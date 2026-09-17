#!/usr/bin/env python3
"""asr_caption.py — PATH 2 core: timestamped ASR transcription.

Three interchangeable backends, all writing the same subtitles.{srt,vtt,json}
schema consumed by build_knowledge.py:

  • faster-whisper (default) — local, CTranslate2 Whisper. Best general-purpose.
  • funasr                  — local, Alibaba FunASR. Best multilingual/Chinese
                                (qwen3-asr / paraformer-zh / sensevoice-small).
  • openai-api              — cloud, any OpenAI-compatible ASR endpoint
                                (DashScope Qwen3-ASR, OpenAI Whisper, Groq, …).

Pick with --backend; defaults come from the hardware profile. Output schema
is backend-agnostic so downstream tools do not change.

Usage:
    # default (faster-whisper)
    python3 asr_caption.py --video in.mp4 --out-dir out --language zh
    # local FunASR (Qwen3-ASR, multilingual SOTA)
    python3 asr_caption.py --video in.mp4 --out-dir out --backend funasr \
        --model qwen3-asr --language zh
    # cloud via DashScope OpenAI-compatible
    python3 asr_caption.py --video in.mp4 --out-dir out --backend openai-api \
        --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api-model qwen3-asr-flash --api-key-env DASHSCOPE_API_KEY \
        --language zh
    # cloud via OpenAI Whisper
    python3 asr_caption.py --video in.mp4 --out-dir out --backend openai-api \
        --api-base https://api.openai.com/v1 --api-model whisper-1 \
        --language en

Outputs (in --out-dir):
    subtitles.srt   # timestamped subtitles
    subtitles.vtt   # WebVTT
    subtitles.json  # {language, language_probability, duration, segments:[…]}
                    # → canonical input to build_knowledge.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
HP = HERE / "hardware_profile.py"


def _hp(key: str, fallback: str) -> str:
    """Read a default from hardware_profile.py; fall back if detection fails."""
    try:
        out = subprocess.run([sys.executable, str(HP), "--key", key],
                             capture_output=True, text=True, check=True, timeout=10)
        v = out.stdout.strip()
        return v or fallback
    except Exception:
        return fallback


DEFAULT_BACKEND = _hp("asr_backend_default", "faster-whisper")
DEFAULT_MODEL = _hp("asr_model", "small")
DEFAULT_COMPUTE = _hp("compute_type", "int8")
DEFAULT_DEVICE = _hp("device", "cpu")
DETECTED_PROFILE = _hp("profile", "unknown")
DETECTED_RAM = float(_hp("ram_gb", "0") or "0")
MODEL_WARN = {"large", "large-v1", "large-v2", "large-v3", "medium"}

# Env-var defaults — resolved at parser-build time (NOT import time) so users
# can override per-invocation. Precedence: CLI > env > hardware_profile > built-in.
# Example shell-rc lines:
#   export ASR_BACKEND=openai-api
#   export ASR_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
#   export ASR_API_MODEL=qwen3-asr-flash
#   export ASR_API_KEY_ENV=DASHSCOPE_API_KEY
#   export ASR_LANGUAGE=zh


def extract_wav(video: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio_16k.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", str(wav)],
        check=True,
    )
    return wav


def fmt_ts(sec: float, sep: str = ",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segs: list[dict]) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        out.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text'].strip()}\n")
    return "\n".join(out)


def to_vtt(segs: list[dict]) -> str:
    body = "\n".join(
        f"{fmt_ts(s['start'], sep='.')} --> {fmt_ts(s['end'], sep='.')}\n{s['text'].strip()}\n"
        for s in segs
    )
    return "WEBVTT\n\n" + body


def load_hotwords(spec: str | None) -> str | None:
    """Parse --hotwords: comma/space/、-separated terms, or @file with one
    term per line. Returns the bias string (None when empty)."""
    if not spec:
        return None
    if spec.startswith("@"):
        path = Path(spec[1:])
        terms = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    else:
        for sep in (",", "、", ";", "；"):
            spec = spec.replace(sep, " ")
        terms = spec.split()
    return "、".join(dict.fromkeys(terms)) or None  # dedupe, keep order


# --- per-backend segment converters (pure: backend output → canonical segs) ---

def _segs_from_faster_whisper(seg_iter) -> list[dict]:
    """faster_whisper returns an iterator of segment objects with .start/.end/.text."""
    return [
        {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
         "text": (s.text or "").strip()}
        for s in seg_iter
    ]


def _segs_from_funasr(result: list[dict]) -> list[dict]:
    """FunASR AutoModel.generate returns list[dict] with 'text' + 'timestamp'
    (character/word-level [[start_ms, end_ms], …]). Aggregate the first→last
    timestamp pair to a segment span in seconds."""
    segs: list[dict] = []
    for item in result:
        text = (item.get("text") or "").strip()
        ts = item.get("timestamp") or []
        if ts and len(ts[0]) >= 2 and len(ts[-1]) >= 2:
            start_ms, _ = ts[0][0], ts[0][1]
            _, end_ms = ts[-1][0], ts[-1][1]
            start, end = start_ms / 1000.0, end_ms / 1000.0
        else:
            start, end = 0.0, 0.0  # model didn't return timestamps
        segs.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return segs


def _segs_from_openai_api(segments) -> list[dict]:
    """OpenAI Whisper-API verbose_json segments are pydantic-like objects
    with .start/.end/.text. DashScope/Groq use the same schema."""
    return [
        {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
         "text": (s.text or "").strip()}
        for s in segments
    ]


# --- per-backend runners (heavy imports, I/O) ---

def _run_faster_whisper(args, wav: Path):
    from faster_whisper import WhisperModel

    if args.model in MODEL_WARN and DETECTED_RAM and DETECTED_RAM < 16 \
            and DETECTED_PROFILE != "high-gpu":
        print(f"[warn] model '{args.model}' may exhaust RAM on this machine "
              f"(profile={DETECTED_PROFILE}, {DETECTED_RAM}GB). "
              f"Suggested for this profile: {DEFAULT_MODEL}.", file=sys.stderr)

    print(f"[asr][faster-whisper] loading model '{args.model}' on {args.device} "
          f"({args.compute_type})...", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    language = None if args.language == "auto" else args.language
    hotwords = load_hotwords(args.hotwords)
    if hotwords:
        print(f"[asr][faster-whisper] hotwords ({len(hotwords.split('、'))} terms)", file=sys.stderr)
    print(f"[asr][faster-whisper] transcribing (language={language or 'auto'})...", file=sys.stderr)
    segs_iter, info = model.transcribe(
        str(wav), language=language, beam_size=5, word_timestamps=True,
        vad_filter=True, initial_prompt=hotwords,
    )
    segs = _segs_from_faster_whisper(segs_iter)
    meta = {"language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration}
    return segs, meta


def _run_funasr(args, wav: Path):
    from funasr import AutoModel

    print(f"[asr][funasr] loading model '{args.model}'...", file=sys.stderr)
    # FunASR's AutoModel picks device from env / its own heuristics.
    kwargs = {}
    # Allow device hint for funasr when explicitly set (some models support it)
    if args.device in ("cuda", "cpu"):
        kwargs["device"] = args.device
    model = AutoModel(model=args.model, **kwargs)
    hotwords = load_hotwords(args.hotwords)
    gen_kwargs = {"input": str(wav)}
    if hotwords:
        gen_kwargs["hotword"] = hotwords
    print(f"[asr][funasr] transcribing (language={args.language})...", file=sys.stderr)
    result = model.generate(**gen_kwargs)
    segs = _segs_from_funasr(result)
    # FunASR doesn't always surface language_probability uniformly; report 1.0.
    duration = max((s["end"] for s in segs), default=0.0)
    meta = {"language": args.language if args.language != "auto" else "auto",
            "language_probability": 1.0,
            "duration": duration}
    return segs, meta


def _run_openai_api(args, wav: Path):
    if not args.api_base:
        raise RuntimeError(
            "openai-api backend requires --api-base "
            "(e.g. https://dashscope.aliyuncs.com/compatible-mode/v1 "
            "or https://api.openai.com/v1)"
        )
    api_key_env = args.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"openai-api backend needs env var '{api_key_env}' set with your API key. "
            f"export {api_key_env}=… before running."
        )
    import openai  # openai>=1.0 client SDK
    print(f"[asr][openai-api] model={args.api_model} base={args.api_base}", file=sys.stderr)
    client = openai.OpenAI(api_key=api_key, base_url=args.api_base)
    hotwords = load_hotwords(args.hotwords)
    with open(wav, "rb") as f:
        kwargs = dict(model=args.api_model, file=f, response_format="verbose_json")
        if args.language and args.language != "auto":
            kwargs["language"] = args.language
        if hotwords:
            # OpenAI Whisper API accepts a 224-token prompt that biases vocab.
            kwargs["prompt"] = hotwords[:1000]
        resp = client.audio.transcriptions.create(**kwargs)
    segs = _segs_from_openai_api(resp.segments)
    meta = {
        "language": getattr(resp, "language", args.language or "unknown"),
        "language_probability": 1.0,
        "duration": float(getattr(resp, "duration", 0.0) or 0.0),
    }
    return segs, meta


# Dispatch table — set after the run* functions are defined.
BACKENDS: dict[str, Callable] = {}


def _register_backends() -> None:
    BACKENDS["faster-whisper"] = _run_faster_whisper
    BACKENDS["funasr"] = _run_funasr
    BACKENDS["openai-api"] = _run_openai_api


_register_backends()


def _build_parser() -> argparse.ArgumentParser:
    # Read env at parse time, not import time, so tests + per-shell invocations work.
    env_backend     = os.environ.get("ASR_BACKEND")      or DEFAULT_BACKEND
    env_model       = os.environ.get("ASR_MODEL")        or DEFAULT_MODEL
    env_language    = os.environ.get("ASR_LANGUAGE")     or "zh"
    env_device      = os.environ.get("ASR_DEVICE")       or DEFAULT_DEVICE
    env_compute     = os.environ.get("ASR_COMPUTE_TYPE") or DEFAULT_COMPUTE
    env_api_base    = os.environ.get("ASR_API_BASE")     or None
    env_api_model   = os.environ.get("ASR_API_MODEL")    or None
    env_api_key_env = os.environ.get("ASR_API_KEY_ENV")  or "OPENAI_API_KEY"

    ap = argparse.ArgumentParser(
        description="ASR → timestamped subtitles. Backends: faster-whisper / funasr / openai-api.\n"
                    "Run with --recommend to print a hardware-based backend suggestion.\n"
                    "Any OpenAI-compatible endpoint works for --backend openai-api\n"
                    "(--api-base URL + --api-model NAME).")
    ap.add_argument("--video", required=False, type=Path,
                    help="input video (not required with --recommend)")
    ap.add_argument("--out-dir", required=False, type=Path,
                    help="output directory (not required with --recommend)")
    ap.add_argument("--backend", choices=sorted(BACKENDS.keys()),
                    default=env_backend,
                    help=f"ASR backend. Precedence: CLI > $ASR_BACKEND > hardware profile. "
                         f"(default {env_backend})")
    # faster-whisper & funasr use --model; openai-api uses --api-model.
    ap.add_argument("--model", default=env_model,
                    help="backend-specific model id. faster-whisper: tiny/base/small/medium/large-v3 "
                         "(or any HuggingFace Whisper id). funasr: qwen3-asr / paraformer-zh / "
                         "sensevoice-small / any FunASR AutoModel name. (default "
                         f"{env_model}; $ASR_MODEL overrides)")
    ap.add_argument("--language", default=env_language,
                    help=f"language code or 'auto' (default {env_language}; $ASR_LANGUAGE overrides)")
    ap.add_argument("--device", default=env_device,
                    help=f"cpu | cuda | auto (default {env_device}). "
                         "Used by faster-whisper and funasr; ignored by openai-api.")
    ap.add_argument("--compute-type", default=env_compute,
                    help=f"int8 | int8_float16 | float16 | float32 "
                         f"(default {env_compute}). faster-whisper only.")
    ap.add_argument("--hotwords", default=None,
                    help="domain terms to bias transcription: comma/space/、separated, "
                         "or @terms.txt (one per line). Passed through to each backend "
                         "via its native bias mechanism (initial_prompt / hotword / prompt).")
    # openai-api specific. Defaults come from env so users can set once in shell rc.
    ap.add_argument("--api-base", default=env_api_base,
                    help="openai-api only: endpoint base URL, e.g. "
                         "https://dashscope.aliyuncs.com/compatible-mode/v1 "
                         "(any OpenAI-compatible URL; default $ASR_API_BASE)")
    ap.add_argument("--api-model", default=env_api_model,
                    help="openai-api only: model id at the endpoint, e.g. "
                         "qwen3-asr-flash / whisper-1 / any custom model name "
                         "(default $ASR_API_MODEL)")
    ap.add_argument("--api-key-env", default=env_api_key_env,
                    help="openai-api only: name of env var holding the API key "
                         f"(default {env_api_key_env}; $ASR_API_KEY_ENV overrides)")
    ap.add_argument("--recommend", action="store_true",
                    help="print a hardware-based backend recommendation and exit "
                         "(does not run ASR; ignores --video/--out-dir).")
    return ap


def main() -> int:
    args = _build_parser().parse_args()

    # --recommend: print hardware-based suggestion, then exit. No ASR run.
    if args.recommend:
        _print_recommendation()
        return 0

    if not args.video or not args.out_dir:
        print("[err] --video and --out-dir are required (unless using --recommend)",
              file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2

    runner = BACKENDS.get(args.backend)
    if runner is None:
        print(f"[err] unknown backend '{args.backend}'. Choices: {sorted(BACKENDS)}",
              file=sys.stderr)
        return 3

    # openai-api uses --api-model; others fall back to --model.
    if args.backend == "openai-api" and not args.api_model:
        print("[err] --backend openai-api requires --api-model "
              "(or set $ASR_API_MODEL)", file=sys.stderr)
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[asr] extracting 16k mono wav...", file=sys.stderr)
    wav = extract_wav(args.video, args.out_dir)

    try:
        segs, meta = runner(args, wav)
    except ImportError as e:
        print(f"[err] backend '{args.backend}' needs an extra dep: {e}. "
              f"Run scripts/setup_models.sh --with-funasr (for funasr) or "
              f"--with-openai-client (for openai-api).", file=sys.stderr)
        return 4
    except RuntimeError as e:
        print(f"[err] {e}", file=sys.stderr)
        return 5

    (args.out_dir / "subtitles.srt").write_text(to_srt(segs), encoding="utf-8")
    (args.out_dir / "subtitles.vtt").write_text(to_vtt(segs), encoding="utf-8")
    (args.out_dir / "subtitles.json").write_text(
        json.dumps({**meta, "segments": segs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] backend={args.backend} {len(segs)} segments -> "
          f"{args.out_dir}/subtitles.{{srt,vtt,json}}")
    return 0


def _print_recommendation() -> None:
    """Read hardware_profile and print a cloud-vs-local ASR recommendation."""
    sys.path.insert(0, str(HERE))
    import hardware_profile as hp_mod
    try:
        d = hp_mod.detect()
    except Exception as e:
        print(f"[err] hardware detection failed: {e}", file=sys.stderr)
        return
    backend = d.get("recommended_asr_backend", "faster-whisper")
    reason = d.get("recommended_backend_reason", "")
    route = "local" if backend in ("faster-whisper", "funasr") else "cloud"
    print(f"hardware profile : {d.get('profile', '?')}  ({d.get('note', '')})")
    print(f"  ram            : {d.get('ram_gb')} GB")
    if d.get("apple_chip"):
        print(f"  apple silicon  : {d['apple_chip']}")
    if d.get("nvidia_vram_gb"):
        print(f"  nvidia vram    : {d['nvidia_vram_gb']} GB")
    print()
    print(f"recommended route: {route.upper()}")
    print(f"recommended backend: {backend}")
    print(f"  reason         : {reason}")
    print()
    # Preset commands so the user can copy-paste.
    if backend == "faster-whisper":
        print("suggested command:")
        print(f"  --backend faster-whisper --model {d.get('asr_model', 'small')} "
              f"--language <zh|en|auto>")
    elif backend == "funasr":
        print("install once:")
        print("  bash scripts/setup_models.sh --with-funasr")
        print("suggested command:")
        print("  --backend funasr --model qwen3-asr --language zh")
    elif backend == "openai-api":
        print("install once:")
        print("  bash scripts/setup_models.sh --with-openai-client")
        print("suggested command (any OpenAI-compatible endpoint; pick yours):")
        print("  --backend openai-api --api-base <YOUR_ENDPOINT> --api-model <MODEL> "
              "--api-key-env <ENV_VAR_WITH_KEY>")
    print()
    print("override precedence: --backend flag > $ASR_BACKEND > this recommendation")


if __name__ == "__main__":
    raise SystemExit(main())
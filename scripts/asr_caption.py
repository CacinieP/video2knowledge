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
    ap = argparse.ArgumentParser(
        description="ASR → timestamped subtitles. Backends: faster-whisper / funasr / openai-api.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", choices=sorted(BACKENDS.keys()),
                    default=DEFAULT_BACKEND,
                    help=f"ASR backend (default {DEFAULT_BACKEND}, from hardware profile)")
    # faster-whisper & funasr use --model; openai-api uses --api-model.
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="backend-specific model id. faster-whisper: tiny/base/small/medium/large-v3. "
                         "funasr: qwen3-asr / paraformer-zh / sensevoice-small / …")
    ap.add_argument("--language", default="zh", help="language code or 'auto'")
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help=f"cpu | cuda | auto (default {DEFAULT_DEVICE}, from profile). "
                         "Used by faster-whisper and funasr; ignored by openai-api.")
    ap.add_argument("--compute-type", default=DEFAULT_COMPUTE,
                    help=f"int8 | int8_float16 | float16 | float32 "
                         f"(default {DEFAULT_COMPUTE}, from profile). "
                         "faster-whisper only; ignored by funasr/openai-api.")
    ap.add_argument("--hotwords", default=None,
                    help="domain terms to bias transcription: comma/space/、separated, "
                         "or @terms.txt (one per line). Passed through to each backend "
                         "via its native bias mechanism (initial_prompt / hotword / prompt).")
    # openai-api specific
    ap.add_argument("--api-base", default=None,
                    help="openai-api only: endpoint base URL, e.g. "
                         "https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-model", default=None,
                    help="openai-api only: model id at the endpoint, e.g. "
                         "qwen3-asr-flash / whisper-1 / whisper-large-v3-turbo")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="openai-api only: name of env var holding the API key "
                         "(default OPENAI_API_KEY). Use DASHSCOPE_API_KEY for DashScope, "
                         "GROQ_API_KEY for Groq, etc.")
    return ap


def main() -> int:
    args = _build_parser().parse_args()

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
        print("[err] --backend openai-api requires --api-model", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
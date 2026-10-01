#!/usr/bin/env python3
"""asr_caption.py — PATH 2 core: timestamped ASR transcription, emitting
SRT/VTT/JSON subtitles.

Three interchangeable backends, all writing the same subtitles.{srt,vtt,json}
schema consumed by build_knowledge.py and batch_run.py:

  • faster-whisper (default) — local, CTranslate2 Whisper. Best general-purpose.
  • openai-api              — cloud, any OpenAI-compatible ASR endpoint
                              (DashScope Qwen3-ASR, OpenAI Whisper, Groq, …).
  • mimo-asr                — cloud, Xiaomi MiMo (mimo-v2.5-asr) via its
                              OpenAI-compatible chat endpoint; audio-only
                              chunks cut on silence, chunk bounds used as the
                              segment timestamps.

funasr is deliberately NOT a backend here. It is a different engine, not a
different model size: it pins an old tokenizers and pulls its own torch, so it
cannot co-install with the main venv. It lives in scripts/asr_funasr.py and is
selected one level up, by `batch_run.py --asr-backend funasr --asr-python …`.
See `python3 hardware_profile.py --recommend` for the per-host advice.

The default --model / --compute-type / --device are auto-selected from the host's
hardware profile (see scripts/hardware_profile.py). Override any of them on the
CLI; the warnings are driven by the detected profile, not a hardcoded RAM number.

API keys are read from the ENVIRONMENT ONLY, never from argv — a key passed on
the command line ends up in the process list, in shell history, and in every log
line that echoes the command. Pass the *name* of the variable instead
(--api-key-env), which is not itself a secret.

Usage:
    python3 asr_caption.py --video in.mp4 --out-dir out --language zh
    python3 asr_caption.py --video in.mp4 --out-dir out --model large-v3   # explicit override
    # cloud via DashScope OpenAI-compatible
    python3 asr_caption.py --video in.mp4 --out-dir out --backend openai-api \
        --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api-model qwen3-asr-flash --api-key-env DASHSCOPE_API_KEY
    # cloud via MiMo — no timestamps from the gateway, so chunks are cut on silence
    python3 asr_caption.py --video in.mp4 --out-dir out --backend mimo-asr \
        --api-key-env MIMO_API_KEY --chunk-seconds 30 --concurrency 4

Outputs (in --out-dir):
    subtitles.srt   # timestamped subtitles
    subtitles.vtt   # WebVTT
    subtitles.json  # {language, language_probability, duration, segments:[…]}
                    # → canonical input to build_knowledge.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


# Hardcoded, NOT read from $ASR_BACKEND: with no --backend flag this must behave
# exactly as it did before cloud backends existed. backend_run.py-style routing
# of funasr happens one level up (batch_run.py --asr-backend).
DEFAULT_BACKEND = "faster-whisper"
DEFAULT_MODEL = _hp("asr_model", "small")
DEFAULT_COMPUTE = _hp("compute_type", "int8")
DEFAULT_DEVICE = _hp("device", "cpu")
DETECTED_PROFILE = _hp("profile", "unknown")
DETECTED_RAM = float(_hp("ram_gb", "0") or "0")
MODEL_WARN = {"large", "large-v1", "large-v2", "large-v3", "medium"}

# Env-var defaults for the cloud backends, resolved at parser-build time (not
# import time) so one `export` in a shell rc is enough. Precedence is
# CLI > env > built-in. These names are cloud-only and cannot affect the default
# faster-whisper path:
#   export ASR_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
#   export ASR_API_MODEL=qwen3-asr-flash
#   export ASR_API_KEY_ENV=DASHSCOPE_API_KEY
#   export V2K_ASR_CHUNK_SEC=30
#   export V2K_ASR_CONCURRENCY=4

MIMO_DEFAULT_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5-asr"


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
    term per line. Returns the initial_prompt string (None when empty)."""
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


def _prepend_nvidia_dll_dirs() -> None:
    """Windows only: put the pip-installed CUDA runtime on the DLL search path.

    CTranslate2 needs cublas64_12.dll / cudnn, which the `nvidia-cublas-cu12`
    and `nvidia-cudnn-cu12` wheels place under
    `site-packages/nvidia/<pkg>/bin` — NOT on PATH by default. Without this,
    `--device cuda` dies with

        RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

    even on a machine with a perfectly good driver. os.add_dll_directory is
    process-scoped, which is exactly the lifetime we need.
    """
    if os.name != "nt":
        return
    import glob
    import site
    roots: list[str] = []
    for getter in (site.getsitepackages, lambda: [site.getusersitepackages()]):
        try:
            roots.extend(getter())
        except Exception:
            pass
    for base in (sys.prefix, sys.base_prefix):
        roots.append(os.path.join(base, "Lib", "site-packages"))
    seen: set[str] = set()
    added = 0
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            if d in seen or not os.path.isdir(d):
                continue
            seen.add(d)
            try:
                os.add_dll_directory(d)
                added += 1
            except Exception:
                pass
    if added:
        print(f"[asr] added {added} NVIDIA runtime dir(s) to the DLL search path",
              file=sys.stderr)


# --- per-backend segment converters (pure: backend output → canonical segs) ---

def _segs_from_faster_whisper(seg_iter) -> list[dict]:
    """faster_whisper yields segment objects with .start/.end/.text."""
    return [
        {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
         "text": (s.text or "").strip()}
        for s in seg_iter
    ]


def _segs_from_openai_api(segments) -> list[dict]:
    """OpenAI Whisper-API verbose_json segments are pydantic-like objects with
    .start/.end/.text. DashScope Qwen-ASR and Groq use the same shape."""
    return [
        {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
         "text": (s.text or "").strip()}
        for s in segments
    ]


# --- per-backend runners (heavy imports, I/O) -------------------------------

def _run_faster_whisper(args, wav: Path):
    from faster_whisper import WhisperModel

    print(f"[asr] loading model '{args.model}' on {args.device} ({args.compute_type})...",
          file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    language = None if args.language == "auto" else args.language
    hotwords = load_hotwords(args.hotwords)
    if hotwords:
        print(f"[asr] hotwords ({len(hotwords.split('、'))} terms): "
              f"{hotwords[:100]}", file=sys.stderr)
    print(f"[asr] transcribing (language={language or 'auto'})...", file=sys.stderr)
    segs_iter, info = model.transcribe(
        str(wav), language=language, beam_size=5, word_timestamps=True,
        vad_filter=True, initial_prompt=hotwords,
    )
    segs = _segs_from_faster_whisper(segs_iter)
    meta = {"language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration}
    return segs, meta


def _run_openai_api(args, wav: Path):
    """Any OpenAI-compatible /audio/transcriptions endpoint.

    The endpoint returns real segment timestamps, so this is a drop-in for
    faster-whisper at the subtitle level. The only genuinely different thing is
    the key: it is read from the named environment variable and never from argv.
    """
    if not args.api_base:
        raise RuntimeError(
            "openai-api backend requires --api-base "
            "(e.g. https://dashscope.aliyuncs.com/compatible-mode/v1 "
            "or https://api.openai.com/v1)")
    api_key_env = args.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"openai-api backend needs env var '{api_key_env}' set with your API key. "
            f"export {api_key_env}=... before running.")
    try:
        import openai  # openai>=1.0 client SDK
    except ImportError as e:
        raise ImportError(
            f"the openai client SDK is not installed ({e}). "
            f"Run scripts/setup_models.sh --with-openai-client") from e
    print(f"[asr][openai-api] model={args.api_model} base={args.api_base}", file=sys.stderr)
    client = openai.OpenAI(api_key=api_key, base_url=args.api_base)
    hotwords = load_hotwords(args.hotwords)
    with open(wav, "rb") as f:
        kwargs = dict(model=args.api_model, file=f, response_format="verbose_json")
        if args.language and args.language != "auto":
            kwargs["language"] = args.language
        if hotwords:
            # The Whisper API accepts a prompt that biases vocabulary.
            kwargs["prompt"] = hotwords[:1000]
        resp = client.audio.transcriptions.create(**kwargs)
    segs = _segs_from_openai_api(resp.segments)
    meta = {
        "language": getattr(resp, "language", None) or args.language or "unknown",
        "language_probability": 1.0,
        "duration": float(getattr(resp, "duration", 0.0) or 0.0),
    }
    return segs, meta


# --- mimo-asr backend (Xiaomi MiMo, chat-shaped cloud ASR) ------------------

def _ffprobe_duration(wav: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _silence_mids(wav: Path) -> list[float]:
    """Midpoints of silence intervals (ffmpeg silencedetect) for cut snapping."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(wav),
         "-af", "silencedetect=noise=-32dB:d=0.35", "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)", proc.stderr)]
    if len(starts) > len(ends):  # trailing silence runs to EOF
        ends.append(_ffprobe_duration(wav))
    return sorted((s + e) / 2 for s, e in zip(starts, ends) if e - s >= 0.35)


def _chunk_bounds(wav: Path, chunk_sec: float) -> list[tuple[float, float]]:
    """[start,end) spans of at most chunk_sec, boundaries snapped to nearby
    silence so a cut does not land in the middle of a word."""
    if chunk_sec <= 0:
        raise RuntimeError(
            f"--chunk-seconds must be > 0 (got {chunk_sec}); a non-positive "
            f"chunk length cannot terminate the cutting loop")
    dur = _ffprobe_duration(wav)
    mids = _silence_mids(wav)
    bounds, t = [0.0], 0.0
    while dur - t > chunk_sec:
        target = t + chunk_sec
        window = [m for m in mids if t + chunk_sec * 0.55 <= m <= target + 6.0]
        cut = min(window, key=lambda m: abs(m - target)) if window else target
        bounds.append(min(cut, dur))
        t = cut
    bounds.append(dur)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] > 0.5]


def _extract_span(wav: Path, start: float, dur: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(wav),
         "-ar", "16000", "-ac", "1", str(dest)],
        check=True)


def _mimo_transcribe_chunk(api_base: str, model: str, api_key: str,
                           path: Path, tries: int = 6) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": [
                   {"type": "input_audio",
                    "input_audio": {"data": f"data:audio/wav;base64,{data}",
                                    "format": "wav"}}]}]}
    last = ""
    for attempt in range(tries):
        req = urllib.request.Request(
            f"{api_base}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read().decode())
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            return (msg.get("content") or "").strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            last = f"HTTP {e.code}: {body}"
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                # gateway throttling / server flakiness: measured 500-storms
                # lasting >15s killed every ~100-min lecture at the old
                # 3-tries/5-10s backoff — 6 tries x exp backoff rides them out
                time.sleep(min(60, 5 * (2 ** attempt)) + random.uniform(0, 3))
                continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # DNS failure, refused connection, TLS error, read timeout. Never
            # swallowed: after the last try it becomes a visible failure.
            last = f"{type(e).__name__}: {e}"
            if attempt < tries - 1:
                time.sleep(min(60, 5 * (2 ** attempt)) + random.uniform(0, 3))
                continue
            break
    raise RuntimeError(f"mimo-asr request failed — {last}")


def _run_mimo_asr(args, wav: Path):
    """Xiaomi MiMo ASR (mimo-v2.5-asr) via the OpenAI-compatible chat endpoint.

    The gateway accepts audio-only user messages (a text part is rejected — the
    prompt is injected server-side) and returns plain text WITHOUT timestamps.
    So this backend cuts the wav on silence-aligned <= --chunk-seconds bounds,
    transcribes the chunks concurrently, and uses each chunk's [start,end] as the
    segment timestamps. Those are coarser than word-level and, unlike the local
    backends, are not corrected afterwards — ASR jargon errors (TileLang ->
    transliterations) are better fixed by a downstream cleanup pass than here.
    """
    from concurrent.futures import ThreadPoolExecutor

    api_key_env = args.api_key_env or "MIMO_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"mimo-asr backend needs env var '{api_key_env}' set with your API key. "
            f"export {api_key_env}=... before running.")
    api_base = (args.api_base or MIMO_DEFAULT_BASE).rstrip("/")
    model = args.api_model or MIMO_DEFAULT_MODEL
    chunk_sec = args.chunk_seconds or 30.0
    conc = max(1, args.concurrency or 4)

    spans = _chunk_bounds(wav, chunk_sec)
    total = _ffprobe_duration(wav)
    print(f"[asr][mimo-asr] model={model} base={api_base}", file=sys.stderr)
    print(f"[asr][mimo-asr] {len(spans)} chunks × ~{chunk_sec:.0f}s "
          f"(audio {total / 60:.1f} min, concurrency {conc})", file=sys.stderr)

    tmp = wav.parent / f"_mimo_chunks_{wav.stem}"
    tmp.mkdir(exist_ok=True)
    try:
        # chunk-text cache: a ~2.5h lecture is 300+ chunks and ~20 min of work;
        # the old code threw all of it away when one chunk exhausted its
        # retries. Completed chunk texts survive in t_*.txt, so a rerun pays
        # only for the chunks that actually failed. spans.json guards against
        # reusing texts cut on different boundaries (changed --chunk-seconds).
        spans_file = tmp / "spans.json"
        if spans_file.is_file():
            try:
                old = json.loads(spans_file.read_text(encoding="utf-8"))
                # full comparison, not just the count: same chunk count on
                # different boundaries would splice stale texts at wrong times
                if [[float(a), float(b)] for a, b in old] != \
                        [[float(a), float(b)] for a, b in spans]:
                    for t in tmp.glob("t_*.txt"):
                        t.unlink()
            except Exception:
                pass
        spans_file.write_text(json.dumps([list(s) for s in spans]),
                              encoding="utf-8")

        def one(idx: int):
            start, end = spans[idx]
            cache = tmp / f"t_{idx:04d}.txt"
            if cache.is_file():
                text = cache.read_text(encoding="utf-8").strip()
                if text:
                    return idx, text
            cpath = tmp / f"chunk_{idx:04d}.wav"
            _extract_span(wav, start, end - start, cpath)
            try:
                text = _mimo_transcribe_chunk(api_base, model, api_key, cpath)
            finally:
                cpath.unlink(missing_ok=True)
            cache.write_text(text, encoding="utf-8")
            return idx, re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        segs = []
        done = 0
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for idx, text in ex.map(one, range(len(spans))):
                start, end = spans[idx]
                if text:
                    segs.append({"start": round(start, 2), "end": round(end, 2),
                                 "text": text})
                done += 1
                if done % 25 == 0:
                    print(f"[asr][mimo-asr] {done}/{len(spans)} chunks", file=sys.stderr)
        shutil.rmtree(tmp, ignore_errors=True)  # success: drop the chunk cache
    finally:
        try:
            tmp.rmdir()
        except OSError:
            pass

    meta = {"language": args.language or "zh", "language_probability": 1.0,
            "duration": round(total, 2)}
    return segs, meta


# Dispatch table — defined after the runners so every value is a real callable.
# funasr is absent on purpose; see the module docstring.
BACKENDS: dict[str, Callable] = {
    "faster-whisper": _run_faster_whisper,
    "openai-api": _run_openai_api,
    "mimo-asr": _run_mimo_asr,
}


class _SecretSafeParser(argparse.ArgumentParser):
    """Never print the value that follows a key-shaped flag.

    argparse's error path quotes every unrecognised token together with its
    value. For an ordinary typo that is exactly what you want. For a mistyped
    `--api-key sk-…` it prints the secret to stderr, into a log, and into
    whatever captures the output — the one place it had not already been.

    So: if a token looks like a key flag and is not a real option here, its
    value is replaced before argparse ever formats a message. The error still
    names the flag the user mistyped, which is the part they need.
    """

    _KEYISH = re.compile(r"^--[A-Za-z0-9-]*key[A-Za-z0-9-]*$", re.IGNORECASE)
    _REAL = frozenset({"--api-key-env", "--keep-wav"})

    def parse_known_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        known = {a.option_strings[0] for a in self._actions}
        for i, tok in enumerate(argv[:-1]):
            if tok in self._REAL or tok in known:
                continue
            if self._KEYISH.match(tok):
                argv[i + 1] = "<redacted: it looked like a key>"
        return super().parse_known_args(argv, namespace)


def _build_parser() -> argparse.ArgumentParser:
    # Two separate controls, and it is worth being precise about which one does
    # what, because neither of them is a complete fix on its own.
    #
    # 1. allow_abbrev=False stops the *misuse*: with abbreviation on,
    #    `--api-key sk-…` is an unambiguous prefix of `--api-key-env`, so
    #    argparse accepts the secret as the NAME of an environment variable and
    #    the "needs env var 'sk-…'" error then quotes it back at the user.
    #
    # 2. _SecretSafeParser below stops the *echo*. allow_abbrev on its own does
    #    not: argparse still reports "unrecognized arguments: --api-key
    #    sk-…", printing the value just as loudly. Verified, not assumed.
    #
    # What neither can do is keep the secret out of the process list or your
    # shell history — by the time any argument parser runs, the command line
    # already contains it. That is why the help text says to pass the NAME of an
    # env var and never the key itself, and why the key is read from the
    # environment rather than from argv.
    ap = _SecretSafeParser(
        allow_abbrev=False,
        description="ASR → timestamped subtitles. Backends: faster-whisper "
                    "(default) / openai-api / mimo-asr. For FunASR use "
                    "asr_funasr.py (batch_run.py --asr-backend funasr).")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", choices=sorted(BACKENDS.keys()),
                    default=DEFAULT_BACKEND,
                    help=f"ASR engine. 'faster-whisper' is local and the default; "
                         f"'openai-api' and 'mimo-asr' upload the audio to a third "
                         f"party. funasr is NOT here — use asr_funasr.py.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"whisper model size: tiny/base/small/medium/large-v3 "
                         f"(default {DEFAULT_MODEL}, from hardware profile). "
                         f"faster-whisper only.")
    ap.add_argument("--language", default="zh", help="language code or 'auto'")
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help=f"cpu | cuda | auto (default {DEFAULT_DEVICE}, from profile). "
                         f"faster-whisper only; ignored by the cloud backends.")
    ap.add_argument("--compute-type", default=DEFAULT_COMPUTE,
                    help=f"int8 | int8_float16 | float16 | float32 "
                         f"(default {DEFAULT_COMPUTE}, from profile). "
                         f"faster-whisper only.")
    ap.add_argument("--keep-wav", action="store_true",
                    help="keep the 16kHz mono wav intermediate "
                         "(default: delete it once subtitles.* are written)")
    ap.add_argument("--hotwords", default=None,
                    help="domain terms to bias transcription: comma/space "
                         "separated, or @terms.txt (one per line). Passed to "
                         "faster-whisper as initial_prompt and to openai-api as "
                         "prompt — sharply reduces errors on jargon, names, "
                         "formulas, product ids")
    # --- cloud backends -----------------------------------------------------
    ap.add_argument("--api-base", default=os.environ.get("ASR_API_BASE") or None,
                    help="openai-api / mimo-asr only: endpoint base URL, e.g. "
                         "https://dashscope.aliyuncs.com/compatible-mode/v1. Any "
                         "OpenAI-compatible URL works (default $ASR_API_BASE; "
                         "mimo-asr falls back to its own default)")
    ap.add_argument("--api-model", default=os.environ.get("ASR_API_MODEL") or None,
                    help="openai-api / mimo-asr only: model id at the endpoint, "
                         "e.g. qwen3-asr-flash / whisper-1 (default $ASR_API_MODEL). "
                         "Required for openai-api")
    ap.add_argument("--api-key-env",
                    default=os.environ.get("ASR_API_KEY_ENV") or "OPENAI_API_KEY",
                    help="cloud backends only: the NAME of the environment variable "
                         "holding the API key (not the key itself — a key on the "
                         "command line leaks into the process list, shell history and "
                         f"every log line). Default {os.environ.get('ASR_API_KEY_ENV') or 'OPENAI_API_KEY'}; "
                         f"$ASR_API_KEY_ENV overrides")
    ap.add_argument("--chunk-seconds", type=float,
                    default=float(os.environ.get("V2K_ASR_CHUNK_SEC", "30")),
                    help="mimo-asr only: max chunk length in seconds; boundaries snap "
                         "to silence so words are not cut mid-sentence "
                         "(default 30, $V2K_ASR_CHUNK_SEC)")
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("V2K_ASR_CONCURRENCY", "4")),
                    help="mimo-asr only: parallel chunk transcriptions "
                         "(default 4, $V2K_ASR_CONCURRENCY)")
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
    if args.backend == "openai-api" and not args.api_model:
        print("[err] --backend openai-api requires --api-model "
              "(or set $ASR_API_MODEL)", file=sys.stderr)
        return 3

    # Warn when a heavy model is forced on a low-RAM profile. faster-whisper
    # only: the cloud backends never load a local model, so there is no RAM to
    # exhaust however heavy --model looks.
    if args.backend == "faster-whisper" and args.model in MODEL_WARN and DETECTED_RAM \
            and DETECTED_RAM < 16 and DETECTED_PROFILE != "high-gpu":
        print(f"[warn] model '{args.model}' may exhaust RAM on this machine "
              f"(profile={DETECTED_PROFILE}, {DETECTED_RAM}GB). "
              f"Suggested for this profile: {DEFAULT_MODEL}.", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "faster-whisper" and args.device == "cuda":
        _prepend_nvidia_dll_dirs()

    print("[asr] extracting 16k mono wav...", file=sys.stderr)
    wav = extract_wav(args.video, args.out_dir)

    try:
        segs, meta = runner(args, wav)
    except ImportError:
        if args.backend == "faster-whisper":
            print("[err] faster-whisper not installed. Run scripts/setup_models.sh first.",
                  file=sys.stderr)
        else:
            print(f"[err] backend '{args.backend}' needs a package that is not "
                  f"installed. Run scripts/setup_models.sh --with-openai-client.",
                  file=sys.stderr)
        return 3
    except RuntimeError as e:
        # A cloud failure must never look like a success. Every backend raises
        # RuntimeError with the reason; we print it and return non-zero WITHOUT
        # writing subtitles.*, so batch_run.py sees a failed stage rather than an
        # empty transcript that downstream turns into a plausible-looking doc.
        print(f"[err] {e}", file=sys.stderr)
        return 5

    (args.out_dir / "subtitles.srt").write_text(to_srt(segs), encoding="utf-8")
    (args.out_dir / "subtitles.vtt").write_text(to_vtt(segs), encoding="utf-8")
    (args.out_dir / "subtitles.json").write_text(
        json.dumps({**meta, "segments": segs}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    if not segs:
        # Writing an empty transcript is legitimate for a genuinely wordless
        # clip (build_knowledge.py has a documented path for it), but it is also
        # exactly what a silently-broken run looks like — so say so, loudly, on
        # stderr, where batch_run.py's stageA.log will pick it up.
        print("[warn] 0 segments: the backend returned no speech. If this video "
              "does have a voice track, the ASR step failed rather than found "
              "silence — do not treat the downstream doc as real.",
              file=sys.stderr)

    # The 16 kHz mono wav is a ~115 MB/hour intermediate that nothing downstream
    # reads — subtitles.* is the only thing the rest of the pipeline consumes,
    # and it is already written. Left in place it dominates the run dir: a
    # 61.5-hour course library leaves ~7 GB of dead WAVs. batch_run.py's own
    # docstring has always claimed "wav deleted after", so this closes the gap
    # between the documented and the actual behaviour. --keep-wav opts out for
    # anyone re-transcribing the same audio without re-extracting it.
    if args.keep_wav:
        print(f"[asr] kept {wav.name} ({wav.stat().st_size/1e6:.0f} MB)",
              file=sys.stderr)
    else:
        try:
            wav.unlink()
        except OSError as e:
            print(f"[warn] could not remove {wav}: {e}", file=sys.stderr)

    print(f"[ok] backend={args.backend} {len(segs)} segments -> "
          f"{args.out_dir}/subtitles.{{srt,vtt,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

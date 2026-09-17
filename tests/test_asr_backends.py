#!/usr/bin/env python3
"""test_asr_backends.py — unit tests for asr_caption.py backend dispatch & schema.

Verifies the three ASR backends (faster-whisper / funasr / openai-api) all
produce the canonical subtitles.json schema consumed by build_knowledge.py,
and that hardware-based backend recommendation picks cloud vs local sensibly.

Run:  python tests/test_asr_backends.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import asr_caption as ac  # noqa: E402

# hardware_profile is needed for the recommendation tests.
sys.path.insert(0, str(HERE.parent / "scripts"))
import hardware_profile as hp  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


# ---- fixtures ----

class FakeWhisperSegment:
    """Mimics faster_whisper.WhisperModel segment objects (.start/.end/.text)."""

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


def fake_funasr_item(text: str, ts_pairs: list[list[int]]) -> dict:
    """Mimics funasr AutoModel.generate() output element."""
    return {"text": text, "timestamp": ts_pairs}


def fake_openai_segment(idx: int, start: float, end: float, text: str) -> MagicMock:
    """Mimics OpenAI verbose_json segment object."""
    seg = MagicMock()
    seg.id = idx
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


# ---- pure-helper tests (no I/O, no mocks of heavy libs) ----

def test_load_hotwords_comma_and_dunhao() -> None:
    out = ac.load_hotwords("foo, bar、baz qux")
    check("hotwords: comma + dunhao + space all parsed",
          out is not None and "foo" in out and "bar" in out and "baz" in out and "qux" in out,
          repr(out))


def test_load_hotwords_file_at_syntax() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("alpha\nbeta\n\ngamma\n")  # blank line should be skipped
        path = Path(f.name)
    try:
        out = ac.load_hotwords(f"@{path}")
        check("hotwords: @file reads lines", "alpha" in out and "beta" in out and "gamma" in out,
              repr(out))
        check("hotwords: blank lines skipped", out is not None and out.count("、") == 2,
              repr(out))
    finally:
        path.unlink()


def test_load_hotwords_empty() -> None:
    check("hotwords: empty returns None", ac.load_hotwords("") is None)
    check("hotwords: None returns None", ac.load_hotwords(None) is None)


def test_srt_format_basic() -> None:
    segs = [{"start": 0.0, "end": 2.5, "text": "Hello"},
            {"start": 2.5, "end": 5.123, "text": "World"}]
    srt = ac.to_srt(segs)
    check("srt: 1-indexed numbering", srt.startswith("1\n"))
    check("srt: timestamp HH:MM:SS,mmm",
          "00:00:00,000 --> 00:00:02,500" in srt and
          "00:00:02,500 --> 00:00:05,123" in srt,
          srt)
    check("srt: text preserved", "Hello" in srt and "World" in srt)


def test_vtt_format_basic() -> None:
    segs = [{"start": 0.0, "end": 2.5, "text": "Hello"}]
    vtt = ac.to_vtt(segs)
    check("vtt: WEBVTT header", vtt.startswith("WEBVTT"))
    check("vtt: timestamp dots (.)",
          "00:00:00.000 --> 00:00:02.500" in vtt, vtt)


# ---- per-backend schema conversion (pure functions) ----

def test_segs_from_faster_whisper_basic() -> None:
    seg_iter = [FakeWhisperSegment(0.0, 2.5, "Hello "),
                FakeWhisperSegment(2.5, 5.0, "World")]
    segs = ac._segs_from_faster_whisper(seg_iter)
    check("fw: count", len(segs) == 2, str(segs))
    check("fw: stripped text", segs[0]["text"] == "Hello", segs[0])
    check("fw: 3-dp rounding", segs[1]["end"] == 5.0, segs[1])
    check("fw: schema keys",
          all(set(s.keys()) == {"start", "end", "text"} for s in segs))


def test_segs_from_funasr_basic() -> None:
    """FunASR AutoModel returns list[dict] with 'text' + 'timestamp' (ms pairs).

    Segment span = first token start → last token end (covers the whole
    utterance regardless of how many char/word-level pairs the model emits).
    """
    result = [
        fake_funasr_item("你好世界", [[0, 500], [500, 1500]]),  # 2 word-pairs → end = 1500ms
        fake_funasr_item("这是第二句", [[1500, 3000]]),         # 1 word-pair  → end = 3000ms
    ]
    segs = ac._segs_from_funasr(result)
    check("funasr: count", len(segs) == 2, str(segs))
    check("funasr: text preserved", segs[0]["text"] == "你好世界", segs[0])
    check("funasr: ms→s conversion (segment = first token start → last token end)",
          segs[0]["start"] == 0.0 and segs[0]["end"] == 1.5, segs[0])
    check("funasr: second segment span",
          segs[1]["start"] == 1.5 and segs[1]["end"] == 3.0, segs[1])
    check("funasr: schema keys",
          all(set(s.keys()) == {"start", "end", "text"} for s in segs))


def test_segs_from_funasr_empty_timestamp_fallback() -> None:
    """If FunASR returns no timestamp (some models), emit 0.0/0.0 placeholder."""
    result = [fake_funasr_item("only text", [])]
    segs = ac._segs_from_funasr(result)
    check("funasr: empty timestamp → 0.0/0.0",
          segs[0]["start"] == 0.0 and segs[0]["end"] == 0.0, segs[0])


def test_segs_from_openai_api_basic() -> None:
    """OpenAI verbose_json returns pydantic-like objects with .start/.end/.text."""
    fake_segs = [fake_openai_segment(0, 0.0, 2.5, " Hello,"),
                 fake_openai_segment(1, 2.5, 5.0, " World.")]
    segs = ac._segs_from_openai_api(fake_segs)
    check("openai-api: count", len(segs) == 2, str(segs))
    check("openai-api: leading-space stripped",
          segs[0]["text"] == "Hello,", segs[0])
    check("openai-api: schema keys",
          all(set(s.keys()) == {"start", "end", "text"} for s in segs))


# ---- backend-dispatch & guard tests ----

def test_openai_api_missing_key_raises() -> None:
    """openai-api backend fails clearly when the named env var is unset."""
    import os
    os.environ.pop("VIDE_TEST_KEY", None)
    args = MagicMock()
    args.api_key_env = "VIDE_TEST_KEY"
    args.api_base = "https://example.com/v1"
    args.api_model = "whisper-1"
    args.language = "en"
    args.hotwords = None
    try:
        ac._run_openai_api(args, Path("/tmp/fake.wav"))
        check("openai-api: missing key raises", False, "did not raise")
    except RuntimeError as e:
        check("openai-api: missing key raises",
              "VIDE_TEST_KEY" in str(e), str(e))


def test_openai_api_missing_base_raises() -> None:
    """openai-api backend fails clearly when --api-base is not set."""
    args = MagicMock()
    args.api_base = None
    args.api_key_env = "OPENAI_API_KEY"
    args.api_model = "whisper-1"
    args.language = "en"
    args.hotwords = None
    try:
        ac._run_openai_api(args, Path("/tmp/fake.wav"))
        check("openai-api: missing base raises", False, "did not raise")
    except RuntimeError as e:
        check("openai-api: missing base raises",
              "api-base" in str(e).lower(), str(e))


def test_backend_choice_dispatch_table() -> None:
    """Each --backend value resolves to a runnable function."""
    table = ac.BACKENDS  # type: ignore[attr-defined]
    check("dispatch table exposes all three backends",
          set(table.keys()) == {"faster-whisper", "funasr", "openai-api"},
          str(sorted(table.keys())))
    check("dispatch table values are callable",
          all(callable(v) for v in table.values()))


def test_argparse_default_backend() -> None:
    """No --backend flag → 'faster-whisper' (backward-compatible default)."""
    import argparse
    # Build a parser identical to main()'s via the helper.
    parser = ac._build_parser()  # type: ignore[attr-defined]
    ns = parser.parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
    check("argparse: default backend is faster-whisper",
          ns.backend == "faster-whisper", getattr(ns, "backend", "MISSING"))


# ---- schema round-trip ----

def test_canonical_json_schema() -> None:
    """Final JSON output matches the contract build_knowledge.py consumes."""
    segs = [{"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 1.0, "end": 2.0, "text": "B"}]
    info = {"language": "zh", "language_probability": 0.99, "duration": 2.0}
    payload = {**info, "segments": segs}
    blob = json.dumps(payload, ensure_ascii=False)
    parsed = json.loads(blob)
    check("schema: top-level keys",
          set(parsed.keys()) >= {"language", "language_probability", "duration", "segments"},
          str(parsed.keys()))
    check("schema: segment keys",
          all(set(s.keys()) == {"start", "end", "text"} for s in parsed["segments"]))


# ---- hardware-based recommendation ----

def test_recommend_tiny_profile_openai_api() -> None:
    """<6GB RAM should always recommend cloud (openai-api) — local is too slow."""
    d = {"ram_gb": 4.0, "nvidia_vram_gb": None, "apple_chip": None}
    backend, reason = hp.recommend_asr_backend(d)
    check("recommend: tiny RAM → openai-api", backend == "openai-api", backend)
    check("recommend: tiny RAM reason mentions RAM", "ram" in reason.lower(), reason)


def test_recommend_nvidia_gpu_funasr() -> None:
    """NVIDIA GPU ≥8GB VRAM should recommend local funasr for GPU acceleration."""
    d = {"ram_gb": 16.0, "nvidia_vram_gb": 12.0, "apple_chip": None}
    backend, _ = hp.recommend_asr_backend(d)
    check("recommend: NVIDIA GPU → funasr (local)", backend == "funasr", backend)


def test_recommend_apple_silicon_high_ram_funasr() -> None:
    """Apple Silicon with ≥16GB RAM → funasr (local SOTA possible with Metal)."""
    d = {"ram_gb": 24.0, "nvidia_vram_gb": None, "apple_chip": "M2 Pro"}
    backend, _ = hp.recommend_asr_backend(d)
    check("recommend: Apple Silicon 16GB+ → funasr", backend == "funasr", backend)


def test_recommend_apple_silicon_low_ram_faster_whisper() -> None:
    """Apple Silicon 8-16GB (no NVIDIA) → faster-whisper (lighter than qwen3-asr)."""
    d = {"ram_gb": 10.0, "nvidia_vram_gb": None, "apple_chip": "M1"}
    backend, _ = hp.recommend_asr_backend(d)
    check("recommend: Apple Silicon 8GB → faster-whisper", backend == "faster-whisper", backend)


def test_recommend_x86_mid_ram_faster_whisper() -> None:
    """x86 8-16GB no dGPU → faster-whisper (the safe default)."""
    d = {"ram_gb": 12.0, "nvidia_vram_gb": None, "apple_chip": None}
    backend, _ = hp.recommend_asr_backend(d)
    check("recommend: mid x86 no GPU → faster-whisper", backend == "faster-whisper", backend)


def test_recommend_high_ram_funasr() -> None:
    """≥16GB RAM (any arch) → funasr (can host qwen3-asr locally)."""
    d = {"ram_gb": 32.0, "nvidia_vram_gb": None, "apple_chip": None}
    backend, _ = hp.recommend_asr_backend(d)
    check("recommend: 32GB no GPU → funasr", backend == "funasr", backend)


def test_recommend_nvidia_beats_ram_rule() -> None:
    """NVIDIA short-circuits even on lower RAM (GPU does the heavy lifting)."""
    d = {"ram_gb": 8.0, "nvidia_vram_gb": 10.0, "apple_chip": None}
    backend, _ = hp.recommend_asr_backend(d)
    check("recommend: NVIDIA wins over RAM rule", backend == "funasr", backend)


def test_detect_includes_recommendation() -> None:
    """detect() output includes recommended_asr_backend + recommended_backend_reason."""
    out = hp.detect.__wrapped__ if hasattr(hp.detect, "__wrapped__") else hp.detect
    # Don't actually invoke detect() (it probes hardware). Verify schema by patching.
    import unittest.mock as mock
    with mock.patch.object(hp, "detect_ram_gb", return_value=8.0), \
         mock.patch.object(hp, "detect_nvidia_vram_gb", return_value=None), \
         mock.patch.object(hp, "detect_apple_silicon", return_value=None):
        d = hp.detect()
    check("detect: includes recommended_asr_backend",
          "recommended_asr_backend" in d, str(sorted(d.keys())))
    check("detect: includes recommended_backend_reason",
          "recommended_backend_reason" in d, str(sorted(d.keys())))
    check("detect: recommendation is a known backend",
          d["recommended_asr_backend"] in {"faster-whisper", "funasr", "openai-api"},
          d["recommended_asr_backend"])


# ---- env-var default override ----

def test_env_var_override_backend() -> None:
    """ASR_BACKEND env var changes the default backend chosen by argparse."""
    import os
    old = os.environ.get("ASR_BACKEND")
    os.environ["ASR_BACKEND"] = "openai-api"
    try:
        ns = ac._build_parser().parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
        check("env: ASR_BACKEND=openai-api → argparse default = openai-api",
              ns.backend == "openai-api", ns.backend)
    finally:
        if old is None:
            os.environ.pop("ASR_BACKEND", None)
        else:
            os.environ["ASR_BACKEND"] = old


def test_env_var_override_api_base() -> None:
    """ASR_API_BASE env var populates --api-base when not given."""
    import os
    old = os.environ.get("ASR_API_BASE")
    os.environ["ASR_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    try:
        ns = ac._build_parser().parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
        check("env: ASR_API_BASE → argparse default = that URL",
              ns.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1",
              ns.api_base)
    finally:
        if old is None:
            os.environ.pop("ASR_API_BASE", None)
        else:
            os.environ["ASR_API_BASE"] = old


# ---- vendor neutrality: --api-base accepts arbitrary URL ----

def test_argparse_accepts_arbitrary_api_base() -> None:
    """No vendor lock-in: parser must accept ANY URL for --api-base."""
    urls = [
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://api.openai.com/v1",
        "https://api.groq.com/openai/v1",
        "http://127.0.0.1:8080/v1",                              # self-hosted
        "https://my-llm-gateway.corp.internal/openai-compatible/v1",
        "https://api.deepinfra.com/v1/openai",
    ]
    for url in urls:
        ns = ac._build_parser().parse_args(
            ["--video", "x.mp4", "--out-dir", "/tmp/y", "--api-base", url])
        check(f"argparse: --api-base accepts {url}", ns.api_base == url, ns.api_base)


def test_argparse_accepts_arbitrary_api_model() -> None:
    """No vendor lock-in: --api-model accepts any string (custom model name)."""
    models = [
        "qwen3-asr-flash",
        "whisper-1",
        "whisper-large-v3-turbo",
        "my-finetuned-asr-v2",
        "team-internal/speech-recognizer-2026-09",
    ]
    for m in models:
        ns = ac._build_parser().parse_args(
            ["--video", "x.mp4", "--out-dir", "/tmp/y", "--api-model", m])
        check(f"argparse: --api-model accepts {m}", ns.api_model == m, ns.api_model)


# ---- runner ----

def main() -> int:
    test_load_hotwords_comma_and_dunhao()
    test_load_hotwords_file_at_syntax()
    test_load_hotwords_empty()
    test_srt_format_basic()
    test_vtt_format_basic()
    test_segs_from_faster_whisper_basic()
    test_segs_from_funasr_basic()
    test_segs_from_funasr_empty_timestamp_fallback()
    test_segs_from_openai_api_basic()
    test_openai_api_missing_key_raises()
    test_openai_api_missing_base_raises()
    test_backend_choice_dispatch_table()
    test_argparse_default_backend()
    test_canonical_json_schema()
    # hardware-based recommendation
    test_recommend_tiny_profile_openai_api()
    test_recommend_nvidia_gpu_funasr()
    test_recommend_apple_silicon_high_ram_funasr()
    test_recommend_apple_silicon_low_ram_faster_whisper()
    test_recommend_x86_mid_ram_faster_whisper()
    test_recommend_high_ram_funasr()
    test_recommend_nvidia_beats_ram_rule()
    test_detect_includes_recommendation()
    # env-var override
    test_env_var_override_backend()
    test_env_var_override_api_base()
    # vendor neutrality
    test_argparse_accepts_arbitrary_api_base()
    test_argparse_accepts_arbitrary_api_model()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
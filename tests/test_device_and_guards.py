#!/usr/bin/env python3
"""test_device_and_guards.py — regressions for three defects found by running a
real 61.5-hour course library through the default path.

1. hardware_profile: `device` came straight from the profile tier table
   (`nvidia_vram >= 8` -> high-gpu, else cpu). A 6 GB RTX 3060 Laptop was told to
   use the CPU and ran 2.5x realtime instead of 17.9x — a 61.5 h library went from
   5 h to 24 h. And on Windows `get_cuda_device_count()` returned 1 while the CUDA
   runtime DLLs were absent, so trusting the count sent every run into
   "RuntimeError: Library cublas64_12.dll is not found".

2. build_knowledge: with 0 recognised segments the script still called the LLM.
   A small model handed an empty transcript does not decline, it generates until
   it hits the context ceiling — measured 9+ minutes with no output, and because
   ollama serialises per model it blocked every later video in the batch.

3. asr_caption: the 16 kHz mono wav (~115 MB/hour) was never removed, though
   batch_run.py has always documented "wav deleted after" — ~7 GB of dead WAVs
   for this library.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import asr_caption  # noqa: E402
import build_knowledge  # noqa: E402
import hardware_profile as hp  # noqa: E402


# --------------------------------------------------------------------------
# 1. ASR device selection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vram,cuda_ok,expected", [
    (None, False, "cpu"),      # no NVIDIA card at all
    (None, True, "cpu"),       # Apple silicon / integrated graphics
    (6.0, False, "cpu"),       # driver visible but runtime DLLs missing
    (2.0, True, "cpu"),        # too little VRAM for a comfortable ASR load
    (3.0, True, "cuda"),       # threshold, runtime verified
    (6.0, True, "cuda"),       # the 3060 Laptop that used to be sent to the CPU
    (16.0, True, "cuda"),
])
def test_select_asr_device(vram, cuda_ok, expected):
    assert hp.select_asr_device(vram, cuda_ok) == expected


def test_detect_prefers_cuda_on_a_6gb_card(monkeypatch):
    """The whole point: a 6 GB NVIDIA card must not be classified 'cpu' just
    because it misses the 8 GB tier cutoff."""
    monkeypatch.setattr(hp, "detect_ram_gb", lambda: 15.8)
    monkeypatch.setattr(hp, "detect_nvidia_vram_gb", lambda: 6.0)
    monkeypatch.setattr(hp, "detect_apple_silicon", lambda: None)
    monkeypatch.setattr(hp, "detect_cuda_usable", lambda: True)
    d = hp.detect()
    assert d["profile"] == "mid"          # tier unchanged...
    assert d["device"] == "cuda"          # ...but ASR no longer follows it
    assert d["cuda_ok"] is True


def test_detect_falls_back_to_cpu_int8_without_a_usable_runtime(monkeypatch):
    monkeypatch.setattr(hp, "detect_ram_gb", lambda: 15.8)
    monkeypatch.setattr(hp, "detect_nvidia_vram_gb", lambda: 6.0)
    monkeypatch.setattr(hp, "detect_apple_silicon", lambda: None)
    monkeypatch.setattr(hp, "detect_cuda_usable", lambda: False)
    d = hp.detect()
    assert d["device"] == "cpu"
    assert d["cuda_ok"] is False
    # int8_float16 is CUDA-only; CTranslate2's CPU backend raises on load.
    assert d["compute_type"] == "int8"


def test_detect_upgrades_int8_to_int8_float16_on_cuda(monkeypatch):
    """A tier that settles on plain int8 should still get float16 math when a
    real CUDA device was verified: same int8 weights, no extra VRAM, faster."""
    monkeypatch.setattr(hp, "detect_ram_gb", lambda: 15.8)
    monkeypatch.setattr(hp, "detect_nvidia_vram_gb", lambda: 6.0)
    monkeypatch.setattr(hp, "detect_apple_silicon", lambda: None)
    monkeypatch.setattr(hp, "detect_cuda_usable", lambda: True)
    monkeypatch.setitem(hp.PROFILES["mid"], "compute", "int8")
    try:
        assert hp.detect()["compute_type"] == "int8_float16"
    finally:
        hp.PROFILES["mid"]["compute"] = "int8_float16"


def test_detect_cuda_usable_returns_a_bool():
    """Must never raise — a detection failure has to degrade to 'cpu', not
    take down the caller that shells out to --key device."""
    assert isinstance(hp.detect_cuda_usable(), bool)


# --------------------------------------------------------------------------
# 2. build_knowledge must not call the LLM on a wordless video
# --------------------------------------------------------------------------

def _write_subtitles(path: Path, segments) -> Path:
    path.write_text(json.dumps({"language": "zh", "duration": 90.0,
                                "segments": segments}, ensure_ascii=False),
                    encoding="utf-8")
    return path


def test_zero_segment_video_skips_the_llm(tmp_path, monkeypatch):
    subs = _write_subtitles(tmp_path / "subtitles.json", [])
    out = tmp_path / "out"

    def boom(*a, **k):
        raise AssertionError("LLM must not be called for a video with no speech")

    monkeypatch.setattr(build_knowledge, "build_analysis", boom)
    monkeypatch.setattr(sys, "argv", [
        "build_knowledge.py", "--subtitles", str(subs), "--out-dir", str(out),
        "--format", "knowledge", "--lang", "zh"])
    assert build_knowledge.main() == 0

    md = (out / "knowledge.md").read_text(encoding="utf-8")
    assert "未识别到语音内容" in md
    # and it must point at the only path that can actually handle a wordless clip
    assert "build_notes.py" in md


def test_blank_text_segments_also_skip_the_llm(tmp_path, monkeypatch):
    """Segments that exist but carry only whitespace are the same situation."""
    subs = _write_subtitles(tmp_path / "subtitles.json",
                            [{"start": 0.0, "end": 2.0, "text": "   "}])
    out = tmp_path / "out"

    def boom(*a, **k):
        raise AssertionError("LLM must not be called on a blank transcript")

    monkeypatch.setattr(build_knowledge, "build_analysis", boom)
    monkeypatch.setattr(sys, "argv", [
        "build_knowledge.py", "--subtitles", str(subs), "--out-dir", str(out),
        "--format", "knowledge", "--lang", "zh"])
    assert build_knowledge.main() == 0
    assert "未识别到语音内容" in (out / "knowledge.md").read_text(encoding="utf-8")


def test_real_transcript_still_calls_the_llm(tmp_path, monkeypatch):
    """The guard must not swallow normal videos."""
    # Several segments of realistic length, not one 8-character line: a real
    # lecture clears the speech-density floor, and this test is about that.
    subs = _write_subtitles(tmp_path / "subtitles.json", [
        {"start": 0.0, "end": 2.0, "text": "今天讲音阶的指法，先看右手的基本手型。"},
        {"start": 2.0, "end": 4.0, "text": "拇指和食指的间距决定了能不能连续弹。"},
        {"start": 4.0, "end": 6.0, "text": "然后我们加上中指和无名指一起练习。"},
    ])
    out = tmp_path / "out"
    seen = {}

    def fake_analysis(host, model, raw_text, source, lang=None,
                      char_limit=None, cache=None):
        seen["raw_text"] = raw_text
        return {"summary": "S", "timeline": "T", "key_points": "- K",
                "bullets": "- B", "qa": "- Q", "glossary": "- G"}

    monkeypatch.setattr(build_knowledge, "build_analysis", fake_analysis)
    monkeypatch.setattr(sys, "argv", [
        "build_knowledge.py", "--subtitles", str(subs), "--out-dir", str(out),
        "--format", "knowledge", "--lang", "zh"])
    assert build_knowledge.main() == 0
    assert "音阶" in seen["raw_text"]
    assert "未识别到语音内容" not in (out / "knowledge.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 3. the wav intermediate is cleaned up unless --keep-wav
# --------------------------------------------------------------------------

class _FakeSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeInfo:
    language, language_probability, duration = "zh", 0.99, 4.0


def _install_fake_faster_whisper(monkeypatch, segments):
    class _Model:
        def __init__(self, model, device=None, compute_type=None):
            self.device, self.compute_type = device, compute_type

        def transcribe(self, path, **kw):
            return iter(segments), _FakeInfo()

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = _Model
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


def _run_asr(tmp_path, monkeypatch, extra_args=()):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not really a video")
    out = tmp_path / "out"

    def fake_extract(video, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / "audio_16k.wav"
        wav.write_bytes(b"RIFF" + b"\0" * 2048)
        return wav

    monkeypatch.setattr(asr_caption, "extract_wav", fake_extract)
    _install_fake_faster_whisper(monkeypatch, [_FakeSeg(0.0, 2.0, " 测试")])
    monkeypatch.setattr(sys, "argv", [
        "asr_caption.py", "--video", str(video), "--out-dir", str(out),
        "--device", "cpu", "--compute-type", "int8", *extra_args])
    rc = asr_caption.main()
    return rc, out


def test_wav_is_deleted_by_default(tmp_path, monkeypatch):
    rc, out = _run_asr(tmp_path, monkeypatch)
    assert rc == 0
    assert not (out / "audio_16k.wav").exists()
    # the subtitles are what downstream consumes, and they must survive
    assert (out / "subtitles.srt").exists()
    assert (out / "subtitles.json").exists()


def test_keep_wav_opt_out(tmp_path, monkeypatch):
    rc, out = _run_asr(tmp_path, monkeypatch, extra_args=["--keep-wav"])
    assert rc == 0
    assert (out / "audio_16k.wav").exists()
    assert (out / "subtitles.srt").exists()

#!/usr/bin/env python3
"""test_asr_cloud.py — the cloud ASR backends of asr_caption.py.

Covers the three things that are easy to get quietly wrong:

1. the API key comes from the ENVIRONMENT only. A key on the command line lands
   in the process list, in shell history and in every log line that echoes the
   command, so there is deliberately no `--api-key` option and a missing variable
   has to fail with a message naming it.
2. a cloud failure is never a success. Silent transcription failure is how a
   whole batch ends up "successful" with placeholder documents, so every failure
   path prints its reason to stderr, returns non-zero, and writes no
   subtitles.* that a downstream step could mistake for a real transcript.
3. the dispatch table is exactly the three engines this script owns. funasr
   lives in asr_funasr.py behind `batch_run.py --asr-backend funasr` (it needs a
   different interpreter), so it must not creep back in here.

Nothing in this file touches the network: urllib and the openai SDK are stubbed.
"""
from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import asr_caption as ac  # noqa: E402


# The cloud defaults must be deterministic regardless of the developer's shell.
CLOUD_ENV = ["ASR_API_BASE", "ASR_API_MODEL", "ASR_API_KEY_ENV",
             "V2K_ASR_CHUNK_SEC", "V2K_ASR_CONCURRENCY"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in CLOUD_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)


def _args(**kw):
    """A parsed-args stand-in; only the fields a runner reads are set."""
    a = types.SimpleNamespace(
        backend="faster-whisper", model="small", language="zh", device="cpu",
        compute_type="int8", hotwords=None, keep_wav=False,
        api_base=None, api_model=None, api_key_env="OPENAI_API_KEY",
        chunk_seconds=30.0, concurrency=4)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _wav(tmp_path, name="audio_16k.wav"):
    w = tmp_path / name
    w.write_bytes(b"RIFF" + b"\0" * 2048)
    return w


# --------------------------------------------------------------------------
# dispatch table
# --------------------------------------------------------------------------

def test_dispatch_table_exposes_exactly_three_backends():
    assert set(ac.BACKENDS) == {"faster-whisper", "openai-api", "mimo-asr"}


def test_dispatch_values_are_callable():
    assert all(callable(v) for v in ac.BACKENDS.values())


def test_funasr_is_not_a_backend_here():
    """funasr needs its own interpreter (own torch + old tokenizers) and lives
    in asr_funasr.py, selected by batch_run.py --asr-backend funasr."""
    assert "funasr" not in ac.BACKENDS
    assert (HERE.parent / "scripts" / "asr_funasr.py").is_file()


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def test_default_backend_is_faster_whisper():
    """Backward compatibility: with no --backend the run must be exactly the one
    that existed before cloud backends did."""
    ns = ac._build_parser().parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
    assert ns.backend == "faster-whisper"
    assert ac.DEFAULT_BACKEND == "faster-whisper"


def test_faster_whisper_local_defaults_are_untouched():
    """The default path must not drift: profile-derived model/device/compute and
    the zh language default are what every existing caller relies on."""
    ns = ac._build_parser().parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
    assert ns.model == ac.DEFAULT_MODEL
    assert ns.device == ac.DEFAULT_DEVICE
    assert ns.compute_type == ac.DEFAULT_COMPUTE
    assert ns.language == "zh"
    assert ns.keep_wav is False


def test_backend_choices_reject_funasr():
    with pytest.raises(SystemExit):
        ac._build_parser().parse_args(
            ["--video", "x.mp4", "--out-dir", "/tmp/y", "--backend", "funasr"])


def test_cloud_backends_are_selectable():
    for b in ("openai-api", "mimo-asr"):
        ns = ac._build_parser().parse_args(
            ["--video", "x.mp4", "--out-dir", "/tmp/y", "--backend", b])
        assert ns.backend == b


def test_cloud_defaults():
    ns = ac._build_parser().parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
    assert ns.api_base is None
    assert ns.api_model is None
    assert ns.api_key_env == "OPENAI_API_KEY"
    assert ns.chunk_seconds == 30.0
    assert ns.concurrency == 4


def test_api_base_and_model_are_vendor_neutral():
    """No lock-in to one provider — any OpenAI-shaped endpoint must parse."""
    for url in ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                "https://api.openai.com/v1",
                "https://api.groq.com/openai/v1",
                "http://127.0.0.1:8080/v1"):
        for model in ("qwen3-asr-flash", "whisper-1", "my-finetuned-asr-v2"):
            ns = ac._build_parser().parse_args(
                ["--video", "x.mp4", "--out-dir", "/tmp/y",
                 "--api-base", url, "--api-model", model])
            assert ns.api_base == url
            assert ns.api_model == model


def test_env_populates_cloud_defaults(monkeypatch):
    monkeypatch.setenv("ASR_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("ASR_API_MODEL", "whisper-1")
    monkeypatch.setenv("ASR_API_KEY_ENV", "EXAMPLE_KEY")
    ns = ac._build_parser().parse_args(["--video", "x.mp4", "--out-dir", "/tmp/y"])
    assert ns.api_base == "https://api.example.com/v1"
    assert ns.api_model == "whisper-1"
    assert ns.api_key_env == "EXAMPLE_KEY"


def test_cli_beats_env(monkeypatch):
    monkeypatch.setenv("ASR_API_MODEL", "from-env")
    ns = ac._build_parser().parse_args(
        ["--video", "x.mp4", "--out-dir", "/tmp/y", "--api-model", "from-cli"])
    assert ns.api_model == "from-cli"


# --------------------------------------------------------------------------
# the key must never come from argv
# --------------------------------------------------------------------------

def test_there_is_no_api_key_option():
    """A key on the command line is visible to every process on the box and
    lands in shell history. Only the NAME of the variable may be passed."""
    opts = {o for a in ac._build_parser()._actions for o in a.option_strings}
    assert "--api-key" not in opts
    assert "--api-key-env" in opts


def test_passing_api_key_on_the_command_line_is_rejected():
    with pytest.raises(SystemExit):
        ac._build_parser().parse_args(
            ["--video", "x.mp4", "--out-dir", "/tmp/y",
             "--api-key", "sk-should-not-be-here"])


def test_openai_api_missing_key_fails_naming_the_variable(monkeypatch):
    monkeypatch.delenv("V2K_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError) as ei:
        ac._run_openai_api(_args(backend="openai-api", api_key_env="V2K_TEST_KEY",
                                 api_base="https://api.example.com/v1",
                                 api_model="whisper-1"), Path("/tmp/fake.wav"))
    assert "V2K_TEST_KEY" in str(ei.value)


def test_mimo_missing_key_fails_naming_the_variable(monkeypatch):
    monkeypatch.delenv("V2K_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError) as ei:
        ac._run_mimo_asr(_args(backend="mimo-asr", api_key_env="V2K_TEST_KEY"),
                         Path("/tmp/fake.wav"))
    assert "V2K_TEST_KEY" in str(ei.value)


def test_openai_api_missing_base_fails():
    with pytest.raises(RuntimeError) as ei:
        ac._run_openai_api(_args(backend="openai-api", api_base=None,
                                 api_model="whisper-1"), Path("/tmp/fake.wav"))
    assert "api-base" in str(ei.value).lower()


def test_openai_api_reads_the_key_from_the_environment(monkeypatch, tmp_path):
    """The configured client must receive the ENV value — proof the secret
    travels out-of-band, not through the args namespace."""
    monkeypatch.setenv("V2K_TEST_KEY", "secret-from-env")
    seen = {}

    class _Client:
        def __init__(self, api_key=None, base_url=None):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            self.audio = types.SimpleNamespace(transcriptions=_Transcriptions())

    class _Resp:
        language, duration = "zh", 12.0
        segments = [types.SimpleNamespace(start=0.0, end=2.0, text=" 你好 ")]

    class _Transcriptions:
        def create(self, **kw):
            seen["kwargs"] = kw
            return _Resp()

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    segs, meta = ac._run_openai_api(
        _args(backend="openai-api", api_key_env="V2K_TEST_KEY",
              api_base="https://api.example.com/v1", api_model="whisper-1"),
        _wav(tmp_path))

    assert seen["api_key"] == "secret-from-env"
    assert seen["base_url"] == "https://api.example.com/v1"
    assert seen["kwargs"]["model"] == "whisper-1"
    assert seen["kwargs"]["response_format"] == "verbose_json"
    assert segs == [{"start": 0.0, "end": 2.0, "text": "你好"}]
    assert meta["language"] == "zh"
    assert meta["duration"] == 12.0


def test_openai_api_hotwords_become_a_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    seen = {}

    class _Client:
        def __init__(self, api_key=None, base_url=None):
            self.audio = types.SimpleNamespace(transcriptions=_Transcriptions())

    class _Resp:
        language, duration = "zh", 1.0
        segments = []

    class _Transcriptions:
        def create(self, **kw):
            seen.update(kw)
            return _Resp()

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    ac._run_openai_api(
        _args(backend="openai-api", api_key_env="V2K_TEST_KEY",
              api_base="https://api.example.com/v1", api_model="whisper-1",
              hotwords="音阶、指法"),
        _wav(tmp_path))
    assert seen["prompt"] == "音阶、指法"


# --------------------------------------------------------------------------
# mimo: silence-aligned chunking
# --------------------------------------------------------------------------

def test_silence_mids_parses_and_filters_ffmpeg_output(monkeypatch):
    """Midpoints are cut candidates; pauses shorter than 0.35 s are not."""
    monkeypatch.setattr(
        ac.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(stderr=(
            "[silencedetect] silence_start: 1.0\n"        # 0.2 s -> too short
            "[silencedetect] silence_end: 1.2\n"
            "[silencedetect] silence_start: 10.0\n"       # 0.5 s -> a real pause
            "[silencedetect] silence_end: 10.5\n")))
    assert ac._silence_mids(Path("x.wav")) == [10.25]


def test_silence_mids_closes_a_trailing_silence_at_eof(monkeypatch):
    """ffmpeg omits silence_end for silence that runs to EOF; without closing it
    the last cut candidate is lost."""
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 30.0)
    monkeypatch.setattr(
        ac.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(stderr=(
            "[silencedetect] silence_start: 20.0\n")))
    assert ac._silence_mids(Path("x.wav")) == [25.0]


def test_chunk_bounds_snap_to_nearby_silence(monkeypatch):
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 100.0)
    monkeypatch.setattr(ac, "_silence_mids", lambda p: [28.0, 55.0, 80.0, 99.0])
    assert ac._chunk_bounds(Path("x.wav"), 30.0) == [
        (0.0, 28.0), (28.0, 55.0), (55.0, 80.0), (80.0, 100.0)]


def test_chunk_bounds_fall_back_to_even_splits(monkeypatch):
    """Continuous speech has no silence to snap to — cut on the clock instead of
    failing or looping."""
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 100.0)
    monkeypatch.setattr(ac, "_silence_mids", lambda p: [])
    assert ac._chunk_bounds(Path("x.wav"), 30.0) == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 100.0)]


def test_chunk_bounds_reject_a_non_positive_chunk(monkeypatch):
    """target = t + 0 never advances, so this would hang the run."""
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 100.0)
    monkeypatch.setattr(ac, "_silence_mids", lambda p: [])
    with pytest.raises(RuntimeError) as ei:
        ac._chunk_bounds(Path("x.wav"), 0.0)
    assert "chunk-seconds" in str(ei.value)


# --------------------------------------------------------------------------
# mimo: chat request, retry and error visibility
# --------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mimo_reply(text):
    return _FakeResponse({"choices": [{"message": {"content": text}}]})


def test_mimo_request_returns_the_message_content(monkeypatch, tmp_path):
    monkeypatch.setattr(ac.urllib.request, "urlopen",
                        lambda req, timeout=None: _mimo_reply("今天讲音阶"))
    got = ac._mimo_transcribe_chunk("https://x/v1", "mimo-v2.5-asr", "k",
                                    _wav(tmp_path, "c.wav"))
    assert got == "今天讲音阶"


def test_mimo_request_retries_a_429_then_succeeds(monkeypatch, tmp_path):
    """Throttling is expected, not a failure — but only if it actually clears."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {},
                                        __import__("io").BytesIO(b"slow down"))
        return _mimo_reply("ok")

    monkeypatch.setattr(ac.time, "sleep", lambda s: None)
    monkeypatch.setattr(ac.urllib.request, "urlopen", fake_urlopen)
    got = ac._mimo_transcribe_chunk("https://x/v1", "m", "k", _wav(tmp_path, "c.wav"))
    assert got == "ok"
    assert calls["n"] == 3


def test_mimo_request_gives_up_with_the_server_body(monkeypatch, tmp_path):
    """A permanent 4xx is a real error: it must surface with the reason, not be
    retried into oblivion or dropped."""
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError("u", 400, "Bad Request", {},
                                     __import__("io").BytesIO(b"unsupported audio"))

    monkeypatch.setattr(ac.time, "sleep", lambda s: None)
    monkeypatch.setattr(ac.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as ei:
        ac._mimo_transcribe_chunk("https://x/v1", "m", "k", _wav(tmp_path, "c.wav"))
    assert "400" in str(ei.value) and "unsupported audio" in str(ei.value)


def test_mimo_request_wraps_a_connection_failure(monkeypatch, tmp_path):
    """A dead endpoint must become a visible RuntimeError, not a bare URLError
    escaping as a traceback."""
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(ac.time, "sleep", lambda s: None)
    monkeypatch.setattr(ac.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as ei:
        ac._mimo_transcribe_chunk("https://x/v1", "m", "k", _wav(tmp_path, "c.wav"),
                                  tries=2)
    assert "Name or service not known" in str(ei.value)


def test_mimo_run_uses_chunk_bounds_as_timestamps(monkeypatch, tmp_path):
    """The gateway returns no timestamps, so each chunk's [start,end] IS the
    segment — that is the whole reason the chunking exists. Reasoning tags are
    stripped here too: they would otherwise land in the transcript verbatim."""
    monkeypatch.setenv("MIMO_API_KEY", "k")
    spans = [(0.0, 28.0), (28.0, 55.0), (55.0, 80.0), (80.0, 100.0)]
    monkeypatch.setattr(ac, "_chunk_bounds", lambda w, c: spans)
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 100.0)
    seen_spans = []

    def fake_extract(wav, start, dur, dest):
        seen_spans.append((start, dur))
        dest.write_bytes(b"RIFF" + b"\0" * 16)

    monkeypatch.setattr(ac, "_extract_span", fake_extract)
    # index 2 came back silent -> no segment; index 1 carries <think>…</think>
    texts = {0: "第一段", 1: "<think>先分析音频</think>第二段", 2: "", 3: "第四段"}
    monkeypatch.setattr(
        ac, "_mimo_transcribe_chunk",
        lambda base, model, key, path: texts[int(path.stem.split("_")[1])])

    segs, meta = ac._run_mimo_asr(
        _args(backend="mimo-asr", api_key_env="MIMO_API_KEY"), _wav(tmp_path))

    assert segs == [
        {"start": 0.0, "end": 28.0, "text": "第一段"},
        {"start": 28.0, "end": 55.0, "text": "第二段"},
        {"start": 80.0, "end": 100.0, "text": "第四段"},
    ]
    assert seen_spans == [(0.0, 28.0), (28.0, 27.0), (55.0, 25.0), (80.0, 20.0)]
    assert meta["duration"] == 100.0
    assert meta["language"] == "zh"
    # the temp chunk directory must not survive the run
    assert not (tmp_path / f"_mimo_chunks_audio_16k").exists()


def test_mimo_run_keeps_chunk_text_cache_when_a_chunk_fails(monkeypatch, tmp_path):
    """A 300-chunk lecture must not lose 300 chunks of paid work because chunk
    299 hit a 500-storm. On failure the wav scratch is gone but the completed
    chunk texts and the spans manifest stay behind for the rerun."""
    monkeypatch.setenv("MIMO_API_KEY", "k")
    monkeypatch.setattr(ac, "_chunk_bounds", lambda w, c: [(0.0, 5.0)])
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 5.0)
    monkeypatch.setattr(ac, "_extract_span",
                        lambda wav, s, d, dest: dest.write_bytes(b"RIFF"))
    monkeypatch.setattr(ac, "_mimo_transcribe_chunk",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ac._run_mimo_asr(_args(backend="mimo-asr", api_key_env="MIMO_API_KEY"),
                         _wav(tmp_path))
    d = tmp_path / "_mimo_chunks_audio_16k"
    assert d.is_dir()                                   # cache survives...
    assert not list(d.glob("chunk_*.wav"))              # ...scratch wavs do not
    assert (d / "spans.json").is_file()


def test_mimo_run_rerun_pays_only_for_failed_chunks(tmp_path, monkeypatch):
    """The rerun after a mid-lecture failure must skip every chunk whose text
    is already cached — otherwise 'resume' re-pays the whole transcript."""
    monkeypatch.setenv("MIMO_API_KEY", "k")
    spans = [(0.0, 30.0), (30.0, 60.0)]
    monkeypatch.setattr(ac, "_chunk_bounds", lambda w, c: spans)
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 60.0)
    monkeypatch.setattr(ac, "_extract_span",
                        lambda wav, s, d, dest: dest.write_bytes(b"RIFF"))
    calls = {"n": 0}

    def flaky(base, model, key, path):
        calls["n"] += 1
        if calls["n"] == 1:          # first pass: chunk 0 fails
            raise RuntimeError("HTTP 500")
        return "第二段" if "0001" in path.stem else "第一段"

    monkeypatch.setattr(ac, "_mimo_transcribe_chunk", flaky)

    with pytest.raises(RuntimeError):
        ac._run_mimo_asr(_args(backend="mimo-asr", api_key_env="MIMO_API_KEY"),
                         _wav(tmp_path))
    first_pass = calls["n"]
    assert first_pass == 2           # both chunks attempted (concurrent map)

    segs, _ = ac._run_mimo_asr(
        _args(backend="mimo-asr", api_key_env="MIMO_API_KEY"), _wav(tmp_path))
    # chunk 0 was transcribed once (retry), chunk 1 served from cache
    assert calls["n"] == 3
    assert [s["text"] for s in segs] == ["第一段", "第二段"]
    assert not (tmp_path / "_mimo_chunks_audio_16k").exists()


def test_mimo_run_cache_is_discarded_when_chunking_changes(tmp_path, monkeypatch):
    """Cached texts were cut on the OLD chunk boundaries; after --chunk-seconds
    changes they would be spliced at wrong timestamps. The spans manifest
    guards this: mismatched span count wipes the cache."""
    monkeypatch.setenv("MIMO_API_KEY", "k")
    d = tmp_path / "_mimo_chunks_audio_16k"
    d.mkdir()
    (d / "t_0000.txt").write_text("旧切分的陈旧文本", encoding="utf-8")
    (d / "spans.json").write_text(json.dumps([[0.0, 90.0]]), encoding="utf-8")

    monkeypatch.setattr(ac, "_chunk_bounds", lambda w, c: [(0.0, 30.0)])
    monkeypatch.setattr(ac, "_ffprobe_duration", lambda p: 30.0)
    monkeypatch.setattr(ac, "_extract_span",
                        lambda wav, s, d_, dest: dest.write_bytes(b"RIFF"))
    monkeypatch.setattr(ac, "_mimo_transcribe_chunk",
                        lambda base, model, key, path: "新切分文本")

    segs, _ = ac._run_mimo_asr(
        _args(backend="mimo-asr", api_key_env="MIMO_API_KEY"), _wav(tmp_path))
    assert [s["text"] for s in segs] == ["新切分文本"]   # stale text NOT used


# --------------------------------------------------------------------------
# segment conversion + output schema
# --------------------------------------------------------------------------

def test_segs_from_openai_api_basic():
    raw = [types.SimpleNamespace(start=0.0, end=2.5, text=" Hello, "),
           types.SimpleNamespace(start=2.5, end=5.0, text="World.")]
    segs = ac._segs_from_openai_api(raw)
    assert segs == [{"start": 0.0, "end": 2.5, "text": "Hello,"},
                    {"start": 2.5, "end": 5.0, "text": "World."}]
    assert all(set(s) == {"start", "end", "text"} for s in segs)


def test_segs_from_faster_whisper_basic():
    raw = [types.SimpleNamespace(start=0.0, end=2.5, text=" Hello "),
           types.SimpleNamespace(start=2.5, end=5.0, text="World ")]
    segs = ac._segs_from_faster_whisper(raw)
    assert segs == [{"start": 0.0, "end": 2.5, "text": "Hello"},
                    {"start": 2.5, "end": 5.0, "text": "World"}]


def test_subtitles_json_schema_survives_a_round_trip(monkeypatch, tmp_path):
    """The contract build_knowledge.py and batch_run.py read. Both backends must
    produce it, unchanged."""
    segs = [{"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 1.0, "end": 2.0, "text": "B"}]
    for backend in ("faster-whisper", "mimo-asr"):
        meta = {"language": "zh", "language_probability": 0.99, "duration": 2.0}
        parsed = json.loads(json.dumps({**meta, "segments": segs},
                                       ensure_ascii=False))
        assert {"language", "language_probability", "duration",
                "segments"} <= set(parsed)
        assert all(set(s) == {"start", "end", "text"} for s in parsed["segments"])


def test_srt_and_vtt_formatting_unchanged():
    segs = [{"start": 0.0, "end": 2.5, "text": "Hello"},
            {"start": 2.5, "end": 5.123, "text": "World"}]
    srt = ac.to_srt(segs)
    assert srt.startswith("1\n")
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "00:00:02,500 --> 00:00:05,123" in srt
    vtt = ac.to_vtt(segs[:1])
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in vtt


# --------------------------------------------------------------------------
# main(): failure must be loud and must not fake a transcript
# --------------------------------------------------------------------------

def _run_main(tmp_path, monkeypatch, extra_args, runner=None, backend="faster-whisper"):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"not really a video")
    out = tmp_path / "out"

    def fake_extract(video, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        w = out_dir / "audio_16k.wav"
        w.write_bytes(b"RIFF" + b"\0" * 2048)
        return w

    monkeypatch.setattr(ac, "extract_wav", fake_extract)
    if runner is not None:
        monkeypatch.setitem(ac.BACKENDS, backend, runner)
    monkeypatch.setattr(sys, "argv", ["asr_caption.py", "--video", str(video),
                                      "--out-dir", str(out), *extra_args])
    return ac.main(), out


def test_main_reports_a_cloud_failure_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """No key -> non-zero, reason on stderr, and NO subtitles.json. A partially
    written transcript is how a failed run gets mistaken for a real one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc, out = _run_main(tmp_path, monkeypatch,
                        ["--backend", "openai-api", "--api-base", "https://x/v1",
                         "--api-model", "whisper-1"])
    assert rc == 5
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert not (out / "subtitles.json").exists()
    assert not (out / "subtitles.srt").exists()


def test_main_rejects_openai_api_without_a_model(tmp_path, monkeypatch, capsys):
    rc, out = _run_main(tmp_path, monkeypatch,
                        ["--backend", "openai-api", "--api-base", "https://x/v1"])
    assert rc == 3
    assert "api-model" in capsys.readouterr().err
    assert not (out / "subtitles.json").exists()


def test_main_writes_subtitles_on_success(tmp_path, monkeypatch, capsys):
    segs = [{"start": 0.0, "end": 2.0, "text": "今天讲音阶"}]
    meta = {"language": "zh", "language_probability": 0.99, "duration": 2.0}
    rc, out = _run_main(tmp_path, monkeypatch,
                        ["--backend", "mimo-asr"],
                        runner=lambda a, w: (segs, meta), backend="mimo-asr")
    assert rc == 0
    data = json.loads((out / "subtitles.json").read_text(encoding="utf-8"))
    assert data["segments"] == segs
    assert data["language"] == "zh"
    assert (out / "subtitles.srt").exists()
    # wav cleanup applies to the cloud backends too
    assert not (out / "audio_16k.wav").exists()


def test_main_warns_loudly_on_zero_segments(tmp_path, monkeypatch, capsys):
    """Zero segments is legitimate for a wordless clip, but it is also exactly
    what a silently-broken run looks like — so it has to say so."""
    rc, out = _run_main(tmp_path, monkeypatch,
                        ["--backend", "mimo-asr"],
                        runner=lambda a, w: ([], {"language": "zh",
                                                 "language_probability": 1.0,
                                                 "duration": 5.0}),
                        backend="mimo-asr")
    assert rc == 0
    assert "0 segments" in capsys.readouterr().err
    assert json.loads((out / "subtitles.json").read_text(
        encoding="utf-8"))["segments"] == []


def test_main_keeps_the_wav_only_with_keep_wav(tmp_path, monkeypatch):
    segs = [{"start": 0.0, "end": 2.0, "text": "x"}]
    meta = {"language": "zh", "language_probability": 1.0, "duration": 2.0}
    rc, out = _run_main(tmp_path, monkeypatch,
                        ["--backend", "mimo-asr", "--keep-wav"],
                        runner=lambda a, w: (segs, meta), backend="mimo-asr")
    assert rc == 0
    assert (out / "audio_16k.wav").exists()

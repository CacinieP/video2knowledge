#!/usr/bin/env python3
"""test_llm_degradation.py — a knowledge doc must never silently be a placeholder.

The failure this pins: when the text model is unreachable or absent,
`ask_llm` returned None, `build_analysis` fell back to placeholder text, and
`build_knowledge.py` wrote a complete-looking knowledge.md/.html/.docx/.pdf/
cards.csv and returned 0. In a batch that is the worst shape of bug — it is
indistinguishable from success from the outside.

It was hit for real: OLLAMA_MODELS pointed at a directory that did not hold the
weights, so /api/tags answered 200 with an empty list and /api/generate returned
404 "model not found". Two knowledge docs were produced that contained nothing
but "(本地模型不可用...)" and "启用本地模型自动生成问答", and the run reported
success.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import build_knowledge as bk  # noqa: E402


def _subs(tmp_path: Path, n: int = 5) -> Path:
    # The text has to clear build_knowledge's speech-density floor, otherwise
    # these tests would exercise the no-speech path instead of the degradation
    # path they are about. 5 segments x 14 meaningful characters = 70, over the
    # floor of 40 with room to spare. A real lecture segment is about this long.
    p = tmp_path / "subtitles.json"
    p.write_text(json.dumps({
        "language": "zh", "duration": 60.0,
        "segments": [{"start": i * 10.0, "end": i * 10.0 + 9.0,
                      "text": f"第{i}句是测试用的字幕内容示例"}
                     for i in range(n)],
    }, ensure_ascii=False), encoding="utf-8")
    return p


def _argv(tmp_path: Path, subs: Path, *extra: str) -> list[str]:
    return ["build_knowledge.py", "--subtitles", str(subs),
            "--out-dir", str(tmp_path / "out"), "--format", "knowledge",
            "--lang", "zh", "--host", "http://127.0.0.1:1",  # nothing listening
            *extra]


# --- the marker itself -------------------------------------------------------

def test_fallback_summary_is_recognised_as_degraded():
    fields = bk._heuristic_fallback("a\nb\nc", {k: "" for k in
                                                ("summary", "timeline", "key_points",
                                                 "qa", "glossary", "bullets")})
    assert bk.is_degraded(fields)


def test_real_analysis_is_not_flagged():
    good = {"summary": "这段视频讲音阶指法。", "timeline": "- t",
            "key_points": "- k", "qa": "- Q", "glossary": "- g", "bullets": "- b"}
    assert not bk.is_degraded(good)


def test_is_degraded_survives_junk_input():
    assert bk.is_degraded(None)
    assert bk.is_degraded("not a dict")
    assert bk.is_degraded({})


# --- end to end --------------------------------------------------------------

def test_unreachable_model_exits_nonzero_but_still_writes_the_file(
        tmp_path, monkeypatch, capsys):
    subs = _subs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, subs))
    rc = bk.main()
    err = capsys.readouterr().err
    # artifacts are kept for inspection...
    assert (tmp_path / "out" / "knowledge.md").is_file()
    # ...but the run must not report success
    assert rc == 4, f"expected exit 4 (degraded), got {rc}"
    assert "DEGRADED" in err
    # and the message must say what to actually check
    assert "api/tags" in err


def test_written_document_carries_the_marker_a_driver_can_test_for(
        tmp_path, monkeypatch):
    subs = _subs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, subs))
    bk.main()
    md = (tmp_path / "out" / "knowledge.md").read_text(encoding="utf-8")
    assert bk.DEGRADED_MARKER in md


def test_ask_llm_reports_the_failure_instead_of_returning_none_silently(
        capsys):
    """A silent None is what let this run to completion looking healthy."""
    assert bk.ask_llm("http://127.0.0.1:1", "some-model", "hi") is None
    err = capsys.readouterr().err
    assert "[llm]" in err and "some-model" in err


def test_no_cards_when_degraded(tmp_path, monkeypatch):
    """Placeholder QA must not become Anki cards — that is how a degraded run
    looks like a finished deliverable one level downstream."""
    subs = _subs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, subs, "--format", "all"))
    bk.main()
    csv = tmp_path / "out" / "cards.csv"
    if csv.is_file():
        body = csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(body) <= 1          # header only


# --- knowledge_doc_status: catching invented summaries -----------------------

# What a 2B model actually returned for a 0-segment piano-performance video when
# handed an empty transcript. Well-formed, plausible, and entirely invented.
_HALLUCINATED = {
    "summary": "这段视频详细讲解了如何使用Python的requests库来发送HTTP请求，"
               "包括获取网页内容、处理JSON数据和设置请求头等关键操作。",
    "timeline": "- [00:03] 展示产品使用场景",
    "key_points": "- 讲解 HTTP 请求", "qa": "- Q: ...", "glossary": "- g",
    "bullets": "- b",
}
_NO_SPEECH = {
    "summary": f"**{bk.NO_SPEECH_MARKER}** 本视频没有可用的旁白/讲解，"
               "因此无法生成文字总结。",
    "timeline": "_(无语音时间轴)_", "key_points": "- (无语音内容)",
    "qa": "- (无语音内容)", "glossary": "- (无语音内容)", "bullets": "- (无语音内容)",
}


def test_empty_transcript_with_an_invented_summary_is_degraded():
    """The failure no marker search can find: the summary string itself is
    fluent and plausible, so only the segment count gives it away."""
    assert bk.knowledge_doc_status(_HALLUCINATED, 0) == bk.STATUS_DEGRADED


def test_empty_transcript_with_the_placeholder_is_no_speech_not_degraded():
    """That placeholder IS the correct answer for a wordless video — a driver
    must not retry it or count it as a deliverable, but it is not a failure."""
    assert bk.knowledge_doc_status(_NO_SPEECH, 0) == bk.STATUS_NO_SPEECH


def test_a_real_document_with_segments_is_ok():
    good = dict(_HALLUCINATED)
    good["summary"] = "这段视频讲解了音阶与琶音的正确指法与练习方法。"
    assert bk.knowledge_doc_status(good, 235) == bk.STATUS_OK


def test_unreachable_model_is_degraded_even_with_segments():
    fields = bk._heuristic_fallback("a\nb", {k: "" for k in
                                            ("summary", "timeline", "key_points",
                                             "qa", "glossary", "bullets")})
    assert bk.knowledge_doc_status(fields, 235) == bk.STATUS_DEGRADED


def test_status_constants_are_distinct():
    assert len({bk.STATUS_OK, bk.STATUS_DEGRADED, bk.STATUS_NO_SPEECH}) == 3

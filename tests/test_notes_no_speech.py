"""A wordless video still gets a note — just not a fabricated one.

`build_notes.py` asked the text model to condense each node's narration window
into a "note". On the no-speech half of a music-course library that window held
what Paraformer made of a piano: "嗯嗯，背". The model did not decline. It
produced, among 125 fabricated node bodies across 11 notes:

    "用艾宾浩斯记忆曲线安排复习时间。"

— an Ebbinghaus spaced-repetition tip, from a two-syllable transcript.

This is the same fabrication #17 stopped in `build_knowledge.py`, one layer
over. There the fix was to refuse to build a document at all. Here the right
answer is different and worth stating: **the frames are real.** A score on
screen is exactly what an illustrated note is for, and the VLM reads it
honestly. What cannot be produced is a sentence claiming to condense narration
that does not exist.

So: keep the images and the 画面 descriptions, drop the note text and the 原声
citation, and say so once at the top of the note.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_notes as bn  # noqa: E402

META = {"title": "4536251王道和弦进行加花技巧", "video": "v.mp4",
        "duration": "00:50", "date": "2026-01-01"}


def _sec(t=0.0, title="", note="", desc="乐谱显示大调音阶", excerpt="", speech=True):
    return {"t": t, "file": "frames/f.jpg", "title": title, "note": note,
            "desc": desc, "excerpt": excerpt, "n_lines": 1, "speech": speech}


JUNK = [{"start": 0.0, "end": 5.0, "text": "嗯嗯，背"},
        {"start": 5.0, "end": 9.0, "text": "嗯嗯嗯嗯"}]
REAL = [{"start": 0.0, "end": 5.0,
         "text": "今天我们讲一下属七和弦的解决方式，它要导向上主和弦，"
                 "这样听上去才不会突兀。"},
        {"start": 5.0, "end": 9.0,
         "text": "共同音保持不动是最自然的连接方式，也是和声写作的基本功。"}]


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def test_junk_transcript_is_not_usable_speech():
    assert not bn.has_usable_speech(JUNK)


def test_real_transcript_is_usable_speech():
    assert bn.has_usable_speech(REAL)


# --------------------------------------------------------------------------
# the note body
# --------------------------------------------------------------------------

def test_a_wordless_note_keeps_the_frames_and_the_frame_description(tmp_path):
    md = bn.render_markdown([_sec(speech=False)], META, tmp_path)
    # relpath rewrites the ref against tmp_path, so match the file name
    assert "f.jpg" in md and "![00:00]" in md
    assert "**画面**：乐谱显示大调音阶" in md


def test_a_wordless_note_states_itself_at_the_top(tmp_path):
    """Otherwise a run of 画面-only nodes reads as a broken export."""
    md = bn.render_markdown([_sec(speech=False)], META, tmp_path)
    assert "纯画面笔记" in md
    # and only once, not on every node
    assert md.count("纯画面笔记") == 1


def test_a_normal_note_does_not_carry_the_banner(tmp_path):
    md = bn.render_markdown([_sec(speech=True, title="属七和弦的解决")],
                            META, tmp_path)
    assert "纯画面笔记" not in md


def test_a_wordless_node_never_quotes_the_filler(tmp_path):
    """"原声：嗯嗯，背" presents a hallucination as a citation."""
    md = bn.render_markdown(
        [_sec(speech=False, excerpt="", title="", note="")], META, tmp_path)
    # no blockquote under any node (the banner text mentions the word, so
    # match the rendered form rather than the bare string)
    assert "> 原声：" not in md
    assert "嗯" not in md


def test_a_normal_node_still_quotes_its_narration(tmp_path):
    md = bn.render_markdown(
        [_sec(speech=True, title="小标题", note="要点。", excerpt="属七和弦")],
        META, tmp_path)
    assert "> 原声：属七和弦" in md


def test_the_distilled_view_also_states_it(tmp_path):
    md = bn.render_markdown([_sec(speech=False)], META, tmp_path, verbatim=False)
    assert "纯画面笔记" in md


def test_heading_falls_back_to_the_frame_description(tmp_path):
    """With no title and no note, the description is the node's whole content."""
    md = bn.render_markdown([_sec(speech=False, title="", note="",
                                  desc="乐谱显示属七和弦")], META, tmp_path)
    assert "## [00:00] 乐谱显示属七和弦" in md


# --------------------------------------------------------------------------
# build_sections must not call the model at all
# --------------------------------------------------------------------------

def _frames(n=3):
    return [{"t": float(i * 10), "file": f"f{i}.jpg"} for i in range(n)]


def test_no_llm_call_for_a_wordless_video(tmp_path, monkeypatch):
    """The strongest form of the guarantee: the model is never asked."""
    monkeypatch.setattr(bn, "section_note",
                        lambda *a, **k: pytest.fail("LLM must not be asked"))
    monkeypatch.setattr(bn, "describe_frame", lambda *a, **k: "乐谱显示大调音阶")
    secs = bn.build_sections("h", "model", "vlm", _frames(), JUNK,
                             describe=True, lang="zh", workers=1)
    assert len(secs) == 3
    assert all(not s["title"] and not s["note"] for s in secs)
    assert all(s["speech"] is False for s in secs)


def test_llm_still_called_for_a_normal_video(monkeypatch):
    seen = {"n": 0}

    def fake(host, model, narration, lang):
        seen["n"] += 1
        return "小标题", "提炼要点。"

    monkeypatch.setattr(bn, "section_note", fake)
    monkeypatch.setattr(bn, "describe_frame", lambda *a, **k: "乐谱")
    secs = bn.build_sections("h", "model", "vlm", _frames(), REAL,
                             describe=True, lang="zh", workers=1)
    assert seen["n"] == 3
    assert all(s["speech"] for s in secs)
    assert all(s["title"] == "小标题" for s in secs)


def test_excerpts_survive_for_a_normal_video(monkeypatch):
    """The fix must not cost real videos their quotes."""
    monkeypatch.setattr(bn, "section_note", lambda *a, **k: ("t", "n"))
    monkeypatch.setattr(bn, "describe_frame", lambda *a, **k: "d")
    # frames aligned with the segments: t=0 covers 0-5, t=5 covers 5-9
    frames = [{"t": 0.0, "file": "f0.jpg"}, {"t": 5.0, "file": "f1.jpg"}]
    secs = bn.build_sections("h", "m", "v", frames, REAL,
                             describe=True, lang="zh", workers=1)
    assert all(s["excerpt"] for s in secs)
    assert "属七和弦" in secs[0]["excerpt"]
    assert "共同音" in secs[1]["excerpt"]

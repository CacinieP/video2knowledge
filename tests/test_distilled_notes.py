"""Two illustrated-note views, and a frame description aimed at study value.

Both changes came from measuring the output rather than from reading it.

1. The quoted narration is 53% of the characters in a note (measured over 295
   nodes). A reader revising wants the knowledge; a reader checking the
   teacher's exact wording wants the quote. They are not the same document, so
   they are now two files produced from one pass — no extra model calls.

2. The VLM prompt asked for "主体物品、人物动作、屏幕文字或演示步骤", and
   "人物动作" won every time: measured output was "女士弹琴，手势讲解，字幕
   提示优秀突出" — none of which helps anyone revise. Re-prioritised toward
   score/board content, with an explicit instruction not to describe people.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_notes as bn  # noqa: E402


def _section(t=0.0, title="要点", note="这是提炼后的知识点。",
             desc="乐谱显示 C 大调和弦。", excerpt="原声逐字内容。",
             file="frames/frame_000001.jpg"):
    return {"t": t, "file": file, "desc": desc, "note": note,
            "title": title, "excerpt": excerpt}


META = {"title": "测试课", "video": "a.mp4", "duration": "10:00",
        "date": "2026-01-01"}


# --------------------------------------------------------------------------
# the two views
# --------------------------------------------------------------------------

def test_verbatim_view_keeps_the_quote():
    md = bn.render_markdown([_section()], META, Path("."), verbatim=True)
    assert "> 原声：原声逐字内容。" in md
    assert "图文笔记" in md
    assert "纯享版" not in md


def test_distilled_view_drops_the_quote():
    md = bn.render_markdown([_section()], META, Path("."), verbatim=False)
    assert "原声" not in md
    assert "原声逐字内容" not in md


def test_distilled_view_keeps_everything_else():
    md = bn.render_markdown([_section()], META, Path("."), verbatim=False)
    assert "乐谱显示 C 大调和弦。" in md, "the frame description is study value"
    assert "这是提炼后的知识点。" in md
    assert "frame_000001.jpg" in md
    assert "## [00:00] 要点" in md


def test_distilled_view_points_at_the_full_one():
    md = bn.render_markdown([_section()], META, Path("."), verbatim=False)
    assert "notes.md" in md
    assert "纯享版" in md


def test_distilled_is_materially_shorter():
    """The whole reason for the second file: ~42% smaller on the real corpus."""
    secs = [_section(t=i * 60.0, excerpt="逐字内容" * 40) for i in range(10)]
    full = bn.render_markdown(secs, META, Path("."), verbatim=True)
    light = bn.render_markdown(secs, META, Path("."), verbatim=False)
    assert len(light) < len(full) * 0.75


def test_both_views_agree_on_node_count():
    secs = [_section(t=i * 60.0) for i in range(5)]
    full = bn.render_markdown(secs, META, Path("."), verbatim=True)
    light = bn.render_markdown(secs, META, Path("."), verbatim=False)
    pat = re.compile(r"(?m)^## \[")
    assert len(pat.findall(full)) == len(pat.findall(light)) == 5


def test_default_stays_verbatim():
    """A caller that passes nothing must keep today's behaviour."""
    assert "> 原声" in bn.render_markdown([_section()], META, Path("."))


# --------------------------------------------------------------------------
# the frame-description prompt
# --------------------------------------------------------------------------

def test_prompt_prioritises_teaching_content():
    p = bn.DESC_PROMPT
    assert "乐谱" in p or "板书" in p
    assert "屏幕文字" in p


def test_prompt_forbids_describing_people():
    """The failure it exists to prevent, quoted verbatim in the prompt."""
    assert "不要描述人物动作" in bn.DESC_PROMPT
    assert "女士弹琴" in bn.DESC_PROMPT, "names the bad output it is fixing"


def test_prompt_has_a_fallback_for_boring_frames():
    """A frame with nothing readable still needs an answer, not a blank."""
    assert "没有可读教学信息" in bn.DESC_PROMPT or "无字幕" in bn.DESC_PROMPT


def test_prompt_still_forbids_reasoning_leak():
    assert "不要推理过程" in bn.DESC_PROMPT

#!/usr/bin/env python3
"""test_path2_coverage.py — a Path 2 knowledge doc must see the whole transcript.

The bug: `build_analysis` sliced `raw_text[:char_limit]` with a default of 8000
BEFORE the map-reduce stage. Two consequences, both silent:

1. Every lecture longer than ~25 minutes lost its tail. Measured on a 35-minute
   lesson: 12241 chars of subtitle, 8000 reached the model, 35% of the content
   never seen, and the document still looked complete — the summary simply
   described the first half of the video.
2. `long_mode = len(sub) > CHUNK * 1.5` (13500) could never be true when sub was
   capped at 8000, so the map-reduce chunker was dead code on the plain Path 2
   path. The machinery was written, tested, and unreachable.

`char_limit` is a speed knob, not a coverage knob. With no --char-limit, the
whole transcript must reach the chunker.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import build_knowledge as bk  # noqa: E402


class _StubCache:
    def __init__(self, prompts: list):
        self.prompts = prompts

    def ask(self, host, prompt):
        self.prompts.append(prompt)
        return "结果"

    def flush(self):
        pass


def _transcript(n_chars: int) -> str:
    """A raw_text shaped like the real thing: one '[mm:ss] ' line per segment."""
    lines, i = [], 0
    while sum(len(x) for x in lines) < n_chars:
        lines.append(f"[{i // 60:02d}:{i%60:02d}] 第{i}行字幕内容")
        i += 1
    return "\n".join(lines)


def _run(raw: str, char_limit: int = 0):
    """Call build_analysis with a stubbed model; return the prompts it produced."""
    prompts: list = []
    old_ping, old_ask = bk.ping, bk.ask_llm
    bk.ping, bk.ask_llm = (lambda h: True), (lambda h, m, p: "结果")
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            bk.build_analysis("http://x", "m", raw, "s", lang="zh",
                              char_limit=char_limit,
                              cache=_StubCache(prompts))
    finally:
        bk.ping, bk.ask_llm = old_ping, old_ask
    return prompts, err.getvalue()


# --- the whole transcript must reach the model -------------------------------

@pytest.mark.parametrize("size", [9000, 20000, 60000])
def test_the_tail_of_the_transcript_reaches_some_prompt(size):
    """The property that actually matters: no part of the video is invisible to
    the model. Checking the chunker is an implementation detail — checking the
    prompts is the contract."""
    raw = _transcript(size)
    tail_line = raw.splitlines()[-1]
    prompts, _ = _run(raw)
    assert prompts, "no LLM call was made"
    assert any(tail_line in p for p in prompts), "the last line never reached the model"


def test_long_mode_actually_engages():
    """The regression that made map-reduce unreachable: a 13500 threshold
    against an 8000 cap meant the branch could never run on Path 2."""
    raw = _transcript(20000)
    prompts, _ = _run(raw)
    # chunked => the same chunk is asked about once per field (6 fields) plus
    # summary/reduce passes, so clearly more calls than the 6 fields of a short
    # transcript would produce
    assert len(prompts) > 6


def test_short_transcript_is_asked_about_directly():
    raw = _transcript(4000)
    prompts, _ = _run(raw)
    assert 0 < len(prompts) <= 8
    assert any(raw.splitlines()[-1] in p for p in prompts)


# --- an explicit cap is still honoured, and now says so ----------------------

def test_explicit_char_limit_drops_the_tail():
    raw = _transcript(20000)
    head_line, tail_line = raw.splitlines()[0], raw.splitlines()[-1]
    prompts, _ = _run(raw, char_limit=8000)
    assert any(head_line in p for p in prompts), "the kept head must still be there"
    assert not any(tail_line in p for p in prompts), "the tail should be gone"


def test_truncation_is_never_silent():
    raw = _transcript(20000)
    _, err = _run(raw, char_limit=8000)
    assert "char_limit=8000" in err
    assert str(len(raw) - 8000) in err, "the message must say how much was dropped"


def test_no_warning_when_nothing_is_dropped():
    _, err = _run(_transcript(4000), char_limit=0)
    assert "truncated" not in err


# --- the default itself ------------------------------------------------------

def test_default_char_limit_is_unbounded():
    """Guards the specific value that caused this: a default low enough to
    truncate a real lecture is the bug, whatever number it is."""
    import inspect
    sig = inspect.signature(bk.build_analysis)
    assert sig.parameters["char_limit"].default == 0

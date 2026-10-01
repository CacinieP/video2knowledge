"""Tests for the robustness fixes: LLM-output label stripping.

The frame-extraction and ollama-call fixes need ffmpeg / a live ollama server
and are covered by the real-video regression in tests/; the label strip is pure
string logic and is cheap to pin down here.
"""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_notes import _strip_line_label  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("第一行：视谱能力训练", "视谱能力训练"),
    ("第二行：先分手再合手", "先分手再合手"),
    ("  第一行：  有前后空格  ", "有前后空格"),
    ("第一行:半角冒号", "半角冒号"),
    ("Line 1: warm up", "warm up"),
])
def test_strip_line_label(raw, expected):
    assert _strip_line_label(raw) == expected


@pytest.mark.parametrize("raw", [
    "视谱能力训练",            # no label at all -> untouched
    "第一行的重要不是标签",      # label must be at the START only
    "见 Line 1: 的引用",        # mid-string, not a prefix
    "line 2 - dash form",     # only ':' is a label separator, not '-'
])
def test_strip_line_label_leaves_normal_text(raw):
    assert _strip_line_label(raw) == raw

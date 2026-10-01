"""Tests for card de-duplication in build_knowledge.py.

`cards_from_qa` used to be a pure parser, so whatever the model produced went
straight into cards.csv. A lecture is summarised with a map/reduce over
chunks and the model re-asks the same question in each chunk, because a
teacher restates the same fact all lesson. Measured on the 286-video course:
145 of 1608 cards (9%) were repeats, worst case 67 rows for 6 distinct
questions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_knowledge as bk  # noqa: E402


# --------------------------------------------------------------------------
# parsing still works
# --------------------------------------------------------------------------

def test_parses_markdown_pairs():
    rows = bk.cards_from_qa("Q: 什么是全音？\nA: 相邻两个白键之间。\n"
                            "Q: 什么是半音？\nA: 白键与黑键之间。", "s.json")
    assert rows == [
        ["什么是全音？", "相邻两个白键之间。", "", "", "s.json"],
        ["什么是半音？", "白键与黑键之间。", "", "", "s.json"],
    ]


def test_parses_dict_list():
    rows = bk.cards_from_qa([{"q": "Q1", "a": "A1"}, {"question": "Q2", "answer": "A2"}], "s")
    assert [r[0] for r in rows] == ["Q1", "Q2"]


def test_parses_nested_string_list():
    rows = bk.cards_from_qa(["Q: one\nA: 1", "Q: two\nA: 2"], "s")
    assert [r[0] for r in rows] == ["one", "two"]


# --------------------------------------------------------------------------
# the defect
# --------------------------------------------------------------------------

def test_exact_duplicates_are_removed():
    rows = bk.cards_from_qa("Q: 什么是全音？\nA: A\nQ: 什么是全音？\nA: B", "s")
    assert len(rows) == 1
    assert rows[0][1] == "A", "first wins: a complete answer keeps its slot"


def test_punctuation_and_spacing_variants_collapse():
    """The model rephrases; punctuation and spacing must not defeat dedup."""
    rows = bk.cards_from_qa(
        "Q: 什么是全音？\nA: A\n"
        "Q: 什么是 全音?\nA: B\n"
        "Q:  什么是全音\nA: C",
        "s")
    assert len(rows) == 1


def test_distinct_questions_survive():
    rows = bk.cards_from_qa(
        "Q: 白键之间是什么音程？\nA: 全音\n"
        "Q: 黑键之间是什么音程？\nA: 全音\n"
        "Q: 白键和黑键之间呢？\nA: 半音",
        "s")
    assert len(rows) == 3


def test_empty_question_is_dropped():
    rows = bk.cards_from_qa("Q: \nA: orphan answer\nQ: real?\nA: yes", "s")
    assert [r[0] for r in rows] == ["real?"]


def test_repeated_trailing_question():
    """A 'Q:' with no following 'A:' must not duplicate the previous card."""
    rows = bk.cards_from_qa("Q: a?\nA: 1\nQ: a?", "s")
    assert len(rows) == 1


# --------------------------------------------------------------------------
# the measured regression
# --------------------------------------------------------------------------

def test_real_67_row_lecture_collapses_to_6():
    """Verbatim shape of 1-3）全音与半音: 67 rows, 6 distinct questions."""
    reps = [
        "钢琴上相邻的两个白键之间是什么音程关系？",
        "钢琴上相邻的两个黑键之间是什么音程关系？",
        "钢琴上相邻的两个白键和黑键之间是什么音程关系？",
    ]
    body = "".join(f"Q: {q}\nA: 答\n" for q in reps * 22)
    body += "Q: 白键与黑键这一节主要讲的是什么？\nA: 概述\n"
    parsed = bk._parse_qa_rows(body, "s")
    assert len(parsed) == 67
    rows = bk.cards_from_qa(body, "s")
    assert len(rows) == 4


def test_dedup_reports_what_it_dropped():
    body = "Q: a?\nA: 1\nQ: a?\nA: 2\nQ: b?\nA: 3\nQ: \nA: 4"
    parsed = bk._parse_qa_rows(body, "s")
    rows, dropped = bk._dedup_cards(parsed)
    assert len(rows) == 2
    assert dropped == 2


def test_dedup_is_stable_across_chunk_ordering():
    """map/reduce concatenates chunk results; order must not change the count."""
    a, b = "Q: x?\nA: 1", "Q: y?\nA: 2"
    assert len(bk.cards_from_qa([a, b], "s")) == 2
    assert len(bk.cards_from_qa([b, a], "s")) == 2


def test_card_key_strips_scaffolding():
    assert bk._card_key("什么是 全音？") == bk._card_key("什么是全音")
    assert bk._card_key("Q: 音程?") != bk._card_key("Q: 节奏?")

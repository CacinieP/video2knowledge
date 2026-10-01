#!/usr/bin/env python3
"""test_funasr_adapter.py — regressions for scripts/asr_funasr.py.

The functions under test are all pure (text/timestamps in, segments out), so
this suite needs neither torch nor funasr installed and stays green on CI.

What each case protects, all of it learned from a real run over a 61.5-hour
course library:

* timestamp alignment — the pipeline's `text` is LONGER than its `timestamp`
  list, because ct-punc inserts punctuation into a string the ASR timestamped
  before punctuation existed. Indexing both with one counter walks off the end
  of the list and silently truncates the tail of every transcript.
* segment boundaries — cut only where both sides have a real time, so nothing
  collapses onto t=0.
* output compatibility — the whole point of the script is that it is a drop-in
  for asr_caption.py, so the SRT/VTT/JSON must match byte for byte.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from asr_funasr import (  # noqa: E402
    align_text_ts, fmt_ts, load_hotwords, segment_from_chars, to_srt, to_vtt,
)


# --- timestamp alignment -----------------------------------------------------

def test_punctuation_inserted_by_ctpunc_consumes_no_timestamp():
    """text is 749 chars but timestamp has 689 entries on real output — the
    60 extra are punctuation that ct-punc added after the ASR ran. The naive
    "one index walks both" version drops the last 60 characters."""
    text = "你好，你好。"                        # 6 chars: 4 hanzi + 2 punct
    ts = [[0, 100], [100, 200], [200, 300], [300, 400]]   # only the 4 hanzi
    chars, out = align_text_ts(text, ts)
    assert chars == text                    # nothing dropped
    assert len(out) == len(text)
    # punctuation gets a zero-width stamp at the preceding end time
    assert out[2] == [200.0, 200.0]         # the '，' — zero width, no stamp spent
    assert out[0] == [0.0, 100.0]
    assert out[3] == [200.0, 300.0]          # 3rd hanzi
    assert out[4] == [300.0, 400.0]          # 4th hanzi
    assert out[5] == [400.0, 400.0]          # the '。'


def test_space_separated_tokens_are_removed():
    """SeACo-Paraformer returns '记 忆 力 呢'; timestamps are per token, so
    keeping the spaces would misalign every char after the first."""
    text = "记 忆 力 呢"
    ts = [[0, 100], [100, 200], [200, 300], [300, 400]]
    chars, out = align_text_ts(text, ts)
    assert chars == "记忆力呢"
    assert len(out) == 4
    assert out[3] == [300.0, 400.0]


def test_short_timestamp_list_does_not_invent_times():
    """Defensive: a real character with no timestamp must be dropped, not given
    a fabricated zero-width stamp — otherwise the tail of a transcript carries
    text pinned to t=0."""
    chars, out = align_text_ts("一二三四五", [[0, 10], [10, 20]])
    assert chars == "一二"
    assert len(out) == 2


# --- segmentation ------------------------------------------------------------

def _ts(n, step=100):
    return [[i * step, (i + 1) * step] for i in range(n)]


def test_segments_carry_real_start_and_end():
    segs = segment_from_chars("今天讲音阶的指法。踏板很重要。", _ts(13))
    assert segs
    for s in segs:
        assert s["end"] > s["start"] >= 0
        assert s["text"]


def test_sentence_ending_on_punctuation_is_not_dropped():
    """The unit bug this protects: inserted punctuation used to carry a
    pre-divided (second-valued) stamp, so `end` came out ~1000x too small,
    every punctuation-terminated segment failed `end > start`, and the whole
    sentence vanished. Nothing warned; the doc just quietly got shorter."""
    segs = segment_from_chars("第一句话。第二句话。", _ts(8))
    assert "".join(s["text"] for s in segs).replace(" ", "") == "第一句话。第二句话。"


def test_segments_sort_and_do_not_overlap_backwards():
    segs = segment_from_chars("第一句。第二句。第三句。", _ts(12))
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)
    for a, b in zip(segs, segs[1:]):
        assert b["start"] >= a["start"] - 1e-6


def test_sentence_punctuation_ends_a_segment():
    segs = segment_from_chars("一二三四五六七八。", _ts(9))
    assert any(s["text"].endswith("。") for s in segs)


def test_long_run_without_punctuation_is_still_split():
    """A 40-char sentence with no 。 must not become one 40-char subtitle."""
    text = "啊" * 40
    segs = segment_from_chars(text, _ts(40))
    assert len(segs) >= 2


def test_empty_input_is_not_an_error():
    assert segment_from_chars("", []) == []
    assert segment_from_chars("字", []) == []


# --- output compatibility with asr_caption.py --------------------------------

def test_srt_and_vtt_match_the_asr_caption_format():
    segs = [{"start": 0.0, "end": 1.5, "text": "第一句"},
            {"start": 2.25, "end": 3.0, "text": "第二句"}]
    assert to_srt(segs) == (
        "1\n00:00:00,000 --> 00:00:01,500\n第一句\n\n"
        "2\n00:00:02,250 --> 00:00:03,000\n第二句\n")
    assert to_vtt(segs) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.500\n第一句\n\n"
        "00:00:02.250 --> 00:00:03.000\n第二句\n")


def test_fmt_ts_matches_asr_caption_rounding():
    assert fmt_ts(0) == "00:00:00,000"
    assert fmt_ts(1.5) == "00:00:01,500"
    assert fmt_ts(3661.007, sep=".") == "01:01:01.007"
    # hours are not wrapped at 24 — a 10-hour lecture must not print "10:00:00"
    assert fmt_ts(359999.999) == "99:59:59,999"


# --- hotwords ----------------------------------------------------------------

def test_load_hotwords_accepts_the_same_specs_as_asr_caption():
    assert load_hotwords("背谱,视谱,音阶") == "背谱 视谱 音阶"
    assert load_hotwords("背谱、视谱；音阶") == "背谱 视谱 音阶"
    assert load_hotwords("背谱 视谱 音阶") == "背谱 视谱 音阶"
    assert load_hotwords("背谱,背谱,视谱") == "背谱 视谱"      # dedupe, keep order
    assert load_hotwords("") is None
    assert load_hotwords(None) is None


def test_load_hotwords_from_file(tmp_path):
    f = tmp_path / "terms.txt"
    f.write_text("背谱\n视谱\n\n音阶\n", encoding="utf-8")
    assert load_hotwords("@" + str(f)) == "背谱 视谱 音阶"


# --- drop-in guarantee -------------------------------------------------------

def test_subtitles_json_schema_is_identical_to_asr_caption(tmp_path, monkeypatch):
    """build_knowledge/fuse/gen_apkg read these exact keys; a renamed field
    would break every downstream script while every test still passed."""
    from asr_funasr import write_outputs
    segs = [{"start": 0.0, "end": 1.0, "text": "测试"}]
    write_outputs(tmp_path, segs, 1.0, "zh")
    d = json.loads((tmp_path / "subtitles.json").read_text(encoding="utf-8"))
    assert set(d) == {"language", "language_probability", "duration", "segments"}
    assert set(d["segments"][0]) == {"start", "end", "text"}
    for f in ("subtitles.srt", "subtitles.vtt", "subtitles.json"):
        assert (tmp_path / f).is_file()
    # provenance marker so a later run can tell engines apart
    assert "funasr" in (tmp_path / "asr_engine.txt").read_text(encoding="utf-8")

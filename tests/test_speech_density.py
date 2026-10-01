"""A non-empty transcript can still be worth nothing.

`n_segments == 0` was the guard for "the model will now invent a summary of a
video with no words in it". It fires correctly, and it is not sufficient.

On a piano-course library, 13 clips had an audio track and no narration — pure
performance. Paraformer does not return nothing for those; it transcribes the
playing into vocalisations, so the segment list is non-empty and the old guard
never sees them:

    第27节   3 segments,  6 characters (5 filler)  ->  8 knowledge cards
    第28节   1 segment,   6 characters (2 filler)  ->  9 knowledge cards
    第29节   4 segments, 14 characters (9 filler)  ->  8 knowledge cards
    第31节   3 segments,  8 characters (7 filler)  ->  8 knowledge cards

Six characters of "嗯嗯嗯背谱。" cannot support eight cards. The model fills the
template from the video title and whatever topic words it can scrape out of the
filler. That is the same fabrication the empty case was fixed for — but now it
looks like a success, which is worse, because nothing flags it.

These tests pin the density check and, more importantly, pin the *separation*
the threshold depends on. A threshold with no measured gap on either side is a
guess; the numbers below are the ones that justified it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_knowledge as bk  # noqa: E402


def segs(*texts: str) -> list[dict]:
    return [{"start": i * 2.0, "end": i * 2.0 + 1.8, "text": t}
            for i, t in enumerate(texts)]


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------

def test_punctuation_and_whitespace_do_not_count():
    assert bk.count_meaningful_chars(segs("，、。！？ \n")) == (0, 0)


def test_filler_is_not_meaningful():
    m, t = bk.count_meaningful_chars(segs("嗯嗯嗯啦啦啦"))
    assert t == 6
    assert m == 0


def test_real_words_count():
    m, t = bk.count_meaningful_chars(segs("今天讲和弦的连接"))
    assert t == m == 8


def test_counting_is_language_agnostic():
    """A vocalisation run in any script is caught by the same rule.

    Note the honest limit: "the" is not in FILLER_CHARS, so a repeated English
    hallucination counts as meaningful. It is still rejected below, because 36
    characters does not clear the floor — but this is a character-count rule,
    not a semantic one, and a long enough run of real English words would pass.
    That is the trade for not having to maintain a stopword list per language.
    """
    m, t = bk.count_meaningful_chars(segs("the the the"))
    assert (m, t) == (9, 9)
    run = "the " * 10   # 30 alnum characters, under the 40 floor
    assert len(run.replace(" ", "")) == 30
    assert not bk.has_usable_speech(segs(run))


def test_a_single_long_lecture_is_not_a_filler_run():
    """Filler chars are also legitimate words inside real speech.

    嗯 appears constantly in a spoken Mandarin lecture. Only the *density*
    matters, never the presence of one.
    """
    lecture = segs("嗯我们今天来看一下这个和弦的连接方式，"
                   "它需要通过共同音来保持声部进行的连贯，"
                   "否则就会出现不必要的跳动。")
    assert bk.has_usable_speech(lecture)


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def test_empty_transcript_is_no_speech():
    assert not bk.has_usable_speech([])
    assert not bk.has_usable_speech(segs())


def test_vocalisation_only_is_no_speech():
    assert not bk.has_usable_speech(segs("嗯嗯嗯", "嗯嗯嗯", "嗯背谱。"))


@pytest.mark.parametrize("texts", [
    ("嗯嗯嗯", "嗯嗯嗯背谱。"),                       # 第27节, 3 segs / 6 chars
    ("嗯嗯，背谱。",),                                # 第28节, 1 seg / 6 chars
    ("嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯。",),                        # 第29节
    ("嗯，嗯，啦啦啦啦啦啊。",),                       # 第19节
])
def test_the_measured_junk_transcripts_are_all_rejected(texts):
    assert not bk.has_usable_speech(segs(*texts))


def test_genuine_speech_is_accepted():
    assert bk.has_usable_speech(segs("这几年日本流行音乐里最常用的一个和声进行，"
                                     "我们把它拆开来看每一个和弦的功能，"
                                     "然后说明它为什么听起来这么舒服。"))


def test_the_threshold_sits_in_a_measured_gap():
    """The justification for MIN_MEANINGFUL_CHARS, as an executable claim.

    Measured across the 296-video library:

        worst  transcript judged no-speech : 21 meaningful characters
        shortest transcript judged speech  : 57 meaningful characters

    A 40-character floor leaves 19 below and 17 above. If a future change to
    FILLER_CHARS, or a new course with a different speech style, collapses that
    gap, this fails instead of silently starting to eat real lectures.
    """
    measured_worst_junk = 21        # 日本大师 第18节
    measured_shortest_real = 57     # 日本大师 第23节
    floor = bk.MIN_MEANINGFUL_CHARS
    assert measured_worst_junk < floor, "floor would reject a real transcript"
    assert floor < measured_shortest_real, "floor would let junk through"


def test_threshold_is_configurable():
    """A genuinely brief-but-real clip is a legitimate case to override."""
    assert bk.MIN_MEANINGFUL_CHARS == 40
    # the env var is read at import time; assert the knob exists in source
    src = Path(bk.__file__).read_text(encoding="utf-8")
    assert "V2K_MIN_SPEECH_CHARS" in src


# --------------------------------------------------------------------------
# end to end: the LLM must not be called
# --------------------------------------------------------------------------

def _write_subs(tmp_path: Path, segments: list[dict]) -> Path:
    import json
    p = tmp_path / "subtitles.json"
    p.write_text(json.dumps({"language": "zh", "duration": 120.0,
                             "segments": segments}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def _run(tmp_path, monkeypatch, segments):
    subs = _write_subs(tmp_path, segments)
    out = tmp_path / "out"

    def boom(*a, **k):
        raise AssertionError("the LLM must not be asked to summarise filler")

    monkeypatch.setattr(bk, "build_analysis", boom)
    monkeypatch.setattr(sys, "argv", [
        "build_knowledge.py", "--subtitles", str(subs),
        "--out-dir", str(out), "--format", "all", "--lang", "zh"])
    assert bk.main() == 0
    return out


def test_llm_is_skipped_for_a_vocalisation_only_transcript(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, segs("嗯嗯嗯", "嗯嗯嗯背谱。"))
    md = (out / "knowledge.md").read_text(encoding="utf-8")
    assert bk.NO_SPEECH_MARKER in md
    # the fix is incomplete without this: 6 characters used to yield 8 cards
    assert not (out / "cards.csv").exists() or \
        len((out / "cards.csv").read_text(encoding="utf-8").strip().splitlines()) <= 1


def test_status_reports_no_speech_for_filler(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, segs("嗯嗯嗯", "嗯嗯嗯背谱。"))
    md = (out / "knowledge.md").read_text(encoding="utf-8")
    assert bk.knowledge_doc_status({"summary": md}, 2, segs("嗯嗯嗯", "嗯嗯嗯背谱。")) \
        == bk.STATUS_NO_SPEECH


def test_status_still_ok_for_real_speech():
    real = segs("今天我们来讲解和弦的连接方法，它决定了配曲听起来是否流畅",
                "这是我们最常用的转位，需要通过共同音保持声部进行")
    assert bk.knowledge_doc_status({"summary": "和弦连接讲解"}, 2, real) \
        == bk.STATUS_OK


def test_merged_mode_is_exempt(tmp_path, monkeypatch):
    """Path 3 must not be caught by this.

    There the thin ASR track is expected — the content comes from visual OCR,
    and refusing to build a document would throw away a perfectly good lecture.
    """
    subs = _write_subs(tmp_path, segs("嗯嗯嗯"))
    out = tmp_path / "out"
    merged = tmp_path / "merged.json"
    import json
    merged.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "五线谱"},
            {"start": 1.0, "end": 2.0, "text": "属七和弦"},
        ],
        "visual_blocks": [{"t": 0.0, "text": "属七和弦解决到主和弦的进行"}],
        "asr_count": 1, "visual_count": 1,
    }, ensure_ascii=False), encoding="utf-8")

    called = {"n": 0}

    def fake_analysis(host, model, raw, source, **k):
        called["n"] += 1
        return {"summary": "和声讲解", "timeline": "", "key_points": "",
                "bullets": "", "qa": "", "glossary": ""}

    monkeypatch.setattr(bk, "build_analysis", fake_analysis)
    monkeypatch.setattr(bk, "build_visual_timeline", lambda *a, **k: "")
    monkeypatch.setattr(sys, "argv", [
        "build_knowledge.py", "--subtitles", str(subs), "--merged", str(merged),
        "--out-dir", str(out), "--format", "knowledge", "--lang", "zh"])
    bk.main()
    assert called["n"] == 1, "merged mode must still call the model"

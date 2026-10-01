"""Key frames that land past the last subtitle produce empty nodes.

Found on the first full Path 3 run in the library, an 18:28 lecture:

    frame times : ... 1019, 1076, 1110, 1111, 1112, 1113
    last subtitle: 1108.7s

Four of twenty-four nodes sat past the end of the transcript. Subtitles stop at
the last spoken word; frame extraction runs to the end of the file. The two
clocks are not the same, and nothing reconciled them.

`build_sections` gives each key frame a narration window of `[t, next frame t)`,
with the last one running to infinity. So a frame after the last subtitle has an
empty window *by construction* — not because that stretch is silent, but
because there are no subtitles left to find. The node renders as a bare image
under the literal heading "画面节点" with no body. Six percent of the note spent
on nodes that cannot say anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_notes as bn  # noqa: E402

# --------------------------------------------------------------------------
# the measured case, as data
# --------------------------------------------------------------------------

# 18:28 lecture. Subtitles end at 1108.7s; the file runs to 1118s.
REAL_FRAME_TIMES = [1.0, 2.0, 8.0, 11.0, 40.0, 42.0, 47.0, 129.0, 181.0, 245.0,
                    304.0, 472.0, 478.0, 704.0, 710.0, 835.0, 854.0, 894.0,
                    1019.0, 1076.0, 1110.0, 1111.0, 1112.0, 1113.0]
REAL_LAST_SUBTITLE = 1108.735
REAL_TAIL_COUNT = 4


def _drop_tail(times, segs, tolerance=1.0):
    """Mirror the filter build_notes.main() applies."""
    if not segs:
        return list(times)
    last = segs[-1]["end"]
    return [t for t in times if t <= last + tolerance]


def _segs(n=189, end=REAL_LAST_SUBTITLE):
    return [{"start": i, "end": min(i + 0.9, end), "text": "讲解内容"}
            for i in range(0, int(end), int(end / n))]


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------

def test_the_real_tail_is_dropped():
    kept = _drop_tail(REAL_FRAME_TIMES, _segs())
    assert len(kept) == len(REAL_FRAME_TIMES) - REAL_TAIL_COUNT


def test_no_kept_frame_exceeds_the_last_subtitle():
    kept = _drop_tail(REAL_FRAME_TIMES, _segs())
    assert all(f <= REAL_LAST_SUBTITLE + 1.0 for f in kept)


def test_a_frame_landing_just_inside_the_tolerance_survives():
    """A subtitle can end a fraction before its own sentence's last frame."""
    segs = [{"start": 0.0, "end": 100.0, "text": "x"}]
    kept = _drop_tail([50.0, 100.5], segs)
    assert kept == [50.0, 100.5]


def test_a_frame_well_past_the_tolerance_goes():
    segs = [{"start": 0.0, "end": 100.0, "text": "x"}]
    assert _drop_tail([50.0, 130.0], segs) == [50.0]


def test_no_subtitles_means_no_filtering_here():
    """With no transcript there is no reference point; the old path handles it."""
    assert _drop_tail([10.0, 20.0], []) == [10.0, 20.0]


# --------------------------------------------------------------------------
# every frame past the end is a hard error, not an empty note
# --------------------------------------------------------------------------

def test_all_frames_past_the_end_exits_instead_of_writing_an_empty_note(tmp_path,
                                                                     monkeypatch):
    subs = tmp_path / "subtitles.json"
    subs.write_text(json.dumps({"language": "zh", "duration": 10.0, "segments": [
        {"start": 0.0, "end": 5.0, "text": "只有一句"}]}, ensure_ascii=False),
        encoding="utf-8")
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps({"video": "v.mp4", "frames": [
        {"t": 60.0, "file": "a.jpg"}, {"t": 90.0, "file": "b.jpg"}]}),
        encoding="utf-8")
    out = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", [
        "build_notes.py", "--subtitles", str(subs), "--frames", str(frames),
        "--out-dir", str(out)])
    assert bn.main() == 2, "a note with no usable frame must not be written"
    assert not (out / "notes.md").exists()


# --------------------------------------------------------------------------
# heading fallback: a frame with no narration still shows what it teaches
# --------------------------------------------------------------------------

def _sec(t=0.0, title="", note="", desc="", excerpt=""):
    return {"t": t, "file": "f.jpg", "title": title, "note": note,
            "desc": desc, "excerpt": excerpt, "n_lines": 0}


META = {"title": "T", "video": "v.mp4", "duration": "00:10", "date": "2026-01-01"}


def test_heading_falls_back_to_the_frame_description(tmp_path):
    """The picture is the only thing such a node has; do not waste it on a literal."""
    md = bn.render_markdown(
        [_sec(title="", desc="乐谱显示音名与和弦")], META, tmp_path)
    assert "## [00:00] 乐谱显示音名与和弦" in md


def test_heading_falls_back_further_to_the_literal(tmp_path):
    md = bn.render_markdown([_sec(title="", desc="")], META, tmp_path)
    assert "## [00:00] 画面节点" in md


def test_a_real_title_still_wins(tmp_path):
    md = bn.render_markdown(
        [_sec(title="属七和弦的解决", desc="乐谱")], META, tmp_path)
    assert "## [00:00] 属七和弦的解决" in md


def test_the_fallback_appears_in_the_distilled_view_too(tmp_path):
    md = bn.render_markdown([_sec(title="", desc="板书标注拍号变化")], META,
                            tmp_path, verbatim=False)
    assert "## [00:00] 板书标注拍号变化" in md


# --------------------------------------------------------------------------
# honesty about frame-limited density
# --------------------------------------------------------------------------

def test_the_frame_limited_message_names_both_numbers():
    """A user who asked for "one node per 45s" and got bunched deserves to know why."""
    src = Path(bn.__file__).read_text(encoding="utf-8")
    assert "frame-limited" in src
    assert "distinct key frames" in src
    # and it must not claim the nodes are evenly spaced
    assert "instead of a clock" in src

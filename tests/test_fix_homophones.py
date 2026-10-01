"""Tests for scripts/fix_homophones.py.

The behaviour worth locking down is the *safety* behaviour: a corrector that
rewrites text it should have left alone is worse than no corrector at all. So
most of these assert on what must NOT change, and on the cases where offset
arithmetic goes wrong.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fix_homophones as fh  # noqa: E402


@pytest.fixture(scope="module")
def index():
    idx, firsts = fh.build_index(fh.DEFAULT_GLOSSARY)
    max_len = max(len(t) for t in fh.DEFAULT_GLOSSARY if len(t) >= 2)
    return idx, firsts, max_len


def _propose(text, index, blocked=()):
    idx, firsts, max_len = index
    return fh.propose(text, set(fh.DEFAULT_GLOSSARY), idx, max_len,
                      set(blocked), firsts)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad,good", [
    ("爬音", "琶音"),
    ("适谱", "视谱"),
    ("吊号", "调号"),
    ("合声", "和声"),
    ("川指", "穿指"),
    ("乐剧", "乐句"),
    ("音成", "音程"),
])
def test_detects_known_mishearings(bad, good, index):
    fixes = _propose(f"今天我们讲{bad}这个内容", index)
    assert [(f["from"], f["to"]) for f in fixes] == [(bad, good)]


def test_correct_spelling_is_left_alone(index):
    """琶音 must never be 'corrected' to anything — it is already a term."""
    assert _propose("这是一个琶音的练习", index) == []


def test_tone_difference_blocks_correction(index):
    """音阶 yīn jiē vs 音介 yīn jiè differ by tone, so the text is trusted.

    A tone mismatch means the audio was not ambiguous, which means the ASR
    probably heard it correctly and rewriting it would introduce an error.
    """
    assert _propose("这是一个音介的练习", index) == []


@pytest.mark.parametrize("text", [
    "曲是",       # 曲式  — ends on a function char
    "声不",       # 声部
    "调是",       # 调式
    "和手",       # 合手
])
def test_function_char_guard_rejects_common_word_slices(text, index):
    """宁可漏检不可错纠: these are two ordinary words sliced in half."""
    assert _propose(text + "的一些问题", index) == []


def test_longest_fragment_wins(index):
    """A 3-char match must not be shadowed by a 2-char prefix of it."""
    fixes = _propose("变化音很多", index)
    assert fixes == []


def test_non_overlapping_scan(index):
    """音成近的音成远 yields two independent corrections."""
    fixes = _propose("音成近的音成远", index)
    assert [(f["from"], f["to"]) for f in fixes] == [("音成", "音程")] * 2
    assert fixes[0]["end"] <= fixes[1]["start"]


def test_blocked_terms_are_skipped(index):
    """实度 → 十度 is a genuine false positive (it comes from 坚实度);
    --block escapes it without a special case in the defaults."""
    blocked = _propose("实度的练习", index, blocked={"实度"})
    assert all(f["from"] != "实度" for f in blocked)


@pytest.mark.parametrize("frag", sorted(fh.DEFAULT_BLOCKED))
def test_default_blocklist_is_actually_applied(frag, index):
    """The blocklist must be wired into the scan, not just documented.

    These are the fragments the function-character guard cannot catch: neither
    edge character is grammatical, so 坚实度 and 何首乌 both look like terms.
    """
    hits = _propose(f"这里有{frag}的一些问题", index, blocked=fh.DEFAULT_BLOCKED)
    assert all(f["from"] != frag for f in hits), hits


def test_default_blocklist_can_be_overridden(index):
    """--no-default-block lets a caller who knows better re-enable them."""
    idx, firsts, max_len = index
    text = "这里有实度的一些问题"
    assert fh.propose(text, set(fh.DEFAULT_GLOSSARY), idx, max_len,
                      fh.DEFAULT_BLOCKED, firsts) == []
    assert fh.propose(text, set(fh.DEFAULT_GLOSSARY), idx, max_len,
                      set(), firsts) != []


def test_context_is_recorded(index):
    fixes = _propose("我们先练习爬音然后放松", index)
    assert fixes and "放松" in fixes[0]["context"]


def test_first_char_prefilter_does_not_change_results(index):
    """The prefilter is an optimisation; it must not alter what is found."""
    idx, _, max_len = index
    text = "今天讲爬音、适谱、吊号、合声、川指、乐剧、音成、手行、延音"
    with_pre = fh.propose(text, set(fh.DEFAULT_GLOSSARY), idx, max_len,
                          set(), fh.build_index(fh.DEFAULT_GLOSSARY)[1])
    without_pre = fh.propose(text, set(fh.DEFAULT_GLOSSARY), idx, max_len, set())
    assert with_pre == without_pre


# --------------------------------------------------------------------------
# application: the offset bugs this rewrite exists to fix
# --------------------------------------------------------------------------

def _segs(*texts):
    return [{"start": float(i), "end": float(i + 1), "text": t}
            for i, t in enumerate(texts)]


def test_same_typo_twice_in_one_segment_is_fixed_twice(index):
    """The old substring-replace implementation fixed only the first."""
    segs = _segs("音成近的音成远")
    fixes = _propose(fh.concat_text(segs), index)
    touched, anomalies = fh.apply_fixes(segs, fixes)
    assert segs[0]["text"] == "音程近的音程远"
    assert touched == 1
    assert anomalies == []


def test_typo_is_not_fixed_in_the_wrong_segment(index):
    """A clean segment must stay clean even if a dirty one holds the same word."""
    segs = _segs("干净的音程", "这里有音成问题")
    fixes = _propose(fh.concat_text(segs), index)
    touched, _ = fh.apply_fixes(segs, fixes)
    assert segs[0]["text"] == "干净的音程"      # unchanged
    assert segs[1]["text"] == "这里有音程问题"
    assert touched == 1


def test_fix_spanning_a_segment_boundary(index):
    """ASR cuts mid-phrase; a term split across two cues is still one term.

    爬音 lands with 爬 at the end of cue 1 and 音 at the start of cue 2. The
    replacement goes into the earlier cue and the later one drops the consumed
    character, so no timestamped cue is left empty.
    """
    segs = _segs("我们今天讲一爬", "音的指法")
    fixes = _propose(fh.concat_text(segs), index)
    assert [(f["from"], f["to"]) for f in fixes] == [("爬音", "琶音")]
    touched, anomalies = fh.apply_fixes(segs, fixes)
    # the whole term lands in the earlier cue; the later one drops 音
    assert segs[0]["text"] == "我们今天讲一琶音"
    assert segs[1]["text"] == "的指法"
    # the invariant that actually matters: splitting and rejoining must not
    # lose or duplicate a character
    assert fh.concat_text(segs) == "我们今天讲一琶音的指法"
    assert touched == 2
    assert anomalies == []


def test_term_entirely_inside_the_second_segment(index):
    """The boundary is before the term, so only that segment changes."""
    segs = _segs("我们今天讲一", "下爬音的指法")
    fixes = _propose(fh.concat_text(segs), index)
    touched, _ = fh.apply_fixes(segs, fixes)
    assert segs[0]["text"] == "我们今天讲一"
    assert segs[1]["text"] == "下琶音的指法"
    assert touched == 1


def test_length_changing_fixes_do_not_shift_later_ones(index):
    """Right-to-left application must keep offsets valid for pending edits."""
    segs = _segs("音成近的音成远，这里是爬音的指法")
    fixes = _propose(fh.concat_text(segs), index)
    fh.apply_fixes(segs, fixes)
    assert segs[0]["text"] == "音程近的音程远，这里是琶音的指法"


def test_apply_is_idempotent(index):
    """Running twice must not double-apply or find new work: the second pass
    sees a corrected transcript, so it proposes nothing."""
    segs = _segs("音成近的音成远，这里是爬音的指法")
    fh.apply_fixes(segs, _propose(fh.concat_text(segs), index))
    once = fh.concat_text(segs)
    assert _propose(once, index) == []
    touched, _ = fh.apply_fixes(segs, _propose(once, index))
    assert touched == 0
    assert fh.concat_text(segs) == once


@pytest.mark.parametrize("text", [
    "音成近的音成远，这里是爬音的指法",
    "触见的时候要轻，触贱也是同样的道理",
    "今天的合声和音成都很重要",
    "手行要放松，延因要保持",
    "乐剧的结构和调式的变化",
    "一个没有任何术语的普通句子",
])
def test_rejoin_preserves_every_other_character(text, index):
    """Whatever else happens, the only changes are the proposed ones.

    Applied left-to-right, the corrected transcript must equal the original
    with exactly the reported spans substituted — no dropped characters, no
    duplicated ones, no silent edits between the reported offsets.
    """
    segs = _segs(text)
    fixes = _propose(text, index)
    expected, prev = [], 0
    for f in sorted(fixes, key=lambda f: f["start"]):
        expected.append(text[prev:f["start"]])
        expected.append(f["to"])
        prev = f["end"]
    expected.append(text[prev:])
    fh.apply_fixes(segs, fixes, text)
    assert fh.concat_text(segs) == "".join(expected)


def test_offsets_survive_a_different_length_replacement(index):
    """爬音→琶音 is same-length; use a term that is not."""
    segs = _segs("主调转换到和弦进行")
    text = fh.concat_text(segs)
    fixes = [{"start": 0, "end": 2, "from": "主调", "to": "功能和声",
              "context": "主调转换到和弦进行"}]
    assert text[fixes[0]["start"]:fixes[0]["end"]] == "主调"
    fh.apply_fixes(segs, fixes, text)
    assert segs[0]["text"] == "功能和声转换到和弦进行"


def test_offset_drift_is_reported_not_guessed(index):
    """A stale offset must surface, not silently no-op."""
    segs = _segs("实际内容完全不同")
    fixes = [{"start": 0, "end": 2, "from": "音成", "to": "音程", "context": ""}]
    touched, anomalies = fh.apply_fixes(segs, fixes)
    assert touched == 0
    assert segs[0]["text"] == "实际内容完全不同"   # untouched
    assert len(anomalies) == 1
    assert anomalies[0]["reason"] == "offset drift"


def test_empty_segments_are_skipped():
    segs = [{"start": 0.0, "end": 1.0, "text": ""}, {"start": 1.0, "end": 2.0, "text": "音成"}]
    fixes = [{"start": 0, "end": 2, "from": "音成", "to": "音程", "context": ""}]
    touched, _ = fh.apply_fixes(segs, fixes)
    assert segs[1]["text"] == "音程"
    assert touched == 1


def test_no_fixes_is_a_no_op():
    segs = _segs("一切正常")
    assert fh.apply_fixes(segs, []) == (0, [])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _run(tmp_path, payload, *extra):
    sub = tmp_path / "subtitles.json"
    sub.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script = Path(fh.__file__).resolve()
    r = subprocess.run([sys.executable, str(script), "--subtitles", str(sub), *extra],
                       capture_output=True, text=True)
    return r, sub


def test_cli_dry_run_touches_nothing(tmp_path):
    payload = {"language": "zh", "segments": _segs("今天讲爬音的指法")}
    r, sub = _run(tmp_path, payload, "--dry-run")
    assert r.returncode == 0
    assert "NOT applied" in r.stdout
    assert json.loads(sub.read_text(encoding="utf-8"))["segments"][0]["text"] \
        == "今天讲爬音的指法"


def test_cli_rewrites_json_and_writes_report(tmp_path):
    payload = {"language": "zh", "segments": _segs("今天讲爬音的指法")}
    r, sub = _run(tmp_path, payload)
    assert r.returncode == 0
    assert "琶音" in json.loads(sub.read_text(encoding="utf-8"))["segments"][0]["text"]
    report = json.loads((tmp_path / "homophone_fixes.json").read_text(encoding="utf-8"))
    assert report["changes"][0]["from"] == "爬音"
    assert report["anomalies"] == []


def test_cli_resyncs_srt_and_vtt(tmp_path):
    """Three subtitle files that disagree is the bug, not a feature."""
    payload = {"language": "zh", "segments": _segs("今天讲爬音的指法")}
    r, sub = _run(tmp_path, payload)
    assert r.returncode == 0
    srt = (tmp_path / "subtitles.srt").read_text(encoding="utf-8")
    vtt = (tmp_path / "subtitles.vtt").read_text(encoding="utf-8")
    assert "琶音" in srt and "爬音" not in srt
    assert vtt.startswith("WEBVTT") and "琶音" in vtt


def test_cli_no_sync_leaves_other_files_alone(tmp_path):
    payload = {"language": "zh", "segments": _segs("今天讲爬音的指法")}
    r, _ = _run(tmp_path, payload, "--no-sync")
    assert r.returncode == 0
    assert "[warn]" in r.stdout
    assert not (tmp_path / "subtitles.srt").exists()


def test_cli_clean_subtitles(tmp_path):
    payload = {"language": "zh", "segments": _segs("这是一个琶音的练习")}
    r, sub = _run(tmp_path, payload)
    assert r.returncode == 0
    assert "clean" in r.stdout
    assert not (tmp_path / "homophone_fixes.json").exists()


def test_cli_zero_segments(tmp_path):
    r, _ = _run(tmp_path, {"language": "zh", "segments": []})
    assert r.returncode == 0
    assert "0 segments" in r.stdout


def test_cli_missing_file(tmp_path):
    script = Path(fh.__file__).resolve()
    r = subprocess.run([sys.executable, str(script), "--subtitles",
                        str(tmp_path / "nope.json")], capture_output=True, text=True)
    assert r.returncode == 2
    assert "not found" in r.stderr


def test_cli_extra_glossary(tmp_path):
    """A user term joins the built-ins; text that is already right stays put."""
    payload = {"language": "zh", "segments": _segs("这是一个手眼协调的练习")}
    r, sub = _run(tmp_path, payload, "--glossary", "手眼协调")
    assert r.returncode == 0
    assert json.loads(sub.read_text(encoding="utf-8"))["segments"][0]["text"] \
        == "这是一个手眼协调的练习"


def test_cli_block_flag(tmp_path):
    payload = {"language": "zh", "segments": _segs("实度很重要")}
    sub = tmp_path / "subtitles.json"
    sub.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script = Path(fh.__file__).resolve()
    r = subprocess.run([sys.executable, str(script), "--subtitles", str(sub),
                        "--block", "实度"], capture_output=True, text=True)
    assert r.returncode == 0
    assert json.loads(sub.read_text(encoding="utf-8"))["segments"][0]["text"] == "实度很重要"


def test_cli_default_block_applies_without_a_flag(tmp_path):
    """A user who never reads the docs still does not get 坚实十度."""
    payload = {"language": "zh", "segments": _segs("它的坚实度很重要")}
    r, sub = _run(tmp_path, payload)
    assert r.returncode == 0
    assert "clean" in r.stdout
    assert json.loads(sub.read_text(encoding="utf-8"))["segments"][0]["text"] == "它的坚实度很重要"


def test_cli_no_default_block_re_enables(tmp_path):
    payload = {"language": "zh", "segments": _segs("它的坚实度很重要")}
    r, sub = _run(tmp_path, payload, "--no-default-block")
    assert r.returncode == 0
    assert "十度" in json.loads(sub.read_text(encoding="utf-8"))["segments"][0]["text"]

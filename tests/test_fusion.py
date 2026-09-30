#!/usr/bin/env python3
"""test_fusion.py — synthetic tests for the dual-path fusion pieces.

No network, no models, no media — pure function-level checks plus one CLI
round-trip of merge_visual.py on temp JSON files. Run:

    python tests/test_fusion.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import merge_visual as mv          # noqa: E402
import hotwords_from_ocr as hf     # noqa: E402
import build_knowledge as bk       # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if not cond:
        print(f"  FAIL {name} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok   {name}")


# --- merge_visual: timestamp attach + re-attach (regression) -----------------

def test_timestamp_attach_and_reattach():
    visual = [
        {"start": 0.0, "end": 100.0, "text": "幻灯片A:现金流量表结构分析"},
        {"start": 100.0, "end": 200.0, "text": "幻灯片B:编制方法与列报要求"},
    ]
    asr = [
        {"start": 10.0, "end": 12.0, "text": "我们看第一张现金流量表"},
        {"start": 13.0, "end": 15.0, "text": "它的结构分为三大部分"},
        {"start": 101.0, "end": 103.0, "text": "接下来讲编制方法"},
    ]
    r = mv.merge(asr, visual, semantic=False)
    check("attach: seg0 gets slide A", r["segments"][0]["visual"].startswith("幻灯片A"))
    check("attach: seg1 re-attaches slide A (no unused cover)",
          r["segments"][1]["visual"].startswith("幻灯片A") and r["reattached"] == 1)
    check("attach: seg2 gets slide B",
          r["segments"][2]["visual"].startswith("幻灯片B"))
    check("attach: match field is 'time' in pure-timestamp mode",
          all(s["match"] == "time" for s in r["segments"]))


# --- merge_visual: semantic swap ----------------------------------------------

def test_semantic_swap():
    slide_a = "长期股权投资 权益法 成本法 初始投资成本 后续计量调整"
    slide_b = "现金流量表 结构分析 编制基础 列报格式"
    visual = [
        {"start": 0.0, "end": 100.0, "text": slide_a},
        {"start": 100.0, "end": 200.0, "text": slide_b},
    ]
    # speaker still on slide A's topic while slide B is already up
    asr = [{"start": 100.5, "end": 102.5,
            "text": "权益法下初始投资成本还要调整长期股权投资的账面价值"}]
    r = mv.merge(asr, visual)
    s0 = r["segments"][0]
    check("swap: re-bound to adjacent slide A", s0["visual"] == slide_a,
          f"match={s0['match']}")
    check("swap: match flag", s0["match"] == "semantic-swap")
    check("swap: note present", "换绑" in s0.get("note", ""))
    check("swap: counted", r["semantic_swaps"] == 1)


def test_no_swap_when_timestamp_matches():
    # narration overlaps the timestamp-attached slide -> no judgement needed
    visual = [
        {"start": 0.0, "end": 100.0, "text": "现金流量表 结构分析 编制基础"},
        {"start": 100.0, "end": 200.0, "text": "长期股权投资 权益法 初始成本"},
    ]
    asr = [{"start": 101.0, "end": 103.0,
            "text": "权益法要求比较初始成本与享有份额"}]
    r = mv.merge(asr, visual)
    check("no-swap: kept timestamp slide",
          r["segments"][0]["match"] == "time"
          and "权益法" in r["segments"][0]["visual"])


def test_weak_attribution_flagged_not_swapped():
    # fresh slide (on screen <60s) + digressing narration -> flagged weak
    visual = [
        {"start": 0.0, "end": 100.0, "text": "长期股权投资 权益法 初始投资成本"},
        {"start": 100.0, "end": 150.0, "text": "现金流量表 结构分析 编制基础"},
    ]
    # narration shares nothing with EITHER slide (speaker digressing)
    asr = [{"start": 101.0, "end": 103.0,
            "text": "那我们休息十分钟再继续下一部分"}]
    r = mv.merge(asr, visual)
    s0 = r["segments"][0]
    check("weak: kept timestamp slide", "现金流量表" in s0["visual"])
    check("weak: flagged", s0["match"] == "weak" and "错位" in s0.get("note", ""))
    check("weak: counted", r["weak_attribution"] == 1)
    # same digression on a SETTLED slide (>=60s) -> normal elaboration, no flag
    visual_settled = [
        {"start": 0.0, "end": 100.0, "text": "长期股权投资 权益法 初始投资成本"},
        {"start": 100.0, "end": 220.0, "text": "现金流量表 结构分析 编制基础"},
    ]
    r2 = mv.merge(asr, visual_settled)
    check("weak: settled slide unflagged", r2["weak_attribution"] == 0
          and r2["segments"][0]["match"] == "time")


def test_short_narration_not_judged():
    visual = [{"start": 0.0, "end": 100.0, "text": "现金流量表 结构分析 编制基础"},
              {"start": 100.0, "end": 200.0, "text": "长期股权投资 权益法 初始成本"}]
    asr = [{"start": 101.0, "end": 101.6, "text": "好"}]
    r = mv.merge(asr, visual)
    check("short: no semantic judgement", r["segments"][0]["match"] == "time"
          and r["weak_attribution"] == 0)


# --- hotwords_from_ocr ---------------------------------------------------------

def test_extract_terms():
    blocks = [
        {"start": 0, "end": 10, "text": "第三章 长期股权投资\n权益法与成本法的适用范围"},
        {"start": 10, "end": 20, "text": "| 权益法 | 30% |\n| 成本法 | 70% |"},
        {"start": 20, "end": 30,
         "text": "长期股权投资的后续计量 适用IFRS 9\n金融资产分类"},
    ]
    terms = hf.extract_terms(blocks, max_terms=20)
    check("terms: recurring long term kept", "长期股权投资" in terms,
          f"terms={terms}")
    check("terms: header/table term kept", "权益法" in terms)
    check("terms: acronym from a single slide kept", "IFRS" in terms)
    check("terms: sub-ngram suppressed by parent term", "股权" not in terms)
    check("terms: no function-word junk",
          not any(t in ("我们", "老师", "以下", "一下") for t in terms))


def test_coverage():
    segs = [{"start": 0, "end": 1, "text": "今天我们讲权益法和长期股权投资"}]
    covered, missing = hf.coverage(["权益法", "长期股权投资", "IFRS"], segs)
    check("coverage: 2 of 3 covered", covered == 2 and missing == ["IFRS"])


def test_hotwords_cli(tmp_path: Path):
    caps = tmp_path / "captions.json"
    caps.write_text(json.dumps([
        {"start": 0, "end": 10, "text": "长期股权投资 权益法"},
        {"start": 10, "end": 20, "text": "权益法 适用 IFRS"},
    ], ensure_ascii=False), encoding="utf-8")
    subs = tmp_path / "subtitles.json"
    subs.write_text(json.dumps({"segments": [
        {"start": 0, "end": 5, "text": "权益法的适用"},
    ]}, ensure_ascii=False), encoding="utf-8")
    vocab = tmp_path / "course_hotwords.txt"
    out = tmp_path / "ocr_hotwords.txt"
    r = subprocess.run([sys.executable, str(HERE.parent / "scripts" / "hotwords_from_ocr.py"),
                        "--captions", str(caps), "--subtitles", str(subs),
                        "--course-vocab", str(vocab), "--manual", "会计要素,权益法",
                        "--out", str(out), "--fail-under", "0.9"],
                       capture_output=True, text=True)
    check("cli: exits 3 when coverage below threshold", r.returncode == 3,
          f"rc={r.returncode} stderr={r.stderr}")
    eff = out.read_text(encoding="utf-8").splitlines()
    check("cli: manual terms first", eff[:2] == ["会计要素", "权益法"])
    check("cli: OCR term merged in", "长期股权投资" in eff)
    vocab_lines = vocab.read_text(encoding="utf-8").splitlines()
    check("cli: course vocab accumulated", "长期股权投资" in vocab_lines
          and "IFRS" in vocab_lines)


# --- build_knowledge: QA pairs + note passthrough -----------------------------

def test_qa_pair_dedup():
    lines = ["Q: 什么是权益法?", "A: 长期股权投资的后续计量方法之一",
             "Q: 什么是权益法 ?", "A: 完全等价的重复问题",
             "Q: 现金流量表分为几类?", "A: 三类"]
    pairs = bk._dedupe_qa_pairs(bk._parse_qa_pairs(lines))
    check("qa: dup question dropped as a PAIR",
          len(pairs) == 2 and pairs[0][1].startswith("长期股权投资"))


def test_interleaved_note_marker():
    segs = [{"start": 101.0, "end": 103.0, "text": "权益法调整账面价值",
             "visual": "| 权益法 | 调整 |", "match": "semantic-swap",
             "note": "时间戳画面[01:40]与讲述词面重叠极低,已换绑"}]
    text = bk.build_interleaved_text(segs)
    check("interleave: ⚠️ note surfaced to the LLM", "⚠️" in text
          and "换绑" in text and "🖼️画面" in text)


def test_merge_cli_roundtrip(tmp_path: Path):
    subs = tmp_path / "subtitles.json"
    subs.write_text(json.dumps({"segments": [
        {"start": 1.0, "end": 3.0, "text": "现金流量表的结构"},
        {"start": 4.0, "end": 6.0, "text": "经营活动现金流量的列报"},
    ]}, ensure_ascii=False), encoding="utf-8")
    caps = tmp_path / "captions.json"
    caps.write_text(json.dumps([
        {"start": 0.0, "end": 10.0, "text": "现金流量表 结构分析 编制基础"},
    ], ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "merged.json"
    r = subprocess.run([sys.executable, str(HERE.parent / "scripts" / "merge_visual.py"),
                        "--subtitles", str(subs), "--visual", str(caps),
                        "--out", str(out)], capture_output=True, text=True)
    check("cli: merge exits 0", r.returncode == 0, r.stderr)
    data = json.loads(out.read_text(encoding="utf-8"))
    check("cli: schema has match/counts",
          "semantic_swaps" in data and all("match" in s for s in data["segments"]))
    check("cli: second segment re-attached", data["reattached"] == 1)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("merge_visual:")
        test_timestamp_attach_and_reattach()
        test_semantic_swap()
        test_no_swap_when_timestamp_matches()
        test_weak_attribution_flagged_not_swapped()
        test_short_narration_not_judged()
        print("hotwords_from_ocr:")
        test_extract_terms()
        test_coverage()
        test_hotwords_cli(tmp)
        print("build_knowledge:")
        test_qa_pair_dedup()
        test_interleaved_note_marker()
        print("cli round-trip:")
        test_merge_cli_roundtrip(tmp)
    print(f"\nALL {PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

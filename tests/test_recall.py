#!/usr/bin/env python3
"""test_recall.py — unit tests for recall_check normalization & scoring.

Run:  python tests/test_recall.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import recall_check as rc  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if not cond:
        print(f"  FAIL {name} {detail}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok {name}")


def test_number_normalization() -> None:
    nums = rc.extract_numbers("成本２０４,００元，利率５．５％，共3,000件")
    check("fullwidth digits normalized", "20400" in nums, str(nums))
    check("fullwidth percent number", "5.5" in nums, str(nums))
    check("comma-stripped thousands", "3000" in nums, str(nums))


def test_scoring_math() -> None:
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        (run / "knowledge.md").write_text(
            "# t\n- [00:10] 未确认融资费用的计算\n- 数字 1731.8 万元\n"
            "- [45:00] 尾部考点\n", encoding="utf-8")
        (run / "subtitles.json").write_text(json.dumps(
            {"segments": [{"start": 0, "end": 2700, "text": "x"}]}),
            encoding="utf-8")
        golden = {"terms": ["未确认融资费用", "不存在的词"],
                  "numbers": ["1731.8", "999"], "formulas": [],
                  "quarters": 2}
        r = rc.score(run, golden)
    check("term recall 1/2", abs(r["terms"]["recall"] - 0.5) < 1e-9)
    check("number recall 1/2", abs(r["numbers"]["recall"] - 0.5) < 1e-9)
    check("timeline both quarters covered",
          r["timeline_coverage"]["recall"] == 1.0)
    check("overall bounded", 0.0 <= r["overall"] <= 1.0)


def main() -> int:
    print("recall_check:")
    test_number_normalization()
    test_scoring_math()
    print(f"\nALL {PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

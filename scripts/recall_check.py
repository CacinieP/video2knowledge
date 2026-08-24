#!/usr/bin/env python3
"""recall_check.py — quantify "缺内容": check produced artifacts against a
golden must-have list (terms / numbers / formulas / timeline coverage).

Two modes:

  --draft  generate a golden DRAFT from a run dir (auto-extract high-confidence
           must-haves: numbers seen on-screen, numbers confirmed by BOTH the
           audio and the slide, slide-line terms). Human then prunes it —
           minutes, not hours.

  default  score a run dir against a golden file; prints per-category recall
           and the MISSING list (the actionable diff). Exit code 0 always —
           this is a measurement, not a gate.

Golden schema (JSON):
  {"video": "018 ...", "duration": 2069.0,
   "terms": ["未确认融资费用", ...],
   "numbers": ["1731.8", "20400"],          # normalized digits
   "formulas": ["(P/A,5%,5)"],              # normalized substrings
   "quarters": 4}                            # timeline coverage segments

Usage:
  python scripts/recall_check.py --run-dir runs/... --draft > golden.json
  python scripts/recall_check.py --run-dir runs/... --golden golden.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,8}")
NUM_RE = re.compile(r"\d[\d,，.．]*\d|\d")
FW = str.maketrans("０１２３４５６７８９，．％×（）", "0123456789,.%×()")


def norm_text(s: str) -> str:
    return (s or "").translate(FW).replace(",", "").replace("，", "")


def norm_number(s: str) -> str:
    # strip a trailing ".0" decimal part WITHOUT eating integer zeros
    # (rstrip(".0") turned 20400 into 204 — exactly the class of numeric
    # corruption this tool exists to catch)
    t = norm_text(s)
    return re.sub(r"\.0+$", "", t) or s


def extract_numbers(text: str) -> set[str]:
    return {norm_number(m) for m in NUM_RE.findall(norm_text(text))}


def run_corpus(run_dir: Path) -> tuple[str, float, list[float]]:
    """All user-facing artifact text + duration + timeline timestamps."""
    parts, duration, ts = [], 0.0, []
    for name in ("knowledge.md", "notes.md"):
        f = run_dir / name
        if f.is_file():
            parts.append(f.read_text(encoding="utf-8"))
    csv = run_dir / "cards.csv"
    if csv.is_file():
        parts.append(csv.read_text(encoding="utf-8"))
    sub = run_dir / "subtitles.json"
    if sub.is_file():
        try:
            segs = json.loads(sub.read_text(encoding="utf-8"))["segments"]
            duration = segs[-1]["end"] if segs else 0.0
        except Exception:
            pass
    for m in re.finditer(r"\[(\d{1,3}):(\d{2})\]",
                         "\n".join(p for p in parts if "knowledge" in str(p)
                                   or True)):
        ts.append(int(m.group(1)) * 60 + int(m.group(2)))
    return "\n".join(parts), duration, ts


def draft(run_dir: Path) -> dict:
    sub = json.loads((run_dir / "subtitles.json").read_text(encoding="utf-8"))
    asr_text = "\n".join(s["text"] for s in sub["segments"])
    duration = sub["segments"][-1]["end"] if sub["segments"] else 0.0
    vis_nums, vis_lines = set(), []
    cap = run_dir / "captions.json"
    if cap.is_file():
        try:
            for v in json.loads(cap.read_text(encoding="utf-8")):
                t = str(v.get("text", ""))
                vis_nums |= extract_numbers(t)
                vis_lines += [ln.strip() for ln in t.splitlines()
                              if 3 <= len(ln.strip()) <= 14]
        except Exception:
            pass
    asr_nums = extract_numbers(asr_text)
    # both-channel numbers: spoken AND written — the highest-confidence facts
    confirmed = sorted(vis_nums & asr_nums, key=lambda x: -len(x))[:40]
    terms: list[str] = []
    for ln in vis_lines:
        for m in CJK_RUN.findall(ln):
            if m not in terms:
                terms.append(m)
    return {"video": run_dir.name, "duration": duration,
            "numbers": confirmed[:30], "terms": terms[:60],
            "formulas": [], "quarters": 4, "_draft": True}


def score(run_dir: Path, golden: dict) -> dict:
    corpus, duration, ts = run_corpus(run_dir)
    ncorpus = norm_text(corpus)
    nnums = extract_numbers(corpus)
    out = {"video": golden.get("video", run_dir.name)}

    miss_t = [t for t in golden.get("terms", []) if t not in corpus]
    out["terms"] = {"n": len(golden.get("terms", [])),
                    "recall": 1 - len(miss_t) / max(1, len(golden.get("terms", []))),
                    "missing": miss_t}

    miss_n = [n for n in golden.get("numbers", []) if norm_number(n) not in nnums]
    out["numbers"] = {"n": len(golden.get("numbers", [])),
                      "recall": 1 - len(miss_n) / max(1, len(golden.get("numbers", []))),
                      "missing": miss_n}

    miss_f = [f for f in golden.get("formulas", []) if norm_text(f) not in ncorpus]
    out["formulas"] = {"n": len(golden.get("formulas", [])),
                       "recall": 1 - len(miss_f) / max(1, len(golden.get("formulas", []))),
                       "missing": miss_f}

    q = int(golden.get("quarters", 4)) or 4
    seg = duration / q if duration else 1
    def _in(i: int, t: float) -> bool:
        if i * seg <= t < (i + 1) * seg:
            return True
        # last quarter is closed: a [45:00] marker on a 45:00 video counts
        return i == q - 1 and t <= duration + 0.5
    cov = [bool([t for t in ts if _in(i, t)]) for i in range(q)]
    out["timeline_coverage"] = {"quarters": q, "covered": sum(cov),
                                "recall": sum(cov) / q,
                                "missing": [f"Q{i+1}" for i, c in enumerate(cov) if not c]}

    ws = {"terms": .3, "numbers": .3, "formulas": .1, "timeline_coverage": .3}
    def _w(k: str) -> float:
        return ws[k] if (out[k].get("n") or k == "timeline_coverage") else 0.0
    tot = sum(_w(k) * out[k]["recall"] for k in ws)
    wsum = sum(_w(k) for k in ws)
    out["overall"] = round(tot / max(wsum, 1e-9), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--draft", action="store_true",
                    help="emit a golden draft (JSON) from this run's sources")
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        print(f"[err] run dir not found: {args.run_dir}", file=sys.stderr)
        return 2
    if args.draft:
        print(json.dumps(draft(args.run_dir), ensure_ascii=False, indent=1))
        return 0
    if not args.golden:
        print("[err] need --golden or --draft", file=sys.stderr)
        return 2
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    r = score(args.run_dir, golden)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

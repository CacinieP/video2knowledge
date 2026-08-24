#!/usr/bin/env python3
"""merge_visual.py — fuse ASR subtitles with VLM visual captions by timestamp.

Inputs:
  --subtitles : subtitles.json from asr_caption.py (Path 2): {segments:[{start,end,text}]}
  --visual    : captions.json from mm_caption.py (Path 1): [{start,end,text}]
  --out       : merged.json (default <visual dir>/merged.json)

Output schema (merged.json):
  {
    "asr_source": "...", "visual_source": "...",
    "segments": [
      {"start":..,"end":..,"text":"<ASR line>","visual":"<on-screen text/table>",
       "match":"time|semantic-swap|weak","note":"..."},
      ...
    ],
    "visual_blocks": [{"start":..,"end":..,"text":"<full OCR of a slide>"}]
  }

Alignment rules, in order:
  1. TIMESTAMP: for each ASR segment, attach the visual caption whose window
     [vc.start-2, vc.end] covers the segment midpoint (±tolerance). An UNUSED
     caption is preferred; a caption attaches to AT MOST ONE consecutive run of
     segments so a huge table is not repeated on every narration line; when no
     unused caption covers the moment, the used one that covers it is
     re-attached (a long-lived slide keeps its table instead of going empty).
  2. SEMANTIC CHECK (default on, --no-semantic to disable): the speaker often
     lags or leads the slide deck — narrating slide N while slide N+1 is
     already up, or ASR timestamps drift. After the timestamp pick, score the
     word-level overlap between the narration and the attached slide text
     (CJK bigrams + Latin words, function words dropped). When the attached
     slide shares (almost) nothing with the narration while an ADJACENT slide
     (within --swap-window seconds) clearly matches better, the narration is
     re-bound to that neighbour ("semantic-swap") and the doc builder is told
     via a note. When nothing matches at all the attachment is kept but marked
     "weak" — the speaker may simply be elaborating verbally, so we flag
     instead of guessing. Both guards are deliberately conservative
     (see --weak-overlap / --swap-margin) to keep timestamp authority.

This is the "dual-path fusion" bridge between Path 1 (VLM) and Path 2 (ASR).
The output is consumed by build_knowledge.py --merged, which interleaves audio
text with visual tables/formulas (and surfaces swap/weak notes as ⚠️ markers)
so the LLM sees both at each timestamp.

Usage:
    python3 merge_visual.py --subtitles run/subtitles.json --visual run/captions.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
CJK_STOP_BIGRAMS = {
    "我们", "这个", "那个", "什么", "怎么", "可以", "一个", "就是", "还是",
    "但是", "所以", "然后", "现在", "这里", "那里", "大家", "老师", "同学",
    "比如", "例如", "一下", "时候", "这样", "那样", "已经", "应该", "这些",
    "那些", "自己", "通过", "关于", "以及", "可能", "因为", "如果", "注意",
    "下面", "上面", "表格", "如图", "图示", "之一", "等等", "一起", "一下",
}
EN_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "is", "are",
    "on", "with", "by", "at", "as", "be", "this", "that", "it", "from",
    "table", "chart", "figure", "note", "example", "page", "part",
}


def load_segments(path: Path) -> list[dict]:
    """Load {start,end,text} from subtitles.json or captions.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments") or data.get("captions") or []


def _tokens(text: str) -> set[str]:
    """Content tokens for overlap scoring: CJK char-bigrams + Latin words,
    minus function words. Coarse on purpose — it only powers a RELATIVE
    attached-vs-neighbour comparison, never an absolute judgement."""
    toks: set[str] = set()
    for run in CJK_RE.findall(text):
        if len(run) == 1:
            toks.add(run)
            continue
        for k in range(len(run) - 1):
            bg = run[k:k + 2]
            if bg not in CJK_STOP_BIGRAMS:
                toks.add(bg)
    for w in LATIN_RE.findall(text):
        lw = w.lower()
        if len(lw) >= 2 and lw not in EN_STOP:
            toks.add(lw)
    return toks


def _overlap(a: set[str], b: set[str]) -> float:
    """Containment-style overlap: |A∩B| / min(|A|,|B|). For narration-vs-slide
    this reads as "fraction of the smaller side that is shared" — stable when
    a big table dwarfs one narration line."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _fmt(sec: float) -> str:
    m, s = divmod(int(max(sec, 0)), 60)
    return f"{m:02d}:{s:02d}"


def find_visual_for(mid: float, visual: list[dict], used: set[int],
                    tolerance: float) -> int:
    """Index of the first UNUSED visual caption whose [start-tol, end] window
    covers `mid`; falling back to the first USED one that covers it (re-attach:
    a long-lived slide keeps its table instead of going visually empty); else -1.

    Returns the index (not the dict) so the caller avoids an O(n) list.index
    lookup per segment (O(n^2) overall on long videos)."""
    fallback = -1
    for i, vc in enumerate(visual):
        if vc["start"] - tolerance <= mid <= vc["end"]:
            if i not in used:
                return i
            if fallback < 0:
                fallback = i
    return fallback


def merge(asr: list[dict], visual: list[dict], tolerance: float = 2.0,
          semantic: bool = True, weak_overlap: float = 0.06,
          swap_margin: float = 0.15, swap_floor: float = 0.12,
          swap_window: float = 90.0, weak_dwell: float = 60.0) -> dict:
    segs = []
    used: set[int] = set()
    reattached = 0
    swaps = 0
    weak = 0
    vtok = [_tokens(str(v.get("text", ""))) for v in visual]
    for a in asr:
        mid = (a["start"] + a["end"]) / 2
        i = find_visual_for(mid, visual, used, tolerance)
        final_i, match, note = i, "time", None
        if i >= 0 and semantic:
            a_toks = _tokens(str(a.get("text", "")))
            # too little narration / too little OCR text to judge — trust time
            if len(a_toks) >= 3 and len(vtok[i]) >= 3:
                ov_att = _overlap(a_toks, vtok[i])
                if ov_att < weak_overlap:
                    best_j, best_ov = -1, 0.0
                    for j in (i - 1, i + 1):
                        if not 0 <= j < len(visual):
                            continue
                        vj = visual[j]
                        lo, hi = vj["start"] - tolerance, vj["end"]
                        dist = (lo - mid) if mid < lo else ((mid - hi) if mid > hi else 0.0)
                        if dist > swap_window:
                            continue  # adjacent slide but far away in time
                        ov = _overlap(a_toks, vtok[j])
                        if ov > best_ov:
                            best_j, best_ov = j, ov
                    if best_j >= 0 and best_ov >= max(ov_att + swap_margin, swap_floor):
                        final_i = best_j
                        swaps += 1
                        match = "semantic-swap"
                        note = (f"时间戳画面[{_fmt(visual[i]['start'])}]与讲述词面重叠极低,"
                                f"相邻画面[{_fmt(visual[best_j]['start'])}]匹配更优,已换绑")
                    elif visual[i]["end"] - visual[i]["start"] >= weak_dwell:
                        # settled slide (on screen >= weak_dwell): the narrator
                        # elaborating verbally off-slide is the NORM here, not a
                        # misattribution — keep the timestamp match unflagged so
                        # ⚠️ stays a rare, meaningful signal for the LLM
                        pass
                    else:
                        weak += 1
                        match = "weak"
                        note = "讲述与画面词面重叠极低,画面归属可能错位,供参考"
        vtext = ""
        if final_i >= 0:
            vtext = str(visual[final_i].get("text", ""))
            if final_i in used:
                reattached += 1
            else:
                used.add(final_i)
        seg = {"start": a["start"], "end": a["end"],
               "text": a["text"], "visual": vtext, "match": match}
        if note:
            seg["note"] = note
        segs.append(seg)
    return {
        "asr_count": len(asr),
        "visual_count": len(visual),
        "used_visual": len(used),
        "reattached": reattached,
        "semantic_swaps": swaps,
        "weak_attribution": weak,
        "segments": segs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fuse ASR subtitles with VLM visual captions")
    ap.add_argument("--subtitles", required=True, type=Path)
    ap.add_argument("--visual", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output merged.json (default: <visual dir>/merged.json)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="seconds of slack when matching ASR midpoint to a visual window")
    ap.add_argument("--no-semantic", action="store_true",
                    help="disable the semantic alignment check (pure timestamp "
                         "matching — the pre-2026-08 behavior)")
    ap.add_argument("--weak-overlap", type=float, default=0.06,
                    help="narration/slide overlap below this counts as 'no "
                         "match' (default 0.06)")
    ap.add_argument("--swap-margin", type=float, default=0.15,
                    help="an adjacent slide must beat the attached one by at "
                         "least this overlap to trigger a semantic swap "
                         "(default 0.15)")
    ap.add_argument("--swap-window", type=float, default=90.0,
                    help="max seconds between the narration moment and an "
                         "adjacent slide's window for a swap to be considered "
                         "(default 90)")
    ap.add_argument("--weak-dwell", type=float, default=60.0,
                    help="a slide on screen >= this many seconds is 'settled': "
                         "narration elaborating off-slide is normal and is NOT "
                         "flagged weak (default 60)")
    args = ap.parse_args()

    for p, name in [(args.subtitles, "subtitles"), (args.visual, "visual")]:
        if not p.is_file():
            print(f"[err] {name} not found: {p}", file=sys.stderr)
            return 2

    asr = load_segments(args.subtitles)
    visual = load_segments(args.visual)
    if not asr:
        print(f"[err] no ASR segments in {args.subtitles}", file=sys.stderr)
        return 2
    print(f"[merge] {len(asr)} ASR segments x {len(visual)} visual frames"
          + ("" if args.no_semantic else " (semantic alignment on)"), file=sys.stderr)

    result = merge(asr, visual, args.tolerance, semantic=not args.no_semantic,
                   weak_overlap=args.weak_overlap, swap_margin=args.swap_margin,
                   swap_window=args.swap_window, weak_dwell=args.weak_dwell)
    # keep full visual blocks for reference
    result["asr_source"] = str(args.subtitles)
    result["visual_source"] = str(args.visual)
    result["visual_blocks"] = [{"start": v["start"], "end": v["end"], "text": v["text"]}
                                for v in visual]

    out = args.out or args.visual.parent / "merged.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] merged {result['used_visual']}/{len(visual)} visual frames into "
          f"{len(asr)} ASR segments ({result['reattached']} re-attached, "
          f"{result['semantic_swaps']} semantic swaps, "
          f"{result['weak_attribution']} weak) -> {out}", file=sys.stderr)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

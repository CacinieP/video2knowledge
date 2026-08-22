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
      {"start":..,"end":..,"text":"<ASR line>","visual":"<on-screen text/table for this moment>"},
      ...
    ],
    "visual_blocks": [{"start":..,"end":..,"text":"<full OCR of a slide>"}]
  }

Alignment rule: for each ASR segment, attach the visual caption whose time window
[vc.start-2, vc.end] covers the segment midpoint (±tolerance seconds). An UNUSED
visual caption is preferred; a visual caption is attached to AT MOST ONE
consecutive run of ASR segments to avoid repeating a huge table on every line.
Re-attach fallback (as documented): when no unused caption covers the moment,
a USED one that covers it is re-attached — a slide that stays on screen for many
ASR segments keeps its table attached instead of going visually empty. Empty
`visual` only when no caption window covers the moment at all.

This is the "dual-path fusion" bridge between Path 1 (VLM) and Path 2 (ASR). The
output is consumed by build_knowledge.py --merged, which interleaves audio text
with visual tables/formulas so the LLM sees both at each timestamp (instead of
ASR-only, which misses everything on screen).

Usage:
    python3 merge_visual.py --subtitles run/subtitles.json --visual run/captions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_segments(path: Path) -> list[dict]:
    """Load {start,end,text} from subtitles.json or captions.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments") or data.get("captions") or []


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


def merge(asr: list[dict], visual: list[dict], tolerance: float = 2.0) -> dict:
    segs = []
    used: set[int] = set()
    reattached = 0
    for a in asr:
        mid = (a["start"] + a["end"]) / 2
        i = find_visual_for(mid, visual, used, tolerance)
        vtext = ""
        if i >= 0:
            vtext = visual[i]["text"]
            if i in used:
                reattached += 1
            else:
                used.add(i)
        segs.append({
            "start": a["start"], "end": a["end"],
            "text": a["text"], "visual": vtext,
        })
    return {
        "asr_count": len(asr),
        "visual_count": len(visual),
        "used_visual": len(used),
        "reattached": reattached,
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
    print(f"[merge] {len(asr)} ASR segments x {len(visual)} visual frames",
          file=sys.stderr)

    result = merge(asr, visual, args.tolerance)
    # keep full visual blocks for reference
    result["asr_source"] = str(args.subtitles)
    result["visual_source"] = str(args.visual)
    result["visual_blocks"] = [{"start": v["start"], "end": v["end"], "text": v["text"]}
                                for v in visual]

    out = args.out or args.visual.parent / "merged.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] merged {result['used_visual']}/{len(visual)} visual frames into "
          f"{len(asr)} ASR segments ({result['reattached']} re-attached) -> {out}",
          file=sys.stderr)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

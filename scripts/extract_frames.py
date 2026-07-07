#!/usr/bin/env python3
"""extract_frames.py — sample frames from a video at a fixed interval.

Output: <out_dir>/frame_%06d.jpg  +  frames.json
frames.json = [{"file": "...", "t": <sec>}, ...] so downstream scripts (mm_caption.py)
can pair each image with its timestamp without re-deriving it.

Usage:
    python3 extract_frames.py --video in.mp4 --out-dir frames --interval 2.0
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def ffprobe_duration(video: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def extract(video: Path, out_dir: Path, interval: float, fps_filter: float | None) -> list[dict]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fps_filter is not None:
        filt = f"fps={fps_filter}"
    else:
        filt = f"fps={1.0 / interval}"
    pattern = str(out_dir / "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", filt, "-q:v", "3", "-y", pattern],
        check=True,
    )

    frames = sorted(out_dir.glob("frame_*.jpg"))
    duration = ffprobe_duration(video)
    n = len(frames)
    # Even spacing across actual duration (more accurate than re-deriving from index*interval).
    records = []
    for i, f in enumerate(frames):
        t = round(i * (duration / n), 3) if n else 0.0
        records.append({"file": str(f), "t": t})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between frames (default 2.0)")
    ap.add_argument("--fps", type=float, default=None,
                    help="alternative: explicit fps filter (overrides --interval)")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2

    recs = extract(args.video, args.out_dir, args.interval, args.fps)
    manifest = args.out_dir / "frames.json"
    manifest.write_text(json.dumps(
        {"video": str(args.video.resolve()),
         "interval": args.interval, "fps": args.fps,
         "count": len(recs), "frames": recs},
        ensure_ascii=False, indent=2,
    ))
    print(f"[ok] extracted {len(recs)} frames -> {args.out_dir}")
    print(f"[ok] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""extract_frames.py — sample representative frames from a video.

Two modes:
  - interval (default): uniform `fps=1/interval` sampling. Timestamps are derived
    from even spacing across the probed duration. Good for motion video.
  - dedup:               DENSE uniform sampling + perceptual-hash (dHash) dedup.
                         Samples at --dedup-fps (default 1fps), computes a 64-bit
                         dHash per frame, and keeps only frames whose Hamming
                         distance from the last KEPT frame exceeds --dedup-hamming.
                         Finally caps to --max-frames by uniform downsampling.

`dedup` is the GENERAL mode for slide/PPT/screencast/talking-head videos: it needs
no scene-detection threshold tuning (which fails on fade animations and varies per
video), and its frame count is bounded by --max-frames so downstream VLM cost stays
predictable on any input. Timestamps stay accurate because fps sampling is uniform.

Output: <out_dir>/frame_%06d.jpg  +  frames.json
frames.json = {"video","mode","interval","fps","dedup_fps","dedup_hamming",
               "max_frames","count","frames":[{"file","t"}, ...]} so downstream
scripts (mm_caption.py) can pair each image with its timestamp.

Usage:
    python3 extract_frames.py --video in.mp4 --out-dir frames --interval 2.0
    python3 extract_frames.py --video slides.mp4 --out-dir frames --mode dedup
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def ffprobe_duration(video: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


# --- dHash (no Pillow / image deps) ------------------------------------------
# 9x8 grayscale -> each row: pixel[j+1] > pixel[j] -> 8*8 = 64 bits.

def read_pgm(data: bytes) -> np.ndarray:
    """Minimal PGM (P5) reader: returns (h,w) uint8 array."""
    assert data[:2] == b"P5", "expected P5 pgm"
    idx = 2
    tokens = []
    while len(tokens) < 3:
        while data[idx:idx + 1] in (b" ", b"\n", b"\t", b"\r"):
            idx += 1
        if data[idx:idx + 1] == b"#":
            while data[idx:idx + 1] not in (b"\n", b""):
                idx += 1
            continue
        start = idx
        while data[idx:idx + 1] not in (b" ", b"\n", b"\t", b"\r"):
            idx += 1
        tokens.append(int(data[start:idx]))
    w, h, _maxval = tokens
    idx += 1  # single whitespace after maxval
    return np.frombuffer(data[idx:idx + w * h], dtype=np.uint8).reshape(h, w)


def _dhash_bits(pix: np.ndarray) -> np.ndarray:
    """9x8 -> 64-bit bool array (left<right per row)."""
    return pix[:, 1:] > pix[:, :-1]


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# --- sampling ----------------------------------------------------------------

def extract_interval(video: Path, out_dir: Path, interval: float, fps_filter: float | None) -> list[dict]:
    """Uniform fps sampling. Timestamps = even spacing across probed duration."""
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


def _hash_all_frames(video: Path, fps: float) -> list[tuple[int, np.ndarray]]:
    """Return [(frame_index_0based, dhash_bits)] for every frame at `fps`.

    Uses an intermediate PGM pipe (9x8 grayscale) so hashing needs no image lib:
    one ffmpeg pass writes a PGM stream to stdout, parsed incrementally.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps},scale=9:8,format=gray", "-f", "image2pipe",
         "-vcodec", "pgm", "-"],
        capture_output=True, check=True,
    )
    blob = proc.stdout
    # PGM P5 frame size for 9x8: header (~12-16B) + 72B pixel data. Split on the
    # "P5\n" magic that starts each frame.
    parts = blob.split(b"P5\n", 1)
    out = []
    idx = 0
    chunks = blob.split(b"P5\n")
    # first chunk is empty (before first magic); the rest are frame bodies (header tail + pixels)
    for body in chunks[1:]:
        # body = "<W> <H>\n<MAXVAL>\n<72 bytes pixels>"
        try:
            # header: "9 8\n255\n" then 72 bytes
            # parse header width/height (fixed 9 8 here)
            nl1 = body.index(b"\n")
            nl2 = body.index(b"\n", nl1 + 1)
            w, h = (int(x) for x in body[:nl1].split())
            _maxval = int(body[nl1 + 1:nl2])
            px = np.frombuffer(body[nl2 + 1:nl2 + 1 + w * h], dtype=np.uint8).reshape(h, w)
            out.append((idx, _dhash_bits(px)))
            idx += 1
        except Exception:
            # truncated trailing frame — stop
            break
    return out


def extract_dedup(video: Path, out_dir: Path, fps: float, hamming: int,
                  max_frames: int) -> list[dict]:
    """Dense uniform sampling + dHash dedup + cap to max_frames.

    Returns records with accurate timestamps. Keeps frame j iff its dHash differs
    from the last KEPT frame by > `hamming` bits, so visually-identical frames
    (static slides, pauses) are dropped while real content changes are captured.
    Independent of scene-detection thresholds -> works on any video style.
    """
    all_hashes = _hash_all_frames(video, fps)
    duration = ffprobe_duration(video)
    if not all_hashes:
        return []
    # frame i (0-based) at `fps` -> timestamp i/fps (fps sampling is uniform)
    def t_of(i: int) -> float:
        return round(i / fps, 3)

    # Keep first frame always, then only on perceptual change.
    kept_idx = [all_hashes[0][0]]
    last = all_hashes[0][1]
    for i, h in all_hashes[1:]:
        if _hamming(h, last) > hamming:
            kept_idx.append(i)
            last = h

    # Cap to max_frames by uniform downsampling (preserves temporal spread).
    if len(kept_idx) > max_frames:
        step = len(kept_idx) / max_frames
        kept_idx = [kept_idx[int(j * step)] for j in range(max_frames)]

    # Extract each kept frame precisely at its timestamp via -ss seek + 1 frame.
    records = []
    for n, fi in enumerate(kept_idx, 1):
        jpg = out_dir / f"frame_{n:06d}.jpg"
        t = t_of(fi)
        # -ss before -i for fast seek; then grab exactly one frame.
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", str(t), "-i", str(video), "-frames:v", "1",
             "-q:v", "3", "-y", str(jpg)],
            check=True,
        )
        if jpg.exists():
            records.append({"file": str(jpg), "t": t})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["interval", "dedup"], default="interval",
                    help="sampling mode: interval (uniform fps) or dedup "
                         "(dense sample + perceptual-hash dedup; general-purpose)")
    # interval-mode args
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between frames (default 2.0, interval mode)")
    ap.add_argument("--fps", type=float, default=None,
                    help="explicit fps filter (overrides --interval, interval mode)")
    # dedup-mode args
    ap.add_argument("--dedup-fps", type=float, default=1.0,
                    help="dense sample rate in dedup mode (default 1.0 fps). "
                         "Raise for fast-changing video, keep low for slides.")
    ap.add_argument("--dedup-hamming", type=int, default=10,
                    help="keep a frame iff dHash Hamming distance from last kept "
                         "frame exceeds this (0-64; default 10). Lower=more frames.")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="hard cap on kept frames (default 120). Bounded VLM cost.")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dedup":
        recs = extract_dedup(args.video, args.out_dir, args.dedup_fps,
                             args.dedup_hamming, args.max_frames)
        print(f"[ok] extracted {len(recs)} dedup frames "
              f"(fps={args.dedup_fps}, hamming>{args.dedup_hamming}, "
              f"cap={args.max_frames}) -> {args.out_dir}", file=sys.stderr)
    else:
        recs = extract_interval(args.video, args.out_dir, args.interval, args.fps)
        print(f"[ok] extracted {len(recs)} frames -> {args.out_dir}", file=sys.stderr)

    manifest = args.out_dir / "frames.json"
    manifest.write_text(json.dumps(
        {"video": str(args.video.resolve()),
         "mode": args.mode,
         "interval": args.interval, "fps": args.fps,
         "dedup_fps": args.dedup_fps, "dedup_hamming": args.dedup_hamming,
         "max_frames": args.max_frames,
         "count": len(recs), "frames": recs},
        ensure_ascii=False, indent=2,
    ))
    print(f"[ok] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

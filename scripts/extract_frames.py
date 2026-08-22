#!/usr/bin/env python3
"""extract_frames.py — sample representative frames from a video.

Two modes:
  - interval (default): uniform `fps=1/interval` sampling. Timestamps are derived
    from even spacing across the probed duration. Good for motion video.
  - dedup:               DENSE uniform sampling + perceptual-hash (dHash) dedup.
                         Samples at --dedup-fps (default 1fps), computes a
                         (--hash-size)^2-bit dHash per frame (default 8 -> 64
                         bits; --hash-mode dual appends an aHash for flat /
                         gradient content dHash cannot see), and keeps only
                         frames whose Hamming distance from the last KEPT
                         frame exceeds --dedup-hamming. A detected change is
                         not kept at its FIRST frame: the extractor waits for
                         the frame to SETTLE (its successor is nearly
                         identical), so fade/slide animations are captured in
                         their final stable state, not mid-transition. Finally
                         caps to --max-frames (see extract_frames.cap_by_time).

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
# (n+1)x n grayscale -> each row: pixel[j+1] > pixel[j] -> n*n bits (8 -> 64).

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
    """(n+1) x n -> n*n-bit flat bool array (left<right per row)."""
    return (pix[:, 1:] > pix[:, :-1]).ravel()


def _ahash_bits(pix: np.ndarray) -> np.ndarray:
    """(n+1) x n -> n*n-bit flat bool array (pixel > mean, left n columns).

    dHash is blind to horizontally-monotone content (solid colors, single-axis
    gradients: every row comparison has the same sign -> flat hash at ANY
    size). aHash sees overall brightness structure, so it catches exactly the
    content dHash misses. In dual mode both are concatenated; thresholds are
    then over 2*n^2 bits, so scale --dedup-hamming ~2.5x (e.g. 25 for n=8).
    """
    block = pix[:, :-1]
    return (block > block.mean()).ravel()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# --- sampling ----------------------------------------------------------------

def cap_by_time(ts: list[float], max_frames: int, fps: float = 1.0,
                gap: float | None = None) -> list[float]:
    """Cap kept-frame timestamps to `max_frames` by cluster-stratified selection
    instead of blind uniform downsampling.

    Frames are clustered by time gaps (a gap > `gap` seconds starts a new
    cluster: a burst of changes within a few seconds is one "topic burst").
    Every cluster keeps at least its FINAL frame (the settled state of that
    burst); the remaining budget is distributed across clusters in proportion
    to their size, and picks inside a cluster are spread evenly (first and
    last always included). A 100-frame animation burst therefore no longer
    starves isolated key slides of budget, and every burst keeps its most
    complete state even under pressure.

    `gap` defaults to max(5.0, 2/fps) seconds. Returns the selected timestamps
    in order. Also imported by build_notes.py to cap key frames for notes.
    """
    if max_frames <= 0 or len(ts) <= max_frames:
        return list(ts)
    g = gap if gap is not None else max(5.0, 2.0 / fps)
    clusters: list[list[float]] = [[ts[0]]]
    for prev, t in zip(ts, ts[1:]):
        if t - prev > g:
            clusters.append([t])
        else:
            clusters[-1].append(t)
    if len(clusters) > max_frames:
        # More bursts than budget: keep the final frame of the max_frames
        # LARGEST bursts (ties prefer LATER ones — end-of-video summary
        # slides tend to matter more than early ones).
        ranked = sorted(range(len(clusters)),
                        key=lambda c: (-len(clusters[c]), -clusters[c][0]))
        return [clusters[c][-1] for c in sorted(ranked[:max_frames])]
    # Guarantee the final frame of every cluster, then hand out the remaining
    # budget proportionally to (size - 1) via largest-remainder rounding.
    weights = [len(c) - 1 for c in clusters]
    remaining = max_frames - len(clusters)
    total = sum(weights)
    raw = [remaining * w / total for w in weights] if total else [0.0] * len(clusters)
    add = [int(r) for r in raw]
    by_frac = sorted(range(len(clusters)), key=lambda i: -(raw[i] - add[i]))
    for i in by_frac[:remaining - sum(add)]:
        add[i] += 1
    out: list[float] = []
    for c, a in zip(clusters, add):
        k = 1 + a
        if k == 1:
            out.append(c[-1])
        else:  # even spread, first and last of the burst always included
            step = (len(c) - 1) / (k - 1)
            out.extend(c[round(j * step)] for j in range(k))
    return out

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


def _hash_all_frames(video: Path, fps: float, hash_size: int = 8,
                     hash_mode: str = "dhash") -> list[tuple[int, np.ndarray, float]]:
    """Return [(frame_index_0based, hash_bits, mean_luma)] for every frame at
    `fps`.

    Uses an intermediate PGM pipe ((n+1) x n grayscale) so hashing needs no image
    lib: one ffmpeg pass writes a PGM stream to stdout, parsed incrementally.
    hash_mode 'dual' concatenates dHash + aHash (2*n^2 bits) to cover content
    that is horizontally monotone (invisible to dHash alone). mean_luma feeds
    the blank-frame gate in _find_settled.
    """
    w, h = hash_size + 1, hash_size
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "image2pipe",
         "-vcodec", "pgm", "-"],
        capture_output=True, check=True,
    )
    blob = proc.stdout
    # PGM P5 frame size for (n+1) x n: header (~12-16B) + n*(n+1) B pixel data.
    # Split on the "P5\n" magic that starts each frame.
    chunks = blob.split(b"P5\n")
    # first chunk is empty (before first magic); the rest are frame bodies (header tail + pixels)
    out = []
    idx = 0
    for body in chunks[1:]:
        # body = "<W> <H>\n<MAXVAL>\n<pixel bytes>"
        try:
            nl1 = body.index(b"\n")
            nl2 = body.index(b"\n", nl1 + 1)
            pw, ph = (int(x) for x in body[:nl1].split())
            _maxval = int(body[nl1 + 1:nl2])
            px = np.frombuffer(body[nl2 + 1:nl2 + 1 + pw * ph], dtype=np.uint8).reshape(ph, pw)
            bits = _dhash_bits(px)
            if hash_mode == "dual":
                bits = np.concatenate([bits, _ahash_bits(px)])
            out.append((idx, bits, float(px.mean())))
            idx += 1
        except Exception:
            # truncated trailing frame — stop
            break
    return out


BLANK_RANGE = (16.0, 240.0)  # mean luma outside this = blank/flash frame


def _find_settled(all_hashes: list[tuple[int, np.ndarray, float]], start: int,
                  settle_eps: int, window_frames: int,
                  blank: tuple[float, float] = BLANK_RANGE
                  ) -> tuple[int, np.ndarray]:
    """From a change detected at `start`, return the first frame that SETTLES:
    its immediate successor is nearly identical (Hamming <= settle_eps) AND it
    is not a blank/flash frame (mean luma outside `blank`).

    A hard cut settles at `start` itself; a fade/scroll transition advances
    until consecutive frames stop moving (bounded by `window_frames`). The
    blank gate matters because sign-based hashes (dHash/aHash) are invariant
    to uniform brightness scaling: a frame at 1% into a fade from black hashes
    identically to the fully-lit slide, so without the gate the extractor
    would happily keep a visually black JPEG. If nothing settles within the
    window, fall back to the frame where change was first detected.
    """
    end = min(start + window_frames, len(all_hashes) - 1)
    for k in range(start, end + 1):
        _idx, bits, mean = all_hashes[k]
        if not (blank[0] <= mean <= blank[1]):
            continue
        nxt = all_hashes[k + 1] if k + 1 < len(all_hashes) else None
        if nxt is None or _hamming(nxt[1], bits) <= settle_eps:
            return all_hashes[k][0], bits
    return all_hashes[start][0], all_hashes[start][1]


def _jpg_luma(jpg: Path) -> float:
    """Mean luma of a JPEG via a tiny ffmpeg decode (9x8 gray PGM)."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(jpg), "-vf", "scale=9:8,format=gray",
         "-f", "image2pipe", "-vcodec", "pgm", "-"],
        capture_output=True, check=True,
    )
    body = r.stdout.split(b"P5\n", 1)[1]
    nl1 = body.index(b"\n")
    nl2 = body.index(b"\n", nl1 + 1)
    w, h = (int(x) for x in body[:nl1].split())
    px = np.frombuffer(body[nl2 + 1:nl2 + 1 + w * h], dtype=np.uint8)
    return float(px.mean())


def _extract_jpg_settled(video: Path, t: float, jpg: Path, fps: float,
                         blank: tuple[float, float] = BLANK_RANGE) -> bool:
    """Write one JPEG at `t`; if it decodes blank/black (the fps-filter's
    renumbered timestamps can point the seek a fraction of a sample EARLIER
    than the frame the hash actually saw mid-fade), retry slightly later.

    Returns True if the file exists. Bounded to 2 retries (<= 1 sample period).
    """
    for k, tt in enumerate((t, t + 0.5 / fps, t + 1.0 / fps)):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", f"{tt:.3f}", "-i", str(video), "-frames:v", "1",
             "-q:v", "3", "-y", str(jpg)],
            check=True,
        )
        if not jpg.exists():
            return False
        if k == 2:  # last try: accept whatever we got (legitimately dark content)
            return True
        m = _jpg_luma(jpg)
        if blank[0] <= m <= blank[1]:
            return True
    return True


def extract_dedup(video: Path, out_dir: Path, fps: float, hamming: int,
                  max_frames: int, hash_size: int = 8, hash_mode: str = "dhash",
                  settle_window: float = 2.0, settle_eps: int = 0) -> list[dict]:
    """Dense uniform sampling + dHash dedup + settle + cap to max_frames.

    Returns records with accurate timestamps. A frame is kept when its dHash
    differs from the last KEPT frame by > `hamming` bits, but the kept frame is
    the first SETTLED one after the change (successor nearly identical), so
    transition mid-states (fade halfway, slide mid-animation) are skipped.
    Independent of scene-detection thresholds -> works on any video style.
    """
    all_hashes = _hash_all_frames(video, fps, hash_size, hash_mode)
    duration = ffprobe_duration(video)
    if not all_hashes:
        return []
    # frame i (0-based) at `fps` -> timestamp i/fps (fps sampling is uniform)
    def t_of(i: int) -> float:
        return round(i / fps, 3)

    eps = settle_eps or max(2, hamming // 5)
    window_frames = max(1, int(settle_window * fps))

    # Keep first frame always, then only on perceptual change (at its settled frame).
    kept_idx = [all_hashes[0][0]]
    last = all_hashes[0][1]
    i = 1
    while i < len(all_hashes):
        idx, h, _mean = all_hashes[i]
        if _hamming(h, last) > hamming:
            j, hs = _find_settled(all_hashes, i, eps, window_frames)
            kept_idx.append(j)
            last = hs
            i = j + 1
        else:
            i += 1

    kept_idx = cap_by_time([t_of(i) for i in kept_idx], max_frames, fps=fps)

    # Extract each kept frame precisely at its timestamp via -ss seek + 1 frame.
    records = []
    for n, t in enumerate(kept_idx, 1):
        jpg = out_dir / f"frame_{n:06d}.jpg"
        # -ss before -i for fast seek; blank frames (fade mid-states) retry later.
        if _extract_jpg_settled(video, t, jpg, fps) and jpg.exists():
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
                         "frame exceeds this (default 10 of 64 bits; scale ~4x "
                         "for --hash-size 16). Lower=more frames.")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="hard cap on kept frames (default 120). Bounded VLM cost.")
    ap.add_argument("--hash-size", type=int, default=8,
                    help="dHash size: (n+1) x n grayscale -> n*n-bit hash "
                         "(default 8 = 64 bits). Raise to 16 (256 bits) to catch "
                         "small-but-important changes on dense slides.")
    ap.add_argument("--hash-mode", choices=["dhash", "dual"], default="dhash",
                    help="dhash: gradient hash only (default). dual: dHash+aHash "
                         "concatenated (2*n^2 bits) — use when dHash misses "
                         "changes on flat/gradient content; scale --dedup-hamming "
                         "~2.5x in dual mode (e.g. 25 for n=8).")
    ap.add_argument("--settle-window", type=float, default=2.0,
                    help="seconds to wait for a changed frame to settle before "
                         "keeping it (default 2.0; transitions longer than this "
                         "keep the frame where change was first detected)")
    ap.add_argument("--settle-eps", type=int, default=0,
                    help="consecutive-frame Hamming <= this counts as settled "
                         "(default 0 = auto: max(2, hamming//5))")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dedup":
        recs = extract_dedup(args.video, args.out_dir, args.dedup_fps,
                             args.dedup_hamming, args.max_frames,
                             hash_size=args.hash_size, hash_mode=args.hash_mode,
                             settle_window=args.settle_window,
                             settle_eps=args.settle_eps)
        nbits = args.hash_size * args.hash_size * (2 if args.hash_mode == "dual" else 1)
        print(f"[ok] extracted {len(recs)} dedup frames "
              f"(fps={args.dedup_fps}, {args.hash_mode} hamming>"
              f"{args.dedup_hamming} of {nbits} bits, settle<={args.settle_window}s, "
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
         "hash_size": args.hash_size, "hash_mode": args.hash_mode,
         "settle_window": args.settle_window,
         "max_frames": args.max_frames,
         "count": len(recs), "frames": recs},
        ensure_ascii=False, indent=2,
    ))
    print(f"[ok] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

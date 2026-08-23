#!/usr/bin/env python3
"""test_pgm_stream.py — regression: pixel data containing the b"P5\\n" magic
must never truncate the PGM stream.

The original parser split the ffmpeg PGM stream on the literal magic; pixel
bytes 0x50 0x35 0x0A split a frame in two, the bogus remainder failed header
parsing, and the except-break silently dropped EVERY later frame. This test
builds a real (rawvideo, lossless) video whose 9x8 grayscale frames embed the
magic at a different position per frame, then requires the streaming parser
to recover all frames with distinct hashes and the correct luma ramp.

Needs ffmpeg on PATH. Run:  python tests/test_pgm_stream.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

from extract_frames import _hash_all_frames, _hamming  # noqa: E402

N = 10


def build_defective_video(dir_: Path) -> Path:
    frames = []
    for i in range(N):
        px = np.full((8, 9), 40 + i * 18, dtype=np.uint8)
        px[i % 8, (i * 2) % 6:(i * 2) % 6 + 3] = [0x50, 0x35, 0x0A]  # "P5\n"
        frames.append(px.tobytes())
    raw = dir_ / "frames.raw"
    raw.write_bytes(b"".join(frames))
    avi = dir_ / "defect.avi"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", "9x8", "-r", "1",
         "-i", str(raw), "-c:v", "rawvideo", str(avi)],
        check=True, capture_output=True,
    )
    return avi


def main() -> int:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        avi = build_defective_video(Path(td))
        out = _hash_all_frames(avi, 1.0, 8, "dhash")
        check("all frames parsed (old parser stopped at 1)", len(out) == N, f"{len(out)}/{N}")
        check("frame order intact (distinct hashes)",
              all(_hamming(out[i][1], out[i + 1][1]) > 0 for i in range(N - 1)))
        means = [m for _, _, m in out]
        check("luma ramp correct", all(means[i] < means[i + 1] for i in range(N - 1)),
              f"{[round(m) for m in means]}")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

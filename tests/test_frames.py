#!/usr/bin/env python3
"""test_frames.py — synthetic tests for dedup frame selection (cap_by_time).

No network, no models, no media — pure function-level checks. Run:

    python tests/test_frames.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import extract_frames as ef  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if not cond:
        print(f"  FAIL {name} {detail}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok {name}")


def test_no_cap_needed() -> None:
    ts = [0.0, 10.0, 20.0]
    check("under budget returns all", ef.cap_by_time(ts, 5) == ts)


def test_single_cluster_even_spread() -> None:
    # one dense burst (1 fps, gaps <= 5s) -> first and last always included
    ts = [float(i) for i in range(30)]
    sel = ef.cap_by_time(ts, 4)
    check("burst spread keeps ends", sel[0] == 0.0 and sel[-1] == 29.0, str(sel))


def test_over_budget_stratifies_across_time() -> None:
    # regression: 20 isolated single-frame "slide" clusters (one per minute,
    # gaps > 5s) followed by dense demo bursts. The old largest-burst ranking
    # picked only the demo bursts and dropped the whole slide section.
    slides = [float(i * 60) for i in range(20)]            # 0..1140s
    demo = [1200.0 + i * 2 for i in range(16)]             # dense 32s burst
    sel = ef.cap_by_time(slides + demo, 12)
    early = [t for t in sel if t < 1200.0]
    check("capped set covers early slides", len(early) >= 4, str(sel))
    check("capped set covers demo tail", sel[-1] >= demo[-1], str(sel))
    check("capped set stays in budget", len(sel) <= 12, str(sel))
    check("capped set sorted", sel == sorted(sel), str(sel))


def test_over_budget_span_and_count() -> None:
    # more clusters than budget across the whole video: selection spans
    # start-to-end and returns exactly max_frames distinct picks
    ts = [float(i * 30) for i in range(40)]                # 40 isolated slides
    sel = ef.cap_by_time(ts, 8)
    check("spans start", sel[0] == ts[0], str(sel))
    check("spans end", sel[-1] == ts[-1], str(sel))
    check("exact budget", len(sel) == 8, str(sel))
    check("distinct picks", len(set(sel)) == len(sel), str(sel))


# ---- tail emission (final-state frames of similar-runs) ----
def _bits(n: int, flipped: int):
    import numpy as np
    b = np.zeros(n, dtype=np.uint8)
    b[:flipped] = 1
    return b


def _dedup_with_samples(samples, tmpdir: Path):
    """Run extract_dedup on synthetic hashes (no ffmpeg): patches hashing,
    duration probe and jpg writer; restores on exit."""
    import numpy as np  # noqa: F401
    saved = (ef._hash_all_frames, ef.ffprobe_duration, ef._extract_jpg_settled)
    ef._hash_all_frames = lambda *a, **k: [
        (i, b, 128.0) for i, b in enumerate(samples)]
    ef.ffprobe_duration = lambda p: float(len(samples))
    ef._extract_jpg_settled = lambda v, t, jpg, fps: (
        Path(jpg).write_bytes(b"x") or True)
    try:
        return ef.extract_dedup(Path("v.mp4"), Path(tmpdir), 1.0, 20, 360)
    finally:
        (ef._hash_all_frames, ef.ffprobe_duration,
         ef._extract_jpg_settled) = saved


def test_tail_catches_micro_edit_before_flip() -> None:
    # A x5 -> A'(dist 8) x4 -> flip to B(dist>=40): the last A' sample must
    # be kept — a sub-threshold edit (changed number, added bullet) the old
    # head-only walk silently lost.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        samples = ([_bits(64, 0)] * 5 + [_bits(64, 8)] * 4
                   + [_bits(64, 48)] * 5)
        ts = [r["t"] for r in _dedup_with_samples(samples, td)]
    check("micro-edit tail kept", ts == [0.0, 8.0, 9.0], str(ts))


def test_tail_not_fired_on_noise_run() -> None:
    # A x5 -> noise(dist 3) x4 -> flip: tail differs from anchor by only
    # 3 <= tail_eps(5) -> no extra frame.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        samples = ([_bits(64, 0)] * 5 + [_bits(64, 3)] * 4
                   + [_bits(64, 56)] * 5)
        ts = [r["t"] for r in _dedup_with_samples(samples, td)]
    check("noise run emits no tail", ts == [0.0, 9.0], str(ts))


def test_tail_not_fired_on_gradual_drift() -> None:
    # V0..V28 in steps of 4: crossing at V24 (dist 24>20); tail V20 is only
    # 4 bits from the incoming anchor V24 (<= hamming) -> skipped, and the
    # end-of-video tail V28 is only 4 from the anchor -> skipped.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        samples = [_bits(64, 0)] * 2 + [_bits(64, k) for k in (4, 8, 12, 16, 20, 24, 28)]
        ts = [r["t"] for r in _dedup_with_samples(samples, td)]
    check("drift crossing no tail spam", ts == [0.0, 7.0], str(ts))


def main() -> int:
    print("extract_frames.cap_by_time:")
    test_no_cap_needed()
    test_single_cluster_even_spread()
    test_over_budget_stratifies_across_time()
    test_over_budget_span_and_count()
    print("extract_frames tail emission:")
    test_tail_catches_micro_edit_before_flip()
    test_tail_not_fired_on_noise_run()
    test_tail_not_fired_on_gradual_drift()
    print(f"\nALL {PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

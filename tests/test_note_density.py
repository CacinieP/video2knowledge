"""The auto node budget for illustrated notes.

Density is the one number that decides whether a note feels complete or
patchy, and it is easy to change by accident: the old 1-per-4min rule, then
1-per-90s, both looked reasonable in isolation and both under-served short
clips once measured against the real library.

Measured on a 306-video / 61.5 h course library, the mean clip is 12.1 min.
Under the previous rule (`round(dur/90)` floored at 8) that floor bound the
*majority* of the library to 8 nodes no matter how long the video ran — the
real defect was the floor, not the ceiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_notes as bn  # noqa: E402


# The real function, not a copy of the formula. An earlier version of this file
# re-implemented the rule inline, which made every assertion below a tautology:
# the test stayed green no matter what build_notes.py actually did — and when
# the code moved to 45s the `--help` text silently kept advertising 90s.
budget = bn.auto_node_budget


# --------------------------------------------------------------------------
# the library's actual length distribution
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mins,expected", [
    (12.1, 16),    # the library mean — was 8 before
    (5, 12),       # floored, deliberately
    (20, 27),
    (26, 35),
    (41, 55),      # was clamped to 48 before
])
def test_budget_at_real_lengths(mins, expected):
    assert budget(mins * 60) == expected


def test_short_clips_are_not_starved():
    """The defect: a 12-min clip used to get 8 nodes regardless of length."""
    assert budget(12 * 60) >= 14
    assert budget(12 * 60) > 8


def test_long_lecture_is_not_clamped_to_48():
    assert budget(41 * 60) > 48
    assert budget(41 * 60) == 55


def test_floor_and_ceiling_hold():
    assert budget(0) == 12
    assert budget(1) == 12
    assert budget(10_000) == 60


def test_budget_is_monotonic():
    """A longer video must never get fewer nodes."""
    prev = 0
    for secs in range(0, 5400, 60):
        cur = budget(secs)
        assert cur >= prev, f"non-monotonic at {secs}s"
        prev = cur


def test_density_is_about_one_per_45s():
    for mins in (15, 25, 40):
        n = budget(mins * 60)
        assert abs(n - mins * 60 / 45) <= 1


def test_denser_than_the_previous_rule():
    """Explicitly non-regression against the 1-per-90s, floor-8 version."""
    for mins in (6, 10, 12, 20, 30, 45):
        old = max(8, min(48, round(mins * 60 / 90)))
        assert budget(mins * 60) > old, f"{mins}min did not get denser"


# --------------------------------------------------------------------------
# the constants themselves, and the help text that describes them
# --------------------------------------------------------------------------

def test_constants_are_the_documented_ones():
    assert (bn.NODE_SECONDS, bn.NODE_MIN, bn.NODE_MAX) == (45, 12, 60)


def test_help_text_agrees_with_the_code():
    """The drift this guards: code said 45s, `--help` still said 90s.

    Nothing else catches it. The behaviour tests pass either way, the note looks
    fine, and the only symptom is that a user reading `--help` picks a density
    that is not what they get.
    """
    import subprocess
    import sys as _sys

    out = subprocess.run(
        [_sys.executable, str(Path(bn.__file__)), "--help"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    help_text = " ".join(out.split())

    assert f"{bn.NODE_SECONDS}s" in help_text, \
        f"--help does not mention the {bn.NODE_SECONDS}s rate"
    assert f"{bn.NODE_MIN}-{bn.NODE_MAX}" in help_text, \
        f"--help does not mention the {bn.NODE_MIN}-{bn.NODE_MAX} clamp"

    # the stale numbers, if either ever reappears
    assert "per 90s" not in help_text
    assert "8-48" not in help_text


def test_printed_budget_uses_the_constant():
    """The stderr line must not hardcode a second copy of the rate."""
    src = Path(bn.__file__).read_text(encoding="utf-8")
    assert "~1 per {NODE_SECONDS}s" in src
    assert "~1 per 45s" not in src

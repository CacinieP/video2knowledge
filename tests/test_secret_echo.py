"""A mistyped `--api-key` must not print the key.

Found while deciding whether to keep `allow_abbrev=False`. The instinct was to
call it a security control; running it showed that only half of that was true.

With abbreviation on, `--api-key sk-…` is an unambiguous prefix of
`--api-key-env`, so argparse accepts the secret as the *name of an environment
variable* and the resulting "needs env var 'sk-…'" error quotes it back.
`allow_abbrev=False` stops that misuse — and then argparse still reports
`unrecognized arguments: --api-key sk-…`, printing the value just as loudly.

So the flag alone does not do what its comment claimed. What argparse *can* fix
is the echo, because that happens inside the process. What no argument parser
can fix is the process list and the shell history: the secret was on the command
line before any of this code ran. Hence: redact the echo, keep the flag, and say
plainly in the help that the key itself never belongs on the command line.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "asr_caption.py"

sys.path.insert(0, str(ROOT / "scripts"))
import asr_caption  # noqa: E402

SECRET = "sk-DO-NOT-PRINT-ME-1234567890"


def _run(*extra: str) -> str:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--video", "x.mp4", "--out-dir", "o",
         "--backend", "openai-api", "--api-model", "m", *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    return r.stdout + r.stderr


# --------------------------------------------------------------------------
# the echo
# --------------------------------------------------------------------------

def test_a_mistyped_api_key_is_not_echoed():
    out = _run("--api-key", SECRET)
    assert out, "expected an argparse error"
    assert SECRET not in out
    assert "redacted" in out


def test_the_flag_they_mistyped_is_still_named():
    """Redaction must not make the error useless."""
    out = _run("--api-key", SECRET)
    assert "--api-key" in out


def test_an_ordinary_typo_still_reports_its_value():
    """Only key-shaped flags are redacted. A typo needs the value back."""
    out = _run("--api-modle", "whisper-1")
    assert "--api-modle" in out
    assert "whisper-1" in out


def test_the_real_key_flag_is_untouched():
    """`--api-key-env` takes a variable NAME, which is not a secret."""
    ap = asr_caption._build_parser()
    assert "--api-key-env" in ap._REAL
    ns = ap.parse_args(["--video", "v.mp4", "--out-dir", "o",
                        "--api-key-env", "MY_SECRET_VAR_NAME"])
    assert ns.api_key_env == "MY_SECRET_VAR_NAME"


# --------------------------------------------------------------------------
# the flag itself
# --------------------------------------------------------------------------

def test_abbreviation_is_still_disabled():
    """`--keep` must not resolve to `--keep-wav`; that is the whole point."""
    ap = asr_caption._build_parser()
    assert ap.allow_abbrev is False


def test_long_options_still_work_in_full():
    ns = asr_caption._build_parser().parse_args([
        "--video", "v.mp4", "--out-dir", "o", "--language", "zh",
        "--keep-wav", "--chunk-seconds", "30", "--concurrency", "4",
        "--api-base", "https://example.invalid/v1", "--api-model", "m",
    ])
    assert ns.keep_wav and ns.chunk_seconds == 30 and ns.concurrency == 4
    assert ns.api_base.endswith("/v1")


# --------------------------------------------------------------------------
# the class itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--api-key", "--apikey", "--openai-key",
                                  "--API-KEY", "--key"])
def test_key_shaped_flags_are_recognised(flag):
    assert asr_caption._SecretSafeParser._KEYISH.match(flag)


@pytest.mark.parametrize("flag", ["--api-base", "--api-model", "--language",
                                  "--keep-wav", "--video", "-v"])
def test_ordinary_flags_are_not(flag):
    assert not asr_caption._SecretSafeParser._KEYISH.match(flag)

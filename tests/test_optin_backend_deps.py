"""Opt-in dependency setup for the optional ASR backends.

setup_models.sh grew two opt-in flags — `--with-funasr` and
`--with-openai-client` — because the scripts that need those dependencies
already told users to run them:

    asr_caption.py   "Run scripts/setup_models.sh --with-openai-client"
    asr_funasr.py    a four-line manual venv recipe in its module docstring

Neither existed. The scripts pointed at a flag the installer did not have, so
the documented recovery path for both optional backends was a dead end.

These tests are mostly about that class of drift: a *reference* to a flag in
one file, with no check that the flag exists in another. Nothing at runtime
catches it — the failure is an ImportError, or a user hand-assembling a venv,
long after the docs looked right.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SETUP = ROOT / "scripts" / "setup_models.sh"
ASR_CAPTION = ROOT / "scripts" / "asr_caption.py"
ASR_FUNASR = ROOT / "scripts" / "asr_funasr.py"


# Git Bash on a Chinese-locale Windows host writes the script's em-dashes and
# box characters in the console codepage, not UTF-8. Decoding strictly raises
# UnicodeDecodeError inside the reader thread and fails the test for a reason
# that has nothing to do with the script. Every assertion below is on ASCII.
DECODE = {"encoding": "utf-8", "errors": "replace"}


def _bash() -> str | None:
    """A bash that can actually run this script, or None.

    `shutil.which("bash")` is not enough on Windows: the first hit is normally
    the MicrosoftApps `bash.EXE` shim, which is the *WSL launcher*. With no
    distro installed it prints an install prompt and exits non-zero, so every
    test would fail against a stub. Probe candidates and keep the first one
    that runs a command.
    """
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        shutil.which("bash"),
        shutil.which("sh"),
    ]
    for cand in candidates:
        if not cand or not Path(cand).exists():
            continue
        try:
            r = subprocess.run([cand, "-c", "exit 0"],
                               capture_output=True, timeout=20, **DECODE)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            return cand
    return None


requires_bash = pytest.mark.skipif(_bash() is None,
                                    reason="no usable bash on this host")


# --------------------------------------------------------------------------
# the flags exist
# --------------------------------------------------------------------------

def test_setup_declares_both_optin_flags():
    src = SETUP.read_text(encoding="utf-8")
    for flag in ("--with-funasr", "--with-openai-client"):
        assert flag in src, f"setup_models.sh does not handle {flag}"


@requires_bash
def test_setup_script_is_syntactically_valid():
    r = subprocess.run([_bash(), "-n", str(SETUP)], capture_output=True, text=True, **DECODE)
    assert r.returncode == 0, r.stderr


@requires_bash
def test_help_works_without_ollama_or_a_venv():
    """`--help` must not require anything.

    The default path hard-requires ollama and exits 1 without it. Arg parsing
    happens before that specifically so someone on a machine with neither ollama
    nor a venv can still find out what the flags are.
    """
    r = subprocess.run([_bash(), str(SETUP), "--help"],
                       capture_output=True, text=True, timeout=60, **DECODE)
    assert r.returncode == 0, r.stderr
    assert "--with-funasr" in r.stdout
    assert "--with-openai-client" in r.stdout


def test_unknown_arg_is_reported_but_not_fatal():
    """Static check, deliberately.

    Running the script for real to observe the warning would execute the whole
    default path: probe ollama, `ollama serve`, and — if the profile VLM is not
    already present — a multi-GB `ollama pull`, then build a venv and install
    packages. A test must never do that to the developer's machine. So the
    contract is asserted against the source instead: an unrecognised argument
    falls into the `*)` arm, which logs and continues.
    """
    src = SETUP.read_text(encoding="utf-8")
    unknown_arm = re.search(r'\*\)\s*(.+?);;\s*\n', src)
    assert unknown_arm, "no catch-all arm for unknown arguments"
    arm = unknown_arm.group(1)
    assert "log" in arm, "unknown args must be reported, not silently swallowed"
    assert "exit" not in arm, \
        "an unknown arg must not abort the run (only --help may exit early)"


# --------------------------------------------------------------------------
# the drift this exists to prevent
# --------------------------------------------------------------------------

def test_asr_caption_error_points_at_a_flag_that_exists():
    """The ImportError is the only recovery path a user gets. It must resolve."""
    src = ASR_CAPTION.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")

    refs = set(re.findall(r"setup_models\.sh\s+(--[a-z-]+)", src))
    assert refs, "expected asr_caption.py to reference a setup_models.sh flag"
    for flag in refs:
        assert flag in setup, \
            f"asr_caption.py tells the user to run setup_models.sh {flag}, " \
            f"but the script has no such flag"


def test_asr_funasr_docstring_points_at_the_flag():
    """The manual recipe stays, but the supported path comes first."""
    doc = ASR_FUNASR.read_text(encoding="utf-8")
    assert "setup_models.sh --with-funasr" in doc
    # the hand-rolled fallback must survive for people who prefer it
    assert "-m venv .venv-funasr" in doc


def test_every_flag_named_in_the_repo_exists_in_the_script():
    """Repo-wide sweep: any `setup_models.sh <flag>` mention must be real.

    Broader than the two known cases on purpose — this is the check that would
    have caught the original dead end before anyone hit it.
    """
    setup = SETUP.read_text(encoding="utf-8")
    missing = []
    for py in sorted((ROOT / "scripts").glob("*.py")):
        for flag in set(re.findall(r"setup_models\.sh\s+(--[a-z-]+)",
                                   py.read_text(encoding="utf-8"))):
            if flag not in setup:
                missing.append(f"{py.name} -> {flag}")
    for md in sorted((ROOT / "references").glob("*.md")) + [ROOT / "SKILL.md",
                                                            ROOT / "README.md"]:
        for flag in set(re.findall(r"setup_models\.sh\s+(--[a-z-]+)",
                                   md.read_text(encoding="utf-8"))):
            if flag not in setup:
                missing.append(f"{md.name} -> {flag}")
    assert not missing, f"docs/scripts reference non-existent setup flags: {missing}"


# --------------------------------------------------------------------------
# the opt-in contract itself
# --------------------------------------------------------------------------

def test_default_run_installs_neither_optional_backend():
    """The important property: a plain run must not pull these in.

    funasr drags in its own torch (~2.5 GB) and the cloud SDK is the dependency
    for a backend that uploads audio off the host. Neither may be something a
    bare `setup_models.sh` decides to do.
    """
    src = SETUP.read_text(encoding="utf-8")
    assert "WITH_FUNASR=0" in src
    assert "WITH_OPENAI_CLIENT=0" in src
    # ...and every use of them must be guarded
    for var in ("WITH_FUNASR", "WITH_OPENAI_CLIENT"):
        uses = [m.start() for m in re.finditer(rf'if \[ "\${var}" = "1" \]', src)]
        assert uses, f"{var} is set but never tested — the opt-in would not hold"
    assert re.search(r'if \[ "\$WITH_OPENAI_CLIENT" = "1" \]; then', src)
    assert re.search(r'if \[ "\$WITH_FUNASR" = "1" \]; then', src)


def test_funasr_gets_its_own_venv_not_the_main_one():
    """funasr + faster-whisper in one venv is the thing that does not resolve."""
    src = SETUP.read_text(encoding="utf-8")
    assert "FUNASR_VENV_DIR" in src
    # the funasr install must target the funasr interpreter, never $VENV_PY
    block = src.split('if [ "$WITH_FUNASR" = "1" ]; then')[1].split("\nfi")[0]
    assert "$VENV_PY" not in block, \
        "funasr install targets the main venv; it must use its own interpreter"
    assert "$FUNASR_PY" in block


def test_openai_sdk_goes_into_the_main_venv():
    """The opposite case: the cloud SDK co-installs fine, so it belongs in $VENV_PY."""
    src = SETUP.read_text(encoding="utf-8")
    block = src.split('if [ "$WITH_OPENAI_CLIENT" = "1" ]; then')[1].split("\nfi")[0]
    assert "openai>=1.0" in block
    assert "$VENV_PY" in block


def test_cloud_flag_is_labelled_as_uploading():
    """Whoever turns this on must see that audio leaves the host."""
    src = SETUP.read_text(encoding="utf-8")
    assert "UPLOADS audio" in src or "upload your audio" in src.lower()


def test_reported_python_paths_use_set_u_safely():
    """`set -u` is on; the final report reads FUNASR_PY even when unused."""
    src = SETUP.read_text(encoding="utf-8")
    assert 'FUNASR_PY=""' in src, "FUNASR_PY must be initialised for `set -u`"
    r = subprocess.run([_bash(), "-n", str(SETUP)], capture_output=True, text=True, **DECODE)
    assert r.returncode == 0, r.stderr

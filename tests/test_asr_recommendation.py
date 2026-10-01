"""Tests for the ASR backend recommendation in hardware_profile.py.

The recommendation is advice the user reads instead of guessing which of the
three engines to reach for. Two things matter: it must not fire on hosts that
are perfectly capable, and the reason string has to say something true — an
earlier version recommended FunASR *because* it saw >=8 GB VRAM and called it
"GPU-accelerated", which is wrong (Paraformer runs on CPU here, and 8 GB is the
profile-tier gate, not the ASR gate).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import hardware_profile as hp  # noqa: E402


def _d(**kw):
    base = {"ram_gb": 16.0, "nvidia_vram_gb": None, "apple_chip": None,
            "cuda_ok": False}
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# the measured host
# --------------------------------------------------------------------------

def test_measures_the_actual_batch_host():
    """RTX 3060 Laptop, 6 GB VRAM, 16 GB RAM — the machine the 94% vs 44%
    A/B was run on. Its GPU is too small to be the deciding factor."""
    backend, why = hp.recommend_asr_backend(
        _d(ram_gb=15.8, nvidia_vram_gb=6.0, cuda_ok=True))
    assert backend == "funasr"
    assert "94%" in why and "44%" in why


def test_recommendation_is_not_driven_by_vram_tier():
    """8 GB VRAM is the profile-tier gate, not the ASR gate.

    A host with a large GPU but ordinary RAM must land on the same answer as
    one with no GPU at all, because Paraformer's advantage is accuracy on
    Mandarin and it runs on CPU.
    """
    with_gpu = hp.recommend_asr_backend(
        _d(ram_gb=16.0, nvidia_vram_gb=24.0, cuda_ok=True))[0]
    without = hp.recommend_asr_backend(
        _d(ram_gb=16.0, nvidia_vram_gb=None, cuda_ok=False))[0]
    assert with_gpu == without == "funasr"


def test_never_calls_paraformer_gpu_accelerated():
    """The old wording was wrong twice: Paraformer runs on CPU, and a GPU
    present does not make it so."""
    _, why = hp.recommend_asr_backend(
        _d(ram_gb=16.0, nvidia_vram_gb=24.0, cuda_ok=True))
    assert "GPU-accelerated" not in why
    assert "CPU" in why


# --------------------------------------------------------------------------
# the tiers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ram,expected", [
    (4.0, "openai-api"),
    (5.9, "openai-api"),
    (8.0, "funasr"),
    (16.0, "funasr"),
    (64.0, "funasr"),
])
def test_ram_thresholds(ram, expected):
    assert hp.recommend_asr_backend(_d(ram_gb=ram))[0] == expected


def test_mid_range_prefers_faster_whisper_over_cloud():
    """8 GB is the floor for funasr; below it but above 6 GB, local whisper
    still beats sending the audio to a third party."""
    backend, why = hp.recommend_asr_backend(
        _d(ram_gb=7.0, nvidia_vram_gb=None, apple_chip=None, cuda_ok=False))
    assert backend == "faster-whisper"
    assert "FunASR" in why


def test_low_ram_warns_about_the_privacy_cost():
    """Recommending cloud is a privacy decision, so the reason has to say the
    audio leaves the machine rather than presenting it as a free win."""
    _, why = hp.recommend_asr_backend(_d(ram_gb=4.0))
    assert "third party" in why or "uploads" in why


def test_apple_silicon_is_called_out():
    _, why = hp.recommend_asr_backend(_d(ram_gb=32.0, apple_chip="M2 Max"))
    assert "M2 Max" in why


def test_gpu_warning_mentions_the_vlm_contention():
    """The measured trap: whisper and the VLM on the same GPU turns an 8-token
    request from 3.8s into >60s. A host with a free GPU should hear about it."""
    _, why = hp.recommend_asr_backend(
        _d(ram_gb=16.0, nvidia_vram_gb=6.0, cuda_ok=True))
    assert "VLM" in why and "VRAM" in why


def test_always_returns_a_reason():
    for kw in (_d(ram_gb=4.0), _d(ram_gb=7.0), _d(ram_gb=16.0)):
        backend, why = hp.recommend_asr_backend(kw)
        assert backend and len(why) > 20


# --------------------------------------------------------------------------
# detect() wiring
# --------------------------------------------------------------------------

def test_detect_exposes_the_recommendation():
    d = hp.detect()
    assert d["recommended_asr_backend"] in {"funasr", "faster-whisper",
                                            "openai-api"}
    assert d["recommended_backend_reason"]


def test_detect_keeps_every_pre_existing_field():
    """PR #1 and #6 callers read these; adding fields must not drop any."""
    d = hp.detect()
    for k in ("os", "os_release", "arch", "ram_gb", "apple_chip",
              "nvidia_vram_gb", "cuda_ok", "profile", "asr_model",
              "compute_type", "device", "vlm_model", "text_model", "note"):
        assert k in d, k


def test_recommend_flag_runs_without_asr(capsys):
    import subprocess
    r = subprocess.run(
        [sys.executable, str(Path(hp.__file__).resolve()), "--recommend"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    assert "recommended ASR backend:" in r.stdout
    assert "batch_run.py" in r.stdout or "asr_caption.py" in r.stdout

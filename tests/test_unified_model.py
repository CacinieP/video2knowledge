"""Unified minicpm-v4.7 lineup (2026-10): ONE local model per host.

Every tier's VLM and text fields must agree on openbmb/minicpm-v4.7 — the
whole point of the refresh is that a single 1.6 GB pull serves Path 1
captioning/OCR and Step 2 text synthesis, so a tier silently drifting back to
a two-model lineup (or an old qwen pick) is a regression this catches.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import hardware_profile as hp  # noqa: E402

V47 = "openbmb/minicpm-v4.7:latest"


def test_every_tier_uses_v47_for_vlm_and_text():
    for name, prof in hp.PROFILES.items():
        assert prof["vlm"] == V47, f"{name}: vlm drifted from the unified pull"
        assert prof["text"] == V47, f"{name}: text drifted from the unified pull"


def test_asr_ladder_unchanged():
    """ASR stays tiered and hardware-driven — only the LLM/VLM was unified."""
    expected = {"tiny": "tiny", "low": "base", "low-mac": "small", "mid": "small",
                "high": "medium", "high-gpu": "large-v3", "max": "large-v3"}
    assert {k: v["asr"] for k, v in hp.PROFILES.items()} == expected


def test_fallback_text_model_is_the_unified_pull():
    """Detection-failure fallback must not force a second model download."""
    assert hp.FALLBACK_TEXT_MODEL == V47


def test_detect_reports_unified_models(monkeypatch):
    monkeypatch.setattr(hp, "detect_ram_gb", lambda: 12.0)
    monkeypatch.setattr(hp, "detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(hp, "detect_apple_silicon", lambda: None)
    d = hp.detect()
    assert d["profile"] == "mid"
    assert d["vlm_model"] == V47
    assert d["text_model"] == V47


def test_script_defaults_point_at_v47():
    """Entry points that hardcode the default model keep it on the unified one."""
    root = Path(__file__).resolve().parent.parent
    for fname in ("mm_caption.py", "batch_run.py", "setup_models.sh",
                  "build_knowledge.py", "build_notes.py"):
        src = (root / "scripts" / fname).read_text(encoding="utf-8-sig")
        assert "openbmb/minicpm-v4.7" in src, f"{fname}: v4.7 default missing"

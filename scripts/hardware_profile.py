#!/usr/bin/env python3
"""hardware_profile.py — detect the host machine and recommend a config profile
(ASR model size, compute type, device, VLM) that fits its RAM and GPU.

Single source of truth for the whole skill. Other scripts / setup_models.sh read
this via:
    python3 hardware_profile.py            # human-readable summary
    python3 hardware_profile.py --json     # machine-readable dict
    python3 hardware_profile.py --key asr_model

Profile table (see references/hardware-profiles.md for rationale):

  profile   RAM        GPU            ASR model   compute      VLM
  -------   --------   -------------  ----------  -----------  ----------------------
  tiny      < 6 GB     any            tiny        int8         moondream (if 4GB+)
  low       6–8 GB     none/integrated base        int8         minicpm-v4.6 (Q4)
  low-mac   6–8 GB     Apple Silicon  small       int8         minicpm-v4.6 (Q4)
  mid       8–16 GB    any            small       int8         minicpm-v4.6 (Q4)
  high      16–32 GB   any            medium      int8_float16 qwen2.5vl:3b
  high-gpu  >= 8 GB    NVIDIA >=8GB    large-v3    float16      qwen2.5vl:7b
  max       > 32 GB    any            large-v3    float16      qwen2.5vl:7b

NVIDIA GPUs short-circuit to high-gpu (CUDA + float16 is always faster than CPU)
regardless of total RAM, as long as VRAM >= 8 GB.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys

# --- thresholds -------------------------------------------------------------
PROFILES = {
    "tiny":     {"min_ram": 0,  "asr": "tiny",     "compute": "int8",         "device": "cpu",  "vlm": "moondream",                       "note": "极低配/老设备，仅保证能跑"},
    "low":      {"min_ram": 6,  "asr": "base",     "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.6:latest",     "note": "6-8GB 无独立GPU"},
    "low-mac":  {"min_ram": 6,  "asr": "small",    "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.6:latest",     "note": "Apple Silicon 6-8GB（Metal 加速抽帧）"},
    "mid":      {"min_ram": 8,  "asr": "small",    "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.6:latest",     "note": "8-16GB 通用"},
    "high":     {"min_ram": 16, "asr": "medium",   "compute": "int8_float16", "device": "auto", "vlm": "qwen2.5vl:3b",                    "note": "16-32GB，可上 medium"},
    "high-gpu": {"min_ram": 8,  "asr": "large-v3", "compute": "float16",      "device": "cuda", "vlm": "qwen2.5vl:7b",                    "note": "NVIDIA >=8GB VRAM，CUDA 全速"},
    "max":      {"min_ram": 32, "asr": "large-v3", "compute": "float16",      "device": "auto", "vlm": "qwen2.5vl:7b",                    "note": "工作站/服务器 >32GB"},
}


# --- detection --------------------------------------------------------------

def detect_os() -> str:
    return platform.system().lower()  # darwin / linux / windows


def detect_arch() -> str:
    return platform.machine().lower()  # arm64 / x86_64 / aarch64


def detect_ram_gb() -> float:
    """Total physical RAM in GB. Returns best-effort; 0 if unknown."""
    osn = detect_os()
    try:
        if osn == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, check=True)
            return int(out.stdout.strip()) / 1073741824
        if osn == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 / 1024
        if osn == "windows":
            # wmic is deprecated but widely present; fall back is acceptable.
            out = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, check=True)
            for tok in out.stdout.split():
                if tok.isdigit():
                    return int(tok) / 1073741824
    except Exception:
        pass
    return 0.0


def detect_apple_silicon() -> str | None:
    """Return chipset name (e.g. 'Apple M2 Pro') on Apple Silicon, else None."""
    if detect_os() != "darwin" or detect_arch() != "arm64":
        return None
    try:
        out = subprocess.run(["system_profiler", "SPHardwareDataType"],
                             capture_output=True, text=True, check=True, timeout=10)
        for line in out.stdout.splitlines():
            if "Chip:" in line or "Chipset" in line:
                return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return None


def detect_nvidia_vram_gb() -> float | None:
    """Return VRAM in GB of first NVIDIA GPU, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        first = out.stdout.strip().splitlines()[0]
        return int(first) / 1024  # MiB -> GiB
    except Exception:
        return None


# --- profile selection ------------------------------------------------------

def select_profile(ram_gb: float, nvidia_vram: float | None = None,
                   apple_chip: str | None = None) -> str:
    """Pick the best profile for this machine. All args optional for testing."""
    if nvidia_vram is not None and nvidia_vram >= 8:
        return "high-gpu"
    if ram_gb >= 32:
        return "max"
    if ram_gb >= 16:
        return "high"
    if apple_chip and 6 <= ram_gb < 8:
        return "low-mac"
    if ram_gb >= 8:
        return "mid"
    if ram_gb >= 6:
        return "low"
    return "tiny"


def detect() -> dict:
    """Run all detection and return a full profile dict."""
    ram = detect_ram_gb()
    nvidia = detect_nvidia_vram_gb()
    apple = detect_apple_silicon()
    pname = select_profile(ram, nvidia_vram=nvidia, apple_chip=apple)
    prof = PROFILES[pname]
    return {
        "os": detect_os(),
        "arch": detect_arch(),
        "ram_gb": round(ram, 1),
        "apple_chip": apple,
        "nvidia_vram_gb": round(nvidia, 1) if nvidia else None,
        "profile": pname,
        "asr_model": prof["asr"],
        "compute_type": prof["compute"],
        "device": prof["device"],
        "vlm_model": prof["vlm"],
        "note": prof["note"],
    }


# --- CLI --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--key", help="print a single field (asr_model, compute_type, "
                                  "device, vlm_model, profile, ram_gb)")
    args = ap.parse_args()
    d = detect()
    if args.key:
        if args.key not in d:
            print(f"[err] unknown key '{args.key}'. valid: {sorted(d)}", file=sys.stderr)
            return 2
        print(d[args.key])
        return 0
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return 0
    chip = f"  chip:    {d['apple_chip']}\n" if d["apple_chip"] else ""
    nv = f"  nvidia:  {d['nvidia_vram_gb']} GB VRAM\n" if d["nvidia_vram_gb"] else ""
    print(f"hardware profile: {d['profile']}  ({d['note']})")
    print(f"  os/arch: {d['os']} / {d['arch']}")
    print(f"  ram:     {d['ram_gb']} GB")
    if chip:
        print(chip.rstrip())
    if nv:
        print(nv.rstrip())
    print(f"  -> asr_model:    {d['asr_model']}")
    print(f"  -> compute_type: {d['compute_type']}")
    print(f"  -> device:       {d['device']}")
    print(f"  -> vlm_model:    {d['vlm_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""hardware_profile.py — detect the host machine and recommend a config profile
(ASR model size, compute type, device, VLM) that fits its RAM and GPU.

Single source of truth for the whole skill. Other scripts / setup_models.sh read
this via:
    python3 hardware_profile.py            # human-readable summary
    python3 hardware_profile.py --json     # machine-readable dict
    python3 hardware_profile.py --key asr_model

Profile table (see references/hardware-profiles.md for rationale).
VLM lineup refreshed 2026-08: qwen3.5 is the only current Qwen generation
with the full small-size ladder (0.8b/2b/4b/9b, unified text+vision, 256K
context); qwen3.6/3.8 only ship 27b+ so they serve the `max` tier. ModelBest's
end-side models cover the low tiers: minicpm-v4.6 (1B, ultra-efficient image/
video understanding, strong CJK OCR) and minicpm5 (text, 688 MB Q4).

  profile   RAM        GPU            ASR model   compute      VLM
  -------   --------   -------------  ----------  -----------  ----------------------
  tiny      < 6 GB     any            tiny        int8         qwen3.5:0.8b (1.0 GB)
  low       6–8 GB     none/integrated base        int8         minicpm-v4.6 (1.6 GB)
  low-mac   6–8 GB     Apple Silicon  small       int8         minicpm-v4.6 (1.6 GB)
  mid       8–16 GB    any            small       int8         minicpm-v4.6 (1.6 GB)
  high      16–32 GB   any            medium      int8_float16 qwen3.5:4b (3.4 GB)
  high-gpu  >= 8 GB    NVIDIA >=8GB    large-v3    float16      qwen3.5:9b (6.6 GB)
  max       > 32 GB    any            large-v3    float16      qwen3.8:27b (18 GB)

NVIDIA GPUs short-circuit to high-gpu (CUDA + float16 is always faster than CPU)
regardless of total RAM, as long as VRAM >= 8 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys

# --- thresholds -------------------------------------------------------------
PROFILES = {
    "tiny":     {"min_ram": 0,  "asr": "tiny",     "compute": "int8",         "device": "cpu",  "vlm": "qwen3.5:0.8b",                   "note": "极低配/老设备，仅保证能跑"},
    "low":      {"min_ram": 6,  "asr": "base",     "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.6:latest",    "note": "6-8GB 无独立GPU"},
    "low-mac":  {"min_ram": 6,  "asr": "small",    "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.6:latest",    "note": "Apple Silicon 6-8GB（Metal 加速抽帧）"},
    "mid":      {"min_ram": 8,  "asr": "small",    "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.6:latest",    "note": "8-16GB 通用"},
    "high":     {"min_ram": 16, "asr": "medium",   "compute": "int8_float16", "device": "auto", "vlm": "qwen3.5:4b",                     "note": "16-32GB，可上 medium"},
    "high-gpu": {"min_ram": 8,  "asr": "large-v3", "compute": "float16",      "device": "cuda", "vlm": "qwen3.5:9b",                     "note": "NVIDIA >=8GB VRAM，CUDA 全速"},
    "max":      {"min_ram": 32, "asr": "large-v3", "compute": "float16",      "device": "auto", "vlm": "qwen3.8:27b",                    "note": "工作站/服务器 >32GB"},
}


# --- detection --------------------------------------------------------------

def detect_os() -> str:
    return platform.system().lower()  # darwin / linux / windows


def detect_arch() -> str:
    return platform.machine().lower()  # arm64 / x86_64 / aarch64


def _win_ram_gb() -> float:
    """Total physical RAM via the native Win32 API.

    Works on every Windows since 2000 (kernel32.GlobalMemoryStatusEx) and needs
    no subprocess — this is the primary probe on all Windows versions because
    wmic.exe was removed starting with Win11 24H2 / Server 2025.
    """
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")
    return stat.ullTotalPhys / 1073741824


def detect_ram_gb() -> float:
    """Total physical RAM in GB. Returns best-effort; 0 if unknown.

    Per-OS probe chains (first hit wins; every layer is best-effort):
      darwin   : sysctl -n hw.memsize                      (all macOS)
      linux    : /proc/meminfo MemTotal                     (all Linux)
                 -> sysconf(_SC_PHYS_PAGES * _SC_PAGE_SIZE) (POSIX fallback)
      windows  : ctypes GlobalMemoryStatusEx                (Win2000+, primary)
                 -> wmic ComputerSystem                     (XP..Win11 23H2;
                    removed in 24H2+/Server 2025, kept for old boxes)
                 -> PowerShell Get-CimInstance              (Win8+, last resort)
    """
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
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1073741824
        if osn == "windows":
            try:
                return _win_ram_gb()
            except Exception:
                pass
            try:
                out = subprocess.run(
                    ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                    capture_output=True, text=True, check=True)
                for tok in out.stdout.split():
                    if tok.isdigit():
                        return int(tok) / 1073741824
            except Exception:
                pass
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, check=True, timeout=30)
            return float(out.stdout.strip()) / 1073741824
    except Exception:
        pass
    try:  # generic POSIX last resort
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1073741824
    except Exception:
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


def _win_nvidia_vram_gb() -> float | None:
    """NVIDIA VRAM from the Windows display-class registry key.

    Used when nvidia-smi is not on PATH (e.g. driver installed but bin dir not
    exported, or Git-Bash PATH differences). Reads
      HKLM\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-...}\\00NN
      DriverDesc                     -> must mention NVIDIA
      HardwareInformation.qwMemorySize -> bytes (QWORD; DWORD on old drivers)
    A sanity clamp (0.3-128 GB) rejects garbage or wrong-unit values so a bad
    probe can never mis-trigger the high-gpu profile.
    """
    import winreg

    cls = (r"SYSTEM\CurrentControlSet\Control\Class"
           r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls) as k0:
            for i in range(16):
                try:
                    with winreg.OpenKey(k0, f"{i:04d}") as k:
                        try:
                            desc = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                        except OSError:
                            continue
                        if "nvidia" not in desc.lower():
                            continue
                        try:
                            raw, _typ = winreg.QueryValueEx(
                                k, "HardwareInformation.qwMemorySize")
                        except OSError:
                            continue
                        gb = int(raw) / 1073741824
                        if 0.3 < gb < 128:
                            return gb
                except OSError:
                    continue
    except OSError:
        pass
    return None


def detect_nvidia_vram_gb() -> float | None:
    """Return VRAM in GB of first NVIDIA GPU, or None.

    Probe chain:
      1. nvidia-smi --query-gpu=memory.total   (any OS, driver's CLI on PATH)
      2. Windows registry qwMemorySize          (nvidia-smi missing/not on PATH)
    Non-NVIDIA dGPUs (Intel Arc, AMD) are not probed: CTranslate2 only
    accelerates via CUDA, so they never change the profile — ASR stays on CPU
    and Ollama uses whatever backend it was built with.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        first = out.stdout.strip().splitlines()[0]
        return int(first) / 1024  # MiB -> GiB
    except Exception:
        pass
    if detect_os() == "windows":
        return _win_nvidia_vram_gb()
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
        "os_release": platform.release(),  # e.g. 11 (Win11), 24.04 (Ubuntu)
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
                                  "device, vlm_model, profile, ram_gb, os_release)")
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
    print(f"  os/arch: {d['os']} {d['os_release']} / {d['arch']}")
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

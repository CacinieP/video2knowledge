#!/usr/bin/env python3
"""hardware_profile.py — detect the host machine and recommend a config profile
(ASR model size, compute type, device, VLM) that fits its RAM and GPU.

Single source of truth for the whole skill. Other scripts / setup_models.sh read
this via:
    python3 hardware_profile.py            # human-readable summary
    python3 hardware_profile.py --json     # machine-readable dict
    python3 hardware_profile.py --key asr_model

Profile table (see references/hardware-profiles.md for rationale).
Local inference is ONE model: ModelBest's minicpm-v4.7 (1B, native multimodal
— image/video understanding with strong CJK OCR, plus general text reasoning;
released 2026-10 as the successor of v4.6). It serves every tier for BOTH
roles — Path-1 visual captioning/OCR and Path-3 text synthesis — so a single
1.8 GB pull covers the whole local pipeline and the `vlm`/`text` fields below
always agree. The default is the PINNED `:1b` tag: on this family `:latest`
resolves to the 35B MoE (22 GB) and would swamp the low tiers — never point a
profile at it. A bigger local model is an explicit override (--model /
--vlm-model / V2K_TEXT_MODEL), and the alternative for every LLM/VLM stage is
any OpenAI-compatible /chat/completions endpoint (--api-base; see
references/path1-multimodal.md). ASR stays tiered and hardware-driven.

  profile   RAM        GPU            ASR model   compute      VLM & text
  -------   --------   -------------  ----------  -----------  ----------------------
  tiny      < 6 GB     any            tiny        int8         minicpm-v4.7:1b (1.8 GB)
  low       6–8 GB     none/integrated base        int8         minicpm-v4.7:1b (1.8 GB)
  low-mac   6–8 GB     Apple Silicon  small       int8         minicpm-v4.7:1b (1.8 GB)
  mid       8–16 GB    any            small       int8         minicpm-v4.7:1b (1.8 GB)
  high      16–32 GB   any            medium      int8*        minicpm-v4.7:1b (1.8 GB)

  * int8_float16 when CUDA is present, auto-downgraded to int8 on CPU-only
    machines (CTranslate2's CPU backend rejects int8_float16 — and plain
    float16 — at load; verified on Apple Silicon, where "max" used to emit
    float16 on the CPU device and faster-whisper refused to start).
  high-gpu  >= 8 GB    NVIDIA >=8GB    large-v3    float16      minicpm-v4.7:1b (1.8 GB)
  max       > 32 GB    any            large-v3    float16†     minicpm-v4.7:1b (1.8 GB)

  † float16 only when the machine really lands on CUDA; on CPU-only hosts the
    resolver downgrades it to int8, same as the int8_float16 tier.

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
    "tiny":     {"min_ram": 0,  "asr": "tiny",     "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "极低配/老设备，仅保证能跑"},
    "low":      {"min_ram": 6,  "asr": "base",     "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "6-8GB 无独立GPU"},
    "low-mac":  {"min_ram": 6,  "asr": "small",    "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "Apple Silicon 6-8GB（Metal 加速抽帧）"},
    "mid":      {"min_ram": 8,  "asr": "small",    "compute": "int8",         "device": "cpu",  "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "8-16GB 通用（MiniCPM v4.7 本地全链路）"},
    "high":     {"min_ram": 16, "asr": "medium",   "compute": "int8_float16", "device": "auto", "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "16-32GB，可上 medium"},
    "high-gpu": {"min_ram": 8,  "asr": "large-v3", "compute": "float16",      "device": "cuda", "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "NVIDIA >=8GB VRAM，CUDA 全速"},
    "max":      {"min_ram": 32, "asr": "large-v3", "compute": "float16",      "device": "auto", "vlm": "openbmb/minicpm-v4.7:1b",    "text": "openbmb/minicpm-v4.7:1b",    "note": "工作站/服务器 >32GB"},
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


def detect_cuda_usable() -> bool:
    """True if CTranslate2 can actually LOAD a model on CUDA right now.

    `get_cuda_device_count()` only proves a driver is visible. The CUDA runtime
    libraries themselves ship separately, and on Windows they are not on PATH
    by default even when the driver is fine — ctranslate2 then reports a device
    and dies at load time with

        RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

    so a device count is not evidence. Probing the DLLs the loader actually
    needs is: if they resolve, CUDA works; if not, saying `device=cuda` sends
    every run into that RuntimeError. Returns False on any doubt — CPU is the
    safe answer, just a slower one.
    """
    if detect_os() != "windows":
        return True  # non-Windows resolves CUDA through the system loader
    try:
        import ctranslate2  # noqa: F401  (import is the cheap part)
    except Exception:
        return False
    try:
        import ctypes
        import glob
        import os
        import site
    except Exception:
        return False

    # cublas is the one CTranslate2 always needs; cudnn for the conv path.
    roots = []
    for getter in (lambda: site.getsitepackages(), lambda: [site.getusersitepackages()]):
        try:
            roots.extend(getter())
        except Exception:
            pass
    try:  # uv / venv layouts keep packages under Lib\site-packages
        roots.extend(glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")))
    except Exception:
        pass

    for dll in ("cublas64_12.dll",):
        found = any(os.path.isfile(os.path.join(r, dll)) for r in roots)
        if not found:
            # last resort: already somewhere on PATH?
            from shutil import which
            found = which(dll) is not None
        if not found:
            return False
    try:
        ctypes.CDLL(os.path.join(next(r for r in roots
                                      if os.path.isfile(os.path.join(r, "cublas64_12.dll"))),
                                 "cublas64_12.dll"))
    except Exception:
        return False
    return True


def select_asr_device(nvidia_vram: float | None, cuda_ok: bool) -> str:
    """ASR device, decoupled from the profile tier.

    The tier gate (`nvidia_vram >= 8` -> high-gpu) is about picking a whole
    configuration; it is the wrong test for "should ASR use the GPU". Measured
    on an RTX 3060 Laptop (6 GB), faster-whisper small on CUDA int8_float16 ran
    at 17.9x realtime versus 2.5x on CPU — a 4-7x difference — and the tier
    table sent that machine to `mid`/cpu. The whisper models fit in far less
    than 8 GB (small int8 ~1.2 GB, large-v3 int8_float16 ~3 GB), so 3 GB of
    usable VRAM plus a working runtime is the honest threshold.
    """
    if nvidia_vram and nvidia_vram >= 3.0 and cuda_ok:
        return "cuda"
    return "cpu"


def recommend_asr_backend(d: dict) -> tuple[str, str]:
    """Return (backend, reason) for this host — which ASR should you actually use.

    This is advice, not configuration: nothing here changes what the scripts do
    unless you pass the answer back as `--asr-backend`. It exists because the
    three backends are good at very different things and the right one is not
    obvious from RAM alone.

    Measured on a 286-video Mandarin course (61.5 h) on an RTX 3060 Laptop,
    6 GB VRAM, both engines on the same 30-minute lecture:

        faster-whisper (small + 39 hotwords)   44% correct on homophone terms
        Paraformer-zh (FunASR, CPU)            94%, and 12-16x realtime

    Paraformer wins on Mandarin accuracy and it wins on CPU, so the VRAM
    question barely matters for it. An earlier version of this function
    recommended funasr *because* it saw >= 8 GB VRAM and described it as
    "GPU-accelerated"; that was wrong twice over — Paraformer runs on CPU
    here, and 8 GB is the profile-tier gate, not the ASR gate (see
    `select_asr_device`, which is the honest one).

    Cloud is recommended only when the host genuinely cannot keep up, because
    sending a course recording to a third party is a real privacy decision and
    should be the fallback, not the default.
    """
    ram = float(d.get("ram_gb") or 0)
    nv = d.get("nvidia_vram_gb")
    apple = d.get("apple_chip")
    cuda_ok = bool(d.get("cuda_ok"))

    # Nothing local will be tolerable below this.
    if ram < 6:
        return ("openai-api",
                f"only {ram} GB RAM — local ASR is too slow to be usable. "
                f"A cloud OpenAI-compatible endpoint is the practical option; "
                f"note that it uploads the audio to a third party.")

    if ram >= 8:
        why = (f"Paraformer-zh (FunASR) transcribes Mandarin homophones far more "
               f"accurately than faster-whisper — measured 94% vs 44% on a "
               f"piano course. It runs on CPU, so {ram} GB RAM is enough")
        if apple:
            why += f", and Apple Silicon ({apple}) makes CPU ASR fast"
        elif nv and cuda_ok:
            why += (f". Your {nv} GB GPU stays free for the VLM — do not run "
                    f"whisper and the VLM at the same time or they fight over "
                    f"VRAM")
        return ("funasr", why + ".")

    return ("faster-whisper",
            f"{ram} GB RAM with no usable GPU — faster-whisper 'small' is the "
            f"local default and fits comfortably. If Mandarin accuracy matters "
            f"more than speed, FunASR on CPU still beats it and is the better "
            f"trade here.")


def detect() -> dict:
    """Run all detection and return a full profile dict."""
    ram = detect_ram_gb()
    nvidia = detect_nvidia_vram_gb()
    apple = detect_apple_silicon()
    cuda_ok = detect_cuda_usable() if nvidia else False
    pname = select_profile(ram, nvidia_vram=nvidia, apple_chip=apple)
    prof = PROFILES[pname]
    compute = prof["compute"]
    device = select_asr_device(nvidia, cuda_ok)
    # int8_float16 and plain float16 are CUDA-only compute types: CTranslate2's
    # CPU backend raises "Requested ... compute type, but the target device or
    # backend do not support efficient ..." at load for both (hit on an M5 Max
    # where the "max" profile emitted float16 onto the CPU device and
    # faster-whisper refused to start). Downgrade to plain int8 when CUDA is
    # not actually usable...
    if compute in ("int8_float16", "float16") and device != "cuda":
        compute = "int8"
    # ...and upgrade the tiers that settle on plain int8 now that a real CUDA
    # device was verified. Same int8 weights, float16 math: strictly faster,
    # and it costs no extra VRAM over int8.
    elif compute == "int8" and device == "cuda":
        compute = "int8_float16"
    d = {
        "os": detect_os(),
        "os_release": platform.release(),  # e.g. 11 (Win11), 24.04 (Ubuntu)
        "arch": detect_arch(),
        "ram_gb": round(ram, 1),
        "apple_chip": apple,
        "nvidia_vram_gb": round(nvidia, 1) if nvidia else None,
        "cuda_ok": cuda_ok,
        "profile": pname,
        "asr_model": prof["asr"],
        "compute_type": compute,
        "device": device,
        "vlm_model": prof["vlm"],
        "text_model": prof["text"],
        "note": prof["note"],
    }
    backend, why = recommend_asr_backend(d)
    d["recommended_asr_backend"] = backend
    d["recommended_backend_reason"] = why
    return d


# Step 2's text model when detection cannot run at all (stripped container with
# no /proc and no nvidia-smi): keep the unified v4.7 pull — one model already
# serves every other stage, and a separate fallback text model would force a
# second download on machines that could not even run the detector.
FALLBACK_TEXT_MODEL = "openbmb/minicpm-v4.7:1b"

_default_text_model: str | None = None


def default_text_model() -> str:
    """Resolve Step 2's text model: V2K_TEXT_MODEL > profile table > fallback.

    Cached per process — detect() shells out to nvidia-smi (10 s timeout) and
    the answer cannot change mid-run. Never raises: a failed probe must not
    stop a run that a flat default would have completed.
    """
    global _default_text_model
    if _default_text_model is None:
        _default_text_model = os.environ.get("V2K_TEXT_MODEL") or ""
        if not _default_text_model:
            try:
                _default_text_model = detect()["text_model"]
            except Exception:
                _default_text_model = FALLBACK_TEXT_MODEL
    return _default_text_model


# --- CLI --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--key", help="print a single field (asr_model, compute_type, "
                                  "device, vlm_model, text_model, profile, ram_gb, "
                                  "os_release, cuda_ok, recommended_asr_backend)")
    ap.add_argument("--recommend", action="store_true",
                    help="print the recommended ASR backend and why, plus a "
                         "ready-to-paste command. Runs no ASR.")
    args = ap.parse_args()
    d = detect()
    if args.recommend:
        backend = d["recommended_asr_backend"]
        print(f"recommended ASR backend: {backend}")
        print(f"  {d['recommended_backend_reason']}")
        if backend == "funasr":
            print("\n  batch_run.py --asr-backend funasr --asr-python <funasr-venv>/Scripts/python.exe")
        elif backend == "openai-api":
            print("\n  asr_caption.py --backend openai-api --api-base <URL> "
                  "--api-model <MODEL> --api-key-env <ENV_VAR>")
        else:
            print("\n  (default) batch_run.py --asr-model "
                  f"{d['asr_model']}")
        return 0
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
    nv = f"  nvidia:  {d['nvidia_vram_gb']} GB VRAM (cuda_ok={d['cuda_ok']})\n" if d["nvidia_vram_gb"] else ""
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
    print(f"  -> text_model:   {d['text_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

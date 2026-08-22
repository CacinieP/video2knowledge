# Hardware Profiles

`scripts/hardware_profile.py` is the single source of truth for "what model /
backend fits this machine." Every script reads it so you never have to guess
ASR model size or compute type by hand.

## How it works

On first run, `setup_models.sh` and `asr_caption.py` call
`hardware_profile.py`, which detects OS, CPU arch, total RAM, Apple Silicon
chip, and NVIDIA VRAM. It then maps the machine to one of the profiles below
and uses the corresponding defaults for the whole pipeline.

Inspect yours:

```bash
python3 scripts/hardware_profile.py            # human-readable
python3 scripts/hardware_profile.py --json     # machine-readable
python3 scripts/hardware_profile.py --key asr_model
```

Override anything with env vars (highest priority) or CLI flags:

```bash
ASR_DEFAULT_MODEL=medium bash scripts/setup_models.sh
# or per-run:
python3 scripts/asr_caption.py --video v.mp4 --out-dir o --model large-v3 --device cuda
```

## Profile table

| Profile | Trigger | ASR model | compute | device | VLM | Notes |
|---|---|---|---|---|---|---|
| `tiny` | RAM < 6 GB | `tiny` | int8 | cpu | `qwen3.5:0.8b` (~1.0 GB) | 老设备/上网本，仅保证能跑，字幕较粗 |
| `low` | 6–8 GB, no dGPU | `base` | int8 | cpu | `minicpm-v4.6` | 通用低配 |
| `low-mac` | 6–8 GB, Apple Silicon | `small` | int8 | cpu | `minicpm-v4.6` | M1/A 系列芯片，Metal 加速抽帧 |
| `mid` | 8–16 GB | `small` | int8 | cpu | `minicpm-v4.6` | **主流笔记本**（含 8GB MacBook） |
| `high` | 16–32 GB | `medium` | int8_float16 (int8 on CPU-only¹) | auto | `qwen3.5:4b` | 16G+，可上 medium |
| `high-gpu` | NVIDIA ≥ 8 GB VRAM | `large-v3` | float16 | **cuda** | `qwen3.5:9b` | 独显直通，CUDA 全速 |
| `max` | RAM > 32 GB | `large-v3` | float16 | auto | `qwen3.8:27b` | 工作站/服务器 |

### Model lineup (refreshed 2026-08)

- **qwen3.5** (Alibaba, ~Feb 2026): the only current Qwen generation with the
  full small-size ladder — 0.8b (1.0 GB) / 2b (2.7 GB) / 4b (3.4 GB) / 9b
  (6.6 GB) — unified text+vision (early fusion), 256K context. One pull serves
  both Path 1 (VLM) and Step 2 (text) on `high`+ machines.
- **qwen3.6** (~Apr 2026) / **qwen3.8** (~Aug 2026): ship 27b+ only (17–18 GB);
  qwen3.8 adds native image/video understanding (incl. hour-scale videos), so
  it is the `max`-tier pick. For a more battle-tested 27b, `qwen3.6:27b` works
  too.
- **ModelBest end-side models** (openbmb): `minicpm-v4.6` (1B, 1.6 GB,
  ultra-efficient image/video understanding, strong CJK OCR) holds the
  low/mid tiers; `minicpm5` (688 MB Q4) is the low-RAM text-model override.
  `minicpm-v4.5`/`minicpm-o4.5` (8B, GPT-4o-class omni) are solid
  alternatives where a 6 GB-class download fits.
- Legacy picks (`moondream`, `qwen2.5vl:3b/7b`) still work if already pulled,
  but new machines should use the lineup above.
- ASR: the whisper ladder (tiny→large-v3) is unchanged — faster-whisper
  (CTranslate2) remains the engine, and the 2026 leaderboard newcomers
  (Canary-Qwen 2.5B, Voxtral, FireRedASR) run on different inference stacks.
  `--model turbo` (large-v3-turbo, 809M) is the speed/accuracy sweet spot on
  GPUs where `large-v3` is too slow.

**NVIDIA short-circuit:** any machine with a CUDA GPU reporting ≥ 8 GB VRAM is
forced to `high-gpu` regardless of total RAM — CUDA + float16 always beats CPU,
and `large-v3` fits in 8 GB VRAM.

¹ `int8_float16` is CUDA-only: CTranslate2's CPU backend raises
"target device or backend do not support efficient int8_float16" at model load.
`hardware_profile.py` auto-downgrades it to `int8` when no NVIDIA GPU is
detected, so the `high` profile works on CPU-only 16–32 GB machines.

## ASR model sizing rationale

faster-whisper loads at `compute_type` into RAM/VRAM:

| Model | RAM (int8) | RAM (float16) | Word timestamps | Typical use |
|---|---|---|---|---|
| `tiny` | ~0.5 GB | ~1 GB | rough | 4GB / quick draft |
| `base` | ~0.7 GB | ~1.3 GB | ok | 6GB netbook |
| `small` | ~1.2 GB | ~2 GB | good | **8–16GB 主流** |
| `medium` | ~3.5 GB | ~5.5 GB | great | 16GB+ |
| `large-v3` | ~6.5 GB | ~10 GB | best | NVIDIA / 32GB+ |

Apple Silicon note: faster-whisper (CTranslate2) has limited Metal support, so
all Apple profiles default to `device=cpu, compute=int8` — fastest and safest on
macOS. On Linux/NVIDIA, `device=cuda, compute=float16` gives a large speedup.

## Detection details

All probes are best-effort with layered per-OS fallbacks; detection never
blocks the pipeline — if every layer fails, `mid` (small/int8/cpu/
minicpm-v4.6) is the safe fallback.

### RAM detection chain (per OS / version)

| OS / version | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| macOS (any) | `sysctl -n hw.memsize` | — | — |
| Linux (any) | `/proc/meminfo` `MemTotal:` | `sysconf(_SC_PHYS_PAGES × _SC_PAGE_SIZE)` (POSIX) | — |
| Windows 2000+ (all, incl. Win11 24H2+) | ctypes `GlobalMemoryStatusEx` (native kernel32, no subprocess) | `wmic ComputerSystem get TotalPhysicalMemory` (XP–Win11 23H2; **removed in Win11 24H2 / Server 2025**) | PowerShell `Get-CimInstance Win32_ComputerSystem` (Win8+) |

**Why the ctypes probe is primary on Windows:** older versions of this skill
called `wmic` directly. On Win11 24H2+ `wmic.exe` no longer exists, so the RAM
probe silently returned 0 and every such machine fell to the `tiny` profile
(tiny ASR + moondream VLM) regardless of its actual 16–32 GB. The native
`GlobalMemoryStatusEx` API works on every Windows since 2000 and needs no
subprocess, so it is now the first probe on all versions; `wmic` and PowerShell
CIM remain as fallbacks for exotic cases.

### NVIDIA VRAM detection chain

| Environment | Primary | Fallback |
|---|---|---|
| Any OS, NVIDIA driver + `nvidia-smi` on PATH | `nvidia-smi --query-gpu=memory.total` | — |
| Windows, `nvidia-smi` missing / not on PATH | registry `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-…}\00NN` → `DriverDesc` mentions NVIDIA → `HardwareInformation.qwMemorySize` (bytes) | sanity clamp 0.3–128 GB; invalid values return None (never mis-trigger `high-gpu`) |

Non-NVIDIA dGPUs (Intel Arc, AMD) are intentionally **not** probed: CTranslate2
(faster-whisper) only accelerates via CUDA, so they cannot change the profile —
ASR stays on CPU and Ollama uses whatever backend it was built with.

### Other probes

- **OS version**: `platform.release()` (reported as `os_release`, e.g. `11` on
  Win11, `24.04` on Ubuntu) — useful when diagnosing which probe chain ran.
- **Apple Silicon chip**: `system_profiler SPHardwareDataType`.

## Common machines → expected profile

| Machine | Profile | ASR | VLM |
|---|---|---|---|
| 4 GB old laptop / Raspberry Pi 4 | `tiny` | tiny | qwen3.5:0.8b |
| 8 GB Intel MacBook / ThinkPad | `mid` | small | minicpm-v4.6 |
| 8 GB M1 / M2 MacBook Air | `mid` | small | minicpm-v4.6 |
| 8 GB iPhone-class (A18 Pro) Mac | `mid` | small | minicpm-v4.6 |
| 16 GB M2/M3 Pro, 16 GB PC | `high` | medium | qwen3.5:4b |
| 24–32 GB M-Max / workstation | `max` | large-v3 | qwen3.8:27b |
| Any + RTX 3060/4060 (8 GB) | `high-gpu` | large-v3 | qwen3.5:9b |
| Any + RTX 3090/4090 (24 GB) | `high-gpu` | large-v3 | qwen3.5:9b |

## Tuning the profiles

Edit `PROFILES` in `scripts/hardware_profile.py`. Each entry sets `asr`,
`compute`, `device`, `vlm`, and a human note. Re-run `hardware_profile.py` to
confirm. Keep `select_profile()`'s thresholds in sync with the `min_ram`
values if you change the trigger ranges.

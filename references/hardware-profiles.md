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
| `tiny` | RAM < 6 GB | `tiny` | int8 | cpu | `moondream` (~1.6 GB) | 老设备/上网本，仅保证能跑，字幕较粗 |
| `low` | 6–8 GB, no dGPU | `base` | int8 | cpu | `minicpm-v4.6` | 通用低配 |
| `low-mac` | 6–8 GB, Apple Silicon | `small` | int8 | cpu | `minicpm-v4.6` | M1/A 系列芯片，Metal 加速抽帧 |
| `mid` | 8–16 GB | `small` | int8 | cpu | `minicpm-v4.6` | **主流笔记本**（含 8GB MacBook） |
| `high` | 16–32 GB | `medium` | int8_float16 | auto | `qwen2.5vl:3b` | 16G+，可上 medium |
| `high-gpu` | NVIDIA ≥ 8 GB VRAM | `large-v3` | float16 | **cuda** | `qwen2.5vl:7b` | 独显直通，CUDA 全速 |
| `max` | RAM > 32 GB | `large-v3` | float16 | auto | `qwen2.5vl:7b` | 工作站/服务器 |

**NVIDIA short-circuit:** any machine with a CUDA GPU reporting ≥ 8 GB VRAM is
forced to `high-gpu` regardless of total RAM — CUDA + float16 always beats CPU,
and `large-v3` fits in 8 GB VRAM.

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

- **RAM**: `sysctl hw.memsize` (macOS) / `/proc/meminfo` (Linux) /
  `wmic ComputerSystem` (Windows).
- **Apple Silicon chip**: `system_profiler SPHardwareDataType`.
- **NVIDIA VRAM**: `nvidia-smi --query-gpu=memory.total`.

Detection is best-effort and never blocks the pipeline — if anything fails,
`mid` (small/int8/cpu/minicpm-v4.6) is the safe fallback.

## Common machines → expected profile

| Machine | Profile | ASR | VLM |
|---|---|---|---|
| 4 GB old laptop / Raspberry Pi 4 | `tiny` | tiny | moondream |
| 8 GB Intel MacBook / ThinkPad | `mid` | small | minicpm-v4.6 |
| 8 GB M1 / M2 MacBook Air | `mid` | small | minicpm-v4.6 |
| 8 GB iPhone-class (A18 Pro) Mac | `mid` | small | minicpm-v4.6 |
| 16 GB M2/M3 Pro, 16 GB PC | `high` | medium | qwen2.5vl:3b |
| 24–32 GB M-Max / workstation | `max` | large-v3 | qwen2.5vl:7b |
| Any + RTX 3060/4060 (8 GB) | `high-gpu` | large-v3 | qwen2.5vl:7b |
| Any + RTX 3090/4090 (24 GB) | `high-gpu` | large-v3 | qwen2.5vl:7b |

## Tuning the profiles

Edit `PROFILES` in `scripts/hardware_profile.py`. Each entry sets `asr`,
`compute`, `device`, `vlm`, and a human note. Re-run `hardware_profile.py` to
confirm. Keep `select_profile()`'s thresholds in sync with the `min_ram`
values if you change the trigger ranges.

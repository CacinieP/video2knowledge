# Path 2 — ASR Transcription (faster-whisper)

Path 2 extracts the audio track and transcribes it with **faster-whisper** (the
CTranslate2 backend of Whisper), producing word/segment-level timestamps. This is
the recommended path whenever the video has a clear speech track.

## When to choose Path 2

- Lecture, talk, interview, podcast video, tutorial with narration.
- You need accurate, fine-grained timestamps (segment + word level).
- You want the highest fidelity text — Whisper models are SOTA for general speech.

Avoid Path 2 when: the video is silent/slide-only, audio is non-speech, or you
need what is visually on screen (use Path 1 or run both).

## Model sizing

The default `--model` / `--compute-type` / `--device` are **auto-selected from
your hardware profile** (`scripts/hardware_profile.py`) — you don't have to size
them by hand. The table below shows the RAM footprint at `compute_type=int8` so
you understand what the profile picker chose; see
`references/hardware-profiles.md` for the full profile → model mapping.

| Model | RAM (int8) | Word timestamps | Profile that picks it |
|---|---|---|---|
| `tiny` | ~0.5 GB | rough | `tiny` (RAM < 6 GB) |
| `base` | ~0.7 GB | ok | `low` (6–8 GB, no dGPU) |
| `small` | ~1.2 GB | good | `low-mac`, `mid` (8–16 GB) |
| `medium` | ~3.5 GB | great | `high` (16–32 GB) |
| `large-v3` | ~6.5 GB | best | `high-gpu`, `max` |

`asr_caption.py` **warns** (but does not block) if you force a heavy model on a
low-RAM profile. Override explicitly with `--model`.

## Device & compute type

These come from the hardware profile. The defaults by profile:

- **Apple Silicon / CPU-only** (`mid`, `low`, `low-mac`): `device=cpu`,
  `compute=int8`. faster-whisper (CTranslate2) has limited Metal support; CPU
  int8 is the fastest, most reliable path on macOS.
- **16 GB+** (`high`): `device=auto`, `compute=int8_float16`.
- **NVIDIA** (`high-gpu`): `device=cuda`, `compute=float16` — large speedup.

Override per-run with `--device` and `--compute-type`.

## Language

- `--language zh` (default): assume Chinese. Good for Chinese lectures.
- `--language en`, `--language ja`, …: any Whisper language code.
- `--language auto`: auto-detect (slightly slower, can misfire on code-switching).

## VAD filter

`vad_filter=True` is hardcoded — it trims silence, which dramatically improves
segment timestamps and speed for lectures with long pauses. Disable in
`asr_caption.py` only if you need exact wall-clock silence boundaries.

## Outputs

```
<out-dir>/
├── audio_16k.wav                # intermediate (16k mono, for the model)
├── subtitles.srt                # timestamped subtitles
├── subtitles.vtt                # WebVTT
└── subtitles.json               # {language, duration, segments:[{start,end,text}]}
```

`subtitles.json` is the canonical input to `build_knowledge.py`. `segments` is a
list of `{start,end,text}` — same schema Path 1 emits, so Step 2 is path-agnostic.

## End-to-end example

```bash
source ~/.zcode/skills/video2knowledge/.venv/bin/activate
cd ~/.zcode/skills/video2knowledge
python3 scripts/asr_caption.py \
  --video ~/Movies/lecture.mp4 \
  --out-dir runs/lecture-asr \
  --model small --language zh
python3 scripts/build_knowledge.py \
  --subtitles runs/lecture-asr/subtitles.json \
  --out-dir runs/lecture-asr --format all
python3 scripts/gen_apkg.py \
  --csv runs/lecture-asr/cards.csv \
  --out runs/lecture-asr/cards.apkg --deck "课程知识卡"
```

## Running both paths (cross-validation)

For high-value content, run Path 2 for accurate text + Path 1 for visual context,
then let the LLM in Step 2 merge them. Point `build_knowledge.py` at the ASR
subtitles (better text), and paste key multimodal captions into the prompt or a
custom template's `{{summary}}` slot.

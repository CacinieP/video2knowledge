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

## Model sizing (important on 8 GB RAM)

faster-whisper loads into RAM/VRAM at `compute_type=int8` by default.

| Model | RAM (int8) | Word timestamps | Recommendation |
|---|---|---|---|
| `tiny` | ~0.5 GB | rough | quick & dirty |
| `base` | ~0.7 GB | ok | short clips |
| **`small`** | ~1.2 GB | good | **default — best for 8 GB** |
| `medium` | ~3.5 GB | great | ≥16 GB RAM only |
| `large-v3` | ~6.5 GB | best | ≥16 GB RAM only |

`asr_caption.py` **warns** (but does not block) if you pick `medium`/`large*` on a
machine reporting <16 GB. On the A18 Pro / 8 GB target, stick to `small`.

## Device & compute type

- `--device cpu` (default): fastest, most reliable on macOS. faster-whisper uses
  CTranslate2 which has limited MPS support; CPU `int8` is the safe default.
- `--device auto`: let the library choose (may pick Metal on newer builds).
- `--compute-type int8` (default): smallest memory, good speed. Use
  `int8_float16` on Apple Silicon if you observe good CPU/GPU balance.

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

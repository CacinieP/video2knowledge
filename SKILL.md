---
name: video2knowledge
description: >-
  Convert videos into timestamped subtitles and structured knowledge artifacts.
  Two ingestion paths: (1) a native multimodal VLM (at most 4B params, via
  Ollama) reads sampled video frames into timestamped captions; (2)
  faster-whisper ASR transcribes the audio into timestamped subtitles. Then
  refine into a structured knowledge doc (custom template supported), a
  self-contained HTML page, a knowledge-card CSV, and an Anki apkg deck.
  Fully local — no video, audio, or output ever leaves the host; the repo
  tracks only code and config changes. Use when transcribing or summarizing a
  video, building study cards from a lecture or recording, turning a silent or
  screen-recording video into notes, or producing reviewable knowledge from any
  video file.
---

# Video2Knowledge

Turn a video into: **timestamped subtitles → knowledge doc → HTML / Anki cards**.
Two local ingestion paths, all-local inference (no cloud API). **Processing stays
fully local** — videos, subtitles, and outputs are never uploaded; they live in a
local `runs/` folder (gitignored). The repo tracks only code/config changes.

## Prerequisites & First-Time Setup

Required on the host: `ollama`, `ffmpeg`, `python3` (or `uv`). One-time:

```bash
bash scripts/setup_models.sh
```

This (idempotently) **auto-detects your machine's hardware profile** (RAM / GPU /
Apple Silicon / NVIDIA) via `scripts/hardware_profile.py`, then pulls the
recommended VLM, creates a venv at `.venv/`, and installs `faster-whisper` +
`genanki`. See what it picked:

```bash
python3 scripts/hardware_profile.py
```

Profiles range from `tiny` (4 GB machines → whisper-tiny + qwen3.5:0.8b) through
`high-gpu` (NVIDIA ≥8 GB → whisper-large-v3 + qwen3.5:9b on CUDA). Full table
and tuning in `references/hardware-profiles.md`. Override any choice with env
vars (`VLM_MODEL=`, `ASR_DEFAULT_MODEL=`, ...) or CLI flags.

Activate the venv before running any python step:

```bash
source .venv/bin/activate
```

## Choose an Ingestion Path (Step 1)

Decision tree:

- **Clear speech track** (lecture, talk, interview, narration) → **Path 2 (ASR)**.
  Faster, more accurate text, fine-grained timestamps. → `references/path2-asr.md`
- **Silent / slide-only / screen recording**, or you need on-screen text & diagrams
  → **Path 1 (multimodal)**. → `references/path1-multimodal.md`
- **Slide/PPT video with a speech track** (the common online-course case): the core
  content is on screen (tables, formulas, examples) AND the speaker narrates →
  **Path 3 (dual-path fusion)**. Runs ASR for narration + VLM OCR for slides and
  fuses them by timestamp. → `references/path3-fusion.md`

Paths 1 & 2 emit the same segment schema (`{start,end,text}`), so Step 2 is
path-agnostic. Path 3 emits a fused `merged.json` consumed by Step 2's `--merged`.

### Path 1 — Multimodal captions

```bash
python3 scripts/mm_caption.py \
  --video VIDEO --out-dir OUT \
  --model openbmb/minicpm-v4.6:latest --interval 2.0
```
Outputs: `OUT/captions.srt`, `OUT/captions.json`, `OUT/frames/`.

For slide/PPT videos, use dedup sampling (general-purpose, no threshold tuning)
plus the table/formula OCR prompt:

```bash
python3 scripts/mm_caption.py \
  --video VIDEO --out-dir OUT \
  --mode dedup --prompt-ocr
```

### Path 2 — ASR transcription

```bash
python3 scripts/asr_caption.py \
  --video VIDEO --out-dir OUT \
  --model small --language zh \
  --hotwords "术语一, 术语二"   # optional jargon biasing via initial_prompt
```
Outputs: `OUT/subtitles.{srt,vtt,json}`.

### Path 3 — Dual-path fusion (ASR × VLM)

Run Path 2 then Path 1 (OCR), fuse, and build from the merged file:

```bash
# 1a. ASR
python3 scripts/asr_caption.py --video VIDEO --out-dir OUT --language zh
# 1b/1c. dedup frames + VLM OCR
python3 scripts/mm_caption.py --video VIDEO --out-dir OUT --mode dedup --prompt-ocr
# 2. fuse by timestamp
python3 scripts/merge_visual.py --subtitles OUT/subtitles.json --visual OUT/captions.json --out OUT/merged.json
# 3. build (Step 2 with --merged)
python3 scripts/build_knowledge.py --subtitles OUT/subtitles.json --merged OUT/merged.json --out-dir OUT --format all
```
The default text model is `qwen3.5:4b` (unified vision+text — on `high` machines
the same pull serves Path 1 and Step 2; override with
`--model openbmb/minicpm5:Q4_K_M` for low-RAM/fast runs).

## Refine into Knowledge Artifacts (Step 2)

Point `build_knowledge.py` at either path's `.json` output:

```bash
python3 scripts/build_knowledge.py \
  --subtitles OUT/subtitles.json \
  --out-dir OUT --format all
```

Produces, in `OUT/`:

| Artifact | File | Section |
|---|---|---|
| 2.1 Knowledge doc (templated) | `knowledge.md` | uses `assets/default-template.md` or `--template <file>` |
| 2.2 Self-contained HTML | `knowledge.html` | clickable `[mm:ss]` timeline |
| 2.2b Office/print export | `knowledge.docx` / `knowledge.pdf` | `--format docx` / `--format pdf` (auto in `--format all` when python-docx/fpdf2 installed) |
| 2.3 Knowledge cards | `cards.csv` | `question,answer,tags,timestamp,source` |
| 2.4 Illustrated notes (图文笔记) | `notes.md` + `notes.html` | key frames × narration, see below |
| 2.4b Notes office/print | `notes.docx` / `notes.pdf` | `build_notes.py --docx --pdf` (frames embedded) |

Then convert cards to Anki (2.3 final):

```bash
python3 scripts/gen_apkg.py --csv OUT/cards.csv --out OUT/cards.apkg --deck "视频知识卡"
```

### Illustrated notes (2.4, 图文笔记)

A scrollable note that interleaves deduped KEY FRAMES with the narration around
each timestamp — the "watch it back as a reading" artifact. Needs `frames.json`
(from `extract_frames.py`, usually via Path 1's dedup mode) plus either path's
subtitles:

```bash
python3 scripts/build_notes.py \
  --subtitles OUT/subtitles.json \
  --frames OUT/frames/frames.json \
  --out-dir OUT --max-frames 12 --describe-frames
```

Per key frame: LLM section title → frame image → optional VLM 画面 description
(`--describe-frames`) → 1-2 sentence note condensed from the narration window →
verbatim 原声 excerpt. Emits `notes.md` (relative image refs, renders anywhere
the frames dir travels) and `notes.html` (frames embedded as base64 — a single
shareable file). Degrades to frames + raw excerpts when no model is reachable.

See `references/outputs.md` for schema, single-format runs, and degraded mode.

## Custom Templates (2.1)

Default template: `assets/default-template.md`. Override with `--template <file>`:

```bash
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o \
  --template ./my-lecture-template.md --format knowledge
```

Templates are plain Markdown using `{{placeholders}}` (`{{title}}`, `{{summary}}`,
`{{timeline}}`, `{{key_points}}`, `{{qa}}`, `{{glossary}}`, `{{source}}`,
`{{duration}}`, `{{date}}`, `{{meta}}`). Only placeholders you include are filled;
everything else stays verbatim. Full spec + 3 example templates
(lecture / meeting / tutorial) in `references/templates.md`.

## Local-Only Processing & Change Tracking

**All video processing stays on your machine.** Videos, extracted frames,
subtitles, knowledge docs, and cards are written to a local `runs/` folder that is
**gitignored** — nothing about your media is ever uploaded or committed.

The GitHub repo tracks **only code and config changes** (scripts, references,
templates, README). This gives a clear history of how the skill evolved, without
exposing any user's media. Use descriptive `feat:`/`fix:`/`docs:` commit messages.

If you want a local record of a specific run, write a `manifest.json` into that
run folder (path taken, models, args, output list) — but keep it local:

```bash
cat > "$RUN/manifest.json" <<EOF
{"video":"~/Movies/lecture.mp4","path":"2","model":"small",
 "outputs":["subtitles.srt","knowledge.md","knowledge.html","cards.csv","cards.apkg"]}
EOF
```

## End-to-End Example

ASR path on a lecture, full pipeline (everything stays local):

```bash
source .venv/bin/activate
RUN=runs/$(date +%Y%m%d-%HMMSS)-lecture
mkdir -p "$RUN"

# Step 1 — subtitles
python3 scripts/asr_caption.py \
  --video ~/Movies/lecture.mp4 \
  --out-dir "$RUN" --language zh

# Step 2 — knowledge doc / HTML / CSV
python3 scripts/build_knowledge.py \
  --subtitles "$RUN/subtitles.json" \
  --out-dir "$RUN" --format all

# 2.3 — Anki deck
python3 scripts/gen_apkg.py \
  --csv "$RUN/cards.csv" --out "$RUN/cards.apkg"

# Optional local record (stays on your machine; runs/ is gitignored)
cat > "$RUN/manifest.json" <<EOF
{"video":"~/Movies/lecture.mp4","path":"2","model":"small","outputs":["subtitles.srt","knowledge.md","knowledge.html","cards.csv","cards.apkg"]}
EOF
```

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/setup_models.sh` | Idempotent model/venv setup (profile-aware) |
| `scripts/hardware_profile.py` | Detect machine → recommend ASR/VLM/backend profile |
| `scripts/extract_frames.py` | Frame sampling: `--mode interval` (uniform fps) or `--mode dedup` (dense sample + dHash dedup with **settle-frame** selection, blank-frame gate, optional `--hash-mode dual` dHash+aHash, and cluster-stratified `--max-frames` budget) → `frames.json` |
| `scripts/mm_caption.py` | Path 1: VLM captioning → `captions.{srt,json}`; `--mode dedup --prompt-ocr` for slide tables/formulas, with an OCR text-change gate that drops frames whose text is ≥90% similar to the last kept one |
| `scripts/asr_caption.py` | Path 2: faster-whisper → `subtitles.{srt,vtt,json}`; `--hotwords` biases transcription via initial_prompt |
| `scripts/merge_visual.py` | Path 3: fuse ASR `subtitles.json` × VLM `captions.json` by timestamp → `merged.json` (re-attach fallback keeps long-lived slides attached) |
| `scripts/build_knowledge.py` | Step 2: subtitles → knowledge.md / .html / cards.csv; `--merged` for dual-path fusion with `{{visual_timeline}}` section + map-reduce; `--format docx` / `--format pdf` for office/print |
| `scripts/build_notes.py` | Step 2.4: illustrated notes (图文笔记): key frames × narration → notes.md + self-contained notes.html; `--docx` / `--pdf` for office/print |
| `scripts/md_export.py` | Shared Markdown → DOCX/PDF exporter (python-docx + fpdf2, CJK font auto-detect, ffmpeg-JPEG normalize); standalone CLI for any pipeline .md |
| `scripts/gen_apkg.py` | Step 2.3: cards.csv → Anki `.apkg` |
| `scripts/batch_run.py` | Batch a whole course library: ASR + Ollama two-thread pipeline, resumable markers, priority `--order`, summary CSV |

## References (load as needed)

- `references/hardware-profiles.md` — profile table, sizing rationale, tuning
- `references/path1-multimodal.md` — VLM details, API format, sampling strategy
- `references/path2-asr.md` — model sizing, device/compute, language options
- `references/path3-fusion.md` — dual-path fusion (ASR × VLM OCR), dedup sampling, `--merged` build
- `references/templates.md` — placeholder spec + custom template examples
- `references/outputs.md` — HTML/CSV/APKG schemas, single-format runs, degraded mode

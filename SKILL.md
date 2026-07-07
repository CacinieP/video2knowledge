---
name: video2knowledge
description: >-
  Convert videos into timestamped subtitles and structured knowledge artifacts
  with full GitHub traceability. Two ingestion paths: (1) a native multimodal
  VLM (at most 4B params, via Ollama) reads sampled video frames into
  timestamped captions; (2) faster-whisper ASR transcribes the audio into
  timestamped subtitles. Then refine into a structured knowledge doc (custom
  template supported), a self-contained HTML page, a knowledge-card CSV, and an
  Anki apkg deck. Use when transcribing or summarizing a video, building study
  cards from a lecture or recording, turning a silent or screen-recording video
  into notes, or producing reviewable knowledge from any video file.
---

# Video2Knowledge

Turn a video into: **timestamped subtitles → knowledge doc → HTML / Anki cards**.
Two local ingestion paths, all-local inference (no cloud API), every run committed
to a public GitHub repo for traceability.

## Prerequisites & First-Time Setup

Required on the host: `ollama`, `ffmpeg`, `python3` (or `uv`). One-time:

```bash
bash scripts/setup_models.sh
```

This (idempotently) pulls the VLM `openbmb/minicpm-v4.6:latest`, creates a venv at
`.venv/`, and installs `faster-whisper` + `genanki`. Whisper model weights download
on first transcription. **On 8 GB RAM machines the ASR default is `small`** (see
`references/path2-asr.md` for sizing).

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
- **Both** (cross-validate high-value content): run Path 2 for text, Path 1 for
  visual context, feed ASR subtitles to Step 2 and fold visual captions in by hand
  or via a custom template.

Both paths emit the same segment schema (`{start,end,text}`), so Step 2 is
path-agnostic.

### Path 1 — Multimodal captions

```bash
python3 scripts/mm_caption.py \
  --video VIDEO --out-dir OUT \
  --model openbmb/minicpm-v4.6:latest --interval 2.0
```
Outputs: `OUT/captions.srt`, `OUT/captions.json`, `OUT/frames/`.

### Path 2 — ASR transcription

```bash
python3 scripts/asr_caption.py \
  --video VIDEO --out-dir OUT \
  --model small --language zh
```
Outputs: `OUT/subtitles.{srt,vtt,json}`.

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
| 2.3 Knowledge cards | `cards.csv` | `question,answer,tags,timestamp,source` |

Then convert cards to Anki (2.3 final):

```bash
python3 scripts/gen_apkg.py --csv OUT/cards.csv --out OUT/cards.apkg --deck "视频知识卡"
```

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

## Traceability & GitHub Logging

Every run **must** be committed to the skill repo (`CacinieP/video2knowledge`,
public) for traceability. Workflow:

1. Create a per-run folder: `runs/<YYYYMMDD-HHMMSS>-<video-basename>/`.
2. Write all script outputs there (frames, subtitles, knowledge.md/html/csv, apkg).
3. Write `runs/<...>/manifest.json` recording: source video path, path taken
   (1/2/both), models, all CLI args, start/end time, output file list, success/fail.
4. Commit and push:

```bash
git add runs/<...>
git commit -m "run(<ts>): <video-basename> path=<1|2> model=<...>"
git push
```

`runs/` and `.venv/` handling: commit `runs/`, **gitignore** `.venv/` and any large
raw videos (keep only a path reference in `manifest.json`). Commit template and
config changes to the skill itself with clear `feat:`/`fix:` messages.

The public repo is the audit trail: anyone can reproduce a run from its
`manifest.json` + the committed skill version.

## End-to-End Example

ASR path on a lecture, full pipeline with traceability:

```bash
source .venv/bin/activate
RUN=runs/$(date +%Y%m%d-%H%M%S)-lecture
mkdir -p "$RUN"

# Step 1 — subtitles
python3 scripts/asr_caption.py \
  --video ~/Movies/lecture.mp4 \
  --out-dir "$RUN" --model small --language zh

# Step 2 — knowledge doc / HTML / CSV
python3 scripts/build_knowledge.py \
  --subtitles "$RUN/subtitles.json" \
  --out-dir "$RUN" --format all

# 2.3 — Anki deck
python3 scripts/gen_apkg.py \
  --csv "$RUN/cards.csv" --out "$RUN/cards.apkg"

# Traceability
cat > "$RUN/manifest.json" <<EOF
{"video":"~/Movies/lecture.mp4","path":"2","model":"small",
 "language":"zh","outputs":["subtitles.srt","knowledge.md","knowledge.html",
 "cards.csv","cards.apkg"]}
EOF
git add "$RUN" && git commit -m "run: lecture path=2 model=small" && git push
```

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/setup_models.sh` | Idempotent model/venv setup (VLM pull + pip) |
| `scripts/extract_frames.py` | ffmpeg frame sampling → `frames.json` |
| `scripts/mm_caption.py` | Path 1: VLM captioning → `captions.{srt,json}` |
| `scripts/asr_caption.py` | Path 2: faster-whisper → `subtitles.{srt,vtt,json}` |
| `scripts/build_knowledge.py` | Step 2: subtitles → knowledge.md / .html / cards.csv |
| `scripts/gen_apkg.py` | Step 2.3: cards.csv → Anki `.apkg` |

## References (load as needed)

- `references/path1-multimodal.md` — VLM details, API format, sampling strategy
- `references/path2-asr.md` — model sizing, device/compute, language options
- `references/templates.md` — placeholder spec + custom template examples
- `references/outputs.md` — HTML/CSV/APKG schemas, single-format runs, degraded mode

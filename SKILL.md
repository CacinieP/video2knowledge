---
name: video2knowledge
description: >-
  Convert videos into timestamped subtitles and structured knowledge artifacts.
  Two ingestion paths: (1) a native multimodal VLM (at most 4B params, via
  Ollama) reads sampled video frames into timestamped captions; (2) ASR
  transcribes the audio into timestamped subtitles — faster-whisper locally, or
  FunASR Paraformer for Mandarin (measured 44% -> 94% on domain homophones,
  背谱 vs 被谱) with a tone-aware pinyin repair pass afterwards. Then refine
  into a structured knowledge doc (custom template supported), a self-contained
  HTML page, a deduplicated knowledge-card CSV, an Anki apkg deck, and
  illustrated notes (图文笔记) in two views — the full one with quoted narration
  and a distilled 纯享版 for revision. Fully local by default — no video, audio,
  or output ever leaves the host; cloud ASR/LLM endpoints are opt-in and
  explicitly labelled as uploading to a third party. The repo tracks only code
  and config changes. Use when transcribing or summarizing a video, building
  study cards from a lecture or recording, turning a silent or screen-recording
  video into notes, or producing reviewable knowledge from any video file.
---

# Video2Knowledge

Turn a video into: **timestamped subtitles → knowledge doc → HTML / Anki cards / 图文笔记**.
Two local ingestion paths, all-local inference by default (no cloud API).
**Processing stays fully local** — videos, subtitles, and outputs are never
uploaded; they live in a local `runs/` folder (gitignored). The repo tracks only
code/config changes.

> The one exception is opt-in and never automatic: `--backend openai-api` /
> `mimo-asr` (ASR) and `--api-base` (knowledge doc, any OpenAI-compatible
> endpoint) **send that audio or subtitle text to the third party you named**.
> Not passing them keeps everything on the host. Say so explicitly when you
> recommend them.

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
  Faster, more accurate text, fine-grained timestamps. For Mandarin, prefer the
  FunASR Paraformer backend — ask the machine with
  `python3 scripts/hardware_profile.py --recommend`.
  → `references/path2-asr.md`
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

**Which engine?** Ask the machine first:

```bash
python3 scripts/hardware_profile.py --recommend
```

Measured on a 61.5 h Mandarin course, both engines on the same 30-minute lecture
with the same 39-term hotword list:

| | homophone terms correct | speed |
|---|---|---|
| faster-whisper `small` + hotwords | 44% | 2.5× realtime |
| **Paraformer-zh (FunASR, CPU)** | **94%** | 12–16× realtime |

Hotwords cannot fix this: 背谱/被谱 are identical in pinyin *and* tone, so the
audio itself is ambiguous. The recommendation keys on RAM, not VRAM — Paraformer's
advantage is Mandarin accuracy and it runs on CPU. If you have a GPU, leave it
free for the VLM: running whisper and the VLM together turns an 8-token request
from 3.8 s into over 60 s.

Default (local, faster-whisper):

```bash
python3 scripts/asr_caption.py \
  --video VIDEO --out-dir OUT \
  --model small --language zh \
  --hotwords "术语一, 术语二"   # optional jargon biasing via initial_prompt
```

**Paraformer** (recommended for Mandarin; needs its own venv — funasr's
tokenizers pin does not co-install with the main environment):

```bash
python3 .venv-funasr/bin/python scripts/asr_funasr.py \
  --video VIDEO --out-dir OUT --language zh --hotwords "背谱,视谱,音阶"
```

Cloud engines exist for hosts that cannot transcribe locally, and are opt-in —
they upload the audio to a third party. `--backend openai-api` takes any
OpenAI-compatible endpoint; `--backend mimo-asr` is for chat-shaped gateways that
return no timestamps. See `references/path2-asr.md`.

Outputs: `OUT/subtitles.{srt,vtt,json}`.

**Then repair homophones.** ASR hears sounds, not words, so domain terms come
out spelled wrong and the error propagates into every downstream artifact.
Measured: 20 classes of genuine error in 46 places, 100% precision.

```bash
python3 scripts/fix_homophones.py --subtitles OUT/subtitles.json
```

Matches on tone-aware pinyin, not edit distance, and rewrites `subtitles.srt` /
`.vtt` to match. Run it before Step 2. `--dry-run` to preview, `--block 实度`
to blacklist a fragment, `--glossary @terms.txt` to add your own.

### Path 3 — Dual-path fusion (ASR × VLM)

Run Path 2 then Path 1 (OCR), feed OCR terms back into ASR hotwords, fuse,
and build from the merged file:

```bash
# 1a. ASR
python3 scripts/asr_caption.py --video VIDEO --out-dir OUT --language zh
# 1b/1c. dedup frames + VLM OCR
python3 scripts/mm_caption.py --video VIDEO --out-dir OUT --mode dedup --prompt-ocr
# 1d. OCR terms -> hotwords + coverage check (re-run ASR with
#     --hotwords @OUT/ocr_hotwords.txt when coverage is poor)
python3 scripts/hotwords_from_ocr.py --captions OUT/captions.json \
  --subtitles OUT/subtitles.json --out OUT/ocr_hotwords.txt
# 2. fuse by timestamp + semantic alignment check
python3 scripts/merge_visual.py --subtitles OUT/subtitles.json --visual OUT/captions.json --out OUT/merged.json
# 3. build (Step 2 with --merged)
python3 scripts/build_knowledge.py --subtitles OUT/subtitles.json --merged OUT/merged.json --out-dir OUT --format all
```

The fusion loop has three cross-path pieces: (1) `hotwords_from_ocr.py`
extracts on-screen terms (recurring CJK n-grams, slide titles, Latin
acronyms — heuristic, no LLM) and reports which ones the ASR never heard,
the likely mis-heard jargon; (2) `merge_visual.py` attaches visuals by
timestamp, then conservatively re-binds a narration line to an ADJACENT slide
when its word overlap with the timestamp-attached slide is near zero and the
neighbour matches clearly better (speaker lag / timestamp drift), flagging
uncertain cases as `weak` instead of guessing — except on SETTLED slides
(≥60 s on screen), where verbal elaboration is the norm and no flag is raised
(`--no-semantic` restores pure timestamp matching); (3) `build_knowledge.py
--merged` feeds the LLM interleaved audio+visual text (chunk boundaries never
cut a slide table; prompt-echo lines are fingerprint-filtered from every list
field; field caps scale with video duration, 3× at 3 h) with ⚠️ markers on
swap/weak notes, and re-ranks map-reduced list fields in one global pass with
the summary as context. Chunk-level LLM responses are cached in
`build_cache.json`, so an interrupted 3-hour build resumes instead of
restarting. In batches, `batch_run.py` accumulates OCR terms into
`course_hotwords.txt` so later videos transcribe better, and `--asr-verify`
re-transcribes a video whose OCR-term coverage fell below 50 %.
The default text model follows the same hardware profile: `openbmb/minicpm5-2b`
(2.5B dense, 1.6 GB) on `low`/`low-mac`/`mid`, and the Path 1 VLM pull itself on
`high`+ (qwen3.5/qwen3.8 are unified vision+text, so nothing is downloaded
twice). Override with `--model`, or `V2K_TEXT_MODEL=` to pin one for every run.
Quality measurement: `scripts/recall_check.py --run-dir <dir> --draft` emits a
golden must-have list (terms / both-channel numbers / timeline coverage) that a
human prunes in minutes; `--golden` then scores produced artifacts against it
(baselines in `tests/golden/BASELINE.md`) — "missing content" becomes a number
you can track across parameter changes.

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
| 2.3 Knowledge cards | `cards.csv` | `question,answer,tags,timestamp,source` — **deduplicated**: a long lecture is summarised map/reduce over chunks and the model re-asks the same question in each, so every repeat is dropped (measured: 145 of 1608 cards, worst case 117 rows → 11) |
| 2.4 Illustrated notes (图文笔记) | `notes.md` + `notes.html` | key frames × narration, see below |
| 2.4b Notes office/print | `notes.docx` / `notes.pdf` | `build_notes.py --docx --pdf` (frames embedded) |
| 2.4c **Distilled notes (纯享版)** | `notes-distilled.md` / `.docx` / `.pdf` | same nodes and frames, no quoted narration — 42% smaller, and free (rendered from output already computed) |

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
  --out-dir OUT --describe-frames
```

**Node density is automatic.** `--max-frames 0` (the default) sizes the note to
~1 node per 45 seconds, clamped to 12–60: a 12-minute clip gets 16 nodes, a
41-minute lecture 55, a 3-hour one 60. The old `1-per-4min` budget gave a
37-minute lecture only 9 nodes, and the `1-per-90s` that followed still floored
at 8 — measured over a 306-video / 61.5h library the mean clip is 12.1 min, so
the floor bound most of the library to 8 nodes no matter how long the video
ran. The floor was what had to move, not the ceiling. Pass an explicit
`--max-frames N` to pin it. Selection is cluster-stratified: every change burst
keeps its settled final frame, so an animation burst no longer starves isolated
key slides of their own node.

Per key frame: LLM section title → frame image → optional VLM 画面 description
(`--describe-frames`) → 1-2 sentence note condensed from the narration window →
verbatim 原声 excerpt.

The 画面 prompt asks for **teaching information, not people**: note pitches, chord
symbols, time signatures, annotations and on-screen text — and explicitly refuse
to describe the performer ("女士弹琴", "手势讲解" are not what a revision note
needs). A frame with no readable teaching content gets its topic named instead
("钢琴演奏画面，无字幕"). This one prompt change is the difference between a note
that helps you study and one that narrates the video.

**Two views, one pass.** Every run writes a twin pair from the same sections —
no extra model call:

| File | Contains | For |
|---|---|---|
| `notes.md` | frames + notes + quoted 原声 | cross-checking the teacher's exact wording |
| `notes-distilled.md` | same nodes, same frames, **no quoted narration** | revising — measured over 295 nodes, the blockquotes were 53% of the characters |

`--docx` / `--pdf` export both variants (`notes.docx` + `notes-distilled.docx`,
and likewise for PDF). `notes.html` embeds frames as base64 into a single
shareable file — that one is the full view only, no distilled HTML twin.
Degrades to frames + raw excerpts when no model is reachable.

**Give the VLM a vision-capable model explicitly.** `--describe-frames` without
`--vlm-model` falls back to the *text* model, which cannot see; the run warns and
drops the descriptions instead of failing once per frame:

```bash
python3 scripts/build_notes.py ... --describe-frames \
  --model openbmb/minicpm5-2b --vlm-model openbmb/minicpm-v4.6:latest
```

See `references/outputs.md` for schema, single-format runs, and degraded mode.

## Custom Templates (2.1)

Default template: `assets/default-template.md`. Override with `--template <file>`:

```bash
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o \
  --template ./my-lecture-template.md --format knowledge
```

Templates are plain Markdown using `{{placeholders}}` (`{{title}}`, `{{summary}}`,
`{{bullets}}` (音画合并要点速览), `{{timeline}}`, `{{key_points}}`, `{{qa}}`, `{{glossary}}`,
`{{source}}`, `{{duration}}`, `{{date}}`, `{{meta}}`). Only placeholders you include are filled;
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

Mandarin lecture, full pipeline, everything local (swap in the Paraformer line
below if `hardware_profile.py --recommend` says so — it needs its own venv):

```bash
source .venv/bin/activate
RUN=runs/$(date +%Y%m%d-%HMMSS)-lecture
mkdir -p "$RUN"

# Step 1 — subtitles
python3 scripts/asr_caption.py \
  --video ~/Movies/lecture.mp4 \
  --out-dir "$RUN" --language zh
#   ...or, for Mandarin:
# .venv-funasr/bin/python scripts/asr_funasr.py \
#   --video ~/Movies/lecture.mp4 --out-dir "$RUN" --language zh

# Step 1b — repair domain homophones (run before Step 2)
python3 scripts/fix_homophones.py --subtitles "$RUN/subtitles.json" --dry-run

# Step 1c — key frames for the illustrated note
python3 scripts/extract_frames.py \
  --video ~/Movies/lecture.mp4 --out-dir "$RUN/frames" --mode dedup

# Step 2 — knowledge doc / HTML / CSV
python3 scripts/build_knowledge.py \
  --subtitles "$RUN/subtitles.json" \
  --out-dir "$RUN" --format all

# 2.3 — Anki deck
python3 scripts/gen_apkg.py \
  --csv "$RUN/cards.csv" --out "$RUN/cards.apkg"

# 2.4 — illustrated notes: notes.md + notes-distilled.md (+ notes.html)
python3 scripts/build_notes.py \
  --subtitles "$RUN/subtitles.json" \
  --frames "$RUN/frames/frames.json" \
  --out-dir "$RUN" --describe-frames \
  --model openbmb/minicpm5-2b --vlm-model openbmb/minicpm-v4.6:latest

# Optional local record (stays on your machine; runs/ is gitignored)
cat > "$RUN/manifest.json" <<EOF
{"video":"~/Movies/lecture.mp4","path":"2","model":"small","outputs":["subtitles.srt","knowledge.md","knowledge.html","cards.csv","cards.apkg","notes.md","notes-distilled.md","notes.html"]}
EOF
```

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/setup_models.sh` | Idempotent model/venv setup (profile-aware) |
| `scripts/hardware_profile.py` | Detect machine → recommend ASR/VLM/backend profile |
| `scripts/extract_frames.py` | Frame sampling: `--mode interval` (uniform fps) or `--mode dedup` (dense sample + dHash dedup with **settle-frame** selection, blank-frame gate, **tail-frame emission** — the final state of each similar-run is kept so sub-threshold micro-edits before a slide change are not lost, `--tail-eps` to tune; optional `--hash-mode dual` dHash+aHash, and cluster-stratified `--max-frames` budget; at `--hash-size 16` use `--dedup-hamming 20`) → `frames.json` |
| `scripts/mm_caption.py` | Path 1: VLM captioning → `captions.{srt,json}`; `--mode dedup --prompt-ocr` for slide tables/formulas, with an OCR text-change gate that drops frames whose text is ≥90% similar to the last kept one |
| `scripts/asr_caption.py` | Path 2: faster-whisper (default, local) → `subtitles.{srt,vtt,json}`; `--hotwords` biases transcription via initial_prompt. Also hosts the opt-in cloud backends `--backend openai-api` / `mimo-asr` (`--api-base` / `--api-model` / `--api-key-env`) — **these upload your audio to a third party** |
| `scripts/asr_funasr.py` | Path 2 alternative: FunASR Paraformer-large → the same `subtitles.{srt,vtt,json}`; better on Mandarin homophones and jargon (measured 44% → 94% on 背谱/视谱), runs on CPU, `--jobs` transcribes a list in one model load. Needs its own venv |
| `scripts/fix_homophones.py` | Post-ASR repair pass against a domain glossary — tone-aware pinyin matching, not edit distance; rewrites `.srt`/`.vtt` to match and logs to `homophone_fixes.json`. `--dry-run` / `--block WORD` / `--glossary @file` / `--no-default-block`. Run before Step 2 |
| `scripts/merge_visual.py` | Path 3: fuse ASR `subtitles.json` × VLM `captions.json` by timestamp → `merged.json` (re-attach fallback keeps long-lived slides attached; semantic alignment check swaps clearly-mismatched attachments to adjacent slides and flags weak ones) |
| `scripts/hotwords_from_ocr.py` | Path 3 loop: extract salient terms from VLM OCR → `ocr_hotwords.txt` (+ accumulating `course_hotwords.txt`), check ASR coverage of those terms, exit 3 under `--fail-under` to trigger re-transcription |
| `scripts/build_knowledge.py` | Step 2: subtitles → knowledge.md / .html / cards.csv; `--merged` for dual-path fusion with `{{visual_timeline}}` section, ⚠️ swap/weak markers, map-reduce + global re-rank; `--format docx` / `--format pdf` for office/print; `--api-base` / `--api-model` for an OpenAI-compatible endpoint instead of local Ollama. Cards are **deduplicated on the normalized question**, not the raw string — models rarely repeat a card verbatim ("白键之间是全音" vs "白键和相邻白键间隔是一个全音" are the same card) |
| `scripts/build_notes.py` | Step 2.4: illustrated notes (图文笔记): key frames × narration → `notes.md` (with quoted 原声) **and** `notes-distilled.md` (same nodes, no quotes) from one pass, plus a self-contained `notes.html`; node budget auto-sizes to ~1 per 45 s (clamped 12–60), `--describe-frames` runs a teaching-focused VLM line, `--docx` / `--pdf` export both variants |
| `scripts/md_export.py` | Shared Markdown → DOCX/PDF exporter (python-docx + fpdf2, CJK font auto-detect, ffmpeg-JPEG normalize); standalone CLI for any pipeline .md |
| `scripts/gen_apkg.py` | Step 2.3: cards.csv → Anki `.apkg` |
| `scripts/batch_run.py` | Batch a whole course library: ASR + Ollama two-thread pipeline, resumable markers, priority `--order`, summary CSV, `--asr-backend funasr` |
| `scripts/recall_check.py` | Measure "缺内容" in a finished run: checks the produced artifacts against sources/coverage instead of trusting that a file exists; `--draft` emits a golden-set draft for the next run to compare against |

## References (load as needed)

- `references/hardware-profiles.md` — profile table, sizing rationale, tuning
- `references/path1-multimodal.md` — VLM details, API format, sampling strategy
- `references/path2-asr.md` — model sizing, device/compute, language options
- `references/path3-fusion.md` — dual-path fusion (ASR × VLM OCR), dedup sampling, `--merged` build
- `references/templates.md` — placeholder spec + custom template examples
- `references/outputs.md` — HTML/CSV/APKG schemas, single-format runs, degraded mode

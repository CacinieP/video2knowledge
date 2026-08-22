# Path 3 — Dual-Path Fusion (ASR × VLM)

Path 3 combines Path 2 (ASR) and Path 1 (VLM OCR) for **slide/PPT/screencast
videos** where the core content is on screen — tables, formulas, examples,
on-screen text that ASR simply cannot capture. It runs both ingestions and fuses
them by timestamp so the knowledge doc contains the *narration grounded in the
slides*.

## When to choose Path 3

- The video is a slide deck / PPT / screencast (the two together cover most
  online-course and tutorial content).
- You need on-screen tables, formulas, charts, and worked examples — not just
  what the speaker says.
- You want to cross-validate the speech against what is actually shown.

Path 3 is strictly additive over Path 2: it reuses the ASR subtitles and layers
VLM OCR on top. If the video has no slides, skip it (Path 2 alone is enough).

## Pipeline

```
            ┌─ Path 2: video → ASR → subtitles.json ─────────┐
video ──────┤                                              ├─→ merge_visual.py ─→ build_knowledge.py --merged
            └─ Path 1-OCR: video → dedup frames → VLM OCR → captions.json ┘
```

## Step 1a — ASR (same as Path 2)

```bash
source .venv/bin/activate
python3 scripts/asr_caption.py --video slides.mp4 --out-dir run --language zh
```

## Step 1b — Frame extraction with perceptual dedup

The general-purpose sampling mode is `dedup`: it samples densely (default 1 fps),
computes a **dHash** per frame (default 64-bit; `--hash-size 16` → 256-bit for
small-but-important changes on dense slides; `--hash-mode dual` appends an aHash
for flat/gradient content dHash cannot see), and keeps only frames whose Hamming
distance from the last kept frame exceeds a threshold. A detected change is kept
at the first **settled** frame — the one whose successor is nearly identical
(`--settle-window`, default 2s) and which is not blank/black (mean-luma gate) —
so fades and slide animations are captured in their final stable state. Over
`--max-frames` (default 120), the budget is split by **change bursts**
(cluster-stratified): every burst keeps its settled final frame and the rest is
distributed proportionally, so an animation burst cannot starve isolated key
slides. This drops the hundreds of visually-identical static-slide frames (a
25-min PPT video yields ~30 key frames, not ~750), bounding VLM cost — and it
needs **no scene-detection threshold tuning** (ffmpeg's `select=gt(scene,T)`
fails on fade animations and varies per video). Timestamps stay accurate because
fps sampling is uniform.

```bash
python3 scripts/extract_frames.py --video slides.mp4 --out-dir run/frames --mode dedup
# knobs: --dedup-fps 1.0  --dedup-hamming 10  --max-frames 120
#        --hash-size 8  --hash-mode dhash|dual  --settle-window 2.0
# dual mode: scale --dedup-hamming ~2x, e.g. 20 (2*n^2 bits)
```

`dedup` uses only numpy (hand-written dHash/aHash over an ffmpeg PGM pipe) — no
Pillow or other image dependency.

## Step 1c — VLM OCR captioning

```bash
python3 scripts/mm_caption.py \
  --video slides.mp4 --out-dir run \
  --mode dedup --prompt-ocr
```

`--prompt-ocr` switches the VLM (default `openbmb/minicpm-v4.6:latest`) to a
full-fidelity transcription prompt that preserves Markdown tables and newlines,
and raises `num_predict` so whole tables are not truncated. The prompt is written
to avoid the small-VLM failure mode of echoing the instruction menu back as
content. An **OCR text-change gate** then drops any frame whose OCR text is ≥90%
similar to the last kept frame's (visual change without knowledge change —
cursor blinks, partial redraws, animation noise), so every emitted caption
carries new on-screen text and wasted VLM calls disappear (`--no-ocr-dedup` to
disable, `--ocr-dedup-threshold` to tune). Output schema is the same
`{start,end,text}` list as Path 1 captions.

## Step 2 — Fuse by timestamp

```bash
python3 scripts/merge_visual.py \
  --subtitles run/subtitles.json \
  --visual run/captions.json \
  --out run/merged.json
```

Each ASR segment is annotated with the on-screen content at its moment. A visual
frame attaches to **at most one consecutive run** of ASR segments so a large
table is not repeated on every narration line; when no unused frame covers a
moment, the used frame that covers it is re-attached (a slide that stays on
screen keeps its table instead of going visually empty). Output: `merged.json`
with `{start,end,text,visual}` per segment plus a `visual_blocks` reference list.

## Step 3 — Build the fused knowledge doc

```bash
python3 scripts/build_knowledge.py \
  --subtitles run/subtitles.json \
  --merged run/merged.json \
  --out-dir run --format all
```

In merged mode the LLM is fed **interleaved audio + visual** text
(`[mm:ss] 🎙️narration / 🖼️slide-table`), and a `{{visual_timeline}}` section
(tables/formulas/definitions per slide, deduped) is added to the knowledge doc.
The raw-text cap is auto-raised (default 8000 → 20000) and `build_analysis` uses
**map-reduce** over 4500-char chunks so small/mid models do not degrade on long
inputs.

## Text model sizing

The default text model is **`qwen3.5:4b`** (unified vision+text — on `high`
machines the same pull serves Path 1). On an 8 GB machine a 1B model degrades
badly on the ~20k-char fused context (repeated output, dropped items); a 3–4B
reads the content and reasons about it. `qwen3.5:4b` loads in ~3.4 GB, leaving
headroom on 16 GB; use `qwen3.5:2b` (~2.7 GB) for 8 GB machines. For low-RAM
runs, override with `--model openbmb/minicpm5:Q4_K_M` (faster but
shallower). The map-reduce chunking keeps each call small enough that even 1B
stays usable on short videos.

## End-to-end example (everything local)

```bash
source .venv/bin/activate
RUN=runs/$(date +%Y%m%d-%HMMSS)-slides
mkdir -p "$RUN"

# 1a ASR
python3 scripts/asr_caption.py --video ~/Movies/slides.mp4 --out-dir "$RUN" --language zh
# 1b+1c dedup frames + VLM OCR
python3 scripts/mm_caption.py --video ~/Movies/slides.mp4 --out-dir "$RUN" --mode dedup --prompt-ocr
# 2 fuse
python3 scripts/merge_visual.py --subtitles "$RUN/subtitles.json" --visual "$RUN/captions.json" --out "$RUN/merged.json"
# 3 knowledge doc / HTML / CSV
python3 scripts/build_knowledge.py --subtitles "$RUN/subtitles.json" --merged "$RUN/merged.json" --out-dir "$RUN" --format all
# Anki deck
python3 scripts/gen_apkg.py --csv "$RUN/cards.csv" --out "$RUN/cards.apkg" --deck "幻灯片知识卡"
```

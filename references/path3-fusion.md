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
                                  │
                                  └─→ hotwords_from_ocr.py ─→ ocr_hotwords.txt / course_hotwords.txt
                                       (OCR 术语回灌 ASR：跨路径反馈回路)
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
`--max-frames` (default 360, a 3h lecture's worth of changes), the budget is split by **change bursts**
(cluster-stratified): every burst keeps its settled final frame and the rest is
distributed proportionally, so an animation burst cannot starve isolated key
slides. This drops the hundreds of visually-identical static-slide frames (a
25-min PPT video yields ~30 key frames, not ~750), bounding VLM cost — and it
needs **no scene-detection threshold tuning** (ffmpeg's `select=gt(scene,T)`
fails on fade animations and varies per video). Timestamps stay accurate because
fps sampling is uniform.

```bash
python3 scripts/extract_frames.py --video slides.mp4 --out-dir run/frames --mode dedup
# knobs: --dedup-fps 1.0  --dedup-hamming 10  --max-frames 360
#        --hash-size 8  --hash-mode dhash|dual  --settle-window 2.0
# dual mode: scale --dedup-hamming ~2x, e.g. 20 (2*n^2 bits)
```

`dedup` uses only numpy (hand-written dHash/aHash over an ffmpeg PGM pipe, parsed by a strict streaming reader that reads each frame's declared pixel count and never scans pixel data for the P5 magic) — no
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

## Step 2 — Fuse by timestamp (with semantic alignment check)

```bash
python3 scripts/merge_visual.py \
  --subtitles run/subtitles.json \
  --visual run/captions.json \
  --out run/merged.json
```

Each ASR segment is first annotated by **timestamp**: the on-screen content
whose window `[start-2, end]` covers the segment midpoint. A visual frame
attaches to **at most one consecutive run** of ASR segments so a large table
is not repeated on every narration line; when no unused frame covers a moment,
the used frame that covers it is re-attached (a slide that stays on screen
keeps its table instead of going visually empty).

Then a conservative **semantic alignment check** (default on, `--no-semantic`
to disable) catches speaker-lag: the narration discusses slide N while slide
N+1 is already on screen (or ASR timestamps drifted). After the timestamp
pick, word-level overlap between narration and slide text is scored (CJK
bigrams + Latin words, function words dropped). When the attached slide
shares (almost) nothing with the narration (`--weak-overlap`, default 0.06)
while an **adjacent** slide (within `--swap-window`, default 90 s) matches
clearly better (`--swap-margin`, default 0.15), the narration is re-bound to
that neighbour (`"match": "semantic-swap"` + a note). When nothing matches at
all, the attachment is kept but flagged `"weak"` — the speaker may just be
elaborating verbally, so we flag instead of guessing. Timestamps stay the
authority; the semantic pass only fixes obvious mismatches.

Output: `merged.json` with `{start,end,text,visual,match,note}` per segment
plus a `visual_blocks` reference list and `semantic_swaps` / `weak_attribution`
counts (surfaced by `build_knowledge.py` and the batch logs).

## Step 2b — OCR terms → ASR hotwords (cross-path feedback)

The pipeline runs ASR *before* OCR, so the on-screen vocabulary cannot bias
the initial transcription — exactly where jargon errors happen. Close the loop
after OCR:

```bash
python3 scripts/hotwords_from_ocr.py \
  --captions run/captions.json \
  --subtitles run/subtitles.json \
  --manual "术语一, 术语二" \
  --course-vocab runs/batch/course_hotwords.txt \
  --out run/ocr_hotwords.txt
```

- **Extract** salient on-screen terms from the OCR text — recurring CJK
  n-grams, slide-title/table-header phrases, Latin acronyms (IFRS, FVOCI) —
  pure heuristic, no LLM call.
- **Check coverage**: which extracted terms never appear in the ASR text
  (those are the likely mis-heard jargon). `--fail-under 0.5` exits 3 when
  coverage drops below 50 % — the cue to re-run
  `asr_caption.py --hotwords @run/ocr_hotwords.txt`.
- **Accumulate**: with `--course-vocab`, terms are appended to a shared
  vocabulary; in a batch (`batch_run.py`) stage A merges it into the hotwords
  of later videos, so the loop pays off without re-running anything.

## Step 3 — Build the fused knowledge doc

```bash
python3 scripts/build_knowledge.py \
  --subtitles run/subtitles.json \
  --merged run/merged.json \
  --out-dir run --format all
```

In merged mode the LLM is fed **interleaved audio + visual** text
(`[mm:ss] 🎙️narration / 🖼️slide-table`); swap/weak notes from the semantic
check surface as `⚠️` markers so the model knows an attribution is corrected
or uncertain. A `{{visual_timeline}}` section (tables/formulas/definitions per
slide, deduped) is added to the knowledge doc. The raw-text cap is auto-raised
(default 8000 → 20000) and `build_analysis` uses **map-reduce** over 4500-char
chunks so small/mid models do not degrade on long inputs; each list field
(timeline/key_points/qa/glossary) then gets one **global re-rank pass** with
the overall summary as context, restoring cross-chunk order and merging
chunk-boundary duplicates. QA items are deduped and capped as Q/A *pairs*
(never orphaning an answer line).

## Text model sizing

Step 2's text model is picked from the same hardware profile, in a field
separate from the VLM because this stage never sees a frame. The default is
**`openbmb/minicpm5-2b`** (2.5B dense, 1.6 GB Q4_K_M, 131K context) on
`low`/`low-mac`/`mid`, and the VLM pull itself on `high`+ — qwen3.5/qwen3.8
are unified vision+text, so the top tiers download one model, not two. On an
8 GB machine a 1B model degrades badly on the ~20k-char fused context
(repeated output, dropped items); the 2B reads the content and reasons about
it, and it averaged 53.9 over OpenBMB's 34-benchmark set against Qwen3.5-4B's
51.1 at roughly half the RAM. Below the `tiny` tier the table drops to
`openbmb/minicpm5:Q4_K_M` (688 MB, shallower). The map-reduce chunking keeps
each call small enough that even the 688 MB model stays usable on short
videos. Pin a different one with `--model` or `V2K_TEXT_MODEL=`.

## End-to-end example (everything local)

```bash
source .venv/bin/activate
RUN=runs/$(date +%Y%m%d-%HMMSS)-slides
mkdir -p "$RUN"

# 1a ASR
python3 scripts/asr_caption.py --video ~/Movies/slides.mp4 --out-dir "$RUN" --language zh
# 1b+1c dedup frames + VLM OCR
python3 scripts/mm_caption.py --video ~/Movies/slides.mp4 --out-dir "$RUN" --mode dedup --prompt-ocr
# 1d OCR terms -> hotwords (coverage check; re-run ASR with @ocr_hotwords.txt
#    when coverage is poor)
python3 scripts/hotwords_from_ocr.py --captions "$RUN/captions.json" \
    --subtitles "$RUN/subtitles.json" --out "$RUN/ocr_hotwords.txt"
# 2 fuse (timestamp + semantic alignment)
python3 scripts/merge_visual.py --subtitles "$RUN/subtitles.json" --visual "$RUN/captions.json" --out "$RUN/merged.json"
# 3 knowledge doc / HTML / CSV
python3 scripts/build_knowledge.py --subtitles "$RUN/subtitles.json" --merged "$RUN/merged.json" --out-dir "$RUN" --format all
# Anki deck
python3 scripts/gen_apkg.py --csv "$RUN/cards.csv" --out "$RUN/cards.apkg" --deck "幻灯片知识卡"
```

For whole course libraries, `batch_run.py` wires this chain (including the
OCR-hotwords loop and optional `--asr-verify` re-transcription) with resumable
per-video run dirs — see its `--help`.

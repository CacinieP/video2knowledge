# Route 3: speech plus meaningful visual evidence

Use for narrated slides, whiteboards and screen demonstrations when the requested
notes need visible text, tables, formulas or operations. Use Route 2 for speech
alone; use Route 1 without usable speech. Resolve paths/interpreters as in
[runtime.md](runtime.md), and choose tested installed models from the live profile.

## Baseline recipe

This is one Bash invocation. Rebind variables if executing steps in fresh shells.
The example uses Whisper for Mandarin; substitute the separate FunASR interpreter
for the ASR step when selected. For English use `--language en` / `--lang en`,
and omit Mandarin music-term correction. Also omit `--describe-frames` for
English-only notes: the current local frame-description prompt is Chinese,
even when `--lang en` is set. Keep original frames and OCR for formula/table
verification; English headings/notes do not imply English VLM captions.

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
VIDEO="/absolute/path/to/slides.mp4"
OUT="/absolute/path/to/runs/slides"
VLM="openbmb/minicpm-v4.6:latest"  # example; use installed tested vision model
TEXT="openbmb/minicpm5-2b"          # example; use installed tested text model
"$PY" "$SKILL_ROOT/scripts/asr_caption.py" --video "$VIDEO" --out-dir "$OUT" --language zh
"$PY" "$SKILL_ROOT/scripts/mm_caption.py" --video "$VIDEO" --out-dir "$OUT" --model "$VLM" --mode dedup --prompt-ocr
"$PY" "$SKILL_ROOT/scripts/hotwords_from_ocr.py" --captions "$OUT/captions.json" --subtitles "$OUT/subtitles.json" --out "$OUT/ocr_hotwords.txt"
# If coverage is poor and terms are relevant, rerun ASR with --hotwords @... BEFORE the steps below.
# Apply only with a relevant Mandarin domain glossary; omit for unrelated domains.
"$PY" "$SKILL_ROOT/scripts/fix_homophones.py" --subtitles "$OUT/subtitles.json"
"$PY" "$SKILL_ROOT/scripts/merge_visual.py" --subtitles "$OUT/subtitles.json" --visual "$OUT/captions.json" --out "$OUT/merged.json"
"$PY" "$SKILL_ROOT/scripts/build_knowledge.py" --subtitles "$OUT/subtitles.json" --merged "$OUT/merged.json" --out-dir "$OUT" --model "$TEXT" --format html --lang zh
"$PY" "$SKILL_ROOT/scripts/build_notes.py" --subtitles "$OUT/subtitles.json" --frames "$OUT/frames/frames.json" --out-dir "$OUT" --model "$TEXT" --describe-frames --vlm-model "$VLM" --lang zh
```

Deliver knowledge Markdown/HTML and illustrated notes plus frame files. Add
CSV/Anki, DOCX or PDF only as requested; see [outputs.md](outputs.md). The notes
builder uses frames and narration, not `merged.json`; it describes selected frames
separately. Do not claim that every merged OCR table appears verbatim in notes.

## Hotword feedback and temporal alignment

`hotwords_from_ocr.py --fail-under 0.5` uses exit 3 for low OCR-term coverage.
This is an advisory signal, not proof ASR is wrong: a slide may contain unspoken
terms. Inspect the terms before a second ASR pass. After re-transcribing, reapply
relevant corrections, rerun merge and rebuild dependent artifacts.

The merger attaches frames by time, conservatively swaps clearly mismatched
adjacent slides, and marks weak/uncertain attribution. `--no-semantic` restores
pure timestamp attachment. Keep uncertainty visible in the delivered notes.

Fused knowledge input defaults to a 400,000-character cap; plain ASR defaults
to no character cap. Map-reduce uses 9,000-character chunks and activates for
long inputs; these are implementation limits, not guarantees of complete recall.
Use `recall_check.py` against a reviewed golden set only as a diagnostic.

## Libraries and cloud boundaries

`batch_run.py` wires the Mandarin fusion chain with state-aware retries. It is
not a general silent-video or multilingual batch driver. Read
[batch-resume.md](batch-resume.md) before using it.

The OCR stage above still calls Ollama. An optional cloud text/notes endpoint
can receive fused text and selected notes images; it does not replace
`mm_caption.py`'s local OCR API. Read [cloud.md](cloud.md).

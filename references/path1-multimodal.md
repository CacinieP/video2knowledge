# Route 1: visual-only notes and on-screen OCR

Use when there is no usable speech or the request is explicitly visual-only.
An audio track containing music/filler does not make the source a lecture.
Read [runtime.md](runtime.md) first. Commands use absolute paths and the main
venv interpreter, never an activated shell assumed to persist.

## Illustrated notes for a silent video

This recipe deliberately supplies an empty speech transcript: the notes builder
then labels the result pure-visual and does not invent narrator quotations.
Set `$VLM` from the live profile or an already-tested installed vision model.

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
VIDEO="/absolute/path/to/silent-demo.mp4"
OUT="/absolute/path/to/runs/silent-demo"
VLM="openbmb/minicpm-v4.7:latest"  # profile pick on every tier; use a tested installed VLM
mkdir -p "$OUT"
"$PY" "$SKILL_ROOT/scripts/extract_frames.py" --video "$VIDEO" --out-dir "$OUT/frames" --mode dedup
"$PY" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"segments": []}), encoding="utf-8")' "$OUT/visual-only.json"
"$PY" "$SKILL_ROOT/scripts/build_notes.py" --subtitles "$OUT/visual-only.json" --frames "$OUT/frames/frames.json" --out-dir "$OUT" --describe-frames --vlm-model "$VLM" --lang zh
```

Deliver `notes.md`, `notes-distilled.md`, `notes.html` and their frame folder.
The two Markdown views may be identical because there is no spoken quotation.
Check that descriptions are present; if the VLM fails, label frame-only output
partial rather than claiming visual content was understood. Without a model,
frames can still be extracted, but that is not a completed semantic note.

The local frame-description prompt is currently Chinese; `--lang en` does not
translate it. For an English-only silent task, report this limitation instead of
promising English semantic visual notes or silently modifying the prompt.

## Requested on-screen transcription

Use `mm_caption.py` for timestamped visual descriptions, or `--prompt-ocr` for
visible text/tables/formulas. With the same bindings as above (in the SAME shell):

```bash
"$PY" "$SKILL_ROOT/scripts/mm_caption.py" --video "$VIDEO" --out-dir "$OUT" --model "$VLM" --mode dedup --prompt-ocr
```

Outputs: `captions.json` (list of `{start,end,text}`), `captions.srt`, `frames/`.
These are visual OCR/captions, not spoken subtitles. Do not present them as
verbatim teacher speech or run the Mandarin homophone corrector over OCR text.
`mm_caption.py` defaults to `openbmb/minicpm-v4.7:latest` — the unified
vision+text model every profile now picks — but the installer profile does NOT
automatically change this CLI default, so pass `--model` explicitly whenever you
override the profile (`VLM_MODEL=`, a legacy VLM, or a different tag).

For a requested knowledge document, `build_knowledge.py --subtitles captions.json`
can consume visual text. Very short visual text may hit its speech-density guard;
report this limitation and deliver OCR + visual notes rather than disabling guards
or fabricating a summary/cards. Do not give captions to the notes builder as if
there were spoken narration; use the empty-speech recipe above.

## Sampling and limits

- Use `--mode dedup` for slides/boards/screens; the frame cap is 360 by default.
- Use `--interval 2.0` for uniformly sampled general video.
- `--hash-size 16 --dedup-hamming 20` can preserve smaller slide changes.
- `--hash-mode dual` adds flat/gradient sensitivity; adjust thresholds accordingly.
- Sampling fewer frames reduces requests, not the resident model weight memory.

The captions endpoint is the Ollama `/api/generate` API. `--api-base` sends each
frame to an OpenAI-compatible `/chat/completions` vision endpoint instead (no
Ollama, no local GPU); `--api-model` defaults to the hosted twin
`MiniCPM-V-4.7-1B`, so prompts are unchanged when switching local↔cloud, and the
flags cascade from `V2K_VLM_API_BASE` / `V2K_VLM_API_MODEL` / `V2K_VLM_API_KEY_ENV`
(or the `V2K_LLM_*` equivalents) so a batch can go cloud by environment alone.
That route uploads every selected frame. `build_notes.py` has its own cloud VLM
route; see [cloud.md](cloud.md).

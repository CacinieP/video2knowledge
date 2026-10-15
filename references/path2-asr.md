# Route 2: spoken transcript and knowledge notes

Use for usable speech when screen content is not needed. For narrated slides,
boards or demonstrations whose visuals matter, read [path3-fusion.md](path3-fusion.md).
Read [runtime.md](runtime.md) for absolute interpreters and install constraints.

## Local baseline

The following is one Bash invocation. For English, use `--language en` here and
`--lang en` on the knowledge/notes builders. The latter currently supports zh/en.

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
VIDEO="/absolute/path/to/lecture.mp4"
OUT="/absolute/path/to/runs/lecture"
"$PY" "$SKILL_ROOT/scripts/asr_caption.py" --video "$VIDEO" --out-dir "$OUT" --language zh
# Apply only when this Mandarin input needs the domain glossary; this is NOT dry-run.
"$PY" "$SKILL_ROOT/scripts/fix_homophones.py" --subtitles "$OUT/subtitles.json"
"$PY" "$SKILL_ROOT/scripts/build_knowledge.py" --subtitles "$OUT/subtitles.json" --out-dir "$OUT" --format html --lang zh
```

`--format html` writes both `knowledge.md` and `knowledge.html`, avoiding an
unrequested PDF/font dependency. ASR also writes `subtitles.srt`, `.vtt`, `.json`;
its WAV intermediate is normally removed unless `--keep-wav` is passed.
Use `--hotwords "term1,term2"` or `--hotwords @/absolute/path/terms.txt` for jargon.
If previewing repairs with `--dry-run`, inspect and then rerun without that flag
before producing repaired downstream outputs. Skip the built-in Mandarin music
correction for unrelated languages/domains, or supply a relevant glossary/blocks.

## Mandarin Paraformer alternative

Prefer an already-working installed backend. `hardware_profile.py --recommend`
can suggest Paraformer; its measured advantage comes from a particular Mandarin
course sample, not a guarantee for arbitrary audio. Setup is opt-in:

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
cd "$SKILL_ROOT"
bash "$SKILL_ROOT/scripts/setup_models.sh" --with-funasr
ASR_PY="$SKILL_ROOT/.venv-funasr/bin/python"
[ -x "$ASR_PY" ] || ASR_PY="$SKILL_ROOT/.venv-funasr/Scripts/python.exe"
"$ASR_PY" "$SKILL_ROOT/scripts/asr_funasr.py" --video "/absolute/path/to/lecture.mp4" --out-dir "/absolute/path/to/runs/lecture" --language zh --hotwords "背谱,视谱,音阶"
```

Use the main venv for correction/build/export afterwards. FunASR uses its own
venv and Mandarin/CPU engine; do not pass Whisper `--model`, `--device` or
`--compute-type` flags to it. For library jobs, see [batch-resume.md](batch-resume.md).

## Cards or additional formats

Only when requested, generate `cards.csv` with `build_knowledge.py --format csv`
(or choose `--format all` for all knowledge formats). Then package with a stable
`gen_apkg.py --source-id`, as described in [outputs.md](outputs.md). Illustrated
notes require a separate frame extraction and `build_notes.py` invocation.

Empty/low-information speech is not a source for invented knowledge: inspect the
transcript and return no-speech or use Route 1 if the video contains useful visuals.
Cloud ASR/LLM are opt-in; read [cloud.md](cloud.md) for parameters and uploads.

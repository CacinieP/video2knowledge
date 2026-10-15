# Batch scope, checkpoints and retries

## Use the correct entry point

`batch_run.py` handles narrated Mandarin course libraries with the full fusion
chain. It hardcodes `--language zh`, defaults to CPA hotwords/deck names, and
requires non-empty ASR output. Override the domain/deck and backend explicitly.
For English or silent libraries use the corresponding per-video recipes; do not
claim this batch driver auto-detects languages or skips unusable ASR into Route 1.

Inspect a plan first (dry-run creates the batch log/summary but does not alter
video checkpoints):

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
"$PY" "$SKILL_ROOT/scripts/batch_run.py" --root "/absolute/path/to/course-library" --out-root "/absolute/path/to/runs/course" --deck "My course" --hotwords "" --dry-run
```

Run exactly the same command without `--dry-run`. Use `--limit 1` for an initial
smoke run; `--order` prioritizes top-level directory substrings. Set
`--text-model` / `--vlm-model` to already-tested models. For FunASR add
`--asr-backend funasr --asr-python "/absolute/path/.venv-funasr/bin/python"`
(or the Windows interpreter). There is no general `--language` or cloud-ASR flag
on this batch driver. Corrector built-ins include Mandarin music terms: for an
unrelated course choose relevant glossary/blocks or `--no-homophone-fix`.

## Read state before changing it

- `.asr_done`: ASR stage checkpoint; a matching retry can reuse it.
- `.done`: whole-chain marker; trusted only with a matching `run_manifest.json`
  and the driver's minimum artifact checks, not just marker existence.
- `.failed`: last stage error; the driver clears it for selected retries.
- `stageA.log` / `stageB.log`, `batch.log`, `summary.csv`: inspect before retrying.
- `run_manifest.json`: hashed ASR/full fingerprints. It is NOT a readable record
  of the input, model name and all arguments.
- `build_cache.json`: inner LLM responses; retained on unchanged failed retries,
  removed when the driver's full fingerprint changes.

Stage logs may not contain the complete original argv. Recover it from the
original launch command/task configuration when available; hashes cannot recover
missing parameters. If input/configuration is not verifiable, report the missing
fields and a conditional command template rather than claiming an exact resume.

After fixing a backend/dependency failure rerun the SAME root, output root and
arguments. Do not create a timestamped output root on every retry or manually
clear all markers/caches. Batch exit 1 means at least one selected video failed.
Exit 0 and `.done` still do not guarantee every optional export or accurate
content; inspect the outputs the user requested.

## Fingerprint limits and migration

Fingerprint source path, size and nanosecond mtime, options, manual hotword/
glossary file contents, pipeline Python code/Markdown templates and relevant
environment configuration. Hashes avoid storing credential values. OCR vocabulary
accumulated across the batch deliberately does not invalidate prior runs.

Changing the text-model flag rebuilds downstream outputs while retaining ASR;
source/ASR changes invalidate transcription too. Some code/environment changes
conservatively invalidate both stages. Replacing model weights under the same name
or a video while retaining its size/mtime is not detected automatically.

Directory names include a digest of the original relative path (including file
extension). Old unhashed directories are preserved but not automatically reused;
the first upgraded run reprocesses into new directories. Explain this before a
large legacy-library rerun. Never rename old outputs over new ones by guesswork.

For a readable audit record, write a separate optional `manifest.json` with
input path, route, actual models/backends, non-secret arguments and artifact
paths. Keep both the inputs and generated media local unless sharing was requested.

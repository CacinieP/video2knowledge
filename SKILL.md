---
name: video2knowledge
description: >-
  Turn local videos into timestamped transcripts, source-grounded study notes,
  illustrated HTML notes and Anki cards. Use for lecture/course notes, narrated
  tutorials, silent screen recordings, transcription, study-card generation or
  resuming an interrupted video2knowledge batch. Choose ASR, visual-only or
  audio/visual fusion from the source and requested outputs. Process locally
  by default; use cloud ASR/LLM/VLM only when the user chooses that destination.
---

# Video2Knowledge

## 1. Resolve the task and execution environment

- Resolve the skill root from this SKILL.md's location, not the user's working
  directory. Resolve the video/library and output directory to absolute paths.
  Ask only for a missing/ambiguous input path or a choice that blocks execution.
- Default to the input's language and local inference. Use the requested output;
  for a generic "make notes" request, deliver Markdown + HTML. Add cards,
  DOCX/PDF or a whole-library run only when useful for the user's request.
- Check `ffmpeg`, `ffprobe`, Python and model service availability. Inspect
  `scripts/hardware_profile.py --json` / `--recommend` for model suggestions;
  recommendations do not prove the model can infer. Check a representative
  image/text request before a long run. Never call a text-only model a VLM.
- First-time setup: run `bash scripts/setup_models.sh` with cwd set to the skill
  root. It creates the main venv and downloads the profile VLM. State expected
  downloads first and respect limits on downloads, service startup and cloud use.
  Run `--with-funasr` only for the separate Mandarin backend; cloud ASR SDKs
  use `--with-openai-client`. Read [runtime.md](references/runtime.md) for setup.
- Use absolute interpreter and script paths on EVERY tool invocation. Do not
  rely on shell activation, cwd or variables surviving between tool calls.
  Re-establish variables in each fresh shell, or put absolute paths into the
  tool's argument list. Shell recipes below are Bash, including Windows Git Bash.

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"  # actual installed location
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
"$PY" "$SKILL_ROOT/scripts/hardware_profile.py" --json
```

Use the setup script's printed `run python as:` path when the venv location is
customized. A FunASR command uses the separate interpreter directly:
`"/absolute/path/.venv-funasr/bin/python" "/absolute/path/scripts/asr_funasr.py" ...`
(or `.venv-funasr/Scripts/python.exe` on Windows). Never prefix it with `python3`.

## 2. Choose exactly one initial route

Apply the rows in this order. Inspect streams with `ffprobe`; an audio track's
presence alone does not prove there is speech. Sample the video/audio or inspect
an initial transcription when uncertain. Music/filler-only ASR is not narration.

| Source / requested evidence | Route | Read before execution |
|---|---|---|
| Usable speech AND meaningful slides, board, screen text, tables or demonstrations are needed | 3: ASR + visual OCR + fusion | [path3-fusion.md](references/path3-fusion.md) |
| Usable speech; the request only needs what was said | 2: ASR | [path2-asr.md](references/path2-asr.md) |
| No usable speech, or explicitly visual-only input | 1: visual notes / OCR | [path1-multimodal.md](references/path1-multimodal.md) |

Do not run ASR for a known silent recording. On a narrated screen recording,
prefer fusion when the screen carries needed information. Prefer installed
models that work; use the profile's `vlm_model` and `text_model` as explicit
arguments, since `mm_caption.py` itself defaults to MiniCPM-V 4.6 rather than
reading the profile. Do not switch models solely to chase a new version.

## 3. Execute the selected recipe

- Use one stable absolute run directory for this input and chosen configuration.
  Preserve the original video. Keep generated media under a local output folder;
  do not commit/upload it as part of repository maintenance.
- Route 2: transcribe, apply relevant domain correction, then build the requested
  knowledge outputs. Preview corrections if appropriate, but do not leave a
  promised repair at `--dry-run`: rerun without it to actually apply the changes.
  For non-Mandarin input, do not apply the built-in Mandarin music glossary blindly.
- Route 3: ASR → visual OCR → hotword coverage → optional ASR verification →
  correction of the FINAL transcript → fusion → knowledge/illustrated notes.
  Read the referenced recipe rather than inventing CLI flags.
- Route 1: for illustrated notes, extract frames and give `build_notes.py` an
  explicit empty speech transcript; use a VLM to describe frames. This preserves
  the pure-visual label and avoids presenting visual captions as teacher speech.
  For requested on-screen transcription, run `mm_caption.py --prompt-ocr` and
  deliver `captions.json` / `.srt`; label them visual OCR, not spoken subtitles.
- Keep knowledge output language consistent: `--lang zh` or `--lang en` on the
  builders; ASR language is a separate `--language` option. For unsupported
  builder languages, explain the limitation instead of promising native output.
  Local VLM frame descriptions currently use a Chinese prompt even with
  `--lang en`. For English-only notes omit `--describe-frames`, keeping original
  frames plus English narration notes; use bilingual captions only if acceptable.
  Do not silently monkeypatch prompts to promise unsupported language behavior.
  Preserve formula/table evidence as original frames and OCR, since the minimal
  HTML renderer is not a mathematical typesetter.
- Add Anki only from usable, source-grounded questions. Read
  [outputs.md](references/outputs.md) for `--source-id`, file mapping and migration.
- For a Mandarin narrated COURSE LIBRARY with visual content, use
  `batch_run.py --root ... --out-root ...` and first inspect `--dry-run`.
  Read [batch-resume.md](references/batch-resume.md). This batch driver hardcodes
  Chinese and the fusion chain; do not use it unchanged for an English library
  or silent-only library. Per-video scripts support the broader routes.

## 4. Handle failures and resume

| Observation | Action |
|---|---|
| Missing interpreter/dependency/model | Report the exact missing component; use setup within the user's constraints. Do not silently switch to cloud. |
| Service responds, inference fails | Report the inference error and chosen backend/model; retry only a transient failure, or use an already-working route consistent with the request. |
| `build_knowledge.py` exits 4 / degraded marker | Preserve outputs for inspection, mark incomplete, fix the backend and rerun. Do not import placeholder questions into Anki. |
| Empty / low-information ASR | Do not invent narration; switch to visual notes if useful, or report no usable speech. |
| Batch exits 1 / `.failed` | Inspect stage logs and rerun the same batch command/output root after fixing the cause. Let the driver handle markers and fingerprints. |
| Optional PDF cannot render | Keep valid Markdown/HTML, report PDF as missing; install/configure a CJK font only if PDF is needed. |

Batch retries reuse valid `.asr_done` checkpoints, clear old failure markers,
and validate `.done` against `run_manifest.json`. The manifest contains hashes,
not a readable audit record. Write an optional separate `manifest.json` for
human-readable input, route, model, arguments and artifact paths. Do not claim
that these files are interchangeable. Do not delete markers or caches blindly.

## 5. Validate and deliver

- Check the requested files exist, are non-empty where appropriate and open or
  parse. Check transcript segments for meaningful text and valid timestamps.
  Compare samples near the beginning, middle and end against the source; a zero
  exit code or file presence alone does not establish content correctness.
- Inspect knowledge summaries for degradation/placeholders, notes for missing
  frames and unexpected absent descriptions, and CSV questions/answers before
  packaging cards. A visual-only video may correctly have no knowledge cards.
- Use `recall_check.py --golden` only with a reviewed golden list for that input.
  Its keyword/number recall is a diagnostic, not a factuality score or approval.
- Report **complete**, **partial** or **blocked** based on requested outputs.
  Return artifact links/absolute paths, route and models/backends, local/cloud
  use, any ungenerated outputs/degradation/known omissions, and the exact resume
  command if incomplete. Do not call a partial result complete.

## Cloud use and data boundaries

Use cloud endpoints only when the user chooses them; check existing
`V2K_LLM_API_BASE`, `V2K_LLM_API_MODEL`, `V2K_CLOUD_PROTOCOL`, `ASR_API_*` and
`OLLAMA_HOST` settings before calling tools. An explicit CLI flag is not the only
way to enable uploads. A remote Ollama host also sends data off the machine.
Describe the selected destination and data:

- cloud ASR: extracted audio;
- cloud knowledge/notes: subtitle or fused text (which may contain OCR content);
- cloud VLM notes with `--describe-frames`: selected frame images AND text.

Keep keys in configured environment variables/credential stores, never argv or
run manifests. `mm_caption.py` uses the Ollama API; a cloud notes endpoint does
not replace the route-3 OCR stage. Read [cloud.md](references/cloud.md).

## References to load only as needed

- [runtime.md](references/runtime.md): root/interpreter resolution, install and preflight.
- [path1-multimodal.md](references/path1-multimodal.md): silent visual notes and OCR.
- [path2-asr.md](references/path2-asr.md): speech, FunASR and domain correction.
- [path3-fusion.md](references/path3-fusion.md): narrated visual evidence and fusion.
- [batch-resume.md](references/batch-resume.md): supported batch scope, state and retries.
- [outputs.md](references/outputs.md): output schemas, Anki IDs and acceptance.
- [cloud.md](references/cloud.md): opt-in endpoint commands and upload boundaries.
- [hardware-profiles.md](references/hardware-profiles.md): sizing details; check live detection.
- [templates.md](references/templates.md): custom knowledge document templates.
- [engineering-notes.md](references/engineering-notes.md): historical sample measurements.

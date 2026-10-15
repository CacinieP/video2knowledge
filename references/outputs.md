# Output mapping and completion checks

## Pick outputs deliberately

| Requested result | Script / option | Files to check |
|---|---|---|
| Speech transcript | ASR backend | `subtitles.json`, `.srt`, `.vtt` |
| On-screen OCR / visual captions | `mm_caption.py` | `captions.json`, `.srt`, `frames/` |
| Knowledge notes | `build_knowledge.py --format html` | `knowledge.md`, `knowledge.html` |
| Study questions | `build_knowledge.py --format csv` | `cards.csv` |
| Anki deck | `gen_apkg.py` after CSV validation | `cards.apkg` |
| Illustrated notes | `build_notes.py` with frames | `notes.md`, `notes-distilled.md`, `notes.html` |
| Knowledge DOCX/PDF | explicit `--format docx` / `pdf` | `knowledge.docx` / `.pdf` |
| Illustrated DOCX/PDF | `build_notes.py --docx --pdf` | full and distilled `.docx` / `.pdf` |

`build_knowledge.py --format all` includes knowledge Markdown/HTML/CSV and
attempts office exports when libraries are available. It does NOT run
`build_notes.py` or `gen_apkg.py`. Those require separate commands.

Markdown illustrated notes need their relative frame files; share the containing
folder or use self-contained `notes.html`. The distilled Markdown omits narrator
quotations; there is no separate distilled HTML output. Visual-only notes can
legitimately have no narration or cards.

## Schemas and Anki identity

JSON ingestion accepts a list of `{start,end,text}` or an object with a
`segments` / `captions` list. ASR timestamps refer to speech; visual-caption
timestamps refer to sampled screen states. Never interchange those descriptions.
CSV columns are `question,answer,tags,timestamp,source`; inspect questions and
answers for actual source support, not only row count.

Question deduplication normalizes punctuation, whitespace and case. It does not
perform semantic paraphrase matching; differently worded duplicates may remain.
The `timestamp` column is not guaranteed to be populated for every model's QA.

Anki uses model ID `1830540287`, a name-hashed deck ID and a GUID derived from
deck name, source ID, timestamp and question. Row reordering/insertion and answer
edits do not change existing identity. Questions or timestamps that change can
create new notes. The default manual source ID is the absolute CSV path; use an
explicit stable ID to preserve identity after moving directories/machines:

```bash
"$PY" "$SKILL_ROOT/scripts/gen_apkg.py" --csv "$OUT/cards.csv" --out "$OUT/cards.apkg" --deck "My course" --source-id "course/lecture-01"
```

Use the same source ID for repeated exports of the SAME video, and different IDs
for different videos. Batch exports use the relative video path automatically.
The v1.9.0 scheme differs from older positional GUIDs: existing old decks require
manual migration and a first upgraded import may create duplicates. Do not
promise automatic migration or claim this prevents every kind of duplication.

## Rendering and optional formats

`knowledge.html` is a minimal Markdown renderer, not a full table/math/player
renderer. Its timestamp links currently target `#t-mm:ss` fragments without
matching generated targets or an embedded video player. Treat them as timestamp
text; do not promise clickable playback or working section navigation.
`notes.html` embeds frame images and has no player integration either.

For local illustrated notes, `--lang en` changes narration-derived headings/notes,
not the hardcoded Chinese VLM frame-description prompt. Omit `--describe-frames`
for English-only output, or explicitly report bilingual captions. Preserve
formula/table screenshots and OCR; minimal HTML is not full mathematical typesetting.

DOCX/PDF use `python-docx` / `fpdf2`. Chinese PDF also needs a supported system
font (Linux example: `fonts-wqy-microhei`) or `V2K_PDF_FONT`. Missing libraries
may cause warnings/skipped exports; a missing font can abort PDF generation.
Use Markdown/HTML as the baseline and report missing requested office outputs.

## Interpret status and validate content

| State | Interpretation | Deliver as |
|---|---|---|
| Valid requested artifacts, source samples checked | expected completion | complete, with limits stated |
| `build_knowledge.py` exit 4 / fallback markers | text inference degraded | partial / incomplete |
| Empty or low-information speech | no basis for spoken knowledge | no-speech; offer visual notes |
| Notes without a reachable model | frames/raw excerpts may still exist | partial when semantic notes were requested |
| Missing optional requested export | format not delivered | partial; list the missing format |
| Batch exit 1 | at least one selected video failed | partial; list failures and resume command |

`knowledge_doc_status()` classifies analysis and transcript information; it does
not prove a model registry contains working weights. Empty-speech builds can
write explicitly labelled placeholder documents, but must not fabricate facts or
cards. `is_degraded()` inspects the analysis/fallback marker, not the model list.

Before delivery, parse/check requested files and meaningful transcript segments;
compare beginning/middle/end samples with source evidence. Check missing images,
degraded markers and card questions. Use reviewed golden-set recall only as a
diagnostic: finding a keyword in raw quotations does not prove accurate synthesis.

Report: complete/partial/blocked; artifact paths/links; selected route and actual
models/backends; local/cloud destination and data; missing outputs/degradation;
exact resume command when needed. Read [batch-resume.md](batch-resume.md) for
checkpoint hashes versus the optional readable audit manifest.

# Outputs Reference (Step 2.2 / 2.3 / 2.4)

`build_knowledge.py --format` controls which artifacts are produced from the
subtitles. `--format all` (default) emits all three. Illustrated notes (2.4)
come from the separate `build_notes.py`.

## 2.2 — HTML (`knowledge.html`)

A single, self-contained HTML file (no external assets) so it can be shared,
emailed, or opened offline.

- Inline CSS, system font stack (`-apple-system, PingFang SC`).
- `[mm:ss]` timestamps in the timeline become clickable anchors
  (`<a class="ts">`) — wire them to your video player if you host one; standalone
  they jump to `#t-mm:ss` fragments.
- Markdown subset supported: `#`–`######` headings, `- ` bullets, `> ` quotes,
  `**bold**`, paragraphs.

Rendering is intentionally minimal (no full markdown engine dependency). For a
richer render, convert `knowledge.md` with your own tool (pandoc, md-to-html) and
drop it in the run folder.

## 2.3 — CSV (`cards.csv`) and Anki (`cards.apkg`)

### cards.csv schema

| Column | Meaning |
|---|---|
| `question` | Card front (from the `Q:` lines of `{{qa}}`) |
| `answer` | Card back (from the `A:` lines) |
| `tags` | Free tags (empty by default — fill via `--template`-driven edits or post-processing) |
| `timestamp` | `mm:ss` pointing back into the source video |
| `source` | The subtitle file the card was derived from |

Parsed from the LLM's `{{qa}}` section: any line starting with `Q:` opens a card,
the following `A:` line supplies the answer. Cards are emitted in order.

**Deduplication.** `cards_from_qa` is a parser, so repeats used to survive
straight into the deck. A long lecture is summarised map/reduce over chunks and
the model re-asks the same question in each one. Measured over a real course:
145 of 1608 cards were duplicates, worst case 117 rows collapsing to 11.

Matching is on the **normalized question** — punctuation, whitespace and case
stripped — not the raw string, because models rarely repeat verbatim. "两个白键
之间是全音" and "白键和相邻白键间隔是一个全音" are the same card with different
wording. First occurrence wins, and the row order is preserved.

### gen_apkg.py

Wraps `cards.csv` into an Anki package with a fixed model
(`Video2Knowledge Card`, model id `1830540287`) and a stable deck id derived from
the deck name. Stable ids + `guid_for(question|n)` mean **rerunning on the same
CSV updates notes in place instead of duplicating them** when reimported into Anki.

Card styling: centered question + small timestamp badge on front; answer + source
on the back (with a divider).

Install the deck:

1. Open Anki desktop.
2. File → Import → select `cards.apkg`.
3. The deck `视频知识卡` (or your `--deck` name) appears in the deck list.

## 2.4 — Illustrated notes (`notes.md` + `notes-distilled.md` + `notes.html`)

`build_notes.py` interleaves deduped key frames with the narration around each
timestamp. Inputs: `--subtitles` (either path's `.json`) + `--frames`
(`frames.json` from `extract_frames.py`). Per key frame (capped by
`--max-frames`; default 0 = AUTO — one node per ~45 s of video, clamped 12-60,
cluster-stratified so every change burst keeps its settled final frame):

```markdown
## [mm:ss] <LLM section title, <=12 chars>
![mm:ss](<relative path to frame jpg>)
**画面**：<optional VLM description, --describe-frames>
<1-2 sentence note condensed from the narration window>
> 原声：<verbatim subtitle excerpt>
```

### The auto node budget

`max(12, min(60, round(duration / 45)))`. The floor is the part that matters:
it was 8 for a long stretch, and over a 306-video / 61.5h library the mean clip
is 12.1 min — so the floor bound the majority of the library to 8 nodes however
long the video ran. Raising the rate alone does nothing for those; the floor had
to move. At 45 s a 12-min clip gets 16 nodes and a 41-min lecture 55.

### Two views, one pass

| File | Contains |
|---|---|
| `notes.md` | frames + notes + quoted 原声 |
| `notes-distilled.md` | same nodes, same frames, **no quoted narration** |
| `notes.html` | base64-embedded single file — the full view only |

The distilled twin is rendered from the sections already in memory, so it costs
**no extra model call**. Measured over 295 nodes, the blockquotes were 53% of
the characters; a reader revising wants the knowledge, a reader cross-checking
the teacher's exact wording wants the quote. `--docx` / `--pdf` export both
variants.

### The 画面 line

`--describe-frames` asks the VLM for **teaching information, not people**:
scores/whiteboard/formula/chart content (note names, chords, time signatures,
annotations), then on-screen text, and explicitly *not* performer action. A
frame with no readable teaching content gets its topic named instead. Same frame,
before and after: `女士弹琴，手势讲解` → `乐谱显示音名与和弦`.

It costs one VLM call per key frame; section titles/notes cost one text-model
call each. Without a reachable Ollama both degrade to raw narration excerpts
(the note still emits). If `--vlm-model` is omitted, `supports_vision()` checks
the fallback and the run **warns and skips** the descriptions rather than
failing once per frame with a text-only model.

## Generating only one artifact

```bash
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format knowledge  # 2.1 only
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format html       # 2.2 only
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format csv        # 2.3 only
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format docx       # 2.2b office
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format pdf        # 2.2b print
python3 scripts/gen_apkg.py --csv o/cards.csv --out o/cards.apkg                      # 2.3 final
```

## DOCX / PDF export (2.2b, office & print)

`scripts/md_export.py` converts any pipeline `.md` (knowledge.md, notes.md) to
`.docx` (python-docx) or `.pdf` (fpdf2) — pure-python backends, no LaTeX, no
Word. Wired in as:

- `build_knowledge.py --format docx` / `--format pdf` (explicit), and **auto**
  with `--format all` when the libs are installed (missing libs warn + skip,
  never fail an `all` run; explicit formats fail loudly with an install hint);
- `build_notes.py --docx` / `--pdf` — illustrated notes with key frames
  embedded;
- standalone: `python3 scripts/md_export.py --md run/notes.md --out run/notes.pdf`.

Rendering: headings/bold/italic/code spans, lists, tables (gridded in docx,
headed in pdf), blockquotes, fenced code, embedded images, horizontal rules.
Details that matter in practice:

- **PDF CJK font** is auto-detected per OS (msyh/simhei on Windows, PingFang/
  Songti on macOS, Noto CJK/WenQuanYi on Linux) and embedded; override with
  `V2K_PDF_FONT=/path/to/font.ttf|ttc`.
- **ffmpeg 8 JPEGs** carry no JFIF APP0 (and add a COM comment), which
  python-docx rejects — md_export normalizes both transparently before
  embedding.

## Degraded mode

`build_knowledge.py` distinguishes three states — check with
`knowledge_doc_status()` before trusting a run folder that looks complete:

| Status | When | What you get |
|---|---|---|
| `ok` | model reachable, subtitles non-empty | full artifacts |
| `degraded` | no Ollama text model | all artifacts from **raw subtitles**, summary marked "(本地模型不可用...)", cards empty; rerun after `ollama serve`. Script exits **4** |
| `no-speech` | subtitles parsed but zero segments | **no fabricated document** |

The `no-speech` case used to be the worst failure mode in the pipeline: a silent
video produced a confident-looking knowledge doc full of invented content. A
melody-improvisation clip came back as *"how to send an HTTP request with the
Python requests library"*, complete with a fabricated `-[00:03]` timestamp.
Zero segments is a fact about the source, not a reason to ask a model to fill
the gap.

Two related traps worth knowing:

- `OLLAMA_MODELS` pointing at an empty directory makes `/api/tags` return **200
  with an empty list** and `/api/generate` return 404. A naive reachability
  check passes, the run looks fine, and the output is garbage. `is_degraded()`
  checks the model list, not just the ping.
- Never report a run as complete on "the file exists". Use `recall_check.py` or
  `knowledge_doc_status()`.

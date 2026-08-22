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

## 2.4 — Illustrated notes (`notes.md` + `notes.html`)

`build_notes.py` interleaves deduped key frames with the narration around each
timestamp. Inputs: `--subtitles` (either path's `.json`) + `--frames`
(`frames.json` from `extract_frames.py`). Per key frame (capped by
`--max-frames`, default 12):

```markdown
## [mm:ss] <LLM section title, <=12 chars>
![mm:ss](<relative path to frame jpg>)
**画面**：<optional VLM description, --describe-frames>
<1-2 sentence note condensed from the narration window>
> 原声：<verbatim subtitle excerpt>
```

- `notes.md` uses RELATIVE image refs — it renders in VS Code/Typora/GitHub as
  long as the frames dir travels with the note.
- `notes.html` embeds every frame as a base64 data URL — a single shareable
  file, no sidecar files needed.
- `--describe-frames` costs one VLM call per key frame; section titles/notes
  cost one text-model call each. Without a reachable Ollama both degrade to
  raw narration excerpts (the note still emits).

## Generating only one artifact

```bash
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format knowledge  # 2.1 only
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format html       # 2.2 only
python3 scripts/build_knowledge.py --subtitles s.json --out-dir o --format csv        # 2.3 only
python3 scripts/gen_apkg.py --csv o/cards.csv --out o/cards.apkg                      # 2.3 final
```

## Degraded mode

If no Ollama text model is reachable, `build_knowledge.py` still writes all three
artifacts using **raw subtitles** and clearly marks the summary as
"(本地模型不可用...)". Cards will be empty in that case — rerun after
`ollama serve` / `setup_models.sh`.

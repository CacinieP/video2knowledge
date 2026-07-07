# Outputs Reference (Step 2.2 / 2.3)

`build_knowledge.py --format` controls which artifacts are produced from the
subtitles. `--format all` (default) emits all three.

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

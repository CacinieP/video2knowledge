# Knowledge Document Templates (Step 2.1)

The knowledge doc is rendered by substituting `{{placeholders}}` in a template file.
`build_knowledge.py` scans the template text and replaces each known placeholder
with generated content; unknown placeholders are left intact so you can spot typos.

## Available placeholders

| Placeholder | Content |
|---|---|
| `{{title}}` | Document title (video basename, or `--title`) |
| `{{source}}` | Source file name |
| `{{duration}}` | Video duration `mm:ss` |
| `{{date}}` | Generation date `YYYY-MM-DD` |
| `{{summary}}`, `{{bullets}}` | 3–5 sentence summary |
| `{{timeline}}` | Markdown bullet list `[mm:ss] key event` (≤12) |
| `{{key_points}}` | Markdown bullet list (≤10) |
| `{{qa}}` | Markdown `Q:` / `A:` pairs (5–10) — also feeds 2.3 cards |
| `{{glossary}}` | Markdown bullet list of terms |
| `{{meta}}` | Free-form metadata line (segment count, model) |

## Built-in templates

`assets/default-template.md` is the generic default (title → metadata quote →
summary → timeline → key points → QA → glossary → meta). It is used when
`--template` is not passed.

`assets/templates/` ships three ready-to-use course templates:

| File | Shape | Use when |
|---|---|---|
| `course-notes.md` | summary → 必看要点 → 知识要点 → 时间轴 → 画面要点 → 自测题 → 术语 → 复习清单 | the general case: a lecture you will study afterwards |
| `revision-cheatsheet.md` | TL;DR → 核心清单 → 易混对照 → 自测, timeline collapsed in `<details>` | last-minute review; you only want the load-bearing 10% |
| `tutorial-steps.md` | 目标 → 操作步骤 → 参数与要点 → 画面 → 时间轴 → 常见问题 → 术语 → 自检 | software / DAW / instrument walkthroughs, where the reader follows along |

```bash
python3 scripts/build_knowledge.py \
  --subtitles runs/lec/subtitles.json --out-dir runs/lec \
  --template assets/templates/revision-cheatsheet.md --format all
```

Pick per batch, not per video — a course should keep one shape so the notes
stack. Everything outside `{{placeholders}}` is kept verbatim, so a template can
carry its own scaffolding (checklists, `<details>` blocks, instructions to the
reader) around the generated content.

> Tip: pass `--title` while batch-processing. `{{source}}` is the *subtitle*
> file's name (`subtitles.json`), so without `--title` the header reads
> `来源: subtitles.json`; `{{title}}` is the one that carries the real name.

## Custom templates

Write any `.md` file using any subset of the placeholders. Only the placeholders
you include are filled; everything else (your headings, prose, branding) is kept
verbatim. Examples:

### Lecture / course notes

```markdown
# {{title}} — 课程笔记

- 授课日期: {{date}}  | 时长: {{duration}}  | 来源: {{source}}

## 本节目标
{{summary}}

## 章节时间轴
{{timeline}}

## 必背知识点
{{key_points}}

## 自测题
{{qa}}

## 术语
{{glossary}}
```

### Meeting minutes

```markdown
# 会议纪要: {{title}}
> {{date}} · {{duration}} · {{source}}

## 议题摘要
{{summary}}

## 关键节点
{{timeline}}

## 决议与行动项
{{key_points}}

## 待跟进 Q&A
{{qa}}
```

### Technical tutorial

```markdown
# {{title}}
`{{source}}` · {{duration}}

## TL;DR
{{summary}}

## 步骤时间轴
{{timeline}}

## 操作要点
{{key_points}}

## FAQ
{{qa}}

## 关键术语
{{glossary}}

---
{{meta}}
```

## Using a custom template

```bash
python3 scripts/build_knowledge.py \
  --subtitles runs/lec/subtitles.json \
  --out-dir runs/lec \
  --template ./my-lecture-template.md \
  --format all
```

The template path is relative to your current directory (or absolute). Keep
templates in the skill repo (e.g. `runs/<name>/template.md`) so every artifact is
traceable.

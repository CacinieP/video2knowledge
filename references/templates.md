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
| `{{summary}}` | 3–5 sentence summary |
| `{{timeline}}` | Markdown bullet list `[mm:ss] key event` (≤12) |
| `{{key_points}}` | Markdown bullet list (≤10) |
| `{{qa}}` | Markdown `Q:` / `A:` pairs (5–10) — also feeds 2.3 cards |
| `{{glossary}}` | Markdown bullet list of terms |
| `{{meta}}` | Free-form metadata line (segment count, model) |

## Built-in default

`assets/default-template.md` ships a generic structure: title → metadata quote →
summary → timeline → key points → QA → glossary → meta. It is used when
`--template` is not passed.

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

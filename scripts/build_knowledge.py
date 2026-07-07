#!/usr/bin/env python3
"""build_knowledge.py — STEP 2: refine timestamped subtitles into knowledge artifacts.

Consumes SRT or JSON subtitles (from either path) and produces:
  2.1  knowledge.md   — structured knowledge doc, rendered through a template
                        (default assets/default-template.md, override with --template)
  2.2  knowledge.html — styled, self-contained HTML (clickable timeline)
  2.3  cards.csv      — knowledge-point cards (question, answer, tags, timestamp, source)

The summarization/QA/glossary extraction is delegated to an Ollama text model so the
whole pipeline stays local (privacy + traceability). If no model is reachable, the
script still emits artifacts using the raw subtitles (degraded mode, clearly marked).

Usage:
    python3 build_knowledge.py --subtitles out/subtitles.json --out-dir out \\
        --model openbmb/minicpm5:Q4_K_M --format all
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "default-template.md"

# --- subtitle loading --------------------------------------------------------

def load_subtitles(path: Path) -> tuple[list[dict], str]:
    """Return (segments, source_name). Accepts JSON or SRT.

    JSON may be: a list of {start,end,text} (Path 1 captions.json), or an object
    with a 'segments'/'captions' key (Path 2 subtitles.json).
    """
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(txt)
        if isinstance(data, list):
            return data, path.name
        segs = data.get("segments") or data.get("captions") or []
        return segs, path.name
    # SRT -> segments
    segs = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        segs.append({"start": start, "end": end, "text": " ".join(lines[2:])})
    return segs, path.name


def fmt_mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# --- Ollama summarization ----------------------------------------------------

def http_json(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ping(host: str) -> bool:
    try:
        # GET /api/tags (POST not allowed -> 405)
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
            json.loads(r.read().decode())
        return True
    except Exception:
        return False


def ask_llm(host: str, model: str, prompt: str) -> str | None:
    try:
        r = http_json(f"{host}/api/generate",
                      {"model": model, "prompt": prompt, "stream": False,
                       "think": False,
                       "options": {"temperature": 0.3}})
        return r.get("response", "").strip()
    except (urllib.error.URLError, OSError):
        return None


def _extract_json(text: str) -> dict | None:
    """Tolerantly extract the first JSON object from an LLM response.
    Strips ```json fences and finds the balanced {...} block."""
    if not text:
        return None
    t = text.strip()
    # Strip code fences if present.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    # Balanced-brace scan for the first complete object.
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = t[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try the next brace
        start = t.find("{", start + 1)
    return None


def build_analysis(host: str, model: str | None, raw_text: str, source: str) -> dict:
    """Ask the LLM for summary / timeline / key points / QA / glossary.

    Strategy: call the model once PER field with a narrow, plain-markdown prompt.
    Small local models follow simple single-task prompts far more reliably than a
    single complex JSON request. Falls back to a heuristic if no model is reachable
    or a field comes back empty.
    """
    fields = {
        "summary": "", "timeline": "", "key_points": "",
        "qa": "", "glossary": "",
    }
    if not (model and ping(host)):
        return _heuristic_fallback(raw_text, fields)

    sub = raw_text[:8000]
    tasks = [
        ("summary",
         "任务：读懂下面这段视频字幕，然后用中文写3-5句话总结它的核心内容与目的。\n"
         "要求：\n"
         "- 用自己的话概括，禁止照抄字幕原句\n"
         "- 只输出总结正文，不要前缀、不要小标题\n"
         "- 示例风格：\"本视频介绍了XX的使用方法，重点演示了A、B、C三个核心功能，"
         "并说明了D的注意事项，帮助用户快速上手。\""),
        ("timeline",
         "任务：从下面这段视频字幕中，提炼最多8个关键操作/事件节点。\n"
         "要求：\n"
         "- 每行格式严格为 `- [mm:ss] 概括性事件描述(不超过15字)`\n"
         "- 事件描述要概括，不要照抄原句\n"
         "- 只输出列表，不要前缀\n"
         "- 示例：`- [00:15] 设置本地同步目录`"),
        ("key_points",
         "任务：从下面这段视频字幕中，提炼最多8个核心知识点/操作要点。\n"
         "要求：\n"
         "- 每行格式 `- 知识点(一句话概括)`\n"
         "- 要点是提炼后的结论，禁止照抄字幕原句\n"
         "- 只输出列表\n"
         "- 示例：`- 修改文件后所有设备会自动同步`"),
        ("qa",
         "任务：基于下面这段视频字幕，设计6到10组中文问答，用于学习测试。\n"
         "严格要求（必须遵守）：\n"
         "- 每组两行：第一行 `Q: 问题`，第二行 `A: 答案`\n"
         "- 问题和答案都必须能从字幕中找到依据，禁止编造字幕里没有的内容\n"
         "- 问题用疑问句(如\"如何...?\"\"...是什么?\"\"在哪里...?\")\n"
         "- 答案用自然语言回答，不要照抄字幕原句，但内容必须忠于字幕\n"
         "- 只输出问答，不要前缀、不要编号、不要解释\n"
         "- 示例（假设字幕提到历史版本功能）：\n"
         "  Q: 如何恢复文件到之前的版本?\n"
         "  A: 右键点击文件选择历史版本，即可还原。"),
        ("glossary",
         "任务：从下面这段视频字幕中，提取最重要的术语/专有名词。\n"
         "要求：\n"
         "- 每行格式 `- 术语`\n"
         "- 只保留名词性术语，不要整句\n"
         "- 只输出列表\n"
         "- 示例：`- 同步空间`"),
    ]
    for key, instruction in tasks:
        resp = ask_llm(host, model, instruction + "\n\n字幕:\n" + sub)
        if resp and len(resp.strip()) > 3:
            # Models sometimes wrap markdown in ``` fences — strip them for clean MD.
            cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", resp.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
            fields[key] = cleaned
        else:
            fields[key] = _heuristic_fallback(raw_text, {key: ""})[key]
    return fields


def _heuristic_fallback(raw_text: str, fields: dict) -> dict:
    """Fill empty fields with raw-subtitle-derived placeholders."""
    n = raw_text.count("\n") + 1
    first = raw_text.splitlines()[0][:60] if raw_text else ""
    defaults = {
        "summary": f"(本地模型不可用，原始字幕共 {n} 行)",
        "timeline": f"- [未分段] {first}",
        "key_points": "- 原始字幕见下方；启用 Ollama 文本模型可生成结构化要点",
        "qa": "- Q: (启用本地模型自动生成问答)\n  A: ...",
        "glossary": "- (启用本地模型自动生成术语表)",
    }
    for k in fields:
        if not fields.get(k):
            fields[k] = defaults.get(k, "")
    return fields


# --- rendering ---------------------------------------------------------------

def render_template(template_path: Path, ctx: dict) -> str:
    tpl = template_path.read_text(encoding="utf-8")
    for k, v in ctx.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    # leave unknown placeholders intact (visible to the user)
    return tpl


def md_to_self_html(md: str, title: str) -> str:
    """Minimal markdown -> styled HTML. Handles headings, lists, bold, paragraphs.
    Good enough for a knowledge doc; not a full markdown engine."""
    lines = md.splitlines()
    out = ["<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
           f"<title>{html.escape(title)}</title>",
           "<style>",
           "body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:820px;"
           "margin:40px auto;padding:0 20px;line-height:1.65;color:#1f2328}",
           "h1,h2,h3{color:#0a2540}blockquote{border-left:4px solid #0a7;background:#f6f8fa;"
           "padding:.5em 1em;color:#555}code{background:#f6f8fa;padding:.1em .3em;border-radius:4px}",
           "a.ts{color:#0a7;text-decoration:none}a.ts:hover{text-decoration:underline}",
           ".meta{color:#666;font-size:.9em}</style></head><body>"]
    in_ul = False
    for ln in lines:
        s = html.escape(ln)
        if re.match(r"^#{1,6} ", s):
            if in_ul:
                out.append("</ul>"); in_ul = False
            lvl = len(re.match(r"^#+", s).group(0))
            out.append(f"<h{lvl}>{s[lvl+1:]}</h{lvl}>")
        elif s.startswith("- "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            # clickable timestamps [mm:ss]
            cell = re.sub(r"\[(\d{2}:\d{2})\]",
                          r"[<a class='ts' href='#t-\1'>\1</a>]", s[2:])
            out.append(f"<li>{cell}</li>")
        elif s.startswith("> "):
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append(f"<blockquote>{s[2:]}</blockquote>")
        elif s.strip() == "":
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append("")
        else:
            if in_ul:
                out.append("</ul>"); in_ul = False
            s2 = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
            out.append(f"<p>{s2}</p>")
    if in_ul:
        out.append("</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def cards_from_qa(qa, source: str) -> list[list[str]]:
    """Parse Q/A pairs into CSV rows. Accepts:
    - a markdown string with 'Q:' / 'A:' lines
    - a list of dicts with q/question and a/answer keys (LLM JSON output)
    - a list of strings like 'Q: ... A: ...'
    """
    rows: list[list[str]] = []
    if isinstance(qa, list):
        for item in qa:
            if isinstance(item, dict):
                q = item.get("q") or item.get("question") or item.get("Q") or ""
                a = item.get("a") or item.get("answer") or item.get("A") or ""
                if q:
                    rows.append([str(q).strip(), str(a).strip(), "", "", source])
            elif isinstance(item, str):
                rows.extend(cards_from_qa(item, source))
        return rows
    if not isinstance(qa, str):
        return rows
    cur_q, cur_a = None, None
    for ln in qa.splitlines():
        s = ln.strip()
        if s.lower().startswith("q:"):
            if cur_q is not None:
                rows.append([cur_q, (cur_a or "").strip(), "", "", source])
            cur_q, cur_a = s[2:].strip(), None
        elif s.lower().startswith("a:"):
            cur_a = s[2:].strip()
    if cur_q is not None:
        rows.append([cur_q, (cur_a or "").strip(), "", "", source])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2: subtitles -> knowledge artifacts")
    ap.add_argument("--subtitles", required=True, type=Path,
                    help="subtitles.json or subtitles.srt from Step 1")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="knowledge-doc template (default assets/default-template.md)")
    ap.add_argument("--format", choices=["knowledge", "html", "csv", "all"], default="all")
    ap.add_argument("--model", default=os.environ.get("V2K_TEXT_MODEL", "openbmb/minicpm5:Q4_K_M"),
                    help="Ollama text model for summarization/QA")
    ap.add_argument("--title", default=None, help="document title (default: video basename)")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()

    if not args.subtitles.is_file():
        print(f"[err] subtitles not found: {args.subtitles}", file=sys.stderr)
        return 2
    if not args.template.is_file():
        print(f"[err] template not found: {args.template}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    segs, source = load_subtitles(args.subtitles)
    raw_text = "\n".join(f"[{fmt_mmss(s['start'])}] {s['text']}" for s in segs)

    title = args.title or args.subtitles.stem.replace("_", " ")
    duration = segs[-1]["end"] if segs else 0.0
    print(f"[v2k] {len(segs)} segments, {duration:.0f}s; summarizing with {args.model}...",
          file=sys.stderr)

    analysis = build_analysis(args.host, args.model, raw_text, source)

    def as_md(v) -> str:
        """Coerce any analysis value into a markdown string for template/HTML."""
        if isinstance(v, list):
            return "\n".join(f"- {item}" if isinstance(item, str)
                             else f"- {item.get('q') or item.get('question','')}: "
                                  f"{item.get('a') or item.get('answer','')}"
                             for item in v) or "(无)"
        if isinstance(v, dict):
            return "\n".join(f"- **{k}**: {val}" for k, val in v.items()) or "(无)"
        return str(v) if v else "(无)"

    ctx = {
        "title": title,
        "source": html.escape(source),
        "duration": fmt_mmss(duration),
        "date": dt.date.today().isoformat(),
        "summary": as_md(analysis["summary"]),
        "timeline": as_md(analysis["timeline"]),
        "key_points": as_md(analysis["key_points"]),
        "qa": as_md(analysis["qa"]),
        "glossary": as_md(analysis["glossary"]),
        "meta": f"segments={len(segs)} model={args.model}",
    }

    want = {"knowledge", "html", "csv"} if args.format == "all" else {args.format}

    if "knowledge" in want or "html" in want:
        md = render_template(args.template, ctx)
        md_path = args.out_dir / "knowledge.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"[ok] knowledge doc -> {md_path}")

    if "html" in want:
        md = (args.out_dir / "knowledge.md").read_text(encoding="utf-8") \
            if (args.out_dir / "knowledge.md").exists() \
            else render_template(args.template, ctx)
        h = md_to_self_html(md, title)
        html_path = args.out_dir / "knowledge.html"
        html_path.write_text(h, encoding="utf-8")
        print(f"[ok] html -> {html_path}")

    if "csv" in want or "all" in want:
        rows = cards_from_qa(analysis["qa"], source)
        csv_path = args.out_dir / "cards.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer", "tags", "timestamp", "source"])
            w.writerows(rows)
        print(f"[ok] {len(rows)} cards -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

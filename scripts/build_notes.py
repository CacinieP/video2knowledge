#!/usr/bin/env python3
"""build_notes.py — STEP 2.4: illustrated notes (图文笔记).

Interleaves deduped key frames with the narration around each timestamp into a
scrollable illustrated note. Each section is:

    [mm:ss] section title        <- LLM-condensed from the narration window
    ┌ key frame image            <- from extract_frames.py (dedup mode)
    ├ 画面: <VLM description>     <- optional, --describe-frames
    ├ 1-2 sentence note          <- LLM-condensed narration (degrades to excerpt)
    └ > verbatim narration excerpt (blockquote)

Consumes:
    --subtitles  subtitles.json (Path 2) or captions.json (Path 1)
    --frames     frames.json from extract_frames.py: {"frames":[{"file","t"}]}

Outputs (in --out-dir):
    notes.md     markdown note; images referenced by RELATIVE path, so it
                 renders in VS Code/Typora/GitHub when the frames dir travels
                 with the note
    notes.html   fully self-contained (frames embedded as base64 data URLs) —
                 a single shareable file

All LLM/VLM calls are optional: without a reachable Ollama the note degrades to
key frames + raw narration excerpts (clearly marked).

Usage:
    python3 build_notes.py --subtitles out/subtitles.json \\
        --frames out/frames/frames.json --out-dir out \\
        --max-frames 12 --describe-frames --model openbmb/minicpm5-2b
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_knowledge import ask_llm, fmt_mmss, load_subtitles, ping  # noqa: E402
from extract_frames import cap_by_time  # noqa: E402
from hardware_profile import default_text_model  # noqa: E402

DESC_PROMPT = (
    "请用中文描述这一帧中**可用于学习的教学信息**（不超过40字），按优先级：\n"
    "1) 乐谱/板书/公式/图表上的具体内容（如音名、和弦、拍号、标注）；\n"
    "2) 屏幕文字、字幕、界面上的教学提示；\n"
    "3) 演示的器材或操作步骤。\n"
    "**不要描述人物动作或外观**（如“女士弹琴”“手势讲解”“老师微笑”）——"
    "这些对复习没有帮助。画面里没有可读教学信息时，"
    "就写画面主题（如“钢琴演奏画面，无字幕”）。"
    "只输出描述本身，不要推理过程，不要英文。"
)


def http_generate(host: str, payload: dict, timeout: int = 420,
                  retries: int = 1) -> dict:
    import time
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps({**payload, "keep_alive": -1}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError):
            # VLM frame descriptions measured ~160s under contention; the old
            # 180s no-retry bound turned slow windows into failed videos
            if attempt == retries:
                raise
            time.sleep(30)


def supports_vision(host: str, model: str) -> bool:
    """True if `model` declares the 'vision' capability (ollama /api/show).

    Needed because the --vlm-model fallback below reuses the TEXT model, and
    on the low/mid profiles that model is text-only (minicpm5-2b, qwen3.5:0.8b
    on tiny). Handing a text-only model a base64 image returns HTTP 500 for
    every key frame, and one description per frame in a tight retry loop is
    enough to take the ollama server down mid-run (observed: 23/23 failures
    followed by connection-refused for the rest of the batch).
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{host}/api/show", data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            caps = json.loads(r.read().decode()).get("capabilities") or []
        return "vision" in caps
    except Exception:
        return False  # unknown -> assume no vision, degrade instead of storming


def describe_frame(host: str, model: str, jpg: Path) -> str:
    """One-line VLM description of a key frame (used for the 画面 line)."""
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    r = http_generate(host, {"model": model, "prompt": DESC_PROMPT, "images": [b64],
                             "stream": False, "think": False,
                             "options": {"temperature": 0.2, "num_predict": 160}})
    text = r.get("response", "").strip()
    if "</think>" in text:  # some models leak the reasoning chain
        text = text.rsplit("</think>", 1)[1].strip()
    return text.replace("\n", " ")


def _strip_line_label(line: str) -> str:
    """Drop a leading '第一行：' / 'Line 1:' label the model echoed back.

    The prompt asks for two lines AND says "no extra prefixes"; a small model
    cannot hold both and copies the labels verbatim, which otherwise ships as
    "## [00:04] 第一行：视谱能力训练".
    """
    return re.sub(r"^\s*(?:第[一二]行|line\s*\d+)\s*[:：]\s*", "", line,
                  flags=re.IGNORECASE).strip()


def section_note(host: str, model: str, narration: str, lang: str) -> tuple[str, str]:
    """Condense one section's narration into (title, note). Falls back to
    ('', excerpt) when the model is unreachable or answers empty."""
    if not narration.strip():
        return "", ""
    if not (model and ping(host)):
        excerpt = " / ".join(narration.splitlines()[:2])[:120]
        return "", excerpt
    task = ("任务：下面是视频某一段的旁白字幕。请输出两行：\n"
            "第一行：该段小标题（不超过12个字，概括这段在做什么）\n"
            "第二行：1-2句话的要点笔记（这步做了什么、关键参数/材料/注意事项）\n"
            "不要时间戳，不要序号，不要多余前缀。") if lang == "zh" else (
        "Task: the lines below are narration subtitles for one video section. "
        "Output exactly two lines:\n"
        "Line 1: a section title (<=8 words, what this section does)\n"
        "Line 2: a 1-2 sentence note (what is done; key materials/params/caveats)\n"
        "No timestamps, no numbering, no preamble.")
    resp = ask_llm(host, model, task + "\n\n字幕:\n" + narration[:2000])
    if not resp or not resp.strip():
        return "", " / ".join(narration.splitlines()[:2])[:120]
    lines = [l.strip() for l in resp.strip().splitlines() if l.strip()]
    # Small models echo the "第一行：/第二行：" labels from the prompt verbatim
    # (the prompt asks for two lines AND says "no extra prefixes"; a 2B model
    # cannot hold both). Strip them, or every section title ships as
    # "## [00:04] 第一行：视谱能力训练".
    lines = [_strip_line_label(l) for l in lines]
    lines = [l for l in lines if l]
    if len(lines) >= 2:
        return lines[0][:24], " ".join(lines[1:])[:300]
    if lines:
        return lines[0][:24], " / ".join(narration.splitlines()[:2])[:120]
    return "", " / ".join(narration.splitlines()[:2])[:120]


def resolve_frame_file(frames_json: Path, file_ref: str) -> Path:
    """Locate a frame image from its recorded path.

    extract_frames.py writes the path as given on its command line, which may be
    absolute, relative to its cwd, or (in checked-in examples) relative to a
    long-gone repo root. Try, in order: as-is / relative to the frames dir /
    bare filename in the frames dir / relative to the frames dir's parent; return
    the first candidate that exists, else the frames-dir guess for a clear error.
    """
    p = Path(file_ref)
    cands = [p,
             frames_json.parent / p,
             frames_json.parent / p.name,
             frames_json.parent.parent / p]
    for c in cands:
        if c.is_file():
            return c.resolve()
    return (frames_json.parent / p).resolve()


def build_sections(host: str, model: str | None, vlm_model: str | None,
                   keyframes: list[dict], segs: list[dict], describe: bool,
                   lang: str, workers: int = 4) -> list[dict]:
    """One section per key frame: image + narration window + LLM title/note.

    Sections are independent, so they are generated in parallel (each is one
    text-LLM call plus optionally one VLM call; Ollama batches concurrent
    requests). ex.map preserves input order regardless of completion order.
    """
    n = len(keyframes)
    items = []
    for i, fr in enumerate(keyframes):
        t0 = fr["t"]
        t1 = keyframes[i + 1]["t"] if i + 1 < n else float("inf")
        window = [s for s in segs if t0 <= s["start"] < t1]
        narration = "\n".join(f"[{fmt_mmss(s['start'])}] {s['text'].strip()}"
                              for s in window if s.get("text", "").strip())
        items.append((i, fr, t0, window, narration))

    def one(item):
        i, fr, t0, window, narration = item
        title, note = section_note(host, model, narration, lang)
        desc = ""
        if describe and vlm_model:
            try:
                desc = describe_frame(host, vlm_model, Path(fr["file"]))
            except Exception as e:
                print(f"[notes] frame description failed @ {t0:.0f}s: {e}",
                      file=sys.stderr)
        excerpt = " / ".join(s["text"].strip() for s in window[:3]
                             if s.get("text", "").strip())
        sec = {"t": t0, "file": fr["file"], "desc": desc,
               "title": title, "note": note, "excerpt": excerpt,
               "n_lines": len([s for s in window if s.get("text", "").strip()])}
        print(f"[notes] section {i+1}/{n} @ {fmt_mmss(t0)}"
              f" ({sec['n_lines']} lines"
              f"{', VLM' if desc else ''}{', LLM' if title else ''})",
              file=sys.stderr)
        return sec

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, max(1, n))) as ex:
        return list(ex.map(one, items))


# --- rendering ----------------------------------------------------------------

def render_markdown(sections: list[dict], meta: dict, out_dir: Path,
                     verbatim: bool = True) -> str:
    """Render the illustrated note.

    `verbatim=False` produces the distilled view: the same nodes, the same
    frames, the same LLM notes, minus the quoted narration. Measured over 295
    nodes, the blockquotes are 53% of the characters — a reader revising wants
    the knowledge, and a reader cross-checking the teacher's exact wording
    wants the quote. Producing both from one pass costs no extra model calls.
    """
    suffix = "" if verbatim else " · 纯享版"
    lines = [f"# {meta['title']} — 图文笔记{suffix}", "",
             f"> 来源 `{meta['video']}` · 时长 {meta['duration']} · "
             f"{len(sections)} 个关键节点 · {meta['date']}", ""]
    if not verbatim:
        lines += ["> 精简版：只保留画面与提炼要点，原始逐字引用见 "
                  "`notes.md`。", ""]
    for sec in sections:
        ts = fmt_mmss(sec["t"])
        heading = sec["title"] or "画面节点"
        lines.append(f"## [{ts}] {heading}")
        lines.append("")
        try:  # relative ref so the md renders wherever the frames dir travels
            rel = os.path.relpath(sec["file"], out_dir)
        except ValueError:  # different drives on Windows — fall back to absolute
            rel = sec["file"]
        lines.append(f"![{ts}]({rel.replace(os.sep, '/')})")
        lines.append("")
        if sec["desc"]:
            lines.append(f"**画面**：{sec['desc']}")
            lines.append("")
        if sec["note"]:
            lines.append(sec["note"])
            lines.append("")
        if verbatim and sec["excerpt"]:
            lines.append(f"> 原声：{sec['excerpt'][:200]}")
            lines.append("")
    return "\n".join(lines)


def render_html(sections: list[dict], meta: dict) -> str:
    parts = ["<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
             f"<title>{html.escape(meta['title'])} · 图文笔记</title>",
             "<style>",
             "body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:760px;"
             "margin:40px auto;padding:0 20px;line-height:1.7;color:#1f2328}",
             "h1{color:#0a2540}h2{color:#0a2540;border-bottom:2px solid #eaeef2;"
             "padding-bottom:.3em}",
             "img{max-width:100%;border-radius:8px;border:1px solid #eaeef2;"
             "display:block;margin:.8em 0}",
             ".ts{color:#0a7;font-size:.85em;font-weight:600;"
             "font-variant-numeric:tabular-nums}",
             ".desc{color:#57606a}.meta{color:#666;font-size:.9em}",
             "blockquote{border-left:4px solid #0a7;background:#f6f8fa;"
             "padding:.5em 1em;color:#555;margin:.6em 0}",
             "</style></head><body>",
             f"<h1>{html.escape(meta['title'])}</h1>",
             f"<p class='meta'>来源 {html.escape(meta['video'])} · "
             f"时长 {meta['duration']} · {len(sections)} 个关键节点 · {meta['date']}</p>"]
    for sec in sections:
        ts = fmt_mmss(sec["t"])
        parts.append(f"<h2><span class='ts'>[{ts}]</span> "
                     f"{html.escape(sec['title'] or '画面节点')}</h2>")
        try:
            b64 = base64.b64encode(Path(sec["file"]).read_bytes()).decode()
            parts.append(f"<img src='data:image/jpeg;base64,{b64}' "
                         f"alt='{ts}'>")
        except OSError as e:
            parts.append(f"<p class='desc'>(帧图缺失: {html.escape(str(e))})</p>")
        if sec["desc"]:
            parts.append(f"<p class='desc'>画面：{html.escape(sec['desc'])}</p>")
        if sec["note"]:
            parts.append(f"<p>{html.escape(sec['note'])}</p>")
        if sec["excerpt"]:
            parts.append(f"<blockquote>原声：{html.escape(sec['excerpt'][:200])}</blockquote>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2.4: illustrated notes (图文笔记)")
    ap.add_argument("--subtitles", required=True, type=Path,
                    help="subtitles.json (Path 2) or captions.json (Path 1)")
    ap.add_argument("--frames", required=True, type=Path,
                    help="frames.json from extract_frames.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="key-frame cap for the note (default 0 = AUTO: ~one "
                         "node per 90s of video, clamped 8-48 — a 3h "
                         "lecture gets 48 nodes, a 45-min one gets 30; an "
                         "explicit number fixes it). Cluster-stratified: every "
                         "change burst keeps its settled frame")
    ap.add_argument("--docx", action="store_true",
                    help="also export notes.docx (needs python-docx)")
    ap.add_argument("--pdf", action="store_true",
                    help="also export notes.pdf (needs fpdf2; CJK font auto-detected,"
                         " override with V2K_PDF_FONT)")
    ap.add_argument("--model", default=default_text_model(),
                    help="Ollama text model for section titles/notes "
                         "(default: the profile's text model; unset behavior "
                         "degrades gracefully)")
    ap.add_argument("--describe-frames", action="store_true",
                    help="also run the VLM on each key frame for a one-line "
                         "画面 description (uses --vlm-model)")
    ap.add_argument("--vlm-model", default=None,
                    help="VLM for frame descriptions (default: same as --model; "
                         "a vision-capable model is required)")
    ap.add_argument("--title", default=None, help="note title (default: video basename)")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="prompt/output language (default zh)")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST",
                                                     "http://localhost:11434"))
    args = ap.parse_args()

    for p, label in ((args.subtitles, "subtitles"), (args.frames, "frames")):
        if not p.is_file():
            print(f"[err] {label} not found: {p}", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    segs, _ = load_subtitles(args.subtitles)
    fdata = json.loads(args.frames.read_text(encoding="utf-8"))
    frames = [dict(fr, file=str(resolve_frame_file(args.frames, fr["file"])))
              for fr in fdata.get("frames", [])]
    if not frames:
        print(f"[err] no frames listed in {args.frames}", file=sys.stderr)
        return 2
    # cluster-stratified cap (shared with extract_frames.py): every change
    # burst keeps its settled final frame, remaining budget split proportionally
    # — an animation burst no longer starves isolated key slides of sections.
    #
    # --max-frames 0 (default) = AUTO: one node per ~45s of video, clamped
    # 12-60. The old 1-per-4min gave a 37min lecture only 9 illustrated nodes,
    # and the 1-per-90s that followed still floored at 8 — which is most of a
    # typical clip: measured over a 306-video / 61.5h library the mean length
    # is 12.1 min, so the floor bound the majority of the library to 8 nodes
    # no matter how long the video ran. The floor is what had to move, not the
    # ceiling: at 45s a 12-min clip gets 16 nodes and a 41-min one gets 55.
    if args.max_frames <= 0:
        duration = segs[-1]["end"] if segs else 0.0
        args.max_frames = max(12, min(60, round(duration / 45)))
        print(f"[notes] auto node budget: {args.max_frames} "
              f"(~1 per 45s of {fmt_mmss(duration)})", file=sys.stderr)
    keep_ts = set(cap_by_time([fr["t"] for fr in frames], args.max_frames))
    frames = [fr for fr in frames if fr["t"] in keep_ts]

    model = args.model if ping(args.host) else None
    vlm_model = None
    if args.describe_frames:
        vlm_model = args.vlm_model or args.model
        if vlm_model and not supports_vision(args.host, vlm_model):
            print(f"[notes] {vlm_model} is text-only — skipping 画面 descriptions "
                  f"instead of failing once per key frame. Pass --vlm-model "
                  f"<multimodal>, e.g. openbmb/minicpm-v4.6:latest, to enable them.",
                  file=sys.stderr)
            vlm_model = None
        if not ping(args.host):
            vlm_model = None
    if not model:
        print("[notes] Ollama unreachable — degrading to raw excerpts "
              "(titles/notes marked accordingly)", file=sys.stderr)

    print(f"[notes] {len(frames)} key frames x {len(segs)} segments; "
          f"text={model or 'off'} vlm={vlm_model or 'off'}", file=sys.stderr)
    sections = build_sections(args.host, model, vlm_model, frames, segs,
                              describe=vlm_model is not None, lang=args.lang)

    duration = segs[-1]["end"] if segs else 0.0
    meta = {"title": args.title
                    or Path(fdata.get("video", "video")).stem.replace("_", " "),
            "video": Path(fdata.get("video", "unknown")).name,
            "duration": fmt_mmss(duration), "date": dt.date.today().isoformat()}

    md = render_markdown(sections, meta, args.out_dir, verbatim=True)
    md_path = args.out_dir / "notes.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[ok] illustrated note (md) -> {md_path}")

    # Distilled twin: same nodes and frames, no quoted narration. Generated
    # from the sections already in hand, so it costs no extra model call.
    md_light = render_markdown(sections, meta, args.out_dir, verbatim=False)
    md_light_path = args.out_dir / "notes-distilled.md"
    md_light_path.write_text(md_light, encoding="utf-8")
    print(f"[ok] distilled note (md) -> {md_light_path}")

    html_path = args.out_dir / "notes.html"
    html_path.write_text(render_html(sections, meta), encoding="utf-8")
    print(f"[ok] illustrated note (self-contained html) -> {html_path}")

    if args.docx or args.pdf:  # optional office/print exports, degrade gracefully
        try:
            from md_export import md_to_docx, md_to_pdf
            if args.docx:
                p = md_to_docx(md, args.out_dir / "notes.docx", base_dir=args.out_dir)
                print(f"[ok] illustrated note (docx) -> {p}")
                p = md_to_docx(md_light, args.out_dir / "notes-distilled.docx",
                               base_dir=args.out_dir)
                print(f"[ok] distilled note (docx) -> {p}")
            if args.pdf:
                p = md_to_pdf(md, args.out_dir / "notes.pdf",
                              base_dir=args.out_dir, title=str(meta["title"]))
                print(f"[ok] illustrated note (pdf) -> {p}")
                p = md_to_pdf(md_light, args.out_dir / "notes-distilled.pdf",
                              base_dir=args.out_dir, title=str(meta["title"]))
                print(f"[ok] distilled note (pdf) -> {p}")
        except RuntimeError as e:
            print(f"[warn] export skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

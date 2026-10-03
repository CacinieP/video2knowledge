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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_knowledge import (ask_llm, configure_cloud, fmt_mmss,  # noqa: E402
                             has_usable_speech, load_subtitles, ping, _CLOUD)
from extract_frames import cap_by_time  # noqa: E402
from hardware_profile import default_text_model  # noqa: E402
from net import cloud_auth_header, pop_dead_proxy  # noqa: E402

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
    if _CLOUD["api_base"]:
        return describe_frame_cloud(model, jpg)
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    r = http_generate(host, {"model": model, "prompt": DESC_PROMPT, "images": [b64],
                             "stream": False, "think": False,
                             "options": {"temperature": 0.2, "num_predict": 160}})
    text = r.get("response", "").strip()
    if "</think>" in text:  # some models leak the reasoning chain
        text = text.rsplit("</think>", 1)[1].strip()
    return text.replace("\n", " ")


# --- cloud VLM (OpenAI-compatible, same endpoint as the text LLM) ------------
#
# When --api-base is set the text side already routes through build_knowledge's
# configure_cloud()/ask_llm(). The 画面 descriptions were still hitting the
# Ollama /api/generate shape (base64 "images" array), which no /chat/completions
# provider understands — so a cloud run rendered every node without its frame
# description unless a local Ollama happened to be running. Same probe-once
# discipline as supports_vision(): never find out per-frame in a retry loop.

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="  # 1x1 px — exists purely to be accepted
)


def _cloud_auth_header() -> str:
    """Delegates to net.cloud_auth_header so build_knowledge.py can share it
    without importing this module (that import had become circular)."""
    anthropic_mode = os.environ.get("V2K_CLOUD_PROTOCOL", "").lower() == "anthropic"
    return cloud_auth_header(anthropic_mode, _CLOUD["api_key"] or "")


def _cloud_chat(model: str, content, max_tokens: int, temperature: float):
    """One cloud call. `content` is either a string or a vision content array
    (OpenAI shape). Returns text or raises.

    V2K_CLOUD_PROTOCOL=anthropic routes through POST {base}/messages instead
    of /chat/completions, translating both the request and the vision content
    blocks. Needed because MiniMax's agent endpoint serves the anthropic
    protocol only — its /chat/completions answers 50115 direct_route_not_
    configured, while /messages returns 200.
    """
    import urllib.error
    import urllib.request
    base = _CLOUD["api_base"].rstrip("/")
    anthropic_mode = os.environ.get("V2K_CLOUD_PROTOCOL", "").lower() == "anthropic"
    if anthropic_mode:
        payload = _to_anthropic_payload(model, content, max_tokens, temperature)
        url = f"{base}/messages"
    else:
        payload = {"model": model, "stream": False, "temperature": temperature,
                   "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": content}]}
        url = f"{base}/chat/completions"
    # urllib honours HTTP_PROXY/HTTPS_PROXY from the environment. When that
    # points at a local port nobody is listening on, every cloud call dies
    # with WinError 10061 before leaving the machine -- which looks exactly
    # like an upstream outage. net.dead_proxy_in_env() probes the port and
    # only bypasses a proxy that is really dead. The module-level urlopen is
    # kept on purpose: the cloud tests monkeypatch it, and swapping in an
    # opener took 12 tests red.
    saved = pop_dead_proxy()
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": _cloud_auth_header(),
                 "anthropic-version": "2023-06-01"})
    last = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read().decode())
            if anthropic_mode:
                text = "".join(b.get("text", "") for b in (resp.get("content") or [])
                               if b.get("type") == "text").strip()
            else:
                msg = resp.get("choices", [{}])[0].get("message", {})
                text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "",
                              flags=re.DOTALL).strip()
            if not text:
                raise ValueError("empty content in cloud response")
            return text
        except (urllib.error.HTTPError, urllib.error.URLError, OSError,
                ValueError) as e:
            last = e
            code = getattr(e, "code", 0) if isinstance(e, urllib.error.HTTPError) else 0
            if attempt == 0 and (code in (408, 429) or code >= 500):
                time.sleep(5)
                continue
            raise
        finally:
            os.environ.update(saved)
    raise RuntimeError(f"cloud VLM request failed — {last}")


def _to_anthropic_payload(model: str, content, max_tokens: int, temperature: float):
    """OpenAI-shaped request -> anthropic /messages shape.

    image_url {url: "data:image/jpeg;base64,..."} becomes
    image {source: {type: base64, media_type: image/jpeg, data: ...}} so the
    same describe_frame_cloud() caller works on both protocols.
    """
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    else:
        blocks = []
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                media, _, b64 = url.partition("base64,")
                media_type = media.split("data:")[-1].split(";")[0] or "image/jpeg"
                blocks.append({"type": "image",
                               "source": {"type": "base64",
                                          "media_type": media_type, "data": b64}})
            else:
                blocks.append({"type": "text", "text": part.get("text", "")})
    out = {"model": model, "max_tokens": max_tokens,
           "messages": [{"role": "user", "content": blocks}]}
    # MiniMax rejects temperature 0.0 with 201 only when reasoning is on; keep
    # the caller-supplied value and let the endpoint decide, matching the
    # OpenAI path which never dropped it either.
    out["temperature"] = temperature
    return out


def describe_frame_cloud(model: str, jpg: Path) -> str:
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
               {"type": "text", "text": DESC_PROMPT}]
    text = _cloud_chat(model, content, max_tokens=1024, temperature=0.2)
    return text.replace("\n", " ")


def supports_vision_cloud(model: str) -> bool:
    """Probe the cloud model with a 1x1 image once per run.

    The reasoning models common on /chat/completions endpoints burn budget on
    thinking before answering; 512 tokens covers the probe with room to spare
    (measured: glm-5.3-flash used ~150 reasoning tokens for this yes/no).
    """
    try:
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
                   {"type": "text", "text": "图里是什么颜色？只答两个字。"}]
        _cloud_chat(model, content, max_tokens=512, temperature=0.0)
        return True
    except Exception as e:
        print(f"[notes] cloud VLM probe failed for '{model}': "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return False


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


# --- node density ------------------------------------------------------------

# One node per NODE_SECONDS of video, clamped to [NODE_MIN, NODE_MAX].
#
# Density is the single number deciding whether a note feels complete or patchy,
# and the *floor* is the part that bites. This went 1-per-4min -> 1-per-90s with
# a floor of 8, and both looked fine in isolation while under-serving most of a
# real library: measured over 306 videos / 61.5 h the mean clip is 12.1 min, so
# the floor pinned the majority of the library to 8 nodes however long the video
# ran. Raising the rate alone would not have moved those; the floor had to.
NODE_SECONDS = 45
NODE_MIN = 12
NODE_MAX = 60


def auto_node_budget(duration_s: float) -> int:
    """How many illustrated nodes a video of `duration_s` should get."""
    return max(NODE_MIN, min(NODE_MAX, round(duration_s / NODE_SECONDS)))


def collapse_trailing_visual_nodes(sections: list[dict]) -> list[dict]:
    """Collapse a trailing run of narration-less nodes to just the last one.

    A lecture's final board shot yields several perception-distinct frames
    seconds apart — the board is still being written, or a hand crosses it —
    all after the last spoken word. Each becomes a section with an empty
    narration window, no note, and a near-identical 画面 description, so the
    note ends with 2-4 nodes repeating the same final board (measured: 07:12,
    07:13 and 07:14 all reading "伯努利方程求解，令z=y⁻¹…"). The last of the
    run is the fully-settled board — keep that one alone.

    No-op for wordless videos: there every node is intentionally visual-only
    (see build_sections), and their descriptions are genuinely different
    frames of the performance, not re-reads of one static board.
    """
    if not sections or not sections[0].get("speech", True):
        return sections
    i = len(sections)
    while i > 0 and not sections[i - 1].get("excerpt") \
            and not sections[i - 1].get("note"):
        i -= 1
    dropped = len(sections) - i - 1
    if dropped <= 0:
        return sections
    print(f"[notes] collapsed {dropped} trailing visual-only node(s) into the "
          f"final settled frame", file=sys.stderr)
    return sections[:i] + [sections[-1]]


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

    A transcript with no usable speech still gets its frames and VLM
    descriptions — for a wordless video those are the whole point of the note.
    What it does *not* get is a written "note" per node: asking a small model
    to condense "嗯嗯，背" produces a study tip it invented. Measured over the
    no-speech half of a music-course library, that was 125 fabricated node
    bodies across 11 notes, one of them "用艾宾浩斯记忆曲线安排复习时间" from a
    two-syllable transcript. The picture is real; the sentence under it is not.
    """
    n = len(keyframes)
    speech = has_usable_speech(segs)
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
        if speech:
            title, note = section_note(host, model, narration, lang)
        else:
            # No usable transcript: the frame description is the only honest
            # thing this node can say, and it is real. Ask for nothing else.
            title = note = ""
        desc = ""
        if describe and vlm_model:
            try:
                desc = describe_frame(host, vlm_model, Path(fr["file"]))
            except Exception as e:
                print(f"[notes] frame description failed @ {t0:.0f}s: {e}",
                      file=sys.stderr)
        # Quoting "嗯嗯嗯" as 原声 would present a hallucination as a citation.
        excerpt = "" if not speech else " / ".join(
            s["text"].strip() for s in window[:3] if s.get("text", "").strip())
        sec = {"t": t0, "file": fr["file"], "desc": desc,
               "title": title, "note": note, "excerpt": excerpt,
               "n_lines": len([s for s in window if s.get("text", "").strip()]),
               "speech": speech}
        print(f"[notes] section {i+1}/{n} @ {fmt_mmss(t0)}"
              f" ({sec['n_lines']} lines"
              f"{', VLM' if desc else ''}{', LLM' if title else ''})",
              file=sys.stderr)
        return sec

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, max(1, n))) as ex:
        return list(ex.map(one, items))


# --- rendering ----------------------------------------------------------------

# Unicode superscripts the VLM emits inside formulas (y⁻¹, Cx⁻⁶, yⁿ …).
# Microsoft YaHei — the CJK font fpdf2 auto-detects and subsets — lacks ⁻ and
# the non-BMP-safe superscript digits, and fpdf2 silently drops glyphs the
# subset lacks, so an exponent disappears from the PDF while looking fine in
# md/html/docx (those renderers fall back to a font that has them).
_SUPERMAP = {"\u207B": "-", "\u2070": "0", "\u00B9": "1", "\u00B2": "2",
             "\u00B3": "3", "\u2074": "4", "\u2075": "5", "\u2076": "6",
             "\u2077": "7", "\u2078": "8", "\u2079": "9", "\u207F": "n"}
_SUPER_RE = re.compile("[" + "".join(re.escape(c) for c in _SUPERMAP) + "]+")


def pdf_safe_text(text: str) -> str:
    """Map unicode superscript runs to caret form for the PDF export only."""
    def repl(m: re.Match) -> str:
        body = "".join(_SUPERMAP[ch] for ch in m.group(0))
        return f"^{body}" if len(body) == 1 and body != "-" else f"^({body})"
    return _SUPER_RE.sub(repl, text)

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
    if sections and not sections[0].get("speech", True):
        # Say it once, at the top. Otherwise a run of 画面-only nodes reads as
        # a broken export rather than as what it is: a visual note for a video
        # that has no narration to condense.
        lines += ["> **纯画面笔记**：本视频没有可识别的讲解旁白，节点只保留"
                  "画面与画面描述，没有提炼要点和原声引用。", ""]
    for sec in sections:
        ts = fmt_mmss(sec["t"])
        # A node with no narration still has a picture worth showing, so fall
        # back to what the VLM read off it rather than printing the literal
        # "画面节点" placeholder in every such heading.
        heading = sec["title"] or sec["desc"] or "画面节点"
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
                         "node per 45s of video, clamped 12-60 — a 12-min "
                         "clip gets 16 nodes, a 41-min one 55, a 3h lecture "
                         "60; an explicit number fixes it). Cluster-stratified: "
                         "every change burst keeps its settled frame")
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
    ap.add_argument("--api-base", default=os.environ.get("V2K_LLM_API_BASE", ""),
                    help="use an OpenAI-compatible /chat/completions endpoint "
                         "instead of local Ollama for BOTH the section titles/notes "
                         "and the 画面 VLM descriptions (default: unset = local "
                         "Ollama). Vendor-neutral: any compatible URL works.")
    ap.add_argument("--api-model", default=os.environ.get("V2K_LLM_API_MODEL", ""),
                    help="model name at --api-base (required with it)")
    ap.add_argument("--vlm-api-model",
                    default=os.environ.get("V2K_VLM_API_MODEL", ""),
                    help="vision model at --api-base for 画面 descriptions "
                         "(default: same as --api-model — correct when that model "
                         "is multimodal, e.g. glm-5.3-flash)")
    ap.add_argument("--api-key-env", default=os.environ.get("V2K_LLM_API_KEY_ENV",
                                                            "OPENAI_API_KEY"),
                    help="env var holding the API key (default OPENAI_API_KEY). "
                         "The key is never read from argv.")
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
    duration = segs[-1]["end"] if segs else 0.0
    if args.max_frames <= 0:
        args.max_frames = auto_node_budget(duration)
        print(f"[notes] auto node budget: {args.max_frames} "
              f"(~1 per {NODE_SECONDS}s of {fmt_mmss(duration)})", file=sys.stderr)
    # Key frames that land past the last subtitle cannot have a narration
    # window at all — `build_sections` looks forward to the next key frame, so
    # their window is empty by construction and the node renders as a bare
    # image with no text under it.
    #
    # This is not hypothetical: subtitles end at the last spoken word, while
    # frame extraction runs to the end of the file. Measured on an 18:28
    # lecture, 4 of 24 nodes sat at 1110-1113s against a transcript ending at
    # 1108.7s — a quarter of the note spent on frames that cannot say anything.
    # The tolerance covers a subtitle that ends a hair before the last frame of
    # its own sentence.
    if segs:
        last_spoken = segs[-1]["end"]
        before = len(frames)
        frames = [fr for fr in frames if fr["t"] <= last_spoken + 1.0]
        if len(frames) < before:
            print(f"[notes] dropped {before - len(frames)} key frame(s) past the "
                  f"last subtitle ({fmt_mmss(last_spoken)}) — no narration possible",
                  file=sys.stderr)
        if not frames:
            print(f"[err] every key frame is past the last subtitle "
                  f"({fmt_mmss(last_spoken)}); the note would be empty",
                  file=sys.stderr)
            return 2

    keep_ts = set(cap_by_time([fr["t"] for fr in frames], args.max_frames))
    frames = [fr for fr in frames if fr["t"] in keep_ts]
    if len(frames) < args.max_frames:
        # The budget is a ceiling, not a target. Perception dedup found fewer
        # distinct frames than the budget asks for, and a node needs a real
        # image — so the note is frame-limited, not evenly spaced. Say so
        # rather than leaving the user to wonder why 18 minutes yielded 24
        # "evenly spaced" nodes that are visibly bunched.
        print(f"[notes] frame-limited: only {len(frames)} distinct key frames in "
              f"{fmt_mmss(duration)} (budget was {args.max_frames}) — the video "
              f"changes less often than {NODE_SECONDS}s, so nodes follow the "
              f"actual visual changes instead of a clock.", file=sys.stderr)

    if args.api_base and not args.api_model:
        print("[err] --api-base also needs --api-model (or $V2K_LLM_API_MODEL)",
              file=sys.stderr)
        return 2
    if args.api_base:
        # mirrors build_knowledge.py: ask_llm() routes to the cloud endpoint
        # and ping() turns true, so every existing gate keeps working
        configure_cloud(args.api_base, args.api_model, args.api_key_env)

    cloud = bool(_CLOUD["api_base"])
    model = (args.api_model or args.model) if (cloud or ping(args.host)) else None
    vlm_model = None
    if args.describe_frames:
        if cloud:
            vlm_model = args.vlm_api_model or args.api_model
            if vlm_model and not supports_vision_cloud(vlm_model):
                print(f"[notes] {vlm_model} failed the cloud vision probe — "
                      f"skipping 画面 descriptions instead of failing once per "
                      f"key frame. Pass --vlm-api-model <multimodal> to enable "
                      f"them.", file=sys.stderr)
                vlm_model = None
        else:
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
        print("[notes] no LLM reachable — degrading to raw excerpts "
              "(titles/notes marked accordingly)", file=sys.stderr)

    print(f"[notes] {len(frames)} key frames x {len(segs)} segments; "
          f"text={model or 'off'} vlm={vlm_model or 'off'}", file=sys.stderr)
    sections = build_sections(args.host, model, vlm_model, frames, segs,
                              describe=vlm_model is not None, lang=args.lang)
    sections = collapse_trailing_visual_nodes(sections)

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
                # fpdf2's CJK font subset drops the superscript glyphs the VLM
                # loves (⁻ⁿ⁶ …): map them to caret form for PDF only — md/html/
                # docx renderers fall back to a font that has them
                p = md_to_pdf(pdf_safe_text(md), args.out_dir / "notes.pdf",
                              base_dir=args.out_dir, title=str(meta["title"]))
                print(f"[ok] illustrated note (pdf) -> {p}")
                p = md_to_pdf(pdf_safe_text(md_light), args.out_dir / "notes-distilled.pdf",
                              base_dir=args.out_dir, title=str(meta["title"]))
                print(f"[ok] distilled note (pdf) -> {p}")
        except RuntimeError as e:
            print(f"[warn] export skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
        --model openbmb/minicpm5-2b --format all
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "default-template.md"

try:  # same-directory single source of truth for the Step-2 model choice
    from hardware_profile import default_text_model
except ImportError:  # imported as a stray module without scripts/ on sys.path
    def default_text_model() -> str:
        return os.environ.get("V2K_TEXT_MODEL", "openbmb/minicpm5-2b")

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


def load_merged(path: Path) -> dict:
    """Load merged.json from merge_visual.py: {segments:[{start,end,text,visual}], ...}."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_interleaved_text(segs: list[dict]) -> str:
    """Build a raw_text where each ASR line is annotated with its on-screen visual
    content (tables/formulas/examples). This is what the LLM sees in merged mode so
    it can ground the narration in the slides. Only the FIRST occurrence of each
    visual block is inlined (subsequent lines get a short marker) to avoid repeating
    a full table on every line."""
    seen_visual: set[int] = set()
    lines = []
    for i, s in enumerate(segs):
        ts = fmt_mmss(s["start"])
        audio = s["text"].strip()
        vis = (s.get("visual") or "").strip()
        # merge_visual.py semantic-check notes surface as ⚠️ so the model knows
        # this visual attribution is swapped-from-timestamp or uncertain
        warn = f"  | ⚠️{s['note'].strip()}" if s.get("note") else ""
        if vis:
            # mark by content identity to avoid repeating identical visual text
            vid = hash(vis)
            if vid in seen_visual:
                lines.append(f"[{ts}] 🎙️{audio}{warn}  | 🖼️(画面同上)")
            else:
                seen_visual.add(vid)
                lines.append(f"[{ts}] 🎙️{audio}{warn}\n🖼️画面:\n{vis}")
        else:
            lines.append(f"[{ts}] 🎙️{audio}{warn}")
    return "\n".join(lines)


def _dedup_visual_blocks(blocks: list[dict]) -> list[dict]:
    """Drop near-duplicate visual blocks (same slide shown again). Compares a
    normalized fingerprint of each block's text to its predecessor; keeps only
    blocks whose fingerprint differs. Removes prompt-pollution prefixes too."""
    def norm(t: str) -> str:
        # strip common OCR prompt-echo prefixes the VLM sometimes leaks, then
        # collapse whitespace for a stable fingerprint.
        for pre in ("以下是", "这是一张", "请逐字", "下面是", "Here is",
                    "This is", "The following"):
            if t.lstrip().startswith(pre):
                t = t.lstrip()[len(pre):]
        import re as _re
        return _re.sub(r"\s+", "", t)
    out = []
    last_fp = ""
    for b in blocks:
        fp = norm(b.get("text", ""))
        if fp and fp != last_fp:
            out.append(b)
            last_fp = fp
    return out


def build_visual_timeline(blocks: list[dict], host: str, model: str | None,
                          lang: str = "zh") -> str:
    """Produce a concise timestamped list of on-screen key content from the merged
    visual blocks. Asks the LLM to condense each DISTINCT slide's OCR into one line;
    if no model is reachable, emits the raw blocks (truncated per block)."""
    if not blocks:
        return ""
    blocks = _dedup_visual_blocks(blocks)
    label = "画面内容" if lang == "zh" else "on-screen content"
    task = ("任务：下面是视频画面不同时刻的逐字转写（含表格/公式/数字）。请把每一帧"
            "浓缩成一行，格式 `- [mm:ss] <该画面最核心的内容，≤30字，必须保留关键数字与五行/数字组合>`。"
            "时间戳必须用该帧给定的原始 [mm:ss]，不要自己编。去掉重复，只输出列表，不要前缀。"
            ) if lang == "zh" else (
        "Task: below are verbatim per-frame transcriptions of on-screen content at "
        "different timestamps (tables/formulas/numbers). Condense each into one line "
        "`- [mm:ss] <core content, <=25 words, MUST keep key numbers/combinations>`. "
        "Use the original [mm:ss] given for each frame; do not invent timestamps. "
        "Output list only.")
    raw = "\n".join(f"[{fmt_mmss(b['start'])}] {b['text']}" for b in blocks)
    if model and ping(host):
        resp = ask_llm(host, model, task + f"\n\n{label}:\n" + raw[:12000])
        if resp and len(resp.strip()) > 3:
            cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", resp.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
            return cleaned
    # fallback: raw blocks, one line each, truncated
    out = []
    for b in blocks:
        t = fmt_mmss(b["start"])
        one = b["text"].replace("\n", " ").strip()[:40]
        out.append(f"- [{t}] {one}")
    return "\n".join(out)


# --- Ollama summarization ----------------------------------------------------

def http_json(url: str, payload: dict, timeout: int = 1800,
              retries: int = 1) -> dict:
    # keep_alive per-request: server-side OLLAMA_KEEP_ALIVE can silently
    # revert to the 5-min default (service manager restart, boot autostart),
    # which made llama-server recycle mid-burst and pay a model reload every
    # request — pin it client-side so residency does not depend on env
    data = json.dumps({**payload, "keep_alive": -1}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError):
            # a 9k-char chunk prompt can run ~30 min; dying at the end of one
            # must not fail the whole video — retry once after a breather
            if attempt == retries:
                raise
            time.sleep(30)


def ping(host: str) -> bool:
    if _CLOUD["api_base"]:
        return True
    try:
        # GET /api/tags (POST not allowed -> 405)
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
            json.loads(r.read().decode())
        return True
    except Exception:
        return False


def ask_llm(host: str, model: str, prompt: str) -> str | None:
    if _CLOUD["api_base"]:
        return ask_llm_cloud(prompt)
    # num_predict is a HARD cap, not a tuning knob. Without it ollama generates
    # until EOS or until num_ctx is full (16k here), and a small reasoning model
    # (minicpm5-2b, qwen3.5:0.8b) on a large or degenerate prompt — a 90k-char
    # subtitle dump, or a run with zero segments — can degenerate into an
    # unbounded chain that runs for hours. ollama serialises requests per model,
    # so that one call blocks every later stage AND every later video in a batch.
    # 2048 comfortably covers the JSON/QA payloads these prompts ask for;
    # overflow truncates the JSON, which _extract_json already tolerates.
    cap = int(os.environ.get("V2K_NUM_PREDICT", "2048"))
    try:
        r = http_json(f"{host}/api/generate",
                      {"model": model, "prompt": prompt, "stream": False,
                       "think": False,
                       "options": {"temperature": 0.3, "num_ctx": 16384,
                                   "num_predict": cap}})
        return r.get("response", "").strip()
    except (urllib.error.URLError, OSError) as e:
        # Say what went wrong. This returned None in silence, and the caller
        # turns None into a placeholder-filled document — so a model that is
        # simply not there (wrong name, ollama not running, OLLAMA_MODELS
        # pointing at a directory that is not the one holding the weights, so
        # /api/tags answers 200 with an empty list and /api/generate 404s)
        # produced a complete run of "successful" knowledge docs containing
        # nothing but fallback text, and exit code 0 throughout.
        print(f"[llm] generate failed for '{model}': {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


# --- OpenAI-compatible cloud LLM (optional) ----------------------------------
#
# Everything above talks to a local Ollama at `host`. This routes the same
# ask_llm() to any /chat/completions provider instead, chosen by
# `--api-base` or $V2K_LLM_API_BASE. The signature is unchanged, so every
# caller keeps working and the local path is untouched when this is unset.
_CLOUD = {"api_base": None, "api_model": None, "api_key": None}


def configure_cloud(api_base: str, api_model: str, api_key_env: str) -> None:
    """Point ask_llm() at an OpenAI-compatible endpoint instead of Ollama.

    Works with any OpenAI-shaped /chat/completions provider (Qwen/DashScope,
    GLM, DeepSeek, a self-hosted gateway). Reasoning models wrap their answer
    in <think>…</think>, which is stripped — left in, it lands in the
    knowledge doc as a wall of internal monologue.

    The key is read from the environment only, never from argv: a key passed
    on the command line ends up in the process list, in shell history, and in
    every log line that echoes the command.
    """
    if not api_base:
        return
    key_env = api_key_env or "OPENAI_API_KEY"
    key = os.environ.get(key_env, "")
    if not key:
        raise SystemExit(
            f"[err] --api-base needs env var '{key_env}' set with your API key.\n"
            f"       export {key_env}=... before running.")
    _CLOUD.update(api_base=api_base.rstrip("/"), api_model=api_model, api_key=key)
    print(f"[v2k] cloud LLM: model={api_model} base={api_base}", file=sys.stderr)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_llm_cloud(prompt: str) -> str | None:
    base, model, key = _CLOUD["api_base"], _CLOUD["api_model"], _CLOUD["api_key"]
    if not (base and model and key):
        return None
    payload = {"model": model, "stream": False, "temperature": 0.3,
               "max_tokens": int(os.environ.get("V2K_NUM_PREDICT", "2048")),
               "messages": [{"role": "user", "content": prompt}]}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                resp = json.loads(r.read().decode())
            msg = resp.get("choices", [{}])[0].get("message", {})
            text = _strip_think(msg.get("content") or "")
            if not text:
                # A 200 with no usable content is a failure, not an empty
                # answer: returning "" would let the caller write a
                # placeholder-filled document and call it done. This happens
                # when a gateway answers a different schema than expected.
                raise ValueError("no content in choices[0].message")
            return text
        except (urllib.error.HTTPError, urllib.error.URLError,
                OSError, ValueError, KeyError, IndexError) as e:
            # 429 and 5xx are worth one retry; a 401 or 400 will fail again
            # identically, so retrying it just doubles the wait.
            code = getattr(e, "code", 0) if isinstance(e, urllib.error.HTTPError) else 0
            if attempt == 0 and (code in (408, 429) or code >= 500):
                wait = 5 * (attempt + 1)
                print(f"[llm] cloud {code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            # Never silent: a None here becomes a placeholder-filled document
            # that looks like a finished deliverable.
            print(f"[llm] cloud generate failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return None
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


_TS_HEADER_RE = re.compile(r"^\[\d{1,3}:\d{2}\]")


def _chunk_lines(raw_text: str, max_chars: int) -> list[str]:
    """Split raw_text (newline-separated `[ts] ...` lines) into chunks, each
    under max_chars, never breaking a line — and never cutting a timestamp
    UNIT: a unit is a `[ts] 🎙️...` header plus its continuation lines. Visual
    blocks are multi-line (🖼️ header + table rows); a chunk boundary landing
    mid-table would leave both chunks with half a table and the LLM would
    never see the whole formula/numbers. An oversized unit gets a chunk of
    its own (allowed to exceed max_chars rather than be broken)."""
    units: list[list[str]] = []
    for ln in raw_text.splitlines():
        if _TS_HEADER_RE.match(ln) or not units:
            units.append([ln])
        else:
            units[-1].append(ln)
    chunks, cur, cur_len = [], [], 0
    for u in units:
        ulen = sum(len(l) + 1 for l in u)
        if cur and cur_len + ulen > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.extend(u)
        cur_len += ulen
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [raw_text[:max_chars]]


_ECHO_RE = re.compile(
    r"时间戳必须|原样保留|不要新编|只输出|不要解释|不要复述|不要编造|输出格式"
    r"|任务[：:]|^Task:|^\W*\[mm:ss\]\W*$")


def _strip_echo(lines: list[str]) -> list[str]:
    """Drop prompt-echo bullets: small models occasionally leak instruction
    fragments (e.g. "[mm:ss] 时间戳必须原样保留，不要新编") into list output.
    Fingerprints are instruction-specific phrases a real content bullet never
    contains."""
    return [l for l in lines if not _ECHO_RE.search(l)]


def _caps_for(raw_text: str) -> dict[str, int]:
    """Field item caps scaled by video duration (parsed from the [mm:ss]
    timestamps in raw_text). A fixed cap treats a 3h lecture like a 25min
    one: late chapters get diluted out of the timeline. Scale = duration/45min
    clamped to [1, 3]."""
    import math
    max_ts = max((t for t in (_line_ts(l) for l in raw_text.splitlines())
                  if t is not None), default=0.0)
    scale = min(3.0, max(1.0, max_ts / 60.0 / 45.0))
    base = {"timeline": 12, "key_points": 12, "qa": 30,
            "glossary": 16, "bullets": 16}
    return {k: math.ceil(v * scale) for k, v in base.items()}


class _LLMCache:
    """Disk-backed response cache for build_analysis LLM calls, keyed by
    model+prompt hash. A 3h lecture's build makes ~50 chunk calls over 3+
    hours; dying at chunk 45 used to mean redoing all of them (that is how
    016 died twice). Cache lives at <out_dir>/build_cache.json and survives
    crashes/restarts — a retried build replays hits instantly."""

    def __init__(self, path: Path | None, model: str | None):
        self.path, self.model, self.d, self.dirty = path, model, {}, False
        if path and path.is_file():
            try:
                self.d = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.d = {}

    def ask(self, host: str, prompt: str):
        key = hashlib.sha1(
            (str(self.model) + "|" + prompt).encode("utf-8")).hexdigest()
        if key in self.d:
            return self.d[key]
        resp = ask_llm(host, self.model, prompt)
        if resp:
            self.d[key] = resp
            self.dirty = True
        return resp

    def flush(self) -> None:
        if self.path and self.dirty:
            self.path.write_text(
                json.dumps(self.d, ensure_ascii=False), encoding="utf-8")
            self.dirty = False



def _strip_fence(resp: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", resp.strip())
    return re.sub(r"\n?```\s*$", "", cleaned).strip()


def _merge_list_items(items: list[str], max_n: int) -> str:
    """Dedup list items (by normalized text) and cap to max_n lines."""
    seen, out = set(), []
    for it in items:
        key = re.sub(r"\s+", "", it).strip("- ")
        if key and key not in seen:
            seen.add(key)
            out.append(it)
            if len(out) >= max_n:
                break
    return "\n".join(out)


def _parse_qa_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """Parse 'Q:'/'A:' lines into (q, a) pairs. Line-level dedup (the plain
    _merge_list_items path) can orphan an 'A:' line from its 'Q:' — QA must be
    deduped and capped as PAIRS, by question text."""
    pairs, cur_q, cur_a = [], None, None
    for ln in lines:
        s = ln.strip()
        low = s.lower()
        if low.startswith("q:"):
            if cur_q is not None:
                pairs.append((cur_q, cur_a or ""))
            cur_q, cur_a = s[2:].strip(), None
        elif low.startswith("a:"):
            cur_a = s[2:].strip()
    if cur_q is not None:
        pairs.append((cur_q, cur_a or ""))
    return pairs


def _dedupe_qa_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, out = set(), []
    for q, a in pairs:
        key = re.sub(r"\s+", "", q)
        if key and key not in seen:
            seen.add(key)
            out.append((q, a))
    return out


def _rerank_list(host: str, model: str | None, key: str, items: list[str],
                 summary: str, lang: str, cap: int, line_budget: int) -> str | None:
    """One GLOBAL pass over map-reduced list items: merge duplicates, restore
    cross-chunk order, select the top `cap`, with the (already computed)
    overall summary as context. Chunk-local extraction breaks ordering and
    misses cross-boundary duplicates; this pass restores both. Returns None
    when there is nothing to cut, no model, or the response is unusable —
    the caller then falls back to truncation."""
    if not items or len(items) <= cap or not (model and ping(host)):
        return None
    if len(items) > 150:  # 3h lectures: bound the global-pass prompt size
        step = (len(items) - 1) / 149
        items = [items[round(i * step)] for i in range(149)] + [items[-1]]
    zh_names = {"timeline": "关键事件节点", "key_points": "核心知识点",
                "qa": "问答", "glossary": "术语", "bullets": "音画合并要点"}
    if lang == "zh":
        instr = (f"任务：下面是对一段长视频分块抽取的{zh_names.get(key, key)}候选条目，"
                 f"可能重复、顺序混乱或在分块边界处断裂。请结合视频总体摘要，合并重复项、"
                 f"恢复合理顺序，从中选出最重要的至多{cap}条。保持每行原格式不变；"
                 f"[mm:ss] 时间戳必须原样保留，不要新编。只输出最终列表，不要解释。\n\n"
                 f"视频总体摘要：\n{summary[:1500]}")
    else:
        instr = (f"Task: the items below are chunk-extracted "
                 f"{zh_names.get(key, key)} candidates from a long video — "
                 f"possibly duplicated, out of order, or split across chunk "
                 f"boundaries. Using the overall summary for context, merge "
                 f"duplicates, restore a sensible order, and keep the top {cap}. "
                 f"Keep each line's format unchanged; preserve [mm:ss] timestamps "
                 f"verbatim. Output the final list only.\n\n"
                 f"Overall summary:\n{summary[:1500]}")
    try:
        resp = ask_llm(host, model, instr + "\n\n候选条目:\n" + "\n".join(items))
        cleaned = _strip_fence(resp or "")
        out_lines = [l for l in cleaned.splitlines() if l.strip()]
        # sanity: a rerank must not balloon, come back empty, or echo prose
        if 1 <= len(out_lines) <= line_budget + 6 and \
                sum(1 for l in out_lines if l.strip().startswith(("-", "Q:", "A:"))) \
                >= len(out_lines) * 0.6:
            return "\n".join(out_lines)
    except Exception:
        pass
    return None


def _line_ts(line: str) -> int | None:
    """Timestamp in seconds of a `- [mm:ss] ...` list line, else None."""
    m = re.search(r"\[(\d+):(\d{2})\]", line)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _even_spread_ts(items: list[str], budget: int) -> str:
    """Deterministic time-ordered even sampling of timestamped list lines.

    Guards the LLM re-rank, which occasionally returns only early items
    (observed: 24-min lecture timeline stopping at 04:18 while the merged
    pool reached much further). Keeps first and last, spreads between.
    """
    ts_items = sorted(((t, l) for l in items if (t := _line_ts(l)) is not None))
    if not ts_items:
        return ""
    if len(ts_items) <= budget:
        return "\n".join(l for _, l in ts_items)
    step = (len(ts_items) - 1) / (budget - 1)
    return "\n".join(ts_items[round(j * step)][1] for j in range(budget))


def build_analysis(host: str, model: str | None, raw_text: str, source: str,
                   lang: str = "zh", char_limit: int = 0,
                   cache: _LLMCache | None = None) -> dict:
    """Ask the LLM for summary / timeline / key points / QA / glossary.

    Strategy: call the model once PER field with a narrow, plain-markdown prompt.
    Small local models follow simple single-task prompts far more reliably than a
    single complex JSON request. Falls back to a heuristic if no model is reachable
    or a field comes back empty.

    `lang` selects the prompt language ("zh" or "en") — match it to the video's
    language so a Chinese-centric 1B model does not hallucinate cross-language
    content when summarizing foreign-language subtitles.

    `char_limit` bounds the raw_text fed to the model. It is NOT a coverage
    knob: passing a value below the input length drops the tail of the lecture
    silently (a warning is printed when it happens). The default leaves the text
    intact and lets the map-reduce path chunk it, so coverage is complete and
    the cap only bounds per-call context. Pass a lower value to trade
    completeness for speed.
    """
    fields = {
        "summary": "", "timeline": "", "key_points": "",
        "qa": "", "glossary": "", "bullets": "",
    }
    if not (model and ping(host)):
        return _heuristic_fallback(raw_text, fields)

    ask = cache.ask if cache else (lambda h, prompt: ask_llm(h, model, prompt))

    # Do NOT truncate here. The map-reduce machinery below already handles long
    # input by chunking into CHUNK-sized pieces and merging, which keeps each
    # LLM call on a context the small models handle well. Slicing to a small
    # char_limit first silently threw away the rest of the lecture AND — because
    # long_mode needs len(sub) > CHUNK * 1.5 — made the map-reduce branch
    # unreachable in the plain (non-merged) Path 2 flow. Measured on a 35-minute
    # lesson: 12241 chars of subtitle, 8000 reached the model, 35% of the
    # content never seen, no warning.
    if char_limit and len(raw_text) > char_limit:
        print(f"[v2k] WARNING: char_limit={char_limit} truncated "
              f"{len(raw_text) - char_limit} of {len(raw_text)} subtitle chars "
              f"— pass a larger --char-limit for full coverage.",
              file=sys.stderr)
    sub = raw_text[:char_limit] if char_limit else raw_text
    # Bilingual prompt sets. Match `lang` to the video's language to stop a
    # Chinese-centric small model from hallucinating cross-language content.
    TASKS = {
        "zh": [
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
             "- 示例：`- [01:20] 讲者引出主题`"),
            ("key_points",
             "任务：从下面这段视频字幕中，提炼最多8个核心知识点/操作要点。\n"
             "要求：\n"
             "- 每行格式 `- 知识点(一句话概括)`\n"
             "- 要点是提炼后的结论，禁止照抄字幕原句\n"
             "- 只输出列表\n"
             "- 示例：`- 明确目标能提升专注力与成效`"),
            ("bullets",
             "任务：下面这段视频材料中，🎙️ 开头的是讲师原声字幕，🖼️ 开头的是同时刻"
             "幻灯片画面内容（文字/表格/公式）。请把两侧信息融合，输出最多10条要点速览，"
             "每条同时体现画面上的知识与讲解中的补充。\n"
             "要求：\n"
             "- 每行格式 `- **<主题词>** 画面：<幻灯片要点>｜讲解：<讲师的关键补充/案例/提醒> [mm:ss]`\n"
             "- 画面与讲解各不超过25字，取实质内容；材料中没有 🖼️ 行时省略画面部分，"
             "格式改为 `- **<主题词>** <讲解要点> [mm:ss]`\n"
             "- 时间戳取该要点出现的时刻，原样保留\n"
             "- 只输出列表，不要前缀\n"
             "- 示例：`- **可靠性** 画面：完整/中立/准确三要素｜讲解：合理估计不等于不准确 [01:07]`"),
            ("qa",
             "任务：基于下面这段视频字幕，设计6到10组中文问答，用于学习测试。\n"
             "严格要求（必须遵守）：\n"
             "- 每组两行：第一行 `Q: 问题`，第二行 `A: 答案`\n"
             "- 问题和答案都必须能从字幕中找到依据，禁止编造字幕里没有的内容\n"
             "- 问题用疑问句(如\"如何...?\"\"...是什么?\"\"在哪里...?\")\n"
             "- 答案用自然语言回答，不要照抄字幕原句，但内容必须忠于字幕\n"
             "- 只输出问答，不要前缀、不要编号、不要解释\n"
             "- 示例（通用示例，非字幕内容）：\n"
             "  Q: 这段话的核心观点是什么?\n"
             "  A: 协作是成就伟业的关键。"),
            ("glossary",
             "任务：从下面这段视频字幕中，提取最重要的术语/专有名词。\n"
             "要求：\n"
             "- 每行格式 `- 术语`，术语必须是字幕中真实出现的名词\n"
             "- 只保留名词性术语，不要整句、不要动词短语\n"
             "- 禁止输出字幕里没有的内容，禁止照抄示例\n"
             "- 只输出列表\n"
             "- 示例格式（仅示意格式，不要输出此内容）：`- 光合作用`"),
        ],
        "en": [
            ("summary",
             "Task: Read the video subtitles below, then write a 3-5 sentence English "
             "summary of the core content and purpose.\n"
             "Rules:\n"
             "- Paraphrase in your own words; do NOT copy subtitle sentences verbatim\n"
             "- Output only the summary body, no heading, no preamble\n"
             "- Example style: \"This video presents X. It highlights A, B, and C, and "
             "notes the caveat D, helping the viewer quickly understand Y.\""),
            ("timeline",
             "Task: Extract up to 8 key events / milestones from the subtitles below.\n"
             "Rules:\n"
             "- Each line strictly formatted as `- [mm:ss] concise event (<=12 words)`\n"
             "- Paraphrase the event, do NOT copy subtitle sentences\n"
             "- Output only the list, no preamble\n"
             "- Example: `- [01:20] the speaker introduces the main topic`"),
            ("key_points",
             "Task: Extract up to 8 core takeaways / points from the subtitles below.\n"
             "Rules:\n"
             "- Each line formatted as `- key point (one short sentence)`\n"
             "- Points must be distilled conclusions, NOT verbatim subtitle lines\n"
             "- Output only the list\n"
             "- Example: `- clear goals improve focus and outcomes`"),
            ("bullets",
             "Task: In the material below, lines starting with 🎙️ are the speaker's "
             "narration and lines with 🖼️ are the on-screen slide content (text/tables/"
             "formulas) at that moment. Fuse both sides into up to 10 quick-reference "
             "bullets, each combining the slide knowledge with the speaker's addition.\n"
             "Rules:\n"
             "- Each line formatted as `- **<topic>** slide: <slide point> | talk: "
             "<speaker's key addition/case/caveat> [mm:ss]`\n"
             "- Keep each side under ~20 words of substance; when no 🖼️ lines exist, "
             "drop the slide part and use `- **<topic>** <point> [mm:ss]`\n"
             "- Keep the timestamp verbatim from the material\n"
             "- Output only the list\n"
             "- Example: `- **reliability** slide: complete/neutral/accurate | talk: "
             "reasonable estimates are not inaccurate [01:07]`"),
            ("qa",
             "Task: Based on the subtitles below, design 6 to 10 English Q&A pairs "
             "for a study quiz.\n"
             "Strict rules (must follow):\n"
             "- Two lines per pair: first `Q: question`, then `A: answer`\n"
             "- Both question and answer must be grounded in the subtitles; do NOT "
             "invent content not present in the subtitles\n"
             "- Questions must be interrogative (\"How...?\", \"What is...?\", \"Where...?\")\n"
             "- Answers in natural language, paraphrased but faithful to the subtitles\n"
             "- Output only Q&A pairs, no preamble, no numbering, no commentary\n"
             "- Example (generic, not from subtitles):\n"
             "  Q: What is the main message of the talk?\n"
             "  A: It argues that collaboration drives great achievements."),
            ("glossary",
             "Task: Extract the most important terms / proper nouns from the subtitles below.\n"
             "Rules:\n"
             "- Each line formatted as `- term`; the term must be a real noun appearing in the subtitles\n"
             "- Keep only noun-like terms (1-3 words), not full sentences or verb phrases\n"
             "- Do NOT output anything not present in the subtitles; do NOT copy the example\n"
             "- Output only the list\n"
             "- Example format (illustrating format only — do not output it): `- photosynthesis`"),
        ],
    }
    tasks = TASKS.get(lang, TASKS["en"])
    sublabel = "字幕" if lang == "zh" else "subtitles"
    # Map-reduce for long context: list-type fields (timeline/key_points/qa/glossary)
    # are extracted per chunk then merged+deduped, then re-ranked in one global
    # pass with the summary as context (restores cross-chunk order). This keeps
    # each LLM call on a small, accurate context (small/mid models degrade badly
    # on 18k+ char inputs — repeated output, dropped items, bad timestamps).
    # summary is map(summarize)->reduce(summarize).
    # 9000 (not 4500): the default text models (minicpm5-2b, qwen3.5:4b) handle
    # 9k single-task prompts cleanly and it halves the call count, so a 3h
    # lecture (~360k chars, 40 chunks) is tractable while keeping full coverage.
    CHUNK = 9000
    long_mode = len(sub) > CHUNK * 1.5
    chunks = _chunk_lines(sub, CHUNK) if long_mode else [sub]
    LIST_FIELDS = {"timeline", "key_points", "qa", "glossary", "bullets"}
    CAPS = _caps_for(sub)
    for key, instruction in tasks:
        try:
            if long_mode and key in LIST_FIELDS:
                collected = []
                for ch in chunks:
                    resp = ask(host, instruction + f"\n\n{sublabel}:\n" + ch)
                    if resp and len(resp.strip()) > 3:
                        collected.extend(_strip_echo(
                            [l for l in _strip_fence(resp).splitlines() if l.strip()]))
                if key == "qa":
                    pairs = _dedupe_qa_pairs(_parse_qa_pairs(collected))
                    items = [f"Q: {q}\nA: {a}" for q, a in pairs]
                    budget = CAPS[key] * 2  # 2 lines per pair
                else:
                    items = _merge_list_items(collected, 100_000).splitlines()
                    budget = CAPS[key]
                # global re-rank (summary is computed first, so it is ready):
                # restores cross-chunk order and merges chunk-boundary dupes
                reranked = _rerank_list(host, model, key, items,
                                        fields.get("summary", ""), lang,
                                        CAPS[key], budget)
                if key == "timeline" and reranked:
                    # guard: if the re-rank clustered early (max ts < 60% of
                    # the pool's), fall back to deterministic even spread
                    pool_max = max((t for t in (_line_ts(l) for l in items)
                                    if t is not None), default=None)
                    rk_max = max((t for t in (_line_ts(l) for l in reranked.splitlines())
                                  if t is not None), default=None)
                    if pool_max and rk_max is not None and rk_max < pool_max * 0.6:
                        spread = _even_spread_ts(items, CAPS[key])
                        if spread:
                            reranked = spread
                if reranked:
                    fields[key] = reranked
                elif key == "qa":
                    fields[key] = "\n".join(f"Q: {q}\nA: {a}"
                                            for q, a in pairs[:CAPS[key]])
                else:
                    merged = _merge_list_items(items, CAPS[key])
                    fields[key] = merged if merged else _heuristic_fallback(raw_text, {key: ""})[key]
            elif long_mode and key == "summary":
                # summarize each chunk, then summarize the concatenation
                parts = []
                for ch in chunks:
                    resp = ask(host, instruction + f"\n\n{sublabel}:\n" + ch)
                    if resp and len(resp.strip()) > 3:
                        parts.append(_strip_fence(resp))
                joined = "\n".join(parts)
                reduce_instr = ("任务：下面是一段视频各部分的摘要，请融合成一段3-5句话的总体总结，"
                                "保留关键信息，去掉重复，只输出总结正文。") if lang == "zh" else (
                    "Task: the items below are summaries of parts of a video. Fuse them "
                    "into one 3-5 sentence overall summary, keeping key info, dropping "
                    "duplicates. Output only the summary.")

                def _red(text: str) -> str:
                    r = ask(host, reduce_instr + f"\n\n{sublabel}:\n" + text)
                    return _strip_fence(r) if r and r.strip() else ""

                # hierarchical reduce: a 3h lecture makes ~40 part-summaries
                # (>16k chars); reduce in levels of ~12 so no tail is dropped
                level = parts[:]
                while len(joined) > 16000 and len(level) > 1:
                    nxt = []
                    for i in range(0, len(level), 12):
                        grp = "\n".join(level[i:i + 12])
                        nxt.append(_red(grp) or grp[:400])
                    if len(nxt) >= len(level):
                        break
                    level = nxt
                    joined = "\n".join(level)
                resp = ask(host, reduce_instr + f"\n\n{sublabel}:\n" + joined[:12000])
                fields[key] = _strip_fence(resp) if resp and resp.strip() else (
                    joined[:500] or _heuristic_fallback(raw_text, {key: ""})[key])
            else:
                resp = ask(host, instruction + f"\n\n{sublabel}:\n" + sub)
                body = _strip_fence(resp) if resp and len(resp.strip()) > 3 else ""
                if body and key in LIST_FIELDS:
                    body = "\n".join(_strip_echo(body.splitlines()))
                fields[key] = body or _heuristic_fallback(raw_text, {key: ""})[key]
        except Exception:
            fields[key] = _heuristic_fallback(raw_text, {key: ""})[key]
        finally:
            if cache:
                cache.flush()  # crash-safe: each field's chunk calls persist
    # last-line defense: prompt fragments can be re-leaked by ANY llm stage
    # (observed: the re-rank call echoing the key_points instructions verbatim
    # even though the chunk-collection stage had already been filtered)
    for key in LIST_FIELDS:
        if fields.get(key):
            fields[key] = "\n".join(_strip_echo(fields[key].splitlines()))
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
        "bullets": "- (启用本地模型可生成音画合并要点速览)",
    }
    for k in fields:
        if not fields.get(k):
            fields[k] = defaults.get(k, "")
    return fields


# The one string that marks a document as fallback text rather than output.
# Exported as a constant so a caller (batch driver, resume logic) can test for
# degradation instead of pattern-matching Chinese prose.
DEGRADED_MARKER = "本地模型不可用"


def is_degraded(analysis: dict) -> bool:
    """True when the document is mostly placeholder rather than generated.

    A knowledge doc built with no reachable model still looks like a finished
    deliverable: right filename, right sections, plausible length, exit code 0.
    In a batch that is the worst possible failure mode — it reads as success.
    Detecting it by content means a driver can refuse to treat the run as done.
    """
    if not isinstance(analysis, dict):
        return True
    summary = analysis.get("summary")
    if not summary or not str(summary).strip():
        return True          # no summary at all is the same failure, quieter
    return DEGRADED_MARKER in str(summary)


# Emitted by main() when the transcript has no speech at all. Not a degradation —
# it is the honest answer — but a driver still has to be able to tell it apart
# from a real document.
NO_SPEECH_MARKER = "未识别到语音内容"

# What a driver should do with a finished knowledge doc.
STATUS_OK = "ok"                # real content
STATUS_DEGRADED = "degraded"    # model unreachable / returned nothing usable
STATUS_NO_SPEECH = "no-speech"  # transcript empty; placeholder is correct

# --- speech density ----------------------------------------------------------
#
# Zero segments is not the only way a transcript can be useless. On a
# piano-course library, 13 clips had an audio track but no narration — just
# performance. Paraformer does not return nothing for those; it hallucinates the
# playing into vocalisations, so the segment list is non-empty and the
# `n_segments == 0` guard never fires:
#
#     第27节  3 segments,  6 characters, 5 of them filler  ->  8 knowledge cards
#     第28节  1 segment,   6 characters, 2 of them filler  ->  9 knowledge cards
#     第19节 14 segments, 65 characters, 40 of them filler ->  0 cards
#
# Six characters of "嗯嗯嗯背谱。" cannot support eight cards. The model does not
# decline — it fills the template from the title and the topic words it can
# scrape out of the filler. That is the same fabrication the empty-transcript
# case was fixed for, one step further along: now it looks like a success.
#
# So the guard counts characters that could plausibly be words, after removing
# interjections and vocalisations. Measured on the library that separates
# cleanly with room on both sides: the worst junk transcript had 25 meaningful
# characters, the shortest genuine one had 63.
FILLER_CHARS = set("嗯啊哦呃啦哎呐嘛哼唔哈呵唉嘿咦噢诶喔呀噢唉嗯呃")
MIN_MEANINGFUL_CHARS = int(os.environ.get("V2K_MIN_SPEECH_CHARS", "40"))


def count_meaningful_chars(segs: list[dict]) -> tuple[int, int]:
    """Return (meaningful, total) character counts over a segment list.

    "Meaningful" = anything that is not punctuation/whitespace and not a bare
    interjection or vocalisation. Counting the run of syllables rather than
    filtering a fixed stopword list keeps this language-agnostic: the same
    function catches an English "The the the" just as it catches "嗯嗯嗯".
    """
    meaningful = total = 0
    for seg in segs:
        for ch in (seg.get("text") or ""):
            if ch.isspace() or not ch.isalnum():
                continue          # punctuation and whitespace carry no content
            total += 1
            if ch not in FILLER_CHARS:
                meaningful += 1
    return meaningful, total


def has_usable_speech(segs: list[dict]) -> bool:
    """True when the transcript holds enough real words to summarise."""
    if not segs:
        return False
    meaningful, total = count_meaningful_chars(segs)
    if not total:
        return False
    return meaningful >= MIN_MEANINGFUL_CHARS


def knowledge_doc_status(analysis: dict, n_segments: int,
                        segs: list[dict] | None = None) -> str:
    """Classify a finished knowledge doc so a batch driver can trust it.

    `n_segments` is the subtitle segment count the document was built from, and
    it is not optional decoration: it is the only thing that catches the worst
    failure mode, which leaves no marker to grep for.

    With an empty transcript there is no possible source for a real summary, and
    a small model does not decline — it invents one. Observed on a piano-course
    library: a lesson on improv accompaniment came back summarised as "how to
    send HTTP requests with Python's requests library", another as "how to use
    WeChat mini-programs", both with invented `- [00:03]` timestamps. The file
    was well-formed, exit code 0, and `is_degraded()` could not see it, because
    the summary string itself looked perfectly plausible.

    A batch driver treats only `ok` as finished. The other two are legitimate
    outcomes that must not be retried blindly or counted as deliverables.
    """
    if is_degraded(analysis):
        return STATUS_DEGRADED
    if n_segments == 0 or (segs is not None and not has_usable_speech(segs)):
        summary = str(analysis.get("summary", ""))
        return STATUS_NO_SPEECH if NO_SPEECH_MARKER in summary else STATUS_DEGRADED
    return STATUS_OK


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


def _card_key(question: str) -> str:
    """Normalise a question for duplicate detection.

    A model asked the same question twice rarely emits it twice identically —
    one answer may read "两个白键之间是全音" and the other "白键和相邻白键
    间隔是一个全音". Comparing the raw string would keep both. Dropping
    punctuation, whitespace and the interrogative scaffolding leaves the
    content words, which is what actually identifies the question.
    """
    s = re.sub(r"[\s，。、？?！!：:；;（）()【】\[\]「」“”\"'’·\-—…]+", "", question)
    return s.lower()


def _dedup_cards(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    """Keep the first card for each distinct question; count what was dropped.

    First-wins, so a card whose answer is complete keeps its slot and a later
    stub with the same question is discarded rather than overwriting it.
    """
    seen: set[str] = set()
    out: list[list[str]] = []
    dropped = 0
    for r in rows:
        key = _card_key(r[0] if r else "")
        if not key:
            dropped += 1          # a question-less card is not a study card
            continue
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


def cards_from_qa(qa, source: str) -> list[list[str]]:
    """Parse Q/A pairs into CSV rows. Accepts:
    - a markdown string with 'Q:' / 'A:' lines
    - a list of dicts with q/question and a/answer keys (LLM JSON output)
    - a list of strings like 'Q: ... A: ...'

    Duplicates are removed. A long lecture is summarised with a map/reduce
    over chunks, and the model re-asks the same question in several chunks
    because the same fact is restated throughout a lecture — measured on a
    286-video course, 145 of 1608 cards (9%) were repeats, and the worst case
    was one 8-minute lesson producing 67 rows for 6 distinct questions. An
    Anki deck of near-duplicates is worse than a short one: it looks like
    coverage while drilling the same fact over and over.
    """
    rows = _parse_qa_rows(qa, source)
    rows, _dropped = _dedup_cards(rows)
    return rows


def _parse_qa_rows(qa, source: str) -> list[list[str]]:
    rows: list[list[str]] = []
    if isinstance(qa, list):
        for item in qa:
            if isinstance(item, dict):
                q = item.get("q") or item.get("question") or item.get("Q") or ""
                a = item.get("a") or item.get("answer") or item.get("A") or ""
                if q:
                    rows.append([str(q).strip(), str(a).strip(), "", "", source])
            elif isinstance(item, str):
                rows.extend(_parse_qa_rows(item, source))
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
    ap.add_argument("--merged", type=Path, default=None,
                    help="merged.json from merge_visual.py (dual-path fusion). When set, "
                         "the LLM is fed audio+visual interleaved text and a visual "
                         "timeline section is added; --subtitles is still required for "
                         "titles/source but the merged file takes precedence for content.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="knowledge-doc template (default assets/default-template.md)")
    ap.add_argument("--format", choices=["knowledge", "html", "csv", "docx", "pdf", "all"],
                    default="all",
                    help="all = knowledge+html+csv (plus docx+pdf when python-docx/"
                         "fpdf2 are installed; explicit docx/pdf fail loudly "
                         "when the lib is missing)")
    ap.add_argument("--model", default=default_text_model(),
                    help="Ollama text model for summarization/QA (default: the "
                         "profile's text model — openbmb/minicpm5-2b on low/mid, "
                         "the VLM pull itself on high tiers; set V2K_TEXT_MODEL "
                         "or pass --model to override)")
    ap.add_argument("--title", default=None, help="document title (default: video basename)")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="prompt/output language for summary/QA (default zh; set to en "
                         "for English videos to avoid cross-language hallucination)")
    ap.add_argument("--char-limit", type=int, default=None,
                    help="max chars of raw text fed to the model (default 8000; auto-raised "
                         "in merged mode since interleaved audio+visual is the main signal)")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--api-base", default=os.environ.get("V2K_LLM_API_BASE", ""),
                    help="use an OpenAI-compatible /chat/completions endpoint "
                         "instead of local Ollama (default: unset = local). "
                         "Vendor-neutral: any compatible URL works.")
    ap.add_argument("--api-model", default=os.environ.get("V2K_LLM_API_MODEL", ""),
                    help="model name for --api-base (required with it)")
    ap.add_argument("--api-key-env", default=os.environ.get("V2K_LLM_API_KEY_ENV",
                                                            "OPENAI_API_KEY"),
                    help="env var holding the API key (default OPENAI_API_KEY). "
                         "The key is never read from argv.")
    args = ap.parse_args()

    if args.api_base and not args.api_model:
        print("[err] --api-base also needs --api-model (or $V2K_LLM_API_MODEL)",
              file=sys.stderr)
        return 2
    configure_cloud(args.api_base, args.api_model, args.api_key_env)

    if not args.subtitles.is_file():
        print(f"[err] subtitles not found: {args.subtitles}", file=sys.stderr)
        return 2
    if not args.template.is_file():
        print(f"[err] template not found: {args.template}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged_data = None
    visual_timeline = ""
    if args.merged:
        if not args.merged.is_file():
            print(f"[err] merged not found: {args.merged}", file=sys.stderr)
            return 2
        merged_data = load_merged(args.merged)
        mseg = merged_data["segments"]
        raw_text = build_interleaved_text(mseg)
        # merged text interleaves tables/formulas — raise the cap so they survive.
        # 400k covers a 3-hour lecture end to end (~360k interleaved chars at
        # measured course density). Map-reduce chunks the text so the cap only
        # bounds runtime, not coverage; lower it via --char-limit to trade
        # completeness for speed.
        char_limit = args.char_limit or 400000
        print(f"[v2k] merged mode: {len(mseg)} ASR segments, "
              f"{merged_data.get('used_visual', 0)}/{merged_data.get('visual_count', 0)} "
              f"visual frames"
              f" (swaps={merged_data.get('semantic_swaps', 0)},"
              f" weak={merged_data.get('weak_attribution', 0)})"
              f"; interleaved raw_text {len(raw_text)} chars, "
              f"cap {char_limit}; summarizing with {args.model}...", file=sys.stderr)
        visual_timeline = build_visual_timeline(
            merged_data.get("visual_blocks", []), args.host, args.model, lang=args.lang)
        # source name reflects fusion
        segs = mseg
        source = f"{args.merged.name} ({merged_data.get('asr_count','?')} ASR + {merged_data.get('visual_count','?')} visual)"
    else:
        segs, source = load_subtitles(args.subtitles)
        raw_text = "\n".join(f"[{fmt_mmss(s['start'])}] {s['text']}" for s in segs)
        # No default cap here. A 8000-char default (the old value) silently
        # dropped the tail of every lecture longer than ~25 minutes AND, because
        # long_mode needs len(sub) > CHUNK * 1.5, kept the map-reduce branch
        # unreachable on this path. 0 means "feed it all"; the chunker bounds
        # each individual call. --char-limit still overrides, for users who
        # would rather trade coverage for speed.
        char_limit = args.char_limit or 0

    title = args.title or args.subtitles.stem.replace("_", " ")
    duration = segs[-1]["end"] if segs else 0.0

    # Zero recognised speech: a piano-only demo, a silent screen recording, a
    # music bed. Calling the LLM here is actively harmful — a small model handed
    # an empty transcript does not return "nothing to summarise", it generates
    # until it hits the context ceiling. Measured: one such video produced no
    # output for 9+ minutes and, because ollama serialises requests per model,
    # blocked every later video in the batch behind it. It also fabricates a
    # confident summary of a video with no words in it, which is worse than
    # saying nothing.
    #
    # So: skip the model entirely and say so out loud, pointing at Path 1/3 —
    # the illustrated-notes / VLM path, which is the only one that can produce
    # anything for a wordless video.
    #
    # Test the segment text itself, not raw_text: every raw_text line carries a
    # "[mm:ss] " prefix, so a transcript of pure whitespace still looks non-empty
    # ("[00:00]    ".strip() is truthy) and the guard would silently not fire.
    spoken = "".join((s.get("text") or "").strip() for s in segs)
    # Path 3 is exempt from the density check: there the visual OCR text is a
    # legitimate content source, so a thin ASR transcript does not mean the
    # document would be empty.
    no_speech = (not segs or not spoken
                 or (not args.merged and not has_usable_speech(segs)))
    if no_speech:
        if segs and spoken:
            meaningful, total = count_meaningful_chars(segs)
            why = (f"only {meaningful} meaningful of {total} characters "
                   f"across {len(segs)} segments "
                   f"(threshold {MIN_MEANINGFUL_CHARS}) — that is instrumental "
                   f"audio transcribed as vocalisation, not narration")
        else:
            why = "0 segments"
        print(f"[v2k] no usable speech in {args.subtitles.name} "
              f"({why}) — skipping the LLM.\n"
              f"      This video has no usable narration; use Path 1/3 "
              f"(mm_caption.py / build_notes.py) instead.", file=sys.stderr)
        analysis = {
            "summary": "**未识别到语音内容。** 本视频没有可用的旁白/讲解，"
                       "因此无法生成文字总结。这类素材（纯演奏、纯演示、无人声录屏）"
                       "请改用 Path 1/3：先用 `mm_caption.py --mode dedup --prompt-ocr` "
                       "读画面，再用 `build_notes.py` 生成图文笔记。",
            "timeline": "_(无语音时间轴)_",
            "key_points": "- (无语音内容，无法提炼知识点)",
            "bullets": "- (无语音内容)",
            "qa": "- (无语音内容，无法生成问答)",
            "glossary": "- (无语音内容)",
        }
        llm_cache = None
    else:
        if not args.merged:
            print(f"[v2k] {len(segs)} segments, {duration:.0f}s; "
                  f"summarizing with {args.model}...", file=sys.stderr)
        llm_cache = _LLMCache(args.out_dir / "build_cache.json", args.model)
        analysis = build_analysis(args.host, args.model, raw_text, source,
                                  lang=args.lang, char_limit=char_limit,
                                  cache=llm_cache)
        llm_cache.flush()

    degraded = is_degraded(analysis)
    if degraded:
        print(f"[v2k] WARNING: '{args.model}' produced no usable output for "
              f"{args.subtitles.name} — every section is fallback text.",
              file=sys.stderr)

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
        "bullets": as_md(analysis.get("bullets", "")),
        "qa": as_md(analysis["qa"]),
        "glossary": as_md(analysis["glossary"]),
        "visual_timeline": visual_timeline or "(无视觉信息，使用纯ASR模式)",
        "meta": f"segments={len(segs)} model={args.model}"
               + (" +merged" if args.merged else ""),
    }

    want = {"knowledge", "html", "csv"} if args.format == "all" else {args.format}
    if args.format == "all":  # office/print exports are opt-out by absence of lib
        try:
            import docx  # noqa: F401
            want.add("docx")
        except ImportError:
            print("[warn] python-docx missing — skipping knowledge.docx "
                  "(pip install python-docx)", file=sys.stderr)
        try:
            import fpdf  # noqa: F401
            want.add("pdf")
        except ImportError:
            print("[warn] fpdf2 missing — skipping knowledge.pdf "
                  "(pip install fpdf2)", file=sys.stderr)

    if "knowledge" in want or "html" in want or "docx" in want or "pdf" in want:
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

    if "docx" in want:
        from md_export import md_to_docx
        p = md_to_docx(md, args.out_dir / "knowledge.docx", base_dir=args.out_dir)
        print(f"[ok] docx -> {p}")

    if "pdf" in want:
        from md_export import md_to_pdf
        p = md_to_pdf(md, args.out_dir / "knowledge.pdf",
                      base_dir=args.out_dir, title=title)
        print(f"[ok] pdf -> {p}")

    if "csv" in want or "all" in want:
        parsed = _parse_qa_rows(analysis["qa"], source)
        rows, dropped = _dedup_cards(parsed)
        csv_path = args.out_dir / "cards.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer", "tags", "timestamp", "source"])
            w.writerows(rows)
        note = f" ({dropped} duplicate/empty dropped)" if dropped else ""
        print(f"[ok] {len(rows)} cards{note} -> {csv_path}")

    if degraded:
        # The artifacts are still written — they are useful for eyeballing the
        # transcript — but this must not read as a completed run. A batch driver
        # that checks only the exit code will otherwise record hundreds of
        # placeholder documents as finished work.
        print(f"[err] KNOWLEDGE DOC IS DEGRADED: the text model "
              f"'{args.model}' produced nothing usable, so every section is "
              f"fallback text.\n"
              f"      The files in {args.out_dir} are NOT a finished deliverable.\n"
              f"      Check: is ollama running, and is '{args.model}' installed "
              f"(curl {args.host}/api/tags)? On a reachable-but-empty model "
              f"registry this looks like success and is not.",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

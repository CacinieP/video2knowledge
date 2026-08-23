#!/usr/bin/env python3
"""mm_caption.py — PATH 1 core: use an Ollama VLM to caption sampled video frames
into a timestamped subtitle document.

Pipeline:
    video --(ffmpeg)--> frames --(ollama /api/generate w/ image)--> per-frame caption
         --(merge by timestamp)--> SRT + structured JSON

This is the "native multimodal small model" path (<=4B VLM). It is the right choice
when the video has no usable audio track, is a slide/screen demo, or when you want
visual grounding that ASR alone cannot provide.

Usage:
    python3 mm_caption.py --video in.mp4 --out-dir out \\
        --model openbmb/minicpm-v4.6:latest --interval 2.0

Outputs (in --out-dir):
    captions.srt      # timestamped subtitles
    captions.json     # [{"start","end","text"}, ...]
"""
from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = (
    "Describe what is visible in this single video frame in concise Chinese "
    "(<=40 chars). Focus on key objects, on-screen text, actions, and any "
    "slides/diagrams. Do NOT show your reasoning chain. Output the caption ONLY, "
    "no preamble, no thinking tags, no English."
)

# OCR prompt: full-fidelity transcription of on-screen text/tables/formulas.
# Used when the video is a slide/PPT/screencast and ASR cannot capture the visual
# content (tables, charts, formulas, examples). Keeps Markdown table formatting and
# newlines (the default caption prompt collapses them and caps at 40 chars).
#
# Anti-pollution: the prompt deliberately avoids listing "title/body/list/table..."
# as a menu, because small VLMs echo that menu back as content. Instead it states
# the single goal (verbatim transcription) and the few format rules.
PROMPT_OCR = (
    "请逐字转写这张图片画面里所有可见的文字、数字和表格，原样输出，不要做任何总结、"
    "解释或补充。表格用 Markdown 格式（| 列 | 列 |）保留行列结构，其余按原文换行。"
    "只输出转写内容本身，不要复述本指令。"
)


def http_json(url: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ping(host: str) -> None:
    try:
        # GET /api/tags (POST is not allowed on this endpoint -> 405)
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
            json.loads(r.read().decode())
    except (urllib.error.URLError, OSError) as e:
        print(f"[err] cannot reach ollama at {host} ({e}). "
              f"Start it with: ollama serve", file=sys.stderr)
        sys.exit(2)


def caption_frame(host: str, model: str, jpg: Path, prompt: str, ocr: bool) -> str:
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    # think:false disables the reasoning chain on thinking models (MiniCPM, Qwen3, ...)
    # so .response holds only the final answer.
    # num_predict is raised for OCR so full tables/formulas are not truncated; the
    # default caption mode keeps 160 (short captions).
    num_predict = 1024 if ocr else 160
    resp = http_json(
        f"{host}/api/generate",
        {"model": model, "prompt": prompt, "images": [b64],
         "stream": False, "think": False,
         "options": {"temperature": 0.2, "num_predict": num_predict}},
    )
    text = resp.get("response", "").strip()
    # Fallback cleanup in case the model still leaks a reasoning chain.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    # Caption mode collapses newlines (one-line subtitles); OCR mode preserves
    # newlines so Markdown tables survive.
    if not ocr:
        text = text.replace("\n", " ")
    return text


def frames_from(video: Path, out_dir: Path, interval: float, mode: str,
                dedup_fps: float, dedup_hamming: int, max_frames: int,
                hash_size: int = 8, hash_mode: str = "dhash") -> Path:
    """Delegate frame extraction to extract_frames.py (same dir)."""
    here = Path(__file__).resolve().parent
    import subprocess
    cmd = [sys.executable, str(here / "extract_frames.py"),
           "--video", str(video), "--out-dir", str(out_dir / "frames"),
           "--mode", mode]
    if mode == "interval":
        cmd += ["--interval", str(interval)]
    else:  # dedup
        cmd += ["--dedup-fps", str(dedup_fps),
                "--dedup-hamming", str(dedup_hamming),
                "--max-frames", str(max_frames),
                "--hash-size", str(hash_size),
                "--hash-mode", hash_mode]
    subprocess.run(cmd, check=True)
    return out_dir / "frames" / "frames.json"


def _norm_ocr(text: str) -> str:
    """OCR text for similarity: drop ALL whitespace (line breaks and wrap
    points differ between runs) and case."""
    return "".join(text.split()).casefold()


def ocr_texts_differ(prev: str, cur: str, threshold: float = 0.9) -> bool:
    """True when `cur` carries new on-screen knowledge vs `prev`.

    Visual change is not knowledge change: dHash keeps frames for cursor
    blinks, partial redraws and animation noise, and the VLM OCR of two such
    frames is near-identical. This gate (used in --prompt-ocr runs) drops a
    frame whose OCR text is >= `threshold` similar to the last KEPT frame, so
    every emitted caption carries new text and wasted VLM calls disappear.
    """
    a, b = _norm_ocr(prev), _norm_ocr(cur)
    if not a or not b:
        return bool(a) != bool(b)
    return difflib.SequenceMatcher(None, a, b).ratio() < threshold


def to_srt(records: list[dict]) -> str:
    def fmt(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, r in enumerate(records, 1):
        blocks.append(
            f"{i}\n{fmt(r['start'])} --> {fmt(r['end'])}\n{r['text']}\n"
        )
    return "\n".join(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal captioning via Ollama VLM")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="openbmb/minicpm-v4.6:latest")
    ap.add_argument("--mode", choices=["interval", "dedup"], default="interval",
                    help="frame sampling mode (default interval; use dedup for "
                         "slide/PPT/screencast videos)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--dedup-fps", type=float, default=1.0,
                    help="dense sample rate in dedup mode (default 1.0)")
    ap.add_argument("--dedup-hamming", type=int, default=10,
                    help="dHash change threshold in dedup mode (default 10)")
    ap.add_argument("--hash-size", type=int, default=8,
                    help="dHash size n -> n*n bits (default 8 = 64 bits)")
    ap.add_argument("--hash-mode", choices=["dhash", "dual"], default="dhash",
                    help="dual = dHash+aHash for flat/gradient content dHash "
                         "cannot see (scale --dedup-hamming ~2x)")
    ap.add_argument("--max-frames", type=int, default=360,
                    help="frame cap in dedup mode (default 360, covers a 3h "
                         "lecture; cluster-stratified when exceeded)")
    ap.add_argument("--prompt-ocr", action="store_true",
                    help="use the table/formula OCR prompt instead of the short "
                         "caption prompt (for slide/PPT videos where ASR misses "
                         "on-screen text/tables/examples)")
    ap.add_argument("--no-ocr-dedup", action="store_true",
                    help="disable the OCR text-change gate (default: with "
                         "--prompt-ocr, a frame whose OCR text is >= 90% "
                         "similar to the last kept frame is dropped — visual "
                         "change without knowledge change)")
    ap.add_argument("--ocr-dedup-threshold", type=float, default=0.9,
                    help="similarity ratio above which two OCR texts count as "
                         "the same slide (default 0.9)")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ping(args.host)

    manifest = frames_from(args.video, args.out_dir, args.interval, args.mode,
                           args.dedup_fps, args.dedup_hamming, args.max_frames,
                           hash_size=args.hash_size, hash_mode=args.hash_mode)
    frames = json.loads(manifest.read_text())["frames"]
    prompt = PROMPT_OCR if args.prompt_ocr else PROMPT
    ocr_gate = args.prompt_ocr and not args.no_ocr_dedup
    print(f"[mm] {len(frames)} frames to caption with {args.model}"
          f" ({'OCR' if args.prompt_ocr else 'caption'} prompt"
          f"{', text-dedup gate' if ocr_gate else ''})", file=sys.stderr)

    captions = []
    dropped = 0
    prev_text: str | None = None
    last_t = frames[-1]["t"] if frames else 0.0
    for i, fr in enumerate(frames):
        t0 = time.time()
        text = caption_frame(args.host, args.model, Path(fr["file"]), prompt, args.prompt_ocr)
        elapsed = time.time() - t0
        if ocr_gate and prev_text is not None and \
                not ocr_texts_differ(prev_text, text, args.ocr_dedup_threshold):
            dropped += 1
            print(f"[mm] {i+1}/{len(frames)} @ {fr['t']:.1f}s: OCR text unchanged"
                  f" — dropped (visual change, no knowledge change)", file=sys.stderr)
            continue
        captions.append({"start": fr["t"], "end": None, "text": text})
        prev_text = text
        preview = text.replace("\n", " ")[:60]
        print(f"[mm] {i+1}/{len(frames)} @ {fr['t']:.1f}s ({elapsed:.1f}s): {preview}",
              file=sys.stderr)

    # Each kept caption runs until the next KEPT frame (covering dropped
    # duplicates' time); the last one runs `interval` past the final sample.
    for i, c in enumerate(captions):
        c["end"] = captions[i + 1]["start"] if i + 1 < len(captions) \
            else last_t + args.interval
    if dropped:
        print(f"[mm] OCR gate dropped {dropped} near-duplicate frame(s)", file=sys.stderr)

    (args.out_dir / "captions.srt").write_text(to_srt(captions), encoding="utf-8")
    (args.out_dir / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {args.out_dir/'captions.srt'} and captions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

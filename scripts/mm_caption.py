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


def http_json(url: str, payload: dict, timeout: int = 120) -> dict:
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


def caption_frame(host: str, model: str, jpg: Path) -> str:
    b64 = base64.b64encode(jpg.read_bytes()).decode()
    # think:false disables the reasoning chain on thinking models (MiniCPM, Qwen3, ...)
    # so .response holds only the final answer.
    resp = http_json(
        f"{host}/api/generate",
        {"model": model, "prompt": PROMPT, "images": [b64],
         "stream": False, "think": False,
         "options": {"temperature": 0.2, "num_predict": 160}},
    )
    text = resp.get("response", "").strip().replace("\n", " ")
    # Fallback cleanup in case the model still leaks a reasoning chain.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    return text


def frames_from(video: Path, out_dir: Path, interval: float) -> Path:
    """Delegate frame extraction to extract_frames.py (same dir)."""
    here = Path(__file__).resolve().parent
    import subprocess
    subprocess.run(
        [sys.executable, str(here / "extract_frames.py"),
         "--video", str(video), "--out-dir", str(out_dir / "frames"),
         "--interval", str(interval)],
        check=True,
    )
    return out_dir / "frames" / "frames.json"


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
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ping(args.host)

    manifest = frames_from(args.video, args.out_dir, args.interval)
    frames = json.loads(manifest.read_text())["frames"]
    print(f"[mm] {len(frames)} frames to caption with {args.model}", file=sys.stderr)

    captions = []
    for i, fr in enumerate(frames):
        t0 = time.time()
        text = caption_frame(args.host, args.model, Path(fr["file"]))
        elapsed = time.time() - t0
        next_t = frames[i + 1]["t"] if i + 1 < len(frames) else fr["t"] + args.interval
        captions.append({"start": fr["t"], "end": next_t, "text": text})
        print(f"[mm] {i+1}/{len(frames)} @ {fr['t']:.1f}s ({elapsed:.1f}s): {text}", file=sys.stderr)

    (args.out_dir / "captions.srt").write_text(to_srt(captions), encoding="utf-8")
    (args.out_dir / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {args.out_dir/'captions.srt'} and captions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

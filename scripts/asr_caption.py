#!/usr/bin/env python3
"""asr_caption.py — PATH 2 core: faster-whisper transcription with word-level
timestamps, emitting SRT/VTT/JSON subtitles.

This is the recommended path when the video has a clear speech track (lectures,
talks, interviews): it is faster and more accurate than visual captioning.

Default model is 'small' (good fit for 8GB-RAM machines). Upgrade to
'medium' or 'large-v3' on >=16GB RAM via --model.

Usage:
    python3 asr_caption.py --video in.mp4 --out-dir out \\
        --model small --language zh

Outputs (in --out-dir):
    subtitles.srt   # timestamped subtitles
    subtitles.vtt   # WebVTT
    subtitles.json  # segments with start/end/text (consumed by build_knowledge.py)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_WARN = {"large", "large-v1", "large-v2", "large-v3", "medium"}


def extract_wav(video: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio_16k.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", str(wav)],
        check=True,
    )
    return wav


def fmt_ts(sec: float, sep: str = ",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segs: list[dict]) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        out.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text'].strip()}\n")
    return "\n".join(out)


def to_vtt(segs: list[dict]) -> str:
    body = "\n".join(
        f"{fmt_ts(s['start'], sep='.')} --> {fmt_ts(s['end'], sep='.')}\n{s['text'].strip()}\n"
        for s in segs
    )
    return "WEBVTT\n\n" + body


def main() -> int:
    ap = argparse.ArgumentParser(description="faster-whisper ASR -> timestamped subtitles")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="small",
                    help="whisper model size: tiny/base/small/medium/large-v3 (default small)")
    ap.add_argument("--language", default="zh", help="language code or 'auto'")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, safest) | cuda | auto")
    ap.add_argument("--compute-type", default="int8",
                    help="int8 (default) | int8_float16 | float16 | float32")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2
    if args.model in MODEL_WARN and os_total_ram_gb() < 16:
        print(f"[warn] model '{args.model}' may exhaust RAM on <16GB machines. "
              f"Consider --model small.", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[asr] extracting 16k mono wav...", file=sys.stderr)
    wav = extract_wav(args.video, args.out_dir)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[err] faster-whisper not installed. Run scripts/setup_models.sh first.",
              file=sys.stderr)
        return 3

    print(f"[asr] loading model '{args.model}' on {args.device} ({args.compute_type})...",
          file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    language = None if args.language == "auto" else args.language
    print(f"[asr] transcribing (language={language or 'auto'})...", file=sys.stderr)
    segs_iter, info = model.transcribe(
        str(wav), language=language, beam_size=5, word_timestamps=True,
        vad_filter=True,
    )

    segs = []
    for s in segs_iter:
        segs.append({"start": round(s.start, 3), "end": round(s.end, 3),
                     "text": s.text.strip()})

    (args.out_dir / "subtitles.srt").write_text(to_srt(segs), encoding="utf-8")
    (args.out_dir / "subtitles.vtt").write_text(to_vtt(segs), encoding="utf-8")
    (args.out_dir / "subtitles.json").write_text(
        json.dumps({"language": info.language, "language_probability": info.language_probability,
                    "duration": info.duration, "segments": segs},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] {len(segs)} segments -> {args.out_dir}/subtitles.{{srt,vtt,json}}")
    return 0


def os_total_ram_gb() -> float:
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / 1073741824
    except Exception:
        return 16.0  # assume safe


if __name__ == "__main__":
    raise SystemExit(main())

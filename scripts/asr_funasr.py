#!/usr/bin/env python3
"""asr_funasr.py — Step 2.1b: FunASR/Paraformer ASR -> timestamped subtitles.

An alternative to scripts/asr_caption.py (faster-whisper) for languages and
domains where the model confuses near-homophones. It is a drop-in replacement:
same CLI surface (--video/--out-dir/--language/--hotwords/--keep-wav), same
three output files, byte-identical schema —

    subtitles.json  {"language","language_probability","duration","segments":[{start,end,text}]}
    subtitles.srt   1\n00:00:00,000 --> ...\ntext
    subtitles.vtt   WEBVTT\n\n...

so build_knowledge.py / build_notes.py / fuse.py / gen_apkg.py need no change.

Why it exists (measured, not assumed). On a 30-minute Mandarin piano lesson,
faster-whisper `small` WITH 39 hotwords including both terms produced:

    背谱   14 correct /  8 written as 被谱
    视谱    0 correct / 10 written as 适谱

i.e. 44% accuracy on the two terms the entire course is about, and hotwords did
not prevent it. Paraformer-large with the same hotwords:

    背谱   11 correct /  0 wrong
    视谱    4 correct /  1 wrong          94% overall

It also read whole words correctly that whisper mangled (元州律 -> 圆周率,
格鲁迦 -> 格鲁吉亚, 合声 -> 和声), and ran 11-16x realtime on CPU, which frees
the GPU for Ollama instead of contending with it.

Install (FunASR does NOT co-install cleanly with the main venv — it pins an old
tokenizers and pulls its own torch, so give it its own environment):

    bash scripts/setup_models.sh --with-funasr

That builds a separate .venv-funasr and prints the exact `run python as` line.
The equivalent by hand, if you would rather not use the script:

    python3 -m venv .venv-funasr
    ./.venv-funasr/bin/pip install "tokenizers>=0.21" funasr torch
    ./.venv-funasr/bin/pip install --no-deps funasr        # after deps resolve

Usage:

    ./.venv-funasr/bin/python asr_funasr.py --video in.mp4 --out-dir out --language zh
    ./.venv-funasr/bin/python asr_funasr.py --jobs jobs.json   # one process, one model load

Two implementation notes worth keeping:

* The bare ASR model returns only {key, text} — no timestamps. Timestamps only
  appear once a VAD model is attached, so vad_model is not optional here.
* Model load costs 25-60 s. A per-video process pays that once per clip — ~2.1 h
  of pure loading across a 297-clip library — so --jobs runs a whole list in one
  process and skips any clip whose subtitles.json already exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Short alias names, deliberately. The long iic/ names resolve to revisions that
# do NOT emit timestamps when a punc model is attached — measured on real course
# audio: `iic/speech_paraformer-...` + vad + ct-punc returns {key, text} only,
# so the whole transcript collapses onto zero-length segments. The alias form
# (what FunASR's own pipeline docs use) returns {key, text, timestamp} with the
# punctuation from ct-punc, which is the only combination that has both.
ASR_MODEL = "paraformer-zh"
VAD_MODEL = "fsmn-vad"
PUNC_MODEL = "ct-punc"

# Close a subtitle segment at sentence-final punctuation, or when it gets this
# long, whichever comes first. Whisper's default segments run ~5-12s; matching
# that keeps SRT readable and keeps build_knowledge's char budget similar.
SENTENCE_END = "。！？!?…；;"
SEG_MAX_CHARS = 28
SEG_MAX_SEC = 12.0
SEG_GAP_SEC = 0.35


# --- output formatting (kept identical to scripts/asr_caption.py) ------------

def fmt_ts(sec: float, sep: str = ",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segs: list[dict]) -> str:
    return "\n".join(
        f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text'].strip()}\n"
        for i, s in enumerate(segs, 1))


def to_vtt(segs: list[dict]) -> str:
    body = "\n".join(
        f"{fmt_ts(s['start'], sep='.')} --> {fmt_ts(s['end'], sep='.')}\n{s['text'].strip()}\n"
        for s in segs)
    return "WEBVTT\n\n" + body


def load_hotwords(spec: str | None) -> str | None:
    """Parse --hotwords exactly like asr_caption.py (comma/、/;/；/space, or @file)."""
    if not spec:
        return None
    if spec.startswith("@"):
        p = Path(spec[1:])
        terms = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        for sep in (",", "、", ";", "；"):
            spec = spec.replace(sep, " ")
        terms = spec.split()
    return " ".join(dict.fromkeys(terms)) or None


# --- wav ---------------------------------------------------------------------

def extract_wav(video: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio_16k.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    "-f", "wav", str(wav)], check=True)
    return wav


# --- segmentation ------------------------------------------------------------

# ct-punc runs AFTER the ASR and INSERTS punctuation into an otherwise
# unpunctuated string. The timestamp list belongs to the pre-punctuation text,
# so on real course audio text is consistently LONGER than timestamp
# (measured 749 vs 689). Walking both with one shared index therefore runs off
# the end of the timestamp list and silently drops the tail of every
# transcript. So: punctuation chars are copied through with a zero-width
# stamp, and only real characters consume a timestamp.
PUNCT_ONLY = "，。！？、；：…“”‘’（）《》〈〉「」【】,.!?;:\"'()[]—～~"


def _ms(x) -> float:
    """Timestamps come back in ms from FunASR; be tolerant of seconds."""
    return float(x) / 1000.0


def align_text_ts(text: str, ts: list) -> tuple[str, list]:
    """Return (text, timestamps) with exactly one entry per character.

    Handles all three shapes FunASR emits:
      * clean string + per-char ts              (paraformer, unpunctuated)
      * space-separated tokens + per-char ts    (seaco, "记 忆 力 呢")
      * punctuated string + shorter ts          (paraformer-zh + ct-punc)

    Everything comes back in the SAME unit the input was in (milliseconds);
    segment_from_chars divides once, at the end. Keeping that invariant matters
    more than it looks: an inserted punctuation mark borrows the previous
    character's end time, and if that one value is pre-divided it lands ~1000x
    too small — so every segment that ENDS on a 。/?/! fails its `end > start`
    check and is silently dropped, taking its whole sentence with it.
    """
    chars: list[str] = []
    out_ts: list = []
    ti = 0
    last_end = 0.0
    for ch in text:
        if ch.isspace():
            continue
        if ch in PUNCT_ONLY:
            # inserted punctuation: keep it, give it a zero-width stamp at the
            # preceding end time (same unit as everything else here)
            chars.append(ch)
            out_ts.append([last_end, last_end])
            continue
        if ti >= len(ts):
            # a real character with no timestamp: stop rather than invent one
            break
        pair = list(ts[ti]) + [ts[ti][-1]] * 2
        chars.append(ch)
        out_ts.append([float(pair[0]), float(pair[1])])
        last_end = float(pair[1])
        ti += 1
    return "".join(chars), out_ts


def segment_from_chars(text: str, ts: list) -> list[dict]:
    """Turn (text, per-char timestamps) into subtitle segments.

    A cut between chars i and i+1 always has a real time on both sides, so no
    interpolation is needed.
    """
    text, ts = align_text_ts(text, ts)
    if not text or not ts:
        return []
    segs: list[dict] = []
    buf: list[int] = []          # char indices in the current segment

    def flush() -> None:
        if not buf:
            return
        s = _ms(ts[buf[0]][0])
        e = _ms(ts[buf[-1]][1])
        t = "".join(text[i] for i in buf).strip()
        if t and e > s:
            segs.append({"start": round(s, 3), "end": round(e, 3), "text": t})
        buf.clear()

    for i, ch in enumerate(text):
        buf.append(i)
        dur = _ms(ts[buf[-1]][1]) - _ms(ts[buf[0]][0])
        if ch in SENTENCE_END or len(buf) >= SEG_MAX_CHARS or dur >= SEG_MAX_SEC:
            flush()
    flush()

    # absorb tiny fragments (a lone "。", a 1-char segment) into a neighbour
    merged: list[dict] = []
    for s in segs:
        if (merged and len(s["text"]) <= 2
                and s["start"] - merged[-1]["end"] < SEG_GAP_SEC):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] += s["text"]
        elif (merged and s["start"] - merged[-1]["end"] < 0.12
              and len(merged[-1]["text"]) + len(s["text"]) <= SEG_MAX_CHARS + 6):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] += s["text"]
        else:
            merged.append(s)
    return merged


# --- main --------------------------------------------------------------------

def _load_model(hotwords_needed: bool):
    from funasr import AutoModel
    return AutoModel(
        model=ASR_MODEL,
        vad_model=VAD_MODEL,
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model=PUNC_MODEL,
        device="cpu",
        disable_update=True,
    )


def transcribe_one(model, wav: Path, language: str, hotwords: str | None,
                   batch_size_s: int) -> tuple[list[dict], float]:
    gen = dict(input=str(wav), batch_size_s=batch_size_s)
    if hotwords:
        gen["hotword"] = hotwords
    res = model.generate(**gen)
    segs: list[dict] = []
    for r in res:
        text = (r.get("text") or "").replace("\n", "").strip()
        ts = r.get("timestamp") or []
        segs.extend(segment_from_chars(text, ts))
    segs.sort(key=lambda s: s["start"])
    return segs, (segs[-1]["end"] if segs else 0.0)


ENGINE_TAG = "funasr: paraformer-zh + fsmn-vad + ct-punc"


def write_outputs(out_dir: Path, segs: list[dict], duration: float, language: str) -> None:
    (out_dir / "subtitles.srt").write_text(to_srt(segs), encoding="utf-8")
    (out_dir / "subtitles.vtt").write_text(to_vtt(segs), encoding="utf-8")
    (out_dir / "subtitles.json").write_text(
        json.dumps({"language": language, "language_probability": 1.0,
                    "duration": duration, "segments": segs},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    # provenance marker: lets a later run tell a paraformer transcript apart
    # from a faster-whisper one, so switching engines does not re-transcribe
    # (and re-summarise) work that is already done with the right engine
    (out_dir / "asr_engine.txt").write_text(ENGINE_TAG, encoding="utf-8")


def drop_wav(wav: Path, keep: bool) -> None:
    if keep:
        print(f"[asr] kept {wav.name}", file=sys.stderr)
        return
    try:
        wav.unlink()
    except OSError as e:
        print(f"[warn] could not remove {wav}: {e}", file=sys.stderr)


def run_batch(jobs_path: Path, args) -> int:
    """Transcribe a whole list of videos with the model loaded exactly once.

    Model load is 25-60 s (cold 60 s). Spawning one process per video — the
    shape the per-video driver would use — pays that 297 times for this library,
    i.e. ~2.1 h of pure loading against 4.7 h of actual transcription. One
    process, one load.
    """
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    print(f"[asr] {len(jobs)} job(s) from {jobs_path.name}", file=sys.stderr)
    t0 = time.time()
    model = _load_model(True)
    print(f"[asr] model ready in {time.time()-t0:.0f}s", file=sys.stderr)

    hotwords = load_hotwords(args.hotwords)
    done = failed = skipped = 0
    for i, job in enumerate(jobs, 1):
        video = Path(job["video"])
        out_dir = Path(job["out_dir"])
        sub = out_dir / "subtitles.json"
        if sub.is_file() and sub.stat().st_size > 20 and not job.get("redo"):
            print(f"[asr] ({i}/{len(jobs)}) skip (has subtitles) {video.name}", flush=True)
            skipped += 1
            continue
        t1 = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            wav = extract_wav(video, out_dir)
            segs, duration = transcribe_one(model, wav, args.language,
                                            hotwords, args.batch_size_s)
            write_outputs(out_dir, segs, duration, args.language)
            drop_wav(wav, args.keep_wav)
            el = time.time() - t1
            rt = (duration / el) if el else 0
            print(f"[asr] ({i}/{len(jobs)}) ok {video.name}  {len(segs)} seg  "
                  f"{el:.0f}s  {rt:.1f}x", flush=True)
            done += 1
        except Exception as exc:  # keep going: one bad clip must not kill the batch
            failed += 1
            print(f"[asr] ({i}/{len(jobs)}) FAIL {video.name}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    total = time.time() - t0
    print(f"[asr] batch done: {done} ok, {skipped} skipped, {failed} failed "
          f"in {total/60:.1f} min", file=sys.stderr)
    return 1 if failed and not done else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FunASR Paraformer -> SRT/VTT/JSON subtitles")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--jobs", type=Path,
                    help="JSON list of {video, out_dir, redo?} — loads the model once")
    ap.add_argument("--language", default="zh", help="only 'zh' is supported by this model")
    ap.add_argument("--hotwords", default=None,
                    help="space/comma separated terms, or @terms.txt (one per line)")
    ap.add_argument("--keep-wav", action="store_true")
    ap.add_argument("--batch-size-s", type=int, default=300)
    args = ap.parse_args()

    if args.jobs:
        return run_batch(args.jobs, args)

    if not args.video or not args.out_dir:
        print("[err] need --video+--out-dir, or --jobs", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[asr] extracting 16k mono wav...", file=sys.stderr)
    wav = extract_wav(args.video, args.out_dir)

    hotwords = load_hotwords(args.hotwords)
    if hotwords:
        print(f"[asr] hotwords ({len(hotwords.split())} terms): {hotwords[:100]}",
              file=sys.stderr)

    print("[asr] loading Paraformer (vad+punc) on cpu...", file=sys.stderr)
    t0 = time.time()
    model = _load_model(True)
    print(f"[asr] model ready in {time.time()-t0:.0f}s; transcribing...", file=sys.stderr)

    t1 = time.time()
    segs, duration = transcribe_one(model, wav, args.language, hotwords, args.batch_size_s)
    el = time.time() - t1

    write_outputs(args.out_dir, segs, duration, args.language)
    drop_wav(wav, args.keep_wav)

    rt = (duration / el) if el else 0
    print(f"[asr] {el:.0f}s wall, {rt:.1f}x realtime", file=sys.stderr)
    print(f"[ok] {len(segs)} segments -> {args.out_dir}/subtitles.{{srt,vtt,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

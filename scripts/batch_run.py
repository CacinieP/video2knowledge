#!/usr/bin/env python3
"""batch_run.py — batch-process a whole video library through the full pipeline.

Designed for course libraries (e.g. a CPA term of lectures): walks a root dir
of videos, and per video runs the validated Path-3 chain —

    A: asr_caption.py (small + zh + hotwords, wav deleted after)
    B: mm_caption.py --mode dedup --prompt-ocr --hash-size 16 --dedup-hamming 40
       -> merge_visual.py -> build_knowledge.py --merged --format all
       -> build_notes.py --docx --pdf --describe-frames -> gen_apkg.py

Two threads pipeline the two resource pools (faster-whisper CPU vs Ollama),
so stage B of video i overlaps stage A of video i+1. Everything is resumable:
per-video run dirs carry .asr_done / .done / .failed markers — rerun the same
command and finished videos are skipped. A summary CSV + batch.log record
progress; per-video stdout/stderr land in each run dir.

Usage:
    python3 scripts/batch_run.py --root "D:/courses" --out-root runs/batch \
        --order "03,04" --hotwords "@terms.txt" --asr-model small
    python3 scripts/batch_run.py --root ... --dry-run     # show the plan
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HOTWORDS = (
    "会计要素, 资产负债表, 利润表, 现金流量表, 长期股权投资, 权益法, 成本法, "
    "商誉减值, 资产减值, 金融工具, 持有待售, 终止经营, 收入确认, 履约义务, "
    "可变对价, 租赁负债, 使用权资产, 递延所得税, 暂时性差异, 股份支付, "
    "可转换债券, 借款费用, 或有事项, 预计负债, 政府补助, 售后回租, "
    "ESG, 可持续信息披露, 双重重要性"
)

_print_lock = threading.Lock()
_csv_lock = threading.Lock()


def log(tag: str, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    with _print_lock:
        print(line, flush=True)


def venv_python() -> str:
    for cand in (HERE.parent / ".venv" / "Scripts" / "python.exe",
                 HERE.parent / ".venv" / "bin" / "python"):
        if cand.is_file():
            return str(cand)
    return sys.executable


def sanitize(rel: str) -> str:
    """Filesystem-safe run-dir name that keeps CJK readable."""
    s = rel.replace("\\", "/")
    s = re.sub(r"[/:*?\"<>|\s]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")[:120]


def load_hotwords(spec: str) -> str:
    if spec.startswith("@"):
        return ",".join(ln.strip() for ln in
                        Path(spec[1:]).read_text(encoding="utf-8").splitlines()
                        if ln.strip())
    return spec


class Batch:
    def __init__(self, args):
        self.args = args
        self.py = venv_python()
        self.out_root: Path = args.out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.summary = self.out_root / "summary.csv"
        if not self.summary.exists():
            with self.summary.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(
                    ["video", "dur_min", "status", "asr_s", "vlm_s",
                     "build_s", "notes_s", "run_dir"])
        self.logf = open(self.out_root / "batch.log", "a", encoding="utf-8")
        self.log_lock = threading.Lock()
        self.done_count = 0
        self.total = 0
        self.t0 = time.time()

    def record(self, row: list) -> None:
        with _csv_lock:
            with self.summary.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(row)

    def blog(self, tag: str, msg: str) -> None:
        with self.log_lock:
            self.logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{tag}] {msg}\n")
            self.logf.flush()

    def run(self, cmd: list[str], log_path: Path, tag: str) -> float:
        t = time.time()
        with log_path.open("a", encoding="utf-8") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {' '.join(cmd[:6])}...")
        return time.time() - t

    # --- stage A: audio -> subtitles ------------------------------------------
    def stage_a(self, video: Path, run_dir: Path) -> None:
        if (run_dir / ".asr_done").exists() or (run_dir / ".done").exists():
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        lg = run_dir / "stageA.log"
        log("A", f"ASR start: {video.name}")
        s = self.run(
            [self.py, str(HERE / "asr_caption.py"),
             "--video", str(video), "--out-dir", str(run_dir),
             "--model", self.args.asr_model, "--language", "zh",
             "--hotwords", self.args.hotwords], lg, "A")
        subs = run_dir / "subtitles.json"
        n = len(__import__("json").loads(subs.read_text(encoding="utf-8"))
                .get("segments", [])) if subs.exists() else 0
        if n == 0:
            (run_dir / ".failed").write_text("asr: 0 segments\n", encoding="utf-8")
            raise RuntimeError("ASR produced 0 segments (no speech track?)")
        wav = run_dir / "audio_16k.wav"
        if wav.exists():  # ~31 KB/s — do not keep 193 h of it
            wav.unlink()
        (run_dir / ".asr_done").write_text(f"segments={n}\n", encoding="utf-8")
        log("A", f"ASR done: {video.name} ({n} segs, {s:.0f}s)")

    # --- stage B: vision -> fusion -> artifacts --------------------------------
    def stage_b(self, video: Path, run_dir: Path) -> None:
        lg = run_dir / "stageB.log"
        name = video.stem
        title = re.sub(r"^\d+[\s.-]*", "", name) or name
        klass = re.sub(r"[^\w\u4e00-\u9fff（）()]+", "",
                       video.parent.name)[:24]
        s = [0.0, 0.0, 0.0]  # vlm, build, notes
        log("B", f"vision+build start: {name}")
        s[0] = self.run(
            [self.py, str(HERE / "mm_caption.py"),
             "--video", str(video), "--out-dir", str(run_dir),
             "--mode", "dedup", "--prompt-ocr",
             "--model", self.args.vlm_model,
             "--hash-size", "16", "--dedup-hamming", "40"], lg, "B")
        s[1] = self.run(
            [self.py, str(HERE / "merge_visual.py"),
             "--subtitles", str(run_dir / "subtitles.json"),
             "--visual", str(run_dir / "captions.json"),
             "--out", str(run_dir / "merged.json")], lg, "B")
        s[2] = self.run(
            [self.py, str(HERE / "build_knowledge.py"),
             "--subtitles", str(run_dir / "subtitles.json"),
             "--merged", str(run_dir / "merged.json"),
             "--out-dir", str(run_dir), "--format", "all", "--title", title], lg, "B")
        if not self.args.no_notes:
            s[2] += self.run(
                [self.py, str(HERE / "build_notes.py"),
                 "--subtitles", str(run_dir / "subtitles.json"),
                 "--frames", str(run_dir / "frames" / "frames.json"),
                 "--out-dir", str(run_dir), "--max-frames", "12",
                 "--describe-frames", "--docx", "--pdf", "--title", title], lg, "B")
        self.run(
            [self.py, str(HERE / "gen_apkg.py"),
             "--csv", str(run_dir / "cards.csv"),
             "--out", str(run_dir / "cards.apkg"),
             "--deck", f"{self.args.deck}::{klass}::{title}"], lg, "B")
        (run_dir / ".done").write_text(
            f"vlm={s[0]:.0f}s merge+build={s[1]:.0f}s notes={s[2]:.0f}s\n",
            encoding="utf-8")
        self.finish(video, run_dir, "done", s + [0])

    def finish(self, video: Path, run_dir: Path, status: str, s: list) -> None:
        self.done_count += 1
        dur = ""
        try:  # informational duration for the summary row
            dur = f"{__import__('json').loads((run_dir / 'subtitles.json').read_text(encoding='utf-8')).get('duration', 0) / 60:.0f}"
        except Exception:
            pass
        elapsed = time.time() - self.t0
        rate = elapsed / max(self.done_count, 1)
        eta_h = rate * (self.total - self.done_count) / 3600
        log("B", f"({self.done_count}/{self.total}) {status}: {video.name} "
                 f"vlm={s[0]:.0f}s build={s[1]:.0f}s notes={s[2]:.0f}s "
                 f"| ETA {eta_h:.1f}h")
        self.record([str(video), dur, status, *[f"{x:.0f}" for x in s[:3]],
                     str(run_dir)])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Batch course-library processor")
    ap.add_argument("--root", required=True, type=Path, help="video library root")
    ap.add_argument("--out-root", type=Path, default=Path("runs/batch"))
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--vlm-model", default="openbmb/minicpm-v4.6:latest")
    ap.add_argument("--hotwords", default=DEFAULT_HOTWORDS)
    ap.add_argument("--deck", default="2026注会会计")
    ap.add_argument("--order", default="",
                    help='comma list of top-level dir substrings, priority order '
                         '(e.g. "03,04,02,01"); remaining dirs appended after')
    ap.add_argument("--only", default="", help="regex — only process matching paths")
    ap.add_argument("--limit", type=int, default=0, help="stop after N NEW videos (0=all)")
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    videos = [p for p in sorted(args.root.rglob("*"))
              if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".flv")
              and not p.name.startswith(".")]
    if args.only:
        rx = re.compile(args.only)
        videos = [p for p in videos if rx.search(str(p))]
    if args.order:
        def rank(p: Path) -> tuple:
            rel = str(p.relative_to(args.root))
            for i, key in enumerate(k.strip() for k in args.order.split(",")):
                if key and key in rel.split("\\")[0].split("/")[0]:
                    return (i, rel)
            return (99, rel)
        videos.sort(key=rank)

    b = Batch(args)
    plan = [(v, args.out_root / sanitize(str(v.relative_to(args.root)))) for v in videos]
    b.total = sum(1 for v, d in plan
                  if not (d / ".done").exists() and not (d / ".failed").exists())
    todo = [(v, d) for v, d in plan if not (d / ".done").exists()]
    log("plan", f"{len(plan)} videos, {len(plan) - len(todo)} already done, "
                f"{b.total} to process (limit={args.limit or 'all'})")
    if args.dry_run:
        for v, d in plan[:400]:
            mark = "DONE " if (d / ".done").exists() else "todo  "
            print(f"  {mark} {v.relative_to(args.root)}")
        return 0

    processed = {"n": 0}

    def worker_a():
        for v, d in plan:
            if args.limit and processed["n"] >= args.limit:
                return
            if (d / ".done").exists():
                continue
            try:
                b.stage_a(v, d)
                processed["n"] += 1
            except Exception as e:
                log("A", f"FAILED {v.name}: {e}")
                (d / ".failed").write_text(f"stageA: {e}\n", encoding="utf-8")
                b.blog("A", f"FAILED {v}: {e}\n{traceback.format_exc()}")

    def worker_b():
        for v, d in plan:
            if (d / ".failed").exists():
                continue
            while not (d / ".asr_done").exists() and not (d / ".failed").exists() \
                    and not (d / ".done").exists() and a_thread.is_alive():
                time.sleep(5)
            if (d / ".done").exists():
                continue  # already finished in an earlier invocation — silent
            if not (d / ".asr_done").exists():
                if (d / ".failed").exists():
                    b.finish(v, d, "failed", [0, 0, 0])
                continue
            try:
                b.stage_b(v, d)
            except Exception as e:
                log("B", f"FAILED {v.name}: {e}")
                (d / ".failed").write_text(f"stageB: {e}\n", encoding="utf-8")
                b.blog("B", f"FAILED {v}: {e}\n{traceback.format_exc()}")
                b.finish(v, d, "failed", [0, 0, 0])

    a_thread = threading.Thread(target=worker_a, name="stageA", daemon=True)
    a_thread.start()
    worker_b()
    a_thread.join(timeout=0)
    log("end", f"batch complete: {b.done_count} processed this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

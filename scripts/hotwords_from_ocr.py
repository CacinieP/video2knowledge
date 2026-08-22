#!/usr/bin/env python3
"""hotwords_from_ocr.py — feed VLM OCR terms back into ASR (Path-3 cross-path loop).

The dual-path pipeline runs ASR BEFORE VLM OCR, so the on-screen vocabulary
cannot bias the initial transcription — exactly where jargon errors happen
(slide says 权益法/IFRS, ASR hears 全一法/if as). This script closes that loop:

  1. EXTRACT salient on-screen terms from captions.json (VLM OCR): recurring
     CJK n-grams, short header/table-header phrases, Latin acronyms/ids.
     Pure heuristic, no LLM call — deterministic and fully offline.
  2. MERGE them with the manual hotword list and (optionally) an accumulating
     course vocabulary (--course-vocab). In a batch, OCR terms from earlier
     videos seed ASR hotwords for later videos of the same course, so the
     loop needs no re-transcription to pay off.
  3. CHECK coverage: which extracted terms are missing from subtitles.json
     (whitespace/case-normalized containment). Low coverage means the ASR run
     likely mis-heard the jargon — re-run asr_caption.py with the enriched
     hotwords, or let batch_run.py --asr-verify do it automatically
     (--fail-under makes this script exit 3 when coverage drops below a ratio).

Usage:
    python3 hotwords_from_ocr.py --captions run/captions.json \\
        --subtitles run/subtitles.json \\
        --course-vocab runs/batch/course_hotwords.txt \\
        --manual "会计要素, 权益法" --out run/ocr_hotwords.txt

Outputs:
    ocr_hotwords.txt        effective list: manual terms + course vocab /
                            extracted terms, one per line (for --hotwords @)
    course_hotwords.txt     accumulated vocab (old terms first, capped) when
                            --course-vocab is given
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

# Candidates made ONLY of these function characters carry no terminology.
STOP_CHARS = set(
    "的了和与及或在是由从被把为就也都还而且但所以这个那个什么怎么"
    "可以我们你们他们她它很更最不没我你他个种样上下中前后里外之"
    "以及因此但是如果因为关于对于通过进行方面情况时候大家老师同"
    "学现在然后没有之一等等可能这样那样已经应该这些那些自己")
STOP_TERMS = {
    "一下", "例如", "比如", "注意", "下面", "上面", "表格", "如图",
    "图示", "内容", "标题", "目录", "课程", "章节", "本章", "小结",
    "问题", "答案", "要求", "以下", "以上", "某公司", "甲公司", "乙公司",
    "本节", "本讲", "本课", "本节小结", "本章小结", "课后作业", "谢谢观看",
}
EN_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "is", "are",
    "on", "with", "by", "at", "as", "be", "this", "that", "it", "from",
    "table", "chart", "figure", "note", "notes", "example", "page", "part",
}


def load_blocks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("captions") or data.get("segments") or []


def parse_manual(spec: str | None) -> list[str]:
    """'a,b,c' / 'a、b' / '@file' -> ordered deduped term list."""
    if not spec:
        return []
    if spec.startswith("@"):
        terms = [ln.strip() for ln in
                 Path(spec[1:]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        s = spec
        for sep in (",", "、", ";", "；"):
            s = s.replace(sep, " ")
        terms = s.split()
    return list(dict.fromkeys(t for t in terms if t))


def _cjk_ok(term: str) -> bool:
    if term in STOP_TERMS:
        return False
    return any(ch not in STOP_CHARS for ch in term)


def extract_terms(blocks: list[dict], max_terms: int = 40) -> list[str]:
    """Salient on-screen terms: CJK n-grams (2-6 chars) recurring across
    distinct slides or appearing in short header/table-header lines, plus
    Latin acronyms (internal capitals / all-caps). Longer recurring terms
    suppress their own substrings, so 长期股权投资 wins over 股权 when both
    recur. Scored, sorted, capped."""
    df: dict[str, set[int]] = {}      # term -> block indices containing it
    header_hits: dict[str, int] = {}  # term -> short-line occurrences
    acronyms: set[str] = set()
    for bi, b in enumerate(blocks):
        text = b.get("text") or ""
        # table rows -> cells (cells are header-ish: short, term-dense)
        lines: list[str] = []
        for ln in text.splitlines():
            if "|" in ln:
                lines.extend(c.strip(" \t—-·*#>") for c in ln.split("|"))
            else:
                lines.append(ln.strip(" \t—-·*#>"))
        for ln in lines:
            if not ln:
                continue
            short = len(ln) <= 20  # slide title / table header cell
            for run in CJK_RE.findall(ln):
                for n in range(2, 7):
                    if len(run) < n:
                        break
                    for k in range(len(run) - n + 1):
                        t = run[k:k + n]
                        if _cjk_ok(t):
                            df.setdefault(t, set()).add(bi)
                            if short:
                                header_hits[t] = header_hits.get(t, 0) + 1
            for tok in LATIN_RE.findall(ln):
                tl = tok.lower()
                if len(tl) < 2 or tl in EN_STOP:
                    continue
                df.setdefault(tok, set()).add(bi)
                if any(c.isupper() for c in tok[1:]):  # IFRS, FVOCI, MiniCPM
                    acronyms.add(tok)
                if short:
                    header_hits[tok] = header_hits.get(tok, 0) + 1

    scored: list[tuple[int, int, str]] = []
    for t, blocks_set in df.items():
        rec = len(blocks_set)
        hdr = header_hits.get(t, 0)
        # single occurrence suffices for acronyms (IFRS on one slide is still
        # a term) and for LONG header/title phrases (a 4+ char slide title is
        # almost always terminology); short CJK terms must recur or headline
        # twice to keep discourse noise out
        if rec >= 2 or hdr >= 2 or t in acronyms or (hdr >= 1 and len(t) >= 4):
            score = rec * 2 + min(hdr, 3) * 3 + (2 if t in acronyms else 0)
            scored.append((score, len(t), t))
    scored.sort(reverse=True)

    selected: list[tuple[int, str]] = []  # (score, term), substring-aware
    for score, _len, t in scored:
        replaced = False
        keep = True
        for idx, (s2, sel) in enumerate(selected):
            if t in sel:            # parent already selected -> skip child
                keep = False
                break
            if sel in t:            # child selected but longer term scores close
                if score >= s2:
                    selected[idx] = (score, t)
                    replaced = True
                keep = False
                break
        if keep and not replaced:
            selected.append((score, t))
        if len(selected) >= max_terms:
            break
    return [t for _s, t in selected]


def norm_text(s: str) -> str:
    """Comparison form: drop all whitespace, casefold (same trick as the OCR
    dedup gate in mm_caption.py)."""
    return "".join(s.split()).casefold()


def coverage(terms: list[str], asr_segments: list[dict]) -> tuple[int, list[str]]:
    """(covered_count, missing_terms) — a term is covered when it appears
    verbatim (normalized) anywhere in the ASR text."""
    hay = norm_text("".join(str(s.get("text", "")) for s in asr_segments))
    missing = [t for t in terms if norm_text(t) not in hay]
    return len(terms) - len(missing), missing


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract hotword terms from VLM OCR and check ASR coverage")
    ap.add_argument("--captions", required=True, type=Path,
                    help="captions.json from mm_caption.py (Path 1/3)")
    ap.add_argument("--subtitles", type=Path, default=None,
                    help="subtitles.json from asr_caption.py — enables the "
                         "coverage check")
    ap.add_argument("--manual", default=None,
                    help="existing hotword list ('a,b,c' or @file); kept first "
                         "in the output")
    ap.add_argument("--course-vocab", type=Path, default=None,
                    help="accumulating course vocabulary file; new terms are "
                         "appended (capped) so later videos transcribe better")
    ap.add_argument("--out", type=Path, default=None,
                    help="output hotword file (default: <captions dir>/ocr_hotwords.txt)")
    ap.add_argument("--max-terms", type=int, default=40,
                    help="cap on the effective hotword list (default 40 — "
                         "whisper's initial_prompt truncates around 224 tokens)")
    ap.add_argument("--vocab-cap", type=int, default=80,
                    help="cap on the accumulated course vocabulary (default 80)")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="with --subtitles: exit 3 when the coverage ratio "
                         "drops below this (used by batch_run.py --asr-verify)")
    args = ap.parse_args()

    if not args.captions.is_file():
        print(f"[err] captions not found: {args.captions}", file=sys.stderr)
        return 2
    blocks = load_blocks(args.captions)
    if not blocks:
        print(f"[err] no visual blocks in {args.captions}", file=sys.stderr)
        return 2

    extracted = extract_terms(blocks, args.max_terms)

    # accumulated course vocabulary: old terms first, then new by score
    vocab_terms: list[str] = []
    new_to_vocab = 0
    if args.course_vocab:
        if args.course_vocab.is_file():
            vocab_terms = [ln.strip() for ln in
                           args.course_vocab.read_text(encoding="utf-8").splitlines()
                           if ln.strip()]
        new_to_vocab = sum(1 for t in extracted if t not in vocab_terms)
        merged = list(dict.fromkeys(vocab_terms + extracted))
        atomic_write(args.course_vocab, "\n".join(merged[:args.vocab_cap]) + "\n")
        pool = vocab_terms + [t for t in extracted if t not in vocab_terms]
    else:
        pool = extracted

    manual = parse_manual(args.manual)
    effective = list(dict.fromkeys(manual + [t for t in pool if t not in manual]))
    effective = effective[:args.max_terms]

    out = args.out or args.captions.parent / "ocr_hotwords.txt"
    atomic_write(out, "\n".join(effective) + ("\n" if effective else ""))
    print(f"[ok] {len(extracted)} terms extracted, {len(effective)} effective "
          f"(manual {len(manual)})"
          + (f", +{new_to_vocab} new -> {args.course_vocab.name}" if args.course_vocab else "")
          + f" -> {out}", file=sys.stderr)

    if args.subtitles:
        if not args.subtitles.is_file():
            print(f"[warn] subtitles not found, skipping coverage check: "
                  f"{args.subtitles}", file=sys.stderr)
            return 0
        segs = json.loads(args.subtitles.read_text(encoding="utf-8"))
        segs = segs.get("segments", segs) if isinstance(segs, dict) else segs
        covered, missing = coverage(extracted, segs or [])
        total = len(extracted)
        ratio = covered / total if total else 1.0
        print(f"[hotwords] OCR-term coverage in ASR: {covered}/{total} "
              f"({ratio:.0%})", file=sys.stderr)
        if missing:
            preview = "、".join(missing[:10])
            more = f" …(+{len(missing) - 10})" if len(missing) > 10 else ""
            print(f"[hotwords] not heard in ASR (ASR mis-heard jargon?): "
                  f"{preview}{more}", file=sys.stderr)
        if args.fail_under is not None and ratio < args.fail_under:
            print(f"[hotwords] coverage {ratio:.0%} < {args.fail_under:.0%} — "
                  f"re-run asr_caption.py with --hotwords @{out}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

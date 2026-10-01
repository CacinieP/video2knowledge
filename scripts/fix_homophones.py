#!/usr/bin/env python3
"""fix_homophones.py — repair ASR homophone errors against a domain glossary.

Why this exists: ASR hears sounds, not words, so a term whose characters are
near-homophones of the real term comes out spelled wrong — and no amount of
`--hotwords` prevents it. On the 20-lecture piano set this finds 20 classes of
genuine error in 46 places (100% precision against manual review):

    爬音 → 琶音 (x16)     触见/处键/触贱/触剑/触件 → 触键 (x12)
    适谱 → 视谱 (x6)      吊号 → 调号 (x4)       暑期/暑七 → 属七 (x3)
    合声 → 和声 (x3)      音成/阴城 → 音程 (x3)   乐剧 → 乐句 (x2)
    延因/沿因 → 延音 (x2)  试谱 → 视谱             手行 → 手型
    试唱 → 视唱 (x2)      川指 → 穿指             普号 → 谱号

These propagate: they end up in the knowledge doc's headings, the Anki cards,
and the quoted 原声 lines of the illustrated notes. The same scan on
faster-whisper output found 23 classes / 59 hits; Paraformer (PR #7) already
removed the worst of them, which is why the residue here is so concentrated in
genuinely ambiguous syllables.

Method
------
Match on **tone-aware pinyin**, not on edit distance. 被谱 and 背谱 are both
bèi pǔ — identical syllables, identical tones, so the audio genuinely was
ambiguous. And the tone is what makes the rule safe: 音阶 (yīn jiē) is never
corrected to 音介 (yīn jiè), because a tone difference means the audio was *not*
ambiguous and the text is probably right.

Guards, because a corrector that silently rewrites text is worse than none:

* the fragment must not already be a glossary term
* a **function-character guard** rejects fragments that begin or end with a
  high-frequency grammatical character. Real homophone errors are domain terms
  (爬音, 适谱); false positives are usually two common words sliced in
  half — 曲是 → 曲式, 声不 → 声部, 调是 → 调式, 和手 → 合手. Losing a true
  positive here is fine; corrupting common words is not.
* `DEFAULT_BLOCKED` holds the two fragments the guard above cannot catch
  (实度 from 坚实度, 何首 from 何首乌) — see the comment on that constant
* corrections are applied by **offset into the concatenated transcript**, not by
  substring search, so a typo that occurs twice in one segment is fixed twice
  and a typo in segment A is never "found" in an identical segment B
* `subtitles.srt` / `subtitles.vtt` are regenerated from the corrected segments
  — three subtitle files that disagree is a bug, not a feature
* every change is written to `homophone_fixes.json` with surrounding context,
  and `--dry-run` reports without touching anything

Usage:
    python3 scripts/fix_homophones.py --subtitles OUT/subtitles.json
    python3 scripts/fix_homophones.py --subtitles OUT/subtitles.json --dry-run
    python3 scripts/fix_homophones.py --subtitles OUT/subtitles.json \\
        --glossary @music-terms.txt --block 实度
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    from pypinyin import Style, pinyin
except ImportError:  # pragma: no cover
    print("[err] pypinyin is required:  pip install pypinyin", file=sys.stderr)
    raise SystemExit(3)


# Characters that essentially never begin or end a music term but constantly
# end a clause. Their presence at a fragment edge means the "term" is really two
# ordinary words — see the module docstring for the measured cases.
FUNCTION_CHARS = set(
    "的是不了在有和就也都很这那你我他她它们要会能对把被给让从到向跟与或"
    "但而其之以所于上下中里外前后时候个们第只又再还才更最没呢吗吧啊呀"
)

DEFAULT_GLOSSARY = [
    # 钢琴演奏技法
    "背谱", "视谱", "穿指", "跨指", "同指", "连奏", "断奏", "跳音", "延音",
    "手腕", "手型", "坐姿", "重心", "发力", "放松", "高抬指", "低抬指",
    "触键", "键床", "立架", "乐理", "记忆", "练耳", "视唱", "合手", "分手",
    # 音阶与和声
    "音阶", "琶音", "和弦", "八度", "双音", "三度", "六度", "十度", "音程",
    "调式", "主调", "大小调", "属七", "减七", "增七", "九和弦", "和弦进行",
    "功能和声", "半音", "全音", "调内音", "调外音", "经过音", "辅助音",
    "变化音", "声部", "旋律", "节奏", "曲式", "和声", "配器", "织体", "复调",
    # 记谱
    "五线谱", "简谱", "工尺谱", "谱号", "调号", "拍号", "小节", "节拍",
    "乐句", "升降记号", "临时记号", "还原号", "反复记号", "终止线", "连线",
    # 教学常用
    "教师节", "钢琴老师", "教学", "课程", "练习", "方法", "要点", "基本功",
]

# Fragments that are homophones of a term but overwhelmingly belong to an
# ordinary word. Both were measured false positives on the 20-lecture piano
# corpus, where the other 20 classes were all genuine:
#
#   坚实度 -> 坚实十度    实度 (shí dù) and 十度 (shí dù) are the same syllables
#   何首   -> 合手        何首 (hé shǒu) vs 合手 (hé shǒu), and 何首乌 is a word
#
# The function-character guard does not catch these because neither edge
# character is grammatical. Blocking by default keeps measured precision at
# 20/20 classes; `--block` adds more without editing this list.
DEFAULT_BLOCKED = frozenset({"实度", "何首"})


@lru_cache(maxsize=65536)
def _py(word: str) -> str:
    """Tone-marked pinyin syllables, space separated: 背谱 -> 'bèi pǔ'."""
    return " ".join(x[0] for x in pinyin(word, style=Style.TONE))


def build_index(terms: list[str]) -> tuple[dict[str, list[str]], frozenset[str]]:
    """Tone-pinyin -> terms, plus the pinyin of every character a term starts with.

    The second value is a cheap pre-filter: a fragment can only match if its
    first character is a homophone of some term's first character, which skips
    most of the transcript without scoring pinyin for every candidate length.
    It is indexed by *pinyin*, not by character — 爬音 is a mishearing of 琶音,
    so no literal character overlap exists and a char-based filter would drop
    exactly the cases this script exists to fix.
    """
    idx: dict[str, list[str]] = {}
    firsts: set[str] = set()
    for t in terms:
        if len(t) >= 2:
            idx.setdefault(_py(t), []).append(t)
            firsts.add(_py(t[0]))
    return idx, frozenset(firsts)


def has_function_edge(frag: str) -> bool:
    return bool(frag) and (frag[0] in FUNCTION_CHARS or frag[-1] in FUNCTION_CHARS)


def propose(text: str, glossary: set[str], index: dict[str, list[str]],
            max_len: int, blocked: set[str] = frozenset(),
            firsts: frozenset[str] = frozenset()) -> list[dict]:
    """Every non-overlapping fragment that looks like a misheard glossary term.

    Offsets index `text`, the whole concatenated transcript, so a term split
    across two ASR segments is still found as one fragment.

    Non-overlapping, and left-to-right: once a fragment is claimed the scan
    resumes past it, so "音成近的音成远" yields two independent corrections
    rather than overlapping partial matches. Longest fragment wins, because a
    longer match is the more specific claim.

    `blocked` is the escape hatch in both directions: DEFAULT_BLOCKED holds the
    measured false positives, and `--block` adds more. Losing a true positive
    is fine; corrupting an ordinary word is not.
    """
    out: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        hit = None
        if not firsts or _py(text[i]) in firsts:
            for L in range(min(max_len, n - i), 1, -1):
                frag = text[i:i + L]
                if frag in glossary or frag in blocked or has_function_edge(frag):
                    continue
                for cand in index.get(_py(frag), ()):
                    if cand != frag:
                        hit = (frag, cand, L)
                        break
                if hit:
                    break
        if hit:
            frag, cand, L = hit
            out.append({"start": i, "end": i + L, "from": frag, "to": cand,
                        "context": text[max(0, i - 12): i + L + 12]})
            i += L
        else:
            i += 1
    return out


def concat_text(segs: list[dict]) -> str:
    """The exact string `propose` offsets refer to."""
    return "".join(s.get("text") or "" for s in segs)


def apply_fixes(segs: list[dict], fixes: list[dict],
                text: str | None = None) -> tuple[int, list[dict]]:
    """Rewrite segment text in place from the recorded offsets.

    Returns (segments touched, anomalies). A fix whose recorded offsets no
    longer spell `from` is an anomaly: it is reported and skipped rather than
    guessed at, because a silent no-op here would mean the report claims
    corrections that never landed.

    A fragment can straddle a segment boundary (ASR cuts mid-phrase). The
    replacement lands in the earlier segment and the later one loses the
    consumed prefix, which keeps every timestamped cue non-empty.
    """
    if not fixes:
        return 0, []
    if text is None:
        text = concat_text(segs)

    starts, spans, pos = [], [], 0
    for s in segs:
        t = s.get("text") or ""
        starts.append(pos)
        spans.append((pos, pos + len(t)))
        pos += len(t)

    anomalies: list[dict] = []
    dirty: set[int] = set()
    # Right-to-left: a replacement may change length, and edits to the right
    # must not shift the offsets of the edits still to be applied on the left.
    for f in sorted(fixes, key=lambda f: -f["start"]):
        fs, fe = f["start"], f["end"]
        if text[fs:fe] != f["from"]:
            anomalies.append({**f, "reason": "offset drift",
                              "found": text[fs:fe]})
            continue
        i = bisect.bisect_right(starts, fs) - 1
        j = bisect.bisect_right(starts, fe - 1) - 1
        if i < 0 or j >= len(segs) or i > j:  # pragma: no cover - defensive
            anomalies.append({**f, "reason": "offset out of range"})
            continue
        head = segs[i]
        head_t = head.get("text") or ""
        # The fix occupies [fs, min(fe, end of head)); anything after that
        # offset inside the head segment is untouched text that must survive.
        head_cut = min(fe, spans[i][1]) - spans[i][0]
        head_t = head_t[:fs - spans[i][0]] + f["to"] + head_t[head_cut:]
        head["text"] = head_t
        dirty.add(i)
        if j > i:
            tail = segs[j]
            tail_t = tail.get("text") or ""
            tail["text"] = tail_t[fe - spans[j][0]:]
            dirty.add(j)

    return len(dirty), anomalies


def _srt_writers():
    """(to_srt, to_vtt) from asr_caption, so subtitle formats stay canonical.

    Imported by path because asr_caption lives beside this script but probes the
    hardware profile at import time. Refusing to continue without it is
    deliberate: silently leaving .srt/.vtt uncorrected is the exact failure mode
    this script exists to remove.
    """
    import importlib.util
    target = HERE / "asr_caption.py"
    if not target.is_file():
        raise SystemExit(f"[err] {target} not found — cannot regenerate "
                         f"subtitles.srt/.vtt (use --no-sync to skip them)")
    try:
        spec = importlib.util.spec_from_file_location("_v2k_asr_caption", target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.to_srt, mod.to_vtt
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"[err] cannot import to_srt/to_vtt from {target}: {exc}\n"
                         f"       use --no-sync to skip regenerating .srt/.vtt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subtitles", required=True, type=Path)
    ap.add_argument("--glossary", default="",
                    help="extra terms: comma/space separated, or @file (one per line). "
                         "Built-in domain terms are always included.")
    ap.add_argument("--block", action="append", default=[], metavar="WORD",
                    help="never propose this fragment (repeatable), added to "
                         "the built-in blocklist, e.g. --block 实度")
    ap.add_argument("--no-default-block", action="store_true",
                    help="also allow the fragments in DEFAULT_BLOCKED "
                         "(they were measured false positives)")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not regenerate subtitles.srt/.vtt (they will then keep "
                         "the uncorrected text)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the changes without rewriting any file")
    ap.add_argument("--report", type=Path, default=None,
                    help="where to write the change log "
                         "(default <out-dir>/homophone_fixes.json)")
    args = ap.parse_args()

    if not args.subtitles.is_file():
        print(f"[err] not found: {args.subtitles}", file=sys.stderr)
        return 2
    data = json.loads(args.subtitles.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    if not segs:
        print(f"[homophone] {args.subtitles.name}: 0 segments, nothing to do")
        return 0

    terms = list(DEFAULT_GLOSSARY)
    if args.glossary:
        spec = args.glossary
        if spec.startswith("@"):
            terms += [ln.strip() for ln in
                      Path(spec[1:]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            for sep in (",", "、", ";", "；", " "):
                spec = spec.replace(sep, ",")
            terms += [t.strip() for t in spec.split(",") if t.strip()]
    glossary = set(terms)
    index, firsts = build_index(terms)
    max_len = max((len(t) for t in glossary if len(t) >= 2), default=2)

    text = concat_text(segs)
    blocked = set(args.block)
    if not args.no_default_block:
        blocked |= DEFAULT_BLOCKED
    fixes = propose(text, glossary, index, max_len, blocked, firsts)
    if not fixes:
        print(f"[homophone] {args.subtitles.parent.name}: clean")
        return 0

    counts: dict[tuple[str, str], int] = {}
    for f in fixes:
        counts[(f["from"], f["to"])] = counts.get((f["from"], f["to"]), 0) + 1
    for (a, b), n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"[homophone] {a} -> {b}  x{n}")

    if args.dry_run:
        print(f"[homophone] --dry-run: {len(fixes)} change(s) NOT applied")
        return 0

    touched, anomalies = apply_fixes(segs, fixes, text)
    args.subtitles.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    synced = []
    if not args.no_sync:
        to_srt, to_vtt = _srt_writers()
        srt, vtt = args.subtitles.with_suffix(".srt"), args.subtitles.with_suffix(".vtt")
        srt.write_text(to_srt(segs), encoding="utf-8")
        vtt.write_text(to_vtt(segs), encoding="utf-8")
        synced = [srt.name, vtt.name]

    report = args.report or (args.subtitles.parent / "homophone_fixes.json")
    report.write_text(json.dumps(
        {"source": str(args.subtitles), "changes": fixes,
         "anomalies": anomalies, "synced": synced,
         "counts": [{"from": a, "to": b, "n": n} for (a, b), n in counts.items()]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {len(fixes)} change(s) in {touched} segment(s) -> {args.subtitles}")
    if synced:
        print(f"[ok] resynced {' + '.join(synced)}")
    else:
        print(f"[warn] --no-sync: subtitles.srt/.vtt still hold the old spelling")
    print(f"[ok] log -> {report}")
    if anomalies:
        print(f"[err] {len(anomalies)} change(s) could not be applied (offset drift)",
              file=sys.stderr)
        for a in anomalies[:5]:
            print(f"      {a['from']} -> {a['to']} @ {a['start']}:{a['end']} "
                  f"found {a.get('found')!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""test_fusion_upgrade.py — tests for the 4 fusion-scheme upgrades:

1. atomic chunking: a chunk boundary must never cut a multi-line visual block
2. prompt-echo filter: instruction fragments leaked into list output get dropped
3. weak dwell gate: narration elaborating on a SETTLED slide is not flagged
4. duration-scaled caps: longer videos earn proportionally larger field caps

No network, no models. Run:  python tests/test_fusion_upgrade.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import build_knowledge as bk  # noqa: E402
import merge_visual as mv  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if not cond:
        print(f"  FAIL {name} {detail}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok {name}")


def _sample_text(blocks: int = 6, rows: int = 10) -> str:
    """Interleaved-style text: each unit = [ts] header + 🖼️ multi-line table."""
    units = []
    for b in range(blocks):
        ts = f"[{b:02d}:10]"
        table = "\n".join(f"| 列{b} 第{r}行 | 数值{b}{r} |" for r in range(rows))
        units.append(f"{ts} 🎙️第{b}段讲解内容\n🖼️画面:\n{table}")
    return "\n".join(units)


def test_atomic_chunking() -> None:
    raw = _sample_text(blocks=6, rows=10)
    chunks = bk._chunk_lines(raw, 500)          # a block unit is ~330 chars
    check("chunking loses no text",
          "\n".join(chunks) == raw)
    inside_cut = False
    for ch in chunks:
        lines = ch.splitlines()
        # a 🖼️ block must run to the next [ts] header or the chunk end with
        # ALL its rows: if a table row is the chunk's last line, the block
        # was either cut or happens to end exactly there — distinguish by
        # counting rows between 🖼️ markers and [ts] headers
        block_rows = 0
        in_block = False
        for ln in lines:
            if ln.startswith("🖼️"):
                in_block, block_rows = True, 0
            elif bk._TS_HEADER_RE.match(ln):
                if in_block and block_rows != 10:
                    inside_cut = True
                in_block, block_rows = False, 0
            elif in_block and ln.startswith("|"):
                block_rows += 1
        if in_block and block_rows != 10:
            inside_cut = True
    check("no chunk boundary inside a visual block", not inside_cut)


def test_echo_filter() -> None:
    lines = [
        "- **[mm:ss]** 时间戳必须原样保留，不要新编。只输出最终列表，不要解释。",
        "- [mm:ss]",
        "Task: output only the list",
        "- **可靠性** 画面：三要素｜讲解：合理估计不等于不准确 [01:07]",
        "- **增值税** 画面：可抵扣 vs 计入成本 [04:25]",
    ]
    out = bk._strip_echo(lines)
    check("echo bullets dropped", len(out) == 2, str(out))
    check("content bullets kept", "可靠性" in out[0] and "增值税" in out[1])


def test_weak_dwell_gate() -> None:
    asr = [{"start": 10.0, "end": 16.0, "text": "我们来看一个非常重要的考试重点内容"}]
    # settled slide (0-120s on screen): elaboration is normal -> no weak flag
    visual_settled = [{"start": 0.0, "end": 120.0,
                       "text": "资产负债表日后事项的调整处理原则汇总表"}]
    r1 = mv.merge(asr, visual_settled)
    check("settled slide not flagged weak", r1["weak_attribution"] == 0
          and r1["segments"][0]["match"] == "time",
          f'{r1["segments"][0]["match"]}')
    # fresh slide (0-20s): low overlap right after a flip -> weak stands
    visual_fresh = [{"start": 0.0, "end": 20.0,
                     "text": "资产负债表日后事项的调整处理原则汇总表"}]
    r2 = mv.merge(asr, visual_fresh)
    check("fresh-slide mismatch still weak", r2["weak_attribution"] == 1
          and r2["segments"][0]["match"] == "weak")


def test_caps_scale() -> None:
    short = _sample_text(blocks=2)                       # max ts = [01:10]
    mid = _sample_text(blocks=60)                        # max ts = [59:10]
    long_ = _sample_text(blocks=180)                     # max ts = [179:10] ~3h
    c1, c2, c3 = (bk._caps_for(t) for t in (short, mid, long_))
    check("short video keeps base caps", c1["timeline"] == 12, str(c1))
    check("60min video scales up", c2["timeline"] == 16, str(c2))
    check("3h video clamps at 3x", c3["timeline"] == 36
          and c3["qa"] == 90 and c3["glossary"] == 48, str(c3))


def main() -> int:
    print("fusion upgrade 1 — atomic chunking:")
    test_atomic_chunking()
    print("fusion upgrade 2 — echo filter:")
    test_echo_filter()
    print("fusion upgrade 3 — weak dwell gate:")
    test_weak_dwell_gate()
    print("fusion upgrade 4 — duration-scaled caps:")
    test_caps_scale()
    print(f"\nALL {PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

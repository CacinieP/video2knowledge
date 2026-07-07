#!/usr/bin/env python3
"""gen_apkg.py — STEP 2.3: convert cards.csv (from build_knowledge.py) into an
Anki .apkg deck using genanki.

Card model fields: question (front), answer (back), tags, timestamp, source.
Run scripts/setup_models.sh first to install genanki into the venv.

Usage:
    python3 gen_apkg.py --csv out/cards.csv --out out/cards.apkg --deck "视频知识卡"
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

try:
    import genanki
except ImportError:
    print("[err] genanki not installed. Run scripts/setup_models.sh (or "
          "pip install genanki).", file=sys.stderr)
    sys.exit(3)

# Deterministic IDs so reruns are stable (genanki wants unique-but-fixed IDs).
MODEL_ID = 1830540287
DECK_ID_BASE = 9120451233

MODEL = genanki.Model(
    model_id=MODEL_ID,
    name="Video2Knowledge Card",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
        {"name": "Tags"},
        {"name": "Timestamp"},
        {"name": "Source"},
    ],
    templates=[{
        "name": "Card 1",
        "qfmt": '<div class="q">{{Question}}</div>'
                '<div class="ts">{{Timestamp}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="a">{{Answer}}</div>'
                 '<div class="src">{{Source}}</div>',
    }],
    css=(
        ".card{font-family:-apple-system,'PingFang SC',sans-serif;"
        "text-align:center;color:#1f2328;font-size:18px}"
        ".q{font-weight:600}.ts{color:#0a7;font-size:13px;margin-top:6px}"
        ".a{text-align:left;line-height:1.6}.src{color:#888;font-size:12px;margin-top:10px}"
    ),
)


def deck_for(name: str) -> genanki.Deck:
    # Stable deck id derived from name so the same deck name maps to the same id.
    did = (DECK_ID_BASE + sum(ord(c) for c in name)) % (2**31)
    return genanki.Deck(deck_id=did, name=name)


def main() -> int:
    ap = argparse.ArgumentParser(description="cards.csv -> Anki .apkg")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deck", default="视频知识卡")
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"[err] csv not found: {args.csv}", file=sys.stderr)
        return 2

    deck = deck_for(args.deck)
    n = 0
    with args.csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = genanki.Note(
                model=MODEL,
                fields=[row.get("question", ""), row.get("answer", ""),
                        row.get("tags", ""), row.get("timestamp", ""),
                        row.get("source", "")],
                guid=genanki.guid_for(row.get("question", "") + "|" + str(n)),
            )
            deck.add_note(note)
            n += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pkg = genanki.Package(deck)
    pkg.write_to_file(str(args.out))
    print(f"[ok] {n} notes -> {args.out} (deck: {args.deck!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

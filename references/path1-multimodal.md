# Path 1 — Multimodal Captioning (Ollama VLM)

Path 1 reads a video **visually**: sample frames with ffmpeg, send each frame to a
native multimodal small model (≤4B) via Ollama, and stitch the per-frame captions
into a timestamped subtitle document.

## When to choose Path 1

- No usable speech track (silent screen recording, slide-only video, music B-roll).
- Speech is unreliable (heavy accent, noise) and you want visual grounding.
- You need on-screen text / diagrams / charts captured (ASR cannot see these).
- You want to cross-check ASR output against what is actually on screen.

Path 1 is **slower and less precise on timing** than Path 2 (frame interval sets the
granularity), so prefer Path 2 when a clean speech track exists.

## Model

The default VLM is **picked by your hardware profile** (`scripts/hardware_profile.py`):

| Profile | VLM | Size |
|---|---|---|
| `tiny` (RAM < 6 GB) | `moondream` | ~1.6 GB |
| `low` / `low-mac` / `mid` (6–16 GB) | `openbmb/minicpm-v4.6:latest` | 1.6 GB |
| `high` (16–32 GB) | `qwen2.5vl:3b` | ~2 GB |
| `high-gpu` / `max` | `qwen2.5vl:7b` | ~4.5 GB |

On the most common machines (8–16 GB) this resolves to `minicpm-v4.6`, which is
the tested default. Override with `--model` or `VLM_MODEL=`. Full profile table
in `references/hardware-profiles.md`.

Other VLMs that work the same way (via `ollama pull`):
`moondream` (very fast, terse), `qwen2.5vl:3b` (strong OCR / on-screen text).

## Ollama vision API

The script calls `POST {OLLAMA_HOST}/api/generate` with:

```json
{
  "model": "openbmb/minicpm-v4.6:latest",
  "prompt": "<caption prompt>",
  "images": ["<base64 jpg>"],
  "stream": false,
  "options": {"temperature": 0.2}
}
```

- `images` is a list of **base64-encoded** raw image bytes (no data-URL prefix).
- `stream: false` returns the whole response in one JSON object (`.response`).
- Low temperature (0.2) keeps captions consistent across frames.

## Sampling strategy

`extract_frames.py` uses `ffmpeg -vf fps=1/interval`. The default interval is **2.0s**
(~30 frames/min). Tuning:

| Goal | `--interval` |
|---|---|
| Slides / slow content | 3.0–5.0 |
| Talking head | 2.0 (default) |
| Fast demo / tutorial | 0.5–1.0 |

For dense content use `--fps 1` (1 frame/sec) instead of `--interval`. Frame
timestamps in `frames.json` are derived from the video's true duration (even
spacing), not from `index × interval`, so they stay accurate.

## Prompt

The caption prompt (in `mm_caption.py`) asks the model for **≤40-char Chinese
descriptions** focusing on objects, on-screen text, actions, and diagrams, with no
preamble. Edit `PROMPT` in the script to change language/style globally.

## Outputs

```
<out-dir>/
├── frames/
│   ├── frame_000001.jpg ... frame_NNNNNN.jpg
│   └── frames.json                 # {"file","t"} per frame
├── captions.srt                    # timestamped subtitles
└── captions.json                   # [{start,end,text}, ...]
```

`captions.json` is consumed by `build_knowledge.py` exactly like Path 2's
`subtitles.json` (the loader detects the schema automatically).

## Memory

Peak memory is flat because the script captions **sequentially** (one frame at a
time, no batching). The profile picker already matches the VLM size to your RAM
(see table above). If you override to a bigger VLM and hit OOM, raise `--interval`
to reduce frame count, or close other apps.

## End-to-end example

```bash
source ~/.zcode/skills/video2knowledge/.venv/bin/activate
cd ~/.zcode/skills/video2knowledge
python3 scripts/mm_caption.py \
  --video ~/Movies/demo.mp4 \
  --out-dir runs/demo-mm \
  --model openbmb/minicpm-v4.6:latest \
  --interval 2.0
python3 scripts/build_knowledge.py \
  --subtitles runs/demo-mm/captions.json \
  --out-dir runs/demo-mm --format all
```

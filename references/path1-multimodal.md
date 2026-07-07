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

Default: `openbmb/minicpm-v4.6:latest` (~1.6 GB, native VLM, ≤4B params).
Already present if `setup_models.sh` was run. Alternatives that work the same way:

| Model | Size | Good for |
|---|---|---|
| `openbmb/minicpm-v4.6:latest` | 1.6 GB | default, balanced |
| `moondream` (via `ollama pull moondream`) | ~1.6 GB | very fast, terse captions |
| `qwen2.5vl:3b` (if available) | ~2 GB | strong OCR / on-screen text |

Override with `--model`.

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

## Memory on 8 GB machines

MiniCPM-V 4.6 at Q4 fits comfortably in 8 GB alongside macOS. The script captions
**sequentially** (one frame at a time) with no batching, which keeps peak memory
flat. If you switch to a bigger VLM and hit OOM, raise `--interval` to reduce frame
count, or stop other apps.

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

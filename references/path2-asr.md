# Path 2 — ASR Transcription (faster-whisper)

Path 2 extracts the audio track and transcribes it with **faster-whisper** (the
CTranslate2 backend of Whisper), producing word/segment-level timestamps. This is
the recommended path whenever the video has a clear speech track.

## When to choose Path 2

- Lecture, talk, interview, podcast video, tutorial with narration.
- You need accurate, fine-grained timestamps (segment + word level).
- You want the highest fidelity text — Whisper models are SOTA for general speech.

Avoid Path 2 when: the video is silent/slide-only, audio is non-speech, or you
need what is visually on screen (use Path 1 or run both).

## Choose an ASR backend (`--backend`, since 2026-09)

Path 2 has three interchangeable backends. All emit the **same** subtitles
schema so `build_knowledge.py` does not change:

| Backend | `--backend` | Where it runs | Best for | Install |
|---|---|---|---|---|
| faster-whisper | `faster-whisper` *(default)* | local, CPU/CTranslate2 | general English, broad multilingual, no setup | default `setup_models.sh` |
| FunASR (Alibaba) | `funasr` | local, PyTorch | **Chinese / multilingual SOTA** (qwen3-asr, paraformer-zh, sensevoice-small) | opt-in: `bash scripts/setup_models.sh --with-funasr` |
| OpenAI-compatible API | `openai-api` | cloud, any OpenAI-shape endpoint | quick cloud run; no local GPU needed; up-to-date provider models | opt-in: `bash scripts/setup_models.sh --with-openai-client` |

Default is `faster-whisper` from `scripts/hardware_profile.py` (field
`asr_backend_default`). Override globally with env `ASR_BACKEND=funasr` or
per-run with `--backend`.

### Backend A — faster-whisper (default)

Already documented below this section. No change.

### Backend B — FunASR (local, qwen3-asr)

**Why FunASR / Qwen3-ASR for Chinese?** On the HF Open ASR Leaderboard
(2026-09), Qwen3-ASR-1.7B hits **5.76% mean WER** across 8 datasets, vs
Whisper-large-v3's 7.44% — and it's natively strong on Chinese, code-switch,
and 52 languages via auto-detection. For a Chinese lecture video this is the
strongest open-source local choice; Whisper is the safer general fallback.

Install once:
```bash
bash scripts/setup_models.sh --with-funasr
```
First run auto-downloads the model from ModelScope (~2–4 GB).

```bash
python3 scripts/asr_caption.py \
  --video lecture.mp4 --out-dir runs/lecture-asr \
  --backend funasr --model qwen3-asr --language zh
```

Other FunASR model names (`--model` is passed straight to `AutoModel`):
- `paraformer-zh` — Mandarin-only, faster, smaller
- `paraformer-zh-streaming` — streaming Mandarin
- `sensevoice-small` — multilingual tiny (good for short clips, 17 languages)

Notes:
- `--device cpu|cuda` is honored by FunASR when supported; most models auto-pick.
- `--compute-type` is ignored (FunASR uses its own dtype handling).
- `--hotwords` is passed as FunASR's `hotword=` param (same parser, same syntax).
- Segment timestamps come from FunASR's `timestamp` field (ms); the converter
  treats the whole item as one segment spanning the first → last token.

### Backend C — OpenAI-compatible cloud API (any provider)

`--backend openai-api` accepts **any** ASR endpoint that speaks the OpenAI
`/v1/audio/transcriptions` shape. The script does **not** hardcode provider
names or model lists — you bring your own endpoint URL and model id. The
OpenAI Python SDK ≥1.0 handles auth, retries, and the verbose_json schema.

Required CLI flags (only for `openai-api`):
- `--api-base URL` — your endpoint's base URL (no vendor is special-cased)
- `--api-model NAME` — your model id at that endpoint (any string)
- `--api-key-env VAR` — name of the env var holding the API key; **never**
  pass the key itself on the CLI (avoid shell-history leakage). Defaults to
  `OPENAI_API_KEY`.

The endpoint is asked for `verbose_json`; segments are converted verbatim to
`{start, end, text}`. API key is read from the env var named by
`--api-key-env` (default `OPENAI_API_KEY`) — never pass the key itself on
the CLI. `--device` / `--compute-type` are ignored.

**Examples of OpenAI-compatible ASR endpoints** (the script works with ANY URL; this list is illustrative, not a whitelist — your own gateway / a model you host works the same way):

| Example endpoint | `--api-base` | `--api-model` (example) | `--api-key-env` |
|---|---|---|---|
| DashScope (Aliyun, Qwen3-ASR) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3-asr-flash` | `DASHSCOPE_API_KEY` |
| OpenAI | `https://api.openai.com/v1` | `whisper-1` | `OPENAI_API_KEY` |
| Groq | `https://api.groq.com/openai/v1` | `whisper-large-v3-turbo` | `GROQ_API_KEY` |
| Local `whisper.cpp` server / faster-whisper-server (any OpenAI-compatible) | `http://127.0.0.1:8080/v1` | `whisper-1` | (often empty) |
| Your own gateway / a model you host | `https://your-host/v1` | `your-model-name` | `YOUR_API_KEY_ENV` |

Install the SDK once:
```bash
bash scripts/setup_models.sh --with-openai-client
```

#### Persisting defaults in shell rc (recommended)

Instead of typing the same flags every run, set them once:

```bash
# in ~/.zshrc or ~/.bashrc
export ASR_BACKEND=openai-api
export ASR_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
export ASR_API_MODEL=qwen3-asr-flash
export ASR_API_KEY_ENV=DASHSCOPE_API_KEY
export ASR_LANGUAGE=zh
```

Then a single-line run is enough:
```bash
python3 scripts/asr_caption.py --video lecture.mp4 --out-dir runs/lecture-dash
```
Override per-run with the same CLI flag (CLI > env > hardware_profile).

Example — DashScope Qwen3-ASR (cloud, Chinese SOTA, no local GPU):
```bash
export DASHSCOPE_API_KEY=sk-…
python3 scripts/asr_caption.py \
  --video lecture.mp4 --out-dir runs/lecture-dash \
  --backend openai-api \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-model qwen3-asr-flash \
  --api-key-env DASHSCOPE_API_KEY \
  --language zh
```

Example — OpenAI Whisper API:
```bash
export OPENAI_API_KEY=sk-…
python3 scripts/asr_caption.py \
  --video lecture.mp4 --out-dir runs/lecture-openai \
  --backend openai-api \
  --api-base https://api.openai.com/v1 \
  --api-model whisper-1 --language en
```

Notes:
- API key is **never** a CLI flag — it's read from the env var named by `--api-key-env`
  (default `OPENAI_API_KEY`). Reduces accidental leakage into shell history / logs.
- `--hotwords` becomes the OpenAI `prompt` parameter (≤ ~224 tokens; biases
  vocabulary the same way `--initial-prompt` / FunASR `hotword` do).
- The endpoint is asked for `verbose_json`; segments are converted verbatim.
- `language_probability` is reported as `1.0` since OpenAI / DashScope don't
  surface it; `--language` is passed through when set.
- `--device` / `--compute-type` are ignored (the cloud handles all that).

## Model sizing

The default `--model` / `--compute-type` / `--device` are **auto-selected from
your hardware profile** (`scripts/hardware_profile.py`) — you don't have to size
them by hand. The table below shows the RAM footprint at `compute_type=int8` so
you understand what the profile picker chose; see
`references/hardware-profiles.md` for the full profile → model mapping.

| Model | RAM (int8) | Word timestamps | Profile that picks it |
|---|---|---|---|
| `tiny` | ~0.5 GB | rough | `tiny` (RAM < 6 GB) |
| `base` | ~0.7 GB | ok | `low` (6–8 GB, no dGPU) |
| `small` | ~1.2 GB | good | `low-mac`, `mid` (8–16 GB) |
| `medium` | ~3.5 GB | great | `high` (16–32 GB) |
| `large-v3` | ~6.5 GB | best | `high-gpu`, `max` |

`asr_caption.py` **warns** (but does not block) if you force a heavy model on a
low-RAM profile. Override explicitly with `--model`.

**2026-08 note:** the whisper ladder still fits faster-whisper (CTranslate2).
`--model turbo` (large-v3-turbo, 809M) is the speed/accuracy sweet spot when
`large-v3` is too slow on your GPU — near-large quality at several times the
speed. Other 2026 SOTA models with different inference stacks (NVIDIA
Parakeet / Canary, IBM Granite Speech, Mistral Voxtral, Cohere Transcribe) are
still not wired in as defaults — for Chinese/multilingual we now expose
**FunASR** (`--backend funasr`, qwen3-asr / paraformer-zh) and **any
OpenAI-compatible cloud ASR** (`--backend openai-api`, e.g. DashScope,
OpenAI, Groq); see "Choose an ASR backend" above.

## Device & compute type

These come from the hardware profile. The defaults by profile:

- **Apple Silicon / CPU-only** (`mid`, `low`, `low-mac`): `device=cpu`,
  `compute=int8`. faster-whisper (CTranslate2) has limited Metal support; CPU
  int8 is the fastest, most reliable path on macOS.
- **16 GB+** (`high`): `device=auto`, `compute=int8_float16`.
- **NVIDIA** (`high-gpu`): `device=cuda`, `compute=float16` — large speedup.

Override per-run with `--device` and `--compute-type`.

## Language

- `--language zh` (default): assume Chinese. Good for Chinese lectures.
- `--language en`, `--language ja`, …: any Whisper language code.
- `--language auto`: auto-detect (slightly slower, can misfire on code-switching).

## Hotwords (jargon biasing)

`--hotwords` conditions transcription via faster-whisper's `initial_prompt` —
the single cheapest accuracy win for jargon-heavy lectures (person names,
formula symbols, product ids). Accepts comma/space/`、`-separated terms or a
file (`@terms.txt`, one per line):

```bash
python3 scripts/asr_caption.py --video lecture.mp4 --out-dir run \
  --language zh --hotwords "亥姆霍兹自由能, 格林函数, CTranslate2"
# or: --hotwords @terms.txt
```

## VAD filter

`vad_filter=True` is hardcoded — it trims silence, which dramatically improves
segment timestamps and speed for lectures with long pauses. Disable in
`asr_caption.py` only if you need exact wall-clock silence boundaries.

## Outputs

```
<out-dir>/
├── audio_16k.wav                # intermediate (16k mono, for the model)
├── subtitles.srt                # timestamped subtitles
├── subtitles.vtt                # WebVTT
└── subtitles.json               # {language, duration, segments:[{start,end,text}]}
```

`subtitles.json` is the canonical input to `build_knowledge.py`. `segments` is a
list of `{start,end,text}` — same schema Path 1 emits, so Step 2 is path-agnostic.

## End-to-end example

```bash
source ~/.zcode/skills/video2knowledge/.venv/bin/activate
cd ~/.zcode/skills/video2knowledge
python3 scripts/asr_caption.py \
  --video ~/Movies/lecture.mp4 \
  --out-dir runs/lecture-asr \
  --model small --language zh
python3 scripts/build_knowledge.py \
  --subtitles runs/lecture-asr/subtitles.json \
  --out-dir runs/lecture-asr --format all
python3 scripts/gen_apkg.py \
  --csv runs/lecture-asr/cards.csv \
  --out runs/lecture-asr/cards.apkg --deck "课程知识卡"
```

## Running both paths (cross-validation)

For high-value content, run Path 2 for accurate text + Path 1 for visual context,
then let the LLM in Step 2 merge them. Point `build_knowledge.py` at the ASR
subtitles (better text), and paste key multimodal captions into the prompt or a
custom template's `{{summary}}` slot.

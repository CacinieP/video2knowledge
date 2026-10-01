# Path 2 — ASR Transcription (faster-whisper)

Path 2 extracts the audio track and transcribes it with **faster-whisper** (the
CTranslate2 backend of Whisper), producing word/segment-level timestamps. This is
the recommended path whenever the video has a clear speech track.

> **Mandarin homophone trouble? Use the FunASR backend instead.**
> `scripts/asr_funasr.py` (Paraformer-large) is a drop-in replacement producing the
> same `subtitles.{srt,vtt,json}`. On a 30-minute Mandarin piano lesson,
> faster-whisper `small` *with 39 hotwords including both terms* got them wrong
> ~44% of the time (背谱→被谱 8×, 视谱→适谱 10×); Paraformer with the same hotwords
> scored 94%. It also ran 11–16× realtime on CPU, leaving the GPU free.
> See [FunASR backend](#funasr-backend-alternative) below.

## When to choose Path 2

- Lecture, talk, interview, podcast video, tutorial with narration.
- You need accurate, fine-grained timestamps (segment + word level).
- You want the highest fidelity text — Whisper models are SOTA for general speech.

Avoid Path 2 when: the video is silent/slide-only, audio is non-speech, or you
need what is visually on screen (use Path 1 or run both).

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
speed. Newer leaderboard models (NVIDIA Canary-Qwen 2.5B, Mistral Voxtral,
FireRedASR for Mandarin) need different inference stacks, so they are not
wired in as defaults.

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

## Which backend should I use?

Ask the machine:

```bash
python3 scripts/hardware_profile.py --recommend
```

It prints a backend and a reason, plus a ready-to-paste command. The
recommendation is advice only — nothing changes unless you pass it back.

On a 61.5-hour Mandarin course (286 videos), both engines were run over the
same 30-minute lecture with the same 39-term hotword list:

| | homophone terms correct | speed |
|---|---|---|
| faster-whisper `small` + hotwords | 44% (8 of 18) | 2.5× realtime |
| Paraformer-zh (FunASR, CPU) | **94%** (17 of 18) | 12–16× realtime |

Paraformer fixes things hotwords cannot: 背谱/被谱, 视谱/适谱, 穿指/川指 are
identical in pinyin *and* tone, so no acoustic model separates them — the
course vocabulary just has to be spelled correctly downstream
(`scripts/fix_homophones.py` does that).

The recommendation keys on **RAM, not VRAM**. Paraformer's advantage is
Mandarin accuracy and it runs on CPU, so a machine with a large GPU and a
large GPU with a modest machine reach the same answer. VRAM matters for a
different reason: leave it free for the VLM, because running whisper and the
VLM at once turns an 8-token request from 3.8 s into over 60 s.

Cloud backends are only suggested when the host genuinely cannot keep up
(< 6 GB RAM). Sending a course recording to a third party is a real privacy
decision, so it is the fallback and not the default.

## Cloud ASR backends

`scripts/asr_caption.py --backend` also accepts two cloud engines. They write
the same `subtitles.json` schema, so nothing downstream changes.

```bash
# any OpenAI-compatible /audio/transcriptions endpoint
export MY_ASR_KEY=...                      # never pass the key on argv
python3 scripts/asr_caption.py --video IN.mp4 --out-dir OUT \
  --backend openai-api \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-model qwen3-asr-flash \
  --api-key-env MY_ASR_KEY
```

`--api-base` is **not** a whitelist — any OpenAI-shaped URL works, including a
self-hosted whisper.cpp server or your own gateway.

`mimo-asr` is for gateways that expose a *chat* endpoint instead of the
transcription one. Those return no timestamps, so the clip is cut on
silence-aligned boundaries (`ffmpeg silencedetect`) and each chunk's
`[start, end]` becomes its segments' timestamps:

```bash
python3 scripts/asr_caption.py --video IN.mp4 --out-dir OUT \
  --backend mimo-asr --api-base URL --api-model mimo-v2.5-asr \
  --api-key-env MY_KEY --chunk-seconds 600 --concurrency 4
```

**The key is read from an environment variable only.** A key passed on the
command line lands in the process list, in shell history, and in every log
line that echoes the command.

## Cloud LLM

The same treatment applies to the text model. `build_knowledge.py` can route
`ask_llm()` to any OpenAI-compatible `/chat/completions` endpoint instead of
local Ollama:

```bash
export V2K_LLM_API_KEY=...     # or whatever --api-key-env names
python3 scripts/build_knowledge.py --subtitles OUT/subtitles.json \
  --out-dir OUT --api-base https://api.example.com/v1 --api-model some-model
```

Useful when the local machine is busy with ASR, or when a 2B local model is
not producing summaries you would trust. Reasoning models' `<think>` blocks
are stripped — left in, they land in the knowledge doc as a wall of internal
monologue. Cloud failures are reported on stderr and exit non-zero, never
turned into a placeholder-filled document that looks finished.

## FunASR backend (alternative)

`scripts/asr_funasr.py` transcribes with **Paraformer-large + FSMN-VAD + CT-punc**
and writes the same three files, so every downstream script works unchanged.

Install it in a **separate venv** — funasr pins an old `tokenizers` and brings its
own torch, and does not co-install with the main environment:

```bash
python3 -m venv .venv-funasr
./.venv-funasr/bin/pip install "tokenizers>=0.21" torch
./.venv-funasr/bin/pip install --no-deps funasr
```

Single clip, and the same via `batch_run.py`:

```bash
./.venv-funasr/bin/python scripts/asr_funasr.py \
  --video VIDEO --out-dir OUT --language zh --hotwords "背谱,视谱,音阶"

python3 scripts/batch_run.py --root LIBRARY --asr-backend funasr \
  --asr-python ./.venv-funasr/bin/python
```

`paraformer-zh` is Mandarin-only and CPU-only — it has no `--model` /
`--device` / `--compute-type`, so `batch_run.py` omits them for this backend.

### Transcribing a whole library in one model load

Model load costs 25–60 s. A process per clip pays that once per clip, which on a
297-clip library is ~2.1 h of pure loading against ~5.3 h of actual
transcription. `--jobs` takes a JSON list and skips anything already done:

```json
[{"video": "/lib/a.mp4", "out_dir": "/out/a"},
 {"video": "/lib/b.mp4", "out_dir": "/out/b", "redo": true}]
```

```bash
./.venv-funasr/bin/python scripts/asr_funasr.py --jobs jobs.json --language zh
```

Each finished run dir gets an `asr_engine.txt` recording which engine and
segmenter revision produced its transcript, so a later run can tell a
faster-whisper transcript from a FunASR one and re-do only what must be redone.

### Gotchas found by running it

- **Short model aliases matter.** `iic/speech_paraformer-...` + vad + ct-punc
  returns `{key, text}` with **no timestamps**; the whole transcript then
  collapses onto zero-length segments. The alias form (`paraformer-zh`,
  `fsmn-vad`, `ct-punc`) is the only combination that returns both timestamps
  and punctuation.
- **Punctuation is inserted after the fact.** ct-punc adds it to a string the
  ASR timestamped beforehand, so `text` is longer than `timestamp`. Walking both
  with one index silently drops the tail of every transcript.
- **Keep timestamps in one unit.** A punctuation mark borrows the previous
  character's end time; if that one value is pre-divided, every segment ending on
  。 / ？ / ！ fails its `end > start` check and is dropped. Measured: 62% of a
  30-minute lecture silently lost, with no error anywhere.

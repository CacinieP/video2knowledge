# Optional cloud ASR, text and vision

## Decide what leaves the machine

Local is the default workflow, not a guarantee that every existing environment
is local. Inspect `ASR_API_*`, `V2K_LLM_API_BASE`, `V2K_LLM_API_MODEL`,
`V2K_VLM_API_MODEL`, `V2K_CLOUD_PROTOCOL` and `OLLAMA_HOST`. If an inherited
endpoint conflicts with the user's local-only request, do not send a probe there.
Use an isolated invocation with the cloud endpoint variables unset and an
explicit loopback Ollama host, or report the conflict. Do not silently alter the
user's global environment.

| Step | Data sent when enabled |
|---|---|
| `asr_caption.py --backend openai-api` / `mimo-asr` | extracted audio / audio chunks |
| `build_knowledge.py --api-base` | subtitle or fused audio/OCR text |
| `build_notes.py --api-base` | narration text for headings/notes |
| cloud notes plus `--describe-frames` | selected frame images, plus text |
| remote `OLLAMA_HOST` | the text/images passed to that remote server |

Tell the user the selected destination and data. Use cloud only when it matches
the user's choice; an environment setting is not permission to override a
local-only request. Do not claim videos/frames never leave the machine without
checking the selected route. Do not put secret values in command arguments,
example manifests, logs or published files.

## OpenAI-compatible examples

Resolve `$PY`, `$SKILL_ROOT` and `$OUT` as in [runtime.md](runtime.md), in the same
shell invocation. Credentials must already exist in the named environment
variable; do not ask the user to paste them into chat.

```bash
# Audio upload, using the existing MY_ASR_KEY environment variable:
"$PY" "$SKILL_ROOT/scripts/asr_caption.py" --video "/absolute/path/to/lecture.mp4" --out-dir "$OUT" --backend openai-api --api-base "https://api.example.com/v1" --api-model "whisper-1" --api-key-env MY_ASR_KEY
# Text upload, using the existing MY_LLM_KEY environment variable:
"$PY" "$SKILL_ROOT/scripts/build_knowledge.py" --subtitles "$OUT/subtitles.json" --out-dir "$OUT" --format html --api-base "https://api.example.com/v1" --api-model "text-model" --api-key-env MY_LLM_KEY
# Text AND selected frame upload:
"$PY" "$SKILL_ROOT/scripts/build_notes.py" --subtitles "$OUT/subtitles.json" --frames "$OUT/frames/frames.json" --out-dir "$OUT" --describe-frames --api-base "https://api.example.com/v1" --api-model "text-model" --vlm-api-model "vision-model" --api-key-env MY_LLM_KEY
```

Examples use placeholder endpoints, not a provider recommendation. Cloud ASR
requires `setup_models.sh --with-openai-client` or equivalent SDK installation;
text/vision HTTP clients do not require that SDK. `mimo-asr` is a chat-shaped
ASR backend with silence-aligned chunks, concurrency and retries; use its actual
configured model/endpoint and `--api-key-env` rather than inventing timestamps.

`V2K_CLOUD_PROTOCOL=anthropic` changes the text/vision request protocol. The
current adapter can use a rotating desktop credential from the configured
MiniMax auth store. Use this only for that user's explicitly chosen integration;
do not treat it as a generic Anthropic API-key adapter or reuse a credential with
an unrelated destination. See `scripts/net.py` and the provider configuration
when diagnosing it; never print credential-store contents.

## Boundaries

`mm_caption.py` is an Ollama client. A cloud notes endpoint does not automatically
replace route-3 OCR. Unsupported vision capability is probed, then descriptions
may be skipped with a warning. Report missing descriptions as partial when they
were requested; do not claim remote text models saw images.

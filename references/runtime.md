# Runtime and installation

## Resolve paths once, pass them every time

Resolve `SKILL_ROOT` from the installed SKILL.md location. The user can invoke
this skill from any cwd. Use an absolute input path and output path; quote paths
with spaces. Do not depend on activation or variables surviving tool calls.
The following is one Bash invocation; repeat its bindings in each fresh shell.

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
PY="$SKILL_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$SKILL_ROOT/.venv/Scripts/python.exe"
"$PY" "$SKILL_ROOT/scripts/hardware_profile.py" --json
```

For a custom `VENV_DIR`, use setup's printed `run python as:` interpreter.
On Windows, use Git Bash with native Python/uv, not MSYS Python. Native
PowerShell users should pass the same absolute paths with PowerShell syntax.

## First-time setup

Inspect existing tools and local services first. Setup downloads models and
Python packages, may start `ollama serve`, and can take more than five minutes.
Respect the user's constraints; do not run setup when downloads/startup are forbidden.
From the skill root:

```bash
cd "/absolute/path/to/video2knowledge"
bash scripts/setup_models.sh
# Only when selecting Mandarin Paraformer:
bash scripts/setup_models.sh --with-funasr
# Only when choosing a cloud ASR backend needing the SDK:
bash scripts/setup_models.sh --with-openai-client
```

Default packages come from `requirements.txt`; FunASR uses a separate venv.
System prerequisites: Python 3.11+ (or uv with an appropriate native Python),
ffmpeg + ffprobe; local model steps also need Ollama. Examples:

- macOS: `brew install ffmpeg python@3.11 ollama`.
- Debian/Ubuntu: install `ffmpeg python3 python3-venv`, plus Ollama from its
  official distribution. For Chinese PDF export install `fonts-wqy-microhei`.
- Windows: native Python, ffmpeg and Ollama; use `.venv/Scripts/python.exe`.

ASR model names can download weights on first use. If downloads are forbidden,
verify the local ASR model files and pass their existing model directory, or
report the missing weights before execution. An installed Python package alone
does not mean the model is available.

Do not require Ollama for an explicitly cloud-only step or a no-model export.
The setup script still prepares a local VLM; do not run it blindly for a
cloud-only or offline-no-download request. Report missing dependencies precisely.

## Preflight

Check input existence and `ffprobe` streams/duration. Check available disk and
model service/model inventory before starting an expensive run. Read the JSON
hardware profile to obtain `vlm_model` and `text_model` and pass them explicitly
where the recipes say `$VLM` and `$TEXT`. A profile recommendation or HTTP 200
model list is not proof of successful inference: try a short request first.

For a local-only invocation, isolate inherited cloud settings inside that shell
ONLY (repeat in each fresh call), and explicitly use loopback Ollama:

```bash
unset V2K_LLM_API_BASE V2K_LLM_API_MODEL V2K_VLM_API_MODEL V2K_CLOUD_PROTOCOL
export OLLAMA_HOST="http://127.0.0.1:11434"
```

Inspect cloud-related environment settings before claiming processing is local;
see [cloud.md](cloud.md). Resolve the FunASR interpreter independently:

```bash
SKILL_ROOT="/absolute/path/to/video2knowledge"
ASR_PY="$SKILL_ROOT/.venv-funasr/bin/python"
[ -x "$ASR_PY" ] || ASR_PY="$SKILL_ROOT/.venv-funasr/Scripts/python.exe"
"$ASR_PY" "$SKILL_ROOT/scripts/asr_funasr.py" --help
```

Never write `python3 .venv-funasr/bin/python ...`: that asks Python to parse an
interpreter binary as a script. Do not silently replace a missing local backend
with cloud processing.

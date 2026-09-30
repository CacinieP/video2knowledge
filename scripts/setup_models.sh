#!/usr/bin/env bash
# setup_models.sh — idempotent check/download of models for video2knowledge
#
# Path 1 (multimodal): ollama pulls the profile-picked VLM (2026-08 lineup:
#                      qwen3.5 0.8b/4b/9b, minicpm-v4.6 on low/mid, qwen3.8:27b on max)
# Path 2 (ASR):        faster-whisper installed into a local venv; model weights
#                      auto-download on first transcription to ~/.cache/huggingface
#
# Cross-platform: works in bash on Linux, macOS, and Windows (Git Bash / MSYS).
# Never activates the venv (bin/ vs Scripts/ layout differs); every python call
# goes through an absolute interpreter path resolved once below.
#
# Safe to re-run: existing artifacts are skipped. Prints clear status for traceability.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
# venv lives in the repo/run root next to this script — wherever the skill is
# installed (e.g. ~/.agents/skills/video2knowledge), not a hardcoded HOME path.
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

# --- cross-platform helpers ---------------------------------------------------
log() { printf '[setup] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Windows venvs use Scripts/, POSIX venvs use bin/. Resolve once.
venv_python() {
  if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    printf '%s\n' "$VENV_DIR/Scripts/python.exe"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    printf '%s\n' "$VENV_DIR/bin/python"
  else
    return 1
  fi
}

# Interpreter used for hardware profiling only (the venv python once it exists).
profile_python() {
  if VENV_PY="$(venv_python)"; then printf '%s\n' "$VENV_PY"; return 0; fi
  if have python3; then command -v python3; return 0; fi
  if have python;  then command -v python;  return 0; fi
  return 1
}

# --- auto-detect hardware profile (scripts/hardware_profile.py) -------------
# Reads the single source of truth. Override any field with env vars if needed.
if PROFILE_PY="$(profile_python)"; then
  PROFILE_JSON="$("$PROFILE_PY" "$HERE/hardware_profile.py" --json)"
  hp() { "$PROFILE_PY" -c "import sys,json;print(json.loads('''$PROFILE_JSON''').get('$1',''))"; }
  : "${VLM_MODEL:=$(hp vlm_model)}"
  : "${ASR_DEFAULT_MODEL:=$(hp asr_model)}"
  : "${ASR_COMPUTE_TYPE:=$(hp compute_type)}"
  : "${ASR_DEVICE:=$(hp device)}"
  HP_PROFILE="$(hp profile)"
else
  : "${VLM_MODEL:=openbmb/minicpm-v4.6:latest}"
  : "${ASR_DEFAULT_MODEL:=small}"
  : "${ASR_COMPUTE_TYPE:=int8}"
  : "${ASR_DEVICE:=cpu}"
  HP_PROFILE="(python missing — defaults)"
fi

# --- 1. ollama + vision model -------------------------------------------------
# Git Bash / MSYS does not export USERPROFILE (it uses HOME), but the Windows
# ollama CLI's envconfig panics with "%userprofile% is not defined" when it
# expands default paths. Map it from HOME before touching ollama.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if [ -z "${USERPROFILE:-}" ]; then
      if have cygpath; then
        export USERPROFILE="$(cygpath -w "$HOME")"
      else
        export USERPROFILE="$(printf '%s' "$HOME" | sed 's|^/\([a-zA-Z]\)/|\1:/|; s|/|\\\\|g')"
      fi
    fi
    ;;
esac

if ! have ollama; then
  log "ERROR: ollama not found. Install:"
  log "  Linux/macOS: curl -fsSL https://ollama.com/install.sh | sh"
  log "  Windows:     winget install Ollama.Ollama  (or download from ollama.com)"
  exit 1
fi
# Probe the daemon by asking it something; pgrep is not portable (no pgrep in
# Git Bash) and a process name check says nothing about readiness.
if ! ollama list >/dev/null 2>&1; then
  log "ollama daemon not reachable — starting (background)..."
  ollama serve >"${TMPDIR:-/tmp}/ollama.log" 2>&1 &
  for _ in $(seq 1 15); do
    ollama list >/dev/null 2>&1 && break
    sleep 1
  done
  ollama list >/dev/null 2>&1 || { log "ERROR: ollama serve did not come up (see ${TMPDIR:-/tmp}/ollama.log)"; exit 1; }
fi

# ollama list prints names with tags (e.g. "openbmb/minicpm-v4.6:latest").
# Match on the base name to be tag-tolerant.
VLM_BASE="${VLM_MODEL%%:*}"
if ollama list 2>/dev/null | awk '{print $1}' | grep -qE "^${VLM_BASE}(:|@)"; then
  log "VLM already present: $VLM_MODEL (skipping pull)"
else
  log "Pulling VLM: $VLM_MODEL ..."
  ollama pull "$VLM_MODEL"
fi

# --- 2. python venv + faster-whisper -----------------------------------------
if ! have uv && ! have python3 && ! have python; then
  log "ERROR: need uv or python3 to build venv"; exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR ..."
  if have uv; then
    # uv picks a native toolchain python — on Windows Git Bash the PATH python
    # may be an MSYS/mingw build whose venvs (bin/ layout, no ctranslate2
    # wheels) cannot host faster-whisper. uv sidesteps that entirely.
    uv venv "$VENV_DIR" >/dev/null
    uv pip install --python "$(venv_python)" --quiet "faster-whisper>=1.0.3" genanki python-docx fpdf2
  else
    PY_BIN="$(command -v python3 || command -v python)"
    case "$(uname -s)" in
      MINGW*|MSYS*|CYGWIN*)
        case "$PY_BIN" in
          /mingw64/*|/usr/*|/clang64/*|/clangarm64/*)
            log "ERROR: on Windows the venv must be built by NATIVE python, but"
            log "  '$PY_BIN' is an MSYS/mingw interpreter — its venvs cannot host"
            log "  faster-whisper (no ctranslate2 wheels). Install uv (preferred):"
            log "    winget install astral-sh.uv"
            log "  or point PATH at native python (python.org / conda) and retry."
            exit 1
            ;;
        esac
        ;;
    esac
    "$PY_BIN" -m venv "$VENV_DIR"
    "$(venv_python)" -m pip install --quiet --upgrade pip
    "$(venv_python)" -m pip install --quiet "faster-whisper>=1.0.3" genanki python-docx fpdf2
  fi
  log "Installed faster-whisper + genanki into venv"
else
  log "venv exists: $VENV_DIR (skipping create)"
fi
VENV_PY="$(venv_python)" || { log "ERROR: venv dir exists but no python found inside"; exit 1; }
"$VENV_PY" -c "import faster_whisper" 2>/dev/null || {
  log "venv missing faster-whisper — installing..."
  if have uv; then
    uv pip install --python "$VENV_PY" --quiet "faster-whisper>=1.0.3" genanki python-docx fpdf2
  else
    "$VENV_PY" -m pip install --quiet "faster-whisper>=1.0.3" genanki python-docx fpdf2
  fi
}

# --- 2b. CUDA runtime for faster-whisper ------------------------------------
# The PyPI ctranslate2 wheel links against CUDA 12, but the runtime libraries
# are NOT bundled: on Windows they are not on PATH even with a working driver,
# so `--device cuda` dies with
#   RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
# The nvidia-*-cu12 wheels put them under site-packages/nvidia/<pkg>/bin, which
# asr_caption.py adds to the DLL search path itself. Installing them here is
# what turns a 4-7x slower CPU run into a GPU one. ~1.5 GB, no-op on a CPU-only
# box (hardware_profile reports cuda_ok=false and stays on CPU).
if "$VENV_PY" -c "import glob,os,sys; sys.exit(0 if glob.glob(os.path.join(sys.prefix,'Lib','site-packages','nvidia','cublas','bin','cublas64_12.dll')) or glob.glob(os.path.join(sys.prefix,'lib','python*','site-packages','nvidia','cublas','lib','libcublas.so*')) else 1)" 2>/dev/null; then
  log "CUDA runtime: already present (skipping)"
elif have nvidia-smi; then
  log "NVIDIA GPU detected — installing the CUDA 12 runtime for faster-whisper..."
  if have uv; then
    uv pip install --python "$VENV_PY" --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12 2>/dev/null \
      || log "WARN: CUDA runtime install failed — ASR will fall back to CPU (slower but works)"
  else
    "$VENV_PY" -m pip install --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12 2>/dev/null \
      || log "WARN: CUDA runtime install failed — ASR will fall back to CPU (slower but works)"
  fi
else
  log "No NVIDIA GPU — skipping the CUDA runtime (CPU ASR)"
fi

# docx/pdf export libs (optional at runtime; build_knowledge --format docx/pdf,
# build_notes --docx/--pdf degrade with a hint when absent)
"$VENV_PY" -c "import docx, fpdf" 2>/dev/null || {
  log "venv missing python-docx/fpdf2 — installing (docx/pdf export)..."
  if have uv; then
    uv pip install --python "$VENV_PY" --quiet python-docx fpdf2
  else
    "$VENV_PY" -m pip install --quiet python-docx fpdf2
  fi
}

# --- 3. ffmpeg ---------------------------------------------------------------
if ! have ffmpeg; then
  log "ERROR: ffmpeg not found. Install:"
  log "  macOS:   brew install ffmpeg"
  log "  Linux:   sudo apt install ffmpeg"
  log "  Windows: winget install Gyan.FFmpeg"
  exit 1
fi

cat <<EOF

[setup] DONE
  profile       : $HP_PROFILE
    -> VLM      : $VLM_MODEL
    -> ASR      : faster-whisper '$ASR_DEFAULT_MODEL' (compute=$ASR_COMPUTE_TYPE, device=$ASR_DEVICE)
  venv          : $VENV_DIR
  run python as : $VENV_PY
  Override with : VLM_MODEL=... ASR_DEFAULT_MODEL=... bash setup_models.sh
EOF

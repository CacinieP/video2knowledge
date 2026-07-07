#!/usr/bin/env bash
# setup_models.sh — idempotent check/download of models for video2knowledge
#
# Path 1 (multimodal): ollama pulls openbmb/minicpm-v4.6 (<=4B native VLM)
# Path 2 (ASR):        faster-whisper installed into a local venv; model weights
#                      auto-download on first transcription to ~/.cache/huggingface
#
# Safe to re-run: existing artifacts are skipped. Prints clear status for traceability.
set -euo pipefail

VLM_MODEL="${VLM_MODEL:-openbmb/minicpm-v4.6:latest}"
VENV_DIR="${VENV_DIR:-$HOME/.zcode/skills/video2knowledge/.venv}"
ASR_DEFAULT_MODEL="${ASR_DEFAULT_MODEL:-small}"   # small fits 8GB RAM; bump to medium/large-v3 if you have >=16GB

log() { printf '[setup] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. ollama + vision model -------------------------------------------------
if ! have ollama; then
  log "ERROR: ollama not found. Install: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  log "ollama daemon not running — starting (background)..."
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
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
if ! have uv && ! have python3; then
  log "ERROR: need uv or python3 to build venv"; exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv at $VENV_DIR ..."
  if have uv; then
    uv venv "$VENV_DIR" >/dev/null
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    uv pip install --quiet "faster-whisper>=1.0.3" genanki
  else
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet "faster-whisper>=1.0.3" genanki
  fi
  log "Installed faster-whisper + genanki into venv"
else
  log "venv exists: $VENV_DIR (skipping create)"
fi

# --- 3. ffmpeg ---------------------------------------------------------------
if ! have ffmpeg; then
  log "ERROR: ffmpeg not found. Install: brew install ffmpeg"
  exit 1
fi

cat <<EOF

[setup] DONE
  VLM  : $VLM_MODEL
  ASR  : faster-whisper (default model '$ASR_DEFAULT_MODEL', downloaded on first run)
  venv : $VENV_DIR
  Activate with:  source $VENV_DIR/bin/activate
EOF

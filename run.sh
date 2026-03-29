#!/usr/bin/env bash
# run.sh — Start Ali Real Estate Chatbot (with Voice)
# Usage:  chmod +x run.sh && ./run.sh
#         ./run.sh --stop   ← kill everything

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.ali_pids"
MODEL_NAME="ali-realestate"

# ── Voice model config ────────────────────────────────────────────────────────
VOICES_DIR="$SCRIPT_DIR/backend/Voice/models"
PIPER_VOICE_NAME="en_US-lessac-medium"
PIPER_ONNX="$VOICES_DIR/${PIPER_VOICE_NAME}.onnx"
PIPER_JSON="$VOICES_DIR/${PIPER_VOICE_NAME}.onnx.json"
PIPER_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping Ali..."
  if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null && echo "  killed $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  echo "Done."
  exit 0
fi

> "$PID_FILE"   # clear stale pids

# ── 1. Start Ollama ───────────────────────────────────────────────────────────
echo "[1/4] Starting Ollama..."
if ! curl -sf http://localhost:11434 &>/dev/null; then
  ollama serve > /tmp/ollama.log 2>&1 &
  echo $! >> "$PID_FILE"
  sleep 3
  echo "      Ollama started."
else
  echo "      Ollama already running."
fi

# ── 2. Create LLM model ───────────────────────────────────────────────────────
echo "[2/4] Creating model $MODEL_NAME..."
ollama create "$MODEL_NAME" -f "$SCRIPT_DIR/backend/Ollama/Modelfile"
echo "      Model ready."

# ── 3. Download piper voice model (first run only) ───────────────────────────
echo "[3/4] Checking piper voice model..."
mkdir -p "$VOICES_DIR"

if [[ ! -f "$PIPER_ONNX" ]]; then
  echo "      Downloading ${PIPER_VOICE_NAME}.onnx (~65 MB)..."
  curl -L --progress-bar \
    "${PIPER_BASE_URL}/${PIPER_VOICE_NAME}.onnx" \
    -o "$PIPER_ONNX"
fi

if [[ ! -f "$PIPER_JSON" ]]; then
  echo "      Downloading ${PIPER_VOICE_NAME}.onnx.json..."
  curl -L --progress-bar \
    "${PIPER_BASE_URL}/${PIPER_VOICE_NAME}.onnx.json" \
    -o "$PIPER_JSON"
fi

echo "      Piper voice ready at $PIPER_ONNX"

# ── 4. Start FastAPI ──────────────────────────────────────────────────────────
echo "[4/4] Starting FastAPI..."
cd "$SCRIPT_DIR/backend/api"
PIPER_VOICE="$PIPER_ONNX" \
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload > /tmp/ali_api.log 2>&1 &
echo $! >> "$PID_FILE"

sleep 3
echo ""
echo "  Ali is running!"
echo "  Chat UI  →  http://localhost:8000"
echo "  API Docs →  http://localhost:8000/docs"
echo "  Voice    →  ASR (faster-whisper base) + TTS (piper ${PIPER_VOICE_NAME})"
echo "  Stop     →  ./run.sh --stop"

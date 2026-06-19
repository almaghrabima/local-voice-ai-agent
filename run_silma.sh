#!/usr/bin/env bash
# Launch the silma TTS sidecar + the voice agent together.
# The sidecar runs in the tts-benchmark venv (which has f5_tts); the agent
# runs in this project's .venv (which has fastrtc). Stops both on Ctrl-C.
set -euo pipefail

# Usage: ./run_silma.sh [voice] [stt]
#   voice: auto|en|ar   (default: auto)
#   stt:   whisper|nemotron   (default: whisper)
HERE="$(cd "$(dirname "$0")" && pwd)"
TTS_PY="$HOME/Documents/tts-benchmark/.venvs/tts/bin/python"
AGENT_PY="$HERE/.venv/bin/python"
VOICE="${1:-auto}"
STT="${2:-whisper}"

[ -x "$TTS_PY" ]   || { echo "missing $TTS_PY"; exit 1; }
[ -x "$AGENT_PY" ] || { echo "missing $AGENT_PY"; exit 1; }

echo "[run_silma] starting silma sidecar..."
PYTHONUNBUFFERED=1 "$TTS_PY" "$HERE/silma_tts_server.py" &
SIDECAR_PID=$!
trap 'echo "[run_silma] stopping..."; kill $SIDECAR_PID 2>/dev/null || true' EXIT

echo "[run_silma] waiting for sidecar /health (model load can take a minute)..."
for _ in $(seq 1 180); do
  if curl -s -o /dev/null "http://127.0.0.1:8001/health"; then
    echo "[run_silma] sidecar ready."
    break
  fi
  kill -0 $SIDECAR_PID 2>/dev/null || { echo "[run_silma] sidecar died"; exit 1; }
  sleep 1
done

echo "[run_silma] starting voice agent (voice=$VOICE, stt=$STT)..."
PYTHONUNBUFFERED=1 "$AGENT_PY" "$HERE/local_voice_chat_silma.py" --voice "$VOICE" --stt "$STT"

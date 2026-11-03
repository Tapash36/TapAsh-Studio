#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[1/3] TapAsh-Studio: Generating synthetic verification track (test_song.wav)..."
"$PYTHON_BIN" scripts/generate_test_audio.py

echo "[2/3] Launching TapAsh-Studio Backend on http://localhost:8000 ..."
"$PYTHON_BIN" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[3/3] Launching TapAsh-Studio Frontend on http://localhost:3000 ..."
"$PYTHON_BIN" -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

echo "================================================================"
echo " TapAsh-Studio Live! Open: http://localhost:3000"
echo " Backend API:              http://localhost:8000/health"
echo "================================================================"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT
wait

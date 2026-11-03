@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

if not exist "test_song.wav" (
    echo [1/3] Generating synthetic verification track ^(test_song.wav^)...
    "%PYTHON_BIN%" scripts\generate_test_audio.py
)

echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "ChordStudioBackend" /B "%PYTHON_BIN%" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

echo [2/2] Starting Studio Frontend on http://localhost:3000 ...
start "ChordStudioFrontend" /B "%PYTHON_BIN%" -m http.server 3000 --directory frontend

echo ================================================================
echo  Studio Live in ~2 seconds!
echo  Frontend UI:  http://localhost:3000
echo  Backend API:  http://localhost:8000/health
echo ================================================================

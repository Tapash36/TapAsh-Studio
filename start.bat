@echo off
setlocal
title TapAsh-Studio
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

if not exist "test_song.wav" (
    echo [TapAsh-Studio] Generating synthetic verification track ^(test_song.wav^)...
    "%PYTHON_BIN%" scripts\generate_test_audio.py
)

echo [1/2] Starting TapAsh-Studio Backend on http://localhost:8000 ...
start "TapAsh-Studio-Backend" /B "%PYTHON_BIN%" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

echo [2/2] Starting TapAsh-Studio Frontend on http://localhost:3000 ...
start "TapAsh-Studio-Frontend" /B "%PYTHON_BIN%" -m http.server 3000 --directory frontend

echo ================================================================
echo  TapAsh-Studio Live in ~2 seconds!
echo  Frontend UI:  http://localhost:3000
echo  Backend API:  http://localhost:8000/health
echo ================================================================

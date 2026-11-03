---
title: Chord & 6-Stem Audio Studio
emoji: 🎸
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
---

# Real-Time Chord Recognition, Scale/Pitch Transposition & 6-Stem Audio Separation Studio

A full-stack AI audio engineering studio powered by **Meta Hybrid Transformer Demucs (`htdemucs_6s`)**, **Librosa Constant-Q Transform (`chroma_cqt`)**, **Krumhansl-Schmuckler Key Profiles**, and a **60fps Web Audio API** dark-mode interface (with a built-in 24/7 Cloud Web-DSP fallback engine).

## Live Links
- **24/7 Cloud Studio (Direct App URL):** [https://tapash36-chord-stem-studio.static.hf.space](https://tapash36-chord-stem-studio.static.hf.space)
- **Hugging Face Space:** [https://huggingface.co/spaces/tapash36/chord-stem-studio](https://huggingface.co/spaces/tapash36/chord-stem-studio)
- **GitHub Repository:** [https://github.com/Tapash36/chord-stem-studio](https://github.com/Tapash36/chord-stem-studio)

---

## Core Features
1. **Accurate Time-Synced Chord & Scale Detection (`backend/app.py`)**
   - Harmonic/percussive separation via `librosa.effects.harmonic(y, margin=4.0)` and Constant-Q Transform chromagram (`chroma_cqt`).
   - Temporal median filtering and normalized 12-pitch-class template matching across **Major, Minor, Dominant 7th, Major 7th, and Minor 7th** chords.
   - Overarching musical key/scale estimation using **Krumhansl-Schmuckler** key profiles.
2. **6-Stem AI Audio Separation (`htdemucs_6s`)**
   - Isolates `vocals`, `guitar`, `drums`, `bass`, `piano`, and `other` stems as 16-bit PCM WAV files served with full CORS headers.
3. **Real-Time + Phase-Vocoder Pitch & Scale Transposition**
   - **Instant Frontend Transposition (`-12` to `+12` ST):** Modulo-12 scale/chord transposition + Web Audio `AudioBufferSourceNode.detune` (`100` cents/semitone on melodic stems, `0` cents on `drums`).
   - **High-Quality Backend Phase-Vocoder (`/api/transpose-audio`):** Tempo-preserving pitch shift via `librosa.effects.pitch_shift` with seamless playhead hot-swapping.
4. **6-Stem Mixer Matrix & 60fps Chord HUD (`frontend/index.html`)**
   - Per-stem Mute (`M`), Solo (`S`), smooth gain slider (`0.0` to `1.2`), live RMS level meters, and direct `.WAV` stem export.

---

## Running Locally in VS Code

1. Open this folder in **VS Code**:
   ```bash
   code .
   ```
2. Create the virtual environment and install dependencies (if running on a fresh machine):
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r backend\requirements.txt
   ```
3. Launch the studio:
   - **Option A (VS Code Debugger):** Press **`F5`** in VS Code (uses `.vscode/launch.json`).
   - **Option B (VS Code Task):** Press **`Ctrl+Shift+B`** to run both Backend (`http://localhost:8000`) and Frontend (`http://localhost:3000`).
   - **Option C (One-Click Script):** Double-click `start.bat` (Windows) or run `./start.sh` (macOS/Linux).

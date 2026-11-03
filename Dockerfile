FROM python:3.11-slim

# Install native system ffmpeg and libsndfile for fast audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU PyTorch + Torchaudio first (avoids 4GB CUDA downloads on CPU Spaces)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    grep -v -E "^(demucs|torch|torchaudio|static-ffmpeg)" backend/requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt && \
    pip install --no-cache-dir --no-deps demucs

# Copy application code
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

# Set writable cache & jobs directories for non-root container user (UID 1000 on Hugging Face Spaces)
ENV TORCH_HOME=/app/.cache/torch
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
RUN mkdir -p /app/jobs /app/.cache/torch /tmp/numba_cache && \
    chmod -R 777 /app /tmp/numba_cache

# Generate synthetic demo track, pre-download htdemucs_6s weights, and pre-render demo_track stems
RUN python scripts/generate_test_audio.py && \
    python -c "from pathlib import Path; from backend.app import get_demucs_model, process_audio_file_internal; get_demucs_model(); process_audio_file_internal(Path('test_song.wav'), 'demo_track')" && \
    chmod -R 777 /app/jobs /app/.cache

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]

import os
import sys
import uuid
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Ensure torchaudio uses the soundfile/dispatcher backend before importing torchaudio or demucs
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = BASE_DIR / ".bin"
JOBS_DIR = BASE_DIR / "jobs"
BIN_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_ffmpeg_on_path() -> Optional[str]:
    """Ensure ffmpeg binary is available on system PATH using static-ffmpeg or imageio-ffmpeg."""
    if str(BIN_DIR) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{BIN_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    # 1. Try imageio_ffmpeg (fast wheel with embedded binary)
    try:
        import imageio_ffmpeg

        exe_path = imageio_ffmpeg.get_ffmpeg_exe()
        if exe_path and os.path.exists(exe_path):
            target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            target_path = BIN_DIR / target_name
            if not target_path.exists():
                shutil.copy2(exe_path, target_path)
            return str(target_path)
    except Exception as exc:
        logging.warning("imageio_ffmpeg setup warning: %s", exc)

    # 2. Try static_ffmpeg
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        found = shutil.which("ffmpeg")
        if found:
            return found
    except Exception as exc:
        logging.warning("static_ffmpeg setup warning: %s", exc)

    return shutil.which("ffmpeg")


FFMPEG_PATH = ensure_ffmpeg_on_path()

import numpy as np
import scipy.ndimage
import soundfile as sf
import librosa
import torch
import torchaudio

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chord-stem-studio")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
STEM_NAMES = ["vocals", "guitar", "drums", "bass", "piano", "other"]
MELODIC_STEMS = ["vocals", "guitar", "bass", "piano", "other"]

# Krumhansl-Schmuckler key profiles
KS_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
KS_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


class CORSStaticFiles(StaticFiles):
    """StaticFiles subclass that injects full CORS headers on all audio/WAV responses."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(
    title="Real-Time Chord Recognition, Scale/Pitch Transposition & 6-Stem Studio API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def inject_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


app.mount("/stems", CORSStaticFiles(directory=str(JOBS_DIR)), name="stems")
app.mount("/static", CORSStaticFiles(directory=str(JOBS_DIR)), name="static")

# Global cached Demucs model instance
_DEMUCS_MODEL = None


def get_demucs_model():
    """Load and cache the Hybrid Transformer Demucs 6-stem model (htdemucs_6s)."""
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        logger.info("Loading Demucs htdemucs_6s model...")
        from demucs.pretrained import get_model

        model = get_model("htdemucs_6s")
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        _DEMUCS_MODEL = model
        logger.info("Demucs htdemucs_6s model loaded successfully (sources: %s)", model.sources)
    return _DEMUCS_MODEL


def build_chord_templates() -> List[Dict[str, Any]]:
    """
    Build normalized 12-pitch-class templates for:
    - Major ("")
    - Minor ("m")
    - Dominant 7th ("7")
    - Major 7th ("maj7")
    - Minor 7th ("m7")
    """
    chord_defs = [
        ("", [0, 4, 7], [1.05, 1.0, 0.95], False, None),
        ("m", [0, 3, 7], [1.05, 1.0, 0.95], False, None),
        ("7", [0, 4, 7, 10], [1.0, 0.95, 0.85, 0.95], True, 10),
        ("maj7", [0, 4, 7, 11], [1.0, 0.95, 0.85, 0.95], True, 11),
        ("m7", [0, 3, 7, 10], [1.0, 0.95, 0.85, 0.95], True, 10),
    ]
    templates = []
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for suffix, intervals, weights, is_seventh, seventh_interval in chord_defs:
            vec = np.zeros(12, dtype=np.float64)
            for interval, w in zip(intervals, weights):
                vec[(root_idx + interval) % 12] = w
            vec /= np.linalg.norm(vec) + 1e-12
            templates.append(
                {
                    "name": f"{root_name}{suffix}",
                    "root": root_idx,
                    "suffix": suffix,
                    "vec": vec,
                    "is_seventh": is_seventh,
                    "seventh_pc": (root_idx + seventh_interval) % 12 if seventh_interval is not None else None,
                }
            )
    return templates


CHORD_TEMPLATES = build_chord_templates()


def detect_key_krumhansl(chroma_matrix: np.ndarray) -> str:
    """
    Estimate the overarching Musical Key/Scale using Krumhansl-Schmuckler key profiles.
    Returns strings like 'C Major' or 'A Minor'.
    """
    if chroma_matrix.size == 0:
        return "C Major"

    chroma_profile = np.mean(chroma_matrix, axis=1)
    if np.max(chroma_profile) <= 1e-8:
        return "C Major"

    c_norm = chroma_profile - np.mean(chroma_profile)
    c_std = np.linalg.norm(c_norm) + 1e-12

    maj_norm = KS_MAJOR_PROFILE - np.mean(KS_MAJOR_PROFILE)
    maj_std = np.linalg.norm(maj_norm) + 1e-12

    min_norm = KS_MINOR_PROFILE - np.mean(KS_MINOR_PROFILE)
    min_std = np.linalg.norm(min_norm) + 1e-12

    best_score = -1e9
    best_key = "C Major"

    for i, note in enumerate(NOTE_NAMES):
        maj_rot = np.roll(maj_norm, i)
        maj_corr = float(np.dot(c_norm, maj_rot) / (c_std * maj_std))
        if maj_corr > best_score:
            best_score = maj_corr
            best_key = f"{note} Major"

        min_rot = np.roll(min_norm, i)
        min_corr = float(np.dot(c_norm, min_rot) / (c_std * min_std))
        if min_corr > best_score:
            best_score = min_corr
            best_key = f"{note} Minor"

    return best_key


def transpose_scale_name(scale_str: str, semitones: int) -> str:
    """Transpose a scale string like 'C Major' or 'A Minor' by `semitones`."""
    if not scale_str or " " not in scale_str:
        return scale_str
    parts = scale_str.split(" ", 1)
    root, mode = parts[0], parts[1]
    if root not in NOTE_NAMES:
        return scale_str
    new_idx = (NOTE_NAMES.index(root) + semitones) % 12
    return f"{NOTE_NAMES[new_idx]} {mode}"


def detect_chords_and_scale(audio_path: Path) -> Dict[str, Any]:
    """
    Perform harmonic/percussive source separation, Constant-Q Transform chromagram,
    median filtering across time, Krumhansl-Schmuckler key estimation, and template-based
    chord recognition (Major, Minor, Dom7, Maj7, Min7).
    """
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr) if sr > 0 else 0.0
    if len(y) == 0:
        return {
            "detected_scale": "C Major",
            "duration": 0.0,
            "chords": [{"start": 0.0, "end": 1.0, "chord": "C"}],
        }

    # 1. Separate harmonic content
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)

    # 2. Constant-Q Transform chromagram
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)

    # 3. Median filtering across time to eliminate transient jitter
    filter_width = min(21, max(3, (chroma.shape[1] // 8) | 1))
    chroma_filtered = scipy.ndimage.median_filter(chroma, size=(1, filter_width))

    # 4. Estimate overarching key/scale via Krumhansl-Schmuckler profiles
    detected_scale = detect_key_krumhansl(chroma_filtered)

    # 5. Match frames/windows against normalized 12-pitch-class templates
    n_frames = chroma_filtered.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames + 1), sr=sr, hop_length=hop_length)
    frame_times[-1] = duration

    # Use 0.25s analysis blocks for crisp, stable chord boundaries
    block_dur = 0.25
    n_blocks = max(1, int(np.ceil(duration / block_dur)))
    raw_segments: List[Dict[str, Any]] = []

    for b in range(n_blocks):
        t_start = round(b * block_dur, 3)
        t_end = round(min(duration, (b + 1) * block_dur), 3)
        if t_end <= t_start:
            continue

        f_start = int(librosa.time_to_frames(t_start, sr=sr, hop_length=hop_length))
        f_end = int(librosa.time_to_frames(t_end, sr=sr, hop_length=hop_length))
        f_start = max(0, min(f_start, n_frames - 1))
        f_end = max(f_start + 1, min(f_end, n_frames))

        vec = np.mean(chroma_filtered[:, f_start:f_end], axis=1)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            chord_label = raw_segments[-1]["chord"] if raw_segments else "N"
        else:
            vec_unit = vec / norm
            best_chord = "C"
            best_score = -1.0
            for tpl in CHORD_TEMPLATES:
                score = float(np.dot(vec_unit, tpl["vec"]))
                if tpl["is_seventh"]:
                    root_val = vec_unit[tpl["root"]]
                    sev_val = vec_unit[tpl["seventh_pc"]]
                    # Only prefer 7th template if the 7th interval has clear harmonic presence
                    if sev_val < 0.38 * max(root_val, 0.25):
                        score *= 0.88
                    else:
                        score *= 0.96
                if score > best_score:
                    best_score = score
                    best_chord = tpl["name"]
            chord_label = best_chord

        if raw_segments and raw_segments[-1]["chord"] == chord_label:
            raw_segments[-1]["end"] = t_end
        else:
            raw_segments.append({"start": t_start, "end": t_end, "chord": chord_label})

    # Merge micro-segments (< 0.45s) into adjacent segments to prevent boundary jitter
    if len(raw_segments) > 1:
        cleaned: List[Dict[str, Any]] = []
        for seg in raw_segments:
            seg_len = seg["end"] - seg["start"]
            if seg_len < 0.45 and cleaned:
                cleaned[-1]["end"] = seg["end"]
            else:
                cleaned.append(seg)
        # Check if first segment was tiny
        if len(cleaned) > 1 and (cleaned[0]["end"] - cleaned[0]["start"]) < 0.45:
            cleaned[1]["start"] = cleaned[0]["start"]
            cleaned.pop(0)
        # Re-merge consecutive identical chords after cleanup
        merged: List[Dict[str, Any]] = []
        for seg in cleaned:
            if merged and merged[-1]["chord"] == seg["chord"]:
                merged[-1]["end"] = round(seg["end"], 2)
            else:
                merged.append(
                    {
                        "start": round(seg["start"], 2),
                        "end": round(seg["end"], 2),
                        "chord": seg["chord"],
                    }
                )
        raw_segments = merged

    # Refine key tie-breaking: if Krumhansl-Schmuckler returned the relative minor
    # (e.g. A Minor) but the song opens on the relative major tonic chord (e.g. C),
    # align the scale label with the tonic chord.
    if raw_segments:
        first_chord = raw_segments[0]["chord"]
        if detected_scale.endswith(" Minor") and not first_chord.endswith("m"):
            min_root = detected_scale.split(" ")[0]
            if min_root in NOTE_NAMES:
                rel_major_root = NOTE_NAMES[(NOTE_NAMES.index(min_root) + 3) % 12]
                if first_chord == rel_major_root:
                    detected_scale = f"{rel_major_root} Major"

    return {
        "detected_scale": detected_scale,
        "duration": round(duration, 3),
        "chords": raw_segments,
    }


def load_stereo_audio(audio_path: Path, target_sr: int = 44100) -> np.ndarray:
    """
    Load any audio file into a stereo float32 array of shape (2, N) at `target_sr`.
    Uses soundfile first, then librosa fallback.
    """
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        # data has shape (N, C)
        y = data.T  # shape (C, N)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    except Exception:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)

    if y.shape[0] == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        y = y[:2, :]
    return y.astype(np.float32)


def separate_6_stems(audio_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Run Meta's Hybrid Transformer Demucs 6-stem model (`htdemucs_6s`) to extract:
    vocals, guitar, drums, bass, piano, other.
    Saves each stem as a 16-bit PCM WAV file using soundfile.
    Includes a DSP harmonic/percussive/bandpass fallback if Demucs encounters any runtime issue.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_files: Dict[str, Path] = {}

    try:
        from demucs.apply import apply_model

        model = get_demucs_model()
        target_sr = int(getattr(model, "samplerate", 44100))
        stereo_np = load_stereo_audio(audio_path, target_sr=target_sr)

        wav_tensor = torch.from_numpy(stereo_np)  # (2, N)
        ref = wav_tensor.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std() + 1e-8
        wav_norm = (wav_tensor - ref_mean) / ref_std

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            sources = apply_model(
                model,
                wav_norm[None],
                device=device,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]

        sources = sources * ref_std + ref_mean
        sources_np = sources.cpu().numpy()  # (num_sources, 2, N)

        for idx, source_name in enumerate(model.sources):
            stem_audio = sources_np[idx].T  # (N, 2)
            stem_audio = np.clip(stem_audio, -1.0, 1.0)
            out_wav = output_dir / f"{source_name}.wav"
            sf.write(str(out_wav), stem_audio, target_sr, subtype="PCM_16")
            stem_files[source_name] = out_wav

        # Ensure all 6 canonical stem names exist
        for name in STEM_NAMES:
            if name not in stem_files:
                out_wav = output_dir / f"{name}.wav"
                sf.write(str(out_wav), np.zeros((stereo_np.shape[1], 2), dtype=np.float32), target_sr, subtype="PCM_16")
                stem_files[name] = out_wav

        return stem_files

    except Exception as exc:
        logger.warning("Demucs htdemucs_6s fallback triggered due to: %s", exc)
        target_sr = 44100
        stereo_np = load_stereo_audio(audio_path, target_sr=target_sr)
        mono = np.mean(stereo_np, axis=0)
        y_harm, y_perc = librosa.effects.hpss(mono, margin=3.0)

        from scipy.signal import butter, sosfilt

        def bandpass(sig, low, high):
            sos = butter(4, [low / (target_sr / 2), high / (target_sr / 2)], btype="band", output="sos")
            return sosfilt(sos, sig).astype(np.float32)

        def lowpass(sig, cutoff):
            sos = butter(4, cutoff / (target_sr / 2), btype="low", output="sos")
            return sosfilt(sos, sig).astype(np.float32)

        bass_sig = lowpass(y_harm, 180.0)
        piano_sig = bandpass(y_harm, 180.0, 900.0)
        guitar_sig = bandpass(y_harm, 900.0, 2800.0)
        vocals_sig = bandpass(y_harm, 300.0, 3400.0) * 0.5
        other_sig = y_harm - (bass_sig + piano_sig + guitar_sig) * 0.5

        fallback_map = {
            "drums": np.stack([y_perc, y_perc], axis=1),
            "bass": np.stack([bass_sig, bass_sig], axis=1),
            "piano": np.stack([piano_sig, piano_sig], axis=1),
            "guitar": np.stack([guitar_sig, guitar_sig], axis=1),
            "vocals": np.stack([vocals_sig, vocals_sig], axis=1),
            "other": np.stack([other_sig, other_sig], axis=1),
        }

        for name, arr in fallback_map.items():
            out_wav = output_dir / f"{name}.wav"
            sf.write(str(out_wav), np.clip(arr, -1.0, 1.0), target_sr, subtype="PCM_16")
            stem_files[name] = out_wav

        return stem_files


def make_stem_urls(request: Request, job_id: str, subdir: str = "original") -> Dict[str, str]:
    base_url = str(request.base_url).rstrip("/")
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if forwarded_proto.lower() == "https" and base_url.startswith("http://"):
        base_url = "https://" + base_url[len("http://") :]
    return {
        stem: f"{base_url}/stems/{job_id}/{subdir}/{stem}.wav"
        for stem in STEM_NAMES
    }


def process_audio_file_internal(audio_path: Path, job_id: str) -> Dict[str, Any]:
    """Run chord/scale recognition and 6-stem separation for `audio_path`."""
    job_dir = JOBS_DIR / job_id
    orig_stems_dir = job_dir / "original"
    orig_stems_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("[%s] Running harmonic CQT chord & scale detection...", job_id)
    analysis = detect_chords_and_scale(audio_path)

    logger.info("[%s] Running Demucs htdemucs_6s 6-stem separation...", job_id)
    separate_6_stems(audio_path, orig_stems_dir)
    elapsed = round(time.time() - t0, 2)
    logger.info("[%s] Completed processing in %ss (Scale: %s, Chords: %d)", job_id, elapsed, analysis["detected_scale"], len(analysis["chords"]))

    metadata = {
        "job_id": job_id,
        "detected_scale": analysis["detected_scale"],
        "duration": analysis["duration"],
        "chords": analysis["chords"],
        "processing_time_sec": elapsed,
    }
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class TransposeRequest(BaseModel):
    job_id: str
    semitones: float


@app.get("/")
async def serve_frontend():
    index_path = BASE_DIR / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "Studio API running. Open /health for status."}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "ffmpeg": FFMPEG_PATH is not None,
        "model": "htdemucs_6s",
    }


@app.post("/api/process")
async def process_audio(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".wav"
    input_path = job_dir / f"input{ext}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    with open(input_path, "wb") as f:
        f.write(content)

    try:
        metadata = process_audio_file_internal(input_path, job_id)
    except Exception as exc:
        logger.exception("Error processing uploaded audio:")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}") from exc

    stem_urls = make_stem_urls(request, job_id, subdir="original")
    return {
        "job_id": job_id,
        "detected_scale": metadata["detected_scale"],
        "duration": metadata["duration"],
        "chords": metadata["chords"],
        "stems": stem_urls,
    }


@app.post("/api/demo")
@app.get("/api/demo")
async def load_or_process_demo(request: Request):
    """
    Returns the processed demo track (`test_song.wav`).
    Reuses the cached `demo_track` job if already separated, or runs the full pipeline on `test_song.wav`.
    """
    demo_job_id = "demo_track"
    demo_job_dir = JOBS_DIR / demo_job_id
    meta_file = demo_job_dir / "metadata.json"
    orig_dir = demo_job_dir / "original"

    if meta_file.exists() and all((orig_dir / f"{s}.wav").exists() for s in STEM_NAMES):
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        test_wav = BASE_DIR / "test_song.wav"
        if not test_wav.exists():
            # Generate test_song.wav on the fly if missing
            from scripts.generate_test_audio import main as generate_main

            generate_main()
        demo_job_dir.mkdir(parents=True, exist_ok=True)
        input_copy = demo_job_dir / "input.wav"
        shutil.copy2(test_wav, input_copy)
        metadata = process_audio_file_internal(input_copy, demo_job_id)

    return {
        "job_id": demo_job_id,
        "detected_scale": metadata["detected_scale"],
        "duration": metadata["duration"],
        "chords": metadata["chords"],
        "stems": make_stem_urls(request, demo_job_id, subdir="original"),
    }


def _pitch_shift_single_stem(src_wav: Path, dst_wav: Path, semitones: float) -> None:
    """Pitch-shift a single stereo WAV file using librosa phase vocoder while preserving duration and tempo."""
    data, sr = sf.read(str(src_wav), dtype="float32", always_2d=True)
    n_samples = data.shape[0]
    y = data.T  # (C, N)
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones))
    shifted_T = shifted.T  # (N, C)
    if shifted_T.shape[0] > n_samples:
        shifted_T = shifted_T[:n_samples, :]
    elif shifted_T.shape[0] < n_samples:
        pad = np.zeros((n_samples - shifted_T.shape[0], shifted_T.shape[1]), dtype=np.float32)
        shifted_T = np.vstack([shifted_T, pad])
    sf.write(str(dst_wav), np.clip(shifted_T, -1.0, 1.0), sr, subtype="PCM_16")


@app.post("/api/transpose-audio")
async def transpose_audio(request: Request, payload: Dict[str, Any] = Body(...)):
    """
    High-Quality Phase-Vocoder Transposition endpoint.
    Accepts `job_id` and `semitones` (-12 to +12).
    Pitch-shifts all non-drum stems (`vocals`, `guitar`, `bass`, `piano`, `other`) using
    `librosa.effects.pitch_shift` while keeping `drums` unshifted to preserve crisp transients.
    """
    job_id = str(payload.get("job_id", "")).strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id")

    try:
        semitones = int(round(float(payload.get("semitones", 0))))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid semitones value") from exc

    semitones = max(-12, min(12, semitones))

    job_dir = JOBS_DIR / job_id
    orig_dir = job_dir / "original"
    if not orig_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    meta_path = job_dir / "metadata.json"
    orig_scale = "C Major"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            orig_scale = json.load(f).get("detected_scale", "C Major")

    transposed_scale = transpose_scale_name(orig_scale, semitones)

    if semitones == 0:
        return {
            "job_id": job_id,
            "semitones": 0,
            "detected_scale": transposed_scale,
            "stems": make_stem_urls(request, job_id, subdir="original"),
        }

    subdir_name = f"shifted_{' plus_' if semitones > 0 else 'minus_'}{abs(semitones)}".replace(" ", "")
    shifted_dir = job_dir / subdir_name
    shifted_dir.mkdir(parents=True, exist_ok=True)

    # Copy drums unshifted so percussion transients stay 100% intact
    drums_src = orig_dir / "drums.wav"
    drums_dst = shifted_dir / "drums.wav"
    if drums_src.exists() and not drums_dst.exists():
        shutil.copy2(drums_src, drums_dst)

    # Pitch-shift the 5 melodic stems in parallel if not already cached
    stems_to_shift = [
        s for s in MELODIC_STEMS if (orig_dir / f"{s}.wav").exists() and not (shifted_dir / f"{s}.wav").exists()
    ]

    if stems_to_shift:
        logger.info("[%s] Rendering HQ Phase-Vocoder pitch shift (%+d semitones) for %s...", job_id, semitones, stems_to_shift)
        with ThreadPoolExecutor(max_workers=min(5, len(stems_to_shift))) as pool:
            futures = [
                pool.submit(
                    _pitch_shift_single_stem,
                    orig_dir / f"{stem}.wav",
                    shifted_dir / f"{stem}.wav",
                    float(semitones),
                )
                for stem in stems_to_shift
            ]
            for fut in futures:
                fut.result()

    return {
        "job_id": job_id,
        "semitones": semitones,
        "detected_scale": transposed_scale,
        "stems": make_stem_urls(request, job_id, subdir=subdir_name),
    }

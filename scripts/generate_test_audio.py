#!/usr/bin/env python3
"""
Synthesizes an 8-second stereo WAV file (test_song.wav) with a clear
C Major -> A Minor -> F Major -> G Major chord progression (2s per chord)
plus bass, warm pad/piano harmonics, and light percussion transients for
end-to-end verification of chord recognition, scale detection, and 6-stem separation.
"""

import os
import wave
import numpy as np

SAMPLE_RATE = 44100
DURATION = 8.0
OUTPUT_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_song.wav"),
]


def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def synthesize_test_song(sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    num_samples = int(sr * duration)
    t = np.linspace(0.0, duration, num_samples, endpoint=False)

    left = np.zeros(num_samples, dtype=np.float64)
    right = np.zeros(num_samples, dtype=np.float64)

    # 4 chords, 2 seconds each: C -> Am -> F -> G
    # Pitch classes strictly in C Major:
    # C Major: C(0), E(4), G(7)
    # A Minor: A(9), C(0), E(4)
    # F Major: F(5), A(9), C(0)
    # G Major: G(7), B(11), D(2)
    progression = [
        {
            "name": "C",
            "start": 0.0,
            "end": 2.0,
            "bass": 36,  # C2
            "notes": [48, 52, 55, 60, 64, 67, 72],  # C3, E3, G3, C4, E4, G4, C5
            "amp": 1.15,
        },
        {
            "name": "Am",
            "start": 2.0,
            "end": 4.0,
            "bass": 45,  # A2
            "notes": [57, 60, 64, 69, 72, 76],  # A3, C4, E4, A4, C5, E5
            "amp": 0.95,
        },
        {
            "name": "F",
            "start": 4.0,
            "end": 6.0,
            "bass": 41,  # F2
            "notes": [53, 57, 60, 65, 69, 72],  # F3, A3, C4, F4, A4, C5
            "amp": 0.95,
        },
        {
            "name": "G",
            "start": 6.0,
            "end": 8.0,
            "bass": 43,  # G2
            "notes": [55, 59, 62, 67, 71, 74],  # G3, B3, D4, G4, B4, D5
            "amp": 1.0,
        },
    ]

    for seg in progression:
        s_idx = int(seg["start"] * sr)
        e_idx = int(seg["end"] * sr)
        seg_len = e_idx - s_idx
        seg_t = np.linspace(0.0, seg["end"] - seg["start"], seg_len, endpoint=False)

        # Smooth ADSR-like envelope so chord boundaries are clean
        attack = int(0.03 * sr)
        release = int(0.05 * sr)
        env = np.ones(seg_len, dtype=np.float64)
        if attack > 0:
            env[:attack] = np.linspace(0.0, 1.0, attack)
        if release > 0:
            env[-release:] = np.linspace(1.0, 0.15, release)

        # Bass line (pure fundamental + octave harmonic)
        f_bass = midi_to_freq(seg["bass"])
        bass_wave = 0.28 * np.sin(2.0 * np.pi * f_bass * seg_t) + 0.10 * np.sin(
            2.0 * np.pi * (f_bass * 2.0) * seg_t
        )

        # Warm harmonic chord pad / keys (pure triad frequencies to keep CQT clean)
        chord_l = np.zeros(seg_len, dtype=np.float64)
        chord_r = np.zeros(seg_len, dtype=np.float64)
        for i, midi_note in enumerate(seg["notes"]):
            freq = midi_to_freq(midi_note)
            # Fundamental only (avoids 3rd-harmonic cross-talk in CQT)
            tone = np.sin(2.0 * np.pi * freq * seg_t)
            pan = 0.35 + 0.30 * (i / max(1, len(seg["notes"]) - 1))
            chord_l += tone * (1.0 - pan * 0.35)
            chord_r += tone * (0.65 + pan * 0.35)

        chord_l = chord_l / len(seg["notes"])
        chord_r = chord_r / len(seg["notes"])

        # Plucked guitar-like rhythmic arpeggio on triad notes every 0.5s
        pluck_l = np.zeros(seg_len, dtype=np.float64)
        pluck_r = np.zeros(seg_len, dtype=np.float64)
        for beat in range(4):
            b_start = int(beat * 0.5 * sr)
            b_end = min(seg_len, b_start + int(0.45 * sr))
            b_len = b_end - b_start
            bt = np.linspace(0.0, b_len / sr, b_len, endpoint=False)
            p_env = np.exp(-6.0 * bt)
            note_midi = seg["notes"][(beat % 3) + 3]
            f_note = midi_to_freq(note_midi)
            sig = 0.18 * np.sin(2.0 * np.pi * f_note * bt) * p_env
            pluck_l[b_start:b_end] += sig * (0.8 if beat % 2 == 0 else 0.5)
            pluck_r[b_start:b_end] += sig * (0.5 if beat % 2 == 0 else 0.8)

        left[s_idx:e_idx] += seg["amp"] * env * (0.65 * chord_l + bass_wave + pluck_l)
        right[s_idx:e_idx] += seg["amp"] * env * (0.65 * chord_r + bass_wave + pluck_r)

    # Add crisp percussion transients (kick on 0.5s grid, hi-hat on 0.25s grid)
    rng = np.random.default_rng(42)
    for step in range(int(duration / 0.25)):
        p_start = int(step * 0.25 * sr)
        if step % 2 == 0:
            # Kick transient (short pitch drop 110Hz -> 45Hz over 60ms)
            k_len = min(int(0.06 * sr), num_samples - p_start)
            kt = np.linspace(0.0, k_len / sr, k_len, endpoint=False)
            k_freq = 45.0 + 65.0 * np.exp(-45.0 * kt)
            k_phase = 2.0 * np.pi * np.cumsum(k_freq) / sr
            kick = 0.16 * np.sin(k_phase) * np.exp(-35.0 * kt)
            left[p_start : p_start + k_len] += kick
            right[p_start : p_start + k_len] += kick
        else:
            # Hi-hat / snare brush transient (25ms filtered noise burst)
            h_len = min(int(0.025 * sr), num_samples - p_start)
            ht = np.linspace(0.0, h_len / sr, h_len, endpoint=False)
            noise = rng.standard_normal(h_len)
            # High-pass via simple first-difference
            hp_noise = np.diff(noise, prepend=0.0)
            hat = 0.06 * hp_noise * np.exp(-90.0 * ht)
            left[p_start : p_start + h_len] += hat * 0.9
            right[p_start : p_start + h_len] += hat * 1.1

    # Normalize safely to -1.0 dBFS
    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = (stereo / peak) * 0.88

    return stereo


def save_wav(filepath: str, stereo_audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    pcm16 = np.clip(stereo_audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def main() -> None:
    stereo = synthesize_test_song()
    for out_path in OUTPUT_PATHS:
        save_wav(out_path, stereo)
        print(f"Generated stereo test track: {out_path} ({DURATION}s, {SAMPLE_RATE}Hz)")


if __name__ == "__main__":
    main()

"""
phase2_preprocess.py
====================
PHASE 2 — Audio Standardisation + Noise Addition
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(stage, audio):
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    print(f"    [{stage}] Peak={peak:.4f} | RMS={rms:.4f} | Len={len(audio)}")


def load_and_standardise(wav_path: Path) -> np.ndarray:
    """Load a WAV, resample to 16 kHz, force mono, peak-normalise to –1 dBFS."""

    print("  Step 1: Loading audio...")

    audio, sr = librosa.load(
        str(wav_path),
        sr=config.SAMPLE_RATE,
        mono=config.AUDIO_MONO
    )

    print(f"    Sample rate: {sr}")
    print_stats("Loaded", audio)

    print("  Step 2–4: Normalising...")

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.99

    print_stats("Normalised", audio)

    return audio.astype(np.float32)


def _synthesise_noise(
    noise_type: str,
    n_samples: int,
    rng: np.random.Generator
) -> np.ndarray:

    if noise_type == "traffic":

        white = rng.standard_normal(n_samples)

        pink = np.cumsum(white)
        pink -= pink.mean()
        pink /= (np.std(pink) + 1e-8)

        t = np.linspace(0, n_samples / config.SAMPLE_RATE, n_samples)
        rumble = 0.3 * np.sin(2 * np.pi * 80 * t)

        noise = pink + rumble

    elif noise_type == "cafe":

        noise = np.zeros(n_samples)

        for _ in range(20):

            start = rng.integers(0, max(n_samples - 8000, 1))
            length = rng.integers(2000, 8000)

            burst = rng.standard_normal(length)

            fft = np.fft.rfft(burst, n=length)
            freqs = np.fft.rfftfreq(length, d=1.0 / config.SAMPLE_RATE)

            fft[(freqs < 200) | (freqs > 4000)] = 0

            burst = np.fft.irfft(fft, n=length).real

            end = min(start + length, n_samples)

            noise[start:end] += burst[:end - start]

    else:  # room

        white = rng.standard_normal(n_samples)

        noise = np.zeros(n_samples)

        alpha = 0.95

        for i in range(1, n_samples):
            noise[i] = alpha * noise[i - 1] + (1 - alpha) * white[i]

    rms = np.sqrt(np.mean(noise ** 2)) + 1e-8

    noise = noise / rms

    return noise.astype(np.float32)


def add_noise_at_snr(
    clean: np.ndarray,
    noise_type: str,
    snr_db: float,
    rng: np.random.Generator
) -> np.ndarray:

    print(f"    Step 6: Adding {noise_type} noise @ {snr_db} dB")

    noise = _synthesise_noise(noise_type, len(clean), rng)

    print_stats("Noise", noise)

    clean_rms = np.sqrt(np.mean(clean ** 2)) + 1e-8

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))

    noisy = clean + noise * desired_noise_rms

    noisy = np.clip(noisy, -1.0, 1.0)

    print_stats("Noisy", noisy)

    return noisy.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all():

    rng = np.random.default_rng(config.SEED)

    raw_files = sorted(config.AUDIO_RAW_DIR.glob("*_AUDIO.wav"))

    if not raw_files:

        print(f"[WARNING] No audio files found in {config.AUDIO_RAW_DIR}")
        print("Expected: <ID>_AUDIO.wav")

        return

    print("\n" + "=" * 60)
    print(f"PHASE 2 — Preprocessing {len(raw_files)} audio files")
    print("=" * 60 + "\n")


    for i, wav_path in enumerate(
        tqdm(raw_files, desc="Processing Files"),
        start=1
    ):

        pid = wav_path.stem.replace("_AUDIO", "")

        print(f"\n▶ File {i}/{len(raw_files)} : {wav_path.name}")

        # Step 1–4
        audio = load_and_standardise(wav_path)

        # Step 5
        print("  Step 5: Saving clean audio...")

        clean_path = config.CLEAN_AUDIO_DIR / f"{pid}_clean.wav"

        sf.write(
            str(clean_path),
            audio,
            config.SAMPLE_RATE,
            subtype="PCM_16"
        )

        print(f"    Saved: {clean_path}")

        # Step 6–7
        for noise_type in config.NOISE_TYPES:

            for snr in config.SNR_LEVELS:

                noisy = add_noise_at_snr(
                    audio,
                    noise_type,
                    snr,
                    rng
                )

                noise_dir = (
                    config.NOISY_AUDIO_DIR
                    / noise_type
                    / f"{snr}dB"
                )

                noise_dir.mkdir(parents=True, exist_ok=True)

                noisy_path = (
                    noise_dir
                    / f"{pid}_{noise_type}_{snr}dB.wav"
                )

                sf.write(
                    str(noisy_path),
                    noisy,
                    config.SAMPLE_RATE,
                    subtype="PCM_16"
                )

                print(f"      Saved: {noisy_path}")


    print("\n" + "=" * 60)

    print(f"✔ Clean audio  : {config.CLEAN_AUDIO_DIR}")
    print(f"✔ Noisy audio  : {config.NOISY_AUDIO_DIR}")
    print(f"✔ Noise types : {config.NOISE_TYPES}")
    print(f"✔ SNR levels  : {config.SNR_LEVELS} dB")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    preprocess_all()

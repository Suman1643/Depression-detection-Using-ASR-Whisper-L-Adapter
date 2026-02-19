"""
debug_noise_generator.py
========================

This script generates ONLY the synthetic noises used in phase2_preprocess.py
and saves them as WAV files so you can listen to them.

Run:
    python debug_noise_generator.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path

import config


# --------------------------------------------------
# Noise generator (copied from main code)
# --------------------------------------------------

def synthesise_noise(noise_type, n_samples, rng):

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


    # Normalise RMS
    rms = np.sqrt(np.mean(noise ** 2)) + 1e-8

    noise = noise / rms

    return noise.astype(np.float32)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    rng = np.random.default_rng(config.SEED)

    duration_sec = 10   # 10 seconds of noise
    n_samples = int(duration_sec * config.SAMPLE_RATE)

    out_dir = Path("debug_noises")
    out_dir.mkdir(exist_ok=True)

    print("\nGenerating noise samples...\n")

    for noise_type in config.NOISE_TYPES:

        print(f"Generating: {noise_type}")

        noise = synthesise_noise(noise_type, n_samples, rng)

        out_path = out_dir / f"{noise_type}_noise.wav"

        sf.write(out_path, noise, config.SAMPLE_RATE, subtype="PCM_16")

        print(f"Saved: {out_path}")

    print("\nDone. Check the 'debug_noises' folder.")


if __name__ == "__main__":
    main()

"""
config.py
=========
Central configuration for all phases of the depression detection pipeline.
Edit the paths below to match your local setup.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROOT PATHS  (edit these to match your folder layout)
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path("E-label")                        # top-level data folder
AUDIO_RAW_DIR   = ROOT_DIR / "full-extended-audio"       # raw .wav files
TRANSCRIPT_DIR  = ROOT_DIR / "full-extended-transcript"  # transcript csvs

# CSV splits - these are INSIDE the E-label folder based on your error output
TRAIN_CSV   = ROOT_DIR / "train_split.csv"
DEV_CSV     = ROOT_DIR / "dev_split.csv"
TEST_CSV    = ROOT_DIR / "test_split.csv"
LABELS_CSV  = ROOT_DIR / "Detailed_PHQ8_Labels.csv"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES  (created automatically)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR              = Path("outputs")
CLEAN_AUDIO_DIR      = OUT_DIR / "clean_audio"        # after resampling / norm
NOISY_AUDIO_DIR      = OUT_DIR / "noisy_audio"        # noise versions
FEATURE_DIR          = OUT_DIR / "features"           # traditional + whisper
CHECKPOINT_DIR       = OUT_DIR / "checkpoints"        # model weights
RESULTS_DIR          = OUT_DIR / "results"            # tables, plots

for d in [CLEAN_AUDIO_DIR, NOISY_AUDIO_DIR, FEATURE_DIR,
          CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000          # target sample rate (Hz)
AUDIO_MONO    = True            # force mono
TARGET_LUFS   = -23.0           # loudness normalisation target (LUFS)

# Noise conditions
SNR_LEVELS    = [20, 10, 0]     # dB
NOISE_TYPES   = ["traffic", "cafe", "room"]

# ─────────────────────────────────────────────────────────────────────────────
# TRADITIONAL FEATURES  (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
N_MFCC        = 40
HOP_LENGTH    = 512
N_FFT         = 2048

# ─────────────────────────────────────────────────────────────────────────────
# WHISPER  (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base"          # "tiny" | "base" | "small" | "medium" | "large"
WHISPER_DEVICE = "cpu"          # Force CPU for Windows compatibility

# ─────────────────────────────────────────────────────────────────────────────
# L-ADAPTER  (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
ADAPTER_HIDDEN  = 128           # BiGRU hidden size
ADAPTER_OUT_DIM = 64            # output embedding per layer-adapter
BIGRU_LAYERS    = 1

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 2               # depressed / not-depressed
BATCH_SIZE    = 8
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
SEED          = 42

# PHQ-8 binary threshold (scores ≥ 10 → depressed)
PHQ_THRESHOLD = 10

# ─────────────────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────────────────
NUM_WORKERS   = 0               # dataloader workers (must be 0 for Windows)
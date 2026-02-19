"""
config.py — WINDOWS VERSION
===========================
Based on your directory structure:
C:\Users\Intern\Desktop\SumanKumar\Depression detection Using ASR Whisper L-Adapter\

UPDATE THE PATHS BELOW TO MATCH YOUR ACTUAL FOLDER STRUCTURE!
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROOT PATHS — EDIT THESE TO MATCH YOUR SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# Option 1: If your CSV files are in E-label/ folder
ROOT_DIR = Path("E-label")
AUDIO_RAW_DIR = ROOT_DIR / "full-extended-audio"
TRANSCRIPT_DIR = ROOT_DIR / "full-extended-transcript"

TRAIN_CSV = ROOT_DIR / "train_split.csv"
DEV_CSV = ROOT_DIR / "dev_split.csv"
TEST_CSV = ROOT_DIR / "test_split.csv"
LABELS_CSV = ROOT_DIR / "Detailed_PHQ8_Labels.csv"

# Option 2: If CSVs are in current directory (uncomment these instead)
# ROOT_DIR = Path(".")
# AUDIO_RAW_DIR = Path("E-label") / "full-extended-audio"
# TRANSCRIPT_DIR = Path("E-label") / "full-extended-transcript"
# 
# TRAIN_CSV = Path("train_split.csv")
# DEV_CSV = Path("dev_split.csv")
# TEST_CSV = Path("test_split.csv")
# LABELS_CSV = Path("Detailed_PHQ8_Labels.csv")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES (created automatically)
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path("outputs")
CLEAN_AUDIO_DIR = OUT_DIR / "clean_audio"
NOISY_AUDIO_DIR = OUT_DIR / "noisy_audio"
FEATURE_DIR = OUT_DIR / "features"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
RESULTS_DIR = OUT_DIR / "results"

for d in [CLEAN_AUDIO_DIR, NOISY_AUDIO_DIR, FEATURE_DIR,
          CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
AUDIO_MONO = True
TARGET_LUFS = -23.0

SNR_LEVELS = [20, 10, 0]
NOISE_TYPES = ["traffic", "cafe", "room"]

# ─────────────────────────────────────────────────────────────────────────────
# TRADITIONAL FEATURES (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# ─────────────────────────────────────────────────────────────────────────────
# WHISPER (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"  # Force CPU for Windows compatibility

# ─────────────────────────────────────────────────────────────────────────────
# L-ADAPTER (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
ADAPTER_HIDDEN = 128
ADAPTER_OUT_DIM = 64
BIGRU_LAYERS = 1

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
SEED = 42

PHQ_THRESHOLD = 10

# ─────────────────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────────────────
NUM_WORKERS = 0  # Required for Windows

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS — Print paths on import to verify
# ─────────────────────────────────────────────────────────────────────────────
if __name__ != "config":  # Only print when importing
    pass
else:
    print(f"TRAIN: {TRAIN_CSV.absolute()} {TRAIN_CSV.exists()}")
    print(f"DEV: {DEV_CSV.absolute()} {DEV_CSV.exists()}")
    print(f"TEST: {TEST_CSV.absolute()} {TEST_CSV.exists()}")
"""
data_utils.py
=============
Shared data loading and label utilities used across all phases.

Key functions:
  load_all_labels()         → DataFrame with all participants + PHQ labels
  get_split(split_name)     → DataFrame for "train" / "dev" / "test"
  audio_path(pid)           → Path to best available audio for a participant
  phq8_to_binary(score)     → 0 or 1 (threshold = config.PHQ_THRESHOLD)
  participant_summary()     → prints label distribution info
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config


# ─────────────────────────────────────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_labels() -> pd.DataFrame:
    """
    Merge split CSVs with the detailed PHQ-8 item labels.
    Returns one row per participant with columns:
      Participant_ID, Gender, PHQ_Binary, PHQ_Score, Split,
      PHQ_8NoInterest, PHQ_8Depressed, …, PHQ_8Total
    """
    dfs = []
    for split_name, csv_path in [
        ("train", config.TRAIN_CSV),
        ("dev",   config.DEV_CSV),
        ("test",  config.TEST_CSV),
    ]:
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found — skipping {split_name} split")
            continue
        df = pd.read_csv(csv_path)
        df["Split"] = split_name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No split CSV files found. Check config.TRAIN_CSV / DEV_CSV / TEST_CSV."
        )

    combined = pd.concat(dfs, ignore_index=True)
    combined["Participant_ID"] = combined["Participant_ID"].astype(str).str.strip()

    # Merge with detailed PHQ-8 item scores if available
    if config.LABELS_CSV.exists():
        phq = pd.read_csv(config.LABELS_CSV)
        phq["Participant_ID"] = phq["Participant_ID"].astype(str).str.strip()
        combined = combined.merge(phq, on="Participant_ID", how="left",
                                  suffixes=("", "_phq8"))

    return combined


def get_split(split_name: str) -> pd.DataFrame:
    """
    Load and return one split as a DataFrame.

    Parameters
    ----------
    split_name : "train" | "dev" | "test"
    """
    mapping = {"train": config.TRAIN_CSV, "dev": config.DEV_CSV, "test": config.TEST_CSV}
    if split_name not in mapping:
        raise ValueError(f"split_name must be one of {list(mapping)}")
    df = pd.read_csv(mapping[split_name])
    df["Participant_ID"] = df["Participant_ID"].astype(str).str.strip()
    return df


def phq8_to_binary(score: int | float) -> int:
    """Map PHQ-8 total score to binary label (0=not depressed, 1=depressed)."""
    return int(score >= config.PHQ_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
# Audio path resolution (best available source)
# ─────────────────────────────────────────────────────────────────────────────

def audio_path(pid: str, noise_type: str | None = None,
               snr: int | None = None) -> Path | None:
    """
    Resolve the best available audio path for a participant.

    Priority:
      1. Noisy version (if noise_type + snr given)
      2. Clean pre-processed audio
      3. Raw audio
    """
    if noise_type and snr is not None:
        p = config.NOISY_AUDIO_DIR / noise_type / f"{snr}dB" / f"{pid}_{noise_type}_{snr}dB.wav"
        if p.exists():
            return p

    clean = config.CLEAN_AUDIO_DIR / f"{pid}_clean.wav"
    if clean.exists():
        return clean

    raw = config.AUDIO_RAW_DIR / f"{pid}_AUDIO.wav"
    if raw.exists():
        return raw

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data statistics
# ─────────────────────────────────────────────────────────────────────────────

def participant_summary():
    """Print a summary table of label distributions across splits."""
    print("\n" + "=" * 55)
    print("  DAIC-WOZ Dataset Summary")
    print("=" * 55)
    try:
        df = load_all_labels()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return

    for split in ["train", "dev", "test"]:
        sub = df[df["Split"] == split]
        n   = len(sub)
        if n == 0:
            continue
        n_dep    = (sub["PHQ_Binary"] == 1).sum()
        n_nodep  = (sub["PHQ_Binary"] == 0).sum()
        pct_dep  = 100 * n_dep / max(n, 1)
        mean_phq = sub["PHQ_Score"].mean()
        print(f"\n  {split.capitalize()} split  ({n} participants)")
        print(f"    Depressed   (PHQ≥10): {n_dep:3d}  ({pct_dep:.1f}%)")
        print(f"    Not depress (PHQ< 10): {n_nodep:3d}  ({100-pct_dep:.1f}%)")
        print(f"    Mean PHQ-8 score    : {mean_phq:.2f}")

    # Check audio availability
    all_ids = df["Participant_ID"].unique()
    available = sum(1 for pid in all_ids if audio_path(pid) is not None)
    print(f"\n  Audio files found: {available}/{len(all_ids)} participants")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHQ-8 severity mapping
# ─────────────────────────────────────────────────────────────────────────────

PHQ8_SEVERITY = {
    (0,  4):  "Minimal",
    (5,  9):  "Mild",
    (10, 14): "Moderate",
    (15, 19): "Moderately severe",
    (20, 27): "Severe",
}

def phq_severity(score: int) -> str:
    for (lo, hi), label in PHQ8_SEVERITY.items():
        if lo <= score <= hi:
            return label
    return "Unknown"


if __name__ == "__main__":
    participant_summary()
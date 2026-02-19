"""
phase8_noise_experiment.py
==========================
PHASE 8 — Noise Robustness Experiments (Uses Pre-Extracted Adapter Embeddings)

This version uses pre-extracted L-Adapter embeddings from NPZ files for BOTH
clean and noisy audio. Much faster since adapters are already applied!

Run phase4b_whisper_noisy_with_adapters.py first to extract noisy features.

Benefits:
  - 100× faster than re-extracting Whisper + adapters each time
  - Clean audio: Uses outputs/features/<pid>_whisper.npz → applies adapters
  - Noisy audio: Uses outputs/features_noisy/<noise>/<snr>dB/<pid>_adapter.npz (adapters already applied!)

Usage:
  # Step 1: Extract noisy features with adapters (do this once, can use GPU)
  python phase4b_whisper_noisy_with_adapters.py --gpu

  # Step 2: Run noise experiments (fast!)
  python phase8_noise_experiment.py
"""

import pickle
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

import config
from phase3_baseline import extract_features
from phase5_adapters import load_whisper_npz, LAdapterBank
from phase6_fusion import DepressionDetectionModel

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

NOISY_FEATURE_DIR = config.OUT_DIR / "features_noisy"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_split_labels(split_csv: Path) -> list[tuple[str, int]]:
    df = pd.read_csv(split_csv)
    df["Participant_ID"] = df["Participant_ID"].astype(str).str.strip()
    return [(row["Participant_ID"], int(row["PHQ_Binary"]))
            for _, row in df.iterrows()]


def load_clean_features_with_adapters(pid: str, adapter_bank: LAdapterBank, device: str):
    """Load clean Whisper features and apply adapters."""
    # Load raw Whisper hidden states
    hidden = load_whisper_npz(pid, device=device, verbose=False)
    if hidden is None:
        return None
    
    # Apply adapters
    with torch.no_grad():
        embeddings = adapter_bank(hidden)  # List of 24 tensors (1, 64)
    
    return embeddings


def load_noisy_adapter_embeddings(pid: str, noise_type: str, snr: int, device: str):
    """Load pre-extracted noisy adapter embeddings from NPZ file."""
    npz_path = NOISY_FEATURE_DIR / noise_type / f"{snr}dB" / f"{pid}_adapter.npz"
    
    if not npz_path.exists():
        return None
    
    data = np.load(str(npz_path))
    
    # Convert to list of torch tensors (matching adapter_bank output format)
    embeddings = []
    num_enc = 12
    
    # Encoder embeddings
    for i in range(num_enc):
        key = f"encoder_emb_{i}"
        if key in data:
            emb = torch.tensor(data[key], dtype=torch.float32, device=device)
            embeddings.append(emb)
    
    # Decoder embeddings
    for i in range(num_enc):
        key = f"decoder_emb_{i}"
        if key in data:
            emb = torch.tensor(data[key], dtype=torch.float32, device=device)
            embeddings.append(emb)
    
    return embeddings if len(embeddings) > 0 else None


def get_noisy_audio_path(pid: str, noise_type: str, snr: int) -> Path | None:
    """Get path to noisy audio file."""
    path = config.NOISY_AUDIO_DIR / noise_type / f"{snr}dB" / f"{pid}_{noise_type}_{snr}dB.wav"
    return path if path.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Baseline evaluation (uses noisy audio directly)
# ─────────────────────────────────────────────────────────────────────────────

def eval_baseline_condition(model_name: str, noise_type: str | None, snr: int | None,
                             samples: list[tuple[str, int]]) -> dict:
    """Evaluate baseline model on one noise condition."""
    ckpt_path = config.CHECKPOINT_DIR / f"baseline_{model_name}.pkl"
    if not ckpt_path.exists():
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}

    with open(ckpt_path, "rb") as f:
        pipe = pickle.load(f)

    X_list, y_list = [], []
    
    for pid, label in samples:
        # Get audio path
        if noise_type is None:
            # Clean audio
            wav_path = config.CLEAN_AUDIO_DIR / f"{pid}_clean.wav"
            if not wav_path.exists():
                wav_path = config.AUDIO_RAW_DIR / f"{pid}_AUDIO.wav"
        else:
            # Noisy audio
            wav_path = get_noisy_audio_path(pid, noise_type, snr)
        
        if wav_path is None or not wav_path.exists():
            continue
        
        # Extract features
        feats = extract_features(wav_path, pid)
        X_list.append(feats)
        y_list.append(label)

    if len(X_list) < 2:
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}

    X = np.array(X_list)
    y = np.array(y_list)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = np.nan
    
    return {"acc": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# Deep model evaluation (uses pre-extracted adapter embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def eval_deep_condition(noise_type: str | None, snr: int | None,
                        samples: list[tuple[str, int]],
                        model: DepressionDetectionModel,
                        adapter_bank: LAdapterBank,
                        device: str) -> dict:
    """Evaluate deep model on one noise condition using adapter embeddings."""
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for pid, label in samples:
            # Load adapter embeddings
            if noise_type is None:
                # Clean: load raw Whisper + apply adapters
                embeddings = load_clean_features_with_adapters(pid, adapter_bank, device)
            else:
                # Noisy: load pre-computed adapter embeddings
                embeddings = load_noisy_adapter_embeddings(pid, noise_type, snr, device)
            
            if embeddings is None:
                continue

            # embeddings is a list of 24 tensors (1, 64)
            # Stack them for fusion layer
            embeddings_stacked = torch.stack(embeddings, dim=1)  # (1, 24, 64)
            
            # Apply fusion + classifier (skip adapter bank since we already have embeddings)
            fused = model.fusion(embeddings_stacked)  # (1, 64)
            logits = model.classifier(fused)          # (1, 2)
            
            prob   = torch.softmax(logits, dim=-1)[0, 1].item()
            pred   = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(prob)

    if len(all_preds) < 2:
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = np.nan
    
    return {"acc": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_noise_experiments():
    print(f"\n{'='*70}")
    print("PHASE 8 — Noise Robustness (Using Pre-Extracted Adapter Embeddings)")
    print(f"{'='*70}\n")

    device  = "cpu"
    samples = load_split_labels(config.TEST_CSV)
    
    print(f"  Test samples: {len(samples)}")

    # Check if deep model exists
    deep_ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    has_deep  = deep_ckpt.exists()

    deep_model = None
    adapter_bank = None
    
    if has_deep:
        print(f"  Loading deep model...", end=" ", flush=True)
        deep_model = DepressionDetectionModel(whisper_model=config.WHISPER_MODEL)
        deep_model.load_state_dict(torch.load(deep_ckpt, map_location=device))
        deep_model.to(device)
        deep_model.eval()
        print("✓")
        
        # Extract adapter bank for clean audio processing
        adapter_bank = deep_model.adapter_bank

    # Conditions
    conditions = [(None, None)] + [
        (nt, snr) for nt in config.NOISE_TYPES for snr in config.SNR_LEVELS
    ]

    # Check which noisy features are available
    missing_noisy = []
    if has_deep:
        for noise_type, snr in conditions[1:]:  # Skip clean
            test_pid = samples[0][0]
            npz_path = NOISY_FEATURE_DIR / noise_type / f"{snr}dB" / f"{test_pid}_adapter.npz"
            if not npz_path.exists():
                missing_noisy.append(f"{noise_type}_{snr}dB")
    
    if missing_noisy:
        print(f"\n  ⚠️  WARNING: Missing noisy adapter features for {len(missing_noisy)} conditions:")
        for cond in missing_noisy[:3]:
            print(f"      - {cond}")
        if len(missing_noisy) > 3:
            print(f"      ... and {len(missing_noisy) - 3} more")
        print(f"\n  Run this first: python phase4b_whisper_noisy_with_adapters.py --gpu")
        print(f"  (Will evaluate only available conditions)")

    rows = []
    baseline_models = ["SVM", "RandomForest", "LogisticRegression"]
    
    total_evals = len(conditions) * (len(baseline_models) + (1 if has_deep else 0))
    print(f"\n  Total evaluations: {total_evals}")
    print(f"  {'─'*70}\n")

    pbar = tqdm(total=total_evals, desc="  Evaluating", unit="model", ncols=80)

    for noise_type, snr in conditions:
        cond_name = "Clean" if noise_type is None else f"{noise_type}_{snr}dB"

        # Baseline systems
        for bm in baseline_models:
            ckpt = config.CHECKPOINT_DIR / f"baseline_{bm}.pkl"
            if not ckpt.exists():
                pbar.update(1)
                continue
            
            m = eval_baseline_condition(bm, noise_type, snr, samples)
            
            rows.append({
                "System": f"Baseline_{bm}",
                "Condition": cond_name,
                "NoiseType": noise_type or "clean",
                "SNR_dB": snr if snr is not None else "—",
                "Accuracy": round(m["acc"], 3),
                "F1": round(m["f1"], 3),
                "AUC": round(m["auc"], 3),
            })
            pbar.update(1)

        # Deep system
        if has_deep and deep_model is not None and adapter_bank is not None:
            m = eval_deep_condition(noise_type, snr, samples, deep_model, adapter_bank, device)
            
            rows.append({
                "System": "Whisper+LAdapter",
                "Condition": cond_name,
                "NoiseType": noise_type or "clean",
                "SNR_dB": snr if snr is not None else "—",
                "Accuracy": round(m["acc"], 3),
                "F1": round(m["f1"], 3),
                "AUC": round(m["auc"], 3),
            })
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n  [NOTE] No results — train models first (phase3 & phase7).\n")
        return

    # Save results
    out_path = config.RESULTS_DIR / "noise_robustness_table.csv"
    df.to_csv(out_path, index=False)

    # Pretty-print pivot
    print(f"\n{'='*70}")
    print("  Noise Robustness Results (F1 Score)")
    print(f"{'='*70}\n")
    
    pivot = df.pivot_table(index="System", columns="Condition",
                           values="F1", aggfunc="first")
    col_order = ["Clean"] + [
        f"{nt}_{snr}dB"
        for nt in config.NOISE_TYPES
        for snr in sorted(config.SNR_LEVELS, reverse=True)
    ]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    print(pivot.to_string())

    print(f"\n{'='*70}")
    print(f"  ✓ Full table saved to: {out_path}")
    print(f"{'='*70}\n")

    return df


if __name__ == "__main__":
    run_noise_experiments()
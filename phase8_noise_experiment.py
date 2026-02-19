"""
phase8_noise_experiment.py
==========================
PHASE 8 — Noise Robustness Experiments (CPU-Optimized)

Evaluates all systems under clean + 3 noise types x 3 SNR levels.

Clean audio:
  - Baseline : re-extracts MFCC/spectral features from saved clean wav
  - Deep model: loads pre-extracted .npz files (no Whisper re-run needed)

Noisy audio:
  - Both systems must process audio with added noise.
  - Baseline  : adds noise -> extract_features()
  - Deep model: adds noise -> Whisper re-extraction -> adapter forward pass

Output: outputs/results/noise_robustness_table.csv + printed F1 pivot table.

Usage:
  python phase8_noise_experiment.py
"""

import os
import pickle
import json
import tempfile
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config
from phase2_preprocess       import load_and_standardise, add_noise_at_snr
from phase3_baseline         import extract_features
from phase4_whisper_extract  import WhisperHiddenExtractor, load_whisper as _load_whisper
from phase5_adapters         import load_whisper_npz, LAdapterBank
from phase6_fusion           import DepressionDetectionModel

# Use all CPU cores
torch.set_num_threads(os.cpu_count())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_split_labels(split_csv):
    df = pd.read_csv(split_csv)
    df["Participant_ID"] = df["Participant_ID"].astype(str).str.strip()
    return [(row["Participant_ID"], int(row["PHQ_Binary"]))
            for _, row in df.iterrows()]


def get_clean_audio(pid):
    """Return clean audio array for a participant, or None if not found."""
    for path in [
        config.CLEAN_AUDIO_DIR / f"{pid}_clean.wav",
        config.AUDIO_RAW_DIR   / f"{pid}_AUDIO.wav",
    ]:
        if path.exists():
            return load_and_standardise(path)
    return None


def get_noisy_audio(pid, noise_type, snr, rng):
    """
    Return noisy audio for a participant.
    Tries pre-saved noisy file first; falls back to adding noise on-the-fly
    to the clean audio if the pre-saved version doesn't exist.
    """
    # 1. Try pre-saved noisy file (fastest)
    saved = (config.NOISY_AUDIO_DIR / noise_type
             / f"{snr}dB" / f"{pid}_{noise_type}_{snr}dB.wav")
    if saved.exists():
        import librosa
        audio, _ = librosa.load(str(saved), sr=config.SAMPLE_RATE, mono=True)
        return audio

    # 2. Generate on-the-fly from clean audio
    audio = get_clean_audio(pid)
    if audio is None:
        return None
    return add_noise_at_snr(audio, noise_type, snr, rng)


def write_temp_wav(audio):
    """Write audio array to a temp WAV file. Returns Path. Caller must delete."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, config.SAMPLE_RATE, subtype="PCM_16")
    return Path(tmp.name)


def safe_metrics(labels, preds, probs):
    """Compute acc / f1 / auc safely."""
    if len(preds) < 2:
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan
    return {"acc": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_baseline_condition(model_name, noise_type, snr, samples, rng):
    """
    Evaluate a baseline sklearn model on clean or noisy audio.
    Clean  : loads clean wav -> extract_features()
    Noisy  : adds noise to clean wav -> extract_features()
    """
    ckpt_path = config.CHECKPOINT_DIR / f"baseline_{model_name}.pkl"
    if not ckpt_path.exists():
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}

    with open(ckpt_path, "rb") as f:
        pipe = pickle.load(f)

    X_list, y_list = [], []

    for pid, label in samples:
        # Get audio (clean or noisy)
        if noise_type is None:
            audio = get_clean_audio(pid)
        else:
            audio = get_noisy_audio(pid, noise_type, snr, rng)

        if audio is None:
            continue

        tmp_path = write_temp_wav(audio)
        try:
            feats = extract_features(tmp_path, pid)
        except Exception:
            feats = None
        finally:
            tmp_path.unlink(missing_ok=True)

        if feats is not None:
            X_list.append(feats)
            y_list.append(label)

    if len(X_list) < 2:
        return {"acc": np.nan, "f1": np.nan, "auc": np.nan}

    X     = np.array(X_list)
    y     = np.array(y_list)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)[:, 1]
    return safe_metrics(y, preds, probs)


# ─────────────────────────────────────────────────────────────────────────────
# Deep model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_deep_condition(noise_type, snr, samples, model,
                        whisper_extractor, rng, device):
    """
    Evaluate the Whisper + L-Adapter model.

    Clean  : loads pre-extracted .npz files — NO Whisper re-run needed.
             This is fast because all clean features are already on disk.

    Noisy  : adds noise -> runs Whisper encoder -> adapter forward pass.
             Whisper extractor is loaded once and reused across all conditions.
    """
    num_enc = LAdapterBank.WHISPER_ENC_LAYERS.get(config.WHISPER_MODEL, 6)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for pid, label in samples:

            # ── CLEAN: use pre-extracted NPZ (fastest path, no Whisper needed) ──
            if noise_type is None:
                hidden = load_whisper_npz(pid, device=device,
                                          max_enc_layers=num_enc)
                if hidden is None:
                    continue

            # ── NOISY: must re-run Whisper on noisy audio ──
            else:
                if whisper_extractor is None:
                    continue   # can't evaluate noisy without extractor

                audio = get_noisy_audio(pid, noise_type, snr, rng)
                if audio is None:
                    continue

                tmp_path = write_temp_wav(audio)
                try:
                    raw = whisper_extractor.extract(tmp_path)
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    continue
                finally:
                    tmp_path.unlink(missing_ok=True)

                # Convert to tensors, clamp to expected encoder layers
                hidden = {
                    k: torch.tensor(v, dtype=torch.float32,
                                    device=device).unsqueeze(0)
                    for k, v in raw.items()
                    if k.startswith("encoder_layer_")
                    and int(k.split("_")[-1]) < num_enc
                }

                if not hidden:
                    continue

            # Forward pass
            logits = model(hidden)
            prob   = torch.softmax(logits, dim=-1)[0, 1].item()
            pred   = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(prob)

    return safe_metrics(all_labels, all_preds, all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_noise_experiments():
    print(f"\n{'='*70}")
    print("PHASE 8 — Noise Robustness Experiments")
    print(f"{'='*70}\n")

    rng     = np.random.default_rng(config.SEED)
    device  = "cpu"
    samples = load_split_labels(config.TEST_CSV)
    print(f"  Test samples : {len(samples)}")

    # ── Load deep model once (reused across all conditions) ──────────────────
    deep_ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    has_deep  = deep_ckpt.exists()
    deep_model        = None
    whisper_extractor = None

    if has_deep:
        deep_model = DepressionDetectionModel(whisper_model=config.WHISPER_MODEL)
        deep_model.load_state_dict(torch.load(deep_ckpt, map_location=device))
        deep_model.to(device)
        deep_model.eval()
        print(f"  Deep model   : loaded ✓")

        # Whisper extractor is only needed for NOISY conditions
        # (clean conditions use pre-extracted NPZ files)
        try:
            wm = _load_whisper(config.WHISPER_MODEL, device)
            whisper_extractor = WhisperHiddenExtractor(wm, encoder_only=True)
            print(f"  Whisper      : loaded ✓  (for noisy re-extraction)")
        except Exception as e:
            print(f"  Whisper      : FAILED ({e}) — noisy deep eval skipped")
    else:
        print(f"  Deep model   : not found — run phase7 first")

    # ── Conditions ───────────────────────────────────────────────────────────
    # (None, None) = clean; everything else = noisy
    conditions = [(None, None)] + [
        (nt, snr)
        for nt  in config.NOISE_TYPES
        for snr in config.SNR_LEVELS
    ]

    baseline_models = ["SVM", "RandomForest", "LogisticRegression"]
    active_baselines = [
        bm for bm in baseline_models
        if (config.CHECKPOINT_DIR / f"baseline_{bm}.pkl").exists()
    ]

    total = len(conditions) * (len(active_baselines) + (1 if has_deep else 0))
    print(f"  Conditions   : {len(conditions)}  "
          f"(1 clean + {len(conditions)-1} noisy)")
    print(f"  Systems      : {active_baselines + (['Whisper+LAdapter'] if has_deep else [])}")
    print(f"  Total evals  : {total}\n")

    rows = []
    pbar = tqdm(total=total, desc="  Progress", unit="eval", ncols=72)

    for noise_type, snr in conditions:
        cond_name = "Clean" if noise_type is None else f"{noise_type}_{snr}dB"

        # Baseline systems
        for bm in active_baselines:
            m = eval_baseline_condition(bm, noise_type, snr, samples, rng)
            rows.append({
                "System"   : f"Baseline_{bm}",
                "Condition": cond_name,
                "NoiseType": noise_type or "clean",
                "SNR_dB"   : snr if snr is not None else "—",
                "Accuracy" : round(m["acc"], 3),
                "F1"       : round(m["f1"],  3),
                "AUC"      : round(m["auc"], 3),
            })
            pbar.update(1)

        # Deep system
        if has_deep and deep_model is not None:
            m = eval_deep_condition(
                noise_type, snr, samples,
                deep_model, whisper_extractor, rng, device
            )
            rows.append({
                "System"   : "Whisper+LAdapter",
                "Condition": cond_name,
                "NoiseType": noise_type or "clean",
                "SNR_dB"   : snr if snr is not None else "—",
                "Accuracy" : round(m["acc"], 3),
                "F1"       : round(m["f1"],  3),
                "AUC"      : round(m["auc"], 3),
            })
            pbar.update(1)

    pbar.close()

    if not rows:
        print("\n  [NOTE] No results — run phase3 and phase7 first.\n")
        return None

    df = pd.DataFrame(rows)

    # Save full results
    out_path = config.RESULTS_DIR / "noise_robustness_table.csv"
    df.to_csv(out_path, index=False)

    # Print F1 pivot table
    print(f"\n{'='*70}")
    print("  Noise Robustness Results — F1 Score")
    print(f"{'='*70}\n")

    pivot = df.pivot_table(
        index="System", columns="Condition", values="F1", aggfunc="first"
    )
    # Order: Clean first, then noise types sorted by SNR descending
    col_order = ["Clean"] + [
        f"{nt}_{snr}dB"
        for nt  in config.NOISE_TYPES
        for snr in sorted(config.SNR_LEVELS, reverse=True)
    ]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    print(pivot.to_string())

    print(f"\n  Full table saved to: {out_path}")
    print(f"{'='*70}\n")

    # Cleanup Whisper hooks
    if whisper_extractor is not None:
        try:
            whisper_extractor.remove_hooks()
        except Exception:
            pass

    return df


if __name__ == "__main__":
    run_noise_experiments()
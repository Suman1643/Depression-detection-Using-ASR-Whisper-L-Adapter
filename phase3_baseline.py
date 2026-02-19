"""
phase3_baseline_fast.py
=======================

Optimized version of phase3_baseline.py
- Faster pitch extraction (yin instead of pyin)
- Feature caching
- Reduced logging

Core methodology unchanged.
"""

import numpy as np
import pandas as pd
import librosa
import pickle
import time
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import config


# --------------------------------------------------
# Feature cache directory
# --------------------------------------------------

FEATURE_DIR = Path("features_cache")
FEATURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Fast feature extraction
# --------------------------------------------------

def extract_features(wav_path: Path, pid: str) -> np.ndarray:

    cache_file = FEATURE_DIR / f"{pid}.npy"

    # Load cached features if exist
    if cache_file.exists():
        return np.load(cache_file)

    try:
        audio, sr = librosa.load(
            str(wav_path),
            sr=config.SAMPLE_RATE,
            mono=True
        )
    except:
        return np.zeros(89, dtype=np.float32)

    # ---------------- MFCC ----------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.N_MFCC,
        hop_length=config.HOP_LENGTH,
        n_fft=config.N_FFT
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # ---------------- Pitch (FAST) ----------------

    try:
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        f0 = f0[~np.isnan(f0)]

        if len(f0) > 0:
            pitch_feats = np.array([
                f0.mean(),
                f0.std(),
                f0.min(),
                f0.max()
            ], dtype=np.float32)
        else:
            pitch_feats = np.zeros(4, dtype=np.float32)

    except:
        pitch_feats = np.zeros(4, dtype=np.float32)

    # ---------------- Energy ----------------

    rms = librosa.feature.rms(
        y=audio,
        hop_length=config.HOP_LENGTH
    )[0]

    energy_feats = np.array([
        rms.mean(),
        rms.std()
    ], dtype=np.float32)

    # ---------------- Pause ----------------

    silence_thresh = 0.05 * rms.max() if rms.max() > 0 else 1e-6

    is_silent = rms < silence_thresh

    transitions = np.diff(is_silent.astype(int))

    pause_count = np.sum(transitions == 1)
    total_pause = np.sum(is_silent)
    pause_ratio = total_pause / max(len(rms), 1)

    pause_feats = np.array([
        pause_count,
        total_pause,
        pause_ratio
    ], dtype=np.float32)

    # ---------------- Final ----------------

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        pitch_feats,
        energy_feats,
        pause_feats
    ]).astype(np.float32)

    # Save cache
    np.save(cache_file, features)

    return features


# --------------------------------------------------
# Dataset builder
# --------------------------------------------------

def build_dataset(csv_file: Path, label_col="PHQ_Binary"):

    df = pd.read_csv(csv_file)

    X, y, ids = [], [], []

    print(f"\nLoading: {csv_file.name} ({len(df)} samples)")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        pid = str(row["Participant_ID"]).strip()
        label = int(row[label_col])

        wav = config.CLEAN_AUDIO_DIR / f"{pid}_clean.wav"

        if not wav.exists():
            wav = config.AUDIO_RAW_DIR / f"{pid}_AUDIO.wav"

        if not wav.exists():
            continue

        feats = extract_features(wav, pid)

        X.append(feats)
        y.append(label)
        ids.append(pid)

    return np.array(X), np.array(y), ids


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(name, y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan

    print(f"\n[{name}]")
    print(f" Accuracy: {acc:.3f}")
    print(f" F1: {f1:.3f}")
    print(f" AUC: {auc:.3f}")

    print(classification_report(
        y_true,
        y_pred,
        target_names=["Not Depressed", "Depressed"],
        zero_division=0
    ))

    return {
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def run_baseline():

    start = time.time()

    print("=" * 60)
    print("PHASE 3 — FAST BASELINE")
    print("=" * 60)

    # ---------------- Load Data ----------------

    X_train, y_train, _ = build_dataset(config.TRAIN_CSV)
    X_dev, y_dev, _ = build_dataset(config.DEV_CSV)
    X_test, y_test, _ = build_dataset(config.TEST_CSV)

    print("\nFeature Shapes:")
    print("Train:", X_train.shape)
    print("Dev  :", X_dev.shape)
    print("Test :", X_test.shape)

    # ---------------- Models ----------------

    classifiers = {

        "SVM": Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=config.SEED
            ))
        ]),

        "RandomForest": Pipeline([
            ("impute", SimpleImputer()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=config.SEED
            ))
        ]),

        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.SEED
            ))
        ])
    }

    results = []

    # ---------------- Train ----------------

    for name, model in classifiers.items():

        print(f"\nTraining: {name}")

        t0 = time.time()

        model.fit(X_train, y_train)

        print(f"Done in {time.time() - t0:.1f}s")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        res = evaluate(name, y_test, y_pred, y_prob)

        results.append(res)

        # Save model
        path = config.CHECKPOINT_DIR / f"baseline_{name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(model, f)

        print("Saved:", path)

    # ---------------- Save Results ----------------

    df = pd.DataFrame(results)

    out = config.RESULTS_DIR / "baseline_results_fast.csv"

    df.to_csv(out, index=False)

    print("\nSaved:", out)

    print("\nTotal time:", time.time() - start, "seconds")


# --------------------------------------------------

if __name__ == "__main__":

    run_baseline()

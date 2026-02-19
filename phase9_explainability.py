"""
phase9_explainability.py
========================
PHASE 9 — Feature Importance, Explainability & Psychological Interpretation

Analyses:
  1. Fusion weights visualisation — which Whisper layers matter most.
  2. SHAP values on the baseline (RandomForest) for traditional features.
  3. Layer-type breakdown: encoder vs decoder contribution.
  4. Psychological interpretation table (feature → clinical meaning).

Outputs (in outputs/results/):
  • fusion_weights_plot.png
  • shap_summary_plot.png
  • layer_importance_barplot.png
  • interpretation_table.csv

Usage:
  python phase9_explainability.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
from pathlib import Path

matplotlib.use("Agg")  # non-interactive backend for server environments

import config

# Optional imports (graceful degradation if not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[NOTE] shap not installed — SHAP analysis will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fusion weight visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_fusion_weights():
    weights_path = config.RESULTS_DIR / "fusion_weights.npy"
    if not weights_path.exists():
        print("[SKIP] fusion_weights.npy not found — run phase7 first.")
        return

    weights = np.load(str(weights_path))      # (24,)
    layer_names = (
        [f"Enc {i}" for i in range(12)] +
        [f"Dec {i}" for i in range(12)]
    )
    colours = ["#3B82F6"] * 12 + ["#F59E0B"] * 12   # blue=encoder, amber=decoder

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(layer_names, weights, color=colours, edgecolor="white", linewidth=0.5)

    # Annotate top-5 layers
    top5_idx = np.argsort(weights)[-5:]
    for idx in top5_idx:
        ax.annotate(f"{weights[idx]:.3f}",
                    xy=(idx, weights[idx]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold", color="#1F2937")

    ax.set_title("Learned Layer Fusion Weights\n(Blue = Encoder, Amber = Decoder)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Whisper Layer", fontsize=11)
    ax.set_ylabel("Softmax Weight", fontsize=11)
    ax.set_xticks(range(24))
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
    ax.axvline(x=11.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="Encoder | Decoder boundary")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = config.RESULTS_DIR / "fusion_weights_plot.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"✔ Fusion weights plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP analysis on Random Forest baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis():
    if not SHAP_AVAILABLE:
        return

    rf_path = config.CHECKPOINT_DIR / "baseline_RandomForest.pkl"
    X_test_path = config.FEATURE_DIR / "trad_X_test.npy"
    y_test_path = config.FEATURE_DIR / "trad_y_test.npy"

    if not rf_path.exists() or not X_test_path.exists():
        print("[SKIP] RandomForest model or test features not found — run phase3 first.")
        return

    with open(rf_path, "rb") as f:
        pipe = pickle.load(f)

    X_test = np.load(str(X_test_path))
    y_test = np.load(str(y_test_path))

    # Get the RF step from the pipeline (handle imputer+clf pipelines)
    rf = pipe.named_steps.get("clf", pipe)

    # Build feature names for 89-d vector
    mfcc_names  = [f"MFCC_mean_{i}" for i in range(40)]
    mfcc_names += [f"MFCC_std_{i}"  for i in range(40)]
    feat_names  = (mfcc_names +
                   ["Pitch_mean", "Pitch_std", "Pitch_min", "Pitch_max"] +
                   ["Energy_mean", "Energy_std"] +
                   ["Pause_count", "Pause_frames", "Pause_ratio"])

    # Transform X through imputer/scaler steps (before the clf)
    X_trans = X_test
    for step_name, step in pipe.steps[:-1]:   # all steps except last (clf)
        X_trans = step.transform(X_trans)

    explainer = shap.TreeExplainer(rf)
    shap_vals  = explainer.shap_values(X_trans)

    # For binary classification, shap_values can be [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class 1 = depressed

    # Compute mean absolute SHAP per feature
    mean_shap = np.mean(np.abs(shap_vals), axis=0)
    top20_idx  = np.argsort(mean_shap)[-20:][::-1]
    top20_names  = [feat_names[i] for i in top20_idx]
    top20_values = mean_shap[top20_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(20), top20_values[::-1],
            color=["#EF4444" if "MFCC" in n else
                   "#3B82F6" if "Pitch" in n else
                   "#10B981" if "Energy" in n else
                   "#F59E0B" for n in top20_names[::-1]])
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Top-20 Feature Importances (SHAP)\nRandom Forest — Depression Detection",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = config.RESULTS_DIR / "shap_summary_plot.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"✔ SHAP summary plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoder vs Decoder layer importance bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_importance():
    weights_path = config.RESULTS_DIR / "fusion_weights.npy"
    if not weights_path.exists():
        return

    weights = np.load(str(weights_path))
    enc_weights = weights[:12]
    dec_weights = weights[12:]

    summary = {
        "Component": ["Encoder (avg)", "Decoder (avg)",
                       "Encoder (max)", "Decoder (max)",
                       "Encoder (sum)", "Decoder (sum)"],
        "Value": [enc_weights.mean(), dec_weights.mean(),
                  enc_weights.max(),  dec_weights.max(),
                  enc_weights.sum(),  dec_weights.sum()],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(config.RESULTS_DIR / "layer_summary.csv", index=False)

    # Layer-level bar chart with enc/dec split
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, vals, title, colour in zip(
        axes,
        [enc_weights, dec_weights],
        ["Encoder Layers (0–11)", "Decoder Layers (0–11)"],
        ["#3B82F6", "#F59E0B"],
    ):
        ax.bar(range(len(vals)), vals, color=colour, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer Index", fontsize=10)
        ax.set_ylabel("Fusion Weight", fontsize=10)
        ax.set_xticks(range(len(vals)))
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Per-Layer Contribution to Depression Prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = config.RESULTS_DIR / "layer_importance_barplot.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Layer importance plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Psychological interpretation table
# ─────────────────────────────────────────────────────────────────────────────

def build_interpretation_table():
    rows = [
        # Feature Group | Feature Name | Clinical Meaning | DSM-5 Link
        ("Pitch",  "Low mean F0",        "Reduced vocal affect", "Depressed mood, Anhedonia"),
        ("Pitch",  "High F0 variability","Emotional dysregulation", "Psychomotor agitation"),
        ("Pitch",  "Low F0 variability", "Flat affect / monotone speech", "Depressed mood"),
        ("Energy", "Low RMS mean",       "Reduced vocal energy / effort", "Fatigue, loss of energy"),
        ("Energy", "High RMS variability","Inconsistent vocal effort", "Concentration difficulties"),
        ("Pause",  "High pause count",   "Longer response latency", "Psychomotor slowing"),
        ("Pause",  "High pause ratio",   "More silence than speech", "Reduced speech output"),
        ("MFCC",   "Low MFCC-1 (energy cepstral)", "Global spectral energy shift", "Fatigue"),
        ("MFCC",   "MFCC-2/3 change",   "Formant structure change", "Muscle tension, affect"),
        ("Whisper Encoder", "Later layers dominant",
         "High-level semantic/prosodic encoding used", "Abstract depression cues"),
        ("Whisper Decoder", "Early layers dominant",
         "Linguistic surface features relevant", "Verbal content cues"),
        ("Whisper Decoder", "Later layers dominant",
         "Contextual meaning critical", "Hopelessness phrasing"),
    ]
    df = pd.DataFrame(rows, columns=["Feature_Group", "Feature", "Meaning", "DSM5_Link"])
    out = config.RESULTS_DIR / "interpretation_table.csv"
    df.to_csv(str(out), index=False)

    print("\n━━━━ Psychological Feature Interpretation Table ━━━━")
    print(df.to_string(index=False))
    print(f"\n✔ Saved to {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Results summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_comparison():
    baseline_path = config.RESULTS_DIR / "baseline_results.csv"
    deep_path     = config.RESULTS_DIR / "deep_results.json"

    has_baseline = baseline_path.exists()
    has_deep     = deep_path.exists()

    if not has_baseline and not has_deep:
        print("[SKIP] No result files found — run phases 3 & 7 first.")
        return

    rows = []
    if has_baseline:
        df_b = pd.read_csv(baseline_path)
        for _, r in df_b.iterrows():
            rows.append({"Model": r["model"], "Accuracy": r["accuracy"],
                         "F1": r["f1"], "AUC": r["auc"]})
    if has_deep:
        with open(deep_path) as f:
            d = json.load(f)
        rows.append({"Model": "Whisper+LAdapter",
                     "Accuracy": d.get("accuracy", np.nan),
                     "F1": d.get("f1", np.nan),
                     "AUC": d.get("auc", np.nan)})

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colours = ["#6366F1", "#F59E0B", "#10B981", "#EF4444"]

    for ax, metric in zip(axes, ["Accuracy", "F1", "AUC"]):
        vals  = df[metric].values
        names = df["Model"].values
        bars  = ax.bar(names, vals,
                       color=colours[:len(names)],
                       alpha=0.85, edgecolor="white")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Depression Detection — System Comparison (Test Set)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()

    out = config.RESULTS_DIR / "results_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Results comparison plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability():
    print(f"\n{'='*60}")
    print("PHASE 9 — Explainability & Psychological Interpretation")
    print(f"{'='*60}")

    plot_fusion_weights()
    run_shap_analysis()
    plot_layer_importance()
    build_interpretation_table()
    plot_results_comparison()

    print(f"\n✔ All explainability outputs saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    run_explainability()
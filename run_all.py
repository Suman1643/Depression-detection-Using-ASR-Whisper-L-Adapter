"""
run_all.py
==========
Master script — runs the full depression detection pipeline end-to-end.

Usage:
  python run_all.py              # full pipeline
  python run_all.py --phase 3   # single phase only
  python run_all.py --from 4    # start from a specific phase

Phases:
  2 — Audio standardisation + noise generation
  3 — Traditional features + baseline classifiers (SVM / RF / LR)
  4 — Whisper hidden layer extraction
  5 — (Adapter module — no separate run; used by phase 7)
  6 — (Fusion module — no separate run; used by phase 7)
  7 — Full model training (Adapters + Fusion + Classifier)
  8 — Noise robustness experiments
  9 — Explainability + psychological interpretation + plots
"""

import sys
import time
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Depression Detection Pipeline")
parser.add_argument("--phase", type=int, default=None,
                    help="Run only this phase number (2–9).")
parser.add_argument("--from",  type=int, default=None, dest="start_from",
                    help="Start pipeline from this phase number.")
parser.add_argument("--skip-whisper", action="store_true",
                    help="Skip phase 4 (Whisper extraction) — use if already done.")
args = parser.parse_args()


def section(n: int, title: str):
    print(f"\n{'━'*60}")
    print(f"  PHASE {n}: {title}")
    print(f"{'━'*60}")


def elapsed(t0: float) -> str:
    secs = time.time() - t0
    return f"{secs:.1f}s" if secs < 60 else f"{secs/60:.1f}min"


# ─────────────────────────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────────────────────────

def should_run(phase_num: int) -> bool:
    if args.phase is not None:
        return args.phase == phase_num
    if args.start_from is not None:
        return phase_num >= args.start_from
    return True


def main():
    overall_start = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Depression Detection — Full Research Pipeline          ║")
    print("║   Whisper + L-Adapters + Layer Fusion                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results_summary = {}

    # ── Phase 2: Preprocessing ───────────────────────────────────────────────
    if should_run(2):
        section(2, "Audio Standardisation + Noise Addition")
        t0 = time.time()
        try:
            from phase2_preprocess import preprocess_all
            preprocess_all()
            print(f"  ✔ Phase 2 complete ({elapsed(t0)})")
        except Exception as e:
            print(f"  ✗ Phase 2 failed: {e}")

    # ── Phase 3: Baseline ─────────────────────────────────────────────────────
    if should_run(3):
        section(3, "Traditional Features + Baseline Classifiers")
        t0 = time.time()
        try:
            from phase3_baseline import run_baseline
            res = run_baseline()
            results_summary["baseline"] = res
            print(f"  ✔ Phase 3 complete ({elapsed(t0)})")
        except Exception as e:
            print(f"  ✗ Phase 3 failed: {e}")

    # ── Phase 4: Whisper extraction ───────────────────────────────────────────
    if should_run(4) and not args.skip_whisper:
        section(4, "Whisper Hidden Layer Extraction")
        t0 = time.time()
        try:
            from phase4_whisper_extract import extract_all_whisper_features
            extract_all_whisper_features()
            print(f"  ✔ Phase 4 complete ({elapsed(t0)})")
        except ImportError as e:
            print(f"  ✗ Phase 4 failed — openai-whisper not installed: {e}")
            print("    Install with: pip install openai-whisper")
        except Exception as e:
            print(f"  ✗ Phase 4 failed: {e}")
    elif should_run(4) and args.skip_whisper:
        print("\n  [Phase 4 skipped — --skip-whisper flag set]")

    # ── Phase 5/6 are modules used by Phase 7; no standalone run needed ───────
    if should_run(5):
        section(5, "L-Adapter Architecture (sanity check)")
        try:
            from phase5_adapters import LAdapterBank
            import torch
            bank = LAdapterBank()
            print(f"  LAdapterBank instantiated — "
                  f"{sum(p.numel() for p in bank.parameters()):,} params")
        except Exception as e:
            print(f"  ✗ Adapter check failed: {e}")

    if should_run(6):
        section(6, "Layer Fusion (sanity check)")
        try:
            import torch
            from phase6_fusion import LayerFusion
            import config
            lf = LayerFusion(24)
            dummy = [torch.randn(2, config.ADAPTER_OUT_DIM) for _ in range(24)]
            out = lf(dummy)
            print(f"  LayerFusion output shape: {out.shape}  ✔")
        except Exception as e:
            print(f"  ✗ Fusion check failed: {e}")

    # ── Phase 7: Training ─────────────────────────────────────────────────────
    if should_run(7):
        section(7, "Full Model Training (Adapters + Fusion + Classifier)")
        t0 = time.time()
        try:
            from phase6_fusion import run_training
            res = run_training()
            results_summary["deep_model"] = res
            print(f"  ✔ Phase 7 complete ({elapsed(t0)})")
        except Exception as e:
            print(f"  ✗ Phase 7 failed: {e}")

    # ── Phase 8: Noise experiments ────────────────────────────────────────────
    if should_run(8):
        section(8, "Noise Robustness Experiments")
        t0 = time.time()
        try:
            from phase8_noise_experiment import run_noise_experiments
            run_noise_experiments()
            print(f"  ✔ Phase 8 complete ({elapsed(t0)})")
        except Exception as e:
            print(f"  ✗ Phase 8 failed: {e}")

    # ── Phase 9: Explainability ───────────────────────────────────────────────
    if should_run(9):
        section(9, "Explainability + Psychological Interpretation")
        t0 = time.time()
        try:
            from phase9_explainability import run_explainability
            run_explainability()
            print(f"  ✔ Phase 9 complete ({elapsed(t0)})")
        except Exception as e:
            print(f"  ✗ Phase 9 failed: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Pipeline complete — total time: {elapsed(overall_start)}")
    print(f"{'═'*60}")
    if results_summary:
        print("\n  Results summary:")
        for system, res in results_summary.items():
            if isinstance(res, dict):
                for model, m in res.items():
                    if isinstance(m, dict):
                        print(f"    {system} / {model}: "
                              f"Acc={m.get('accuracy','?'):.3f}  "
                              f"F1={m.get('f1','?'):.3f}  "
                              f"AUC={m.get('auc','?'):.3f}")
    print()


if __name__ == "__main__":
    main()
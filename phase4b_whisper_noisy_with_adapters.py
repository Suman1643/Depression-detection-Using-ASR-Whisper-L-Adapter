"""
phase4b_whisper_noisy_with_adapters.py
=======================================
Extract Whisper Features + Apply L-Adapters for Noisy Audio

This extracts Whisper hidden layers AND applies the trained L-Adapters,
saving the final adapter embeddings (24 × 64-dim vectors per file).

Output is much smaller and Phase 8 is much faster!

Pipeline per noisy audio file:
  1. Whisper encoder/decoder (24 layers × 512-dim) → hidden states
  2. L-Adapter bank (24 adapters) → 24 × 64-dim embeddings
  3. Save embeddings to NPZ

Output: outputs/features_noisy/<noise_type>/<snr>dB/<pid>_adapter.npz
  - Contains: 24 adapter embeddings (encoder_emb_0...11, decoder_emb_0...11)
  - Each embedding: (1, 64) instead of (T, 512)

Usage:
  python phase4b_whisper_noisy_with_adapters.py --gpu      # GPU mode
  python phase4b_whisper_noisy_with_adapters.py            # CPU mode
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import argparse

import config
from phase4_whisper_extract import WhisperHiddenExtractor, load_whisper
from phase5_adapters import LAdapterBank


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

NOISY_FEATURE_DIR = config.OUT_DIR / "features_noisy"
NOISY_FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_all_participant_ids():
    """Get all unique participant IDs from train/dev/test splits."""
    ids = set()
    for csv_path in [config.TRAIN_CSV, config.DEV_CSV, config.TEST_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            ids.update(df["Participant_ID"].astype(str).str.strip().tolist())
    return sorted(ids)


def get_noisy_audio_path(pid: str, noise_type: str, snr: int) -> Path | None:
    """Get path to noisy audio file."""
    path = config.NOISY_AUDIO_DIR / noise_type / f"{snr}dB" / f"{pid}_{noise_type}_{snr}dB.wav"
    return path if path.exists() else None


def get_output_path(pid: str, noise_type: str, snr: int) -> Path:
    """Get output path for extracted adapter embeddings."""
    out_dir = NOISY_FEATURE_DIR / noise_type / f"{snr}dB"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{pid}_adapter.npz"  # Note: _adapter.npz instead of _whisper.npz


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_noisy_features(use_gpu=False, encoder_only=False):
    """
    Extract Whisper features + apply L-Adapters for all noisy audio files.
    
    Parameters
    ----------
    use_gpu : bool
        If True, use GPU (much faster). If False, use CPU.
    encoder_only : bool
        If True, only extract encoder layers (2x faster, 12 layers instead of 24)
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 4B — Noisy Audio: Whisper + L-Adapters")
    print(f"{'='*70}")
    
    # Device selection
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"\n  🚀 GPU Mode: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"\n  💻 CPU Mode")
        if use_gpu:
            print(f"     (GPU requested but not available)")
    
    print(f"\n  Configuration:")
    print(f"    Device        : {device}")
    print(f"    Whisper model : {config.WHISPER_MODEL}")
    print(f"    Encoder-only  : {encoder_only}")
    print(f"    Adapter layers: {12 if encoder_only else 24}")
    print(f"    Embedding dim : {config.ADAPTER_OUT_DIM}")
    print(f"    Output dir    : {NOISY_FEATURE_DIR}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Load Whisper model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  [1/3] Loading Whisper model...", end=" ", flush=True)
    whisper_model = load_whisper(config.WHISPER_MODEL, device)
    extractor = WhisperHiddenExtractor(whisper_model, encoder_only=encoder_only)
    print("✓")
    
    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Load trained L-Adapters
    # ══════════════════════════════════════════════════════════════════════
    print(f"  [2/3] Loading trained L-Adapters...", end=" ", flush=True)
    
    adapter_bank = LAdapterBank(whisper_model=config.WHISPER_MODEL)
    
    # Load adapter weights from trained model
    model_ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    if not model_ckpt.exists():
        print(f"\n\n  ✗ ERROR: Trained model not found at {model_ckpt}")
        print(f"     Please run Phase 7 (training) first!")
        return
    
    # Load full model state dict and extract adapter weights
    full_state = torch.load(model_ckpt, map_location=device)
    adapter_state = {
        k.replace('adapter_bank.', ''): v 
        for k, v in full_state.items() 
        if k.startswith('adapter_bank.')
    }
    adapter_bank.load_state_dict(adapter_state)
    adapter_bank.to(device)
    adapter_bank.eval()
    print("✓")
    
    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Get all participants and build task list
    # ══════════════════════════════════════════════════════════════════════
    print(f"  [3/3] Scanning for noisy audio files...", end=" ", flush=True)
    
    all_ids = get_all_participant_ids()
    
    # Build task list
    tasks = []
    for noise_type in config.NOISE_TYPES:
        for snr in config.SNR_LEVELS:
            for pid in all_ids:
                # Check if already extracted
                out_path = get_output_path(pid, noise_type, snr)
                if out_path.exists():
                    continue
                
                # Check if noisy audio exists
                wav_path = get_noisy_audio_path(pid, noise_type, snr)
                if wav_path is None:
                    continue
                
                tasks.append((pid, noise_type, snr, wav_path, out_path))
    
    print("✓")
    
    print(f"\n  Tasks:")
    print(f"    Total participants  : {len(all_ids)}")
    print(f"    Noise conditions    : {len(config.NOISE_TYPES)} types × {len(config.SNR_LEVELS)} SNRs = {len(config.NOISE_TYPES) * len(config.SNR_LEVELS)}")
    print(f"    Max possible files  : {len(all_ids) * len(config.NOISE_TYPES) * len(config.SNR_LEVELS)}")
    print(f"    To process          : {len(tasks)}")
    
    if len(tasks) == 0:
        print(f"\n  ✓ All features already extracted!")
        extractor.remove_hooks()
        return
    
    # Processing statistics
    stats = {
        'success': 0,
        'failed': 0,
        'times_per_file': []
    }
    
    # Group tasks by condition
    by_condition = {}
    for pid, noise_type, snr, wav_path, out_path in tasks:
        key = f"{noise_type}_{snr}dB"
        if key not in by_condition:
            by_condition[key] = []
        by_condition[key].append((pid, wav_path, out_path))
    
    print(f"\n  {'─'*70}")
    print(f"  Starting extraction...")
    print(f"  {'─'*70}\n")
    
    overall_start = time.time()
    
    # Process each condition
    for cond_idx, (condition, files) in enumerate(by_condition.items(), 1):
        print(f"  [{cond_idx}/{len(by_condition)}] Processing {condition} ({len(files)} files)")
        
        pbar = tqdm(files, desc=f"    {condition}", unit="file", leave=False, ncols=80)
        
        for pid, wav_path, out_path in pbar:
            file_start = time.time()
            
            try:
                # ══════════════════════════════════════════════════════════
                # STEP A: Extract Whisper hidden states
                # ══════════════════════════════════════════════════════════
                raw_hidden = extractor.extract(wav_path)
                
                # Convert to torch tensors
                hidden_states = {
                    k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                    for k, v in raw_hidden.items()
                }
                
                # ══════════════════════════════════════════════════════════
                # STEP B: Apply L-Adapters to get embeddings
                # ══════════════════════════════════════════════════════════
                with torch.no_grad():
                    embeddings = adapter_bank(hidden_states)  # List of 24 tensors (batch, 64)
                
                # ══════════════════════════════════════════════════════════
                # STEP C: Save adapter embeddings
                # ══════════════════════════════════════════════════════════
                # Convert to numpy and save
                output_dict = {}
                num_enc = 12
                for i, emb in enumerate(embeddings):
                    emb_np = emb.cpu().numpy()  # (1, 64)
                    if i < num_enc:
                        output_dict[f"encoder_emb_{i}"] = emb_np
                    else:
                        output_dict[f"decoder_emb_{i - num_enc}"] = emb_np
                
                np.savez_compressed(str(out_path), **output_dict)
                
                file_time = time.time() - file_start
                stats['times_per_file'].append(file_time)
                stats['success'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                pbar.write(f"      ✗ ERROR {pid}: {str(e)[:50]}")
        
        pbar.close()
        
        # Show condition summary
        avg_time = np.mean(stats['times_per_file'][-len(files):]) if stats['times_per_file'] else 0
        print(f"      ✓ {len(files) - stats['failed']}/{len(files)} files ({avg_time:.2f}s avg)")
    
    # Final statistics
    total_time = time.time() - overall_start
    
    print(f"\n  {'='*70}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  {'='*70}")
    
    print(f"\n  📊 Results:")
    print(f"    ✓ Successfully extracted  : {stats['success']:4d} files")
    print(f"    ✗ Failed (errors)         : {stats['failed']:4d} files")
    
    if stats['times_per_file']:
        avg_time = np.mean(stats['times_per_file'])
        min_time = np.min(stats['times_per_file'])
        max_time = np.max(stats['times_per_file'])
        
        print(f"\n  ⚡ Performance:")
        print(f"    Total time                : {total_time/60:.1f} minutes")
        print(f"    Average time per file     : {avg_time:.2f} seconds")
        print(f"    Fastest file              : {min_time:.2f} seconds")
        print(f"    Slowest file              : {max_time:.2f} seconds")
        print(f"    Throughput                : {stats['success']/(total_time/60):.1f} files/min")
    
    print(f"\n  💾 Output:")
    print(f"    Location: {NOISY_FEATURE_DIR}")
    print(f"    Format  : .npz (compressed adapter embeddings)")
    print(f"    Layers  : {12 if encoder_only else 24} adapter outputs")
    print(f"    Dim     : {config.ADAPTER_OUT_DIM} per layer")
    
    # Show file size comparison
    if stats['success'] > 0:
        sample_file = list(NOISY_FEATURE_DIR.rglob("*_adapter.npz"))[0]
        file_size_kb = sample_file.stat().st_size / 1024
        total_size_mb = (file_size_kb * stats['success']) / 1024
        
        # Compare to raw Whisper features
        raw_size_estimate_mb = (24 * 512 * 4 * 1500 / (1024*1024)) * stats['success']  # Rough estimate
        adapter_size_mb = total_size_mb
        savings_pct = 100 * (1 - adapter_size_mb / raw_size_estimate_mb)
        
        print(f"    Avg file size: ~{file_size_kb:.1f} KB")
        print(f"    Total size   : ~{total_size_mb:.1f} MB")
        print(f"    💡 Savings   : ~{savings_pct:.0f}% smaller than raw Whisper features!")
    
    print(f"  {'='*70}\n")
    
    extractor.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Whisper + L-Adapter features for noisy audio"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for extraction (much faster)"
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Extract only encoder layers (2x faster, 12 layers instead of 24)"
    )
    args = parser.parse_args()
    
    extract_all_noisy_features(use_gpu=args.gpu, encoder_only=args.encoder_only)
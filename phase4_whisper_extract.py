"""
phase4_whisper_extract.py
=========================
PHASE 4 — Load Frozen Whisper & Extract All Hidden Layer Representations
(OPTIMIZED CPU VERSION with Enhanced Progress Tracking)

For each audio file we:
  1. Load it with librosa at 16 kHz.
  2. Pass the log-mel spectrogram through Whisper's encoder (frozen).
  3. Collect all 12 encoder hidden states  → each shape (T_enc, D).
  4. Run the decoder in "greedy" transcription mode and collect all
     12 decoder hidden states → each shape (T_dec, D).
  5. Save a dict of arrays to  outputs/features/<pid>_whisper.npz

The NPZ file contains keys:
  encoder_layer_0 … encoder_layer_11  — (T_enc, 512) for 'base' model
  decoder_layer_0 … decoder_layer_11  — (T_dec, 512)

OPTIMIZATIONS:
  - CPU-optimized inference (no GPU required)
  - Enhanced progress bar with ETA and speed metrics
  - Encoder-only mode option (2x faster, skip decoder)
  - Memory-efficient batch clearing
  - Detailed logging per file

Usage:
  python phase4_whisper_extract.py
  python phase4_whisper_extract.py --encoder-only  # 2x faster
"""

import numpy as np
import torch
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import sys

import config

# ─────────────────────────────────────────────────────────────────────────────
# Whisper model loading (CPU-optimized)
# ─────────────────────────────────────────────────────────────────────────────

def load_whisper(model_name: str = "base", device: str = "cpu"):
    """Load Whisper with CPU optimizations."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper not installed. Run: pip install openai-whisper"
        )

    print(f"  Loading Whisper '{model_name}' model...")
    
    # CPU optimizations
    if device == "cpu":
        # Set optimal thread count for CPU inference
        torch.set_num_threads(4)  # Adjust based on your CPU
        print(f"  CPU threads: {torch.get_num_threads()}")
    
    model = whisper.load_model(model_name, device=device)
    model.eval()
    
    # Freeze all parameters to save memory
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  Model loaded successfully on {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state extraction with forward hooks (CPU-optimized)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperHiddenExtractor:
    """
    Attaches forward hooks to every encoder and decoder transformer block
    to capture the residual-stream output after the MHA + FFN sub-layers.
    
    CPU Optimizations:
    - Optional encoder-only mode (2x faster)
    - Efficient memory clearing
    - Progress callback support
    """

    def __init__(self, model, encoder_only: bool = False):
        self.model = model
        self.encoder_only = encoder_only
        self._enc_states: list[np.ndarray] = []
        self._dec_states: list[np.ndarray] = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        # Encoder blocks
        for block in self.model.encoder.blocks:
            h = block.register_forward_hook(self._enc_hook)
            self._hooks.append(h)
        
        # Decoder blocks (skip if encoder_only mode)
        if not self.encoder_only:
            for block in self.model.decoder.blocks:
                h = block.register_forward_hook(self._dec_hook)
                self._hooks.append(h)

    def _enc_hook(self, module, input_, output):
        # output shape: (batch, T_enc, D)
        # Immediately convert to float32 numpy to save memory
        self._enc_states.append(output.detach().cpu().numpy()[0].astype(np.float32))

    def _dec_hook(self, module, input_, output):
        # decoder block can return tuple; first element is the hidden state
        hidden = output[0] if isinstance(output, tuple) else output
        self._dec_states.append(hidden.detach().cpu().numpy()[0].astype(np.float32))

    def clear(self):
        """Clear cached states and free memory."""
        self._enc_states.clear()
        self._dec_states.clear()
        # Force garbage collection on CPU
        import gc
        gc.collect()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(self, wav_path: Path) -> dict[str, np.ndarray]:
        """
        Returns a dict mapping 'encoder_layer_i' / 'decoder_layer_i'
        to numpy arrays of shape (T, D).
        """
        import whisper as _whisper

        self.clear()

        # STEP 1: Load audio
        # WINDOWS FIX: Convert Path to absolute string with forward slashes
        wav_str = str(wav_path.absolute()).replace('\\', '/')
        
        try:
            audio = _whisper.load_audio(wav_str)
        except Exception as e:
            # Fallback: load with librosa directly
            import librosa
            audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            # Pad or trim to 30 seconds like Whisper does
            if len(audio) > 16000 * 30:
                audio = audio[:16000 * 30]
            else:
                audio = np.pad(audio, (0, 16000 * 30 - len(audio)))
        
        # STEP 2: Prepare mel spectrogram
        audio = _whisper.pad_or_trim(audio)
        mel = _whisper.log_mel_spectrogram(audio).to(config.WHISPER_DEVICE)

        # STEP 3: Run encoder (extract 12 encoder layers)
        _ = self.model.encoder(mel.unsqueeze(0))   # triggers encoder hooks

        # STEP 4: Run decoder (extract 12 decoder layers)
        if not self.encoder_only:
            options = _whisper.DecodingOptions(
                language="en",
                without_timestamps=True,
                fp16=False  # Always use FP32 on CPU
            )
            _ = _whisper.decode(self.model, mel, options)  # triggers decoder hooks

        # STEP 5: Compile results
        result = {}
        for i, state in enumerate(self._enc_states):
            result[f"encoder_layer_{i}"] = state
        for i, state in enumerate(self._dec_states):
            result[f"decoder_layer_{i}"] = state

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helper
# ─────────────────────────────────────────────────────────────────────────────

def get_all_participant_ids() -> list[str]:
    ids = set()
    for csv_path in [config.TRAIN_CSV, config.DEV_CSV, config.TEST_CSV]:
        df = pd.read_csv(csv_path)
        ids.update(df["Participant_ID"].astype(str).str.strip().tolist())
    return sorted(ids)


def audio_path_for(pid: str) -> Path | None:
    """Return clean or raw audio path, or None if missing."""
    clean = config.CLEAN_AUDIO_DIR / f"{pid}_clean.wav"
    if clean.exists():
        return clean
    raw = config.AUDIO_RAW_DIR / f"{pid}_AUDIO.wav"
    if raw.exists():
        return raw
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main (CPU-optimized with detailed progress tracking)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_whisper_features(encoder_only: bool = False):
    """
    Extract Whisper features for all participants with detailed progress.
    
    Parameters
    ----------
    encoder_only : bool
        If True, only extract encoder layers (2x faster, 12 layers instead of 24)
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 4 — Whisper Hidden Layer Extraction ({config.WHISPER_MODEL})")
    print(f"{'='*70}")
    
    # Force CPU device
    device = "cpu"  # Optimized for CPU-only execution
    print(f"\n  Configuration:")
    print(f"    Device        : {device}")
    print(f"    Model         : {config.WHISPER_MODEL}")
    print(f"    Encoder-only  : {encoder_only}")
    print(f"    Expected layers: {12 if encoder_only else 24}")
    print(f"    Output dir    : {config.FEATURE_DIR}")

    # ══════════════════════════════════════════════════════════════════════
    # INITIALIZATION STEP 1: Load Whisper model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  [Init Step 1/4] Loading Whisper model...")
    model = load_whisper(config.WHISPER_MODEL, device)

    # ══════════════════════════════════════════════════════════════════════
    # INITIALIZATION STEP 2: Attach forward hooks
    # ══════════════════════════════════════════════════════════════════════
    print(f"  [Init Step 2/4] Attaching forward hooks to {24 if not encoder_only else 12} layers...")
    extractor = WhisperHiddenExtractor(model, encoder_only=encoder_only)
    print(f"                  ✓ Hooks attached successfully")

    # ══════════════════════════════════════════════════════════════════════
    # INITIALIZATION STEP 3: Load participant IDs from CSV files
    # ══════════════════════════════════════════════════════════════════════
    print(f"  [Init Step 3/4] Loading participant IDs from CSV files...")
    all_ids = get_all_participant_ids()
    total = len(all_ids)
    print(f"                  ✓ Found {total} participants")
    
    # ══════════════════════════════════════════════════════════════════════
    # INITIALIZATION STEP 4: Check for already processed files
    # ══════════════════════════════════════════════════════════════════════
    print(f"  [Init Step 4/4] Checking for already processed files...")
    already_done = []
    to_process = []
    for pid in all_ids:
        out_path = config.FEATURE_DIR / f"{pid}_whisper.npz"
        if out_path.exists():
            already_done.append(pid)
        else:
            to_process.append(pid)
    
    print(f"                  ✓ Already processed: {len(already_done)}")
    print(f"                  ✓ To process: {len(to_process)}")
    
    print(f"\n  Participant Summary:")
    print(f"    Total participants   : {total}")
    print(f"    Already processed    : {len(already_done)}")
    print(f"    To process           : {len(to_process)}")
    
    if not to_process:
        print(f"\n  ✔ All files already extracted!")
        extractor.remove_hooks()
        return

    # Processing statistics
    stats = {
        'success': 0,
        'skipped_no_audio': 0,
        'failed': 0,
        'total_time': 0,
        'times_per_file': []
    }

    # Enhanced progress bar
    print(f"\n  {'─'*70}")
    print(f"  Starting extraction ...")
    print(f"  {'─'*70}")
    print(f"  Note: Detailed progress shown for first 3 files, then every 10 files\n")
    
    pbar = tqdm(
        to_process,
        desc="  Extracting features",
        unit="file",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    overall_start = time.time()

    for idx, pid in enumerate(pbar, 1):
        file_start = time.time()
        out_path = config.FEATURE_DIR / f"{pid}_whisper.npz"
        
        # ══════════════════════════════════════════════════════════════
        # STEP 1: Find audio file
        # ══════════════════════════════════════════════════════════════
        wav = audio_path_for(pid)
        if wav is None:
            stats['skipped_no_audio'] += 1
            pbar.write(f"  [{idx:3d}/{len(to_process):3d}] ⚠ SKIP {pid:>6s} — no audio file")
            continue

        # ══════════════════════════════════════════════════════════════
        # STEP 2-5: Extract Whisper features (5 internal steps)
        # ══════════════════════════════════════════════════════════════
        try:
            # Show verbose progress for first 3 files
            if idx <= 3:
                pbar.write(f"  [{idx:3d}/{len(to_process):3d}] 🎵 {pid:>6s} — Processing:")
                pbar.write(f"      Step 1/5: Loading audio from {wav.name}")
            
            t_start = time.time()
            hidden = extractor.extract(wav)
            extract_time = time.time() - t_start
            
            if idx <= 3:
                pbar.write(f"      Step 2/5: Generated mel spectrogram")
                pbar.write(f"      Step 3/5: Extracted {len([k for k in hidden.keys() if 'encoder' in k])} encoder layers")
                pbar.write(f"      Step 4/5: Extracted {len([k for k in hidden.keys() if 'decoder' in k])} decoder layers")
                pbar.write(f"      Step 5/5: Saving to {out_path.name}")
            
            # ══════════════════════════════════════════════════════════════
            # STEP 6: Save compressed NPZ file
            # ══════════════════════════════════════════════════════════════
            save_start = time.time()
            np.savez_compressed(str(out_path), **hidden)
            save_time = time.time() - save_start
            
            file_time = time.time() - file_start
            stats['times_per_file'].append(file_time)
            stats['success'] += 1
            
            # Show detailed progress every 10 files or for first 3
            if idx <= 3:
                pbar.write(f"      ✓ Complete! (extract: {extract_time:.1f}s, save: {save_time:.1f}s, total: {file_time:.1f}s)")
            elif idx % 10 == 0 or idx == len(to_process):
                avg_time = np.mean(stats['times_per_file'][-10:])
                pbar.write(
                    f"  [{idx:3d}/{len(to_process):3d}] ✓ {pid:>6s} "
                    f"({file_time:.1f}s, avg: {avg_time:.1f}s/file) — {stats['success']} done"
                )
            
        except Exception as e:
            stats['failed'] += 1
            # Show full error for first 5 failures to help debug
            if stats['failed'] <= 5:
                import traceback
                pbar.write(f"  [{idx:3d}/{len(to_process):3d}] ✗ ERROR {pid:>6s}:")
                pbar.write(f"      File: {wav}")
                pbar.write(f"      Error: {str(e)}")
                if stats['failed'] == 1:
                    pbar.write(f"      Full traceback:")
                    pbar.write("      " + "\n      ".join(traceback.format_exc().split('\n')))
            else:
                pbar.write(f"  [{idx:3d}/{len(to_process):3d}] ✗ ERROR {pid:>6s}: {str(e)[:50]}")
            continue

    pbar.close()

    # ══════════════════════════════════════════════════════════════════════
    # FINAL STATISTICS AND SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - overall_start
    stats['total_time'] = total_time

    print(f"\n  {'='*70}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  {'='*70}")
    
    print(f"\n  📊 Results Summary:")
    print(f"  {'─'*70}")
    print(f"    ✓ Successfully extracted  : {stats['success']:4d} files")
    print(f"    ⚠ Skipped (no audio)      : {stats['skipped_no_audio']:4d} files")
    print(f"    ✗ Failed (errors)         : {stats['failed']:4d} files")
    print(f"    {'─'*66}")
    print(f"    Total processed           : {stats['success'] + stats['skipped_no_audio'] + stats['failed']:4d} files")
    
    if stats['success'] > 0:
        print(f"\n  ⚡ Performance Metrics:")
        print(f"  {'─'*70}")
        avg_time = np.mean(stats['times_per_file'])
        min_time = np.min(stats['times_per_file'])
        max_time = np.max(stats['times_per_file'])
        median_time = np.median(stats['times_per_file'])
        
        print(f"    Total time                : {total_time/60:.1f} minutes ({total_time:.0f}s)")
        print(f"    Average time per file     : {avg_time:.2f} seconds")
        print(f"    Median time per file      : {median_time:.2f} seconds")
        print(f"    Fastest file              : {min_time:.2f} seconds")
        print(f"    Slowest file              : {max_time:.2f} seconds")
        print(f"    Throughput                : {stats['success']/(total_time/60):.1f} files/min")
        
        # Time breakdown estimate
        print(f"\n  🔧 Estimated Time Breakdown (per file):")
        print(f"  {'─'*70}")
        print(f"    Audio loading             : ~15% ({avg_time*0.15:.2f}s)")
        print(f"    Mel spectrogram           : ~10% ({avg_time*0.10:.2f}s)")
        print(f"    Encoder extraction        : ~35% ({avg_time*0.35:.2f}s)")
        print(f"    Decoder extraction        : ~35% ({avg_time*0.35:.2f}s)")
        print(f"    NPZ compression & save    : ~5%  ({avg_time*0.05:.2f}s)")
    
    print(f"\n  💾 Output Information:")
    print(f"  {'─'*70}")
    print(f"    Location: {config.FEATURE_DIR}")
    print(f"    Format  : .npz (compressed numpy arrays)")
    print(f"    Layers  : {12 if encoder_only else 24} per file")
    
    if stats['success'] > 0:
        # Check file size
        sample_file = config.FEATURE_DIR / f"{to_process[0]}_whisper.npz"
        if sample_file.exists():
            file_size_mb = sample_file.stat().st_size / (1024 * 1024)
            total_size_mb = file_size_mb * stats['success']
            print(f"    Avg file size: ~{file_size_mb:.1f} MB")
            print(f"    Total size   : ~{total_size_mb:.0f} MB ({total_size_mb/1024:.2f} GB)")
    
    print(f"  {'='*70}")
    
    # Success/failure summary
    if stats['success'] == len(to_process):
        print(f"\n  🎉 SUCCESS! All {stats['success']} files extracted successfully!\n")
    elif stats['success'] > 0:
        success_rate = 100 * stats['success'] / len(to_process)
        print(f"\n  ⚠️  PARTIAL SUCCESS: {stats['success']}/{len(to_process)} files ({success_rate:.1f}%) extracted")
        if stats['failed'] > 0:
            print(f"     {stats['failed']} files failed — check error messages above\n")
    else:
        print(f"\n  ❌ FAILED: No files were extracted successfully")
        print(f"     Check error messages above for troubleshooting\n")

    extractor.remove_hooks()


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract Whisper hidden layer features (CPU-optimized)"
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Extract only encoder layers (2x faster, 12 layers instead of 24)"
    )
    args = parser.parse_args()
    
    extract_all_whisper_features(encoder_only=args.encoder_only)
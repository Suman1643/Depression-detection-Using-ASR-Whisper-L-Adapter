"""
diagnose_paths.py
=================
Diagnostic tool to check your file paths and help configure the system correctly.

Run this BEFORE running any phases to verify your setup.

Usage:
    python diagnose_paths.py
"""

import os
from pathlib import Path
import pandas as pd

print("="*70)
print("  DEPRESSION DETECTION — PATH DIAGNOSTICS")
print("="*70)

# Get current directory
current_dir = Path.cwd()
print(f"\nCurrent directory: {current_dir}")

# ─────────────────────────────────────────────────────────────────────────
# Step 1: Find CSV files
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("STEP 1: Checking CSV files")
print("─"*70)

csv_files = {
    "train_split.csv": None,
    "dev_split.csv": None,
    "test_split.csv": None,
    "Detailed_PHQ8_Labels.csv": None
}

# Check in current directory
for csv_name in csv_files.keys():
    path = current_dir / csv_name
    if path.exists():
        csv_files[csv_name] = path
        print(f"✓ Found: {csv_name}")
    else:
        print(f"✗ Missing: {csv_name}")

# Check in E-label folder
elabel_dir = current_dir / "E-label"
if elabel_dir.exists():
    print(f"\n✓ Found E-label directory: {elabel_dir}")
    for csv_name in csv_files.keys():
        if csv_files[csv_name] is None:
            path = elabel_dir / csv_name
            if path.exists():
                csv_files[csv_name] = path
                print(f"  ✓ Found: {csv_name} (in E-label/)")

# ─────────────────────────────────────────────────────────────────────────
# Step 2: Find audio directory
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("STEP 2: Checking audio directory")
print("─"*70)

audio_dir = None
possible_paths = [
    current_dir / "E-label" / "full-extended-audio",
    current_dir / "full-extended-audio",
    elabel_dir / "audio",
    current_dir / "audio"
]

for path in possible_paths:
    if path.exists():
        # Check if it actually contains WAV files
        wav_files = list(path.glob("*.wav"))
        if wav_files:
            audio_dir = path
            print(f"✓ Found audio directory: {path}")
            print(f"  Contains {len(wav_files)} .wav files")
            print(f"  First file: {wav_files[0].name}")
            break

if audio_dir is None:
    print("✗ No audio directory found!")
    print("\nSearching for .wav files anywhere in project...")
    wav_search = list(current_dir.rglob("*.wav"))
    if wav_search:
        print(f"  Found {len(wav_search)} .wav files in:")
        parent_dirs = set(f.parent for f in wav_search[:5])
        for d in parent_dirs:
            print(f"    {d}")
    else:
        print("  No .wav files found anywhere!")

# ─────────────────────────────────────────────────────────────────────────
# Step 3: Sample participant IDs from CSVs
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("STEP 3: Reading participant IDs from CSVs")
print("─"*70)

all_pids = []
for csv_name, csv_path in csv_files.items():
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "Participant_ID" in df.columns:
                pids = df["Participant_ID"].astype(str).str.strip().tolist()
                all_pids.extend(pids)
                print(f"✓ {csv_name}: {len(pids)} participants")
                print(f"  Sample IDs: {pids[:3]}")
        except Exception as e:
            print(f"✗ Error reading {csv_name}: {e}")

unique_pids = sorted(set(all_pids))
print(f"\n  Total unique participants: {len(unique_pids)}")

# ─────────────────────────────────────────────────────────────────────────
# Step 4: Check if audio files exist for participants
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("STEP 4: Matching audio files to participant IDs")
print("─"*70)

if audio_dir and unique_pids:
    sample_pids = unique_pids[:10]  # Check first 10
    found_count = 0
    
    for pid in sample_pids:
        # Try different naming patterns
        patterns = [
            f"{pid}_AUDIO.wav",
            f"{pid}.wav",
            f"{pid}_audio.wav",
            f"AUDIO_{pid}.wav"
        ]
        
        found = False
        for pattern in patterns:
            file_path = audio_dir / pattern
            if file_path.exists():
                print(f"✓ Participant {pid}: {pattern}")
                found = True
                found_count += 1
                break
        
        if not found:
            print(f"✗ Participant {pid}: No audio file found")
    
    print(f"\n  Matched {found_count}/{len(sample_pids)} sample participants")

# ─────────────────────────────────────────────────────────────────────────
# Step 5: Generate corrected config.py
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("STEP 5: Generating corrected config")
print("─"*70)

if csv_files["train_split.csv"] and audio_dir:
    print("\n✓ All required paths found!")
    print("\nRecommended config.py settings:")
    print("-" * 50)
    
    # Determine relative paths from current directory
    if audio_dir.is_relative_to(current_dir):
        audio_rel = audio_dir.relative_to(current_dir)
    else:
        audio_rel = audio_dir
    
    print(f'ROOT_DIR = Path("{elabel_dir.name if elabel_dir.exists() else "."}")')
    print(f'AUDIO_RAW_DIR = Path(r"{audio_dir}")')
    
    # CSV paths
    for csv_name, csv_path in csv_files.items():
        if csv_path:
            if csv_path.parent == current_dir:
                print(f'{csv_name.replace(".csv", "").upper().replace("DETAILED_PHQ8_LABELS", "LABELS")}_CSV = Path("{csv_name}")')
            else:
                rel_path = csv_path.relative_to(current_dir) if csv_path.is_relative_to(current_dir) else csv_path
                print(f'{csv_name.replace(".csv", "").upper().replace("DETAILED_PHQ8_LABELS", "LABELS")}_CSV = Path(r"{rel_path}")')
    
    print("-" * 50)
    
    # Save config suggestion to file
    config_suggestion = f"""# Auto-generated config suggestions based on your directory structure
# Copy these lines into config.py (lines 14-22)

from pathlib import Path

ROOT_DIR = Path(r"{elabel_dir if elabel_dir.exists() else current_dir}")
AUDIO_RAW_DIR = Path(r"{audio_dir}")
TRANSCRIPT_DIR = ROOT_DIR / "full-extended-transcript"

TRAIN_CSV = Path(r"{csv_files['train_split.csv'] if csv_files['train_split.csv'] else 'train_split.csv'}")
DEV_CSV = Path(r"{csv_files['dev_split.csv'] if csv_files['dev_split.csv'] else 'dev_split.csv'}")
TEST_CSV = Path(r"{csv_files['test_split.csv'] if csv_files['test_split.csv'] else 'test_split.csv'}")
LABELS_CSV = Path(r"{csv_files['Detailed_PHQ8_Labels.csv'] if csv_files['Detailed_PHQ8_Labels.csv'] else 'Detailed_PHQ8_Labels.csv'}")
"""
    
    with open("config_fix.txt", "w") as f:
        f.write(config_suggestion)
    
    print(f"\n✓ Saved config suggestions to: config_fix.txt")

else:
    print("\n✗ Missing required paths!")
    print("\nPlease ensure:")
    print("  1. CSV files (train_split.csv, dev_split.csv, test_split.csv) exist")
    print("  2. Audio directory with .wav files exists")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("\n1. Update config.py with the paths shown above")
print("2. Run: python phase2_preprocess.py  (to create clean audio)")
print("3. Then run: python phase4_whisper_extract.py")
print("\n" + "="*70 + "\n")
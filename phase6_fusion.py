"""
phase6_fusion.py  +  phase7_classifier.py  (combined)
======================================================
PHASE 6 — Learnable Layer Fusion
PHASE 7 — Full Depression Classifier + Training

Architecture:
  LAdapterBank (6 adapters, whisper-base encoder-only)
       |  6 x (B, ADAPTER_OUT_DIM)
  LayerFusion  (softmax-weighted sum over 6 layers)
       |  (B, ADAPTER_OUT_DIM)
  FC -> ReLU -> Dropout -> FC
       |  (B, 2)  logits

CPU speed optimisations applied:
  - torch.set_num_threads uses all available CPU cores
  - num_workers=0 avoids multiprocessing overhead on Windows/CPU
  - pin_memory=False (no GPU)
  - GRU runs in float32 (optimal for CPU)
  - All debug/per-sample prints removed
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report)

import config
from phase5_adapters import LAdapterBank, load_whisper_npz

# Use all available CPU cores for torch operations
torch.set_num_threads(os.cpu_count())


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Layer Fusion
# ══════════════════════════════════════════════════════════════════════════════

class LayerFusion(nn.Module):
    """Learnable softmax-weighted sum over N encoder-layer embeddings."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, embeddings: list) -> torch.Tensor:
        stacked = torch.stack(embeddings, dim=1)                        # (B, N, D)
        w       = torch.softmax(self.weights, dim=0)                    # (N,)
        return  (stacked * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)    # (B, D)

    def get_weights(self) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.weights, dim=0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Classifier Head
# ══════════════════════════════════════════════════════════════════════════════

class DepressionClassifier(nn.Module):
    """FC -> ReLU -> Dropout -> FC -> logits"""

    def __init__(self,
                 in_dim:      int   = config.ADAPTER_OUT_DIM,
                 hidden_dim:  int   = 128,
                 num_classes: int   = config.NUM_CLASSES,
                 dropout:     float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Full Model
# ══════════════════════════════════════════════════════════════════════════════

class DepressionDetectionModel(nn.Module):
    """LAdapterBank -> LayerFusion -> DepressionClassifier"""

    def __init__(self, whisper_model: str = config.WHISPER_MODEL):
        super().__init__()
        self.adapter_bank = LAdapterBank(whisper_model=whisper_model)
        self.fusion       = LayerFusion(num_layers=self.adapter_bank.num_layers)
        self.classifier   = DepressionClassifier()

    def forward(self, hidden_states: dict) -> torch.Tensor:
        return self.classifier(self.fusion(self.adapter_bank(hidden_states)))

    def get_layer_weights(self) -> np.ndarray:
        return self.fusion.get_weights()

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class WhisperFeatureDataset(Dataset):
    """
    Loads encoder hidden states from pre-extracted NPZ files.
    Clamps all files to exactly num_enc encoder layers for consistency.
    """

    def __init__(self, split_csv: Path, label_col: str = "PHQ_Binary",
                 device: str = "cpu"):
        df = pd.read_csv(split_csv)
        df["Participant_ID"] = df["Participant_ID"].astype(str).str.strip()
        self.device  = device
        self.num_enc = LAdapterBank.WHISPER_ENC_LAYERS.get(config.WHISPER_MODEL, 6)

        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            pid = row["Participant_ID"]
            npz = config.FEATURE_DIR / f"{pid}_whisper.npz"
            if npz.exists():
                self.samples.append((pid, int(row[label_col])))
            else:
                skipped += 1

        if skipped:
            print(f"  [Dataset] Skipped {skipped} participants with missing NPZ files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pid, label = self.samples[idx]
        hidden = load_whisper_npz(pid, device=self.device, max_enc_layers=self.num_enc)

        # Enforce layer cap and remove leading batch dim
        hidden = {
            k: v.squeeze(0)
            for k, v in hidden.items()
            if k.startswith("encoder_layer_")
            and int(k.split("_")[-1]) < self.num_enc
        }

        return hidden, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Pad variable-length encoder sequences within a batch."""
    hidden_list, labels = zip(*batch)
    keys     = sorted(hidden_list[0].keys())   # always consistent order
    collated = {}

    for key in keys:
        tensors = [h[key] for h in hidden_list]
        max_T   = max(t.size(0) for t in tensors)
        D       = tensors[0].size(-1)
        padded  = torch.zeros(len(tensors), max_T, D)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        collated[key] = padded

    return collated, torch.stack(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(split_csv: Path,
                           label_col: str = "PHQ_Binary") -> torch.Tensor:
    df      = pd.read_csv(split_csv)
    counts  = df[label_col].value_counts().sort_index()
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for hidden, labels in loader:
        hidden = {k: v.to(device) for k, v in hidden.items()}
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
        logits = model(hidden)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return total_loss / max(len(loader), 1), acc, f1


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for hidden, labels in loader:
        hidden  = {k: v.to(device) for k, v in hidden.items()}
        labels  = labels.to(device)
        logits  = model(hidden)
        loss    = criterion(logits, labels)
        total_loss += loss.item()
        probs   = torch.softmax(logits, dim=-1)[:, 1]
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return total_loss / max(len(loader), 1), acc, f1, auc, all_preds, all_labels, all_probs


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_training():
    print(f"\n{'='*60}")
    print("PHASE 7 — Training: Adapters + Fusion + Classifier")
    print(f"{'='*60}")

    torch.manual_seed(config.SEED)
    device = config.WHISPER_DEVICE

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = WhisperFeatureDataset(config.TRAIN_CSV, device="cpu")
    dev_ds   = WhisperFeatureDataset(config.DEV_CSV,   device="cpu")
    test_ds  = WhisperFeatureDataset(config.TEST_CSV,  device="cpu")

    if len(train_ds) == 0:
        print("[ERROR] No training samples found. Run phase4 first.")
        return {}

    # num_workers=0 is fastest on CPU-only Windows — avoids process spawn overhead
    loader_kwargs = dict(
        batch_size  = config.BATCH_SIZE,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    dev_loader   = DataLoader(dev_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"  Train={len(train_ds)}  Dev={len(dev_ds)}  Test={len(test_ds)}")
    print(f"  CPU threads: {torch.get_num_threads()}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DepressionDetectionModel(whisper_model=config.WHISPER_MODEL)
    model.to(device)
    print(f"  Trainable parameters: {model.count_trainable_params():,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw        = compute_class_weights(config.TRAIN_CSV).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── Optimizer + LR scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dev_f1 = -1.0
    best_ckpt   = config.CHECKPOINT_DIR / "best_model.pt"
    history     = {"train_loss": [], "dev_loss": [], "dev_f1": [], "dev_auc": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        tr_loss, _, tr_f1      = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dv_loss, _, dv_f1, dv_auc, _, _, _ = evaluate_model(model, dev_loader, criterion, device)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(dv_f1)
        curr_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(tr_loss)
        history["dev_loss"].append(dv_loss)
        history["dev_f1"].append(dv_f1)
        history["dev_auc"].append(dv_auc)

        lr_tag = f"  [LR {prev_lr:.1e}->{curr_lr:.1e}]" if curr_lr < prev_lr else ""
        saved  = ""
        if dv_f1 > best_dev_f1:
            best_dev_f1 = dv_f1
            torch.save(model.state_dict(), best_ckpt)
            saved = "  [saved]"

        print(f"  Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"Train L={tr_loss:.4f} F1={tr_f1:.3f} | "
              f"Dev L={dv_loss:.4f} F1={dv_f1:.3f} AUC={dv_auc:.3f}"
              f"{lr_tag}{saved}")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n[Test] Loading best checkpoint ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, te_acc, te_f1, te_auc, te_preds, te_labels, _ = evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"\n  Test  Acc={te_acc:.3f}  F1={te_f1:.3f}  AUC={te_auc:.3f}")
    print(classification_report(te_labels, te_preds,
                                target_names=["Not Depressed", "Depressed"],
                                zero_division=0))

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {"accuracy": te_acc, "f1": te_f1, "auc": te_auc}
    with open(config.RESULTS_DIR / "deep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(config.RESULTS_DIR / "training_history.csv", index=False)

    weights = model.get_layer_weights()
    np.save(config.RESULTS_DIR / "fusion_weights.npy", weights)

    print(f"\nResults saved to {config.RESULTS_DIR}")
    print(f"Fusion weights (layer importance): {np.round(weights, 4)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Phase 6: Layer Fusion sanity check ===")
    fusion = LayerFusion(num_layers=6)
    out    = fusion([torch.randn(3, config.ADAPTER_OUT_DIM) for _ in range(6)])
    print(f"  Output shape: {out.shape}  Weights: {fusion.get_weights()}")

    print("\n=== Phase 7: Full model ===")
    model = DepressionDetectionModel(whisper_model=config.WHISPER_MODEL)
    print(f"  Trainable params: {model.count_trainable_params():,}")

    run_training()
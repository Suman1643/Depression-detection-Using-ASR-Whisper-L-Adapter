"""
phase5_adapters.py
==================
PHASE 5 — L-Adapter Module

Architecture per layer:
    input  : (batch, T, D_whisper)   — variable-length sequence
    BiGRU  : (batch, T, 2*hidden)    — bidirectional context
    Pool   : (batch, 2*hidden)       — mean-pool over time
    Linear : (batch, adapter_out)    — project to shared embedding dim
    ReLU + LayerNorm

One adapter per encoder layer. whisper-base = 6 encoder layers = 6 adapters.

NPZ files may have 6 or 12 encoder layers (extraction inconsistency).
The loader clamps all files to exactly num_enc layers for the configured model.
Decoder states are always ignored (variable count per participant).
"""

import os
import torch
import torch.nn as nn
import numpy as np

import config


# ─────────────────────────────────────────────────────────────────────────────
# Single L-Adapter
# ─────────────────────────────────────────────────────────────────────────────

class LAdapter(nn.Module):

    def __init__(self,
                 input_dim  = 512,
                 hidden_dim = config.ADAPTER_HIDDEN,
                 output_dim = config.ADAPTER_OUT_DIM,
                 num_layers = config.BIGRU_LAYERS,
                 dropout    = config.DROPOUT):
        super().__init__()

        self.bigru = nn.GRU(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        self.relu       = nn.ReLU()
        self.norm       = nn.LayerNorm(output_dim)

    def forward(self, x, lengths=None):
        if lengths is not None and lengths.numel() > 1:
            x_packed       = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, _     = nn.utils.rnn.pad_packed_sequence(
                self.bigru(x_packed)[0], batch_first=True
            )
        else:
            gru_out, _ = self.bigru(x)

        # Masked mean pooling
        if lengths is not None:
            mask   = (torch.arange(gru_out.size(1), device=x.device)
                      .unsqueeze(0) < lengths.unsqueeze(1).to(x.device))
            mask   = mask.unsqueeze(-1).float()
            pooled = (gru_out * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = gru_out.mean(dim=1)

        return self.norm(self.relu(self.projection(pooled)))


# ─────────────────────────────────────────────────────────────────────────────
# L-Adapter Bank
# ─────────────────────────────────────────────────────────────────────────────

class LAdapterBank(nn.Module):

    WHISPER_DIMS = {
        "tiny": 384, "base": 512, "small": 768, "medium": 1024, "large": 1280,
    }

    WHISPER_ENC_LAYERS = {
        "tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32,
    }

    def __init__(self, whisper_model=config.WHISPER_MODEL):
        super().__init__()

        self.whisper_model = whisper_model
        d_model            = self.WHISPER_DIMS.get(whisper_model, 512)
        self.num_enc       = self.WHISPER_ENC_LAYERS.get(whisper_model, 6)

        self.adapters = nn.ModuleList(
            [LAdapter(input_dim=d_model) for _ in range(self.num_enc)]
        )

    @property
    def num_layers(self) -> int:
        return self.num_enc

    def forward(self, hidden_states: dict) -> list:
        embeddings = []
        for i in range(self.num_enc):
            key = f"encoder_layer_{i}"
            if key not in hidden_states:
                available = sorted(k for k in hidden_states if k.startswith("encoder_layer_"))
                raise KeyError(
                    f"Expected '{key}' not found. Available: {available}. "
                    f"Check load_whisper_npz() clamping."
                )
            embeddings.append(self.adapters[i](hidden_states[key]))
        return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# NPZ loader  (encoder-only, clamped to model's layer count)
# ─────────────────────────────────────────────────────────────────────────────

def load_whisper_npz(pid, device="cpu", max_enc_layers=None):
    """
    Load encoder hidden states from {FEATURE_DIR}/{pid}_whisper.npz.
    Clamps to max_enc_layers (default: model count from config).
    Returns dict of tensors or None if file missing.
    """
    if max_enc_layers is None:
        max_enc_layers = LAdapterBank.WHISPER_ENC_LAYERS.get(config.WHISPER_MODEL, 6)

    path = config.FEATURE_DIR / f"{pid}_whisper.npz"
    if not path.exists():
        return None

    data = np.load(str(path))
    out  = {}
    for k, v in data.items():
        if not k.startswith("encoder_layer_"):
            continue
        try:
            idx = int(k.split("_")[-1])
        except ValueError:
            continue
        if idx >= max_enc_layers:
            continue
        out[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)

    return out if out else None


# ─────────────────────────────────────────────────────────────────────────────
# Sanity test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bank   = LAdapterBank("base")
    params = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    print(f"LAdapterBank — whisper-base | adapters=6 | params={params:,}")

    dummy = {f"encoder_layer_{i}": torch.randn(2, 1500, 512) for i in range(6)}
    embs  = bank(dummy)
    print(f"Output: {len(embs)} embeddings, each {embs[0].shape}")
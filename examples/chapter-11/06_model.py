"""
06_model.py — Chapter 11: TinyTransformerEncoder Architecture

Implements a minimal transformer encoder for text classification.
This script defines the model architecture, prints a summary, and reports
the approximate parameter count (~40K parameters).

Architecture:
  1. Character embedding layer (vocab_size -> d_model)
  2. Sinusoidal positional encoding (added to embeddings, not learned)
  3. Single transformer encoder block:
     a. Multi-head self-attention (n_heads parallel attention operations)
     b. Residual connection + LayerNorm
     c. Feed-forward network (d_model -> d_ff -> d_model, ReLU activation)
     d. Residual connection + LayerNorm
  4. Mean pooling over non-padding token positions
  5. Classification head (d_model -> n_classes)

This script uses PyTorch if available; falls back to a NumPy reference
implementation if PyTorch is not installed.

Run with Python 3.9+. PyTorch optional.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Dataset constants (reproduced for standalone use)
# ---------------------------------------------------------------------------

PAD_IDX   = 0
VOCAB_SIZE = 42   # approximate for this dataset (2 special + ~40 characters)
N_CLASSES  = 6
D_MODEL    = 64
N_HEADS    = 4
D_FF       = 128
MAX_LEN    = 64
DROPOUT    = 0.1


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (shared utility)
# ---------------------------------------------------------------------------

def sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding matrix, shape (max_len, d_model)."""
    assert d_model % 2 == 0
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    positions = np.arange(max_len)[:, None]
    half_dims = np.arange(0, d_model, 2)
    div_term  = np.exp(half_dims * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


# ---------------------------------------------------------------------------
# NumPy reference implementation (fallback when PyTorch unavailable)
# ---------------------------------------------------------------------------

class TinyTransformerEncoderNumpy:
    """
    NumPy reference implementation for architecture inspection only.
    Not differentiable; cannot be trained. Provided for environments
    where PyTorch is unavailable.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, n_classes=N_CLASSES,
                 d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 max_len=MAX_LEN):
        self.vocab_size = vocab_size
        self.n_classes  = n_classes
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_ff       = d_ff
        self.max_len    = max_len

        rng = np.random.default_rng(42)
        # Embedding
        self.embed = rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02
        self.pe    = sinusoidal_pe(max_len, d_model)

        # Multi-head attention projections
        self.W_q = [rng.standard_normal((d_model, d_model // n_heads)).astype(np.float32) * 0.02
                    for _ in range(n_heads)]
        self.W_k = [rng.standard_normal((d_model, d_model // n_heads)).astype(np.float32) * 0.02
                    for _ in range(n_heads)]
        self.W_v = [rng.standard_normal((d_model, d_model // n_heads)).astype(np.float32) * 0.02
                    for _ in range(n_heads)]
        self.W_o = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02

        # Feed-forward
        self.W_ff1 = rng.standard_normal((d_model, d_ff)).astype(np.float32) * 0.02
        self.b_ff1 = np.zeros(d_ff, dtype=np.float32)
        self.W_ff2 = rng.standard_normal((d_ff, d_model)).astype(np.float32) * 0.02
        self.b_ff2 = np.zeros(d_model, dtype=np.float32)

        # Classifier
        self.W_cls1 = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02
        self.b_cls1 = np.zeros(d_model, dtype=np.float32)
        self.W_cls2 = rng.standard_normal((d_model, n_classes)).astype(np.float32) * 0.02
        self.b_cls2 = np.zeros(n_classes, dtype=np.float32)

    def count_parameters(self):
        """Count total trainable parameters."""
        params = {}
        params["embedding"]   = self.vocab_size * self.d_model
        d_k = self.d_model // self.n_heads
        params["attn_Q"]      = self.n_heads * self.d_model * d_k
        params["attn_K"]      = self.n_heads * self.d_model * d_k
        params["attn_V"]      = self.n_heads * self.d_model * d_k
        params["attn_O"]      = self.d_model * self.d_model
        params["ff_layer1"]   = self.d_model * self.d_ff + self.d_ff
        params["ff_layer2"]   = self.d_ff * self.d_model + self.d_model
        params["layernorm1"]  = 2 * self.d_model   # scale + bias
        params["layernorm2"]  = 2 * self.d_model
        params["classifier1"] = self.d_model * self.d_model + self.d_model
        params["classifier2"] = self.d_model * self.n_classes + self.n_classes
        return params


# ---------------------------------------------------------------------------
# PyTorch implementation (preferred)
# ---------------------------------------------------------------------------

def build_pytorch_model(vocab_size=VOCAB_SIZE, n_classes=N_CLASSES,
                         d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                         max_len=MAX_LEN, dropout=DROPOUT):
    """Build the TinyTransformerEncoder using PyTorch nn.Module."""
    import torch
    import torch.nn as nn

    class TinyTransformerEncoder(nn.Module):
        """
        Minimal transformer encoder for text classification.

        Parameters
        ----------
        vocab_size : int   — number of characters in vocabulary
        n_classes  : int   — number of output categories
        d_model    : int   — embedding dimension (default 64)
        n_heads    : int   — number of attention heads (default 4)
        d_ff       : int   — feed-forward hidden dimension (default 128)
        max_len    : int   — maximum sequence length
        dropout    : float — dropout probability
        """
        def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4,
                     d_ff=128, max_len=MAX_LEN, dropout=0.1):
            super().__init__()
            self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
            pe = sinusoidal_pe(max_len, d_model)
            self.register_buffer("pe", torch.from_numpy(pe))
            self.drop    = nn.Dropout(dropout)

            # Transformer block
            self.mha  = nn.MultiheadAttention(d_model, n_heads,
                                              batch_first=True, dropout=dropout)
            self.ln1  = nn.LayerNorm(d_model)
            self.ffn  = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            self.ln2  = nn.LayerNorm(d_model)

            # Classifier
            self.clf  = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            B, T = x.shape
            h    = self.drop(self.embed(x) + self.pe[:T].unsqueeze(0))
            key_pad  = (x == PAD_IDX)
            h2, attn = self.mha(h, h, h, key_padding_mask=key_pad,
                                need_weights=True, average_attn_weights=False)
            h = self.ln1(h + h2)
            h = self.ln2(h + self.ffn(h))
            valid   = (~key_pad).float().unsqueeze(-1)
            pooled  = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
            return self.clf(pooled), attn, key_pad

    model = TinyTransformerEncoder(vocab_size, n_classes, d_model,
                                   n_heads, d_ff, max_len, dropout)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TINY TRANSFORMER ENCODER — ARCHITECTURE SUMMARY")
    print("=" * 60)

    # NumPy parameter count (always available)
    ref = TinyTransformerEncoderNumpy(
        vocab_size=VOCAB_SIZE, n_classes=N_CLASSES, d_model=D_MODEL,
        n_heads=N_HEADS, d_ff=D_FF, max_len=MAX_LEN,
    )
    param_breakdown = ref.count_parameters()
    total_params = sum(param_breakdown.values())

    print(f"\nHyperparameters:")
    print(f"  vocab_size : {VOCAB_SIZE}")
    print(f"  n_classes  : {N_CLASSES}")
    print(f"  d_model    : {D_MODEL}")
    print(f"  n_heads    : {N_HEADS}")
    print(f"  d_ff       : {D_FF}")
    print(f"  max_len    : {MAX_LEN}")
    print(f"  dropout    : {DROPOUT}")

    print(f"\nParameter breakdown:")
    print(f"  {'Component':<20} {'Parameters':<12} {'% of total'}")
    print(f"  {'-'*48}")
    for name, count in sorted(param_breakdown.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total_params
        print(f"  {name:<20} {count:<12,} {pct:5.1f}%")
    print(f"  {'TOTAL':<20} {total_params:<12,} 100.0%")

    print(f"\nNote: positional encoding has 0 parameters (not learned).")
    print(f"      LayerNorm parameters: 2 x d_model per layer (scale + bias).")

    # PyTorch version if available
    try:
        import torch
        model = build_pytorch_model(
            vocab_size=VOCAB_SIZE, n_classes=N_CLASSES, d_model=D_MODEL,
            n_heads=N_HEADS, d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT,
        )
        torch_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nPyTorch model parameter count: {torch_params:,}")
        print(f"(Reference count from breakdown: {total_params:,})")
        print(f"\nPyTorch model structure:")
        print(model)

        # Test forward pass
        test_input = torch.zeros(2, 10, dtype=torch.long)  # batch=2, seq_len=10
        logits, attn, _ = model(test_input)
        print(f"\nTest forward pass:")
        print(f"  Input shape:   {tuple(test_input.shape)}")
        print(f"  Logits shape:  {tuple(logits.shape)}  (batch, n_classes)")
        print(f"  Attn shape:    {tuple(attn.shape)}    (batch, n_heads, seq, seq)")

    except ImportError:
        print(f"\nPyTorch not installed. NumPy reference used above.")
        print(f"To install: pip install torch")
        print(f"The architecture is otherwise identical.")

    print(f"\nArchitecture summary (prose):")
    print(f"  1. Embedding:  {VOCAB_SIZE} chars -> {D_MODEL}-dim vectors")
    print(f"  2. Pos.Enc.:   sinusoidal (not learned), added to embeddings")
    print(f"  3. MHA:        {N_HEADS} heads x {D_MODEL // N_HEADS} dims = {D_MODEL} total")
    print(f"     + Residual + LayerNorm")
    print(f"  4. FFN:        {D_MODEL} -> {D_FF} (ReLU) -> {D_MODEL}")
    print(f"     + Residual + LayerNorm")
    print(f"  5. Mean pool:  average over non-padding positions")
    print(f"  6. Classifier: {D_MODEL} -> {D_MODEL} (ReLU) -> {N_CLASSES}")

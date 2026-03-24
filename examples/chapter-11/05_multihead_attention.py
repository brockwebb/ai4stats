"""
05_multihead_attention.py — Chapter 11: Multi-Head Attention (NumPy)

Demonstrates multi-head attention on a 4-token sequence using NumPy.
Multi-head attention runs h parallel attention operations ("heads"), each
with separate projection matrices, and concatenates their outputs:

  MHA(X) = Concat(head_1, ..., head_h) W_O
  head_j  = Attention(X W_j^Q, X W_j^K, X W_j^V)

Each head can learn to attend to different relationships:
  - One head might focus on adjacent tokens (local context)
  - Another might attend to the first token (global category signal)
  - A third might pick up suffix patterns ("-er", "-ion")

This script runs 2 heads on the same 4-token sequence and prints both
attention weight matrices to show that different heads attend differently.

Note: No figure files are saved. All output is text/tables.
Run with Python 3.9+. Requires only numpy.
"""

import math
import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Reuse attention utilities from 04_attention_numpy.py
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * -1e9
    weights = softmax(scores, axis=-1)
    output  = weights @ V
    return output, weights


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

def multi_head_attention(
    H: np.ndarray,
    num_heads: int = 2,
    seed: int = 0,
) -> tuple:
    """
    Multi-head attention on a single sequence (no batch dimension).

    Parameters
    ----------
    H : np.ndarray, shape (seq_len, d_model)
        Input token representations (embeddings + positional encoding).
    num_heads : int
        Number of attention heads.
    seed : int
        Random seed for reproducible projection weights.

    Returns
    -------
    concatenated : np.ndarray, shape (seq_len, d_model)
        Concatenated head outputs projected back to d_model.
    head_weights : list of np.ndarray, each (seq_len, seq_len)
        Attention weight matrices for each head.
    """
    seq_len, d_model = H.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_k = d_model // num_heads

    rng = np.random.default_rng(seed)
    head_outputs = []
    head_weights  = []

    for head_idx in range(num_heads):
        W_q = rng.standard_normal((d_model, d_k)).astype(np.float32) * 0.1
        W_k = rng.standard_normal((d_model, d_k)).astype(np.float32) * 0.1
        W_v = rng.standard_normal((d_model, d_k)).astype(np.float32) * 0.1

        Q_h = H @ W_q   # (seq_len, d_k)
        K_h = H @ W_k
        V_h = H @ W_v

        out_h, weights_h = scaled_dot_product_attention(
            Q_h[None], K_h[None], V_h[None]
        )
        head_outputs.append(out_h[0])       # (seq_len, d_k)
        head_weights.append(weights_h[0])    # (seq_len, seq_len)

    # Concatenate: (seq_len, d_model)
    concatenated = np.concatenate(head_outputs, axis=-1)

    # Output projection W_O
    W_O = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1
    output = concatenated @ W_O

    return output, head_weights


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng_main = np.random.default_rng(42)

    tokens  = ["farm", "crop", "dairy", "worker"]
    seq_len = len(tokens)
    d_model = 16
    num_heads = 2

    # Random input representations
    H = rng_main.standard_normal((seq_len, d_model)).astype(np.float32) * 0.1

    print("=" * 60)
    print("MULTI-HEAD ATTENTION (NumPy)")
    print("=" * 60)
    print(f"\nSequence:   {tokens}")
    print(f"seq_len:    {seq_len}")
    print(f"d_model:    {d_model}")
    print(f"num_heads:  {num_heads}")
    print(f"d_k per head: {d_model // num_heads}  (d_model / num_heads)")

    output, head_weights = multi_head_attention(H, num_heads=num_heads, seed=0)

    for head_idx, weights in enumerate(head_weights):
        print(f"\n--- Head {head_idx + 1} attention weight matrix ---")
        print(f"  (Random weights; not trained)")
        print(f"\n  {'Query':>8} | " + " | ".join(f"{t:>8}" for t in tokens))
        print("  " + "-" * (12 + 11 * seq_len))
        for i, qt in enumerate(tokens):
            row = " | ".join(f"{weights[i, j]:8.4f}" for j in range(seq_len))
            print(f"  {qt:>8} | {row}")

        print(f"\n  Most attended token per query (Head {head_idx + 1}):")
        for i, qt in enumerate(tokens):
            top_j = int(np.argmax(weights[i]))
            print(f"    {qt!r:>10} -> {tokens[top_j]!r:>10}  "
                  f"(weight = {weights[i, top_j]:.4f})")

    print(f"\n--- Difference between heads ---")
    w1, w2 = head_weights[0], head_weights[1]
    max_diff = np.abs(w1 - w2).max()
    mean_diff = np.abs(w1 - w2).mean()
    print(f"  Max absolute difference between head 1 and head 2: {max_diff:.4f}")
    print(f"  Mean absolute difference:                          {mean_diff:.4f}")
    print(f"  (Non-zero difference confirms heads attend differently)")

    print(f"\n--- Agreements and disagreements ---")
    head1_top = [int(np.argmax(w1[i])) for i in range(seq_len)]
    head2_top = [int(np.argmax(w2[i])) for i in range(seq_len)]
    agree = sum(1 for a, b in zip(head1_top, head2_top) if a == b)
    print(f"  Queries where both heads agree on top attended token: {agree}/{seq_len}")
    for i, qt in enumerate(tokens):
        agreement = "AGREE" if head1_top[i] == head2_top[i] else "DIFFER"
        print(f"  {qt!r:>10}: head1->{tokens[head1_top[i]]!r:>10}  "
              f"head2->{tokens[head2_top[i]]!r:>10}  [{agreement}]")

    print(f"\n--- Output shape ---")
    print(f"  Concatenated + projected output: {output.shape}")
    print(f"  Same as input shape (seq_len={seq_len}, d_model={d_model})")
    print(f"  This allows stacking multiple transformer blocks.")

    print(f"\n--- Why multiple heads? ---")
    print(f"  In a trained model, different heads specialize:")
    print(f"  - One head might learn to link 'hospital' with 'patient'")
    print(f"    (semantic co-occurrence in healthcare text)")
    print(f"  - Another head might focus on word-ending patterns")
    print(f"    ('-er', '-ist', '-or' signal occupational roles)")
    print(f"  - A third might attend to the first word in a description")
    print(f"    (often the most informative for industry classification)")
    print(f"  Multi-head attention lets the model learn all of these")
    print(f"  simultaneously within a single layer.")

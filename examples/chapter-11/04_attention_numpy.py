"""
04_attention_numpy.py — Chapter 11: Scaled Dot-Product Attention (NumPy)

Implements scaled dot-product attention from scratch using only NumPy,
following Vaswani et al. (2017):

  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

where:
  Q (queries)  — what each token is looking for
  K (keys)     — what each token offers
  V (values)   — what each token contributes if attended to
  d_k          — dimension of the key vectors (scaling factor)

This script demonstrates the mechanics on a 4-token toy sequence with
random projection weights, printing the full 4x4 attention weight matrix.
No training occurs; this is a pure mechanics demonstration.

Note: No figure files are saved. All output is text/tables.
Run with Python 3.9+. Requires only numpy.
"""

import math
import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Core attention function
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None,
) -> tuple:
    """
    Scaled dot-product attention.

    Parameters
    ----------
    Q : np.ndarray, shape (batch, seq_len, d_k)
    K : np.ndarray, shape (batch, seq_len, d_k)
    V : np.ndarray, shape (batch, seq_len, d_v)
    mask : np.ndarray, shape (batch, seq_len, seq_len), optional
        Binary mask; masked positions receive -1e9 before softmax.

    Returns
    -------
    output  : np.ndarray, shape (batch, seq_len, d_v)
    weights : np.ndarray, shape (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    # (batch, seq_len, seq_len)
    scores = Q @ K.transpose(0, 2, 1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * -1e9
    weights = softmax(scores, axis=-1)
    output  = weights @ V
    return output, weights


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # A 4-token sequence: "farm crop dairy worker"
    tokens = ["farm", "crop", "dairy", "worker"]
    seq_len = len(tokens)
    d_model = 16
    d_k     = d_model  # projection keeps same size in this demo

    print("=" * 60)
    print("SCALED DOT-PRODUCT ATTENTION (NumPy)")
    print("=" * 60)
    print(f"\nSequence: {tokens}")
    print(f"seq_len:  {seq_len}")
    print(f"d_model:  {d_model}")
    print(f"d_k:      {d_k}")
    print(f"\nFormula: Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V")

    # Toy embeddings (random for demo — not trained)
    vocab = sorted(set(w for t in tokens for w in [t]))
    embed_table = rng.standard_normal((len(vocab) + 10, d_model)).astype(np.float32) * 0.1
    H = rng.standard_normal((seq_len, d_model)).astype(np.float32) * 0.1

    # Random projection matrices
    W_q = rng.standard_normal((d_model, d_k)).astype(np.float32) * 0.1
    W_k = rng.standard_normal((d_model, d_k)).astype(np.float32) * 0.1
    W_v = rng.standard_normal((d_model, d_k)).astype(np.float32) * 0.1

    # Project to Q, K, V
    Q = H @ W_q   # (seq_len, d_k)
    K = H @ W_k
    V = H @ W_v

    print(f"\n--- Matrix shapes ---")
    print(f"  H (token representations): {H.shape}")
    print(f"  Q = H @ W_q:               {Q.shape}")
    print(f"  K = H @ W_k:               {K.shape}")
    print(f"  V = H @ W_v:               {V.shape}")

    # Compute attention (add batch dimension)
    output, weights = scaled_dot_product_attention(
        Q[None], K[None], V[None]
    )
    output  = output[0]   # (seq_len, d_k)
    weights = weights[0]  # (seq_len, seq_len)

    print(f"\n--- Raw attention scores (Q K^T / sqrt({d_k})) ---")
    raw_scores = Q @ K.T / math.sqrt(d_k)
    print(f"  Rows = queries (tokens attending)")
    print(f"  Cols = keys (tokens being attended to)")
    print(f"\n  {'Query':>8} | " + " | ".join(f"{t:>8}" for t in tokens))
    print("  " + "-" * (12 + 11 * seq_len))
    for i, qt in enumerate(tokens):
        row = " | ".join(f"{raw_scores[i, j]:+8.4f}" for j in range(seq_len))
        print(f"  {qt:>8} | {row}")

    print(f"\n--- Attention weight matrix (after softmax) ---")
    print(f"  Each row sums to 1.0.")
    print(f"  Higher weight = token in column is more attended to by token in row.")
    print(f"\n  {'Query':>8} | " + " | ".join(f"{t:>8}" for t in tokens) + " | row_sum")
    print("  " + "-" * (12 + 11 * seq_len + 12))
    for i, qt in enumerate(tokens):
        row = " | ".join(f"{weights[i, j]:8.4f}" for j in range(seq_len))
        row_sum = weights[i].sum()
        print(f"  {qt:>8} | {row} | {row_sum:.4f}")

    print(f"\n--- Most attended token for each query ---")
    for i, qt in enumerate(tokens):
        top_j = int(np.argmax(weights[i]))
        print(f"  {qt!r:>10} attends most to {tokens[top_j]!r:>10}  "
              f"(weight = {weights[i, top_j]:.4f})")

    print(f"\n--- Output shapes ---")
    print(f"  Attention output: {output.shape}  (same shape as input)")
    print(f"  Each output row is a weighted combination of all V rows.")

    # Intuition
    print(f"\n--- What this means ---")
    print(f"  - 'farm' attends to each token according to how similar")
    print(f"    its query vector is to each key vector.")
    print(f"  - The attention weights tell us: given 'farm' as context,")
    print(f"    how much should we blend information from each other token?")
    print(f"  - In a trained model, these weights would reflect learned")
    print(f"    semantic relationships (e.g., 'dairy' and 'farm' are related).")
    print(f"  - Here the weights are random because the projection matrices")
    print(f"    (W_q, W_k, W_v) have not been trained.")

"""
03_embeddings_positional.py — Chapter 11: Sinusoidal Positional Encoding

Demonstrates sinusoidal positional encoding as introduced in Vaswani et al.
(2017) "Attention is all you need."

The encoding assigns each position in a sequence a unique vector of sine and
cosine values at different frequencies. This allows the model to distinguish
position 0 from position 1, position 10 from position 11, and so on, even
though the self-attention mechanism itself has no built-in notion of order.

Formula:
  PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

This script shows:
- A small worked example (5 positions, d_model=8)
- How the pattern varies across positions and dimensions
- Why the encoding is unique for every position

Note: This is a visualization/demonstration script. No figure files are saved.
Run with Python 3.9+. Requires only numpy.
"""

import math
import numpy as np


def sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal positional encoding matrix.

    Parameters
    ----------
    max_len : int
        Maximum sequence length (number of positions).
    d_model : int
        Embedding dimension. Must be even.

    Returns
    -------
    pe : np.ndarray, shape (max_len, d_model)
        Positional encoding matrix. Row i is the encoding for position i.
    """
    assert d_model % 2 == 0, "d_model must be even"
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    positions = np.arange(max_len)[:, None]            # (max_len, 1)
    half_dims  = np.arange(0, d_model, 2)              # 0, 2, 4, ..., d_model-2
    div_term   = np.exp(half_dims * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(positions * div_term)         # even dims
    pe[:, 1::2] = np.cos(positions * div_term)         # odd dims
    return pe


if __name__ == "__main__":
    print("=" * 60)
    print("SINUSOIDAL POSITIONAL ENCODING")
    print("=" * 60)

    # Small example: 5 positions, d_model=8
    max_len_demo = 5
    d_model_demo = 8
    pe_small = sinusoidal_pe(max_len_demo, d_model_demo)

    print(f"\nSmall example: max_len={max_len_demo}, d_model={d_model_demo}")
    print(f"\nFormula:")
    print(f"  PE(pos, 2i)   = sin(pos / 10000^(2i / {d_model_demo}))")
    print(f"  PE(pos, 2i+1) = cos(pos / 10000^(2i / {d_model_demo}))")

    print(f"\nEncoding matrix ({max_len_demo} x {d_model_demo}):")
    header = "  pos | " + "  ".join(f"dim{d:02d}" for d in range(d_model_demo))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pos in range(max_len_demo):
        row_vals = "  ".join(f"{v:+.3f}" for v in pe_small[pos])
        print(f"    {pos}  | {row_vals}")

    print(f"\nKey observations:")
    print(f"  - Each row (position) is a unique vector.")
    print(f"  - Even dimensions (0, 2, 4, ...) use sine; odd use cosine.")
    print(f"  - Lower dimensions oscillate faster (higher frequency).")
    print(f"  - Higher dimensions oscillate slowly (low frequency).")
    print(f"  - Adjacent positions have similar but distinct encodings,")
    print(f"    which helps the model learn relative position.")

    # Show the pattern at larger scale
    max_len_full = 64
    d_model_full = 64
    pe_full = sinusoidal_pe(max_len_full, d_model_full)

    print(f"\n--- Pattern at full scale (max_len={max_len_full}, d_model={d_model_full}) ---")
    print(f"  Shape: {pe_full.shape}")
    print(f"  Value range: [{pe_full.min():.3f}, {pe_full.max():.3f}]")
    print(f"  (Values are always in [-1, 1] by construction of sine/cosine)")

    # Show how position 0 vs 1 vs 32 differ
    print(f"\n--- Positional encoding values for selected positions (first 8 dims) ---")
    print(f"  {'pos':<5} " + "  ".join(f"dim{d:02d}" for d in range(8)))
    print("  " + "-" * 60)
    for pos in [0, 1, 2, 10, 32, 63]:
        vals = "  ".join(f"{v:+.3f}" for v in pe_full[pos, :8])
        print(f"  {pos:<5} {vals}")

    # Cosine similarity between adjacent positions
    print(f"\n--- Cosine similarity between adjacent positions ---")
    print(f"  (Higher = more similar; adjacent positions should be close but not identical)")
    print(f"  {'pos i':<8} {'pos i+1':<10} {'cosine similarity'}")
    print(f"  " + "-" * 35)
    for pos in [0, 1, 5, 10, 30, 62]:
        v1 = pe_full[pos]
        v2 = pe_full[pos + 1]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        print(f"  {pos:<8} {pos+1:<10} {sim:.4f}")

    print(f"\nNote: similarity decreases as positions grow farther apart,")
    print(f"      giving the model a smooth sense of relative distance.")

    # How to add PE to embeddings
    print(f"\n--- How positional encoding is added to embeddings ---")
    print(f"  Given:")
    print(f"    token_embedding  shape: (sequence_length, d_model)")
    print(f"    positional_encoding shape: (sequence_length, d_model)")
    print(f"  Result:")
    print(f"    input_to_attention = token_embedding + positional_encoding")
    print(f"  The addition preserves embedding information while injecting")
    print(f"  position information into each token's representation.")

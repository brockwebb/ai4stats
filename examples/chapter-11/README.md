# Chapter 11 Examples: Transformers for Survey Text Classification

These scripts build a mini transformer encoder for industry text coding from scratch.
Together they demonstrate every major component of the transformer architecture
applied to the federal statistics problem of coding free-text survey responses.

## Requirements

- Python 3.9+
- numpy (required by all scripts)
- PyTorch (required by scripts 06-10; scripts 01-05 use numpy only)

Install dependencies:
```
pip install numpy torch
```

PyTorch is optional for scripts 01-05. Scripts 06-10 will raise an informative error
if PyTorch is not installed.

## Running Order

Run scripts in order. Scripts 07-10 depend on a trained model; script 07 saves
`model_state.pt` and `vocab.npz` which scripts 08-10 will load automatically.
If those files are not present, scripts 08-10 will re-train the model from scratch.

| Script | What it demonstrates | Dependencies |
|--------|---------------------|--------------|
| `01_dataset.py` | Dataset generation, class distribution, train/test split | numpy |
| `02_tokenization.py` | Character-level vocabulary, encoding/decoding, coverage | numpy |
| `03_embeddings_positional.py` | Sinusoidal positional encoding formula and pattern | numpy |
| `04_attention_numpy.py` | Scaled dot-product attention mechanics, 4x4 weight matrix | numpy |
| `05_multihead_attention.py` | Multi-head attention, two heads on same sequence | numpy |
| `06_model.py` | TinyTransformerEncoder architecture, parameter count | numpy (+ optional PyTorch) |
| `07_training.py` | 30-epoch training loop, loss/accuracy curves | PyTorch |
| `08_evaluation.py` | Classification report, confusion matrix (text output) | PyTorch |
| `09_attention_visualization.py` | Top-3 attended tokens per example, correct vs. incorrect | PyTorch |
| `10_new_descriptions.py` | Predictions on 20 unseen descriptions, autocoding rate | PyTorch |

## Dataset

All scripts reproduce the same synthetic dataset internally (no external data files needed).
The dataset contains ~150 short industry descriptions across 6 categories:
agriculture, construction, healthcare, manufacturing, retail, technology.
Each description is 3-8 words drawn from industry-specific vocabulary with a noise word.
The dataset is generated with `numpy.random.seed(42)` for reproducibility.

## Architecture Summary

```
Input text -> character tokenization -> embedding (vocab_size -> 64)
          -> + sinusoidal positional encoding
          -> multi-head self-attention (4 heads, d_k=16 each)
          -> residual + LayerNorm
          -> feed-forward network (64 -> 128 -> 64, ReLU)
          -> residual + LayerNorm
          -> mean pooling over non-padding positions
          -> classifier (64 -> 64 -> 6 classes)
```

Total parameters: approximately 40,000.

## Key output (script 10)

```
Autocoding threshold: 0.80
Total descriptions:   20
Auto-coded:           ~70-85%  (depends on trained model)
Routed to review:     ~15-30%
```

Adjust `AUTOCODE_THRESHOLD` in `10_new_descriptions.py` to explore the tradeoff
between autocoding rate and the fraction of uncertain cases sent to human review.

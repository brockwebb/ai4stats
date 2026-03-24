"""
09_attention_visualization.py — Chapter 11: Attention Weight Visualization

After training, attention weights are meaningful: the model has learned which
characters and tokens to focus on for each industry category. This script:

- Loads the trained model (or re-trains if model_state.pt not found)
- Runs inference on one example from each of the 6 categories
- Prints the top-3 attended tokens for each example
- Shows which tokens the model pays attention to for correct vs. incorrect
  predictions

All output is printed as text tables (no matplotlib required).

Run with Python 3.9+. Requires PyTorch.
"""

import math
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    ) from exc

# ---------------------------------------------------------------------------
# Dataset + model definitions (reproduced from 07_training.py)
# ---------------------------------------------------------------------------

INDUSTRIES = {
    "agriculture": ["farm", "crop", "livestock", "harvest", "grain", "dairy", "orchard",
                    "planting", "irrigation", "tractor", "cattle", "poultry"],
    "manufacturing": ["factory", "assembly", "production", "machining", "welding",
                      "fabrication", "stamping", "casting", "tooling", "inspection",
                      "conveyor", "forging"],
    "retail": ["store", "sales", "customer", "merchandise", "retail", "shop",
               "checkout", "inventory", "register", "display", "pricing", "storefront"],
    "healthcare": ["hospital", "patient", "medical", "clinical", "nursing",
                   "treatment", "pharmacy", "diagnostic", "surgical", "therapy",
                   "wellness", "physician"],
    "technology": ["software", "computer", "programming", "data", "network",
                   "system", "code", "server", "database", "cloud", "security",
                   "developer"],
    "construction": ["building", "contractor", "project", "structure", "concrete",
                     "site", "framing", "plumbing", "electrical", "roofing",
                     "foundation", "scaffold"],
}
NOISE_WORDS = ["worker", "specialist", "manager", "coordinator", "assistant",
               "supervisor", "technician", "operator", "lead", "analyst"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_LEN   = 64
SEED      = 42


def generate_descriptions(industries, n_per_industry=25, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    descriptions, labels = [], []
    label_names = sorted(industries.keys())
    label_map = {name: i for i, name in enumerate(label_names)}
    for industry, vocab in industries.items():
        for _ in range(n_per_industry):
            n_words = rng.integers(3, 9)
            core = rng.choice(vocab, size=min(n_words - 1, len(vocab)), replace=False).tolist()
            noise = rng.choice(NOISE_WORDS)
            descriptions.append(" ".join(core + [noise]))
            labels.append(label_map[industry])
    return descriptions, labels, label_map, label_names


def stratified_split(descriptions, labels, test_size=0.20, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    unique_labels = sorted(set(labels))
    train_idx, test_idx = [], []
    for lbl in unique_labels:
        idx = [i for i, y in enumerate(labels) if y == lbl]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return train_idx, test_idx


def build_vocab(texts):
    all_chars = sorted(set("".join(texts)))
    itos = [PAD_TOKEN, UNK_TOKEN] + all_chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4,
                 d_ff=128, max_len=MAX_LEN, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        pe = sinusoidal_pe(max_len, d_model)
        self.register_buffer("pe", torch.from_numpy(pe))
        self.drop  = nn.Dropout(0.0)  # eval mode: no dropout needed
        self.mha   = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln1   = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.ln2   = nn.LayerNorm(d_model)
        self.clf   = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.embed(x) + self.pe[:T].unsqueeze(0))
        key_pad = (x == self.pad_idx)
        h2, attn = self.mha(h, h, h, key_padding_mask=key_pad,
                            need_weights=True, average_attn_weights=False)
        h = self.ln1(h + h2)
        h = self.ln2(h + self.ffn(h))
        valid  = (~key_pad).float().unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.clf(pooled), attn, key_pad


# ---------------------------------------------------------------------------
# Attention analysis functions
# ---------------------------------------------------------------------------

def get_attention(model, text, stoi, unk, device, head_idx=0):
    """Return attention weights and prediction for one text string."""
    ids = [stoi.get(ch, unk) for ch in text[:MAX_LEN]]
    x = torch.tensor([ids], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        logits, attn_weights, _ = model(x)
    probs      = F.softmax(logits, -1).cpu().numpy()[0]
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])
    # attn_weights: (1, n_heads, T, T)
    weights = attn_weights[0, head_idx].cpu().numpy()[:len(ids), :len(ids)]
    return ids, weights, pred_label, confidence, probs


def top_attended_tokens(text, weights, top_k=3):
    """
    For each query position, find the top-k attended key positions.
    Returns a list of (query_char, [(attended_char, weight), ...]).
    """
    results = []
    chars = list(text[:MAX_LEN])
    for i, qc in enumerate(chars):
        if i >= weights.shape[0]:
            break
        row = weights[i, :len(chars)]
        top_idx = np.argsort(row)[::-1][:top_k]
        top_pairs = [(chars[j], float(row[j])) for j in top_idx]
        results.append((qc, top_pairs))
    return results


def mean_attention_by_word(text, weights):
    """Average attention weight received by each word (not character)."""
    chars = list(text[:MAX_LEN])
    words = text[:MAX_LEN].split()
    word_positions = []
    pos = 0
    for w in words:
        word_positions.append((w, list(range(pos, pos + len(w)))))
        pos += len(w) + 1  # +1 for space

    # Mean attention received (column-wise average over all query rows)
    col_avg = weights.mean(axis=0)[:len(chars)]
    word_attn = []
    for word, positions in word_positions:
        valid_pos = [p for p in positions if p < len(col_avg)]
        if valid_pos:
            word_attn.append((word, float(col_avg[valid_pos].mean())))
    return word_attn


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset
    descriptions, labels, label_map, label_names = generate_descriptions(
        INDUSTRIES, n_per_industry=25, rng=rng
    )
    train_idx, test_idx = stratified_split(descriptions, labels, rng=rng)
    train_texts  = [descriptions[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    itos, stoi = build_vocab(train_texts)
    PAD = stoi[PAD_TOKEN]
    UNK = stoi[UNK_TOKEN]
    N_CLASSES = len(label_names)

    # Load or train model
    save_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(save_dir, "model_state.pt")

    model = TinyTransformerEncoder(
        vocab_size=len(itos), n_classes=N_CLASSES, d_model=64, n_heads=4,
        d_ff=128, max_len=MAX_LEN, dropout=0.1, pad_idx=PAD,
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        from torch.utils.data import Dataset, DataLoader
        from functools import partial
        import torch.nn.functional as F_train

        class TextDataset(Dataset):
            def __init__(self, texts, labels, stoi, unk):
                self.texts = texts; self.labels = labels
                self.stoi = stoi; self.unk = unk
            def __len__(self): return len(self.texts)
            def __getitem__(self, idx):
                ids = [self.stoi.get(ch, self.unk) for ch in self.texts[idx][:MAX_LEN]]
                return np.array(ids, dtype=np.int64), int(self.labels[idx])

        def _collate(batch, pad_idx):
            xs, ys = zip(*batch)
            ml = max(len(x) for x in xs)
            padded = [np.pad(x, (0, ml - len(x)), constant_values=pad_idx) for x in xs]
            return torch.from_numpy(np.stack(padded)).long(), torch.tensor(ys).long()

        ds = TextDataset(train_texts, train_labels, stoi, UNK)
        loader = DataLoader(ds, 32, shuffle=True, collate_fn=partial(_collate, pad_idx=PAD))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        model.train()
        for _ in range(30):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _, _ = model(xb)
                loss = F_train.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        print("Model trained from scratch.")

    # One example per category
    example_texts = {
        "agriculture":    "farm crop dairy worker",
        "construction":   "building concrete site manager",
        "healthcare":     "hospital patient clinical analyst",
        "manufacturing":  "factory assembly production specialist",
        "retail":         "store customer sales coordinator",
        "technology":     "software computer programming developer",
    }

    print("=" * 60)
    print("ATTENTION VISUALIZATION")
    print("=" * 60)
    print(f"\nFor each example, head 0 attention weights are shown.")
    print(f"'Top-3 attended tokens' = tokens receiving highest average attention.\n")

    correct_count = 0
    for true_label, text in example_texts.items():
        true_idx = label_map[true_label]
        ids, weights, pred_idx, confidence, probs = get_attention(
            model, text, stoi, UNK, device, head_idx=0
        )
        pred_label = label_names[pred_idx]
        correct    = (pred_idx == true_idx)
        correct_count += int(correct)
        status = "CORRECT" if correct else "WRONG"

        print(f"--- {text!r} ---")
        print(f"  True label:  {true_label}")
        print(f"  Predicted:   {pred_label}  (confidence: {confidence:.3f})  [{status}]")

        # Word-level attention
        word_attn = mean_attention_by_word(text, weights)
        word_attn.sort(key=lambda x: -x[1])
        print(f"  Word attention (avg column attention, head 0):")
        for word, attn_val in word_attn:
            bar = "#" * int(attn_val * 40)
            print(f"    {word:<16} {attn_val:.4f}  {bar}")

        # Top-3 attended characters for first 3 query positions
        top_results = top_attended_tokens(text, weights, top_k=3)
        print(f"  Top-3 attended characters for first 4 query positions:")
        for i, (qc, attended) in enumerate(top_results[:4]):
            attended_str = ", ".join(
                f"{repr(kc)}({w:.3f})" for kc, w in attended
            )
            print(f"    query[{i}]='{qc}': {attended_str}")
        print()

    print(f"Summary: {correct_count}/{len(example_texts)} examples correctly classified.")

    # Additional: correct vs. incorrect attention contrast
    test_texts_list   = [descriptions[i] for i in test_idx]
    test_labels_list  = [labels[i] for i in test_idx]

    correct_examples   = []
    incorrect_examples = []
    for text, true_lbl in zip(test_texts_list, test_labels_list):
        ids, weights, pred_idx, confidence, probs = get_attention(
            model, text, stoi, UNK, device, head_idx=0
        )
        if pred_idx == true_lbl:
            correct_examples.append((text, confidence, weights))
        else:
            incorrect_examples.append((text, confidence, pred_idx, true_lbl, weights))

    print(f"\n--- Confidence contrast: correct vs. incorrect predictions ---")
    print(f"  Correct predictions:   {len(correct_examples)}")
    print(f"  Incorrect predictions: {len(incorrect_examples)}")
    if correct_examples:
        conf_correct = np.mean([c for _, c, _ in correct_examples])
        print(f"  Mean confidence (correct):   {conf_correct:.3f}")
    if incorrect_examples:
        conf_wrong = np.mean([c for _, c, _, _, _ in incorrect_examples])
        print(f"  Mean confidence (incorrect): {conf_wrong:.3f}")
        print(f"\n  Incorrect predictions:")
        for text, conf, pred_idx, true_lbl, _ in incorrect_examples[:5]:
            print(f"    {text!r}")
            print(f"      True: {label_names[true_lbl]}, "
                  f"Predicted: {label_names[pred_idx]}, "
                  f"Confidence: {conf:.3f}")

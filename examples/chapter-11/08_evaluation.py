"""
08_evaluation.py — Chapter 11: Test Set Evaluation

Evaluates the trained TinyTransformerEncoder on the held-out test set.
Produces a classification report (precision, recall, F1 per class) and a
confusion matrix printed as plain text (no matplotlib required).

Prerequisites: Run 07_training.py first to generate model_state.pt and vocab.npz.
If those files are not found, this script re-trains from scratch automatically.

Run with Python 3.9+. Requires PyTorch.
"""

import math
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    ) from exc

# ---------------------------------------------------------------------------
# Dataset and model (reproduced from 07_training.py)
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


class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi, unk):
        self.texts  = texts
        self.labels = labels
        self.stoi   = stoi
        self.unk    = unk

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        ids = [self.stoi.get(ch, self.unk) for ch in self.texts[idx][:MAX_LEN]]
        return np.array(ids, dtype=np.int64), int(self.labels[idx])


def collate_fn(batch, pad_idx):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = [np.pad(x, (0, max_len - len(x)), constant_values=pad_idx) for x in xs]
    return torch.from_numpy(np.stack(padded)).long(), torch.tensor(ys).long()


class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4,
                 d_ff=128, max_len=MAX_LEN, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        pe = sinusoidal_pe(max_len, d_model)
        self.register_buffer("pe", torch.from_numpy(pe))
        self.drop  = nn.Dropout(dropout)
        self.mha   = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln1   = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.clf = nn.Sequential(
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
# Text classification metrics (no sklearn required)
# ---------------------------------------------------------------------------

def classification_report_text(y_true, y_pred, label_names):
    """Print per-class precision, recall, F1 and overall accuracy."""
    n_classes = len(label_names)
    rows = []
    for cls in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support   = sum(1 for t in y_true if t == cls)
        rows.append((label_names[cls], precision, recall, f1, support))
    return rows


def confusion_matrix_text(y_true, y_pred, label_names):
    """Print confusion matrix as a text table."""
    n = len(label_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("EVALUATION: TinyTransformerEncoder")
    print("=" * 60)

    # Build dataset
    descriptions, labels, label_map, label_names = generate_descriptions(
        INDUSTRIES, n_per_industry=25, rng=rng
    )
    train_idx, test_idx = stratified_split(descriptions, labels, rng=rng)
    train_texts  = [descriptions[i] for i in train_idx]
    test_texts   = [descriptions[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels  = [labels[i] for i in test_idx]

    itos, stoi = build_vocab(train_texts)
    PAD = stoi[PAD_TOKEN]
    UNK = stoi[UNK_TOKEN]
    N_CLASSES = len(label_names)

    from functools import partial
    test_ds = TextDataset(test_texts, test_labels, stoi, UNK)
    _collate = partial(collate_fn, pad_idx=PAD)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=_collate)

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
        print("model_state.pt not found. Run 07_training.py first, or training now...")
        # Re-train inline
        train_ds = TextDataset(train_texts, train_labels, stoi, UNK)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=_collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        model.train()
        for epoch in range(30):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _, _ = model(xb)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        print("Training complete.")

    # Run evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits, _, _ = model(xb)
            probs = F.softmax(logits, -1).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = (all_preds == all_labels).mean()

    print(f"\nTest set: {len(all_labels)} examples")
    print(f"Overall accuracy: {accuracy:.3f}")

    # Per-class report
    print(f"\n--- Per-class classification report ---")
    report_rows = classification_report_text(all_labels.tolist(), all_preds.tolist(), label_names)
    print(f"  {'Class':<16}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}")
    print("  " + "-" * 56)
    for (name, prec, rec, f1, sup) in report_rows:
        print(f"  {name:<16}  {prec:>9.3f}  {rec:>6.3f}  {f1:>6.3f}  {sup:>7}")
    macro_p = np.mean([r[1] for r in report_rows])
    macro_r = np.mean([r[2] for r in report_rows])
    macro_f = np.mean([r[3] for r in report_rows])
    print(f"  {'macro avg':<16}  {macro_p:>9.3f}  {macro_r:>6.3f}  {macro_f:>6.3f}  "
          f"{len(all_labels):>7}")

    # Confusion matrix
    cm = confusion_matrix_text(all_labels.tolist(), all_preds.tolist(), label_names)
    short_names = [n[:7] for n in label_names]

    print(f"\n--- Confusion matrix ---")
    print(f"  Rows = true class, Columns = predicted class")
    col_header = "  " + " " * 10 + "  ".join(f"{n:>7}" for n in short_names)
    print(col_header)
    print("  " + "-" * (12 + 9 * len(label_names)))
    for i, name in enumerate(label_names):
        row = "  ".join(f"{cm[i, j]:>7}" for j in range(len(label_names)))
        correct_mark = " *" if cm[i, i] == cm[i].max() else "  "
        print(f"  {name[:10]:<10}  {row}{correct_mark}")
    print(f"\n  (* = class correctly predicted most often in that row)")

    # Per-class errors
    print(f"\n--- Most common errors per class ---")
    for i, name in enumerate(label_names):
        true_mask = (all_labels == i)
        if true_mask.sum() == 0:
            continue
        pred_for_class = all_preds[true_mask]
        errors = [(j, (pred_for_class == j).sum()) for j in range(N_CLASSES) if j != i]
        errors.sort(key=lambda x: -x[1])
        if errors and errors[0][1] > 0:
            top_err_cls = errors[0][0]
            top_err_cnt = errors[0][1]
            print(f"  {name:<16}: {top_err_cnt} misclassified as {label_names[top_err_cls]!r}")
        else:
            print(f"  {name:<16}: no errors")

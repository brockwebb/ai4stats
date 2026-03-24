"""
10_new_descriptions.py — Chapter 11: Prediction on Unseen Descriptions

Demonstrates the model's behavior on 20 unseen descriptions spanning all
6 industry categories. For each description, this script prints:
  - The description
  - The predicted industry category
  - The confidence score (max softmax probability)
  - Whether to auto-code or route to human review

Routing rule: confidence > 0.80 -> auto-code; else -> human review.

This mirrors the confidence-based human-in-the-loop workflow described in
the chapter: high-confidence predictions are processed automatically while
borderline or low-confidence cases are flagged for expert review.

The script also reports the overall autocoding rate at the default threshold
(0.80), and shows how the rate changes if you adjust the threshold.

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
# Dataset + model (reproduced from 07_training.py for standalone use)
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

# Confidence threshold for automatic coding vs. human review
AUTOCODE_THRESHOLD = 0.80


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
        self.drop  = nn.Dropout(dropout)
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


def predict(model, text, stoi, unk, device, label_names):
    """Predict industry category and confidence for one text description."""
    ids = [stoi.get(ch, unk) for ch in text[:MAX_LEN]]
    x   = torch.tensor([ids], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
    probs      = F.softmax(logits, -1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_label = label_names[pred_idx]
    return pred_label, confidence, probs


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset
    descriptions, labels, label_map, label_names = generate_descriptions(
        INDUSTRIES, n_per_industry=25, rng=rng
    )
    train_idx, _ = stratified_split(descriptions, labels, rng=rng)
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
    else:
        from torch.utils.data import Dataset, DataLoader
        from functools import partial

        class _DS(Dataset):
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

        ds = _DS(train_texts, train_labels, stoi, UNK)
        loader = DataLoader(ds, 32, shuffle=True, collate_fn=partial(_collate, pad_idx=PAD))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        model.train()
        for _ in range(30):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _, _ = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        print("Model trained from scratch (run 07_training.py to save for reuse).")

    # 20 unseen descriptions spanning all 6 categories
    new_descriptions = [
        ("farm harvest grain supervisor",        "agriculture"),
        ("crop irrigation livestock analyst",     "agriculture"),
        ("tractor dairy orchard worker",          "agriculture"),
        ("factory welding stamping operator",     "manufacturing"),
        ("assembly machining production lead",    "manufacturing"),
        ("casting forging tooling specialist",    "manufacturing"),
        ("store checkout merchandise manager",    "retail"),
        ("sales customer display coordinator",    "retail"),
        ("inventory pricing storefront worker",   "retail"),
        ("hospital treatment clinical analyst",   "healthcare"),
        ("nursing pharmacy surgical coordinator", "healthcare"),
        ("patient wellness physician assistant",  "healthcare"),
        ("software database cloud developer",     "technology"),
        ("programming network server specialist", "technology"),
        ("computer security system analyst",      "technology"),
        ("building foundation scaffold manager",  "construction"),
        ("concrete electrical roofing worker",    "construction"),
        ("contractor site structure supervisor",  "construction"),
        ("farm software customer analyst",        "ambiguous"),   # cross-category noise
        ("hospital factory sales worker",         "ambiguous"),   # cross-category noise
    ]

    print("=" * 70)
    print("PREDICTIONS ON NEW DESCRIPTIONS")
    print(f"Autocoding threshold: {AUTOCODE_THRESHOLD}")
    print("=" * 70)
    print(f"\n  {'Description':<42} {'Predicted':<16} {'Conf':>5}  {'Action'}")
    print("  " + "-" * 80)

    results = []
    for desc, true_cat in new_descriptions:
        pred_label, confidence, probs = predict(model, desc, stoi, UNK, device, label_names)
        action = "AUTO-CODE" if confidence >= AUTOCODE_THRESHOLD else "HUMAN REVIEW"
        results.append({
            "description": desc,
            "true_category": true_cat,
            "predicted": pred_label,
            "confidence": confidence,
            "action": action,
        })
        true_str = f"[true:{true_cat}]" if true_cat not in ("ambiguous",) else "[ambiguous]"
        print(f"  {desc:<42} {pred_label:<16} {confidence:>5.3f}  {action}  {true_str}")

    # Summary statistics
    n_total     = len(results)
    n_autocode  = sum(1 for r in results if r["action"] == "AUTO-CODE")
    n_review    = n_total - n_autocode
    autocode_rate = n_autocode / n_total

    print(f"\n--- Summary at threshold = {AUTOCODE_THRESHOLD} ---")
    print(f"  Total descriptions:   {n_total}")
    print(f"  Auto-coded:           {n_autocode}  ({100*autocode_rate:.0f}%)")
    print(f"  Routed to review:     {n_review}  ({100*(1-autocode_rate):.0f}%)")

    # Non-ambiguous accuracy
    non_ambiguous = [r for r in results if r["true_category"] != "ambiguous"]
    if non_ambiguous:
        correct = sum(1 for r in non_ambiguous if r["predicted"] == r["true_category"])
        acc = correct / len(non_ambiguous)
        print(f"  Accuracy (non-ambiguous): {correct}/{len(non_ambiguous)} = {acc:.3f}")

    # Threshold sensitivity
    print(f"\n--- Autocoding rate vs. threshold ---")
    print(f"  {'Threshold':>10}  {'Autocode %':>10}  {'Review count':>12}")
    print("  " + "-" * 38)
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        n_auto = sum(1 for r in results if r["confidence"] >= threshold)
        rate = n_auto / n_total
        print(f"  {threshold:>10.2f}  {100*rate:>9.0f}%  {n_total - n_auto:>12}")

    print(f"\n--- Per-category confidence breakdown ---")
    for cat in label_names:
        cat_results = [r for r in results if r["predicted"] == cat]
        if cat_results:
            avg_conf = np.mean([r["confidence"] for r in cat_results])
            print(f"  {cat:<16}: {len(cat_results):>2} predictions, "
                  f"avg confidence = {avg_conf:.3f}")

    print(f"\nNote: To experiment with the threshold, modify AUTOCODE_THRESHOLD")
    print(f"at the top of this script and re-run.")

"""
07_training.py — Chapter 11: Training the TinyTransformerEncoder

Trains the mini transformer encoder on the synthetic industry coding dataset
for 30 epochs using cross-entropy loss and the Adam optimizer. Prints loss
and accuracy every 5 epochs. Saves the final model state to a file.

Training details:
  - Optimizer: Adam (lr=3e-3, weight_decay=1e-3)
  - Loss: cross-entropy
  - Epochs: 30
  - Batch size: 32
  - Dataset: 150 synthetic descriptions, 80/20 stratified split

Requires PyTorch. If not installed, an informative error is raised.
Run with Python 3.9+.
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
        "PyTorch is required for this script. Install with: pip install torch"
    ) from exc

# ---------------------------------------------------------------------------
# Dataset definition (reproduced from 01_dataset.py)
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


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = [self.stoi.get(ch, self.unk) for ch in self.texts[idx][:MAX_LEN]]
        return np.array(ids, dtype=np.int64), int(self.labels[idx])


def collate_fn(batch, pad_idx):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = [np.pad(x, (0, max_len - len(x)), constant_values=pad_idx) for x in xs]
    return torch.from_numpy(np.stack(padded)).long(), torch.tensor(ys).long()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln2   = nn.LayerNorm(d_model)
        self.clf   = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
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
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(-1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(-1) == yb).sum().item()
            total      += len(yb)
    return total_loss / total, correct / total


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TRAINING: TinyTransformerEncoder")
    print("=" * 60)
    print(f"Device: {device}")

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
    train_ds = TextDataset(train_texts, train_labels, stoi, UNK)
    test_ds  = TextDataset(test_texts,  test_labels,  stoi, UNK)
    _collate = partial(collate_fn, pad_idx=PAD)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=_collate)

    print(f"\nDataset: {len(train_ds)} train, {len(test_ds)} test")
    print(f"Vocab size: {len(itos)}, Classes: {N_CLASSES} ({', '.join(label_names)})")

    # Build model
    model = TinyTransformerEncoder(
        vocab_size=len(itos), n_classes=N_CLASSES,
        d_model=64, n_heads=4, d_ff=128, max_len=MAX_LEN, dropout=0.1, pad_idx=PAD,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Training loop
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Test Loss':>9}  {'Test Acc':>8}")
    print("  " + "-" * 55)

    history = []
    N_EPOCHS = 30
    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader, device)
        scheduler.step()
        history.append((epoch, tr_loss, tr_acc, te_loss, te_acc))
        if epoch % 5 == 0:
            print(f"  {epoch:>4}   {tr_loss:>10.4f}  {tr_acc:>9.3f}  {te_loss:>9.4f}  {te_acc:>8.3f}")

    # Save model state
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "model_state.pt")
    vocab_path = os.path.join(save_dir, "vocab.npz")
    torch.save(model.state_dict(), save_path)
    np.savez(vocab_path, itos=itos, label_names=label_names)
    print(f"\nModel saved to: {save_path}")
    print(f"Vocabulary saved to: {vocab_path}")

    final_tr = history[-1]
    final_te = history[-1]
    print(f"\nFinal epoch {N_EPOCHS}:")
    print(f"  Train loss={final_tr[1]:.4f}, acc={final_tr[2]:.3f}")
    print(f"  Test  loss={final_te[3]:.4f}, acc={final_te[4]:.3f}")

"""
02_tokenization.py — Chapter 11: Character-Level Tokenization

Demonstrates how to build a character-level vocabulary from training data
and use it to convert text descriptions into integer token sequences for
input to a neural network.

Key concepts:
- Why tokenization is necessary (networks require numeric inputs)
- Building vocabulary only from training data (avoiding leakage)
- Encoding (text -> integer IDs) and decoding (IDs -> text)
- Handling unknown characters with <unk> token
- Character frequency distribution

Run with Python 3.9+. No external dependencies.
"""

import math
from collections import Counter

# ---------------------------------------------------------------------------
# Re-create dataset (copied from 01_dataset.py for standalone use)
# ---------------------------------------------------------------------------

import numpy as np

np.random.seed(42)

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
MAX_LEN = 64


def generate_descriptions(industries, n_per_industry=25, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
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
        rng = np.random.default_rng(42)
    unique_labels = sorted(set(labels))
    train_idx, test_idx = [], []
    for lbl in unique_labels:
        idx = [i for i, y in enumerate(labels) if y == lbl]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_vocab(texts):
    """Build character vocabulary from a list of texts."""
    all_chars = sorted(set("".join(texts)))
    itos = [PAD_TOKEN, UNK_TOKEN] + all_chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def encode(text, stoi, max_len=None):
    """Convert text to a list of integer token IDs."""
    UNK = stoi[UNK_TOKEN]
    ids = [stoi.get(ch, UNK) for ch in text]
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def decode(ids, itos):
    """Convert integer token IDs back to text (skipping special tokens)."""
    special = {PAD_TOKEN, UNK_TOKEN}
    return "".join(itos[i] for i in ids if itos[i] not in special)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    descriptions, labels, label_map, label_names = generate_descriptions(
        INDUSTRIES, n_per_industry=25, rng=rng
    )

    train_idx, test_idx = stratified_split(descriptions, labels, rng=rng)
    train_texts = [descriptions[i] for i in train_idx]
    test_texts  = [descriptions[i] for i in test_idx]

    # Build vocab from TRAINING set only (no leakage from test)
    itos, stoi = build_vocab(train_texts)
    PAD = stoi[PAD_TOKEN]
    UNK = stoi[UNK_TOKEN]

    print("=" * 60)
    print("CHARACTER-LEVEL TOKENIZATION")
    print("=" * 60)
    print(f"\nVocabulary size:  {len(itos)} tokens (including <pad> and <unk>)")
    print(f"Special tokens:   index 0 = {itos[0]!r}, index 1 = {itos[1]!r}")
    print(f"Character set:    {''.join(itos[2:])}")

    # Example encoding / decoding
    print("\n--- Encoding examples ---")
    examples = [
        "farm crop worker",
        "hospital patient analyst",
        "software developer",
        "unknown!@#chars",   # contains characters not in vocab
    ]
    for text in examples:
        ids = encode(text, stoi, max_len=MAX_LEN)
        recovered = decode(ids, itos)
        print(f"\n  Input:    {text!r}")
        print(f"  IDs:      {ids}")
        print(f"  Decoded:  {recovered!r}")
        n_unk = sum(1 for i in ids if i == UNK)
        if n_unk:
            print(f"  Note:     {n_unk} character(s) mapped to <unk>")

    # Padding demonstration
    print("\n--- Padding to fixed length ---")
    short = "farm worker"
    ids = encode(short, stoi)
    padded = ids + [PAD] * (16 - len(ids))
    print(f"  Original ({len(ids)} chars):  {ids}")
    print(f"  Padded to 16:           {padded}")
    print(f"  Decoded (no pad):       {decode(padded, itos)!r}")

    # Character frequency distribution
    print("\n--- Character frequency (top 20, training set) ---")
    char_counter = Counter("".join(train_texts))
    total_chars = sum(char_counter.values())
    print(f"  Total characters in training set: {total_chars}")
    print(f"\n  {'Char':<8} {'Count':<8} {'%'}")
    print(f"  {'-'*28}")
    for ch, count in char_counter.most_common(20):
        pct = 100.0 * count / total_chars
        bar = "#" * int(pct * 2)
        print(f"  {ch!r:<8} {count:<8} {pct:4.1f}%  {bar}")

    # Coverage check: any test characters not in training vocab?
    test_chars = set("".join(test_texts))
    train_chars = set("".join(train_texts))
    unseen = test_chars - train_chars
    print(f"\n--- Vocabulary coverage ---")
    print(f"  Characters in train vocab: {len(train_chars)}")
    print(f"  Characters in test set:    {len(test_chars)}")
    if unseen:
        print(f"  Unseen in test:            {sorted(unseen)} -> will map to <unk>")
    else:
        print(f"  All test characters covered by training vocabulary.")

    print("\n--- Token ID table (first 10 tokens) ---")
    print(f"  {'ID':<5} {'Token'}")
    print(f"  {'-'*20}")
    for i, tok in enumerate(itos[:10]):
        print(f"  {i:<5} {tok!r}")
    print(f"  ... ({len(itos) - 10} more)")

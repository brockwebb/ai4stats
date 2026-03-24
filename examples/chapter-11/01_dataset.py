"""
01_dataset.py — Chapter 11: Synthetic Industry Coding Dataset

Demonstrates how to construct a small, reproducible dataset for exploring
transformer-based text classification. The dataset simulates the kind of
free-text industry descriptions that federal survey respondents provide,
with known ground-truth NAICS-style category labels.

This script:
- Generates ~150 descriptions across 6 industry categories
- Shows class distribution
- Prints example descriptions
- Performs a stratified 80/20 train/test split
- Reports vocabulary statistics

Standalone: no external data files required. Run with Python 3.9+.
"""

import numpy as np
from collections import Counter

np.random.seed(42)

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------

INDUSTRIES = {
    "agriculture": [
        "farm", "crop", "livestock", "harvest", "grain", "dairy", "orchard",
        "planting", "irrigation", "tractor", "cattle", "poultry",
    ],
    "manufacturing": [
        "factory", "assembly", "production", "machining", "welding",
        "fabrication", "stamping", "casting", "tooling", "inspection",
        "conveyor", "forging",
    ],
    "retail": [
        "store", "sales", "customer", "merchandise", "retail", "shop",
        "checkout", "inventory", "register", "display", "pricing",
        "storefront",
    ],
    "healthcare": [
        "hospital", "patient", "medical", "clinical", "nursing",
        "treatment", "pharmacy", "diagnostic", "surgical", "therapy",
        "wellness", "physician",
    ],
    "technology": [
        "software", "computer", "programming", "data", "network",
        "system", "code", "server", "database", "cloud", "security",
        "developer",
    ],
    "construction": [
        "building", "contractor", "project", "structure", "concrete",
        "site", "framing", "plumbing", "electrical", "roofing",
        "foundation", "scaffold",
    ],
}

NOISE_WORDS = ["worker", "specialist", "manager", "coordinator", "assistant",
               "supervisor", "technician", "operator", "lead", "analyst"]


def generate_descriptions(industries, n_per_industry=25, rng=None):
    """Generate synthetic occupation/industry descriptions."""
    if rng is None:
        rng = np.random.default_rng(42)

    descriptions = []
    labels = []
    label_names = sorted(industries.keys())
    label_map = {name: i for i, name in enumerate(label_names)}

    for industry, vocab in industries.items():
        for _ in range(n_per_industry):
            n_words = rng.integers(3, 9)
            core_words = rng.choice(vocab, size=min(n_words - 1, len(vocab)),
                                    replace=False).tolist()
            # Add one noise word for realism
            noise = rng.choice(NOISE_WORDS)
            desc = " ".join(core_words + [noise])
            descriptions.append(desc)
            labels.append(label_map[industry])

    return descriptions, labels, label_map, label_names


def stratified_split(descriptions, labels, test_size=0.20, rng=None):
    """Stratified train/test split without sklearn."""
    if rng is None:
        rng = np.random.default_rng(42)

    unique_labels = sorted(set(labels))
    train_idx = []
    test_idx = []

    for lbl in unique_labels:
        idx = [i for i, y in enumerate(labels) if y == lbl]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    return train_idx, test_idx


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    descriptions, labels, label_map, label_names = generate_descriptions(
        INDUSTRIES, n_per_industry=25, rng=rng
    )

    print("=" * 60)
    print("SYNTHETIC INDUSTRY CODING DATASET")
    print("=" * 60)
    print(f"\nTotal examples: {len(descriptions)}")
    print(f"Categories:     {len(INDUSTRIES)}")
    print(f"Label map:      {label_map}")

    # Class distribution
    print("\n--- Class distribution ---")
    counter = Counter(labels)
    for lbl_name in label_names:
        lbl_id = label_map[lbl_name]
        print(f"  {lbl_name:<16} (label {lbl_id}): {counter[lbl_id]} examples")

    # Example descriptions
    print("\n--- Sample descriptions (first 3 per category) ---")
    seen = {lbl: 0 for lbl in label_map.values()}
    for desc, lbl in zip(descriptions, labels):
        if seen[lbl] < 3:
            lbl_name = label_names[lbl]
            print(f"  [{lbl_name:<14}] {desc}")
            seen[lbl] += 1
        if all(v >= 3 for v in seen.values()):
            break

    # Train/test split
    train_idx, test_idx = stratified_split(descriptions, labels, test_size=0.20, rng=rng)
    train_labels = [labels[i] for i in train_idx]
    test_labels  = [labels[i] for i in test_idx]

    print(f"\n--- Train/test split (80/20 stratified) ---")
    print(f"  Training examples: {len(train_idx)}")
    print(f"  Test examples:     {len(test_idx)}")

    train_counter = Counter(train_labels)
    test_counter  = Counter(test_labels)
    print(f"\n  {'Category':<16} Train  Test")
    print(f"  {'-'*32}")
    for lbl_name in label_names:
        lbl_id = label_map[lbl_name]
        print(f"  {lbl_name:<16} {train_counter[lbl_id]:<6} {test_counter[lbl_id]}")

    # Vocabulary stats
    all_chars = sorted(set("".join(descriptions)))
    all_words  = []
    for desc in descriptions:
        all_words.extend(desc.split())
    word_counter = Counter(all_words)

    print(f"\n--- Vocabulary statistics ---")
    print(f"  Unique characters:  {len(all_chars)}")
    print(f"  Character set:      {''.join(all_chars)}")
    print(f"  Total word tokens:  {sum(word_counter.values())}")
    print(f"  Unique word types:  {len(word_counter)}")
    print(f"  Most common words:  {word_counter.most_common(10)}")
    print(f"\n  Avg description length: "
          f"{np.mean([len(d) for d in descriptions]):.1f} characters")
    print(f"  Max description length: {max(len(d) for d in descriptions)} characters")
    print(f"  Min description length: {min(len(d) for d in descriptions)} characters")

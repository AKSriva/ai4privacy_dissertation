# scripts/build_splits_200k_multilingual.py
# Builds multilingual (EN+FR+DE+IT) stratified splits for pii-masking-200k
# Outputs:
#   data/processed/splits_200k/{train,val,test}.parquet
#   data/processed/splits_200k/meta.json
#   data/processed/splits_200k/label2id.json

import json
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE = Path("data/raw/ai4privacy_pii/pii-masking-200k")
OUT = Path("data/processed/splits_200k")

FILES = [
    ("en", "english_pii_43k.jsonl"),
    ("fr", "french_pii_62k.jsonl"),
    ("de", "german_pii_52k.jsonl"),
    ("it", "italian_pii_50k.jsonl"),
]

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.90, 0.05, 0.05


def stream_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_all_rows():
    rows = []
    for lang, fname in FILES:
        path = BASE / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        for r in stream_jsonl(path):
            # Minimal fields required for token classification training
            rows.append(
                {
                    "tokenised_text": r["tokenised_text"],
                    "bio_labels": r["bio_labels"],
                    "language": lang,
                    # keep these for future analyses if needed (commented to reduce parquet size)
                    # "masked_text": r.get("masked_text", None),
                    # "unmasked_text": r.get("unmasked_text", None),
                    # "span_labels": r.get("span_labels", None),
                    # "privacy_mask": r.get("privacy_mask", None),
                }
            )
    return rows


def build_label2id_union(rows):
    labels = set()
    for r in rows:
        labels.update(r["bio_labels"])
    ordered = ["O"] + sorted([x for x in labels if x != "O"])
    return {lab: int(i) for i, lab in enumerate(ordered)}  # ensure python int


def split_stratified_by_language(rows):
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9, "Fractions must sum to 1.0"
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["language"]].append(r)

    rng = random.Random(SEED)
    train, val, test = [], [], []

    for lang, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        # remainder goes to test
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # mix languages in each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def counts_as_int(split_rows):
    vc = pd.Series([r["language"] for r in split_rows]).value_counts()
    return {str(k): int(v) for k, v in vc.items()}  # cast numpy int -> python int


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading all rows...")
    rows = load_all_rows()
    print(f"  Total rows loaded: {len(rows):,}")

    print("[2/4] Building label2id (union across languages)...")
    label2id = build_label2id_union(rows)
    print(f"  num_labels: {len(label2id)}")

    print("[3/4] Stratified split by language...")
    train, val, test = split_stratified_by_language(rows)
    print(f"  train/val/test sizes: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"  train by lang: {counts_as_int(train)}")
    print(f"  val   by lang: {counts_as_int(val)}")
    print(f"  test  by lang: {counts_as_int(test)}")

    print("[4/4] Writing parquet + metadata...")
    pd.DataFrame(train).to_parquet(OUT / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(OUT / "val.parquet", index=False)
    pd.DataFrame(test).to_parquet(OUT / "test.parquet", index=False)

    meta = {
        "seed": int(SEED),
        "fractions": {"train": float(TRAIN_FRAC), "val": float(VAL_FRAC), "test": float(TEST_FRAC)},
        "sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "by_language": {
            "train": counts_as_int(train),
            "val": counts_as_int(val),
            "test": counts_as_int(test),
        },
        "num_labels": int(len(label2id)),
        "source_files": [{"language": lang, "file": fname} for lang, fname in FILES],
    }

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote splits to: {OUT}")
    print("      - train.parquet / val.parquet / test.parquet")
    print("      - meta.json / label2id.json")


if __name__ == "__main__":
    main()

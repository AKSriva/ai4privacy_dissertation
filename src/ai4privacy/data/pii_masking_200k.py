import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Any
import random
from collections import defaultdict


FILES_200K = [
    ("en", "english_pii_43k.jsonl"),
    ("fr", "french_pii_62k.jsonl"),
    ("de", "german_pii_52k.jsonl"),
    ("it", "italian_pii_50k.jsonl"),
]


def stream_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_multilingual_records(base_dir: Path) -> List[dict]:
    """
    Loads all 4 language files and adds 'language' field.
    Does NOT tokenize; uses provided 'tokenised_text' + 'bio_labels'.
    """
    records: List[dict] = []
    for lang, fname in FILES_200K:
        p = base_dir / fname
        for r in stream_jsonl(p):
            r["language"] = lang
            records.append(r)
    return records


def build_label2id_union(records: List[dict]) -> Dict[str, int]:
    """
    Build label2id from union of all BIO labels across all languages.
    Ensures 'O' is 0 for convenience + consistency (if you want).
    """
    labels = set()
    for r in records:
        for lab in r["bio_labels"]:
            labels.add(lab)

    # Stable ordering: O first, then others sorted
    ordered = ["O"] + sorted([l for l in labels if l != "O"])
    return {lab: i for i, lab in enumerate(ordered)}


def stratified_split_by_language(
    records: List[dict],
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Stratifies by language: each language is split separately using same fractions.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "Fractions must sum to 1"

    buckets = defaultdict(list)
    for r in records:
        buckets[r["language"]].append(r)

    rng = random.Random(seed)

    train, val, test = [], [], []

    for lang, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # remainder goes to test
        n_test = n - n_train - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Shuffle final sets to mix languages (optional but useful)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def save_split_manifest(run_dir: Path, train: List[dict], val: List[dict], test: List[dict]) -> None:
    """
    Save lightweight split info for reproducibility (no full texts).
    We store counts + per-language counts; do not store PII content.
    """
    def counts(split):
        c = defaultdict(int)
        for r in split:
            c[r["language"]] += 1
        return dict(c)

    manifest = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_by_language": counts(train),
        "val_by_language": counts(val),
        "test_by_language": counts(test),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

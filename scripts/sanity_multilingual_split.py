from pathlib import Path
from transformers import AutoTokenizer
from src.data.pii_masking_200k import (
    load_multilingual_records,
    build_label2id_union,
    stratified_split_by_language,
)

def main():
    base = Path("data/raw/ai4privacy_pii/pii-masking-200k")
    records = load_multilingual_records(base)
    print("Total records:", len(records))

    label2id = build_label2id_union(records)
    print("Total labels:", len(label2id))
    print("First 10 labels:", list(label2id.keys())[:10])

    train, val, test = stratified_split_by_language(records, seed=42)
    print("Split sizes:", len(train), len(val), len(test))

if __name__ == "__main__":
    main()

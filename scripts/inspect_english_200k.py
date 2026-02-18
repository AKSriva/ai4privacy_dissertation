import json
from pathlib import Path
from collections import Counter
import itertools

DATA_PATH = Path("data/raw/ai4privacy_pii/pii-masking-200k/english_pii_43k.jsonl")

def load_jsonl_stream(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():

    print("=" * 80)
    print("Inspecting:", DATA_PATH)
    print("=" * 80)

    # Peek first 3 records
    data_iter = load_jsonl_stream(DATA_PATH)
    sample = list(itertools.islice(data_iter, 3))

    print("\n--- Keys in first record ---")
    print(sample[0].keys())

    print("\n--- First record preview ---")
    for k, v in sample[0].items():
        if isinstance(v, str):
            print(f"{k}: {v[:200]} ...")
        else:
            print(f"{k}: {v}")

    # Count entity labels (first 5000 only for speed)
    print("\n--- Sampling entity label distribution (first 5000 rows) ---")

    label_counter = Counter()
    total_rows = 0
    malformed = 0

    for row in itertools.islice(load_jsonl_stream(DATA_PATH), 5000):
        total_rows += 1

        if "entities" not in row:
            malformed += 1
            continue

        for ent in row["entities"]:
            label = ent.get("label", "UNKNOWN")
            label_counter[label] += 1

    print(f"Rows sampled: {total_rows}")
    print(f"Malformed rows (no entities key): {malformed}")

    print("\nTop 20 labels:")
    for label, count in label_counter.most_common(20):
        print(f"{label}: {count}")

    print("\nDone.")

if __name__ == "__main__":
    main()

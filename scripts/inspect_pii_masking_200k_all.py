import json
from pathlib import Path
from collections import Counter, defaultdict
import itertools

BASE = Path("data/raw/ai4privacy_pii/pii-masking-200k")

FILES = [
    ("en", "english_pii_43k.jsonl"),
    ("fr", "french_pii_62k.jsonl"),
    ("de", "german_pii_52k.jsonl"),
    ("it", "italian_pii_50k.jsonl"),
]

def stream_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main(n_preview=2, n_label_sample=20000):
    print("="*90)
    print("AI4Privacy pii-masking-200k audit")
    print("Base:", BASE)
    print("="*90)

    global_keys = set()
    label_sets = {}
    label_counts = {}
    token_len_stats = {}

    for lang, fname in FILES:
        path = BASE / fname
        print(f"\n--- {lang.upper()} :: {fname} ---")
        it = stream_jsonl(path)

        first = next(it)
        keys = set(first.keys())
        global_keys |= keys
        print("Keys:", sorted(keys))

        # preview
        print("\nPreview record fields (truncated):")
        for k, v in first.items():
            if isinstance(v, str):
                print(f"  {k}: {v[:120]} ...")
            elif isinstance(v, list):
                print(f"  {k}: list(len={len(v)})  sample={v[:8]}")
            elif isinstance(v, dict):
                items = list(v.items())[:4]
                print(f"  {k}: dict(len={len(v)})  sample={items}")
            else:
                print(f"  {k}: {v}")

        # label sampling
        c = Counter()
        lens = []
        # include first record too
        recs = itertools.chain([first], itertools.islice(it, n_label_sample-1))
        for r in recs:
            bl = r.get("bio_labels", [])
            tt = r.get("tokenised_text", [])
            if bl:
                c.update(bl)
            lens.append(len(tt))

        label_sets[lang] = set(c.keys())
        label_counts[lang] = c
        token_len_stats[lang] = (min(lens), sum(lens)/len(lens), max(lens))

        print(f"\nToken length stats (min/mean/max) over {len(lens)} samples:", token_len_stats[lang])
        print("Top 15 BIO labels:")
        for lbl, cnt in c.most_common(15):
            print(f"  {lbl}: {cnt}")

    print("\n" + "="*90)
    print("Cross-language label set comparison")
    print("="*90)
    all_labels = set.union(*label_sets.values())
    print("Total unique BIO labels across all langs:", len(all_labels))

    for lang in label_sets:
        missing = all_labels - label_sets[lang]
        extra = label_sets[lang] - all_labels  # always empty logically
        print(f"{lang.upper()} labels: {len(label_sets[lang])} | missing vs union: {len(missing)}")

    # show union minus each for quick sanity
    for lang in label_sets:
        missing = sorted(list(all_labels - label_sets[lang]))[:30]
        if missing:
            print(f"\nMissing labels in {lang.upper()} (first 30): {missing}")

    print("\nDone.")

if __name__ == "__main__":
    main()

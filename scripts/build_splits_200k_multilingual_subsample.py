import json
import random
from pathlib import Path
import pandas as pd

SRC_DIR = Path("data/processed/splits_200k")
OUT_ROOT = Path("data/processed")
SEED = 42

SUBSAMPLES = [
    ("splits_200k_10p", 0.10),
    ("splits_200k_25p", 0.25),
    ("splits_200k_50p", 0.50),
]

def _subsample_df(df, frac, seed):
    # stratify by language
    parts = []
    for lang, g in df.groupby("language"):
        n = len(g)
        k = max(1, int(n * frac))
        parts.append(g.sample(n=k, random_state=seed))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def main():
    train = pd.read_parquet(SRC_DIR / "train.parquet")
    val   = pd.read_parquet(SRC_DIR / "val.parquet")
    test  = pd.read_parquet(SRC_DIR / "test.parquet")

    label2id = json.loads((SRC_DIR / "label2id.json").read_text(encoding="utf-8"))
    meta_src = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))

    for name, frac in SUBSAMPLES:
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        tr = _subsample_df(train, frac, SEED)
        # keep val/test unchanged (recommended) so comparisons are apples-to-apples
        va = val.copy()
        te = test.copy()

        tr.to_parquet(out_dir / "train.parquet", index=False)
        va.to_parquet(out_dir / "val.parquet", index=False)
        te.to_parquet(out_dir / "test.parquet", index=False)

        meta = {
            "source": str(SRC_DIR),
            "seed": SEED,
            "train_subsample_frac": frac,
            "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "train_by_language": {k: int(v) for k, v in tr["language"].value_counts().items()},
            "num_labels": int(len(label2id)),
        }

        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")

        print(f"[DONE] {name}: train={len(tr):,} val={len(va):,} test={len(te):,}")

if __name__ == "__main__":
    main()

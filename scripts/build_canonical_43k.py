from __future__ import annotations
import ast
from pathlib import Path
import pandas as pd

RAW = Path(r"data/raw/ai4privacy_pii/pii-masking-43k/PII43k.csv")
OUT = Path(r"data/processed/ai4privacy_43k_canonical.parquet")
LOG = Path(r"outputs/runs/43k_build_log.txt")

def parse_list_cell(x: str):
    try:
        return ast.literal_eval(x)
    except Exception:
        return None

def main():
    # ✅ robust parsing + skip malformed lines
    df = pd.read_csv(
        RAW,
        engine="python",
        on_bad_lines="skip",  # key fix
    )

    # Log basic info
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(
        f"Loaded rows (after skipping bad lines): {len(df)}\n"
        f"Columns: {df.columns.tolist()}\n",
        encoding="utf-8",
    )

    # Parse list columns
    df["tokens"] = df["Tokenised Filled Template"].astype(str).map(parse_list_cell)
    df["tags"] = df["Tokens"].astype(str).map(parse_list_cell)

    # Keep only valid rows where tokens/tags parsed and aligned
    df = df[df["tokens"].notna() & df["tags"].notna()].copy()
    df["len_tokens"] = df["tokens"].map(len)
    df["len_tags"] = df["tags"].map(len)
    df = df[df["len_tokens"] == df["len_tags"]].copy()

    # Canonical columns
    out = pd.DataFrame({
        "template": df["Template"].astype(str),
        "text": df["Filled Template"].astype(str),
        "tokens": df["tokens"],
        "tags": df["tags"],
        "n_tokens": df["len_tokens"],
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print("✅ Saved:", OUT)
    print("Rows:", out.shape[0], "Cols:", out.shape[1])
    print("Log:", LOG)

if __name__ == "__main__":
    main()


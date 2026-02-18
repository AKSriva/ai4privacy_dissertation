from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/ai4privacy_43k_canonical.parquet")
OUT_DIR = Path("data/processed/splits_43k")

def main():
    df = pd.read_parquet(DATA)

    train, temp = train_test_split(df, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

if __name__ == "__main__":
    main()

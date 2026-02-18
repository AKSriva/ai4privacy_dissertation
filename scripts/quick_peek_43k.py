import pandas as pd

path = r"data\raw\ai4privacy_pii\pii-masking-43k\PII43k.csv"

# Fast: read only first 2000 rows
df = pd.read_csv(path, engine="python", nrows=2000)

print("Loaded sample rows:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3).to_string(index=False))

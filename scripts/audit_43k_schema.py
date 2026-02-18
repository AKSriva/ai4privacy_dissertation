import pandas as pd
from collections import Counter

path = r"data\raw\ai4privacy_pii\pii-masking-43k\PII43k.csv"

# Read a decent sample first (fast), then we’ll scale
df = pd.read_csv(path, engine="python", nrows=10000)

print("Shape (sample):", df.shape)
print("Columns:", df.columns.tolist())

# Basic text stats
df["n_chars"] = df["Filled Template"].astype(str).str.len()
df["n_tokens"] = df["Tokenised Filled Template"].astype(str).str.count(",") + 1

print("\nText length (chars) - describe:")
print(df["n_chars"].describe().to_string())

print("\nApprox token count - describe:")
print(df["n_tokens"].describe().to_string())

# Extract tag distribution from "Tokens" column (string representation of list)
# We'll do a lightweight parse (safe for this dataset)
def parse_list_str(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # split on comma + strip quotes/spaces
    parts = [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]
    return parts

tag_counter = Counter()
for s in df["Tokens"].astype(str).tolist():
    tags = parse_list_str(s)
    tag_counter.update(tags)

print("\nTop 20 tags (token-level counts):")
for tag, cnt in tag_counter.most_common(20):
    print(f"{tag:20s} {cnt}")

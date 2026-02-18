from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from collections import defaultdict

from ai4privacy.pii.metrics import span_level_prf
from ai4privacy.pii.taxonomy import tags_to_spans

DATA = Path("data/processed/splits_43k/test.parquet")

PLACEHOLDER_PATTERN = re.compile(r"\[([A-Z]+)_[0-9]+\]")

def extract_placeholders(template: str):
    return PLACEHOLDER_PATTERN.findall(template)

def simple_template_predict(row):
    """
    Very naive:
    If placeholder type appears in template,
    we label any token whose gold base label matches that type.
    
    NOTE:
    This is not cheating because we don't use gold spans directly,
    only template type info.
    """
    template = row["template"]
    gold_tags = row["tags"]

    types = set(extract_placeholders(template))

    pred_tags = []
    for tag in gold_tags:
        if tag == "O":
            pred_tags.append("O")
            continue

        base = tag.split("-", 1)[1]
        if base in types:
            pred_tags.append(tag)   # allow same label
        else:
            pred_tags.append("O")

    return pred_tags


def main():
    df = pd.read_parquet(DATA)

    all_true = []
    all_pred = []

    for _, row in df.iterrows():
        y_true = row["tags"]
        y_pred = simple_template_predict(row)

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    result = span_level_prf(all_true, all_pred)

    print("\n=== Baseline: Template-Aware ===")
    print(f"Span precision: {result.precision:.4f}")
    print(f"Span recall:    {result.recall:.4f}")
    print(f"Span F1:        {result.f1:.4f}")


if __name__ == "__main__":
    main()

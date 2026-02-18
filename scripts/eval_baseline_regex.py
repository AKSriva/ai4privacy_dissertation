from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import pandas as pd

from ai4privacy.pii.metrics import span_level_prf, span_level_prf_by_label


# -----------------------------
# Detokenize tokens -> string + char offsets per token
# -----------------------------
NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "'", '"', "”", "’"}
NO_SPACE_AFTER = {"(", "[", "{", "$", "“", "‘"}

def detokenize_with_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Reconstruct an approximate text from tokens and return per-token char offsets.
    Handles wordpiece-style '##' by concatenating to previous token.
    """
    s_parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    def append(piece: str):
        nonlocal cursor
        s_parts.append(piece)
        cursor += len(piece)

    prev_token: Optional[str] = None

    for tok in tokens:
        if tok.startswith("##"):
            piece = tok[2:]
            # attach to previous without space
            start = cursor
            append(piece)
            end = cursor
            offsets.append((start, end))
            prev_token = tok
            continue

        # decide if we need a leading space
        need_space = True
        if cursor == 0:
            need_space = False
        elif tok in NO_SPACE_BEFORE:
            need_space = False
        elif prev_token in NO_SPACE_AFTER:
            need_space = False

        if need_space:
            append(" ")

        start = cursor
        append(tok)
        end = cursor
        offsets.append((start, end))
        prev_token = tok

    return "".join(s_parts), offsets


# -----------------------------
# Regex patterns (conservative)
# -----------------------------
PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s]+", re.I)),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("IPV6", re.compile(r"\b(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{1,4}\b", re.I)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # credit card: 13–19 digits, allows spaces/dashes
    ("CREDITCARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    # phone (US-ish; conservative)
    ("PHONE", re.compile(r"\b(?:\+?1[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b")),
    ("ZIPCODE", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    # BTC (legacy/segwit) and ETH
    ("BITCOINADDRESS", re.compile(r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b")),
    ("ETHEREUMADDRESS", re.compile(r"\b0x[a-fA-F0-9]{40}\b")),
]

# Labels must match dataset base labels; these do based on what we saw in your tag distribution.


def find_matches(text: str) -> List[Tuple[int, int, str]]:
    """
    Return list of matches as (start_char, end_char, label).
    We keep them conservative and resolve overlaps by keeping longer spans first.
    """
    matches: List[Tuple[int, int, str]] = []
    for label, pat in PATTERNS:
        for m in pat.finditer(text):
            a, b = m.span()
            # guard against empty
            if b > a:
                matches.append((a, b, label))

    # resolve overlaps: longest first, then earliest
    matches.sort(key=lambda x: (-(x[1] - x[0]), x[0], x[2]))
    chosen: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        for x, y in occupied:
            if not (b <= x or a >= y):
                return True
        return False

    for a, b, lab in matches:
        if not overlaps(a, b):
            chosen.append((a, b, lab))
            occupied.append((a, b))

    # sort back by start
    chosen.sort(key=lambda x: x[0])
    return chosen


def charspan_to_token_span(offsets: List[Tuple[int, int]], a: int, b: int) -> Optional[Tuple[int, int]]:
    """
    Map character span [a,b) to token span [i,j) using offsets.
    Returns None if it doesn't align to any token.
    """
    # find first token that intersects
    start_i = None
    end_j = None
    for i, (x, y) in enumerate(offsets):
        if y <= a:
            continue
        if x >= b:
            break
        if start_i is None:
            start_i = i
        end_j = i

    if start_i is None or end_j is None:
        return None
    return (start_i, end_j + 1)


def make_bio_tags(n: int, spans: List[Tuple[int, int, str]]) -> List[str]:
    tags = ["O"] * n
    for i, j, lab in spans:
        if i < 0 or j > n or i >= j:
            continue
        tags[i] = f"B-{lab}"
        for k in range(i + 1, j):
            tags[k] = f"I-{lab}"
    return tags


def predict_row(tokens: List[str]) -> List[str]:
    recon, offsets = detokenize_with_offsets(tokens)
    matches = find_matches(recon)

    token_spans: List[Tuple[int, int, str]] = []
    for a, b, lab in matches:
        tspan = charspan_to_token_span(offsets, a, b)
        if tspan is None:
            continue
        i, j = tspan
        token_spans.append((i, j, lab))

    # resolve token-span overlaps too (again longest first)
    token_spans.sort(key=lambda x: (-(x[1] - x[0]), x[0], x[2]))
    final: List[Tuple[int, int, str]] = []
    occupied = [False] * len(tokens)
    for i, j, lab in token_spans:
        if any(occupied[k] for k in range(i, j)):
            continue
        for k in range(i, j):
            occupied[k] = True
        final.append((i, j, lab))
    final.sort(key=lambda x: x[0])

    return make_bio_tags(len(tokens), final)


def main():
    test_path = Path("data/processed/splits_43k/test.parquet")
    df = pd.read_parquet(test_path)

    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for tokens, gold_tags in zip(df["tokens"].tolist(), df["tags"].tolist()):
        pred_tags = predict_row(tokens)
        # sanity: lengths must match
        if len(pred_tags) != len(gold_tags):
            continue
        y_true_all.extend(gold_tags)
        y_pred_all.extend(pred_tags)

    overall = span_level_prf(y_true_all, y_pred_all)
    by_label = span_level_prf_by_label(y_true_all, y_pred_all)

    print("\n=== Baseline: Regex-only (no template, no ML) ===")
    print(f"Span precision: {overall.precision:.4f}")
    print(f"Span recall:    {overall.recall:.4f}")
    print(f"Span F1:        {overall.f1:.4f}")

    # Save outputs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/runs") / f"baseline_regex_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lab, m in by_label.items():
        rows.append({
            "label": lab,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
        })
    out = pd.DataFrame(rows).sort_values(["f1", "tp"], ascending=[False, False])
    out.to_csv(out_dir / "span_by_label.csv", index=False)

    (out_dir / "summary.txt").write_text(
        "\n".join([
            "Baseline: Regex-only",
            f"Span precision: {overall.precision:.6f}",
            f"Span recall: {overall.recall:.6f}",
            f"Span F1: {overall.f1:.6f}",
            "",
            "Per-label results saved to span_by_label.csv"
        ]),
        encoding="utf-8"
    )

    print(f"\n✅ Saved outputs to: {out_dir}\n")


if __name__ == "__main__":
    main()

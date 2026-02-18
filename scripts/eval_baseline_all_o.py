from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd

# -----------------------------
# Helpers: BIO parsing -> spans
# -----------------------------
def base_label(tag: str) -> str:
    if tag == "O":
        return "O"
    return tag.split("-", 1)[1] if "-" in tag else tag

def prefix(tag: str) -> str:
    if tag == "O":
        return "O"
    return tag.split("-", 1)[0] if "-" in tag else "O"

def tags_to_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    """
    Convert BIO tags to spans: (LABEL, start_idx, end_idx_exclusive)
    """
    spans = []
    i, n = 0, len(tags)
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue

        lab = base_label(t)
        start = i
        i += 1
        while i < n:
            t2 = tags[i]
            if t2 == "O":
                break
            if base_label(t2) != lab:
                break
            # continue only if I- of same label
            if prefix(t2) == "I":
                i += 1
                continue
            # B- starts a new span
            break
        spans.append((lab, start, i))
    return spans

def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0

# -----------------------------
# Token-level metrics
# -----------------------------
def token_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    assert len(y_true) == len(y_pred)
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    labels = sorted(set(y_true) | set(y_pred))
    # Per-label PRF (token-level)
    per = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # Macro-F1 excluding "O"
    non_o = [lab for lab in labels if lab != "O"]
    macro_f1_non_o = sum(per[lab]["f1"] for lab in non_o) / len(non_o) if non_o else 0.0

    return {
        "token_accuracy": correct / total if total else 0.0,
        "macro_f1_non_o": macro_f1_non_o,
        "labels": labels,
        "per_label": per,
    }

# -----------------------------
# Span-level metrics
# -----------------------------
def span_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    true_spans = set(tags_to_spans(y_true))
    pred_spans = set(tags_to_spans(y_pred))

    tp = len(true_spans & pred_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # By label
    labels = sorted({s[0] for s in true_spans} | {s[0] for s in pred_spans})
    by = {}
    for lab in labels:
        tset = set([s for s in true_spans if s[0] == lab])
        pset = set([s for s in pred_spans if s[0] == lab])
        tp_l = len(tset & pset)
        fp_l = len(pset - tset)
        fn_l = len(tset - pset)
        prec_l = safe_div(tp_l, tp_l + fp_l)
        rec_l = safe_div(tp_l, tp_l + fn_l)
        f1_l = (2 * prec_l * rec_l / (prec_l + rec_l)) if (prec_l + rec_l) else 0.0
        by[lab] = {"precision": prec_l, "recall": rec_l, "f1": f1_l, "tp": tp_l, "fp": fp_l, "fn": fn_l}

    macro_f1 = sum(by[lab]["f1"] for lab in labels) / len(labels) if labels else 0.0

    return {
        "span_precision": prec,
        "span_recall": rec,
        "span_f1": f1,
        "span_macro_f1": macro_f1,
        "by_label": by,
    }

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Adjust if your split folder name differs
    test_path = Path("data/processed/splits_43k/test.parquet")
    if not test_path.exists():
        # fallback for your filename if you used make_splits_43k.py
        test_path = Path("data/processed/splits_43k/test.parquet")

    df = pd.read_parquet(test_path)

    # Flatten all tokens across all rows
    y_true_all = []
    y_pred_all = []

    for tags in df["tags"].tolist():
        # baseline: predict all "O"
        y_true_all.extend(tags)
        y_pred_all.extend(["O"] * len(tags))

    tok = token_metrics(y_true_all, y_pred_all)
    spn = span_metrics(y_true_all, y_pred_all)

    print("\n=== Baseline: All-O (predict no PII) ===")
    print(f"Token accuracy:       {tok['token_accuracy']:.4f}")
    print(f"Token macro-F1 (non-O): {tok['macro_f1_non_o']:.4f}")
    print(f"Span precision:       {spn['span_precision']:.4f}")
    print(f"Span recall:          {spn['span_recall']:.4f}")
    print(f"Span F1:              {spn['span_f1']:.4f}")
    print(f"Span macro-F1:        {spn['span_macro_f1']:.4f}")

    # Save a dissertation-friendly table for top labels by FN (missed tokens)
    per = tok["per_label"]
    rows = []
    for lab, d in per.items():
        if lab == "O":
            continue
        rows.append({"label": lab, **d})
    out_df = pd.DataFrame(rows).sort_values("fn", ascending=False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/runs") / f"baseline_all_o_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_dir / "token_level_per_label.csv", index=False)

    # Span-level per label
    span_rows = []
    for lab, d in spn["by_label"].items():
        span_rows.append({"label": lab, **d})
    pd.DataFrame(span_rows).sort_values("fn", ascending=False).to_csv(
        out_dir / "span_level_per_label.csv", index=False
    )

    (out_dir / "summary.txt").write_text(
        "\n".join([
            "Baseline: All-O",
            f"Token accuracy: {tok['token_accuracy']:.6f}",
            f"Token macro-F1 (non-O): {tok['macro_f1_non_o']:.6f}",
            f"Span precision: {spn['span_precision']:.6f}",
            f"Span recall: {spn['span_recall']:.6f}",
            f"Span F1: {spn['span_f1']:.6f}",
            f"Span macro-F1: {spn['span_macro_f1']:.6f}",
        ]),
        encoding="utf-8"
    )

    print(f"\n✅ Saved outputs to: {out_dir}\n")

if __name__ == "__main__":
    main()

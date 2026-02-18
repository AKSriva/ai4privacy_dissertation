from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ai4privacy.pii.metrics import span_level_prf, span_level_prf_by_label
from ai4privacy.pii.taxonomy import tags_to_spans


SPLITS_DIR = Path("data/processed/splits_43k")
RUNS_DIR = Path("outputs/runs")
TABLES_DIR = Path("outputs/tables")
MODEL_RUN_PREFIX = "distilbert_tokenclf_"
SEED = 42

# -----------------------------
# Utilities: find latest run + checkpoint
# -----------------------------


def coerce_tokens(tokens) -> List[str]:
    # numpy array -> list
    if hasattr(tokens, "tolist") and not isinstance(tokens, (str, bytes)):
        tokens = tokens.tolist()

    # string that looks like a list -> try to parse
    if isinstance(tokens, str) and tokens.strip().startswith("[") and tokens.strip().endswith("]"):
        import ast
        try:
            tokens = ast.literal_eval(tokens)
        except Exception:
            pass

    # if still a single string, wrap it
    if isinstance(tokens, str):
        tokens = [tokens]

    # final: force each token to string and drop None
    return ["" if t is None else str(t) for t in tokens]



def find_latest_distilbert_run() -> Path:
    candidates = sorted([p for p in RUNS_DIR.glob(f"{MODEL_RUN_PREFIX}*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No distilbert runs found under {RUNS_DIR} with prefix {MODEL_RUN_PREFIX}")
    return candidates[-1]

def find_latest_checkpoint(hf_dir: Path) -> Path:
    # Prefer highest numbered checkpoint-* if present, else use hf_dir itself.
    ckpts = sorted([p for p in hf_dir.glob("checkpoint-*") if p.is_dir()],
                   key=lambda x: int(re.findall(r"checkpoint-(\d+)", x.name)[0]) if re.findall(r"checkpoint-(\d+)", x.name) else -1)
    return ckpts[-1] if ckpts else hf_dir

# -----------------------------
# Prediction alignment: tokens -> word_ids -> first-subword tags
# -----------------------------
@torch.no_grad()
def predict_tags_for_tokens(
    model,
    tokenizer,
    tokens: List[str],
    id2label: Dict[int, str],
    device: torch.device
) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits  # (1, seq_len, num_labels)
    pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True).word_ids()
    pred_tags: List[str] = []
    prev_w = None
    for j, w in enumerate(word_ids):
        if w is None:
            continue
        if w != prev_w:
            pred_tags.append(id2label[int(pred_ids[j])])
        prev_w = w

    # pred_tags should align to original tokens length (sometimes truncation can shorten)
    if len(pred_tags) != len(tokens):
        # If truncation occurred (rare with this dataset), pad with O to keep lengths stable
        if len(pred_tags) < len(tokens):
            pred_tags += ["O"] * (len(tokens) - len(pred_tags))
        else:
            pred_tags = pred_tags[:len(tokens)]
    return pred_tags

# -----------------------------
# Error analysis helpers
# -----------------------------
def span_surface(tokens: List[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end])

def spans_to_readable(tokens: List[str], spans) -> List[Dict[str, Any]]:
    out = []
    for sp in spans:
        # supports tuple spans (label, start, end) AND EntitySpan objects
        if isinstance(sp, tuple) and len(sp) == 3:
            lab, s, e = sp
        else:
            lab = getattr(sp, "label", None) or getattr(sp, "tag", None)
            s = getattr(sp, "start", None)
            e = getattr(sp, "end", None)
        if lab is None or s is None or e is None:
            continue
        out.append({"label": str(lab), "start": int(s), "end": int(e), "surface": span_surface(tokens, int(s), int(e))})
    return out


# -----------------------------
# Robustness perturbations (keep token count fixed)
# -----------------------------
def perturb_tokens(tokens: List[str], rng: random.Random, p_case: float = 0.15, p_punct: float = 0.08) -> List[str]:
    """
    Simple, controlled perturbation:
    - Randomly flip casing on some alphabetic tokens
    - Randomly append punctuation to some tokens
    Keeps token count the same (important for alignment).
    """
    punct_choices = [".", ",", "!", "?", ":", ";"]
    out = []
    for t in tokens:
        t2 = t
        if rng.random() < p_case and any(c.isalpha() for c in t2):
            # flip casing
            if t2.islower():
                t2 = t2.upper()
            elif t2.isupper():
                t2 = t2.lower()
            else:
                # swap first char case
                t2 = (t2[0].swapcase() + t2[1:]) if len(t2) > 1 else t2.swapcase()

        if rng.random() < p_punct and any(c.isalnum() for c in t2):
            t2 = t2 + rng.choice(punct_choices)

        out.append(t2)
    return out

# -----------------------------
# Main evaluation
# -----------------------------
def evaluate_split(
    df: pd.DataFrame,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device: torch.device,
    perturb: bool = False,
    rng: Optional[random.Random] = None
):
    true_all: List[str] = []
    pred_all: List[str] = []

    # store row-level info for error analysis
    row_records: List[Dict[str, Any]] = []

    for idx, (tokens, gold_tags, text) in enumerate(zip(df["tokens"], df["tags"], df["text"])):
        tokens = coerce_tokens(tokens)
        tok_in = tokens
        if perturb:
            tok_in = perturb_tokens(tokens, rng)

        pred_tags = predict_tags_for_tokens(model, tokenizer, tok_in, id2label, device)

        # safety
        if len(pred_tags) != len(gold_tags):
            continue

        true_all.extend(gold_tags)
        pred_all.extend(pred_tags)

        # For error sampling, compare spans at row-level
        gold_spans = tags_to_spans(gold_tags)
        pred_spans = tags_to_spans(pred_tags)
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        if gold_set != pred_set:
            rec = {
                "row_index": int(idx),
                "text": str(text),
                "tokens_preview": " ".join(tokens[:80]),
                "gold_spans": spans_to_readable(tokens, gold_spans),
                "pred_spans": spans_to_readable(tokens, pred_spans),
                "fp_spans": spans_to_readable(tokens, list(pred_set - gold_set)),
                "fn_spans": spans_to_readable(tokens, list(gold_set - pred_set)),
            }
            row_records.append(rec)

    overall = span_level_prf(true_all, pred_all)
    by_label = span_level_prf_by_label(true_all, pred_all)

    return overall, by_label, row_records

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_path = SPLITS_DIR / "test.parquet"
    df_test = pd.read_parquet(test_path)

    # Locate model run
    run_dir = find_latest_distilbert_run()
    hf_dir = run_dir / "hf"
    if not hf_dir.exists():
        raise FileNotFoundError(f"Expected HuggingFace output folder not found: {hf_dir}")

    ckpt = find_latest_checkpoint(hf_dir)
    print(f"[INFO] Using model checkpoint: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # label map from model config
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # Evaluate clean
    clean_overall, clean_by_label, clean_errors = evaluate_split(
        df_test, model, tokenizer, id2label, device, perturb=False
    )

    # Evaluate perturbed (robustness)
    rng = random.Random(SEED)
    pert_overall, pert_by_label, pert_errors = evaluate_split(
        df_test, model, tokenizer, id2label, device, perturb=True, rng=rng
    )

    # Output directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / f"distilbert_diagnostics_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summaries
    summary = {
        "model_run_dir": str(run_dir),
        "checkpoint_used": str(ckpt),
        "clean": asdict(clean_overall),
        "perturbed": asdict(pert_overall),
        "delta_span_f1": float(clean_overall.f1 - pert_overall.f1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DistilBERT Test Results (Clean) ===")
    print(f"Span precision: {clean_overall.precision:.4f}")
    print(f"Span recall:    {clean_overall.recall:.4f}")
    print(f"Span F1:        {clean_overall.f1:.4f}")

    print("\n=== DistilBERT Robustness (Perturbed tokens) ===")
    print(f"Span precision: {pert_overall.precision:.4f}")
    print(f"Span recall:    {pert_overall.recall:.4f}")
    print(f"Span F1:        {pert_overall.f1:.4f}")
    print(f"Δ Span F1 (clean - perturbed): {clean_overall.f1 - pert_overall.f1:.4f}")

    # Per-label tables
    def by_label_to_df(by_label: Dict[str, Any]) -> pd.DataFrame:
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
        return pd.DataFrame(rows).sort_values(["f1", "tp"], ascending=[False, False])

    df_clean = by_label_to_df(clean_by_label)
    df_pert = by_label_to_df(pert_by_label)

    df_clean.to_csv(out_dir / "span_by_label_clean.csv", index=False)
    df_pert.to_csv(out_dir / "span_by_label_perturbed.csv", index=False)

    # Also write to outputs/tables for dissertation
    df_clean.to_csv(TABLES_DIR / "distilbert_span_by_label_clean.csv", index=False)
    df_pert.to_csv(TABLES_DIR / "distilbert_span_by_label_perturbed.csv", index=False)

    # Error samples (cap to keep readable)
    max_examples = 50
    (out_dir / "errors_clean.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in clean_errors[:max_examples]),
        encoding="utf-8"
    )
    (out_dir / "errors_perturbed.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in pert_errors[:max_examples]),
        encoding="utf-8"
    )

    # Quick leaderboard view
    top_k = 20
    df_top = df_clean.sort_values("f1", ascending=False).head(top_k)
    df_bottom = df_clean.sort_values("f1", ascending=True).head(top_k)

    df_top.to_csv(out_dir / "top20_labels_clean.csv", index=False)
    df_bottom.to_csv(out_dir / "bottom20_labels_clean.csv", index=False)

    print(f"\n✅ Saved diagnostics to: {out_dir}")
    print(f"✅ Saved dissertation tables to: {TABLES_DIR}\n")

if __name__ == "__main__":
    main()

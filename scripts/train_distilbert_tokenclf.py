from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from ai4privacy.pii.metrics import span_level_prf

MODEL_NAME = "distilbert-base-cased"
SPLITS_DIR = Path("data/processed/splits_43k")
OUT_BASE = Path("outputs/runs")

def build_label_maps(train_df: pd.DataFrame):
    labels = sorted({t for tags in train_df["tags"].tolist() for t in tags})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label

def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "tokens": df["tokens"].tolist(),
        "tags": df["tags"].tolist(),
    })

def tokenize_and_align(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
    )
    labels = []

    for i, word_ids in enumerate(tokenized.word_ids(batch_index=None) if False else []):
        pass

    # HF gives word_ids per example; handle example by example
    for idx in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=idx)
        tag_seq = examples["tags"][idx]
        aligned = []
        prev_word = None
        for w in word_ids:
            if w is None:
                aligned.append(-100)
            elif w != prev_word:
                aligned.append(label2id[tag_seq[w]])
            else:
                # Subword token -> label as -100 (ignore) for simplicity
                aligned.append(-100)
            prev_word = w
        labels.append(aligned)

    tokenized["labels"] = labels
    return tokenized

def compute_span_f1(eval_preds, id2label, eval_dataset_tokens):
    logits, label_ids = eval_preds
    pred_ids = np.argmax(logits, axis=-1)

    # Need to reconstruct token-level tag sequences (one per original token list)
    # We used -100 for subwords, so we take only first-subword positions.
    true_all = []
    pred_all = []

    # The dataset stored as examples with tokens; we need word_ids again, so simplest:
    # We'll compute span metrics later using a post-pass in evaluate_model() instead of here.
    return {}

def evaluate_model(trainer: Trainer, tokenizer, df_test: pd.DataFrame, id2label: Dict[int, str], label2id: Dict[str, int]):
    # Predict on test split and compute span-level PRF (entity spans)
    ds_test = to_hf_dataset(df_test)
    ds_tok = ds_test.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id), batched=True)

    preds = trainer.predict(ds_tok)
    logits = preds.predictions
    label_ids = preds.label_ids
    pred_ids = np.argmax(logits, axis=-1)

    # Convert back to token-level BIO tags aligned with original tokens (first subword only)
    true_all = []
    pred_all = []

    for i in range(len(ds_test)):
        tokens = ds_test[i]["tokens"]
        gold_tags = ds_test[i]["tags"]

        # Re-tokenize single example to get word_ids
        enc = tokenizer(tokens, is_split_into_words=True, truncation=True, return_tensors=None)
        word_ids = enc.word_ids()

        # labels/preds include special tokens; align to first subword positions
        aligned_pred = []
        aligned_true = []
        prev = None
        for j, w in enumerate(word_ids):
            if w is None:
                continue
            if w != prev:
                # take this position
                # find corresponding position in model outputs:
                # We need the same encoding length; easiest is to use ds_tok[i]['labels'] length,
                # which matches trainer outputs. We'll index by j in that encoding.
                pass
            prev = w

        # Since aligning back precisely is tricky in pure script, we will use the ds_tok stored word_ids implicitly by re-encoding
        # But we also need preds for this example which correspond to the same encoding.
        enc_len = len(enc["input_ids"])
        example_pred_ids = pred_ids[i][:enc_len]
        example_label_ids = label_ids[i][:enc_len]

        prev = None
        for j, w in enumerate(word_ids):
            if w is None:
                continue
            if w != prev:
                aligned_pred.append(id2label[int(example_pred_ids[j])])
                # map gold tag at word index
                aligned_true.append(gold_tags[w])
            prev = w

        # sanity: lengths match original tokens? (should)
        if len(aligned_true) == len(tokens) == len(aligned_pred):
            true_all.extend(aligned_true)
            pred_all.extend(aligned_pred)

    overall = span_level_prf(true_all, pred_all)
    return overall

def main():
    train_path = SPLITS_DIR / "train.parquet"
    val_path = SPLITS_DIR / "val.parquet"
    test_path = SPLITS_DIR / "test.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    labels, label2id, id2label = build_label_maps(train_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    ds_train = to_hf_dataset(train_df)
    ds_val = to_hf_dataset(val_df)

    ds_train_tok = ds_train.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id), batched=True)
    ds_val_tok = ds_val.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id), batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"distilbert_tokenclf_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate span-level on test
    overall = evaluate_model(trainer, tokenizer, test_df, id2label, label2id)

    summary = (
        f"Model: {MODEL_NAME}\n"
        f"Span precision: {overall.precision:.6f}\n"
        f"Span recall: {overall.recall:.6f}\n"
        f"Span F1: {overall.f1:.6f}\n"
    )

    (out_dir / "span_summary.txt").write_text(summary, encoding="utf-8")
    print("\n=== DistilBERT Token-Classification (CPU) ===")
    print(summary)
    print(f"✅ Saved to: {out_dir}\n")

if __name__ == "__main__":
    main()

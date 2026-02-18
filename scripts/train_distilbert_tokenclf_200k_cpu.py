# scripts/train_distilbert_tokenclf_200k_cpu.py
# CPU-safe training: DistilBERT token classification on pii-masking-200k multilingual splits
# Input: data/processed/splits_200k/{train,val,test}.parquet + label2id.json
# Output: outputs/runs/distilbert_tokenclf_200k_cpu_<timestamp>/hf/

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from seqeval.metrics import f1_score, precision_score, recall_score


SPLITS_DIR = Path("data/processed/splits_200k")
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH   = SPLITS_DIR / "val.parquet"
TEST_PATH  = SPLITS_DIR / "test.parquet"
LABEL2ID_PATH = SPLITS_DIR / "label2id.json"

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 256
SEED = 42

OUTPUT_ROOT = Path("outputs/runs")


def _tolist(x):
    # parquet may store list columns as numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist") and not isinstance(x, (list, str, dict)):
        try:
            return x.tolist()
        except Exception:
            return x
    return x


class PiiTokenDataset(Dataset):
    """
    Uses provided tokenised_text + bio_labels.
    Tokeniser called with is_split_into_words=True; BIO labels aligned via word_ids().
    """
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = _tolist(row["tokenised_text"])
        labels = _tolist(row["bio_labels"])

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        word_ids = enc.word_ids()
        label_ids = []
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id.get(labels[w], self.label2id["O"]) if w < len(labels) else -100)

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_ids)
        return item


def load_label_maps(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {k: int(v) for k, v in label2id.items()}
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []

        for pred_seq, lab_seq in zip(preds, labels):
            seq_true = []
            seq_pred = []
            for p, l in zip(pred_seq, lab_seq):
                if l == -100:
                    continue
                seq_true.append(id2label[int(l)])
                seq_pred.append(id2label[int(p)])
            true_labels.append(seq_true)
            true_preds.append(seq_pred)

        return {
            "precision": float(precision_score(true_labels, true_preds)),
            "recall": float(recall_score(true_labels, true_preds)),
            "f1": float(f1_score(true_labels, true_preds)),
        }
    return compute_metrics


def main():
    # Force CPU
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    set_seed(SEED)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"distilbert_tokenclf_200k_cpu_{ts}"
    run_dir = OUTPUT_ROOT / run_id
    hf_dir = run_dir / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "splits_dir": str(SPLITS_DIR),
        "label2id_path": str(LABEL2ID_PATH),
        "train_path": str(TRAIN_PATH),
        "val_path": str(VAL_PATH),
        "test_path": str(TEST_PATH),
        "device": "cpu",
    }
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    label2id, id2label = load_label_maps(LABEL2ID_PATH)
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    train_ds = PiiTokenDataset(train_df, tokenizer, label2id, MAX_LENGTH)
    val_ds = PiiTokenDataset(val_df, tokenizer, label2id, MAX_LENGTH)
    test_ds = PiiTokenDataset(test_df, tokenizer, label2id, MAX_LENGTH)

    collator = DataCollatorForTokenClassification(tokenizer)

    # CPU-safe settings:
    # - small batch
    # - gradient accumulation to simulate larger effective batch
    # - fp16 off
    # - workers=0 for Windows stability
    args = TrainingArguments(
        output_dir=str(hf_dir),
        overwrite_output_dir=True,

        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=2,
        logging_steps=200,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,   # effective train batch = 32

        num_train_epochs=1,              # start with 1 epoch on CPU; scale later
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,

        fp16=False,
        bf16=False,
        no_cuda=True,

        dataloader_num_workers=0,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    metrics = {
        "val": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
        "num_labels": int(num_labels),
        "run_id": run_id,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    trainer.save_model(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    print(f"[DONE] run_id={run_id}")
    print(f"Run dir: {run_dir}")
    print("Test metrics:", metrics["test"])


if __name__ == "__main__":
    import os
    main()

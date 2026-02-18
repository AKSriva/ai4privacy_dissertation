from typing import Dict, List
from torch.utils.data import Dataset
import torch


class PiiBioDataset(Dataset):
    """
    Uses provided 'tokenised_text' (already wordpiece tokens) + 'bio_labels'
    """
    def __init__(self, records: List[dict], label2id: Dict[str, int], tokenizer, max_length: int = 256):
        self.records = records
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        tokens = r["tokenised_text"]
        labels = r["bio_labels"]

        # Encode using split tokens
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None
        )

        # Align labels 1-to-1 with provided tokens.
        # Because tokens are already wordpiece-level, the tokenizer should produce the same segmentation.
        # We still use word_ids() to map labels to final encoding positions.
        word_ids = enc.word_ids() if hasattr(enc, "word_ids") else None

        label_ids = [-100] * len(enc["input_ids"])

        if word_ids is None:
            # fallback: assume perfect alignment (rare)
            # account for truncation/padding
            usable = min(len(labels), self.max_length)
            for i in range(usable):
                label_ids[i] = self.label2id.get(labels[i], self.label2id["O"])
        else:
            for i, w in enumerate(word_ids):
                if w is None:
                    label_ids[i] = -100
                elif w < len(labels):
                    label_ids[i] = self.label2id.get(labels[w], self.label2id["O"])
                else:
                    label_ids[i] = -100

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_ids)

        # keep language for analysis (not used by model)
        item["language"] = r.get("language", "na")
        return item

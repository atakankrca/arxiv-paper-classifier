import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from arxiv_paper_classifier.utils.label_utils import parse_categories

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _clean_text(text: str) -> list[str]:
    """Lowercase, strip non-alpha, tokenize by whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class TextPreprocessor:
    """Fit vocabulary and label binarizer from raw arXiv records, transform text/labels."""

    def __init__(self, max_seq_len: int = 256, min_freq: int = 5, vocab_size: int = 50000):
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq
        self.vocab_size = vocab_size

        self.vocab: dict[str, int] = {}
        self.mlb: MultiLabelBinarizer = MultiLabelBinarizer()
        self._is_fitted = False

    @property
    def num_classes(self) -> int:
        return len(self.mlb.classes_)

    @property
    def actual_vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, records: list[dict]) -> "TextPreprocessor":
        """Build vocabulary and label binarizer from a list of arXiv records."""
        word_counts: Counter = Counter()
        all_labels: list[list[str]] = []

        for rec in tqdm(records, desc="Fitting preprocessor"):
            title = rec.get("title", "") or ""
            abstract = rec.get("abstract", "") or ""
            tokens = _clean_text(f"{title} {abstract}")
            word_counts.update(tokens)

            cats = parse_categories(rec.get("categories", ""))
            if cats:
                all_labels.append(cats)

        self.vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        sorted_words = sorted(
            [(w, c) for w, c in word_counts.items() if c >= self.min_freq],
            key=lambda x: -x[1],
        )
        for word, _ in sorted_words[: self.vocab_size - 2]:
            self.vocab[word] = len(self.vocab)

        self.mlb.fit(all_labels)
        self._is_fitted = True
        return self

    def transform_text(self, title: str, abstract: str) -> list[int]:
        """Convert title + abstract to a padded/truncated list of token IDs."""
        tokens = _clean_text(f"{title} {abstract}")
        unk_id = self.vocab[UNK_TOKEN]
        ids = [self.vocab.get(t, unk_id) for t in tokens[: self.max_seq_len]]
        pad_len = self.max_seq_len - len(ids)
        ids += [self.vocab[PAD_TOKEN]] * pad_len
        return ids

    def transform_labels(self, categories_str: str) -> np.ndarray:
        """Convert space-separated category string to multi-hot float32 vector."""
        cats = parse_categories(categories_str)
        known = [[c for c in cats if c in self.mlb.classes_]]
        if not known[0]:
            return np.zeros(self.num_classes, dtype=np.float32)
        return self.mlb.transform(known)[0].astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TextPreprocessor":
        with path.open("rb") as f:
            return pickle.load(f)


def stream_records(json_path: Path, max_samples: int | None = None) -> list[dict]:
    """Read arXiv JSONL file lazily, returning up to max_samples records."""
    records = []
    with json_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Reading JSON")):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def time_based_split(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    date_field: str = "update_date",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records by date order into train / val / test."""
    sorted_recs = sorted(records, key=lambda r: r.get(date_field, ""))
    n = len(sorted_recs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return sorted_recs[:train_end], sorted_recs[train_end:val_end], sorted_recs[val_end:]

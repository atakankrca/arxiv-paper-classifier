import torch
from torch.utils.data import Dataset

from arxiv_paper_classifier.data.preprocessing import TextPreprocessor


class ArxivDataset(Dataset):
    """PyTorch Dataset for arXiv paper classification."""

    def __init__(self, records: list[dict], preprocessor: TextPreprocessor):
        self.records = records
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        token_ids = self.preprocessor.transform_text(
            rec.get("title", "") or "",
            rec.get("abstract", "") or "",
        )
        label_vec = self.preprocessor.transform_labels(rec.get("categories", "") or "")
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
            label_vec, dtype=torch.float32
        )


class InferenceDataset(Dataset):
    """Dataset for inference: accepts raw text strings, returns token IDs only."""

    def __init__(self, texts: list[str], preprocessor: TextPreprocessor):
        self.texts = texts
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        token_ids = self.preprocessor.transform_text("", self.texts[idx])
        return torch.tensor(token_ids, dtype=torch.long)

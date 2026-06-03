from pathlib import Path

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from arxiv_paper_classifier.data.dataset import ArxivDataset
from arxiv_paper_classifier.data.preprocessing import (
    TextPreprocessor,
    stream_records,
    time_based_split,
)
from arxiv_paper_classifier.utils.dvc_utils import check_data_exists, pull_data


class ArxivDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor: TextPreprocessor | None = None
        self.train_records: list[dict] = []
        self.val_records: list[dict] = []
        self.test_records: list[dict] = []

    def prepare_data(self) -> None:
        raw_path = Path(self.cfg.data.raw_path)
        if not check_data_exists(raw_path):
            pull_data(str(raw_path))

    def setup(self, stage: str | None = None) -> None:
        processed_dir = Path(self.cfg.data.processed_dir)
        preprocessor_path = processed_dir / "preprocessor.pkl"
        raw_path = Path(self.cfg.data.raw_path)

        max_samples = self.cfg.data.get("max_samples", None)
        records = stream_records(raw_path, max_samples=max_samples)

        if preprocessor_path.exists():
            self.preprocessor = TextPreprocessor.load(preprocessor_path)
        else:
            self.preprocessor = TextPreprocessor(
                max_seq_len=self.cfg.preprocessing.max_seq_len,
                min_freq=self.cfg.preprocessing.min_freq,
                vocab_size=self.cfg.preprocessing.vocab_size,
            )
            self.preprocessor.fit(records)
            self.preprocessor.save(preprocessor_path)

        self.train_records, self.val_records, self.test_records = time_based_split(
            records,
            train_ratio=self.cfg.data.train_split,
            val_ratio=self.cfg.data.val_split,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = ArxivDataset(self.train_records, self.preprocessor)  # type: ignore[arg-type]
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = ArxivDataset(self.val_records, self.preprocessor)  # type: ignore[arg-type]
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = ArxivDataset(self.test_records, self.preprocessor)  # type: ignore[arg-type]
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

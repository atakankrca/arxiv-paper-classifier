from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from arxiv_paper_classifier.utils.label_utils import build_embedding_matrix, download_glove


class MLPClassifier(nn.Module):
    """MLP classifier with pre-trained GloVe word embeddings (mean-pooled)."""

    def __init__(
        self,
        cfg: DictConfig,
        vocab_size: int,
        num_classes: int,
        glove_matrix: np.ndarray | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=0)

        if glove_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(glove_matrix))

        if cfg.get("freeze_embeddings", False):
            self.embedding.weight.requires_grad = False

        layers: list[nn.Module] = []
        in_dim = cfg.embedding_dim
        for _ in range(cfg.num_layers):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = cfg.hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(cfg.hidden_dim, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).float().unsqueeze(-1)
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.mlp(pooled))


def build_baseline(
    cfg: DictConfig,
    vocab: dict[str, int],
    num_classes: int,
    glove_dir: Path | None = None,
) -> MLPClassifier:
    """Instantiate MLPClassifier, optionally loading GloVe weights."""
    glove_matrix = None
    if glove_dir is not None:
        glove_path = download_glove(glove_dir, dim=cfg.embedding_dim)
        glove_matrix = build_embedding_matrix(vocab, glove_path, dim=cfg.embedding_dim)

    return MLPClassifier(
        cfg, vocab_size=len(vocab), num_classes=num_classes, glove_matrix=glove_matrix
    )

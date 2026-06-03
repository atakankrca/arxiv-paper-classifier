from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig

from arxiv_paper_classifier.data.preprocessing import TextPreprocessor
from arxiv_paper_classifier.models.baseline import build_baseline
from arxiv_paper_classifier.models.hierarchical_head import HierarchicalClassificationHead
from arxiv_paper_classifier.models.transformer_encoder import TransformerEncoder
from arxiv_paper_classifier.utils.label_utils import build_group_to_indices, get_group


class TransformerWithHead(nn.Module):
    def __init__(
        self,
        encoder: TransformerEncoder,
        head: HierarchicalClassificationHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, token_ids):
        features = self.encoder(token_ids)
        return self.head(features)


def build_model(
    cfg: DictConfig,
    preprocessor: TextPreprocessor,
    glove_dir: Path | None = None,
) -> nn.Module:
    num_classes = preprocessor.num_classes
    vocab = preprocessor.vocab
    vocab_size = preprocessor.actual_vocab_size

    if cfg.model.name == "baseline":
        return build_baseline(cfg.model, vocab, num_classes, glove_dir)

    if cfg.model.name == "transformer":
        encoder = TransformerEncoder(cfg.model, vocab_size)

        leaf_classes = list(preprocessor.mlb.classes_)
        all_groups = sorted({get_group(c) for c in leaf_classes})
        group_to_index = {g: i for i, g in enumerate(all_groups)}
        group_to_leaf_indices = build_group_to_indices(leaf_classes)
        named_group_to_leaf = {
            g: idxs for g, idxs in group_to_leaf_indices.items() if g in group_to_index
        }

        head = HierarchicalClassificationHead(
            input_dim=cfg.model.embedding_dim,
            num_groups=len(all_groups),
            num_leaves=num_classes,
            group_to_leaf_indices=named_group_to_leaf,
            use_soft_mask=cfg.model.get("use_soft_mask", False),
            dropout=cfg.model.dropout,
        )
        return TransformerWithHead(encoder, head)

    raise ValueError(f"Unknown model name: {cfg.model.name!r}")

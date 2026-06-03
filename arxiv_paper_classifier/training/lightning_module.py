from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelPrecision


class ArxivClassifierModule(L.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig, num_classes: int):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=["model"])

        metric_kwargs = {"num_labels": num_classes, "average": "micro"}
        self.train_f1 = MultilabelF1Score(**metric_kwargs)
        self.val_f1 = MultilabelF1Score(**metric_kwargs)
        self.val_auroc = MultilabelAUROC(**metric_kwargs)
        self.val_precision = MultilabelPrecision(**metric_kwargs)

    def forward(self, token_ids: torch.Tensor) -> Any:
        return self.model(token_ids)

    def _compute_loss(self, output: Any, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            group_logits, leaf_logits = output
            leaf_loss = F.binary_cross_entropy_with_logits(leaf_logits, labels)
            group_weight = self.cfg.model.get("group_loss_weight", 0.3)
            leaf_weight = self.cfg.model.get("leaf_loss_weight", 0.7)
            group_labels = self._leaf_to_group_labels(labels, group_logits.size(1))
            group_loss = F.binary_cross_entropy_with_logits(group_logits, group_labels)
            return leaf_weight * leaf_loss + group_weight * group_loss
        return F.binary_cross_entropy_with_logits(output, labels)

    def _get_leaf_logits(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[1]
        return output

    def _leaf_to_group_labels(self, leaf_labels: torch.Tensor, num_groups: int) -> torch.Tensor:
        batch = leaf_labels.size(0)
        group_labels = torch.zeros(batch, num_groups, device=leaf_labels.device)
        leaves_per_group = self.num_classes // num_groups
        for g in range(num_groups):
            start = g * leaves_per_group
            end = start + leaves_per_group if g < num_groups - 1 else self.num_classes
            group_labels[:, g] = leaf_labels[:, start:end].any(dim=1).float()
        return group_labels

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        token_ids, labels = batch
        output = self(token_ids)
        loss = self._compute_loss(output, labels)
        leaf_logits = self._get_leaf_logits(output)
        preds = torch.sigmoid(leaf_logits)
        self.train_f1(preds, labels.int())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        token_ids, labels = batch
        output = self(token_ids)
        loss = self._compute_loss(output, labels)
        leaf_logits = self._get_leaf_logits(output)
        preds = torch.sigmoid(leaf_logits)
        self.val_f1(preds, labels.int())
        self.val_auroc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True)
        self.log("val/precision", self.val_precision, on_epoch=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        token_ids, labels = batch
        output = self(token_ids)
        loss = self._compute_loss(output, labels)
        leaf_logits = self._get_leaf_logits(output)
        preds = torch.sigmoid(leaf_logits)
        self.log("test/loss", loss, on_epoch=True)
        self.log(
            "test/f1",
            MultilabelF1Score(num_labels=self.num_classes, average="micro").to(preds.device)(
                preds, labels.int()
            ),
            on_epoch=True,
        )

    def on_train_epoch_end(self) -> None:
        current_lr = self.optimizers().param_groups[0]["lr"]  # type: ignore[union-attr]
        self.log("train/lr", current_lr, on_epoch=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if isinstance(batch, tuple):
            token_ids = batch[0]
        else:
            token_ids = batch
        output = self(token_ids)
        return torch.sigmoid(self._get_leaf_logits(output))

    @staticmethod
    def load_for_inference(
        checkpoint_path: Path, model: nn.Module, cfg: DictConfig, num_classes: int
    ) -> "ArxivClassifierModule":
        return ArxivClassifierModule.load_from_checkpoint(
            str(checkpoint_path),
            model=model,
            cfg=cfg,
            num_classes=num_classes,
        )

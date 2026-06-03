from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from lightning import Callback, LightningModule, Trainer


class PlotSavingCallback(Callback):
    """Accumulate per-epoch metrics and save training curve plots at end of training."""

    def __init__(self, plots_dir: Path = Path("plots")):
        self.plots_dir = plots_dir
        self._history: dict[str, list[float]] = defaultdict(list)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for key in ("train/loss_epoch", "train/f1"):
            val = trainer.callback_metrics.get(key)
            if val is not None:
                self._history[key].append(float(val))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for key in ("val/loss", "val/f1", "val/auroc", "val/precision"):
            val = trainer.callback_metrics.get(key)
            if val is not None:
                self._history[key].append(float(val))

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._save_plot(
            series={
                "Train loss": self._history.get("train/loss_epoch", []),
                "Val loss": self._history.get("val/loss", []),
            },
            title="Loss over epochs",
            ylabel="BCE Loss",
            filename="loss_curve.png",
        )
        self._save_plot(
            series={"Val F1 (micro)": self._history.get("val/f1", [])},
            title="Validation F1 over epochs",
            ylabel="F1 Score",
            filename="val_f1_curve.png",
        )
        self._save_plot(
            series={"Val AUROC (micro)": self._history.get("val/auroc", [])},
            title="Validation AUROC over epochs",
            ylabel="AUROC",
            filename="val_auroc_curve.png",
        )

        try:
            for png in self.plots_dir.glob("*.png"):
                mlflow.log_artifact(str(png), artifact_path="plots")
        except Exception:
            pass

    def _save_plot(
        self,
        series: dict[str, list[float]],
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        non_empty = {k: v for k, v in series.items() if v}
        if not non_empty:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, values in non_empty.items():
            ax.plot(range(1, len(values) + 1), values, marker="o", label=label)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = self.plots_dir / filename
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_path}")

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from lightning import Callback, LightningModule, Trainer


class PlotSavingCallback(Callback):
    """Save training curve plots to the plots/ directory at end of training."""

    def __init__(self, plots_dir: Path = Path("plots")):
        self.plots_dir = plots_dir

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        metrics_history = trainer.callback_metrics
        logged_metrics: dict[str, list] = {}

        if hasattr(trainer.logger, "experiment"):
            try:
                run_id = trainer.logger.run_id  # type: ignore[union-attr]
                client = mlflow.tracking.MlflowClient(
                    tracking_uri=trainer.logger.tracking_uri  # type: ignore[union-attr]
                )
                for metric_name in ["train/loss_epoch", "val/loss", "val/f1", "val/auroc"]:
                    history = client.get_metric_history(run_id, metric_name)
                    if history:
                        logged_metrics[metric_name] = [m.value for m in history]
            except Exception:
                pass

        if not logged_metrics:
            logged_metrics = {
                k: [v.item() if hasattr(v, "item") else float(v)]
                for k, v in metrics_history.items()
            }

        self._plot_metric(
            logged_metrics, ["train/loss_epoch", "val/loss"], "Loss", "loss_curve.png"
        )
        self._plot_metric(logged_metrics, ["val/f1"], "Validation F1", "val_f1_curve.png")
        self._plot_metric(logged_metrics, ["val/auroc"], "Validation AUROC", "val_auroc_curve.png")

        if hasattr(trainer.logger, "experiment"):
            try:
                for png in self.plots_dir.glob("*.png"):
                    mlflow.log_artifact(str(png), artifact_path="plots")
            except Exception:
                pass

    def _plot_metric(
        self,
        metrics: dict[str, list],
        keys: list[str],
        title: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for key in keys:
            if key in metrics and metrics[key]:
                ax.plot(metrics[key], label=key)
                plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
        output_path = self.plots_dir / filename
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_path}")

from pathlib import Path

import lightning as L
import mlflow
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from arxiv_paper_classifier.data.datamodule import ArxivDataModule
from arxiv_paper_classifier.data.preprocessing import (
    TextPreprocessor,
    stream_records,
)
from arxiv_paper_classifier.models.factory import build_model
from arxiv_paper_classifier.training.callbacks import PlotSavingCallback
from arxiv_paper_classifier.training.lightning_module import ArxivClassifierModule
from arxiv_paper_classifier.utils.dvc_utils import pull_data, pull_models
from arxiv_paper_classifier.utils.mlflow_utils import log_run_metadata, setup_mlflow

_CONFIGS_DIR = str(Path(__file__).parent.parent / "configs")


def _load_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def download(overrides: list[str] | None = None) -> None:
    """Pull raw data from DVC data-store remote."""
    cfg = _load_cfg(overrides)
    raw_path = Path(cfg.data.raw_path)
    if not raw_path.exists():
        pull_data(str(raw_path))
    else:
        print(f"Data already present at {raw_path}")


def preprocess(overrides: list[str] | None = None) -> None:
    """Fit and save the TextPreprocessor from raw data."""
    cfg = _load_cfg(overrides)
    raw_path = Path(cfg.data.raw_path)
    processed_dir = Path(cfg.data.processed_dir)
    preprocessor_path = processed_dir / "preprocessor.pkl"

    if preprocessor_path.exists():
        print(f"Preprocessor already exists at {preprocessor_path}")
        return

    max_samples = cfg.data.get("max_samples", None)
    records = stream_records(raw_path, max_samples=max_samples)

    preprocessor = TextPreprocessor(
        max_seq_len=cfg.preprocessing.max_seq_len,
        min_freq=cfg.preprocessing.min_freq,
        vocab_size=cfg.preprocessing.vocab_size,
    )
    preprocessor.fit(records)
    preprocessor.save(preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    print(f"  Vocabulary size: {preprocessor.actual_vocab_size}")
    print(f"  Number of classes: {preprocessor.num_classes}")


def train(overrides: list[str] | None = None) -> None:
    """Train the arXiv classifier with the given Hydra config overrides."""
    cfg = _load_cfg(overrides)
    L.seed_everything(cfg.seed)

    datamodule = ArxivDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    preprocessor = datamodule.preprocessor
    assert preprocessor is not None

    glove_dir = Path(cfg.data.processed_dir) if cfg.model.name == "baseline" else None
    model = build_model(cfg, preprocessor, glove_dir=glove_dir)

    lightning_module = ArxivClassifierModule(model, cfg, preprocessor.num_classes)

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=cfg.mlflow.log_model,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
        ),
        PlotSavingCallback(plots_dir=Path("plots")),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=mlflow_logger,
        callbacks=callbacks,
    )

    with mlflow.start_run(
        experiment_id=mlflow_logger.experiment_id,
        run_id=mlflow_logger.run_id,
    ):
        log_run_metadata(cfg)
        trainer.fit(lightning_module, datamodule=datamodule)

    print(f"Training complete. Best checkpoint: {checkpoint_dir / 'best_model.ckpt'}")


def infer(
    checkpoint: str = "models/best_model.ckpt",
    input_text: str = "",
    overrides: list[str] | None = None,
) -> list[str]:
    """Run inference on a single text input and return top predicted categories."""
    cfg = _load_cfg(overrides)

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        pull_models(checkpoint)

    processed_dir = Path(cfg.data.processed_dir)
    preprocessor_path = processed_dir / "preprocessor.pkl"
    preprocessor = TextPreprocessor.load(preprocessor_path)

    model = build_model(cfg, preprocessor)
    lightning_module = ArxivClassifierModule.load_for_inference(
        checkpoint_path, model, cfg, preprocessor.num_classes
    )
    lightning_module.eval()

    token_ids = preprocessor.transform_text("", input_text)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        output = lightning_module(input_tensor)
        if isinstance(output, tuple):
            probs = torch.sigmoid(output[1])[0]
        else:
            probs = torch.sigmoid(output)[0]

    top_k = 5
    top_indices = probs.topk(top_k).indices.tolist()
    classes = list(preprocessor.mlb.classes_)
    results = [(classes[i], float(probs[i])) for i in top_indices]

    for cat, score in results:
        print(f"  {cat}: {score:.4f}")

    return [cat for cat, _ in results]

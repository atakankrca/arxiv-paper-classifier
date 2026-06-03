import subprocess
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf


def get_git_commit() -> str:
    """Return current git commit SHA, or 'unknown' if not in a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def flatten_cfg(cfg: DictConfig, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested OmegaConf DictConfig into a flat dict for MLflow logging."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, DictConfig):
            flat.update(flatten_cfg(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def setup_mlflow(cfg: DictConfig) -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def log_run_metadata(cfg: DictConfig) -> None:
    """Log hyperparameters and git commit to the active MLflow run."""
    mlflow.set_tag("git_commit", get_git_commit())
    params = flatten_cfg(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

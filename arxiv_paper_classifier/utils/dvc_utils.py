import subprocess
from pathlib import Path


def _run_dvc(args: list[str]) -> None:
    result = subprocess.run(["dvc"] + args, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"DVC command failed: dvc {' '.join(args)}")


def pull_data(path: str | None = None) -> None:
    """Pull DVC-tracked data files from the data-store remote."""
    cmd = ["pull", "--remote", "data-store"]
    if path:
        cmd.append(path)
    _run_dvc(cmd)


def pull_models(path: str | None = None) -> None:
    """Pull DVC-tracked model artifacts from the model-store remote."""
    cmd = ["pull", "--remote", "model-store"]
    if path:
        cmd.append(path)
    _run_dvc(cmd)


def push_data(path: str | None = None) -> None:
    """Push DVC-tracked data files to the data-store remote."""
    cmd = ["push", "--remote", "data-store"]
    if path:
        cmd.append(path)
    _run_dvc(cmd)


def push_models(path: str | None = None) -> None:
    """Push DVC-tracked model artifacts to the model-store remote."""
    cmd = ["push", "--remote", "model-store"]
    if path:
        cmd.append(path)
    _run_dvc(cmd)


def check_data_exists(path: Path) -> bool:
    """Return True if the DVC-tracked path exists locally."""
    return path.exists()

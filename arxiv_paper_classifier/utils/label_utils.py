import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILENAME = "glove.6B.100d.txt"


def parse_categories(categories_str: str) -> list[str]:
    """Split space-separated arXiv category string into a list."""
    return [c.strip() for c in categories_str.strip().split() if c.strip()]


def get_group(category: str) -> str:
    """Return top-level group from a category string (e.g. 'cs.LG' → 'cs')."""
    return category.split(".")[0]


def build_hierarchy(all_categories: list[str]) -> dict[str, list[str]]:
    """Build mapping from top-level group to list of leaf categories."""
    hierarchy: dict[str, list[str]] = {}
    for cat in all_categories:
        group = get_group(cat)
        if group not in hierarchy:
            hierarchy[group] = []
        if cat not in hierarchy[group]:
            hierarchy[group].append(cat)
    return hierarchy


def build_group_to_indices(
    leaf_classes: list[str],
) -> dict[str, list[int]]:
    """Map each top-level group to leaf indices in the binarizer class list."""
    group_to_indices: dict[str, list[int]] = {}
    for idx, cat in enumerate(leaf_classes):
        group = get_group(cat)
        if group not in group_to_indices:
            group_to_indices[group] = []
        group_to_indices[group].append(idx)
    return group_to_indices


def download_glove(target_dir: Path, dim: int = 100) -> Path:
    """Download and extract GloVe embeddings if not already present."""
    target_dir.mkdir(parents=True, exist_ok=True)
    glove_file = target_dir / f"glove.6B.{dim}d.txt"
    if glove_file.exists():
        return glove_file

    zip_path = target_dir / "glove.6B.zip"
    if not zip_path.exists():
        print(f"Downloading GloVe 6B embeddings to {zip_path}...")
        response = requests.get(GLOVE_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with zip_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Extracting {glove_file.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(glove_file.name, target_dir)

    return glove_file


def build_embedding_matrix(
    vocab: dict[str, int],
    glove_path: Path,
    dim: int = 100,
) -> np.ndarray:
    """Build embedding weight matrix aligned with vocabulary indices."""
    matrix = np.random.normal(scale=0.1, size=(len(vocab), dim)).astype(np.float32)
    loaded = 0
    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                loaded += 1
    print(f"Loaded {loaded}/{len(vocab)} GloVe vectors")
    return matrix

# arXiv Paper Classifier

Multi-label hierarchical classification of research paper abstracts into arXiv categories.

## Project Description

Given a paper title and abstract, the model predicts a multi-label probability vector over arXiv
categories (e.g., `cs.LG`, `hep-ph`, `math.CO`). Categories follow a two-level hierarchy: a
top-level group (e.g., `cs`, `hep`) and specific subcategories.

**Dataset:** [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(~2M papers in JSON format, key fields: `id`, `title`, `abstract`, `categories`, `update_date`)

**Models:**

- **Baseline:** MLP with pre-trained GloVe word embeddings (mean-pooled)
- **Main:** Custom Transformer Encoder built from scratch (no `nn.Transformer` / HuggingFace)
  with a hierarchical classification head (group-level + leaf-level branches)

**Metrics:** Micro-averaged F1-Score, Precision@K, AUROC

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd arxiv-paper-classifier

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install

# Verify hooks pass
pre-commit run -a
```

> **Note on PyTorch:** The project requires `torch>=2.2.0`. If your system does not have a
> GPU-compatible PyTorch installed, install the appropriate variant first:
>
> ```bash
> # CPU only
> uv pip install torch --index-url https://download.pytorch.org/whl/cpu
> # CUDA 12.1
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### DVC Remotes

The project uses two DVC remotes. Configure them before pulling data:

```bash
# Set data remote (replace with your actual remote path/URL)
dvc remote modify data-store url /path/to/your/data-remote

# Set model remote
dvc remote modify model-store url /path/to/your/model-remote
```

## Train

### 1. Start MLflow server

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 2. Download raw data

```bash
python train.py download
```

### 3. Preprocess

```bash
python train.py preprocess
```

### 4. Train

```bash
# Train with default Transformer model
python train.py train

# Train baseline MLP
python train.py train model=baseline

# Override hyperparameters
python train.py train training.max_epochs=10 data.batch_size=64

# Quick smoke test (small sample)
python train.py train data.max_samples=5000 training.max_epochs=2

# Run full DVC pipeline (download → preprocess → train → infer)
dvc repro
```

### 5. Inference

```bash
python train.py infer --checkpoint models/best_model.ckpt \
  --input_text "Attention is all you need. We propose a new network architecture..."
```

## Project Structure

```
arxiv_paper_classifier/   # Main Python package
configs/                  # Hydra configuration files
data/                     # DVC-tracked data (not in git)
models/                   # DVC-tracked model checkpoints
plots/                    # Training curves (committed to git)
train.py                  # Top-level entry point
dvc.yaml                  # DVC pipeline stages
```

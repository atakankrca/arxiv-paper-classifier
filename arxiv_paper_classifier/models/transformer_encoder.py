import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, embedding_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention built from scratch (no nn.MultiheadAttention)."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        attended = self.attention(q, k, v, mask=mask)
        return self.w_o(self._merge_heads(attended))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer."""

    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ff = FeedForward(embedding_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Custom Transformer Encoder with CLS token.

    Uses no nn.Transformer or HuggingFace modules.
    Returns the CLS token representation for classification.
    """

    def __init__(self, cfg: DictConfig, vocab_size: int):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.token_embedding = TokenEmbedding(vocab_size, cfg.embedding_dim)
        self.pos_encoding = PositionalEncoding(
            cfg.embedding_dim,
            max_seq_len=cfg.max_seq_len + 1,
            dropout=cfg.dropout,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(cfg.embedding_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch = token_ids.size(0)
        padding_mask = token_ids != 0

        x = self.token_embedding(token_ids)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)

        cls_mask_col = torch.ones(batch, 1, dtype=torch.bool, device=token_ids.device)
        full_mask = torch.cat([cls_mask_col, padding_mask], dim=1)

        for layer in self.layers:
            x = layer(x, mask=full_mask)

        return self.norm(x[:, 0])

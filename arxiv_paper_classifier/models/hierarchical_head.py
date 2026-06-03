import torch
import torch.nn as nn


class HierarchicalClassificationHead(nn.Module):
    """
    Two-branch classification head for hierarchical arXiv categories.

    Branch 1: group-level (cs, hep, math, …)
    Branch 2: leaf-level (cs.LG, hep-ph, …)

    Leaf logits are optionally soft-masked by parent group probabilities
    to encourage hierarchical consistency.
    """

    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        num_leaves: int,
        group_to_leaf_indices: dict[str, list[int]],
        use_soft_mask: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_to_leaf_indices = group_to_leaf_indices
        self.num_groups = num_groups
        self.num_leaves = num_leaves
        self.use_soft_mask = use_soft_mask

        self.group_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_groups),
        )
        self.leaf_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_leaves),
        )

        if use_soft_mask:
            parent_mask = torch.zeros(num_leaves, num_groups)
            for g_idx, (group, leaf_indices) in enumerate(group_to_leaf_indices.items()):
                for l_idx in leaf_indices:
                    parent_mask[l_idx, g_idx] = 1.0
            self.register_buffer("parent_mask", parent_mask)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        group_logits = self.group_head(features)
        leaf_logits = self.leaf_head(features)

        if self.use_soft_mask:
            group_probs = torch.sigmoid(group_logits)
            parent_probs = torch.matmul(group_probs, self.parent_mask.t())  # type: ignore[attr-defined]
            leaf_logits = leaf_logits + torch.log(parent_probs.clamp(min=1e-6))

        return group_logits, leaf_logits

"""BC policy network: encoded battle -> action logits.

Simple-first (plan.md, M2): a compact MLP over the flat feature vector from
``features.encoder``, emitting one logit per action in the poke-env singles
action space. Illegal actions are masked by the *caller* (training applies the
mask before cross-entropy; the agent applies it before arg-max), so ``forward``
returns raw logits and stays a pure tensor op.

The 12-Pokemon / temporal Transformer + value head from the PRD are the M3+
upgrade; this module is intentionally small enough to train on a laptop (CPU/MPS).
"""

from __future__ import annotations

import torch
from torch import nn

from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE

# Large finite negative instead of -inf: keeps illegal actions effectively
# impossible while avoiding NaN gradients in log-softmax on MPS/CPU.
NEG_INF = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Push logits of illegal actions far below legal ones (``mask`` True == legal)."""
    return logits.masked_fill(~mask, NEG_INF)


class BCPolicy(nn.Module):
    """Feed-forward policy: ``input_dim`` features -> ``n_actions`` logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_actions: int = GEN9_ACTION_SPACE_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

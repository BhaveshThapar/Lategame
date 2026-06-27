"""Actor-critic policy with a value-classification head (plan.md, M3).

Same flat-feature MLP as the M2 ``BCPolicy``, refactored into a shared ``trunk``
plus two heads: ``policy_head`` (action logits, identical to BC) and ``value_head``
(logits over a fixed set of return bins). The value is learned by **classification**
-- HL-Gauss cross-entropy against a Gaussian-smoothed histogram of the scalar
return (the Metamon recipe; classification trains more stably than regression for
value functions) -- and read back as the expectation over bin centres.

The trunk + policy head are structurally identical to ``BCPolicy.net`` layers
``0/3/6``, so ``load_bc_weights`` warm-starts them from a BC checkpoint, leaving
only the value head fresh. Illegal-action masking reuses ``policy.masked_logits``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE


class ActorCritic(nn.Module):
    """Shared trunk -> (action logits, value-bin logits)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_actions: int = GEN9_ACTION_SPACE_SIZE,
        n_bins: int = 51,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, n_bins)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features)


def value_support(v_min: float, v_max: float, n_bins: int) -> torch.Tensor:
    """Bin centres for the value-classification head."""
    return torch.linspace(v_min, v_max, n_bins)


def hl_gauss_target(
    returns: torch.Tensor, centers: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Gaussian-smoothed soft label over ``centers`` for each scalar return.

    Probability of bin ``i`` is the Gaussian mass between its edges, so a return
    outside ``[v_min, v_max]`` simply piles mass onto the boundary bin after
    re-normalisation.
    """
    dz = (centers[-1] - centers[0]) / (len(centers) - 1)
    edges = torch.cat([centers - dz / 2, centers[-1:] + dz / 2])  # [K+1]
    z = (edges.unsqueeze(0) - returns.unsqueeze(1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))  # [B, K+1]
    probs = cdf[:, 1:] - cdf[:, :-1]  # [B, K]
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def value_from_logits(value_logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Scalar value estimate: expectation of the softmax over bin centres."""
    probs = torch.softmax(value_logits, dim=-1)
    return probs @ centers


_BC_TO_AC = {
    "net.0.weight": "trunk.0.weight",
    "net.0.bias": "trunk.0.bias",
    "net.3.weight": "trunk.3.weight",
    "net.3.bias": "trunk.3.bias",
    "net.6.weight": "policy_head.weight",
    "net.6.bias": "policy_head.bias",
}


def load_bc_weights(model: ActorCritic, bc_state_dict: Mapping[str, torch.Tensor]) -> None:
    """Warm-start ``model``'s trunk + policy head from a BC ``state_dict`` in place.

    The value head is left at its random init. Raises if the BC checkpoint lacks
    the expected layer keys (a shape/architecture mismatch we want to fail loudly).
    """
    try:
        remapped = {dst: bc_state_dict[src] for src, dst in _BC_TO_AC.items()}
    except KeyError as exc:
        raise ValueError(
            f"BC checkpoint missing expected key {exc}; cannot warm-start ActorCritic."
        ) from None
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys while warm-starting from BC: {unexpected}")
    if set(missing) != {"value_head.weight", "value_head.bias"}:
        raise ValueError(f"Unexpected missing keys after BC warm-start: {missing}")

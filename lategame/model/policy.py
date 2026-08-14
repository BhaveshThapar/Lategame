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


def policy_logits(
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Policy logits from a model output, dropping the value head if it returns a tuple.

    Lets the BC loop/agent consume ``BCPolicy`` (logits) and the actor-critic /
    entity-transformer (``(policy_logits, value_logits)``) interchangeably.
    """
    return out[0] if isinstance(out, tuple) else out


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


# --------------------------------------------------------------------------- #
# Factored (doubles) policy helpers -- G4/M6.
#
# A doubles turn commits BOTH active slots, so the head emits `n_slots * slot_actions` logits read
# as that many independent distributions rather than one joint one. A joint head would be
# 107^2 = 11,449 outputs; factoring is what makes a third format trainable at all, and the cost is
# that the two slots are modelled as independent when in one respect they are not (see
# `agents.doubles_agent.resolve_switch_conflict`).
# --------------------------------------------------------------------------- #
def factored_logits(logits: torch.Tensor, n_slots: int = 2) -> torch.Tensor:
    """Reshape a flat ``(B, n_slots * A)`` head into per-slot ``(B, n_slots, A)``."""
    return logits.reshape(logits.shape[0], n_slots, -1)


def factored_masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-slot masking. ``logits`` ``(B, n_slots, A)``, ``mask`` ``(B, n_slots, A)``.

    A slot with NO legal action (an empty field slot) would otherwise be all ``NEG_INF``, which is
    a uniform distribution over illegal actions rather than an error. `pass` (action 0) is always
    re-enabled for such a slot, because that is the order the server actually accepts there.
    """
    out = logits.masked_fill(~mask, NEG_INF)
    dead = ~mask.any(dim=-1)  # (B, n_slots)
    if dead.any():
        out = out.clone()
        out[..., 0] = torch.where(dead, torch.zeros_like(out[..., 0]), out[..., 0])
    return out


def factored_cross_entropy(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Summed per-slot cross-entropy. ``logits`` ``(B, S, A)``, ``action`` ``(B, S)``.

    Summed rather than averaged over slots: each slot is a separate decision the demonstrator
    made, and averaging would halve the gradient on a turn where both slots matter.
    """
    b, s, a = logits.shape
    return torch.nn.functional.cross_entropy(logits.reshape(b * s, a), action.reshape(b * s)) * s


def factored_accuracy(logits: torch.Tensor, action: torch.Tensor) -> tuple[int, int]:
    """``(turns where BOTH slots match, slot-level matches)``.

    Both are reported because they answer different questions: per-slot accuracy is the training
    signal's own scale, while the strict both-slots number is what "imitates the demonstrator's
    TURN" means -- and on a factored head the second is much harder than the first.
    """
    pred = logits.argmax(dim=-1)
    per_slot = int((pred == action).sum().item())
    both = int((pred == action).all(dim=-1).sum().item())
    return both, per_slot

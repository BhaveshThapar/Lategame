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

from rotomai.features.action_space import GEN9_ACTION_SPACE_SIZE
from rotomai.features.doubles_action_space import N_SWITCHES, SWITCH_BASE

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


def factored_cross_entropy_none(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Per-SAMPLE summed cross-entropy, shape ``(B,)`` -- the AWR weighting unit.

    AWR multiplies each sample's loss by its advantage weight, so the reduction must collapse the
    slot axis but NOT the batch axis. Summing over slots (rather than averaging) keeps a two-slot
    decision weighted like the two decisions it is.
    """
    b, s, a = logits.shape
    per = torch.nn.functional.cross_entropy(
        logits.reshape(b * s, a), action.reshape(b * s), reduction="none"
    )
    return per.reshape(b, s).sum(dim=1)


# --------------------------------------------------------------------------- #
# The factored BEHAVIOUR POLICY -- B6f.
#
# BC and AWR only ever need to SCORE a demonstrator's action, so a plain product of two
# independent per-slot distributions is enough for them. PPO also has to SAMPLE from the policy
# and then reproduce that sample's density at update time, and there the independence assumption
# stops being free: two slots may not switch to the same benched Pokemon, and `DoublesAgent`
# repaired that by rewriting slot 1 AFTER sampling (`agents.doubles_agent.resolve_switch_conflict`).
# Harmless when you are only scoring; fatal on-policy, because the action that was PLAYED is then
# not the action that was drawn, so log pi_old(a|s) is the density of something that never happened
# and the importance ratio's denominator is wrong.
#
# So the constraint moves INSIDE the distribution: draw slot 0, delete its switch from slot 1's
# legal set, draw slot 1 from what remains. That makes the policy autoregressive,
#     pi(a|s) = pi_0(a_0 | s, m_0) * pi_1(a_1 | s, m_1'),   m_1' = f(s, a_0) deterministic,
# and `log_softmax` over the restricted row IS the renormalization -- there is no separate step.
# Recomputing f at update time from the stored (mask, a_0) reproduces the behaviour density
# exactly, so the ratio is exact rather than approximately right.
#
# For `sample=False` this is provably the SAME FUNCTION as argmax + resolve_switch_conflict (both
# take slot 1's best legal action other than a_0), which is what lets the greedy eval path stay
# untouched and serve as the reference implementation. See tests/test_factored_policy.py.
# --------------------------------------------------------------------------- #
def restrict_slot1_mask(mask: torch.Tensor, action0: torch.Tensor) -> torch.Tensor:
    """``mask`` with slot 1's copy of slot 0's switch removed. ``action0`` is ``(B,)``.

    Only switches collide: two slots using the same MOVE index is ordinary double-attack play, so
    the test is the codec's own switch range rather than "the actions are equal". Returns ``mask``
    itself when no sample in the batch picked a switch, so the common case allocates nothing.
    """
    is_switch = (action0 >= SWITCH_BASE) & (action0 < SWITCH_BASE + N_SWITCHES)
    if not bool(is_switch.any()):
        return mask
    drop = torch.zeros_like(mask[:, 1])
    drop.scatter_(1, action0.clamp(0, mask.shape[-1] - 1).unsqueeze(1), True)
    out = mask.clone()
    out[:, 1] = out[:, 1] & ~(drop & is_switch.unsqueeze(1))
    return out


def factored_log_prob(log_probs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Joint ``log pi(a|s)``, shape ``(B,)``: the per-slot log-probs SUMMED.

    ``log_probs`` is ``(B, S, A)``, already masked and log-softmaxed. Identical by construction to
    ``-factored_cross_entropy_none(masked_logits, action)`` -- pinned by test, and the reason the
    PPO surrogate needs no new loss code.
    """
    return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def factored_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Summed per-slot entropy, shape ``(B,)``, matching ``factored_log_prob``'s summation.

    A slot with exactly one legal action contributes EXACTLY 0 (its masked distribution is a point
    mass), so a turn that offered no choice adds nothing here -- which is why the reported number
    is only interpretable next to the decision-row count. ``NEG_INF`` is finite on purpose: ``exp``
    underflows to 0 and ``0 * -1e9`` is 0, where a true ``-inf`` would give ``NaN``.
    """
    return -(log_probs.exp() * log_probs).sum(dim=-1).sum(dim=-1)


def factored_has_choice(mask: torch.Tensor) -> torch.Tensor:
    """``(B,)`` bool: some slot offers more than one legal action.

    The same clause as ``data.collect._is_learnable``'s ``mask.sum(axis=1).max() > 1``, in torch.
    A turn failing this has ``log pi == 0`` for every parameter setting, so it contributes exactly
    zero policy gradient -- it is kept (the critic wants the state, and GAE wants the episode
    contiguous) but must not enter the DENOMINATOR of the policy/entropy/KL means.
    """
    return mask.sum(dim=-1).max(dim=-1).values > 1


def sample_factored_action(
    logits: torch.Tensor, mask: torch.Tensor, *, sample: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a factored action pair. ``(action (B,S), log_prob (B,), restricted_mask (B,S,A))``.

    ``logits`` is ``(B, S, A)`` (already through ``factored_logits``). Both slots' log-probs are
    read from the SECOND, restricted ``log_softmax`` -- slot 0's row is mathematically unchanged by
    the restriction, but reading one tensor guarantees bit-identity with what ``train.ppo``
    recomputes, and this whole helper exists so act-time and train-time cannot drift apart.
    """
    lp_first = torch.log_softmax(factored_masked_logits(logits, mask), dim=-1)
    a0 = (
        torch.multinomial(lp_first[:, 0].exp(), 1).squeeze(1)
        if sample
        else lp_first[:, 0].argmax(dim=-1)
    )
    restricted = restrict_slot1_mask(mask, a0)
    log_probs = torch.log_softmax(factored_masked_logits(logits, restricted), dim=-1)
    a1 = (
        torch.multinomial(log_probs[:, 1].exp(), 1).squeeze(1)
        if sample
        else log_probs[:, 1].argmax(dim=-1)
    )
    action = torch.stack([a0, a1], dim=1)
    return action, factored_log_prob(log_probs, action), restricted

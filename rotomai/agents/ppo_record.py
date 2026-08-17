"""On-policy recording bookkeeping, shared by the singles and doubles PPO acting agents.

What an on-policy update needs beyond offline recording is two extra per-turn signals captured in
the SAME ``no_grad`` forward that picks the action -- ``log pi_old(a|s)`` (the importance ratio's
denominator) and ``V(s)`` (the GAE baseline) -- plus, on doubles, a flag saying whether the action
that was recorded is the action that was actually played.

Extracted into a mixin rather than duplicated because the bookkeeping is the part that drifts:
``data.rollout._learner_episodes`` recovers a battle's tag by ``id(recs)`` object identity against
``data.collect._battle_rewards``, so every parallel list here has to stay exactly as long as
``records[tag]`` and in the same order. Two hand-maintained copies of that invariant is one copy
too many.

Mixed in BEFORE a concrete agent (``PPORecordMixin, OfflineRLAgent``), so ``__init__`` chains
through to the agent that sets ``self._torch``. The mixin does not define ``choose_move``: each
agent's action path is genuinely different (one scalar draw over 26 actions vs. a sequential
factored draw over 2x107), and only the recording is shared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from poke_env.battle import AbstractBattle

from rotomai.data.reward import RewardWeights, state_value

#: ``(obs, action, mask)``. ``action`` is a scalar on singles and a ``(2,)`` array on doubles;
#: ``mask`` is ``(26,)`` or ``(2, 107)``. Deliberately not unified -- see ``data.collect``.
Record = tuple[np.ndarray, "int | np.ndarray", np.ndarray]


class PPORecordMixin:
    """Per-battle-tag parallel lists for an on-policy rollout."""

    _reward_weights: RewardWeights = RewardWeights()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.records: dict[str, list[Record]] = {}
        self.values: dict[str, list[float]] = {}
        self.log_probs: dict[str, list[float]] = {}
        self.value_estimates: dict[str, list[float]] = {}
        # Whether the recorded action is the one the server was actually sent. False whenever the
        # codec's non-strict decode fell back to a random legal order -- the row is KEPT (the
        # critic wants the state, and dropping it would splice two unrelated turns together in
        # GAE's contiguous scan) but its policy-gradient weight is zeroed downstream.
        self.valid: dict[str, list[bool]] = {}
        self.seen_battles: dict[str, AbstractBattle] = {}
        self.dropped = 0

    def _load_value_support(self, checkpoint_path: str | Path) -> None:
        """Re-read the checkpoint's value support so bin logits become a scalar baseline.

        The concrete agents load the checkpoint for their weights and discard ``v_min``/``v_max``/
        ``n_bins``; PPO needs them at action time. Called by the subclass with its OWN resolved
        path, because the two agents resolve different env vars and different defaults.
        """
        from rotomai.model.actor_critic import value_from_logits, value_support

        ckpt = self._torch.load(  # type: ignore[attr-defined]
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        self._centers = value_support(
            float(ckpt["v_min"]), float(ckpt["v_max"]), int(ckpt["n_bins"])
        )
        self._value_from_logits = value_from_logits

    def _record(
        self,
        battle: AbstractBattle,
        obs: np.ndarray,
        action: int | np.ndarray,
        mask: np.ndarray,
        log_prob: float,
        value: float,
        *,
        executed: bool = True,
    ) -> None:
        """Append one turn to every parallel list. All six must grow together or not at all."""
        tag = battle.battle_tag
        self.records.setdefault(tag, []).append((obs, action, mask))
        self.values.setdefault(tag, []).append(state_value(battle, self._reward_weights))
        self.log_probs.setdefault(tag, []).append(log_prob)
        self.value_estimates.setdefault(tag, []).append(value)
        self.valid.setdefault(tag, []).append(executed)
        self.seen_battles[tag] = battle

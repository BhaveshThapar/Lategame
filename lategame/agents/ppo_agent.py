"""On-policy PPO acting agent: records (obs, action, mask, log_prob, value) per turn.

Extends the M3 ``OfflineRLAgent`` encode -> mask -> sample path, but captures -- in
the *same* ``no_grad`` forward that picks the action -- the acting log-prob and the
critic's scalar value estimate. Those two extra signals are exactly what an on-policy
PPO update needs (the importance ratio's denominator and the GAE baseline) and which
the offline recording (``data.collect._RecordingMixin``) does not provide.

Recording mirrors that mixin's structure -- per-battle-tag lists of ``(obs, action,
mask)`` records plus the shaped ``state_value`` -- so ``data.collect._battle_rewards``
reconstructs rewards/dones verbatim, with parallel ``log_probs`` / ``value_estimates``
lists added. Sampling is mandatory (``sample=True``): PPO needs the on-policy action
distribution, not greedy arg-max.

Torch and the torch-ful value helpers are imported lazily (inside ``__init__`` /
``choose_move``) so importing this module -- which ``eval.arena`` does to register the
agent -- stays torch-free, matching ``OfflineRLAgent`` and the M0/M1 import contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder

from lategame.agents.offline_rl_agent import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT,
    OfflineRLAgent,
)
from lategame.data.reward import RewardWeights, state_value
from lategame.features.action_space import action_mask, action_to_order
from lategame.features.encoder import embed_battle

_Record = tuple[np.ndarray, int, np.ndarray]


class PPORecordingAgent(OfflineRLAgent):
    """``OfflineRLAgent`` that records log-prob + value for on-policy PPO rollouts."""

    _reward_weights: RewardWeights = RewardWeights()

    def __init__(
        self,
        *args: object,
        checkpoint_path: str | Path | None = None,
        sample: bool = True,
        loop_penalty: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            checkpoint_path=checkpoint_path,
            sample=sample,
            loop_penalty=loop_penalty,
            **kwargs,
        )

        from lategame.model.actor_critic import value_from_logits, value_support

        # The base agent loads the checkpoint but discards the value support; re-read
        # it so we can turn value-bin logits into a scalar baseline at action time.
        path = Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        ckpt = self._torch.load(path, map_location="cpu", weights_only=False)
        self._centers = value_support(
            float(ckpt["v_min"]), float(ckpt["v_max"]), int(ckpt["n_bins"])
        )
        self._value_from_logits = value_from_logits

        # Per-battle-tag recording, parallel lists (see module docstring).
        self.records: dict[str, list[_Record]] = {}
        self.values: dict[str, list[float]] = {}
        self.log_probs: dict[str, list[float]] = {}
        self.value_estimates: dict[str, list[float]] = {}
        self.seen_battles: dict[str, AbstractBattle] = {}
        self.dropped = 0

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not battle.available_moves and not battle.available_switches:
            return self.choose_default_move()

        torch = self._torch
        try:
            obs_np = embed_battle(battle)
            mask_np = action_mask(battle)
        except Exception:
            self.dropped += 1
            return self.choose_default_move()

        obs = torch.from_numpy(obs_np).float().unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        with torch.no_grad():
            logits, value_logits = self._model(obs)
            log_probs = torch.log_softmax(self._masked_logits(logits, mask), dim=1)
            action = int(torch.multinomial(log_probs.exp(), 1).item())
            old_log_prob = float(log_probs[0, action].item())
            value = float(self._value_from_logits(value_logits, self._centers).item())

        try:
            order = action_to_order(action, battle)
        except Exception:
            self.dropped += 1
            return self.choose_default_move()

        tag = battle.battle_tag
        self.records.setdefault(tag, []).append((obs_np, action, mask_np))
        self.values.setdefault(tag, []).append(state_value(battle, self._reward_weights))
        self.log_probs.setdefault(tag, []).append(old_log_prob)
        self.value_estimates.setdefault(tag, []).append(value)
        self.seen_battles[tag] = battle
        return order

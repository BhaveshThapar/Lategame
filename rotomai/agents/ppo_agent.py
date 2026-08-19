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

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder

from rotomai.agents.history_input import observe
from rotomai.agents.offline_rl_agent import (
    CHECKPOINT_ENV_VAR,
    DEFAULT_CHECKPOINT,
    OfflineRLAgent,
)
from rotomai.agents.ppo_record import PPORecordMixin
from rotomai.features.action_space import action_mask, action_to_order
from rotomai.features.encoder import embed_battle


class PPORecordingAgent(PPORecordMixin, OfflineRLAgent):
    """``OfflineRLAgent`` that records log-prob + value for on-policy PPO rollouts."""

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
        self._load_value_support(
            Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        )

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

        # The WINDOW, not the turn: what the model consumed is what the PPO update has to
        # recompute log-probs on, or the ratio is between two different functions. The rollout
        # buffer stacks on dim 0, so a [K+1, D] row flows through as [B, K+1, D] unchanged.
        obs_np = observe(self._frames, battle, obs_np)
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

        self._record(battle, obs_np, action, mask_np, old_log_prob, value)
        return order

"""On-policy PPO acting agent for DOUBLES: a factored per-slot draw, recorded exactly (B6f).

The doubles counterpart of ``agents.ppo_agent``. It inherits ``DoublesAgent``'s checkpoint
fingerprint check (``d1-``/888) and model build, and ``PPORecordMixin``'s parallel-list
bookkeeping, and replaces only the action path -- which is where doubles differs and where the
on-policy requirements bite.

THREE THINGS THIS DOES THAT ``DoublesAgent`` DOES NOT, each because PPO has to reproduce the
behaviour density at update time and the eval path does not:

1. **One masking function, in torch.** ``DoublesAgent`` re-derives the masked distribution in
   numpy with ``-np.inf``; training uses ``model.policy.factored_masked_logits`` with a finite
   ``NEG_INF``. Those agree numerically today only because ``exp(-1e9)`` underflows to zero --
   agreement by coincidence, on exactly the forced-replacement turns doubles is full of. Here the
   distribution comes from the same function ``train.ppo._policy_stats`` calls.

2. **The joint-switch constraint is inside the draw, not a repair after it.**
   ``resolve_switch_conflict`` rewrites slot 1 post-hoc, so the played action is not the drawn one
   and ``log pi_old(a|s)`` describes something that never happened. ``sample_factored_action``
   deletes slot 0's switch from slot 1's legal set BEFORE drawing slot 1, which is exact and
   (for argmax) provably the same function as the repair.

3. **The decode is checked.** ``action_to_order`` falls back to a random legal order on a decode
   failure, silently. The legality test cannot catch that -- the action WAS legal under its mask;
   it is the joint decode that failed -- so the flag is recorded and PPO zero-weights the row.
"""

from __future__ import annotations

import os
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, ForfeitBattleOrder

from lategame.agents.doubles_agent import CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT, DoublesAgent
from lategame.agents.ppo_record import PPORecordMixin
from lategame.features.doubles_action_space import (
    action_mask,
    action_to_order_checked,
    slot_actions,
)
from lategame.features.doubles_encoder import embed_doubles_battle


class DoublesPPORecordingAgent(PPORecordMixin, DoublesAgent):
    """``DoublesAgent`` that samples a factored pair and records its exact joint log-prob."""

    def __init__(
        self,
        *args: object,
        checkpoint_path: str | Path | None = None,
        sample: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, checkpoint_path=checkpoint_path, sample=sample, **kwargs)
        self._load_value_support(
            Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        from lategame.model.policy import factored_logits, sample_factored_action

        torch = self._torch
        if self._turn_cap.hit(battle):
            return ForfeitBattleOrder()

        try:
            obs_np = embed_doubles_battle(battle)
            mask_np = action_mask(battle)
        except Exception:
            self.dropped += 1
            return self.choose_default_move()
        if not mask_np.any():
            self.dropped += 1
            return self.choose_default_move()

        n = slot_actions(battle)
        obs = torch.from_numpy(obs_np).float().unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        with torch.no_grad():
            flat, value_logits = self._model(obs)
            logits = factored_logits(flat[:, : 2 * n], 2)
            pen = self._loop_guard.penalty_vector(battle)
            if pen.any():
                logits = logits - torch.from_numpy(pen[:, :n]).to(logits.dtype).unsqueeze(0)
            action, log_prob, _ = sample_factored_action(logits, mask, sample=self._sample)
            value = float(self._value_from_logits(value_logits, self._centers).item())

        action_np = action[0].numpy()
        order, executed = action_to_order_checked(action_np, battle)
        # The SAMPLED action and the ORIGINAL mask are what get recorded -- not the fallback
        # order's label (whose density is the fallback's, not the policy's), and not the
        # restricted mask (which train.ppo recomputes from `mask` and `action[:, 0]`, so storing
        # it would create a second definition that can drift).
        self._loop_guard.record(battle, action_np)
        self._record(
            battle,
            obs_np,
            action_np,
            mask_np,
            float(log_prob.item()),
            value,
            executed=executed,
        )
        return order

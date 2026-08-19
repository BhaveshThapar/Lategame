"""M3 offline-RL agent: plays from the actor-critic policy head.

Loads an ``actor_critic`` checkpoint produced by ``train.offline_rl`` and plays
exactly like the BC agent -- encode, mask, pick from the action logits -- using
the policy head; the value head is unused at inference in M3 (it powers search /
leaf evaluation only in M7). Torch is imported lazily so registering this class
keeps the M0/M1 commands torch-free.
"""

from __future__ import annotations

import os
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, Player

from rotomai.agents.history_input import _history_tracker, observe
from rotomai.features.action_space import action_mask, action_to_order
from rotomai.features.encoder import OBS_DIM, OBS_VERSION, embed_battle

DEFAULT_CHECKPOINT = "checkpoints/offrl_gen9randombattle.pt"
CHECKPOINT_ENV_VAR = "ROTOMAI_OFFRL_CHECKPOINT"


class OfflineRLAgent(Player):
    def __init__(
        self,
        *args: object,
        checkpoint_path: str | Path | None = None,
        sample: bool = False,
        loop_penalty: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        import torch

        from rotomai.agents.loop_guard import LoopGuard
        from rotomai.model.factory import build_model
        from rotomai.model.policy import masked_logits

        path = Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        if not path.exists():
            raise FileNotFoundError(
                f"Offline-RL checkpoint not found at '{path}'. Train one first "
                f"(`rotomai train-rl`) or set {CHECKPOINT_ENV_VAR}."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("obs_version") != OBS_VERSION or ckpt.get("input_dim") != OBS_DIM:
            raise ValueError(
                f"Checkpoint encoder mismatch (ckpt {ckpt.get('obs_version')}/"
                f"{ckpt.get('input_dim')} vs encoder {OBS_VERSION}/{OBS_DIM}). Retrain."
            )

        model = build_model(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        self._torch = torch
        self._model = model
        self._masked_logits = masked_logits
        self._sample = sample
        self._loop_guard = LoopGuard(loop_penalty)
        self._frames = _history_tracker(model, OBS_DIM)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not battle.available_moves and not battle.available_switches:
            return self.choose_default_move()

        torch = self._torch
        obs = torch.from_numpy(observe(self._frames, battle, embed_battle(battle))).float()
        obs = obs.unsqueeze(0)
        mask = torch.from_numpy(action_mask(battle)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._model(obs)
            logits = self._masked_logits(logits, mask)
            pen = self._loop_guard.penalty_vector(battle)
            if pen.any():
                logits = logits - torch.from_numpy(pen).to(logits.dtype).unsqueeze(0)
            if self._sample:
                action = int(torch.multinomial(torch.softmax(logits, dim=1), 1).item())
            else:
                action = int(logits.argmax(dim=1).item())
        self._loop_guard.record(battle, action)
        return action_to_order(action, battle)

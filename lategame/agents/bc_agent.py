"""M2 behaviour-cloning agent: a trained policy that plays full battles.

Loads a checkpoint produced by ``train.bc``, encodes each battle with the same
``features.encoder`` used at training time, masks illegal actions, and converts
the chosen action back to a ``BattleOrder`` via the shared action codec.

Torch is imported *lazily* (inside ``__init__``) so that merely registering this
class in the agent registry -- and therefore running the torch-free M0/M1
commands -- never requires the optional ``[ml]`` extra. Only actually
instantiating a ``bc`` agent pulls in torch.
"""

from __future__ import annotations

import os
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, Player

from lategame.features.action_space import action_mask, action_to_order
from lategame.features.encoder import OBS_DIM, OBS_VERSION, embed_battle

DEFAULT_CHECKPOINT = "checkpoints/bc_gen9randombattle.pt"
CHECKPOINT_ENV_VAR = "LATEGAME_BC_CHECKPOINT"


class BCAgent(Player):
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
        from torch import nn

        from lategame.agents.loop_guard import LoopGuard
        from lategame.model.policy import masked_logits, policy_logits

        path = Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        if not path.exists():
            raise FileNotFoundError(
                f"BC checkpoint not found at '{path}'. Train one first "
                f"(`lategame train`) or set {CHECKPOINT_ENV_VAR}."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("obs_version") != OBS_VERSION or ckpt.get("input_dim") != OBS_DIM:
            raise ValueError(
                f"Checkpoint encoder mismatch (ckpt {ckpt.get('obs_version')}/"
                f"{ckpt.get('input_dim')} vs encoder {OBS_VERSION}/{OBS_DIM}). Retrain."
            )

        model: nn.Module
        if ckpt.get("model_type", "bc_policy") == "bc_policy":
            from lategame.model.policy import BCPolicy

            model = BCPolicy(
                ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"], n_actions=ckpt["n_actions"]
            )
        else:
            from lategame.model.factory import build_model

            model = build_model(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        self._torch = torch
        self._model = model
        self._masked_logits = masked_logits
        self._policy_logits = policy_logits
        self._sample = sample
        self._loop_guard = LoopGuard(loop_penalty)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not battle.available_moves and not battle.available_switches:
            return self.choose_default_move()

        torch = self._torch
        obs = torch.from_numpy(embed_battle(battle)).float().unsqueeze(0)
        mask = torch.from_numpy(action_mask(battle)).unsqueeze(0)
        with torch.no_grad():
            logits = self._masked_logits(self._policy_logits(self._model(obs)), mask)
            pen = self._loop_guard.penalty_vector(battle)
            if pen.any():
                logits = logits - torch.from_numpy(pen).to(logits.dtype).unsqueeze(0)
            if self._sample:
                action = int(torch.multinomial(torch.softmax(logits, dim=1), 1).item())
            else:
                action = int(logits.argmax(dim=1).item())
        self._loop_guard.record(battle, action)
        return action_to_order(action, battle)

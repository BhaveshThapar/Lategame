"""Learned doubles agent: a FACTORED per-slot policy over a 2 x 107 action space (G4 / M6).

Mirrors ``OfflineRLAgent`` -- encode, mask, pick -- with the three things doubles forces:

* **The head is factored.** The model still emits a flat vector, but it is read as two independent
  107-way distributions (one per active slot) rather than one 214-way one. That is why no model
  change was needed for a third format: ``model.factory.build_model`` already takes ``n_actions``
  from checkpoint metadata, so the doubles head is the singles architecture with a different width.
* **The mask is per slot**, from ``doubles_action_space.action_mask`` -- shape ``(2, 107)``.
* **The two slots are not independent, in exactly one way.** Both may not switch to the same
  benched Pokemon; a factored mask cannot express that (it couples the slots), and Showdown answers
  such an order with a default move rather than an error -- a silently lost turn. So the conflict
  is resolved explicitly after sampling, by re-picking the SECOND slot from its next-best legal
  action. Slot 0 keeps its choice because re-picking both could oscillate.

Torch is imported lazily, so registering this class keeps the M0/M1 commands torch-free.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, ForfeitBattleOrder, Player

from lategame.agents.loop_guard import DoublesLoopGuard
from lategame.agents.turn_cap import TurnCap
from lategame.features.doubles_action_space import (
    N_SWITCHES,
    SWITCH_BASE,
    action_mask,
    action_to_order,
    joint_switch_conflict,
    slot_actions,
)
from lategame.features.doubles_encoder import (
    OBS_DIM_DOUBLES,
    OBS_VERSION_DOUBLES,
    embed_doubles_battle,
)

DEFAULT_CHECKPOINT = "checkpoints/doubles_gen9vgc.pt"
CHECKPOINT_ENV_VAR = "LATEGAME_DOUBLES_CHECKPOINT"


def resolve_switch_conflict(
    logits: np.ndarray, mask: np.ndarray, action: np.ndarray
) -> np.ndarray:
    """Re-pick slot 1 when both slots chose the same switch, else return ``action`` unchanged.

    Falls back to slot 1's next-best LEGAL action, and to a pass if it has no other. Only slot 1
    moves: re-picking both slots could cycle between two conflicting choices forever.

    ONLY SWITCHES COLLIDE. Two slots may use the same move INDEX -- both actives using their own
    move 1 is ordinary play -- so the conflict test is the codec's own `_SWITCH_RANGE` one rather
    than "the two actions are equal", which would silently rewrite a perfectly legal double attack.
    """
    if not (SWITCH_BASE <= int(action[0]) < SWITCH_BASE + N_SWITCHES and action[0] == action[1]):
        return action
    ranked = np.argsort(-logits[1])
    for candidate in ranked:
        if mask[1, candidate] and int(candidate) != int(action[0]):
            action = action.copy()
            action[1] = int(candidate)
            return action
    action = action.copy()
    action[1] = 0  # pass -- legal for a slot with nothing else to do
    return action


class DoublesAgent(Player):
    def __init__(
        self,
        *args: object,
        checkpoint_path: str | Path | None = None,
        sample: bool = False,
        loop_penalty: float = 0.0,
        max_battle_turns: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        import torch

        from lategame.model.factory import build_model

        path = Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        if not path.exists():
            raise FileNotFoundError(
                f"Doubles checkpoint not found at '{path}'. Train one first or set "
                f"{CHECKPOINT_ENV_VAR}."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if (
            ckpt.get("obs_version") != OBS_VERSION_DOUBLES
            or ckpt.get("input_dim") != OBS_DIM_DOUBLES
        ):
            raise ValueError(
                f"Checkpoint encoder mismatch (ckpt {ckpt.get('obs_version')}/"
                f"{ckpt.get('input_dim')} vs doubles encoder {OBS_VERSION_DOUBLES}/"
                f"{OBS_DIM_DOUBLES}). Retrain."
            )

        model = build_model(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        self._torch = torch
        self._model = model
        self._sample = sample
        self._loop_guard = DoublesLoopGuard(loop_penalty)
        self._turn_cap = TurnCap(max_battle_turns)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        torch = self._torch
        # Checked before anything else: past the ceiling this battle is a loop, and every further
        # decision is another wasted request rather than a game state worth encoding.
        if self._turn_cap.hit(battle):
            return ForfeitBattleOrder()
        n = slot_actions(battle)
        mask = action_mask(battle)
        if not mask.any():
            return self.choose_default_move()

        obs = torch.from_numpy(embed_doubles_battle(battle)).float().unsqueeze(0)
        with torch.no_grad():
            flat, _ = self._model(obs)
        logits = flat.squeeze(0)[: 2 * n].reshape(2, n).numpy()
        # Mask BEFORE choosing, per slot. -inf rather than a large negative so a fully-masked row
        # cannot be picked by arithmetic accident.
        logits = np.where(mask, logits, -np.inf)
        # Subtracted AFTER masking, as singles does: -inf - penalty is still -inf, so the guard
        # can never make an illegal action reachable or a legal one unreachable.
        pen = self._loop_guard.penalty_vector(battle)
        if pen.any():
            logits = logits - pen[:, :n]

        action = np.zeros(2, dtype=np.int64)
        for slot in (0, 1):
            row = logits[slot]
            if not np.isfinite(row).any():
                action[slot] = 0  # pass: this slot has no legal action
                continue
            if self._sample:
                probs = np.exp(row - row.max())
                probs[~np.isfinite(row)] = 0.0
                probs = probs / probs.sum()
                # Sampled through torch's GLOBAL rng, not a fresh `np.random.default_rng()` per
                # decision. An unseeded generator built at every call made `--seed` a no-op for
                # every doubles rollout and league opponent -- nothing about a doubles run was
                # reproducible -- which a three-seed protocol cannot tolerate. The distribution is
                # unchanged; only the stream is. Singles already samples through torch.
                action[slot] = int(torch.multinomial(torch.from_numpy(probs).float(), 1).item())
            else:
                action[slot] = int(np.argmax(row))

        if joint_switch_conflict(action, battle):
            action = resolve_switch_conflict(logits, mask, action)
        self._loop_guard.record(battle, action)
        return action_to_order(action, battle)

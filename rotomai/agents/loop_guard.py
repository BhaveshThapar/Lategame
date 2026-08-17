"""Build 14 decision-time anti-repetition (loop guard), singles and doubles.

Builds 8-13 isolated the live switch loop: pp drives a high switch mass (the agent
won't attack), and ``P(return|switch)`` is pp-invariant (~0.50) -- the bounce-back is
a structural coin flip the encoder cannot robustify away, and the persisting rate is a
*sticky-argmax* artifact. A first cut penalized only the switch-*back* action; live it
merely converted the tight A->B->A into a longer roster-cycle (A->B->C->D...) via
fresh-mon escape -- switch mass stayed flat, win rate 0. So the guard pressures the
streak itself: it subtracts an escalating penalty from *every* voluntary switch once a
consecutive-switch run forms, pushing the argmax toward attacking.

Soft, not a mask: a finite penalty can never make the only legal action unreachable
(no forced-switch hang), and it leaves attacks (``action >= 6``) untouched, so the
escalating penalty pushes the argmax toward attacking rather than into a longer switch
run. ``penalty = 0`` is exact identity.

Voluntary switches are defined exactly as the behaviour probe: ``action < 6`` while
``not battle.force_switch``. The run resets on any attack, matching how
``scripts/behavior_probe._max_switch_run`` counts consecutive voluntary switches.

``DoublesLoopGuard`` is the same idea in the per-slot layout, and it exists because the guard
was never ported when doubles arrived. Measured on ``data/vgc_rl.npz`` (the shard B6e's AWR was
trained on): the longest episode ran **12,795 recorded turns over 7 unique observation vectors**,
the top 10 of 899 episodes held 33.7% of the shard and the top 100 held 94.2%, against an OU
shard whose top 10 hold 1.4%. 96.6% of its rewards were exactly zero. The singles guard could
not fire for two independent reasons: ``eval.arena._LOOP_GUARD_AGENTS`` does not contain
``doubles``, and ``LoopGuard.penalty_vector`` returns a 26-wide vector penalizing indices 0-5,
while the doubles layout is per slot and reads ``0`` as *pass* and ``1-6`` as the switches.

NOT SUFFICIENT ALONE, and the reason is in the data: the dominant stuck state has slot 0's only
legal action = ``pass`` and slot 1's = two switches. Every legal option there IS a switch, so a
soft penalty has nothing to push the argmax toward. The guard shortens voluntary switch runs; a
hard per-battle turn cap (``agents.turn_cap``) is what actually terminates that state.
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import AbstractBattle

from rotomai.features.action_space import action_space_size
from rotomai.features.doubles_action_space import N_SWITCHES, SWITCH_BASE, slot_actions

_N_SWITCHES = 6  # action slots 0-5 are switches; 6+ are moves/gimmicks


class LoopGuard:
    """Per-player, per-battle consecutive-voluntary-switch tracker + streak penalty.

    ``penalty`` is the per-step logit penalty; the applied amount escalates as
    ``penalty * max(0, run - free_switches)`` where ``run`` is the count of consecutive
    voluntary switches so far this battle. With ``free_switches = 1`` a single scout
    switch and a double-switch pivot stay free (``run`` 0/1 -> factor 0); the penalty
    first bites on the third consecutive switch (``run == 2``, where the 2-cycle forms)
    and escalates until the sticky argmax flips to an attack.
    """

    def __init__(self, penalty: float, free_switches: int = 1) -> None:
        self.penalty = float(penalty)
        self.free_switches = free_switches
        self._run: dict[str, int] = {}

    def penalty_vector(self, battle: AbstractBattle) -> np.ndarray:
        """Penalty (>= 0) to *subtract* from each action's logit; zeros when idle."""
        pen = np.zeros(action_space_size(battle), dtype=np.float32)
        if self.penalty == 0.0 or bool(battle.force_switch):
            return pen
        factor = max(0, self._run.get(str(battle.battle_tag), 0) - self.free_switches)
        if factor == 0:
            return pen
        pen[: min(_N_SWITCHES, pen.size)] = self.penalty * factor
        return pen

    def record(self, battle: AbstractBattle, action: int) -> None:
        """Update state with the chosen action. Forced switches are skipped (as the
        probe does); an attack resets the run; a voluntary switch extends it."""
        if bool(battle.force_switch):
            return
        tag = str(battle.battle_tag)
        self._run[tag] = self._run.get(tag, 0) + 1 if action < _N_SWITCHES else 0


def _force_switch_pair(battle: AbstractBattle) -> list[bool]:
    """``battle.force_switch`` as a two-slot list, whatever poke-env handed us.

    A ``DoubleBattle`` carries a list; the singles attribute is a bare bool. Same defensive
    read as ``agents.heuristic_agent.doubles_pick``.
    """
    force = battle.force_switch
    if isinstance(force, list):
        return [bool(f) for f in (list(force) + [False, False])[:2]]
    return [bool(force), bool(force)]


class DoublesLoopGuard:
    """Per-slot consecutive-voluntary-switch tracker + streak penalty, per battle.

    Same contract as ``LoopGuard`` -- ``penalty_vector`` returns a non-negative amount to
    SUBTRACT from logits, ``record`` advances the streak -- in the factored ``(2, slot_actions)``
    shape. The two slots keep INDEPENDENT runs: they are separate decisions, and merging them
    would let one slot's attack clear the other slot's loop.

    Penalty lands on the switch indices of the offending slot only (``1..6``); ``pass`` (0) and
    every move (7+) are untouched, so the escalating penalty pushes that slot's argmax toward
    attacking rather than into a longer switch run -- the singles rationale, in the layout
    doubles actually uses. ``penalty == 0.0`` is exact identity.
    """

    def __init__(self, penalty: float, free_switches: int = 1) -> None:
        self.penalty = float(penalty)
        self.free_switches = free_switches
        self._run: dict[tuple[str, int], int] = {}

    def penalty_vector(self, battle: AbstractBattle) -> np.ndarray:
        """Penalty (>= 0) to subtract from each slot's logits; zeros when idle."""
        n = slot_actions(battle)
        pen = np.zeros((2, n), dtype=np.float32)
        if self.penalty == 0.0:
            return pen
        tag = str(battle.battle_tag)
        force = _force_switch_pair(battle)
        hi = min(SWITCH_BASE + N_SWITCHES, n)
        for slot in (0, 1):
            if force[slot]:
                continue  # a forced replacement is not a voluntary switch; never penalize it
            factor = max(0, self._run.get((tag, slot), 0) - self.free_switches)
            if factor:
                pen[slot, SWITCH_BASE:hi] = self.penalty * factor
        return pen

    def record(self, battle: AbstractBattle, action: np.ndarray) -> None:
        """Update both slots' runs. A forced slot is skipped; a move resets; a switch extends."""
        tag = str(battle.battle_tag)
        force = _force_switch_pair(battle)
        for slot in (0, 1):
            if force[slot]:
                continue
            a = int(action[slot])
            switched = SWITCH_BASE <= a < SWITCH_BASE + N_SWITCHES
            key = (tag, slot)
            self._run[key] = self._run.get(key, 0) + 1 if switched else 0

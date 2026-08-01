"""Build 14 decision-time anti-repetition (loop guard).

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
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import AbstractBattle

from lategame.features.action_space import action_space_size

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

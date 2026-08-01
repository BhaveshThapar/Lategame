"""M1 rule-based baseline agent.

A damage/STAB/type-aware policy: each turn it picks the highest expected-damage
move (via the R-CALC seed in ``engine.damage``), and switches out of a clearly
bad type matchup when a benched Pokemon is meaningfully better. It is the fixed
baseline that later learned agents (BC -> offline RL -> self-play) are measured
against -- see plan.md, milestone M1.

Switching is intentionally conservative: tempo loss from over-switching is a
classic failure mode, so we only switch out proactively when our best available
move is weak AND a bench Pokemon offers a clearly better matchup.
"""

from __future__ import annotations

from collections.abc import Sequence

from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.player import BattleOrder, Player

from lategame.engine.damage import score_move

# A move scoring at or below this expected-power proxy counts as "weak" (e.g. a
# resisted 80 BP STAB move scores 80 * 1.5 * 0.5 = 60), which makes us consider
# switching. Matchup is an (offense - defense) type-effectiveness differential;
# we only switch if a bench mon improves it by at least this margin.
WEAK_MOVE_SCORE = 60.0
SWITCH_MARGIN = 1.0

# The decision a heuristic turn resolves to: pick a move, switch to a bench mon, or
# no preference (caller falls back to the default move). ``str`` tag keeps it JSON-cheap
# and easy for the white-box opponent model (Lever 14) to map onto a driver choice.
HeuristicPick = tuple[str, Move] | tuple[str, Pokemon] | None


def matchup(mon: Pokemon | None, opponent: Pokemon | None) -> float:
    """Type-based (offense - defense) differential of ``mon`` vs ``opponent``.

    Offense: how hard ``mon``'s STAB types hit ``opponent``.
    Defense: how hard ``opponent``'s STAB types hit ``mon`` (penalty).
    """
    if mon is None or opponent is None:
        return 0.0
    offense = max(
        (opponent.damage_multiplier(t) for t in mon.types if t is not None),
        default=1.0,
    )
    defense = max(
        (mon.damage_multiplier(t) for t in opponent.types if t is not None),
        default=1.0,
    )
    return offense - defense


def heuristic_pick(
    active: Pokemon | None,
    opponent: Pokemon | None,
    available_moves: Sequence[Move],
    available_switches: Sequence[Pokemon],
) -> HeuristicPick:
    """The pure R-CALC decision rule, shared by ``HeuristicAgent`` and the Lever 14
    white-box opponent model.

    Picks the highest expected-damage move; switches out only when the best move is weak
    AND a bench mon meaningfully improves the type matchup (conservative tempo). Returns
    ``("move", Move)`` / ``("switch", Pokemon)`` / ``None`` (no decision -> default).
    Kept a free function of *raw* poke-env objects so the opponent model can evaluate it on
    the modeled opponent's active/legal set without a live ``Player`` or a second battle POV.
    """
    if available_moves:
        best_move = max(available_moves, key=lambda m: score_move(m, active, opponent))
        best_score = score_move(best_move, active, opponent)
        if available_switches and best_score <= WEAK_MOVE_SCORE:
            switch = max(available_switches, key=lambda m: matchup(m, opponent))
            if matchup(switch, opponent) > matchup(active, opponent) + SWITCH_MARGIN:
                return ("switch", switch)
        return ("move", best_move)

    if available_switches:
        return ("switch", max(available_switches, key=lambda m: matchup(m, opponent)))

    return None


class HeuristicAgent(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        pick = heuristic_pick(
            battle.active_pokemon,
            battle.opponent_active_pokemon,
            battle.available_moves,
            battle.available_switches,
        )
        if pick is None:
            return self.choose_default_move()
        return self.create_order(pick[1])

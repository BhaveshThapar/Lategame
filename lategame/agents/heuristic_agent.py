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

from poke_env.battle import AbstractBattle, Pokemon
from poke_env.player import BattleOrder, Player

from lategame.engine.damage import score_move

# A move scoring at or below this expected-power proxy counts as "weak" (e.g. a
# resisted 80 BP STAB move scores 80 * 1.5 * 0.5 = 60), which makes us consider
# switching. Matchup is an (offense - defense) type-effectiveness differential;
# we only switch if a bench mon improves it by at least this margin.
WEAK_MOVE_SCORE = 60.0
SWITCH_MARGIN = 1.0


class HeuristicAgent(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        opponent = battle.opponent_active_pokemon

        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda m: score_move(m, battle.active_pokemon, opponent),
            )
            best_score = score_move(best_move, battle.active_pokemon, opponent)

            if battle.available_switches and best_score <= WEAK_MOVE_SCORE:
                switch = self._best_switch(battle)
                if switch is not None and self._matchup(switch, opponent) > (
                    self._matchup(battle.active_pokemon, opponent) + SWITCH_MARGIN
                ):
                    return self.create_order(switch)

            return self.create_order(best_move)

        if battle.available_switches:
            switch = self._best_switch(battle)
            if switch is not None:
                return self.create_order(switch)

        return self.choose_default_move()

    def _best_switch(self, battle: AbstractBattle) -> Pokemon | None:
        opponent = battle.opponent_active_pokemon
        if not battle.available_switches:
            return None
        return max(battle.available_switches, key=lambda m: self._matchup(m, opponent))

    @staticmethod
    def _matchup(mon: Pokemon, opponent: Pokemon | None) -> float:
        """Type-based (offense - defense) differential of ``mon`` vs ``opponent``.

        Offense: how hard ``mon``'s STAB types hit ``opponent``.
        Defense: how hard ``opponent``'s STAB types hit ``mon`` (penalty).
        """
        if opponent is None:
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

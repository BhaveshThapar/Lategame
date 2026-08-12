"""The M1 baseline on doubles, which is what the VGC ceiling probe has to be anchored on.

poke-env supplies exactly ONE competent doubles bot (`SimpleHeuristicsPlayer`), and the published
RB and OU bands differ only at the TOP of the skill gradient -- both formats crush naive bots
equally (random 0.007/0.030, maxbasepower 0.107/0.060) and diverge only at
simpleheuristics-vs-heuristic (0.523 vs 0.643). So a ceiling probe that can only measure below the
strongest available bot cannot tell FORMAT_BOUND from MODEL_BOUND, and a second competent doubles
agent is a prerequisite for the measurement rather than a nicety.

Doubles forces two things singles never did, and both are silent failures if got wrong: a move
needs a legal TARGET, and the two slots may not switch to the same benched Pokemon.
"""

from __future__ import annotations

from poke_env.battle import Move, Pokemon, Status
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player import DefaultBattleOrder, DoubleBattleOrder

from lategame.agents.heuristic_agent import HeuristicAgent, best_target


class _FakeDoubleBattle(DoubleBattle):
    """Enough of poke-env's DoubleBattle for the decision path.

    A real subclass, so the `isinstance(battle, DoubleBattle)` branch in `choose_move` is the one
    under test -- `__init__` is deliberately not called (it wants a websocket and a battle tag) and
    the four properties the decision reads are overridden instead.
    """

    def __init__(self, active, opponent_active, moves, switches, targets=None):
        self._a, self._oa, self._m, self._s = active, opponent_active, moves, switches
        self._targets = targets or {}

    @property
    def active_pokemon(self):
        return self._a

    @property
    def opponent_active_pokemon(self):
        return self._oa

    @property
    def available_moves(self):
        return self._m

    @property
    def available_switches(self):
        return self._s

    def get_possible_showdown_targets(self, move, pokemon, dynamax=False):
        return self._targets.get(move.id, [1, 2])


def _decide(battle) -> object:
    # __new__ so no account/server is needed; choose_move touches neither.
    return HeuristicAgent.choose_move(HeuristicAgent.__new__(HeuristicAgent), battle)


# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #
def test_the_target_is_the_foe_the_best_move_hits_hardest():
    """A move's expected damage depends on WHICH foe it lands on, so on doubles the move and the
    target have to be chosen together rather than the move first."""
    chomp = Pokemon(9, species="garchomp")
    foes = [(0, Pokemon(9, species="tangrowth")), (1, Pokemon(9, species="pikachu"))]
    # Earthquake is 2x on pikachu (ground) and 0.5x on tangrowth (grass).
    slot, foe = best_target(chomp, foes, [Move("earthquake", 9)])
    assert (slot, foe.species) == (1, "pikachu")


def test_with_no_moves_the_target_is_the_worst_matchup_for_us():
    """A switch decision must be evaluated against the foe we are losing to, not any foe."""
    pika = Pokemon(9, species="pikachu")
    foes = [(0, Pokemon(9, species="garchomp")), (1, Pokemon(9, species="magikarp"))]
    slot, foe = best_target(pika, foes, [])
    assert foe.species == "garchomp"


def test_no_live_foe_yields_no_target():
    assert best_target(Pokemon(9, species="garchomp"), [], [Move("earthquake", 9)]) is None


# --------------------------------------------------------------------------- #
# The joined order
# --------------------------------------------------------------------------- #
def test_both_slots_decide_and_the_result_is_a_double_order():
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), Pokemon(9, species="arcanine")],
        opponent_active=[Pokemon(9, species="pikachu"), Pokemon(9, species="tangrowth")],
        moves=[[Move("earthquake", 9)], [Move("flamethrower", 9)]],
        switches=[[], []],
    )
    order = _decide(battle)
    assert isinstance(order, DoubleBattleOrder)
    assert not isinstance(order.first_order, DefaultBattleOrder)
    assert not isinstance(order.second_order, DefaultBattleOrder)


def test_a_move_target_showdown_would_reject_is_never_sent():
    """Spread and self-targeting moves have fixed target sets. Guessing a target the simulator
    rejects costs the turn, so the preferred slot is only used when it is in the legal list."""
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[Pokemon(9, species="pikachu"), Pokemon(9, species="tangrowth")],
        moves=[[Move("earthquake", 9)], []],
        switches=[[], []],
        targets={"earthquake": [0]},  # spread move: "no target" is its only legal value
    )
    order = _decide(battle)
    assert order.first_order.move_target == 0


def test_the_preferred_target_is_used_when_it_is_legal():
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[Pokemon(9, species="tangrowth"), Pokemon(9, species="pikachu")],
        moves=[[Move("earthquake", 9)], []],
        switches=[[], []],
        targets={"earthquake": [1, 2]},
    )
    order = _decide(battle)
    assert order.first_order.move_target == 2  # opponent slot 1 -> showdown target 2


def test_the_two_slots_never_switch_to_the_same_bench_mon():
    """An order that switches both slots to one Pokemon is illegal, and the server answers it by
    playing a default move -- so this is a silent turn loss, not an error."""
    bench = [Pokemon(9, species="tangrowth"), Pokemon(9, species="arcanine")]
    battle = _FakeDoubleBattle(
        # Two weak attackers that both want out, against a foe they cannot hurt.
        active=[Pokemon(9, species="magikarp"), Pokemon(9, species="magikarp")],
        opponent_active=[Pokemon(9, species="garchomp"), None],
        moves=[[], []],  # no moves at all -> both slots must switch
        switches=[bench, bench],
    )
    order = _decide(battle)
    first, second = order.first_order, order.second_order
    assert not isinstance(first, DefaultBattleOrder)
    assert not isinstance(second, DefaultBattleOrder)
    assert first.order.species != second.order.species


def test_an_absent_slot_defaults_rather_than_crashing():
    """One active fainted mid-turn is the ordinary case, not an edge case."""
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[Pokemon(9, species="pikachu"), None],
        moves=[[Move("earthquake", 9)], []],
        switches=[[], []],
    )
    order = _decide(battle)
    assert isinstance(order.second_order, DefaultBattleOrder)
    assert not isinstance(order.first_order, DefaultBattleOrder)


def test_a_fainted_foe_is_not_targeted():
    dead = Pokemon(9, species="pikachu")
    dead._status = Status.FNT
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[dead, Pokemon(9, species="tangrowth")],
        moves=[[Move("earthquake", 9)], []],
        switches=[[], []],
        targets={"earthquake": [1, 2]},
    )
    order = _decide(battle)
    assert order.first_order.move_target == 2  # the live foe, not the fainted slot-0 one


def test_singles_is_untouched():
    """Every published number came through the singles path; it must not move."""
    from lategame.agents.heuristic_agent import heuristic_pick

    pick = heuristic_pick(
        Pokemon(9, species="garchomp"),
        Pokemon(9, species="pikachu"),
        [Move("earthquake", 9), Move("dragonclaw", 9)],
        [Pokemon(9, species="tangrowth")],
    )
    assert pick[0] == "move" and pick[1].id == "earthquake"

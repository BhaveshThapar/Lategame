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
from poke_env.player import DefaultBattleOrder, DoubleBattleOrder, PassBattleOrder

from rotomai.agents.heuristic_agent import HeuristicAgent, best_target, doubles_pick


class _FakeDoubleBattle(DoubleBattle):
    """Enough of poke-env's DoubleBattle for the decision path.

    A real subclass, so the `isinstance(battle, DoubleBattle)` branch in `choose_move` is the one
    under test -- `__init__` is deliberately not called (it wants a websocket and a battle tag) and
    the four properties the decision reads are overridden instead.
    """

    def __init__(self, active, opponent_active, moves, switches, targets=None, force=None):
        self._a, self._oa, self._m, self._s = active, opponent_active, moves, switches
        self._targets = targets or {}
        self._force = force or [False, False]

    @property
    def force_switch(self):
        return self._force

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
    assert isinstance(order.second_order, PassBattleOrder)
    assert not isinstance(order.first_order, PassBattleOrder)


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
    from rotomai.agents.heuristic_agent import heuristic_pick

    pick = heuristic_pick(
        Pokemon(9, species="garchomp"),
        Pokemon(9, species="pikachu"),
        [Move("earthquake", 9), Move("dragonclaw", 9)],
        [Pokemon(9, species="tangrowth")],
    )
    assert pick[0] == "move" and pick[1].id == "earthquake"


def test_only_the_slot_being_asked_acts_on_a_partial_replacement():
    """One active fainted, the other still standing: `force_switch` is [False, True] and ONLY that
    slot may act. poke-env still populates `available_switches` for the OTHER slot, so a rule that
    reads just that list emits an order the server REJECTS -- and poke-env then re-requests, which
    measured as 28 choose_move calls for 10 turns and made 98% of collected turns unlabelable."""
    bench = [Pokemon(9, species="tangrowth"), Pokemon(9, species="arcanine")]
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[Pokemon(9, species="pikachu"), None],
        moves=[[Move("earthquake", 9)], []],
        switches=[bench, bench],   # poke-env offers switches for BOTH slots...
        force=[False, True],       # ...but only slot 1 is being asked
    )
    picks = doubles_pick(battle)
    assert picks[0] is None, "slot 0 must pass -- it is not this slot's decision"
    assert picks[1] is not None and isinstance(picks[1][0], Pokemon)


def test_a_forced_slot_may_only_switch_never_move():
    """A replacement request offers switches only; emitting a move there is rejected outright."""
    bench = [Pokemon(9, species="tangrowth")]
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), Pokemon(9, species="magikarp")],
        opponent_active=[Pokemon(9, species="pikachu"), None],
        moves=[[Move("earthquake", 9)], [Move("splash", 9)]],
        switches=[bench, bench],
        force=[True, True],
    )
    for pick in doubles_pick(battle):
        assert pick is None or isinstance(pick[0], Pokemon), f"forced slot must switch, got {pick}"


def test_an_ordinary_turn_is_unaffected_by_the_force_switch_guard():
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), Pokemon(9, species="arcanine")],
        opponent_active=[Pokemon(9, species="pikachu"), Pokemon(9, species="tangrowth")],
        moves=[[Move("earthquake", 9)], [Move("flamethrower", 9)]],
        switches=[[], []],
    )
    picks = doubles_pick(battle)
    assert all(p is not None and isinstance(p[0], Move) for p in picks)


# --------------------------------------------------------------------------- #
# B6f: the doubles loop, and the one-line cause.
# --------------------------------------------------------------------------- #
def test_an_idle_slot_emits_pass_and_the_joined_message_is_a_legal_command():
    """THE doubles loop, as one assertion.

    `DefaultBattleOrder` is poke-env's WHOLE-ORDER sentinel (`/choose default`), and
    `DoubleBattleOrder.message` joins by string surgery -- `first.message + ", " +
    second.message[8:]` -- so half a default produced `/choose default, move earthquake 1`, not a
    legal Showdown command. The server rejected it, poke-env re-requested the identical state, and
    the agent answered identically forever: measured at 4,001 choose_move calls across battle turns
    1-7, against 19 for poke-env's own RandomPlayer over 17 turns.

    That is what put 94.2% of `data/vgc_rl.npz` into 100 of its 899 episodes -- this agent is in
    every collection pool. The per-slot "do nothing" is `pass`.
    """
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[Pokemon(9, species="pikachu"), None],
        moves=[[Move("earthquake", 9)], []],
        switches=[[], []],
    )
    order = _decide(battle)
    assert isinstance(order.second_order, PassBattleOrder)
    assert order.second_order.message == "/choose pass"
    # The joined command is what actually reaches the server, so assert on THAT, not the parts.
    assert "default" not in order.message, f"half-default resurrected: {order.message!r}"
    assert order.message.startswith("/choose ") and order.message.count("/choose") == 1


def test_both_slots_idle_is_still_a_whole_default():
    """`default` is correct when it describes the WHOLE order -- that is what it means. Only half
    of one is malformed."""
    battle = _FakeDoubleBattle(
        active=[None, None],
        opponent_active=[Pokemon(9, species="pikachu"), None],
        moves=[[], []],
        switches=[[], []],
    )
    order = _decide(battle)
    assert isinstance(order, DefaultBattleOrder)
    assert order.message == "/choose default"


def test_a_partial_replacement_sends_pass_for_the_unasked_slot():
    """The state that dominated the shard: slot 0 may only pass, slot 1 must switch. 51.9% of all
    recorded VGC turns carried this signature."""
    bench = [Pokemon(9, species="tangrowth"), Pokemon(9, species="arcanine")]
    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), None],
        opponent_active=[Pokemon(9, species="pikachu"), None],
        moves=[[Move("earthquake", 9)], []],
        switches=[bench, bench],
        force=[False, True],
    )
    order = _decide(battle)
    assert isinstance(order.first_order, PassBattleOrder)
    assert "default" not in order.message
    assert order.message.startswith("/choose pass, switch ")


# --------------------------------------------------------------------------- #
# The search agent's shaped-only fallback reaches doubles too.
# --------------------------------------------------------------------------- #
def test_the_search_fallback_uses_the_doubles_rule_not_the_singles_one():
    """`SearchAgent`'s shaped-only fallback called `heuristic_pick`, the SINGLES rule.

    On a DoubleBattle poke-env hands back `available_moves` as a list of per-slot lists, so that
    call raised `'list' object has no attribute 'base_power'` -- and raised it inside poke-env's
    detached message task, where it is logged and swallowed. The agent then never answers and the
    server plays a default move on the timer, which reads as a very weak arm rather than a broken
    one. This is the one path `_SINGLES_ONLY_AGENTS` deliberately lets through to doubles
    (ROTOMAI_SEARCH_SHAPED_ONLY=1, the VGC M2 ceiling probe), and `arena._singles_only_agents`
    claimed in its own docstring that the fallback was doubles-capable while it was not.
    """
    from rotomai.agents.search_agent import SearchAgent

    battle = _FakeDoubleBattle(
        active=[Pokemon(9, species="garchomp"), Pokemon(9, species="rillaboom")],
        opponent_active=[Pokemon(9, species="pikachu"), Pokemon(9, species="charizard")],
        moves=[[Move("earthquake", 9)], [Move("woodhammer", 9)]],
        switches=[[], []],
    )

    agent = SearchAgent.__new__(SearchAgent)
    # Shaped-only: no trained policy, so `choose_move` reaches the M1 fallback. `_pv` with no
    # `greedy_order` is what that mode actually builds.
    agent._pv = object()
    agent._fm = None
    agent._search = lambda *a, **kw: None  # search returns no order -> fall through

    order = SearchAgent.choose_move(agent, battle)
    assert isinstance(order, DoubleBattleOrder)
    assert order.message.startswith("/choose move ")


def test_the_doubles_join_is_one_function_shared_by_both_agents():
    """Pinned so the fallback cannot drift back to a second copy of the rule. Duplicating it is how
    the singles version survived in `search_agent` after the doubles path was built."""
    import inspect

    from rotomai.agents import heuristic_agent, search_agent

    assert "doubles_order" in inspect.getsource(search_agent.SearchAgent.choose_move)
    assert "doubles_order" in inspect.getsource(heuristic_agent.HeuristicAgent._choose_doubles_move)

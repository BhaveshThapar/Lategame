"""Unit tests for the Build 14 decision-time loop guard (no torch, no server).

Uses a minimal fake battle exposing only what ``LoopGuard`` reads: ``force_switch``,
``battle_tag`` and ``gen`` (for the action-space size).
"""

from rotomai.agents.loop_guard import LoopGuard


class FakeBattle:
    def __init__(self, tag: str = "battle-1", force_switch: bool = False) -> None:
        self.battle_tag = tag
        self.force_switch = force_switch
        self.gen = 9


SWITCH, ATTACK = 0, 6  # any slot < 6 is a switch; >= 6 is a move


def test_short_switch_streak_is_free() -> None:
    g = LoopGuard(penalty=4.0)  # free_switches=1
    b = FakeBattle()
    assert not g.penalty_vector(b).any()  # run 0
    g.record(b, SWITCH)  # run 1  (a lone scout switch)
    assert not g.penalty_vector(b).any()  # run 1 -> factor 0, still free
    g.record(b, SWITCH)  # run 2  (a double-switch pivot)


def test_sustained_streak_penalizes_all_switches_and_escalates() -> None:
    g = LoopGuard(penalty=4.0)
    b = FakeBattle()
    g.record(b, SWITCH)  # run 1
    g.record(b, SWITCH)  # run 2  (the third switch is now the one being decided)
    pen = g.penalty_vector(b)  # factor 1
    assert pen[0] == 4.0 and pen[5] == 4.0  # every switch slot penalized
    assert pen[ATTACK] == 0.0  # attacks are untouched -> argmax pushed to attack
    g.record(b, SWITCH)  # run 3
    assert g.penalty_vector(b)[0] == 8.0  # escalates with the switch run


def test_attack_resets_run() -> None:
    g = LoopGuard(penalty=4.0)
    b = FakeBattle()
    g.record(b, SWITCH)
    g.record(b, SWITCH)
    assert g.penalty_vector(b).any()  # streak active
    g.record(b, ATTACK)  # attacking resets the run
    assert not g.penalty_vector(b).any()
    g.record(b, SWITCH)  # a fresh switch (run 1) is free again
    assert not g.penalty_vector(b).any()


def test_zero_penalty_is_identity() -> None:
    g = LoopGuard(penalty=0.0)
    b = FakeBattle()
    for _ in range(5):
        g.record(b, SWITCH)
    assert not g.penalty_vector(b).any()


def test_forced_switch_is_ignored() -> None:
    g = LoopGuard(penalty=4.0)
    b = FakeBattle()
    g.record(b, SWITCH)  # run 1
    g.record(b, SWITCH)  # run 2
    forced = FakeBattle(force_switch=True)  # same battle_tag
    assert not g.penalty_vector(forced).any()  # no penalty on a forced-switch turn
    g.record(forced, SWITCH)  # a forced switch does not advance the voluntary run
    assert g.penalty_vector(b)[0] == 4.0  # run still 2, factor still 1


# --------------------------------------------------------------------------- #
# B6f: the DOUBLES guard.
#
# Never ported when doubles arrived, and the cost was measurable: on `data/vgc_rl.npz` the longest
# episode ran 12,795 recorded turns over 7 unique observation vectors, and the top 100 of 899
# episodes held 94.2% of the shard -- against an OU shard whose top 10 hold 1.4%. The singles
# guard could not fire, for two independent reasons: `arena._LOOP_GUARD_AGENTS` excludes `doubles`,
# and `LoopGuard` is a 26-wide vector penalizing indices 0-5, while the doubles layout is per slot
# with 0 = PASS and 1-6 = the switches.
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402

from rotomai.agents.loop_guard import DoublesLoopGuard  # noqa: E402
from rotomai.agents.turn_cap import TurnCap  # noqa: E402
from rotomai.features.doubles_action_space import N_SWITCHES, SWITCH_BASE  # noqa: E402

D_SWITCH, D_MOVE, D_PASS = SWITCH_BASE, SWITCH_BASE + N_SWITCHES + 3, 0


class FakeDoubles:
    def __init__(self, tag="battle-d1", force_switch=(False, False)):
        self.battle_tag = tag
        self.force_switch = list(force_switch)
        self.gen = 9


def test_the_doubles_penalty_lands_on_switches_only_and_spares_pass():
    """On the doubles layout index 0 is PASS, not a switch. A singles guard here would penalize
    passing -- often the ONLY legal action on a partial replacement -- and five of the six
    switches, which is not a weaker guard but a differently-wrong one."""
    g = DoublesLoopGuard(penalty=4.0)
    b = FakeDoubles()
    g.record(b, np.array([D_SWITCH, D_SWITCH]))
    g.record(b, np.array([D_SWITCH, D_SWITCH]))
    pen = g.penalty_vector(b)
    assert pen.shape == (2, 107)
    assert pen[0, D_PASS] == 0.0, "pass must never be penalized"
    assert pen[0, SWITCH_BASE] == 4.0 and pen[0, SWITCH_BASE + N_SWITCHES - 1] == 4.0
    assert pen[0, D_MOVE] == 0.0, "moves are untouched -- that is where the argmax is pushed"
    g.record(b, np.array([D_SWITCH, D_SWITCH]))
    assert g.penalty_vector(b)[0, SWITCH_BASE] == 8.0  # escalates


def test_the_two_slots_keep_independent_streaks():
    """Merging them would let one slot's attack clear the other slot's loop -- and the observed
    pathology is one slot passing while the other cycles switches."""
    g = DoublesLoopGuard(penalty=4.0)
    b = FakeDoubles()
    for _ in range(3):
        g.record(b, np.array([D_MOVE, D_SWITCH]))  # slot 0 attacks, slot 1 keeps switching
    pen = g.penalty_vector(b)
    assert not pen[0].any(), "slot 0 attacked every turn; its run is 0"
    assert pen[1, SWITCH_BASE] > 0.0, "slot 1's switch streak must still be penalized"


def test_a_forced_slot_is_never_penalized_and_does_not_advance_its_run():
    g = DoublesLoopGuard(penalty=4.0)
    b = FakeDoubles()
    g.record(b, np.array([D_SWITCH, D_SWITCH]))
    g.record(b, np.array([D_SWITCH, D_SWITCH]))
    forced = FakeDoubles(force_switch=(True, False))
    assert not forced_row(g, forced, 0).any(), "a forced replacement is not a voluntary switch"
    g.record(forced, np.array([D_SWITCH, D_MOVE]))
    assert g.penalty_vector(b)[0, SWITCH_BASE] == 4.0  # slot 0's run is still 2


def forced_row(guard, battle, slot):
    return guard.penalty_vector(battle)[slot]


def test_zero_penalty_is_exact_identity_on_doubles_too():
    g = DoublesLoopGuard(penalty=0.0)
    b = FakeDoubles()
    for _ in range(6):
        g.record(b, np.array([D_SWITCH, D_SWITCH]))
    assert not g.penalty_vector(b).any()


# --------------------------------------------------------------------------- #
# The HARD half: a soft penalty cannot break the state that actually dominated the shard, because
# there every legal option IS a switch (slot 0 may only pass; slot 1 may only pick between two
# replacements). Nothing to push the argmax toward -- so past a ceiling the battle is forfeited.
# --------------------------------------------------------------------------- #
def test_the_turn_cap_is_off_by_default_and_counts_calls_not_battle_turns():
    """It counts `choose_move` CALLS deliberately: the pathology is the server re-requesting the
    same turn, and `battle.turn` does not advance while that happens -- so `battle.turn` is
    precisely the quantity that cannot see the loop."""
    off = TurnCap(None)
    b = FakeDoubles()
    for _ in range(10_000):
        assert off.hit(b) is False
    assert off.turns(b.battle_tag) == 0, "an off cap must not even keep a counter"

    cap = TurnCap(3)
    assert [cap.hit(b) for _ in range(5)] == [False, False, False, True, True]
    assert b.battle_tag in cap.capped


def test_the_turn_cap_is_per_battle():
    cap = TurnCap(2)
    a, b = FakeDoubles("battle-a"), FakeDoubles("battle-b")
    assert not cap.hit(a) and not cap.hit(a) and cap.hit(a)
    assert not cap.hit(b), "a different battle starts its own count"

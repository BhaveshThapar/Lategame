"""Unit tests for the Build 14 decision-time loop guard (no torch, no server).

Uses a minimal fake battle exposing only what ``LoopGuard`` reads: ``force_switch``,
``battle_tag`` and ``gen`` (for the action-space size).
"""

from lategame.agents.loop_guard import LoopGuard


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

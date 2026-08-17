"""The learned doubles agent, and the coupling a factored head cannot see.

The policy emits two independent 107-way distributions, one per active slot. That is what makes a
third format tractable -- a joint head would be 11,449 outputs -- but independence is a modelling
lie in exactly one place: both slots may not switch to the same benched Pokemon. Showdown rejects
such an order and answers with a DEFAULT MOVE rather than an error, so an unresolved conflict is a
silently lost turn that reads as a bad decision.
"""

from __future__ import annotations

import numpy as np

from rotomai.agents.doubles_agent import resolve_switch_conflict


def _logits(slot1_ranking: list[int], n: int = 107) -> np.ndarray:
    """Logits whose slot-1 row ranks `slot1_ranking` highest, in the order given."""
    out = np.full((2, n), -1e9, dtype=np.float32)
    out[0] = -1e9
    for rank, action in enumerate(slot1_ranking):
        out[1, action] = 100.0 - rank
    return out


def test_a_conflicting_switch_re_picks_the_second_slot():
    """Both slots chose switch-to-team[0]; slot 1 must fall to its next-best LEGAL action."""
    mask = np.zeros((2, 107), dtype=bool)
    mask[1, [1, 2, 7]] = True
    action = resolve_switch_conflict(_logits([1, 2, 7]), mask, np.array([1, 1]))
    assert action[0] == 1, "slot 0 keeps its choice -- re-picking both could oscillate"
    assert action[1] == 2


def test_it_skips_illegal_alternatives():
    """The next-best action by logit is not necessarily legal; the mask decides."""
    mask = np.zeros((2, 107), dtype=bool)
    mask[1, [1, 7]] = True  # 2 ranks higher but is NOT legal
    action = resolve_switch_conflict(_logits([1, 2, 7]), mask, np.array([1, 1]))
    assert action[1] == 7


def test_a_slot_with_no_alternative_passes():
    """`pass` is legal for a slot with nothing else to do, and is preferable to an order the
    server would reject outright."""
    mask = np.zeros((2, 107), dtype=bool)
    mask[1, 1] = True  # its ONLY legal action is the conflicting switch
    action = resolve_switch_conflict(_logits([1]), mask, np.array([1, 1]))
    assert action[1] == 0


def test_non_conflicting_actions_are_untouched():
    mask = np.ones((2, 107), dtype=bool)
    for pair in ([1, 2], [7, 7], [0, 0], [12, 40]):
        got = resolve_switch_conflict(_logits([1, 2]), mask, np.array(pair))
        assert list(got) == pair, pair


def test_two_slots_using_the_same_MOVE_index_is_not_a_conflict():
    """Only switches collide -- both actives may use their own move 1, and often should."""
    mask = np.ones((2, 107), dtype=bool)
    assert list(resolve_switch_conflict(_logits([7]), mask, np.array([7, 7]))) == [7, 7]

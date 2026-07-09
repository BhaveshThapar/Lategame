"""Unit tests for the Build-13 ping-pong probe's pure logic (no server, no torch).

Covers the per-row masked-softmax gather, the 2-cycle selector (arm filter + alignment guard,
reusing behavior_probe.two_cycle_rows), the seed-agreement rule, and the pre-registered A/B/C
fork classifier. The torch gather path is exercised by the gate's C-harness at run time.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pingpong_probe.py"
_spec = importlib.util.spec_from_file_location("pingpong_probe", _PATH)
assert _spec is not None and _spec.loader is not None
pp = importlib.util.module_from_spec(_spec)
sys.modules["pingpong_probe"] = pp
_spec.loader.exec_module(pp)


# --------------------------------------------------------------------------- #
# Gather metric
# --------------------------------------------------------------------------- #
def test_gather_action_prob_masks_illegal_and_gathers_per_row():
    # Only actions 3 and 5 legal; logits 0 and ln 3 -> softmax {1/4, 3/4} over the legal pair.
    logits = np.zeros((1, 26))
    logits[0, 5] = np.log(3.0)
    mask = np.zeros((1, 26), bool)
    mask[0, [3, 5]] = True
    assert pp.gather_action_prob(logits, mask, np.array([3]))[0] == pytest.approx(0.25)
    assert pp.gather_action_prob(logits, mask, np.array([5]))[0] == pytest.approx(0.75)
    # Per-row gather: two rows, different target indices.
    two = pp.gather_action_prob(np.zeros((2, 26)), np.ones((2, 26), bool), np.array([0, 1]))
    assert two == pytest.approx([1 / 26, 1 / 26])


# --------------------------------------------------------------------------- #
# 2-cycle selection
# --------------------------------------------------------------------------- #
def _row(action, tag, arm, forced=False, team=None):
    return {
        "action": action, "battle_tag": tag, "arm": arm, "forced": forced,
        "team_order": team or ["A", "B", "C", "D", "E", "F"],
    }


def test_select_2cycle_filters_arm_and_returns_return_action():
    rows = [
        _row(0, "b1", "bc_vs_heuristic"),  # -> A
        _row(1, "b1", "bc_vs_heuristic"),  # -> B
        _row(0, "b1", "bc_vs_heuristic"),  # -> A: 2-cycle at global index 2, return-action 0
        _row(3, "b2", "bc_vs_random"),     # other arm: never selected
    ]
    act = np.array([r["action"] for r in rows])
    tag = np.array([r["battle_tag"] for r in rows])
    arm = np.array([r["arm"] for r in rows])
    sel, ra = pp.select_2cycle(rows, act, tag, arm, "bc_vs_heuristic")
    assert sel.tolist() == [2]
    assert ra.tolist() == [0]
    # The other arm has no 2-cycle to find.
    assert pp.select_2cycle(rows, act, tag, arm, "bc_vs_random")[0].tolist() == []


def test_select_2cycle_alignment_guard():
    rows = [_row(0, "b1", "bc_vs_heuristic")]
    tag = np.array(["b1"])
    arm = np.array(["bc_vs_heuristic"])
    with pytest.raises(SystemExit):  # length mismatch
        pp.select_2cycle(rows, np.array([0, 0]), np.array(["b1", "b1"]), np.array(["x", "x"]), "x")
    with pytest.raises(SystemExit):  # per-row action mismatch
        pp.select_2cycle(rows, np.array([9]), tag, arm, "bc_vs_heuristic")


# --------------------------------------------------------------------------- #
# Fork classifier
# --------------------------------------------------------------------------- #
def test_classify_fork_pre_registered_outcomes():
    # classify_fork(residual, retained_res, baseline, carrier, *, thresholds..., controls_ok).
    kw = dict(residual_min=0.40, retained_max=0.50, baseline_min=0.15, controls_ok=True)
    # A: pp-independent (residual high), full neutralization removes the residual (retained low),
    #    and a single group carries it.
    assert pp.classify_fork(0.70, 0.30, 0.25, "own_bench", **kw) == "FEATURE_CARRIER:own_bench"
    # PP_CARRIED: pp-neutralization alone removes most of the return preference (residual low).
    assert pp.classify_fork(0.30, 0.30, 0.25, "own_bench", **kw) == "PP_CARRIED"
    # C: pp-independent but the residual survives even full neutralization (retained high).
    assert pp.classify_fork(0.70, 0.80, 0.25, "own_bench", **kw) == "DYNAMICS"
    # B: pp-independent, removable, but no single carrier.
    assert pp.classify_fork(0.70, 0.30, 0.25, "NO_CARRIER", **kw) == "DISTRIBUTED"
    assert pp.classify_fork(0.70, 0.30, 0.25, "own_bench+moves", **kw) == "DISTRIBUTED"
    assert pp.classify_fork(0.70, 0.30, 0.25, "SEED_INCONSISTENT", **kw) == "DISTRIBUTED"
    # INCONCLUSIVE: a failed control, or a baseline below the validity floor, dominates.
    assert pp.classify_fork(
        0.70, 0.30, 0.25, "own_bench",
        residual_min=0.40, retained_max=0.50, baseline_min=0.15, controls_ok=False,
    ) == "INCONCLUSIVE"
    assert pp.classify_fork(0.70, 0.30, 0.10, "own_bench", **kw) == "INCONCLUSIVE"

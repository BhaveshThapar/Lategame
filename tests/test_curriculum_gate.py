"""Unit tests for the Lever 13 curriculum gate's pure logic (no server, no training).

Covers the Gate-A pre-flight stat computation (winner/loser start-return separation +
loser-fraction balance) and the Gate-B verdict aggregation (best-iter selection, summary
stats, GREEN/AMBER/RED bands). The full self-play loop is exercised by the server-gated
run; here we only check the math that decides whether a run is worth paying for and how
its curve is scored.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # the gate imports rl_dataset, which imports torch

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "curriculum_gate.py"
_spec = importlib.util.spec_from_file_location("curriculum_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_preflight_stats_separates_win_from_loss():
    # Two episodes (gamma=1): one won (terminal +1), one lost (terminal -1).
    reward = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    done = np.array([False, True, False, True])
    stats = gate.preflight_stats(reward, done, gamma=1.0)
    assert stats["n_episodes"] == 2
    assert stats["n_winners"] == 1
    assert stats["n_losers"] == 1
    assert stats["loser_fraction"] == pytest.approx(0.5)
    assert stats["mean_win_start"] == pytest.approx(1.0)
    assert stats["mean_loss_start"] == pytest.approx(-1.0)
    assert stats["gap"] == pytest.approx(2.0)
    assert gate.preflight_verdict(stats) == "PASS"


def test_preflight_verdict_kills_on_no_losers():
    reward = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    done = np.array([False, True, False, True])
    stats = gate.preflight_stats(reward, done, gamma=1.0)
    assert stats["n_losers"] == 0
    assert gate.preflight_verdict(stats) == "KILL"


def test_preflight_verdict_kills_on_small_gap():
    # Winner start-return barely above loser start-return -> no usable AWR signal.
    reward = np.array([0.2, 0.2, -0.0, 0.0], dtype=np.float32)
    done = np.array([False, True, False, True])
    stats = gate.preflight_stats(reward, done, gamma=1.0)
    assert stats["gap"] < gate._GAP_MIN
    assert gate.preflight_verdict(stats) == "KILL"


def test_preflight_verdict_kills_on_imbalanced_classes():
    # 1 winner, 9 losers -> loser_fraction 0.9 > cap, even with a big gap.
    reward = [0.0, 5.0] + [0.0, -5.0] * 9
    done = [False, True] + [False, True] * 9
    stats = gate.preflight_stats(
        np.array(reward, dtype=np.float32), np.array(done), gamma=1.0
    )
    assert stats["loser_fraction"] > gate._LOSER_FRAC_MAX
    assert gate.preflight_verdict(stats) == "KILL"


def _curve(start: float, per_iter: list[tuple[float, float]]) -> list[dict]:
    """iter-0 baseline (vs_heuristic only) + per-iter (vs_heuristic, vs_iter0) points."""
    curve = [{"iter": 0, "vs_heuristic": start}]
    for k, (vh, vi) in enumerate(per_iter, start=1):
        curve.append({"iter": k, "vs_heuristic": vh, "vs_iter0": vi})
    return curve


def test_seed_record_picks_best_iter_and_final_iter0():
    curve = _curve(0.45, [(0.50, 0.52), (0.58, 0.61), (0.54, 0.55)])
    rec = gate._seed_record("checkpoints/green.pt", 0, curve)
    assert rec["start_vs_heuristic"] == pytest.approx(0.45)
    assert rec["best_iter"] == 2  # 0.58 is the best vs_heuristic
    assert rec["best_vs_heuristic"] == pytest.approx(0.58)
    assert rec["final_vs_iter0"] == pytest.approx(0.55)  # last point, not best
    assert rec["delta_vs_start"] == pytest.approx(0.13)
    assert rec["best_checkpoint"].endswith("iter_02.pt")


def test_verdict_green_amber_red():
    green = [
        gate._seed_record("g", s, _curve(0.46, [(0.57, 0.58), (0.60, 0.62)]))
        for s in range(3)
    ]
    assert gate.verdict(gate._summarize(green)) == "GREEN"

    amber = [
        gate._seed_record("g", s, _curve(0.46, [(0.47, 0.49), (0.46, 0.48)]))
        for s in range(3)
    ]
    assert gate.verdict(gate._summarize(amber)) == "AMBER"

    red = [
        gate._seed_record("g", s, _curve(0.46, [(0.33, 0.30), (0.31, 0.28)]))
        for s in range(3)
    ]
    assert gate.verdict(gate._summarize(red)) == "RED"

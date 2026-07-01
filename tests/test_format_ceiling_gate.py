"""Unit tests for the Lever 15 format-ceiling gate's pure logic (no server, no node).

Covers the Wilson interval, the tie-aware ROC-AUC used for the team-RNG variance
decomposition (M3), and the decision rule that turns (M1 skill band, M2 upper bound,
M3 AUC) into a FORMAT_BOUND / MODEL_BOUND / AMBIGUOUS verdict. The server sweep (M1)
and the node re-sim (M3 data path) are exercised only by the gated run.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "format_ceiling_gate.py"
_spec = importlib.util.spec_from_file_location("format_ceiling_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_wilson_ci_brackets_the_rate_and_clamps():
    lo, hi = gate.wilson_ci(150, 300)
    assert lo < 0.5 < hi
    assert 0.03 < (hi - lo) < 0.12  # ~+/-0.056 at n=300
    lo0, hi0 = gate.wilson_ci(0, 10)
    assert lo0 == 0.0 and hi0 < 1.0
    assert gate.wilson_ci(0, 0) == (0.0, 1.0)


def test_roc_auc_extremes_and_ties():
    # Perfect: higher score always the winner.
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    assert gate.roc_auc(scores, labels) == 1.0
    # Perfectly anti-correlated.
    assert gate.roc_auc(scores, np.array([1, 1, 0, 0])) == 0.0
    # All identical scores -> pure ties -> 0.5.
    assert gate.roc_auc(np.zeros(4), np.array([0, 1, 0, 1])) == 0.5
    # Single class -> undefined.
    assert math.isnan(gate.roc_auc(scores, np.array([1, 1, 1, 1])))


def test_roc_auc_matches_probability_definition():
    # AUC = P(score_pos > score_neg) + 0.5 P(tie), checked by brute force.
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=200).astype(float)  # many ties
    labels = (rng.random(200) < 0.4).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    expected = wins / (len(pos) * len(neg))
    assert abs(gate.roc_auc(scores, labels) - expected) < 1e-9


def _signals(s: float, w: float, g: float, a: float, mirror: float = 0.5) -> tuple:
    m1 = {
        "simpleheuristics": {"rate": s},
        "offrl_green": {"rate": g},
        "mirror": {"rate": mirror},
    }
    m2 = {"search_vs_heuristic": w}
    m3 = {"auc": a}
    return m1, m2, m3


def test_verdict_format_bound_when_nothing_beats_heuristic():
    # Best competent agent at/below BAND_TOP -> pivot, regardless of the (corroborating) AUC.
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.70))
    assert d["verdict"] == "FORMAT_BOUND"
    assert d["next_branch"] == "ou_pivot"
    assert d["mirror_sanity_ok"] is True
    assert d["m3_corroboration"]["rng_bound_corroborated"] is True


def test_verdict_format_bound_even_with_low_auc():
    # AUC below AUC_HI must NOT flip the branch: a caveated proxy only corroborates.
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.45))
    assert d["verdict"] == "FORMAT_BOUND"
    assert d["next_branch"] == "ou_pivot"
    assert d["m3_corroboration"]["rng_bound_corroborated"] is False


def test_verdict_model_bound_on_headroom():
    d = gate.compute_verdict(*_signals(s=0.52, w=0.50, g=0.60, a=0.45))
    assert d["verdict"] == "MODEL_BOUND"
    assert d["next_branch"] == "scale_up"


def test_verdict_ambiguous_between_band_and_headroom():
    # Best competent in (BAND_TOP, HEADROOM) -> tie-break to a cheap scale-up probe.
    d = gate.compute_verdict(*_signals(s=0.55, w=0.50, g=0.47, a=0.70))
    assert d["verdict"] == "AMBIGUOUS"
    assert d["next_branch"] == "scale_up_probe"


def test_mirror_sanity_flag_trips():
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.70, mirror=0.62))
    assert d["mirror_sanity_ok"] is False


def _ou_m1(mirror=0.51, simple=0.60, maxbp=0.20, rand=0.03, green=0.30) -> dict:
    return {
        "mirror": {"rate": mirror},
        "simpleheuristics": {"rate": simple},
        "maxbasepower": {"rate": maxbp},
        "random": {"rate": rand},
        "offrl_green": {"rate": green},
    }


def test_assess_ou_clean_harness_wider_band():
    a = gate.assess_ou(_ou_m1())
    assert a["harness_ok"] is True
    assert a["mirror_sanity_ok"] is True
    assert a["gradient_ok"] is True
    # width 0.60 - 0.03 = 0.57 vs RB 0.523 - 0.007 = 0.516 -> wider
    assert a["band_width"]["wider_than_rb"] is True
    # OU smoke never applies the RB FORMAT/MODEL verdict.
    assert "verdict" not in a


def test_assess_ou_flags_broken_mirror_and_gradient():
    a = gate.assess_ou(_ou_m1(mirror=0.65, maxbp=0.70))  # mirror off; maxbp > simple
    assert a["mirror_sanity_ok"] is False
    assert a["gradient_ok"] is False
    assert a["harness_ok"] is False

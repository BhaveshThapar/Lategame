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
    # The RB-style top-level FORMAT/MODEL verdict is still never applied to the OU smoke.
    assert "verdict" not in a
    # ...but the OU-specific FORMAT-vs-MODEL verdict is: simpleheuristics 0.60 >= HEADROOM 0.58.
    assert a["ou_verdict"]["verdict"] == "MODEL_BOUND"
    assert a["ou_verdict"]["format_bound_rejected"] is True


def test_assess_ou_model_bound_reports_model_gap():
    m1 = _ou_m1(simple=0.64)
    m1["bc_v11"] = {"rate": 0.03, "ci95": [0.01, 0.06]}
    a = gate.assess_ou(m1)
    v = a["ou_verdict"]
    assert v["verdict"] == "MODEL_BOUND"
    assert v["learned_bc"]["rate"] == 0.03
    assert abs(v["model_gap"] - (0.64 - 0.03)) < 1e-9


def test_assess_ou_insufficient_when_competent_below_headroom():
    # A clean harness but no wide-band evidence (simpleheuristics < HEADROOM) -> no MODEL_BOUND.
    a = gate.assess_ou(_ou_m1(simple=0.52, maxbp=0.20))
    assert a["harness_ok"] is True
    assert a["ou_verdict"]["verdict"] == "INSUFFICIENT"
    assert a["ou_verdict"]["format_bound_rejected"] is False


def test_assess_ou_flags_broken_mirror_and_gradient():
    a = gate.assess_ou(_ou_m1(mirror=0.65, maxbp=0.70))  # mirror off; maxbp > simple
    assert a["mirror_sanity_ok"] is False
    assert a["gradient_ok"] is False
    assert a["harness_ok"] is False
    # A dirty harness must not emit a trustworthy verdict.
    assert a["ou_verdict"]["verdict"] == "INSUFFICIENT"
    assert a["ou_verdict"]["format_bound_rejected"] is False


def test_build_matchups_appends_loop_fixed_bc_arm():
    base = gate._build_matchups(None)
    assert base == gate._MATCHUPS
    assert base is not gate._MATCHUPS  # a copy, not the module constant
    with_bc = gate._build_matchups("checkpoints/bc_gen9ou_v11_s0.pt")
    assert with_bc[-1] == ("bc_v11", "bc", "heuristic", "checkpoints/bc_gen9ou_v11_s0.pt")
    assert len(with_bc) == len(base) + 1


def test_build_matchups_can_drop_stale_offrl_green_arm():
    # On OU the RB offrl_green checkpoint can't load (older encoder); the arm is droppable while
    # the loop-fixed bc arm is kept.
    labels = [m[0] for m in gate._build_matchups(
        "checkpoints/bc_gen9ou_v11_s0.pt", include_offrl_green=False
    )]
    assert "offrl_green" not in labels
    assert labels[-1] == "bc_v11"


def test_build_matchups_appends_offrl_ou_arm():
    # Build 16: an offrl checkpoint adds a dedicated offrl_ou (PPO-self-play) learned arm.
    ms = gate._build_matchups(None, include_offrl_green=False, offrl_ckpt="checkpoints/ppo.pt")
    assert ms[-1] == ("offrl_ou", "offrl", "heuristic", "checkpoints/ppo.pt")


def test_build_matchups_appends_both_learned_arms():
    ms = gate._build_matchups(
        "checkpoints/bc.pt", include_offrl_green=False, offrl_ckpt="checkpoints/ppo.pt"
    )
    labels = [m[0] for m in ms]
    assert "offrl_ou" in labels and "bc_v11" in labels
    assert labels[-1] == "bc_v11"  # bc appended after offrl_ou


def test_assess_ou_model_gap_prefers_offrl_arm():
    # With both learned arms present, model_gap is measured against the stronger PPO offrl_ou arm.
    m1 = _ou_m1(simple=0.64)
    m1["bc_v11"] = {"rate": 0.03, "ci95": [0.01, 0.06]}
    m1["offrl_ou"] = {"rate": 0.20, "ci95": [0.16, 0.25]}
    v = gate.assess_ou(m1)["ou_verdict"]
    assert v["verdict"] == "MODEL_BOUND"
    assert v["learned_offrl"]["rate"] == 0.20
    assert v["learned_bc"]["rate"] == 0.03
    assert abs(v["model_gap"] - (0.64 - 0.20)) < 1e-9

"""The bias correction must be reproducible, because a dose is declared against it.

`plan.md` §13 requires a contrast to clear `diff - bias > 0` before it counts. Build 25 simulated
that bias ad hoc and never committed the code, so its table (+0.0045 / +0.0028 / +0.0073) could not
be re-derived -- which is exactly what a pre-registered correction must not be. These tests pin the
properties the correction rests on, plus the reproduction of Build 25's own numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from selection_bias_sim import (  # type: ignore[import-not-found]
    _chi2_ppf,
    _norm_ppf,
    arm_bias,
    estimate_sigma_b,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
_V25_PINNED = [RESULTS / f"ppo_ou_gate_v25{a}_s{s}.json" for a in ("a", "b") for s in (0, 1, 2)]


def test_norm_and_chi2_quantiles_match_known_values():
    """scipy is not in the env, so these are hand-rolled and need pinning."""
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.05) == pytest.approx(-1.6448536, abs=1e-6)
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    # Wilson-Hilferty is asymptotic: ~0.2% off at df=10, but tightens fast, and this script only
    # ever calls it at the pooled plateau dof (~350), where it is good to <0.05%.
    assert _chi2_ppf(0.05, 10) == pytest.approx(3.940, rel=5e-3)
    assert _chi2_ppf(0.05, 100) == pytest.approx(77.929, rel=1e-3)
    assert _chi2_ppf(0.95, 100) == pytest.approx(124.342, rel=1e-3)
    assert _chi2_ppf(0.05, 350) == pytest.approx(307.6476, rel=1e-5)


def test_bias_is_zero_without_dispersion():
    """With every checkpoint equally strong there is nothing to mine: re-scoring at N clears it."""
    rng = np.random.default_rng(0)
    assert arm_bias(160, 0.0, 0.6, 100, 1000, rng) == 0.0


def test_bias_grows_with_draws_and_with_dispersion():
    rng = np.random.default_rng(0)
    small = arm_bias(40, 0.03, 0.6, 100, 20_000, rng)
    large = arm_bias(240, 0.03, 0.6, 100, 20_000, rng)
    assert 0.0 < small < large  # more draws at the argmax => a luckier pick

    flat = arm_bias(160, 0.01, 0.6, 100, 20_000, rng)
    spread = arm_bias(160, 0.05, 0.6, 100, 20_000, rng)
    assert 0.0 < flat < spread  # more real dispersion => more to gain by selecting


def test_reproduces_build_25s_published_bias_table():
    """Build 25 booked +0.0045 / +0.0028 / +0.0073 at sigma_b = 0.0428, draws = iters.

    Reproduced to ~5e-4. The residual is unattributable because Build 25 recorded neither its mu
    nor its exact draw count, which is the whole reason this script now exists.
    """
    rng = np.random.default_rng(0)
    bias = {n: arm_bias(n, 0.0428, 0.55, 100, 200_000, rng) for n in (80, 120, 160)}

    assert bias[120] - bias[80] == pytest.approx(0.0045, abs=6e-4)
    assert bias[160] - bias[120] == pytest.approx(0.0028, abs=6e-4)
    assert bias[160] - bias[80] == pytest.approx(0.0073, abs=6e-4)


def test_differential_bias_is_additive_across_a_chain():
    """#1 + #2 == #3 by construction -- the consistency check the gate table reports."""
    rng = np.random.default_rng(1)
    bias = {n: arm_bias(n, 0.033, 0.65, 100, 50_000, rng) for n in (80, 160, 240)}
    first, second = bias[160] - bias[80], bias[240] - bias[160]
    assert first + second == pytest.approx(bias[240] - bias[80], abs=1e-12)


def test_sigma_b_estimate_rejects_arms_without_a_frozen_plateau():
    """v25c annealed across all 160 iters, so its 'late window' is schedule trend, not dispersion.

    Build 25's sigma_b = 0.0428 came from exactly this shape and is inflated as a result; the
    estimator must refuse rather than silently repeat the mistake.
    """
    with pytest.raises(SystemExit, match="anneal_iters is null"):
        estimate_sigma_b([RESULTS / "ppo_ou_gate_v25c_s0.json"])


def test_frozen_plateau_dispersion_sits_at_the_binomial_floor():
    """The finding that sets Build 26's correction: post-anneal, checkpoints are not resolvably
    different in true strength, and every seed is still drifting UPWARD (which is STILL CLIMBING
    showing up in the curve shape rather than in the gate)."""
    got = estimate_sigma_b(_V25_PINNED)

    assert got.dof > 300
    assert got.sigma_b == 0.0  # detrended residual is at/below p(1-p)/eval_n
    assert got.sigma_b_upper == 0.0  # and so is its 95% bound
    assert len(got.slopes) == 6 and all(s > 0 for s in got.slopes)

    # Charging that drift as dispersion instead is the conservative read, and is what Build 26's
    # pre-registered correction is built on.
    raw = estimate_sigma_b(_V25_PINNED, detrend=False)
    assert raw.sigma_b == pytest.approx(0.0257, abs=5e-4)
    assert raw.sigma_b_upper == pytest.approx(0.0328, abs=5e-4)

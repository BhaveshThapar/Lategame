"""Build 20: the gradient-noise probe's estimators (scripts/grad_noise_diag.py).

The probe's verdict rests on three numbers -- split-half cosine, the McCandlish noise scale, and
the critic's explained variance -- so pin each against a synthetic case whose answer is known in
closed form. No model, no network, no Showdown.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grad_noise_diag import (  # noqa: E402
    NOISE_SCALE_MULT,
    cos_sim,
    explained_variance,
    noise_scale,
    signal_fraction,
    verdict,
)


def _policy_stats(cos: float, b_simple: float) -> dict:
    return {"cos_at_budget": cos, "noise_scale": {"b_simple": b_simple}}


class TestCosSim:
    def test_identical_vectors_are_one(self) -> None:
        g = torch.randn(64)
        assert cos_sim(g, g) == pytest.approx(1.0, abs=1e-5)

    def test_opposed_vectors_are_minus_one(self) -> None:
        g = torch.randn(64)
        assert cos_sim(g, -g) == pytest.approx(-1.0, abs=1e-5)

    def test_pure_signal_is_near_one(self) -> None:
        """A shared direction swamping tiny per-half noise: the halves agree."""
        torch.manual_seed(0)
        signal = torch.randn(4096)
        a = signal + 0.01 * torch.randn(4096)
        b = signal + 0.01 * torch.randn(4096)
        assert cos_sim(a, b) > 0.95

    def test_pure_noise_is_near_zero(self) -> None:
        """Two independent high-dimensional gradients are near-orthogonal -- the NOISE case."""
        torch.manual_seed(0)
        a, b = torch.randn(4096), torch.randn(4096)
        assert abs(cos_sim(a, b)) < 0.1

    def test_degenerate_vector_is_zero_not_nan(self) -> None:
        assert cos_sim(torch.zeros(8), torch.randn(8)) == 0.0


class TestNoiseScale:
    def test_recovers_a_known_noise_scale(self) -> None:
        """Construct E|g_B|^2 = |G|^2 + tr(Sigma)/B exactly; the estimator must invert it."""
        g2, trace = 4.0, 800.0  # B_simple = 200
        b_small, b_big = 100, 1000
        out = noise_scale(g2 + trace / b_small, b_small, g2 + trace / b_big, b_big)
        assert out["resolvable"]
        assert out["g_norm_sq"] == pytest.approx(g2, rel=1e-4)
        assert out["trace_sigma"] == pytest.approx(trace, rel=1e-4)
        assert out["b_simple"] == pytest.approx(200.0, rel=1e-3)

    def test_noise_free_gradient_gives_zero_scale(self) -> None:
        """No variance across batch sizes => tr(Sigma) = 0 => a bigger batch buys nothing."""
        out = noise_scale(4.0, 100, 4.0, 1000)
        assert not out["resolvable"]  # trace == 0 is not a resolvable positive scale
        assert out["trace_sigma"] == pytest.approx(0.0, abs=1e-6)

    def test_unresolvable_signal_reports_inf_not_nan(self) -> None:
        """Noise so dominant that |G|^2 estimates <= 0: report inf, never a silent NaN."""
        out = noise_scale(1.0, 100, 5.0, 1000)  # big batch NOISIER than small: impossible in theory
        assert not out["resolvable"]
        assert math.isinf(out["b_simple"])

    def test_rejects_unordered_batch_sizes(self) -> None:
        with pytest.raises(ValueError, match="b_big > b_small"):
            noise_scale(1.0, 1000, 1.0, 100)


class TestExplainedVariance:
    def test_perfect_critic_is_one(self) -> None:
        returns = torch.randn(256)
        assert explained_variance(returns, returns) == pytest.approx(1.0, abs=1e-5)

    def test_constant_critic_is_zero(self) -> None:
        """Predicting the mean explains none of the variance."""
        returns = torch.randn(256)
        value = torch.full_like(returns, float(returns.mean()))
        assert explained_variance(value, returns) == pytest.approx(0.0, abs=1e-5)

    def test_anticorrelated_critic_is_negative(self) -> None:
        returns = torch.randn(256)
        assert explained_variance(-returns, returns) < 0.0

    def test_zero_variance_returns_is_zero_not_nan(self) -> None:
        constant = torch.ones(64)
        assert explained_variance(torch.zeros(64), constant) == 0.0


class TestSignalFraction:
    def test_equals_one_half_at_the_noise_scale(self) -> None:
        """B_simple is by definition the batch size where signal and noise contribute equally."""
        assert signal_fraction(500.0, 500) == pytest.approx(0.5, abs=1e-4)

    def test_batch_far_above_the_noise_scale_is_nearly_all_signal(self) -> None:
        assert signal_fraction(10.0, 10_000) > 0.99

    def test_batch_far_below_the_noise_scale_is_nearly_all_noise(self) -> None:
        assert signal_fraction(10_000.0, 10) < 0.01

    def test_unresolvable_noise_scale_is_zero_signal(self) -> None:
        assert signal_fraction(math.inf, 1600) == 0.0


class TestWithinBufferSplitIsOptimisticallyBiased:
    """Why the probe compares INDEPENDENT rollouts rather than two halves of one buffer.

    One rollout is ONE draw of league opponents, teams and episodes. Splitting it sees only
    turn-level noise: both halves inherit the same nuisance draw, so they agree more than two
    real PPO iterations ever would. The bias is toward AGREEMENT -- a split-half cosine would
    overstate how well the direction is pinned down, which is the dangerous direction here
    (it could manufacture a SIGNAL_LIMITED verdict and wrongly skip Stage B).

    Modelled below as a per-rollout random effect ``b`` shared by every row of a buffer -- the
    gradient contribution of "this rollout happened to draw a soft opponent".
    """

    @staticmethod
    def _buffer(n: int, dim: int, signal: torch.Tensor, rollout_effect: float) -> torch.Tensor:
        """``n`` per-sample grads: a shared true signal + a per-BUFFER offset + per-row noise."""
        b = rollout_effect * torch.randn(dim)  # drawn once per buffer, shared by all its rows
        return signal + b + torch.randn(n, dim)

    def test_shared_rollout_effect_inflates_the_within_buffer_cosine(self) -> None:
        """The bug: halves of one buffer agree far more than independent rollouts do."""
        torch.manual_seed(0)
        dim, n, signal, effect = 256, 512, torch.zeros(256), 3.0  # no true signal at all

        one = self._buffer(n, dim, signal, effect)
        within = cos_sim(one[: n // 2].mean(dim=0), one[n // 2 :].mean(dim=0))

        across = [
            cos_sim(
                self._buffer(n // 2, dim, signal, effect).mean(dim=0),
                self._buffer(n // 2, dim, signal, effect).mean(dim=0),
            )
            for _ in range(20)
        ]
        # There is NO signal, so the honest answer is ~0 -- which only the cross-rollout read gives.
        assert within > 0.5
        assert abs(sum(across) / len(across)) < 0.25

    def test_independent_rollouts_read_pure_noise_as_zero(self) -> None:
        torch.manual_seed(0)
        dim, signal = 256, torch.zeros(256)
        a = self._buffer(256, dim, signal, rollout_effect=0.0).mean(dim=0)
        b = self._buffer(256, dim, signal, rollout_effect=0.0).mean(dim=0)
        assert abs(cos_sim(a, b)) < 0.2

    def test_independent_rollouts_recover_a_strong_signal(self) -> None:
        torch.manual_seed(0)
        dim = 256
        signal = 4.0 * torch.randn(dim)
        a = self._buffer(256, dim, signal, rollout_effect=0.0).mean(dim=0)
        b = self._buffer(256, dim, signal, rollout_effect=0.0).mean(dim=0)
        assert cos_sim(a, b) > 0.9


class TestVerdict:
    def test_noise_limited(self) -> None:
        """Low cosine + a noise scale well past the buffer => run Stage B."""
        n = 1600
        call, why = verdict(_policy_stats(0.05, NOISE_SCALE_MULT * n + 1), n)
        assert call == "NOISE_LIMITED"
        assert "Run Stage B" in why

    def test_unresolvable_noise_scale_is_noise_limited(self) -> None:
        """inf B_simple (noise swamps the signal) is the strongest noise evidence, not a crash."""
        call, _ = verdict(_policy_stats(0.05, math.inf), 1600)
        assert call == "NOISE_LIMITED"

    def test_signal_limited(self) -> None:
        """High cosine + a noise scale below the buffer => Stage B predicted NULL, skip it."""
        call, why = verdict(_policy_stats(0.80, 400), 1600)
        assert call == "SIGNAL_LIMITED"
        assert "predicted NULL" in why

    def test_ambiguous_when_the_two_signals_disagree(self) -> None:
        """A clean direction but a huge noise scale matches neither arm -- run Stage B anyway."""
        call, _ = verdict(_policy_stats(0.80, math.inf), 1600)
        assert call == "AMBIGUOUS"

    def test_ambiguous_in_the_threshold_gap(self) -> None:
        call, _ = verdict(_policy_stats(0.40, 4000), 1600)
        assert call == "AMBIGUOUS"

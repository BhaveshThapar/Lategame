"""Build 20: the corrected build-vs-build strength gate (scripts/seed_strength_gate.py).

The old protocol (score the single best checkpoint at n=300) could not resolve the effect Build 20
was testing for. These pin the two properties that fix it: pooling every seed's best checkpoint,
and reading a z-test rather than CI overlap alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from seed_strength_gate import best_checkpoints, two_proportion_z  # noqa: E402


class TestTwoProportionZ:
    def test_identical_rates_are_null(self) -> None:
        out = two_proportion_z(150, 300, 150, 300)
        assert out["diff"] == 0.0
        assert out["p_value"] > 0.9

    def test_detects_a_real_improvement(self) -> None:
        """+0.07 over 900 battles/arm -- the Build 20 effect size at the CORRECTED sample."""
        out = two_proportion_z(441, 900, 504, 900)  # 0.490 -> 0.560
        assert out["diff"] == pytest.approx(0.07, abs=0.005)
        assert out["p_value"] < 0.05

    def test_same_effect_is_INVISIBLE_at_the_old_single_checkpoint_sample(self) -> None:
        """The whole reason this gate exists: n=300/arm cannot see +0.07."""
        out = two_proportion_z(147, 300, 168, 300)  # same 0.490 -> 0.560
        assert out["diff"] == pytest.approx(0.07, abs=0.005)
        assert out["p_value"] > 0.05  # underpowered -- a real effect reads as NULL

    def test_regression_is_signed_negative(self) -> None:
        out = two_proportion_z(504, 900, 441, 900)
        assert out["diff"] < 0
        assert out["p_value"] < 0.05

    def test_ci_overlap_can_hide_a_significant_difference(self) -> None:
        """Why the gate reports a z-test and not just CI disjointness.

        Overlapping 95% CIs do NOT imply non-significance -- the converse of the disjointness rule
        is false. At +0.05 over 900 battles/arm the intervals still overlap while p = 0.034, so a
        CI-only reading would call a real effect NULL.
        """
        from format_ceiling_gate import wilson_ci  # type: ignore[import-not-found]

        _, hi_a = wilson_ci(441, 900)  # 0.490
        lo_b, _ = wilson_ci(486, 900)  # 0.540
        assert lo_b < hi_a  # the CIs OVERLAP ...
        assert two_proportion_z(441, 900, 486, 900)["p_value"] < 0.05  # ... yet p < 0.05

    def test_a_big_effect_does_give_disjoint_cis_at_the_corrected_sample(self) -> None:
        """Sanity on the fix: at n=900/arm, +0.07 clears BOTH tests -- the old n=300 saw neither."""
        from format_ceiling_gate import wilson_ci  # type: ignore[import-not-found]

        _, hi_a = wilson_ci(441, 900)
        lo_b, _ = wilson_ci(504, 900)
        assert lo_b > hi_a
        assert two_proportion_z(441, 900, 504, 900)["p_value"] < 0.05


class TestBestCheckpoints:
    def test_takes_every_seed_not_the_across_seed_argmax(self, tmp_path: Path) -> None:
        gate = {
            "best_checkpoint": "ckpt/s1/iter_47.pt",  # the old protocol's single pick
            "records": [
                {"seed": 0, "best_vs_heuristic": 0.57, "best_checkpoint": "ckpt/s0/iter_46.pt"},
                {"seed": 1, "best_vs_heuristic": 0.60, "best_checkpoint": "ckpt/s1/iter_47.pt"},
                {"seed": 2, "best_vs_heuristic": 0.53, "best_checkpoint": "ckpt/s2/iter_45.pt"},
            ],
        }
        p = tmp_path / "gate.json"
        p.write_text(json.dumps(gate))
        assert best_checkpoints(str(p)) == [
            "ckpt/s0/iter_46.pt",
            "ckpt/s1/iter_47.pt",
            "ckpt/s2/iter_45.pt",
        ]

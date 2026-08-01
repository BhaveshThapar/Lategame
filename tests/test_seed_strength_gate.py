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

from seed_strength_gate import (  # noqa: E402
    best_checkpoints,
    pairwise_comparisons,
    seed_level_stats,
    two_proportion_z,
    verdict_of,
)

REPO = Path(__file__).resolve().parents[1]


def _arm(wins: int, n: int) -> dict:
    from format_ceiling_gate import wilson_ci  # type: ignore[import-not-found]

    lo, hi = wilson_ci(wins, n)
    return {"wins": wins, "n": n, "rate": round(wins / n, 4), "ci95": [lo, hi]}


def _seeded_arm(per_seed_wins: list[int], n: int) -> dict:
    """An arm carrying its per-seed breakdown, as ``score_build`` actually builds it."""
    arm = _arm(sum(per_seed_wins), n * len(per_seed_wins))
    arm["per_checkpoint"] = [
        {"checkpoint": f"ckpt/s{i}/best.pt", "wins": w, "n": n, "rate": round(w / n, 4)}
        for i, w in enumerate(per_seed_wins)
    ]
    return arm


# Build 23's REAL per-seed results (results/seed_strength_gate_v23.json), n=300 per checkpoint.
V23B_WINS = [179, 141, 182]  # rates 0.597 / 0.470 / 0.607, pooled 502/900
V23A_WINS = [130, 134, 162]  # rates 0.433 / 0.447 / 0.540, pooled 426/900


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


class TestVerdict:
    """Build 24 plumbing: the verdict was two-way (WIN/NULL) and could not say REGRESSION.

    Build 23 measured a CI-disjoint p = 0.0003 REVERSAL and the gate stamped it ``NULL`` --
    the same label a p = 0.67 nothing gets. These pin the three-way replacement.
    """

    def test_significant_positive_is_win(self) -> None:
        assert verdict_of(diff=0.07, significant=True) == "WIN"

    def test_significant_negative_is_REGRESSION_not_null(self) -> None:
        assert verdict_of(diff=-0.0844, significant=True) == "REGRESSION"

    def test_non_significant_is_null_whatever_the_sign(self) -> None:
        assert verdict_of(diff=0.02, significant=False) == "NULL"
        assert verdict_of(diff=-0.02, significant=False) == "NULL"

    def test_the_build_23_result_now_reads_REGRESSION(self) -> None:
        """The regression proof, on the real artifact: v23b 502/900 -> v23a 426/900."""
        comps = pairwise_comparisons(
            {"v23b": _arm(502, 900), "v23a": _arm(426, 900)}, ["v23b", "v23a"]
        )
        (c,) = comps
        assert c["diff"] == pytest.approx(-0.0844, abs=0.001)
        assert c["p_value"] < 0.001
        assert c["disjoint_ci"] is True
        assert c["verdict"] == "REGRESSION"  # was "NULL" through Build 23

    def test_the_shipped_v23_json_still_carries_the_old_null(self) -> None:
        """Documents WHY this fix exists; the stale label stays on disk until the gate is re-run."""
        path = REPO / "results" / "seed_strength_gate_v23.json"
        if not path.exists():
            pytest.skip("v23 result not present")
        old = json.loads(path.read_text())["comparison"]
        assert old["verdict"] == "NULL" and old["p_value"] < 0.001


class TestPairwiseComparisons:
    """Build 24 runs a 2x2 factorial + a decomposition arm, so the gate must handle k > 2.

    Before this, ``main`` guarded ``if len(names) == 2`` and a 5-arm invocation scored every arm
    and then wrote NO comparison block at all -- silently, after hours of battles.
    """

    def test_k_arms_gives_every_pair(self) -> None:
        arms = {n: _arm(450, 900) for n in ["a", "b", "c", "d"]}
        comps = pairwise_comparisons(arms, ["a", "b", "c", "d"])
        assert len(comps) == 6  # k(k-1)/2, not 0
        assert [(c["baseline"], c["candidate"]) for c in comps] == [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]

    def test_two_arms_still_produce_exactly_one_comparison(self) -> None:
        comps = pairwise_comparisons({"x": _arm(441, 900), "y": _arm(504, 900)}, ["x", "y"])
        assert len(comps) == 1 and comps[0]["verdict"] == "WIN"

    def test_alpha_applies_the_preregistered_bonferroni_level(self) -> None:
        """A 0.05-significant contrast must NOT survive a 4-contrast correction."""
        arms = {"base": _arm(1215, 2700), "cand": _arm(1295, 2700)}  # +0.030, p ~ 0.029
        loose = pairwise_comparisons(arms, ["base", "cand"], alpha=0.05)[0]
        strict = pairwise_comparisons(arms, ["base", "cand"], alpha=0.0125)[0]
        assert 0.0125 < loose["p_value"] < 0.05
        assert loose["verdict"] == "WIN"
        assert strict["verdict"] == "NULL"  # same data, corrected level


class TestSeedLevelStats:
    """Build 24: the pooled z-test treats 900 battles as iid, and the SEEDS ARE NOT.

    Build 23's between-seed sd is ~0.0706 against a within-seed binomial sd of 0.0287 at n=300, so
    the pooled p is a claim about the CHECKPOINTS while the seed-level statistic is the claim about
    the PROCEDURE. These pin that the second one is computed and reported -- never adjudicating.
    """

    def test_build_23_pooled_p_is_a_seed_level_t_of_two(self) -> None:
        """The headline reason this exists: p = 0.0003 pooled, t = -2.04 paired by seed."""
        s = seed_level_stats(
            _seeded_arm(V23B_WINS, 300)["per_checkpoint"],
            _seeded_arm(V23A_WINS, 300)["per_checkpoint"],
        )
        assert s is not None
        assert s["mean_paired_diff"] == pytest.approx(-0.0844, abs=0.001)
        assert s["t"] == pytest.approx(-2.04, abs=0.02)
        assert s["seeds"] == 3
        assert s["n_positive"] == 0  # all three seeds moved the same way ...
        assert s["sign_test_p"] == pytest.approx(0.25)  # ... but 3/3 is only p = 0.25

    def test_it_rides_along_on_every_comparison(self) -> None:
        comps = pairwise_comparisons(
            {"v23b": _seeded_arm(V23B_WINS, 300), "v23a": _seeded_arm(V23A_WINS, 300)},
            ["v23b", "v23a"],
        )
        (c,) = comps
        assert c["p_value"] < 0.001  # pooled: decisive
        assert c["seed_level"]["t"] == pytest.approx(-2.04, abs=0.02)  # seed-level: not

    def test_it_never_changes_the_verdict(self) -> None:
        """The pooled test stays authoritative; the seed-level block is reported, not applied."""
        bare = pairwise_comparisons({"a": _arm(502, 900), "b": _arm(426, 900)}, ["a", "b"])[0]
        rich = pairwise_comparisons(
            {"a": _seeded_arm(V23B_WINS, 300), "b": _seeded_arm(V23A_WINS, 300)}, ["a", "b"]
        )[0]
        assert rich["verdict"] == bare["verdict"] == "REGRESSION"
        keys = ("diff", "z", "p_value")
        assert [rich[k] for k in keys] == [bare[k] for k in keys]

    def test_unequal_seed_counts_refuse_to_pair(self) -> None:
        """Pairing is by POSITION (= seed order). Mismatched arms are not a pairing but a bug."""
        assert (
            seed_level_stats(
                _seeded_arm([179, 141, 182], 300)["per_checkpoint"],
                _seeded_arm([130, 134], 300)["per_checkpoint"],
            )
            is None
        )

    def test_a_single_seed_arm_refuses_to_pair(self) -> None:
        assert (
            seed_level_stats(
                _seeded_arm([179], 300)["per_checkpoint"],
                _seeded_arm([130], 300)["per_checkpoint"],
            )
            is None
        )

    def test_absent_per_checkpoint_is_none_not_a_crash(self) -> None:
        """Historical two-arm result files predate per_checkpoint plumbing through this path."""
        (c,) = pairwise_comparisons({"a": _arm(502, 900), "b": _arm(426, 900)}, ["a", "b"])
        assert c["seed_level"] is None

    def test_raising_n_does_not_move_the_seed_level_statistic(self) -> None:
        """Why Build 24 buys battles for the POOLED test only.

        Tripling every seed's battle count at identical rates leaves the paired statistic
        untouched -- the variance it measures is BETWEEN seeds, and only more seeds shrink it.
        """
        small = seed_level_stats(
            _seeded_arm([179, 141, 182], 300)["per_checkpoint"],
            _seeded_arm([130, 134, 162], 300)["per_checkpoint"],
        )
        big = seed_level_stats(
            _seeded_arm([537, 423, 546], 900)["per_checkpoint"],
            _seeded_arm([390, 402, 486], 900)["per_checkpoint"],
        )
        assert small is not None and big is not None
        assert small["t"] == pytest.approx(big["t"], abs=0.001)
        # ... while the POOLED test gains a full factor of sqrt(3) in z over the same data.
        pooled_small = two_proportion_z(502, 900, 426, 900)
        pooled_big = two_proportion_z(1506, 2700, 1278, 2700)
        assert abs(pooled_big["z"]) > abs(pooled_small["z"]) * 1.6

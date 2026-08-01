"""The merged gate file must carry EVERY seed, because the verdict silently degrades if it doesn't.

`seed_strength_gate.py` scores `records[*]["best_checkpoint"]` and pools to 900 battles/arm. Hand it
a merge that dropped a seed and it pools 600, still prints a verdict, and has quietly lost the power
the protocol exists to buy -- no error, no warning. So the merge rule is pinned here, against the
one artifact that is known-good: the hand-assembled v20 file the last build's verdict was read off.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from merge_gate_seeds import merge  # type: ignore[import-not-found]

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _v20_seeds() -> list[dict]:
    return [json.loads((RESULTS / f"ppo_ou_gate_v20_s{s}.json").read_text()) for s in (0, 1, 2)]


def test_rebuilds_the_committed_v20_byte_for_byte():
    """The test that matters: v20 was assembled by hand, and its verdict was read off it."""
    committed = (RESULTS / "ppo_ou_gate_v20.json").read_text()
    gate = json.loads(committed)

    rebuilt = merge(_v20_seeds(), gate["confirmatory_ladder"]["_source"], gate["_note"])

    assert json.dumps(rebuilt, indent=2) == committed


def test_every_seed_survives_the_merge():
    merged = merge(_v20_seeds(), "src", "note")
    assert merged["seeds"] == [0, 1, 2]
    assert len(merged["records"]) == 3
    assert len({r["best_checkpoint"] for r in merged["records"]}) == 3


def test_summary_std_is_population_not_sample():
    """v20 reports 0.0287 for [0.57, 0.60, 0.53]; the sample std would be 0.0351."""
    merged = merge(_v20_seeds(), "src", "note")
    assert merged["summary"]["best_vs_heuristic_per_seed"] == [0.57, 0.6, 0.53]
    assert merged["summary"]["best_vs_heuristic_mean"] == 0.5667
    assert merged["summary"]["best_vs_heuristic_std"] == 0.0287


def test_top_level_best_checkpoint_is_the_across_seed_argmax():
    """Kept for continuity with v19/v20. seed_strength_gate ignores it and reads records[*]."""
    merged = merge(_v20_seeds(), "src", "note")
    assert merged["best_checkpoint"] == "checkpoints/ppo_ou_budget_s1/iter_47.pt"


def test_refuses_to_pool_seeds_from_different_arms():
    """A mismatched arm means the per-seed files came from different experiments."""
    a, b, c = _v20_seeds()
    b["games_per_opp"] = 16  # b was run under v19's sample budget, not v20's
    with pytest.raises(ValueError, match="disagree on the arm"):
        merge([a, b, c], "src", "note")


def test_refuses_to_pool_seeds_run_at_different_kl_budgets():
    """Build 23's lever. v22's NULL was unattributable because its trust region bound; pooling a
    raised-budget seed with a throttled one would hide exactly that, and print a verdict anyway."""
    a, b, c = _v20_seeds()
    for g in (a, b, c):
        g["target_kl"] = 0.06
    b["target_kl"] = 0.03  # b was resubmitted before the budget was raised
    with pytest.raises(ValueError, match="disagree on the arm"):
        merge([a, b, c], "src", "note")


def test_pre_build23_gates_still_pool_without_a_target_kl_key():
    """v20-v22 predate the field entirely; they must keep merging (None == None)."""
    seeds = _v20_seeds()
    assert all("target_kl" not in g for g in seeds)
    assert merge(seeds, "src", "note")["seeds"] == [0, 1, 2]


def test_refuses_duplicate_seeds():
    """Re-running seed 0 into the s1 slot would pool the same checkpoint twice."""
    a, _, c = _v20_seeds()
    with pytest.raises(ValueError, match="duplicate seed"):
        merge([a, a, c], "src", "note")


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="no per-seed gate files"):
        merge([], "src", "note")

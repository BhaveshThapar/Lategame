"""The pinned gate must be selection-free AND must not be mistakable for a seed-best gate.

Build 25 compares arms of different LENGTHS (80 / 120 / 160), where the argmax-selection bias no
longer cancels in the difference (+0.0073 simulated for 80->160, against Build 24's negligible
+0.0014). The pinned file is the selection-free second read. Two failure modes are pinned here:
a pin that names a checkpoint no run ever wrote, and a pinned file that reads downstream as though
its `best_checkpoint` were a best.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pin_gate_checkpoint import checkpoint_for, pin  # type: ignore[import-not-found]

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _v24d() -> dict:
    """Build 24's decomposition arm: iters 80, and its seeds peaked at 73 / 75 / 54 -- so pinning
    to the terminal iteration genuinely moves every seed, which is the point of the fixture."""
    return json.loads((RESULTS / "ppo_ou_gate_v24d.json").read_text())


def test_pins_every_seed_to_the_terminal_iteration_by_default():
    pinned = pin(_v24d())
    assert pinned["_pinned_iter"] == 80
    assert [r["best_iter"] for r in pinned["records"]] == [80, 80, 80]
    assert [r["best_checkpoint"] for r in pinned["records"]] == [
        "checkpoints/ppo_v24d_s0/iter_80.pt",
        "checkpoints/ppo_v24d_s1/iter_80.pt",
        "checkpoints/ppo_v24d_s2/iter_80.pt",
    ]


def test_the_scored_rate_is_the_curve_value_at_the_pin_not_the_best():
    """v24d's seeds peaked at 73/75/54; the terminal rates are lower, and must be reported as such
    -- silently carrying the old best_vs_heuristic would make the file self-inconsistent."""
    original, pinned = _v24d(), pin(_v24d())
    for orig_rec, pin_rec in zip(original["records"], pinned["records"], strict=True):
        terminal = next(p for p in orig_rec["curve"] if p["iter"] == 80)
        assert pin_rec["best_vs_heuristic"] == terminal["vs_heuristic"]
        assert pin_rec["best_vs_heuristic"] <= orig_rec["best_vs_heuristic"]


def test_summary_and_top_level_best_are_recomputed_from_the_pinned_records():
    pinned = pin(_v24d())
    rates = [r["best_vs_heuristic"] for r in pinned["records"]]
    assert pinned["summary"]["best_vs_heuristic_per_seed"] == rates
    assert pinned["summary"]["best_iters"] == [80, 80, 80]
    best = max(pinned["records"], key=lambda r: r["best_vs_heuristic"])
    assert pinned["best_checkpoint"] == best["best_checkpoint"]


def test_the_file_announces_that_its_best_checkpoint_is_not_a_best():
    """seed_strength_gate reads records[*]['best_checkpoint'] and cannot tell the difference, so
    the artifact has to say so itself -- the merge_gate_seeds lesson."""
    pinned = pin(_v24d())
    assert pinned["_pinned_iter"] == 80
    assert "PINNED" in pinned["_note"]
    assert "NOT a best" in pinned["_note"]
    assert "must not be compared against a seed-best gate file" in pinned["_note"]


def test_seed_strength_gate_reads_the_pinned_paths(tmp_path):
    """The contract that matters: best_checkpoints() reads records[*]['best_checkpoint'] as-is."""
    from seed_strength_gate import best_checkpoints  # type: ignore[import-not-found]

    pinned = pin(_v24d())
    out = tmp_path / "pinned.json"
    out.write_text(json.dumps(pinned))
    assert best_checkpoints(str(out)) == [r["best_checkpoint"] for r in pinned["records"]]


def test_an_explicit_iteration_overrides_the_terminal_default():
    pinned = pin(_v24d(), 50)
    assert pinned["_pinned_iter"] == 50
    assert [r["best_iter"] for r in pinned["records"]] == [50, 50, 50]
    assert pinned["records"][0]["best_checkpoint"] == "checkpoints/ppo_v24d_s0/iter_50.pt"


def test_refuses_a_pin_past_the_end_of_the_run():
    """A pin beyond the curve names a checkpoint that was never written; the strength gate's
    preflight would report it as the gitignored-checkpoints problem, which it is not."""
    with pytest.raises(ValueError, match="no curve point at iteration 200"):
        pin(_v24d(), 200)


def test_fails_loudly_if_the_checkpoint_naming_convention_drifts():
    """Silent drift here would produce a valid-looking path to nothing, in every arm at once."""
    gate = _v24d()
    gate["records"][1]["best_checkpoint"] = "checkpoints/ppo_v24d_s1/iter75.pt"
    with pytest.raises(ValueError, match="naming has drifted"):
        pin(gate)


def test_iteration_zero_pins_to_the_warm_start_itself():
    """Mirrors ppo_continue_gate._best_checkpoint: iter 0 is the init, not a file in out_dir."""
    gate = _v24d()
    pinned = pin(gate, 0)
    assert all(r["best_checkpoint"] == gate["init"] for r in pinned["records"])


def test_checkpoint_for_matches_the_gate_convention():
    assert checkpoint_for("checkpoints/x_s0", "init.pt", 7) == "checkpoints/x_s0/iter_07.pt"
    assert checkpoint_for("checkpoints/x_s0", "init.pt", 160) == "checkpoints/x_s0/iter_160.pt"
    assert checkpoint_for("checkpoints/x_s0", "init.pt", 0) == "init.pt"

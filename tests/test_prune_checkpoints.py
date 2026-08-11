"""Pruning must be derived from what the results cite, and must fail loudly rather than guess.

An arm costs ~40 h of wall-clock and cannot be rebuilt from its curve, so the only acceptable
failure mode for this script is refusing to delete. Four ways it could quietly delete something
load-bearing are pinned here: a cited intermediate, an uncited terminal, a CHUNKED arm whose first
chunk left a short gate file behind (`v26b` ran 160 -> resume -> 320, so a naive read of `iters`
would call 160 its terminal and delete iterations 161-320), and an arm whose gate names a terminal
that never reached disk.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prune_checkpoints import (  # type: ignore[import-not-found]
    PruneError,
    plan_prune,
    referenced_checkpoints,
    terminal_iters,
)

ROOT = Path(__file__).resolve().parent.parent


def _arm(tmp_path: Path, name: str, iters: range, extra: tuple[str, ...] = ()) -> Path:
    arm = tmp_path / "checkpoints" / name
    arm.mkdir(parents=True)
    for i in iters:
        (arm / f"iter_{i:02d}.pt").write_bytes(b"x" * 16)
    (arm / "curve.json").write_text("{}")
    for name_ in extra:
        (arm / name_).write_bytes(b"y" * 8)
    return arm


def _gate(tmp_path: Path, fname: str, gate: dict) -> None:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / fname).write_text(json.dumps(gate))


def test_a_cited_intermediate_is_never_deleted(tmp_path):
    """`best_checkpoint` lands mid-run (v26b's seeds peaked at 300/315/292 of 320), so the cited
    file is usually NOT the terminal -- deleting it would break every re-read of the gate."""
    _arm(tmp_path, "ppo_x_s0", range(1, 11))
    _gate(tmp_path, "gate.json", {
        "iters": 10,
        "records": [{"out_dir": "checkpoints/ppo_x_s0",
                     "best_checkpoint": "checkpoints/ppo_x_s0/iter_04.pt"}],
    })

    plan = plan_prune(
        tmp_path / "checkpoints",
        referenced_checkpoints(tmp_path / "results"),
        terminal_iters(tmp_path / "results"),
    )
    deleted = set(plan.paths())
    assert "checkpoints/ppo_x_s0/iter_04.pt" not in deleted
    assert "checkpoints/ppo_x_s0/iter_10.pt" not in deleted  # terminal
    assert "checkpoints/ppo_x_s0/iter_05.pt" in deleted  # a genuine intermediate


def test_the_terminal_is_kept_even_when_no_result_cites_it(tmp_path):
    """`pin_gate_checkpoint` re-reads the terminal to produce the selection-free second read.
    Build 26's 0.7513 is that read, and it is not reproducible without these files."""
    _arm(tmp_path, "ppo_x_s0", range(1, 11))
    _gate(tmp_path, "gate.json", {
        "iters": 10,
        "records": [{"out_dir": "checkpoints/ppo_x_s0",
                     "best_checkpoint": "checkpoints/ppo_x_s0/iter_04.pt"}],
    })

    plan = plan_prune(
        tmp_path / "checkpoints",
        referenced_checkpoints(tmp_path / "results"),
        terminal_iters(tmp_path / "results"),
    )
    assert "checkpoints/ppo_x_s0/iter_10.pt" in set(plan.arms[0].keep)


def test_a_chunked_arm_takes_its_terminal_from_the_longest_gate(tmp_path):
    """`v26b` ran 160 -> resume -> 320 and both chunks write the same per-seed JSON, so between
    them a 160-iteration gate file exists. Reading `iters` from whichever file sorts last would
    call 160 the terminal and delete the entire second chunk, including the published best."""
    _arm(tmp_path, "ppo_v26b_s0", range(1, 321))
    _gate(tmp_path, "gate_chunk1.json", {
        "iters": 160,
        "records": [{"out_dir": "checkpoints/ppo_v26b_s0"}],
    })
    _gate(tmp_path, "gate_chunk2.json", {
        "iters": 320,
        "records": [{"out_dir": "checkpoints/ppo_v26b_s0"}],
    })

    assert terminal_iters(tmp_path / "results")["checkpoints/ppo_v26b_s0"] == 320

    plan = plan_prune(
        tmp_path / "checkpoints",
        referenced_checkpoints(tmp_path / "results"),
        terminal_iters(tmp_path / "results"),
    )
    assert "checkpoints/ppo_v26b_s0/iter_320.pt" in set(plan.arms[0].keep)
    assert "checkpoints/ppo_v26b_s0/iter_160.pt" in set(plan.paths())


def test_a_terminal_that_never_reached_disk_refuses_the_whole_arm(tmp_path):
    """The unchunked `v26b` attempt died OOM at 304 of 320. An arm in that state must not be
    pruned to a terminal it does not have -- the gate would then be unreproducible either way."""
    _arm(tmp_path, "ppo_x_s0", range(1, 305))
    _gate(tmp_path, "gate.json", {
        "iters": 320,
        "records": [{"out_dir": "checkpoints/ppo_x_s0"}],
    })

    with pytest.raises(PruneError, match="not on disk"):
        plan_prune(
            tmp_path / "checkpoints",
            referenced_checkpoints(tmp_path / "results"),
            terminal_iters(tmp_path / "results"),
        )


def test_curve_json_and_unrecognised_files_survive(tmp_path):
    """`curve.json` is the per-iteration training curve the four-point dose-response (G3) is
    booked from, and anything this script does not recognise is kept rather than guessed at."""
    _arm(tmp_path, "ppo_x_s0", range(1, 11), extra=("resume_state.pt", "notes.txt"))
    _gate(tmp_path, "gate.json", {"iters": 10, "records": [{"out_dir": "checkpoints/ppo_x_s0"}]})

    plan = plan_prune(
        tmp_path / "checkpoints",
        referenced_checkpoints(tmp_path / "results"),
        terminal_iters(tmp_path / "results"),
    )
    kept = set(plan.arms[0].keep)
    assert {"checkpoints/ppo_x_s0/curve.json", "checkpoints/ppo_x_s0/notes.txt"} <= kept
    assert "checkpoints/ppo_x_s0/resume_state.pt" in kept


def test_drop_resume_is_opt_in(tmp_path):
    """Resume state is only needed to EXTEND an arm, and Build 26 booked the update-count axis as
    saturating -- but it is 36 MB against a 5.4 GB arm, so removing it is never the point."""
    _arm(tmp_path, "ppo_x_s0", range(1, 11), extra=("resume_state.pt",))
    _gate(tmp_path, "gate.json", {"iters": 10, "records": [{"out_dir": "checkpoints/ppo_x_s0"}]})

    plan = plan_prune(
        tmp_path / "checkpoints",
        referenced_checkpoints(tmp_path / "results"),
        terminal_iters(tmp_path / "results"),
        keep_resume=False,
    )
    assert "checkpoints/ppo_x_s0/resume_state.pt" in set(plan.paths())


@pytest.mark.skipif(not (ROOT / "checkpoints").is_dir(), reason="no local checkpoint tree")
def test_every_published_build_26_checkpoint_survives_the_real_plan():
    """The integration read: against the real tree, nothing this project has published is a
    deletion candidate. Guards the case where a schema change stops a path being recognised."""
    referenced = referenced_checkpoints(ROOT / "results")
    plan = plan_prune(
        ROOT / "checkpoints", referenced, terminal_iters(ROOT / "results")
    )
    assert referenced, "no checkpoints cited at all -- the scan is broken, not the tree"
    assert referenced.isdisjoint(set(plan.paths()))

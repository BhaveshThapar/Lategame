"""The B6f Stage A gate's arithmetic, without a server.

The verdict rests on two numbers, and only one of them is obvious. Episode LENGTH catches the
12,795-turn pathology that was actually in `data/vgc_rl.npz`; it does NOT catch a short cycle
sitting under the ceiling, which is what a capped run would produce if the cap were the only fix.
UNIQUE OBSERVATIONS PER EPISODE catches that, and the gate requires both.
"""

from __future__ import annotations

import numpy as np
import pytest

from lategame.features.doubles_action_space import GEN9_DOUBLES_SLOT_ACTIONS

pytest.importorskip("numpy")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vgc_loop_probe import (  # noqa: E402
    TOP_DECILE_BAR,
    UNIQUE_OBS_BAR,
    shard_stats,
    verdict,
)

A = GEN9_DOUBLES_SLOT_ACTIONS


def _shard(episode_lengths, *, unique_obs=True, legal_per_slot=4):
    """A synthetic doubles shard with the given per-episode turn counts."""
    obs, mask, reward, done = [], [], [], []
    for ep, length in enumerate(episode_lengths):
        for t in range(length):
            row = np.zeros(8, dtype=np.float32)
            # `unique_obs=False` reproduces the real pathology: the same handful of states
            # re-requested, so the episode is long but carries almost no distinct information.
            row[0] = float(t if unique_obs else t % 3)
            row[1] = float(ep)
            obs.append(row)
            m = np.zeros((2, A), dtype=bool)
            m[0, :legal_per_slot] = True
            m[1, :legal_per_slot] = True
            mask.append(m)
            reward.append(0.0 if t < length - 1 else 1.0)
            done.append(t == length - 1)
    return (
        np.stack(obs),
        np.stack(mask),
        np.asarray(reward, dtype=np.float32),
        np.asarray(done, dtype=bool),
    )


def test_it_reproduces_the_shape_of_the_defect_it_was_written_for():
    """One 500-turn loop beside forty 10-turn battles: the length distribution, the concentration
    and the unique-obs collapse must all show it."""
    obs, mask, reward, done = _shard([500] + [10] * 40, unique_obs=False)
    s = shard_stats(obs, mask, reward, done)
    assert s["n_episodes"] == 41
    assert s["turns_per_episode_max"] == 500
    assert s["turns_per_episode_median"] == 10
    assert s["top1_episode_share"] == pytest.approx(500 / 900, abs=1e-3)
    assert s["unique_obs_ratio_min"] < 0.05, "3 unique states in 500 turns"


def test_a_healthy_shard_clears_both_clauses():
    obs, mask, reward, done = _shard([12] * 60, unique_obs=True)
    s = shard_stats(obs, mask, reward, done)
    assert s["top_decile_share"] <= TOP_DECILE_BAR + 1e-9
    assert s["unique_obs_ratio_mean"] >= UNIQUE_OBS_BAR
    assert s["decision_frac"] == 1.0
    assert s["mean_legal_per_slot_on_decisions"] == pytest.approx(4.0)


def test_the_verdict_needs_BOTH_clauses_not_just_the_length_one():
    """A 39-turn cycle under a 40-turn ceiling passes every length test and is still a loop. That
    is precisely why the gate is not 'episodes got shorter'."""
    healthy = shard_stats(*_shard([12] * 60, unique_obs=True))
    short_cycles = shard_stats(*_shard([12] * 60, unique_obs=False))
    loopy = shard_stats(*_shard([500] + [10] * 40, unique_obs=False))

    assert verdict({"a": loopy, "b": healthy}, "a", "b")["verdict"] == "LOOP_CLOSED"
    # Same episode lengths as the healthy arm, so the concentration clause passes...
    assert short_cycles["top_decile_share"] <= TOP_DECILE_BAR + 1e-9
    # ...and the gate still refuses it.
    v = verdict({"a": loopy, "b": short_cycles}, "a", "b")
    assert v["top_decile_share_ok"] is True
    assert v["unique_obs_ok"] is False
    assert v["verdict"] == "PARTIAL"


def test_forced_turns_are_counted_as_such():
    """The decision density is the number the PPO budget is sized off, so it must read the mask
    rather than the turn count."""
    obs, mask, reward, done = _shard([10] * 5, legal_per_slot=1)
    s = shard_stats(obs, mask, reward, done)
    assert s["decision_frac"] == 0.0
    assert s["n_decision"] == 0
    assert s["mean_legal_per_slot_on_decisions"] == 0.0


def test_an_empty_shard_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="empty shard"):
        shard_stats(
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0, 2, A), dtype=bool),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

"""Recovering the winners-only BC shard from the offline-RL shard.

The original BC shard and the 2,760 replays it came from are both off disk, and re-fetching would
return a different replay set (Showdown's search index is live), confounding every comparison
against an older checkpoint. So the rows are recovered from the RL shard instead, and the question
these tests answer is whether the recovery picks the right POV.

The discriminating test is `test_the_pairwise_rule_beats_the_sign_rule_on_a_shaped_reward`. Both
rules give plausible row counts on real data; only one is correct, and the difference only appears
when the terminal reward carries shaping as well as the victory jump -- which is exactly what
`_episode_rewards` produces and what made the sign rule wrong by 702 rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rebuild_bc_shard import (  # noqa: E402
    RECORDED_BC_ROWS,
    pairs_disagreeing,
    rebuild,
    save,
    segment_episodes,
    winners_by_pair,
)

REAL_SHARD = ROOT / "data" / "gen9ou_v7_rl.npz"
requires_shard = pytest.mark.skipif(
    not REAL_SHARD.exists(), reason="data/ is gitignored; no shard on this machine"
)


def _shard(tmp_path: Path, lengths: list[int], terminal: list[float]) -> Path:
    """A tiny RL shard with the given episode lengths and terminal rewards."""
    rows = sum(lengths)
    obs = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    done = np.zeros(rows, dtype=bool)
    reward = np.zeros(rows, dtype=np.float32)
    at = 0
    for length, last in zip(lengths, terminal, strict=True):
        at += length
        done[at - 1] = True
        reward[at - 1] = last
    path = tmp_path / "rl.npz"
    np.savez_compressed(
        path,
        obs=obs,
        action=np.zeros(rows, dtype=np.int64),
        mask=np.ones((rows, 26), dtype=bool),
        reward=reward,
        done=done,
        obs_version=np.array("v5-09831e17c378"),
        obs_dim=np.array(3),
        battle_format=np.array("gen9ou"),
    )
    return path


# --------------------------------------------------------------------------- #
# Episode segmentation
# --------------------------------------------------------------------------- #


def test_episodes_are_segmented_on_done():
    done = np.array([False, True, False, False, True], dtype=bool)
    episodes = segment_episodes(done)
    assert episodes.starts.tolist() == [0, 2]
    assert episodes.ends.tolist() == [1, 4]
    assert episodes.lengths.tolist() == [2, 3]
    assert len(episodes) == 2


def test_a_shard_with_no_boundaries_is_an_error_rather_than_one_huge_episode():
    with pytest.raises(ValueError, match="all False"):
        segment_episodes(np.zeros(5, dtype=bool))


# --------------------------------------------------------------------------- #
# The winner rule
# --------------------------------------------------------------------------- #


def test_exactly_one_pov_per_pair_is_a_winner():
    won = winners_by_pair(np.array([1.0, -1.0, -2.0, 3.0]))
    assert won.tolist() == [True, False, False, True]


def test_the_pairwise_rule_beats_the_sign_rule_on_a_shaped_reward():
    """The terminal reward is `state_value(terminal) - state_value(last)`, so it mixes the +/-
    victory jump with that step's HP and faint deltas. A POV that LOST while landing a big final
    hit still diffs positive -- and the sign rule then counts both POVs of that replay as winners,
    which is how it over-counts by 702 rows on the real shard."""
    # Pair 0: the loser's last step swung enough that both diffs are positive.
    terminal = np.array([2.5, 0.4, -1.0, 2.0])
    by_sign = (terminal > 0).tolist()
    assert by_sign == [True, True, False, True], "the sign rule takes both POVs of pair 0"
    assert winners_by_pair(terminal).tolist() == [True, False, False, True]


def test_an_unpaired_trailing_episode_falls_back_to_the_sign():
    assert winners_by_pair(np.array([1.0, -1.0, 0.5])).tolist() == [True, False, True]
    assert winners_by_pair(np.array([1.0, -1.0, -0.5])).tolist() == [True, False, False]


def test_a_tie_in_the_pair_resolves_to_the_first_pov_rather_than_neither():
    """Deterministic beats principled here: dropping tied pairs would silently shrink the shard."""
    assert winners_by_pair(np.array([1.0, 1.0])).tolist() == [True, False]


def test_disagreeing_pairs_counts_the_sign_rules_error_surface():
    assert pairs_disagreeing(np.array([1.0, -1.0, 2.0, 3.0, -1.0, -2.0])) == 2


# --------------------------------------------------------------------------- #
# The rebuilt shard
# --------------------------------------------------------------------------- #


def test_the_rebuild_keeps_every_turn_of_each_winning_pov(tmp_path):
    path = _shard(tmp_path, lengths=[3, 2, 4, 1], terminal=[5.0, 1.0, -1.0, 2.0])
    result = rebuild(path)
    assert result.n_episodes == 4
    assert result.n_winner_episodes == 2
    assert result.rows == 3 + 1  # episode 0 (3 turns) + episode 3 (1 turn)
    assert result.obs.shape == (4, 3)


def test_the_rebuilt_shard_carries_its_own_episode_boundaries(tmp_path):
    """History-conditioned training needs them, and the BC schema never had a `done` column."""
    path = _shard(tmp_path, lengths=[3, 2, 4, 1], terminal=[5.0, 1.0, -1.0, 2.0])
    result = rebuild(path)
    assert result.done.tolist() == [False, False, True, True]


def test_the_rebuilt_rows_are_the_original_rows_untouched(tmp_path):
    path = _shard(tmp_path, lengths=[2, 2], terminal=[1.0, -1.0])
    source = np.load(path)
    result = rebuild(path)
    assert np.array_equal(result.obs, source["obs"][:2])


def test_saving_writes_the_collect_schema_plus_done(tmp_path):
    path = _shard(tmp_path, lengths=[2, 2], terminal=[1.0, -1.0])
    out = tmp_path / "bc.npz"
    save(rebuild(path), out)
    written = np.load(out, allow_pickle=False)
    assert set(written.files) == {
        "obs", "action", "mask", "done", "obs_version", "obs_dim", "battle_format"
    }
    assert str(written["obs_version"].item()) == "v5-09831e17c378"


def test_the_saved_shard_loads_through_the_normal_dataset(tmp_path):
    """The whole point of matching `collect.save` is that nothing downstream needs to know."""
    pytest.importorskip("torch")
    from rotomai.data.dataset import BCDataset
    from rotomai.features.encoder import OBS_DIM

    rows = 8
    obs = np.zeros((rows, OBS_DIM), dtype=np.float32)
    done = np.zeros(rows, dtype=bool)
    done[3] = done[7] = True
    path = tmp_path / "rl.npz"
    np.savez_compressed(
        path, obs=obs, action=np.zeros(rows, dtype=np.int64),
        mask=np.ones((rows, 26), dtype=bool),
        reward=np.where(done, np.array([1.0, 1, 1, 2.0] * 2, dtype=np.float32), 0.0),
        done=done, obs_version=np.array("v5-09831e17c378"),
        obs_dim=np.array(OBS_DIM), battle_format=np.array("gen9ou"),
    )
    out = tmp_path / "bc.npz"
    save(rebuild(path), out)
    assert len(BCDataset(out)) > 0
    assert len(BCDataset(out, history=2)) > 0


def test_a_shard_missing_a_trajectory_column_is_refused(tmp_path):
    path = _shard(tmp_path, lengths=[2, 2], terminal=[1.0, -1.0])
    columns = {k: v for k, v in np.load(path).items() if k != "reward"}
    np.savez_compressed(path, **columns)
    with pytest.raises(ValueError, match="not an offline-RL shard"):
        rebuild(path)


# --------------------------------------------------------------------------- #
# The real shard
# --------------------------------------------------------------------------- #


@requires_shard
def test_the_real_shard_reconstructs_to_the_recorded_row_count():
    result = rebuild(REAL_SHARD)
    report = result.report()
    assert result.rows == 61_766
    assert abs(report["delta_rows"]) == 43
    assert report["delta_fraction"] < 0.001, "0.07% of the recorded 61,723"
    assert result.rows_by_sign == 62_425, "the method NOT used, kept as the contrast"
    assert result.rows_by_sign - RECORDED_BC_ROWS == 702


@requires_shard
def test_the_two_rules_actually_disagree_on_the_real_shard():
    """If they agreed, the choice of rule would be untested rather than justified."""
    result = rebuild(REAL_SHARD)
    assert result.disagreeing_pairs > 0
    assert result.rows != result.rows_by_sign

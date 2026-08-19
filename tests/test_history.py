"""Trajectory context: episode boundaries, the window convention, and train == eval.

The load-bearing test is `test_the_live_buffer_and_the_dataset_agree_turn_for_turn`. This project's
two most expensive bugs were both a training observation that did not match the live one, and a
dataset stacker plus a separate live ring buffer is that failure with a third chance to occur. So
both sides come through `data.history`, and this asserts they produce identical tensors.

The second load-bearing test is `test_history_zero_is_byte_identical_to_no_history_at_all`. The
whole design rests on trajectory context being opt-in and free: no `OBS_VERSION` bump, no state-dict
key to ignore, every existing shard and checkpoint untouched. If that stops being true, the
human-replay shard -- whose source replays are gone -- becomes unusable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rotomai.data.history import (
    HistoryBuffer,
    PerBattleHistory,
    episode_start_index,
    stack_history,
    window_indices,
)
from rotomai.features.encoder import OBS_DIM
from rotomai.model.entity_transformer import EntityTransformer
from rotomai.model.factory import build_model, model_metadata

# --------------------------------------------------------------------------- #
# Episode boundaries
# --------------------------------------------------------------------------- #


def test_episode_starts_reset_one_past_each_done():
    done = np.array([False, False, True, False, True, False], dtype=bool)
    assert episode_start_index(done).tolist() == [0, 0, 0, 3, 3, 5]


def test_a_single_episode_starts_everywhere_at_zero():
    assert episode_start_index(np.zeros(4, dtype=bool)).tolist() == [0, 0, 0, 0]


def test_every_row_is_done_makes_every_row_its_own_episode():
    assert episode_start_index(np.ones(4, dtype=bool)).tolist() == [0, 1, 2, 3]


def test_window_indices_pad_rather_than_crossing_an_episode_boundary():
    # Row 4, episode starting at 3: rows 1 and 2 belong to the PREVIOUS battle.
    assert window_indices(4, start=3, history=3) == [-1, -1, 3, 4]


def test_window_indices_are_oldest_first_with_now_last():
    assert window_indices(9, start=0, history=2) == [7, 8, 9]


# --------------------------------------------------------------------------- #
# The window itself
# --------------------------------------------------------------------------- #


def test_stack_history_zero_pads_at_the_head_of_an_episode():
    obs = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    frames = stack_history(obs, index=4, start=3, history=3)
    assert frames.shape == (4, 4)
    assert torch.equal(frames[0], torch.zeros(4))
    assert torch.equal(frames[1], torch.zeros(4))
    assert torch.equal(frames[2], obs[3])
    assert torch.equal(frames[3], obs[4])


def test_stack_history_never_reaches_into_the_previous_episode():
    obs = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    frames = stack_history(obs, index=3, start=3, history=2)
    assert torch.equal(frames[2], obs[3])
    assert not any(torch.equal(frames[i], obs[2]) for i in range(3))


def test_history_zero_returns_the_single_turn_unchanged():
    obs = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert torch.equal(stack_history(obs, 2, 0, 0), obs[2].unsqueeze(0))


def test_a_negative_history_is_rejected():
    with pytest.raises(ValueError):
        window_indices(1, 0, -1)
    with pytest.raises(ValueError):
        HistoryBuffer(-1, 4)


# --------------------------------------------------------------------------- #
# The live buffer
# --------------------------------------------------------------------------- #


def test_the_buffer_right_aligns_the_current_turn():
    buffer = HistoryBuffer(history=2, obs_dim=3)
    first = buffer.push(np.array([1.0, 1.0, 1.0]))
    assert first.shape == (3, 3)
    assert np.array_equal(first[:2], np.zeros((2, 3)))
    assert np.array_equal(first[2], np.ones(3))


def test_the_buffer_drops_the_oldest_frame_past_the_window():
    buffer = HistoryBuffer(history=1, obs_dim=2)
    for value in (1.0, 2.0, 3.0):
        window = buffer.push(np.full(2, value))
    assert np.array_equal(window, np.array([[2.0, 2.0], [3.0, 3.0]]))


def test_reset_clears_the_previous_battle():
    """A buffer carried across battles feeds the last opponent's board into turn 1 of the next."""
    buffer = HistoryBuffer(history=2, obs_dim=2)
    buffer.push(np.full(2, 9.0))
    buffer.reset()
    window = buffer.push(np.full(2, 1.0))
    assert np.array_equal(window[:2], np.zeros((2, 2)))


def test_a_wrong_width_observation_is_rejected():
    with pytest.raises(ValueError, match="expected a 5-d observation"):
        HistoryBuffer(1, 5).push(np.zeros(4))


def test_concurrent_battles_do_not_interleave():
    """poke-env runs several battles through one Player, so per-battle state must key on the tag."""
    tracker = PerBattleHistory(history=1, obs_dim=2)
    tracker.observe("battle-a", np.full(2, 1.0))
    tracker.observe("battle-b", np.full(2, 7.0))
    window = tracker.observe("battle-a", np.full(2, 2.0))
    assert np.array_equal(window, np.array([[1.0, 1.0], [2.0, 2.0]]))


# --------------------------------------------------------------------------- #
# Train == eval
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("history", [1, 2, 4])
def test_the_live_buffer_and_the_dataset_agree_turn_for_turn(history):
    """One battle, played turn by turn live and read row by row offline, must give one answer."""
    rng = np.random.default_rng(0)
    turns = rng.normal(size=(7, 5)).astype(np.float32)
    obs = torch.from_numpy(turns)
    done = np.zeros(len(turns), dtype=bool)
    done[-1] = True
    starts = episode_start_index(done)

    buffer = HistoryBuffer(history, obs_dim=5)
    for index in range(len(turns)):
        live = buffer.push(turns[index])
        offline = stack_history(obs, index, int(starts[index]), history)
        assert np.allclose(live, offline.numpy()), f"turn {index} diverged"


def test_the_two_sides_agree_across_a_battle_boundary():
    """The live buffer resets between battles; the dataset stops at `done` -- one rule, twice."""
    rng = np.random.default_rng(1)
    turns = rng.normal(size=(6, 3)).astype(np.float32)
    obs = torch.from_numpy(turns)
    done = np.array([False, False, True, False, False, True])
    starts = episode_start_index(done)

    tracker = PerBattleHistory(history=2, obs_dim=3)
    for index in range(len(turns)):
        tag = "battle-0" if index < 3 else "battle-1"
        live = tracker.observe(tag, turns[index])
        assert np.allclose(live, stack_history(obs, index, int(starts[index]), 2).numpy())


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def _small(**kw):
    return EntityTransformer(OBS_DIM, d_model=16, n_layers=1, n_heads=2, ff_dim=16, **kw)


def test_history_zero_is_byte_identical_to_no_history_at_all():
    """The opt-in has to be free, or the surviving human-replay shard is stranded."""
    torch.manual_seed(0)
    baseline = _small()
    torch.manual_seed(0)
    explicit = _small(history=0)
    assert baseline.state_dict().keys() == explicit.state_dict().keys()
    assert "time_embed" not in baseline.state_dict()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key]), key
    assert baseline.arch_config() == explicit.arch_config()
    assert "history" not in baseline.arch_config()


def test_a_history_model_carries_a_time_embedding_and_says_so():
    model = _small(history=3)
    assert "time_embed" in model.state_dict()
    assert model.state_dict()["time_embed"].shape == (1, 4, 1, 16)
    assert model.arch_config()["history"] == 3


def test_a_history_model_round_trips_through_the_factory():
    model = _small(history=2)
    rebuilt = build_model(model_metadata(model) | {"input_dim": OBS_DIM, "dropout": 0.1})
    assert isinstance(rebuilt, EntityTransformer)
    assert rebuilt.history == 2
    assert rebuilt.state_dict().keys() == model.state_dict().keys()
    rebuilt.load_state_dict(model.state_dict())  # strict: no key may be missing or extra


def test_the_forward_pass_accepts_stacked_and_flat_windows():
    model = _small(history=2).eval()
    stacked = torch.zeros(3, 3, OBS_DIM)
    with torch.no_grad():
        a = model(stacked)
        b = model(stacked.reshape(3, 3 * OBS_DIM))
    assert torch.allclose(a[0], b[0])
    assert a[0].shape == (3, model.n_actions)


def test_a_wrongly_shaped_window_is_refused_rather_than_reinterpreted():
    model = _small(history=2)
    with pytest.raises(ValueError, match=r"expects obs of shape"):
        model(torch.zeros(3, 2, OBS_DIM))
    with pytest.raises(ValueError, match=r"expects obs of shape"):
        model(torch.zeros(3, OBS_DIM))


def test_a_negative_history_is_refused_at_construction():
    with pytest.raises(ValueError, match="history must be >= 0"):
        _small(history=-1)


def test_the_token_count_grows_with_the_window():
    model = _small(history=3).eval()
    tokens, padding = model._tokenize_window(torch.zeros(2, 4, OBS_DIM))
    assert tokens.shape[1] == 4 * model._n_tokens
    assert padding.shape == (2, 4 * model._n_tokens)


def test_a_padded_frame_is_fully_masked_including_its_global_token():
    """A zero frame's Pokemon and move tokens self-mask; the global token has no present flag."""
    model = _small(history=1).eval()
    window = torch.zeros(1, 2, OBS_DIM)
    window[0, 1] = 1.0  # only the current turn is real
    _, padding = model._tokenize_window(window)
    per_frame = padding.view(1, 2, model._n_tokens)
    assert bool(per_frame[0, 0].all()), "every token of the padded frame must be masked"
    assert not bool(per_frame[0, 1, -1]), "the real frame's global token must stay visible"


def test_trajectory_context_actually_reaches_the_output():
    """A model that ignored the earlier frames would be a very expensive no-op."""
    torch.manual_seed(0)
    model = _small(history=2).eval()
    now = torch.randn(1, 1, OBS_DIM).abs()
    a = torch.cat([torch.randn(1, 2, OBS_DIM).abs(), now], dim=1)
    b = torch.cat([torch.randn(1, 2, OBS_DIM).abs(), now], dim=1)
    with torch.no_grad():
        assert not torch.allclose(model(a)[0], model(b)[0])

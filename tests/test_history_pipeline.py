"""History-conditioned training end to end, without a server.

`tests/test_history.py` pins the window convention and the model surface. This pins the parts that
only show up once a real trainer, a real shard and a real checkpoint are involved: that BC and AWR
run at history > 0, that the checkpoint records it, that the agent rebuilt from that checkpoint
allocates a live buffer, and that the RELEASED flat checkpoint is completely unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rotomai.data.dataset import BCDataset  # noqa: E402
from rotomai.data.rl_dataset import RLDataset  # noqa: E402
from rotomai.features.action_space import GEN9_ACTION_SPACE_SIZE  # noqa: E402
from rotomai.features.encoder import OBS_DIM, OBS_VERSION  # noqa: E402

RELEASED = Path(__file__).resolve().parents[1] / "checkpoints/ppo_v26b_s0/iter_320.pt"


def _write_shard(path: Path, *, rows: int = 40, episode: int = 5, rl: bool = False) -> None:
    """A tiny shard with real episode boundaries and legal actions."""
    rng = np.random.default_rng(0)
    # Uniform [0, 1): the trailing ID channels are read as vocabulary indices via `.long()`, so
    # anything wider than the vocab indexes an embedding out of range. In [0, 1) they all floor to
    # 0, which is the padding index -- meaningless features, valid tensors, which is what a
    # plumbing test wants.
    obs = rng.random(size=(rows, OBS_DIM)).astype(np.float32)
    action = rng.integers(0, GEN9_ACTION_SPACE_SIZE, size=rows).astype(np.int64)
    mask = np.zeros((rows, GEN9_ACTION_SPACE_SIZE), dtype=bool)
    mask[np.arange(rows), action] = True
    mask[:, 0] = True
    done = np.zeros(rows, dtype=bool)
    done[episode - 1 :: episode] = True
    done[-1] = True

    columns = dict(
        obs=obs,
        action=action,
        mask=mask,
        done=done,
        obs_version=np.array(OBS_VERSION),
        obs_dim=np.array(OBS_DIM),
        battle_format=np.array("gen9randombattle"),
    )
    if rl:
        columns |= dict(
            reward=rng.normal(size=rows).astype(np.float32),
            gamma=np.array(0.99, dtype=np.float32),
            hp_value=np.array(0.3, dtype=np.float32),
            fainted_value=np.array(1.0, dtype=np.float32),
            status_value=np.array(0.1, dtype=np.float32),
            victory_value=np.array(1.0, dtype=np.float32),
        )
    np.savez_compressed(path, **columns)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def test_a_bc_dataset_yields_a_window_at_history(tmp_path):
    path = tmp_path / "bc.npz"
    _write_shard(path)
    flat, stacked = BCDataset(path), BCDataset(path, history=3)
    assert flat[7][0].shape == (OBS_DIM,)
    frames, action, mask = stacked[7]
    assert frames.shape == (4, OBS_DIM)
    assert torch.equal(frames[-1], flat[7][0]), "the last frame is always the current turn"
    assert torch.equal(action, flat[7][1]) and torch.equal(mask, flat[7][2])


def test_a_bc_window_stops_at_the_episode_head(tmp_path):
    path = tmp_path / "bc.npz"
    _write_shard(path, episode=5)
    stacked = BCDataset(path, history=3)
    frames, _, _ = stacked[5]  # first row of the second episode
    assert torch.equal(frames[:3], torch.zeros(3, OBS_DIM))


def test_history_without_episode_boundaries_is_refused(tmp_path):
    """The BC schema predates `done`; a shard without it cannot be windowed safely."""
    path = tmp_path / "legacy.npz"
    _write_shard(path)
    columns = {k: v for k, v in np.load(path).items() if k != "done"}
    np.savez_compressed(path, **columns)
    BCDataset(path)  # flat still loads -- no regression for anything that exists
    with pytest.raises(ValueError, match="rebuild_bc_shard"):
        BCDataset(path, history=2)


def test_an_rl_dataset_windows_without_a_new_column(tmp_path):
    """An offline-RL shard always carries `done`; the MC return already resets on it."""
    path = tmp_path / "rl.npz"
    _write_shard(path, rl=True)
    flat, stacked = RLDataset(path), RLDataset(path, history=2)
    frames, action, mask, ret = stacked[9]
    assert frames.shape == (3, OBS_DIM)
    assert torch.equal(frames[-1], flat[9][0])
    assert torch.equal(ret, flat[9][3]), "windowing must not disturb the return target"
    assert torch.equal(action, flat[9][1]) and torch.equal(mask, flat[9][2])


# --------------------------------------------------------------------------- #
# Trainers
# --------------------------------------------------------------------------- #


def test_bc_trains_at_history_and_stamps_it_into_the_checkpoint(tmp_path):
    from rotomai.train.bc import TrainConfig, train_bc

    data, out = tmp_path / "bc.npz", tmp_path / "ckpt.pt"
    _write_shard(data, rows=60)
    train_bc(
        data,
        out,
        TrainConfig(
            epochs=1, batch_size=8, device="cpu", model_type="entity_transformer", history=2
        ),
    )
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["arch"]["history"] == 2
    assert "time_embed" in ckpt["state_dict"]


def test_the_flat_mlp_refuses_history_rather_than_ignoring_it(tmp_path):
    from rotomai.train.bc import TrainConfig, train_bc

    data = tmp_path / "bc.npz"
    _write_shard(data)
    with pytest.raises(ValueError, match="flat MLP over one turn"):
        train_bc(data, tmp_path / "c.pt", TrainConfig(epochs=1, device="cpu", history=2))


def test_pp_augmentation_and_history_are_refused_together(tmp_path):
    """The pp augmentations index a single turn's OBS_LAYOUT offsets; over a window they would
    perturb whichever frame the offsets landed in -- a corrupted experiment, not a smaller one."""
    from rotomai.train.bc import TrainConfig, train_bc

    data = tmp_path / "bc.npz"
    _write_shard(data)
    with pytest.raises(ValueError, match="not defined over a stacked window"):
        train_bc(
            data,
            tmp_path / "c.pt",
            TrainConfig(
                epochs=1, device="cpu", model_type="entity_transformer",
                history=2, pp_aug_frac=0.5,
            ),
        )


def test_offline_rl_trains_at_history_and_warm_starts_from_a_matching_bc(tmp_path):
    from rotomai.train.bc import TrainConfig, train_bc
    from rotomai.train.offline_rl import OfflineRLConfig, train_offline_rl

    bc_data, bc_out = tmp_path / "bc.npz", tmp_path / "bc.pt"
    _write_shard(bc_data, rows=60)
    train_bc(
        bc_data, bc_out,
        TrainConfig(epochs=1, batch_size=8, device="cpu",
                    model_type="entity_transformer", history=2),
    )

    rl_data, rl_out = tmp_path / "rl.npz", tmp_path / "offrl.pt"
    _write_shard(rl_data, rows=60, rl=True)
    train_offline_rl(
        rl_data, rl_out,
        OfflineRLConfig(epochs=1, batch_size=8, device="cpu",
                        model_type="entity_transformer", history=2, bc_init=str(bc_out)),
    )
    ckpt = torch.load(rl_out, map_location="cpu", weights_only=False)
    assert ckpt["arch"]["history"] == 2


# --------------------------------------------------------------------------- #
# The agent, and the released checkpoint
# --------------------------------------------------------------------------- #


def test_an_agent_built_from_a_history_checkpoint_allocates_a_live_buffer(tmp_path):
    from rotomai.agents.history_input import _history_tracker
    from rotomai.model.entity_transformer import EntityTransformer

    flat = EntityTransformer(OBS_DIM, d_model=16, n_layers=1, n_heads=2, ff_dim=16)
    windowed = EntityTransformer(
        OBS_DIM, d_model=16, n_layers=1, n_heads=2, ff_dim=16, history=3
    )
    assert _history_tracker(flat, OBS_DIM) is None
    tracker = _history_tracker(windowed, OBS_DIM)
    assert tracker is not None and tracker.history == 3


@pytest.mark.skipif(not RELEASED.exists(), reason="checkpoints/ is gitignored")
def test_the_released_checkpoint_is_untouched_by_trajectory_context():
    """Every published gen9ou number was produced by this file. It must load, build and act
    exactly as before -- strict state dict, same parameter count, same recorded arch block."""
    from rotomai.model.factory import build_model, model_metadata

    ckpt = torch.load(RELEASED, map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    model.load_state_dict(ckpt["state_dict"])  # strict
    model.eval()

    assert model.history == 0
    assert sum(p.numel() for p in model.parameters()) == 4_555_629
    assert model_metadata(model)["arch"] == ckpt["arch"], "the arch block must not have grown"
    assert "time_embed" not in ckpt["state_dict"]

    with torch.no_grad():
        policy, value = model(torch.zeros(2, int(ckpt["input_dim"])))
    assert policy.shape == (2, GEN9_ACTION_SPACE_SIZE)
    assert value.shape == (2, int(ckpt["n_bins"]))

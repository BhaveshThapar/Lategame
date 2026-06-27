"""Unit tests for the M5 entity transformer, model factory, and warm-start path.

Covers (no Showdown server needed): forward shapes, that flat-obs slicing matches
the encoder block layout, factory dispatch + checkpoint metadata round-trip, the
AC->AC warm-start round-trip, and an end-to-end ``train_offline_rl`` run on a
synthetic shard (fresh + self-play continuation). Skipped if torch is absent.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lategame.features.encoder import (  # noqa: E402
    MOVE_ID_FIELDS,
    OBS_DIM,
    OBS_LAYOUT,
    OBS_VERSION,
    POKEMON_ID_FIELDS,
)
from lategame.features.vocab import vocab_sizes  # noqa: E402
from lategame.model.actor_critic import ActorCritic  # noqa: E402
from lategame.model.entity_transformer import EntityTransformer  # noqa: E402
from lategame.model.factory import (  # noqa: E402
    MODEL_ENTITY_TRANSFORMER,
    build_model,
    model_metadata,
)

_SMALL = dict(d_model=32, n_layers=1, n_heads=2, ff_dim=64)


def _tiny_transformer(n_bins=21, id_embed=True):
    return EntityTransformer(OBS_DIM, n_bins=n_bins, id_embed=id_embed, **_SMALL)


def _valid_obs(n, rng=None):
    """Random obs whose trailing ID channels hold valid (in-range) embedding indices."""
    rng = rng or np.random.default_rng(0)
    sizes = vocab_sizes()
    layout = OBS_LAYOUT
    obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    for tok in range(layout.n_pokemon_tokens):
        base = tok * layout.pokemon_dim + layout.pokemon_numeric_dim
        for off, cat in enumerate(POKEMON_ID_FIELDS):
            obs[:, base + off] = rng.integers(0, sizes[cat], size=n)
    for tok in range(layout.n_moves):
        base = layout.moves_start + tok * layout.move_dim + layout.move_numeric_dim
        for off, cat in enumerate(MOVE_ID_FIELDS):
            obs[:, base + off] = rng.integers(0, sizes[cat], size=n)
    return obs


def test_forward_shapes():
    model = _tiny_transformer(n_bins=21)
    logits, value_logits = model(torch.zeros(4, OBS_DIM))
    assert logits.shape == (4, 26)
    assert value_logits.shape == (4, 21)


def test_rejects_wrong_input_dim():
    with pytest.raises(ValueError):
        EntityTransformer(OBS_DIM + 1, **_SMALL)


def test_layout_offsets_cover_obs_exactly():
    layout = OBS_LAYOUT
    assert layout.n_tokens == layout.team_size * 2 + layout.n_moves + 1
    assert layout.moves_start == layout.n_pokemon_tokens * layout.pokemon_dim
    assert layout.global_start == layout.moves_start + layout.n_moves * layout.move_dim
    assert layout.global_start + layout.global_dim == OBS_DIM


def test_tokenize_padding_mask_tracks_present_flags():
    # All-absent obs except ego-mon 0 and move 0 (their "present" flag is feature 0
    # of the block). The padding mask must mark exactly those two real tokens (plus
    # the always-present global) as attend-able.
    obs = torch.zeros(1, OBS_DIM)
    obs[0, 0] = 1.0  # ego Pokemon slot 0 present
    obs[0, OBS_LAYOUT.moves_start] = 1.0  # move slot 0 present

    model = _tiny_transformer()
    _, padding_mask = model._tokenize(obs)

    assert padding_mask.shape == (1, OBS_LAYOUT.n_tokens)
    keep = ~padding_mask[0]  # True where attend-able
    expected = torch.zeros(OBS_LAYOUT.n_tokens, dtype=torch.bool)
    expected[0] = True  # ego mon 0
    expected[OBS_LAYOUT.n_pokemon_tokens] = True  # first move token
    expected[-1] = True  # global token always present
    assert torch.equal(keep, expected)


def test_build_model_dispatch():
    et = build_model(
        {"model_type": MODEL_ENTITY_TRANSFORMER, "input_dim": OBS_DIM, "n_bins": 21, "arch": _SMALL}
    )
    assert isinstance(et, EntityTransformer)
    assert et.n_bins == 21 and et.d_model == 32 and et.n_heads == 2

    ac = build_model({"model_type": "actor_critic", "input_dim": OBS_DIM, "hidden_dim": 16})
    assert isinstance(ac, ActorCritic)

    # Missing model_type defaults to the MLP (backward compat with M3 checkpoints).
    assert isinstance(build_model({"input_dim": OBS_DIM, "hidden_dim": 16}), ActorCritic)


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError):
        build_model({"model_type": "nonsense", "input_dim": OBS_DIM})


def test_metadata_roundtrip_reproduces_outputs():
    # model_metadata(m) + input_dim is enough for build_model to rebuild m exactly,
    # and load_actor_critic_weights (arch-agnostic) restores identical outputs --
    # this is the AC->AC self-play warm-start contract for the transformer.
    from lategame.model.actor_critic import load_actor_critic_weights

    src = _tiny_transformer(n_bins=21).eval()
    meta = {**model_metadata(src), "input_dim": OBS_DIM}
    rebuilt = build_model(meta).eval()
    load_actor_critic_weights(rebuilt, src.state_dict())

    x = torch.from_numpy(_valid_obs(5))
    with torch.no_grad():
        a_logits, a_val = src(x)
        b_logits, b_val = rebuilt(x)
    assert torch.allclose(a_logits, b_logits, atol=1e-6)
    assert torch.allclose(a_val, b_val, atol=1e-6)


def test_id_embeddings_present_when_enabled_and_absent_when_off():
    sizes = vocab_sizes()
    on = _tiny_transformer(id_embed=True)
    assert set(on.embeddings.keys()) == {f"emb_{c}" for c in (*POKEMON_ID_FIELDS, *MOVE_ID_FIELDS)}
    assert on.embeddings["emb_species"].weight.shape[0] == sizes["species"]

    off = _tiny_transformer(id_embed=False)
    assert len(off.embeddings) == 0
    # Fewer params without the embedding tables + wider projections.
    assert sum(p.numel() for p in off.parameters()) < sum(p.numel() for p in on.parameters())

    x = torch.from_numpy(_valid_obs(3))
    for model in (on, off):
        logits, value = model(x)
        assert logits.shape == (3, 26)
        assert value.shape == (3, 21)


def test_id_embed_flag_round_trips_through_metadata():
    off = _tiny_transformer(id_embed=False).eval()
    meta = {**model_metadata(off), "input_dim": OBS_DIM}
    assert meta["arch"]["id_embed"] is False
    rebuilt = build_model(meta)
    assert isinstance(rebuilt, EntityTransformer) and rebuilt.id_embed is False
    rebuilt.load_state_dict(off.state_dict())


def _make_rl_npz(path, n=64, episode_len=16):
    from lategame.data.collect import TrajectoryDataset, save_rl
    from lategame.data.reward import RewardWeights

    rng = np.random.default_rng(0)
    obs = _valid_obs(n, rng)
    action = rng.integers(0, 26, size=n).astype(np.int64)
    mask = np.ones((n, 26), dtype=bool)
    reward = rng.standard_normal(n).astype(np.float32)
    done = np.zeros(n, dtype=bool)
    done[episode_len - 1 :: episode_len] = True
    done[-1] = True
    ds = TrajectoryDataset(
        obs, action, mask, reward, done, "gen9randombattle", 0.99, RewardWeights()
    )
    save_rl(ds, path)


def test_train_offline_rl_transformer_end_to_end(tmp_path):
    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    data_path = tmp_path / "rl.npz"
    _make_rl_npz(data_path)
    ckpt_path = tmp_path / "et_offrl.pt"

    metrics = train_offline_rl(
        data_path,
        ckpt_path,
        OfflineRLConfig(
            epochs=1,
            batch_size=32,
            n_bins=21,
            device="cpu",
            model_type=MODEL_ENTITY_TRANSFORMER,
            **_SMALL,
        ),
    )
    assert "value_mae" in metrics
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["model_type"] == MODEL_ENTITY_TRANSFORMER
    assert ckpt["arch"]["d_model"] == 32
    # The checkpoint rebuilds via the factory and loads its own weights.
    model = build_model(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    assert isinstance(model, EntityTransformer)


def test_train_offline_rl_transformer_selfplay_continuation(tmp_path):
    # AC->AC warm-start: a second train run pointed at the transformer checkpoint
    # must rebuild the transformer (from the init arch) even with the default
    # config model_type, and re-save it as a transformer.
    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    data_path = tmp_path / "rl.npz"
    _make_rl_npz(data_path)
    first = tmp_path / "iter0.pt"
    train_offline_rl(
        data_path,
        first,
        OfflineRLConfig(
            epochs=1, batch_size=32, n_bins=21, device="cpu",
            model_type=MODEL_ENTITY_TRANSFORMER, **_SMALL,
        ),
    )
    ckpt0 = torch.load(first, map_location="cpu", weights_only=False)

    second = tmp_path / "iter1.pt"
    train_offline_rl(
        data_path,
        second,
        OfflineRLConfig(
            epochs=1, batch_size=32, n_bins=21, device="cpu",
            bc_init=str(first), v_min=ckpt0["v_min"], v_max=ckpt0["v_max"],
        ),
    )
    ckpt1 = torch.load(second, map_location="cpu", weights_only=False)
    assert ckpt1["model_type"] == MODEL_ENTITY_TRANSFORMER
    assert ckpt1["v_min"] == ckpt0["v_min"] and ckpt1["v_max"] == ckpt0["v_max"]


def test_train_bc_dispatches_entity_transformer(tmp_path):
    # BC is no longer hardcoded to the flat MLP: model_type routes through the factory,
    # and the checkpoint stamps model_type/arch so the bc agent rebuilds the transformer.
    from lategame.data.collect import Dataset, save
    from lategame.train.bc import TrainConfig, train_bc

    rng = np.random.default_rng(1)
    obs = _valid_obs(64, rng)
    action = rng.integers(0, 26, size=64).astype(np.int64)
    mask = np.ones((64, 26), dtype=bool)
    data_path = tmp_path / "bc.npz"
    save(Dataset(obs, action, mask, "gen9randombattle"), data_path)

    ckpt_path = tmp_path / "bc_et.pt"
    metrics = train_bc(
        data_path,
        ckpt_path,
        TrainConfig(
            epochs=1, batch_size=32, device="cpu",
            model_type=MODEL_ENTITY_TRANSFORMER, id_embed=True,
        ),
    )
    assert "val_acc" in metrics
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["model_type"] == MODEL_ENTITY_TRANSFORMER
    assert ckpt["arch"]["id_embed"] is True
    model = build_model(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    assert isinstance(model, EntityTransformer)


def test_bc_warm_start_into_transformer_rejected(tmp_path):
    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    data_path = tmp_path / "rl.npz"
    _make_rl_npz(data_path)
    bc_path = tmp_path / "bc.pt"
    torch.save(
        {
            "state_dict": {},
            "input_dim": OBS_DIM,
            "hidden_dim": 32,
            "n_actions": 26,
            "obs_version": OBS_VERSION,
            "battle_format": "gen9randombattle",
            "metrics": {},
        },  # no model_type -> treated as a BC checkpoint
        bc_path,
    )
    with pytest.raises(ValueError):
        train_offline_rl(
            data_path,
            tmp_path / "out.pt",
            OfflineRLConfig(
                epochs=1, batch_size=32, n_bins=21, device="cpu",
                bc_init=str(bc_path), model_type=MODEL_ENTITY_TRANSFORMER, **_SMALL,
            ),
        )

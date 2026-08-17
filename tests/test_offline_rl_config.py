"""OfflineRLConfig must pass identity-embedding settings through to the factory.

Regression guard for the R-ENCODE gap: offrl's ``arch()`` previously omitted the
``id_embed`` keys, so a transformer offrl run silently fell back to random-init
embeddings even when ``prior`` was requested. These tests pin the config -> factory
-> applied-prior path without needing a server or training.
"""

import numpy as np
import pytest
import torch

from rotomai.features.action_space import GEN9_ACTION_SPACE_SIZE
from rotomai.features.embed_prior import load_id_priors
from rotomai.features.encoder import OBS_DIM
from rotomai.model.entity_transformer import EntityTransformer
from rotomai.model.factory import MODEL_ENTITY_TRANSFORMER, build_model
from rotomai.train.offline_rl import OfflineRLConfig


def _meta(config: OfflineRLConfig) -> dict:
    """Mirror the metadata dict ``train_offline_rl`` hands to ``build_model``."""
    return {
        "model_type": config.model_type,
        "input_dim": OBS_DIM,
        "n_actions": GEN9_ACTION_SPACE_SIZE,
        "n_bins": config.n_bins,
        "dropout": config.dropout,
        "hidden_dim": config.hidden_dim,
        "arch": config.arch(),
    }


def test_arch_carries_id_embed_keys():
    arch = OfflineRLConfig(
        model_type=MODEL_ENTITY_TRANSFORMER, id_embed=True, id_embed_init="prior"
    ).arch()
    assert arch["id_embed"] is True
    assert arch["id_embed_dim"] == 32
    assert arch["id_embed_init"] == "prior"


def test_build_model_applies_prior_init():
    config = OfflineRLConfig(model_type=MODEL_ENTITY_TRANSFORMER, id_embed_init="prior")
    model = build_model(_meta(config))
    assert isinstance(model, EntityTransformer)
    assert model.arch_config()["id_embed_init"] == "prior"
    # The prior must actually be loaded into the species embedding -- not a silent
    # random fallback. Padding row stays zero; structured rows match the dex prior.
    prior = load_id_priors(config.id_embed_dim)["species"]
    weight = model.embeddings["emb_species"].weight.detach().numpy()
    assert np.allclose(weight, prior)


def test_random_init_differs_from_prior():
    torch.manual_seed(0)
    rnd = build_model(_meta(OfflineRLConfig(model_type=MODEL_ENTITY_TRANSFORMER)))
    prior = load_id_priors(32)["species"]
    weight = rnd.embeddings["emb_species"].weight.detach().numpy()
    assert not np.allclose(weight, prior)


def test_id_embed_off_drops_embeddings():
    off = build_model(_meta(OfflineRLConfig(model_type=MODEL_ENTITY_TRANSFORMER, id_embed=False)))
    assert isinstance(off, EntityTransformer)
    assert len(off.embeddings) == 0


def _write_rl_shard(path) -> None:
    """A minimal valid offline-RL shard (one episode) so train_offline_rl can run."""
    from rotomai.features.encoder import OBS_VERSION

    k = 8
    np.savez_compressed(
        path,
        obs=np.random.randn(k, OBS_DIM).astype(np.float32),
        action=np.zeros(k, dtype=np.int64),
        mask=np.ones((k, GEN9_ACTION_SPACE_SIZE), dtype=bool),
        reward=np.full(k, 0.1, dtype=np.float32),
        done=np.array([False] * (k - 1) + [True]),
        obs_version=np.array(OBS_VERSION),
        obs_dim=np.array(OBS_DIM),
        gamma=np.array(0.99),
        battle_format=np.array("gen9randombattle"),
    )


def _write_bc_policy_checkpoint(path, hidden_dim=32) -> None:
    """A flat-MLP BC checkpoint stamped exactly as ``train_bc`` writes it (BC_POLICY)."""
    from rotomai.features.encoder import OBS_VERSION
    from rotomai.model.policy import BCPolicy
    from rotomai.train.bc import BC_POLICY

    model = BCPolicy(OBS_DIM, hidden_dim=hidden_dim)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": BC_POLICY,
            "input_dim": OBS_DIM,
            "hidden_dim": hidden_dim,
            "n_actions": model.n_actions,
            "obs_version": OBS_VERSION,
            "battle_format": "gen9randombattle",
            "metrics": {},
        },
        path,
    )


def test_bc_policy_checkpoint_warm_starts_offrl(tmp_path):
    """Regression: a ``bc_policy``-stamped checkpoint must take the BC->AC warm-start
    path. It previously fell through to AC->AC and raised KeyError('n_bins')."""
    from rotomai.train.offline_rl import OfflineRLConfig, train_offline_rl

    data = tmp_path / "rl.npz"
    bc = tmp_path / "bc.pt"
    out = tmp_path / "offrl.pt"
    _write_rl_shard(data)
    _write_bc_policy_checkpoint(bc, hidden_dim=32)

    cfg = OfflineRLConfig(epochs=1, batch_size=4, hidden_dim=32, device="cpu", bc_init=str(bc))
    train_offline_rl(data, out, cfg)
    assert out.exists()


# --- Build 21: the arch flags must reach the model, and must never be discarded ---


def _write_transformer_ac_checkpoint(path, arch: dict, n_bins: int = 51) -> None:
    """An ``entity_transformer`` actor-critic checkpoint, stamped as offrl writes it."""
    from rotomai.features.encoder import OBS_VERSION

    meta = {
        "model_type": MODEL_ENTITY_TRANSFORMER,
        "input_dim": OBS_DIM,
        "n_actions": GEN9_ACTION_SPACE_SIZE,
        "n_bins": n_bins,
        "dropout": 0.1,
        "hidden_dim": 256,
        "arch": arch,
    }
    model = build_model(meta)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": MODEL_ENTITY_TRANSFORMER,
            "arch": arch,
            "input_dim": OBS_DIM,
            "hidden_dim": 256,
            "n_actions": GEN9_ACTION_SPACE_SIZE,
            "n_bins": n_bins,
            "obs_version": OBS_VERSION,
            "battle_format": "gen9randombattle",
            "v_min": -1.0,
            "v_max": 1.0,
            "metrics": {},
        },
        path,
    )


def test_arch_carries_ff_dim():
    """ff_dim was reachable on the dataclass but had no CLI flag, so a 'wide' net
    silently kept ff_dim=256 -- an FFN expansion ratio of 1.0x, not the baseline's 2.0x."""
    assert OfflineRLConfig(model_type=MODEL_ENTITY_TRANSFORMER, ff_dim=512).arch()["ff_dim"] == 512


def test_ff_dim_sizes_the_feedforward():
    wide = build_model(_meta(OfflineRLConfig(model_type=MODEL_ENTITY_TRANSFORMER, ff_dim=512)))
    narrow = build_model(_meta(OfflineRLConfig(model_type=MODEL_ENTITY_TRANSFORMER, ff_dim=256)))
    assert wide.arch_config()["ff_dim"] == 512
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in narrow.parameters())


def _run_cli_train_rl(monkeypatch, argv: list[str]):
    """Parse a train-rl command line and capture the OfflineRLConfig it would train with."""
    import rotomai.train.offline_rl as offrl
    from rotomai.cli import _run_train_rl, build_parser

    captured = {}

    def _fake_train(data, out, config):
        captured["config"] = config
        return {}

    # _run_train_rl imports train_offline_rl at call time, so patching the module attr lands.
    monkeypatch.setattr(offrl, "train_offline_rl", _fake_train)
    _run_train_rl(build_parser().parse_args(argv))
    return captured["config"]


def test_cli_arch_flags_reach_the_config(monkeypatch):
    config = _run_cli_train_rl(
        monkeypatch,
        [
            "train-rl",
            "--model",
            "entity_transformer",
            "--d-model",
            "256",
            "--n-layers",
            "4",
            "--n-heads",
            "8",
            "--ff-dim",
            "512",
        ],
    )
    assert config.arch_explicit is True
    arch = config.arch()
    assert (arch["d_model"], arch["n_layers"], arch["n_heads"], arch["ff_dim"]) == (256, 4, 8, 512)


def test_cli_unset_arch_flags_are_not_explicit(monkeypatch):
    """Silence must stay silence: the self-play loop warm-starts from a checkpoint whose
    arch it does not know, and must keep deferring to it."""
    config = _run_cli_train_rl(monkeypatch, ["train-rl", "--model", "entity_transformer"])
    assert config.arch_explicit is False
    arch = config.arch()
    assert (arch["d_model"], arch["n_layers"], arch["n_heads"], arch["ff_dim"]) == (128, 2, 4, 256)


def test_explicit_arch_conflicting_with_warm_start_raises(tmp_path):
    """The footgun: an AC->AC warm-start overwrites model_meta from the checkpoint, so
    --d-model 256 against a 128-wide checkpoint silently trained a 128-wide net."""
    from rotomai.train.offline_rl import OfflineRLConfig, train_offline_rl

    data = tmp_path / "rl.npz"
    init = tmp_path / "init.pt"
    _write_rl_shard(data)
    _write_transformer_ac_checkpoint(
        init, {"d_model": 64, "n_layers": 1, "n_heads": 4, "ff_dim": 128, "id_embed": False}
    )

    cfg = OfflineRLConfig(
        epochs=1,
        batch_size=4,
        device="cpu",
        model_type=MODEL_ENTITY_TRANSFORMER,
        d_model=256,
        n_layers=4,
        n_heads=8,
        ff_dim=512,
        arch_explicit=True,
        bc_init=str(init),
    )
    with pytest.raises(ValueError, match="d_model: requested 256 != checkpoint 64"):
        train_offline_rl(data, tmp_path / "out.pt", cfg)


def test_warm_start_without_explicit_arch_adopts_checkpoint(tmp_path):
    """Regression for the self-play loop (selfplay.py:179): it passes bc_init=<AC ckpt>
    and no arch fields, relying on the checkpoint's arch winning. That must still work."""
    from rotomai.model.factory import model_metadata
    from rotomai.train.offline_rl import OfflineRLConfig, train_offline_rl

    data = tmp_path / "rl.npz"
    init = tmp_path / "init.pt"
    out = tmp_path / "out.pt"
    _write_rl_shard(data)
    arch = {"d_model": 64, "n_layers": 1, "n_heads": 4, "ff_dim": 128, "id_embed": False}
    _write_transformer_ac_checkpoint(init, arch)

    # Defaults (d_model=128) differ from the checkpoint (64) -- and must yield to it.
    cfg = OfflineRLConfig(epochs=1, batch_size=4, device="cpu", bc_init=str(init))
    train_offline_rl(data, out, cfg)

    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert saved["arch"]["d_model"] == 64
    assert saved["model_type"] == MODEL_ENTITY_TRANSFORMER
    assert model_metadata(build_model(saved))["arch"]["d_model"] == 64

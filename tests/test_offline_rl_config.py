"""OfflineRLConfig must pass identity-embedding settings through to the factory.

Regression guard for the R-ENCODE gap: offrl's ``arch()`` previously omitted the
``id_embed`` keys, so a transformer offrl run silently fell back to random-init
embeddings even when ``prior`` was requested. These tests pin the config -> factory
-> applied-prior path without needing a server or training.
"""

import numpy as np
import torch

from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE
from lategame.features.embed_prior import load_id_priors
from lategame.features.encoder import OBS_DIM
from lategame.model.entity_transformer import EntityTransformer
from lategame.model.factory import MODEL_ENTITY_TRANSFORMER, build_model
from lategame.train.offline_rl import OfflineRLConfig


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
    from lategame.features.encoder import OBS_VERSION

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
    from lategame.features.encoder import OBS_VERSION
    from lategame.model.policy import BCPolicy
    from lategame.train.bc import BC_POLICY

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
    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    data = tmp_path / "rl.npz"
    bc = tmp_path / "bc.pt"
    out = tmp_path / "offrl.pt"
    _write_rl_shard(data)
    _write_bc_policy_checkpoint(bc, hidden_dim=32)

    cfg = OfflineRLConfig(epochs=1, batch_size=4, hidden_dim=32, device="cpu", bc_init=str(bc))
    train_offline_rl(data, out, cfg)
    assert out.exists()

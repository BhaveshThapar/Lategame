"""Build an actor-critic model from checkpoint/config metadata (plan.md, M5).

Single dispatch point so training, inference, and self-play warm-start all
construct the right architecture from a ``model_type`` tag instead of hardcoding a
class. ``meta`` is a checkpoint dict (from ``torch.load``) or a config-shaped dict;
both carry ``input_dim`` / ``n_actions`` / ``n_bins`` plus per-type fields
(``hidden_dim`` for the MLP, an ``arch`` block for the transformer).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE

MODEL_ACTOR_CRITIC = "actor_critic"
MODEL_ENTITY_TRANSFORMER = "entity_transformer"
KNOWN_MODELS = (MODEL_ACTOR_CRITIC, MODEL_ENTITY_TRANSFORMER)


def build_model(meta: Mapping[str, Any]) -> nn.Module:
    """Construct the model described by ``meta`` (defaults to the MLP actor-critic)."""
    model_type = str(meta.get("model_type", MODEL_ACTOR_CRITIC))
    input_dim = int(meta["input_dim"])
    n_actions = int(meta.get("n_actions", GEN9_ACTION_SPACE_SIZE))
    n_bins = int(meta.get("n_bins", 51))
    dropout = float(meta.get("dropout", 0.1))

    if model_type == MODEL_ACTOR_CRITIC:
        from lategame.model.actor_critic import ActorCritic

        return ActorCritic(
            input_dim,
            hidden_dim=int(meta.get("hidden_dim", 256)),
            n_actions=n_actions,
            n_bins=n_bins,
            dropout=dropout,
        )

    if model_type == MODEL_ENTITY_TRANSFORMER:
        from lategame.model.entity_transformer import EntityTransformer

        arch_obj = meta.get("arch") or {}
        arch: Mapping[str, Any] = arch_obj if isinstance(arch_obj, Mapping) else {}
        return EntityTransformer(
            input_dim,
            n_actions=n_actions,
            n_bins=n_bins,
            dropout=dropout,
            d_model=int(arch.get("d_model", 128)),
            n_layers=int(arch.get("n_layers", 2)),
            n_heads=int(arch.get("n_heads", 4)),
            ff_dim=int(arch.get("ff_dim", 256)),
        )

    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {KNOWN_MODELS}.")


def model_metadata(model: nn.Module) -> dict[str, Any]:
    """The ``model_type`` / ``arch`` / dims to stamp into a checkpoint (inverse of
    ``build_model``: ``build_model(model_metadata(m) | extras)`` rebuilds ``m``)."""
    from lategame.model.actor_critic import ActorCritic
    from lategame.model.entity_transformer import EntityTransformer

    if isinstance(model, EntityTransformer):
        return {
            "model_type": MODEL_ENTITY_TRANSFORMER,
            "arch": model.arch_config(),
            "hidden_dim": model.d_model,
            "n_actions": model.n_actions,
            "n_bins": model.n_bins,
        }
    if isinstance(model, ActorCritic):
        return {
            "model_type": MODEL_ACTOR_CRITIC,
            "arch": {},
            "hidden_dim": model.hidden_dim,
            "n_actions": model.n_actions,
            "n_bins": model.n_bins,
        }
    raise TypeError(f"Cannot derive checkpoint metadata for {type(model).__name__}.")

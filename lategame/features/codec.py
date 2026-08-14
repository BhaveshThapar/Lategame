"""Per-format observation/action codec selection (G4: "plug in per-format ... ").

Everything downstream of a battle -- collection, datasets, training, agents -- needs the same
three things: how to encode a POV, how to label an order, and what the legality mask looks like.
Those differ by format, and until doubles existed the singles answers were imported directly.
This is the one place that dispatch lives, so a new format is a new ``FormatCodec`` rather than an
``if`` in every consumer.

**The fingerprint is the safety property.** Datasets and checkpoints record ``(obs_version,
obs_dim)``, and both differ between singles (``v5-`` / 761) and doubles (``d1-`` / 888), so a
cross-format shard is rejected on either field alone rather than being silently mis-encoded --
which would train a policy on garbage that looks like data.

Action shapes deliberately differ and are NOT unified: singles is a scalar action with a
``(26,)`` mask; doubles is a ``(2,)`` per-slot action with a ``(2, 107)`` mask. Flattening doubles
into one 11,449-way index would erase exactly the factoring that makes it tractable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lategame.config import is_doubles_format


@dataclass(frozen=True)
class FormatCodec:
    """How to encode, label and mask for one family of formats."""

    name: str
    obs_version: str
    obs_dim: int
    #: Model output width: 26 for singles, 2 x 107 = 214 for the factored doubles head.
    n_actions: int
    #: True when an action is a per-slot pair rather than a scalar.
    factored: bool
    embed: Callable[[Any], Any]
    order_to_action: Callable[[Any, Any], Any]
    action_mask: Callable[[Any], Any]


def _singles() -> FormatCodec:
    from lategame.features.action_space import (
        GEN9_ACTION_SPACE_SIZE,
        action_mask,
        order_to_action,
    )
    from lategame.features.encoder import OBS_DIM, OBS_VERSION, embed_battle

    return FormatCodec(
        name="singles",
        obs_version=OBS_VERSION,
        obs_dim=OBS_DIM,
        n_actions=GEN9_ACTION_SPACE_SIZE,
        factored=False,
        embed=embed_battle,
        order_to_action=order_to_action,
        action_mask=action_mask,
    )


def _doubles() -> FormatCodec:
    from lategame.features.doubles_action_space import (
        GEN9_DOUBLES_ACTION_SPACE_SIZE,
        action_mask,
        order_to_action,
    )
    from lategame.features.doubles_encoder import (
        OBS_DIM_DOUBLES,
        OBS_VERSION_DOUBLES,
        embed_doubles_battle,
    )

    return FormatCodec(
        name="doubles",
        obs_version=OBS_VERSION_DOUBLES,
        obs_dim=OBS_DIM_DOUBLES,
        n_actions=GEN9_DOUBLES_ACTION_SPACE_SIZE,
        factored=True,
        embed=embed_doubles_battle,
        order_to_action=order_to_action,
        action_mask=action_mask,
    )


_CACHE: dict[bool, FormatCodec] = {}


def codec_for(battle_format: str) -> FormatCodec:
    """The codec for ``battle_format``. Cached: the imports pull torch-free but not free modules."""
    doubles = is_doubles_format(battle_format)
    if doubles not in _CACHE:
        _CACHE[doubles] = _doubles() if doubles else _singles()
    return _CACHE[doubles]


def codec_for_battle(battle: Any) -> FormatCodec:
    """The codec for a live battle, read off its own format string."""
    return codec_for(getattr(battle, "format", None) or "")


__all__ = ["FormatCodec", "codec_for", "codec_for_battle"]

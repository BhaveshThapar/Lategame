"""The live half of trajectory context: turn a per-turn observation into what the model expects.

Every learned agent here does the same two lines -- encode the battle, hand the vector to the model
-- and a history-conditioned model needs the last `K+1` turns instead. Putting that branch in one
place is not tidiness: this project's two most expensive bugs were both a training observation that
did not match the live one (`docs/RESULTS.md`, the opponent-roster and own-team mismatches that took
a 0.71-accuracy BC policy to random-quality play). Three agents each growing their own ring buffer
is the same failure with a third chance to diverge.

The window itself comes from `data.history`, which the DATASET also calls, so
`tests/test_history.py` can assert the two sides produce identical tensors for one turn sequence.

A model with `history == 0` gets exactly what it always got: a flat `[obs_dim]` vector, no buffer
allocated, no behaviour changed.
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import AbstractBattle
from torch import nn

from rotomai.data.history import PerBattleHistory


def _history_tracker(model: nn.Module, obs_dim: int) -> PerBattleHistory | None:
    """A tracker when the model consumes a window, ``None`` when it consumes one turn.

    Read off the MODEL rather than passed in by the caller: the checkpoint is the authority on how
    many frames its `time_embed` was trained for, and a mismatch between the two is unrecoverable
    at inference time.
    """
    history = int(getattr(model, "history", 0) or 0)
    return PerBattleHistory(history, obs_dim) if history else None


def observe(
    tracker: PerBattleHistory | None, battle: AbstractBattle, observation: np.ndarray
) -> np.ndarray:
    """``[obs_dim]`` unchanged, or ``[history + 1, obs_dim]`` ending at this turn."""
    if tracker is None:
        return observation
    return tracker.observe(str(battle.battle_tag), observation)

"""Trajectory context for a single-turn encoder, defined ONCE for training and for live play.

WHY THIS EXISTS AT ALL. `features/encoder.py` emits a 761-d observation of the CURRENT turn and
nothing else -- no time axis anywhere in `OBS_LAYOUT`. The pre-registered ladder result named that
as its diagnosis (`docs/RESULTS.md`: the agent "decides from a 761-d single-turn observation with no
trajectory context"), and the human-level results this project benchmarks against came from
sequence models over historical gameplay. This module is the seam that adds the axis.

WHY NOT A WIDER ENCODER. Changing `embed_battle`'s output would bump `OBS_VERSION`, and that
fingerprint is asserted on every shard and every checkpoint by design (`data/dataset.py`,
`agents/bc_agent.py`). A bump invalidates `data/gen9ou_v7_rl.npz` -- the only surviving human-replay
shard, whose 2,760 source replays are gone and cannot be re-fetched without confounding every
comparison (`scripts/cluster/README.md`). So history is assembled from unchanged 761-d frames
instead: the encoder, the shards and every existing checkpoint stay exactly as they are.

WHY ONE MODULE FOR BOTH SIDES. This project has been burned twice by a training observation that
did not match the live one -- the opponent-roster and own-team mismatches that took the first
human-replay agent to random-quality play despite a healthy 0.71 BC accuracy. A stacker in the
dataset and a separate ring buffer in the agent is that same bug waiting to happen, so both call
`stack_history` here and `tests/test_history.py` asserts they agree tensor-for-tensor.

THE CONVENTION, stated once. A window is `history + 1` frames ending at the current turn, oldest
first: `[t-K, ..., t-1, t]`. Turns before the episode began are ZERO frames, never the tail of the
previous battle. A zero frame is self-masking downstream -- "present" is feature 0 of every Pokemon
and move block, so an all-zero frame reads as an absent entity everywhere.
"""

from __future__ import annotations

import numpy as np
import torch


def episode_start_index(done: np.ndarray) -> np.ndarray:
    """For each row, the index of the first row of ITS episode.

    ``done`` marks the LAST row of each episode, so an episode starts one past the previous flag.
    Without this a window at the head of a battle would silently splice on the tail of the previous
    one -- the same battle-boundary error `discounted_returns` avoids by resetting on ``done``.
    """
    done = np.asarray(done, dtype=bool)
    n = len(done)
    starts = np.zeros(n, dtype=np.int64)
    boundaries = np.flatnonzero(done) + 1
    boundaries = boundaries[boundaries < n]
    starts[boundaries] = boundaries
    np.maximum.accumulate(starts, out=starts)
    return starts


def window_indices(index: int, start: int, history: int) -> list[int]:
    """Row indices for the window ending at ``index``; ``-1`` marks a zero-padded frame."""
    if history < 0:
        raise ValueError("history must be >= 0")
    return [i if i >= start else -1 for i in range(index - history, index + 1)]


def stack_history(
    obs: torch.Tensor, index: int, start: int, history: int
) -> torch.Tensor:
    """``[history + 1, obs_dim]`` ending at ``index``, zero-padded back to ``start``."""
    if history == 0:
        return obs[index].unsqueeze(0)
    frames = torch.zeros(history + 1, obs.shape[1], dtype=obs.dtype)
    for slot, row in enumerate(window_indices(index, start, history)):
        if row >= 0:
            frames[slot] = obs[row]
    return frames


class HistoryBuffer:
    """The live twin of ``stack_history``: a per-battle ring of the last ``history`` observations.

    One instance per battle. ``reset`` is not optional -- a buffer carried across battles would feed
    the previous opponent's board into the first turn of the next game, which is exactly the
    train/eval mismatch this module exists to prevent.
    """

    def __init__(self, history: int, obs_dim: int) -> None:
        if history < 0:
            raise ValueError("history must be >= 0")
        self.history = history
        self.obs_dim = obs_dim
        self._frames: list[np.ndarray] = []

    def reset(self) -> None:
        self._frames.clear()

    def push(self, observation: np.ndarray) -> np.ndarray:
        """Record this turn's observation and return the ``[history + 1, obs_dim]`` window."""
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.obs_dim:
            raise ValueError(f"expected a {self.obs_dim}-d observation, got {vector.shape[0]}")
        self._frames.append(vector)
        if len(self._frames) > self.history + 1:
            del self._frames[0]

        window = np.zeros((self.history + 1, self.obs_dim), dtype=np.float32)
        # Right-aligned: the CURRENT turn is always the last slot, so a model never has to learn
        # where "now" is as a function of how far into the battle it happens to be.
        window[self.history + 1 - len(self._frames) :] = np.stack(self._frames)
        return window


class PerBattleHistory:
    """One ``HistoryBuffer`` per battle tag, for an agent playing several games at once.

    Keyed exactly as ``agents.loop_guard.LoopGuard`` keys its streak counters, and for the same
    reason: poke-env runs concurrent battles through one Player instance, so any per-battle state
    that is not keyed by tag silently interleaves two games.
    """

    def __init__(self, history: int, obs_dim: int) -> None:
        self.history = history
        self.obs_dim = obs_dim
        self._buffers: dict[str, HistoryBuffer] = {}

    def observe(self, battle_tag: str, observation: np.ndarray) -> np.ndarray:
        buffer = self._buffers.get(battle_tag)
        if buffer is None:
            buffer = HistoryBuffer(self.history, self.obs_dim)
            self._buffers[battle_tag] = buffer
        return buffer.push(observation)

    def forget(self, battle_tag: str) -> None:
        self._buffers.pop(battle_tag, None)

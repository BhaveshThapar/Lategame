"""Human-replay ingestion: Showdown spectator logs -> training shards (plan.md, M6).

Every learned method so far plateaus because its demonstrators are no stronger than
the M1 heuristic. The fix the PRD (plan.md sec. 10) always intended: imitate real
human ladder play. This module reconstructs each player's partially-observed POV from
a *public* replay log -- offline, no server -- and emits samples in the exact same
``(obs, action, mask, reward, done)`` shape as self-play collection, so the existing
``train-rl``/``train`` consume them unchanged.

How the public-log constraint shapes this (see ``features/action_space`` for the codec
side): public replays carry no ``|request|`` JSON, so ``available_moves``/``_switches``
are empty and the live ``order_to_action``/``action_mask`` can't run. We instead:

* feed each ``|...|`` line into a poke-env ``Battle`` via ``parse_message`` (server-free);
* label the **move** a player used the instant it's revealed, positionally, with
  ``label_action`` (fake=True codec) -- self-consistent with the encoder's move order;
* label a **switch** only when the target was already revealed earlier in the game (a
  pivot back), so the pre-switch obs and the switch index agree; first-reveal switches
  and forced/post-faint switches are *dropped*, not guessed;
* compute the shaped reward from public HP/faint/status (``data.reward.state_value``),
  diffing decision points exactly like ``collect._battle_rewards``.

The accepted v1 fidelity gap: a player's own bench is revealed only as mons switch in
(live play sees all six from the request). v2 can fill this from randbats set data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from poke_env import to_id_str
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import SingleBattleOrder

from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import Dataset, TrajectoryDataset, save, save_rl
from lategame.data.reward import RewardWeights, state_value
from lategame.features.action_space import (
    GEN9_ACTION_SPACE_SIZE,
    label_action,
    synthesize_action_mask,
)
from lategame.features.encoder import embed_battle

# poke-env logs a warning for every message it can't fully model on a request-less
# battle; that's expected here, so keep this quiet during bulk ingestion.
_LOGGER = logging.getLogger("lategame.ingest")
_LOGGER.setLevel(logging.CRITICAL)

_Record = tuple[np.ndarray, int, np.ndarray]  # (obs, action, mask) -- matches collect._Record
_Sample = tuple[_Record, float]  # record + its shaped state-value


@dataclass
class IngestStats:
    """Per-run counters; the drop rate is the key health signal for the pilot gate."""

    replays: int = 0  # replays seen
    parsed: int = 0  # replays reconstructed without a fatal error
    skipped_replays: int = 0  # replays dropped (no log / malformed / parse error)
    episodes: int = 0  # per-player trajectories kept
    turns: int = 0  # labelled decision turns kept
    dropped_turns: int = 0  # decisions seen but undecodable

    @property
    def drop_rate(self) -> float:
        seen = self.turns + self.dropped_turns
        return self.dropped_turns / seen if seen else 0.0


def _own(parts: list[str], battle: AbstractBattle) -> bool:
    """True when the message's ``pXa: Name`` subject belongs to this POV's player."""
    return len(parts) > 2 and battle.player_role is not None and parts[2][:2] == battle.player_role


def _team_key(ident: str) -> str:
    """Normalise a ``pXa: Name`` identifier to its ``pX: Name`` team-dict key."""
    return ident if len(ident) < 4 or ident[3] == " " else ident[:2] + ident[3:]


def _safe_parse(battle: AbstractBattle, parts: list[str]) -> None:
    try:
        battle.parse_message(parts)
    except Exception:  # noqa: BLE001 -- one odd line must not abort a whole replay
        pass


def _move_sample(
    battle: AbstractBattle, parts: list[str], tera: bool, weights: RewardWeights
) -> _Sample | None:
    """Label the move just parsed (it is now in ``active.moves``) from this state."""
    active = battle.active_pokemon
    if active is None or len(parts) < 4:
        return None
    move = active.moves.get(to_id_str(parts[3]))
    if move is None:
        return None
    try:
        action = label_action(SingleBattleOrder(move, terastallize=tera), battle)
    except Exception:  # noqa: BLE001
        return None
    if not 6 <= action < GEN9_ACTION_SPACE_SIZE:  # must decode to a move slot
        return None
    obs = embed_battle(battle)
    return (obs, action, synthesize_action_mask(battle, action)), state_value(battle, weights)


def _switch_sample(battle: AbstractBattle, ident: str, weights: RewardWeights) -> _Sample | None:
    """Label a voluntary switch *before* applying it -- only if the target is known.

    Recorded only when the incoming mon is already in ``team`` (a pivot back to a
    previously-revealed Pokemon), so the pre-switch obs and the ``team.values()``
    index are mutually consistent. First-reveal switches return ``None`` (dropped).
    """
    if battle.active_pokemon is None:
        return None
    target = battle.team.get(_team_key(ident))
    if target is None or target.active or target.fainted:
        return None
    try:
        action = label_action(SingleBattleOrder(target), battle)
    except Exception:  # noqa: BLE001
        return None
    if not 0 <= action < 6:  # must decode to a switch slot
        return None
    obs = embed_battle(battle)
    return (obs, action, synthesize_action_mask(battle, action)), state_value(battle, weights)


def _reconstruct_pov(
    lines: list[str], username: str, battle_tag: str, gen: int, weights: RewardWeights
) -> tuple[AbstractBattle, list[_Record], list[float], int]:
    """Replay ``lines`` into one player's POV; return its battle + labelled decisions.

    A decision is the first ``|move|`` or qualifying voluntary ``|switch|`` the player
    makes each turn. Move obs is snapshotted *after* the move line (so the move is
    revealed and indexable); switch obs *before* (so the old active is still shown).
    """
    battle = Battle(battle_tag=battle_tag, username=username, gen=gen, logger=_LOGGER)
    records: list[_Record] = []
    values: list[float] = []
    dropped = 0
    started = acted = fainted = tera = False

    for raw in lines:
        if not raw.startswith("|"):
            continue
        parts = raw.split("|")
        tag = parts[1] if len(parts) > 1 else ""

        if tag == "turn":
            started, acted, fainted, tera = True, False, False, False
            _safe_parse(battle, parts)
            continue

        # |win|/|tie| aren't handled by parse_message (the live Player loop calls these
        # directly), so drive them here to set finished/won for the terminal reward.
        if tag == "win":
            try:
                battle.won_by(parts[2] if len(parts) > 2 else "")
            except Exception:  # noqa: BLE001
                pass
            continue
        if tag == "tie":
            try:
                battle.tied()
            except Exception:  # noqa: BLE001
                pass
            continue

        if tag == "switch" and started and not acted and not fainted and _own(parts, battle):
            sample = _switch_sample(battle, parts[2], weights)
            if sample is not None:
                records.append(sample[0])
                values.append(sample[1])
            else:
                dropped += 1
            acted = True
            _safe_parse(battle, parts)
            continue

        _safe_parse(battle, parts)
        if tag == "faint" and _own(parts, battle):
            fainted = True
        elif tag == "-terastallize" and _own(parts, battle):
            tera = True
        elif tag == "move" and started and not acted and _own(parts, battle):
            sample = _move_sample(battle, parts, tera, weights)
            if sample is not None:
                records.append(sample[0])
                values.append(sample[1])
            else:
                dropped += 1
            acted = True

    return battle, records, values, dropped


def _episode_rewards(
    battle: AbstractBattle, records: list[_Record], values: list[float], weights: RewardWeights
) -> list[float]:
    """Per-turn shaped rewards for a finished battle (mirrors collect._battle_rewards).

    ``reward[t] = sv[t+1] - sv[t]``; the last entry diffs against the terminal value so
    it carries the +/- victory jump. Returns ``[]`` for unfinished/empty trajectories.
    """
    if not battle.finished or not records:
        return []
    next_values = values[1:] + [state_value(battle, weights)]
    return [nv - v for v, nv in zip(values, next_values, strict=True)]


@dataclass
class _ShardBuilder:
    """Accumulates labelled POV episodes into the ``collect`` on-disk schema.

    Shared by public-log ingestion (this module) and re-sim ingestion
    (``data.resim``): both yield a per-POV ``(battle, records, values)`` and differ
    only in *how* those are reconstructed, not in how they are aggregated/saved.
    """

    obs: list[np.ndarray] = field(default_factory=list)
    act: list[int] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    rew: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)
    bc_obs: list[np.ndarray] = field(default_factory=list)
    bc_act: list[int] = field(default_factory=list)
    bc_mask: list[np.ndarray] = field(default_factory=list)

    def add_pov(
        self,
        battle: AbstractBattle,
        records: list[_Record],
        values: list[float],
        weights: RewardWeights,
    ) -> int:
        """Append one finished POV's turns; return the count kept (0 if unusable).

        Computes per-turn shaped rewards, marks the last turn ``done``, and mirrors
        winning POVs into the BC shard -- the M2 reward filter.
        """
        rewards = _episode_rewards(battle, records, values, weights)
        if not rewards:
            return 0
        won = bool(battle.won)
        last = len(records) - 1
        for i, ((obs, action, mask), r) in enumerate(zip(records, rewards, strict=True)):
            self.obs.append(obs)
            self.act.append(action)
            self.mask.append(mask)
            self.rew.append(r)
            self.done.append(i == last)
            if won:
                self.bc_obs.append(obs)
                self.bc_act.append(action)
                self.bc_mask.append(mask)
        return len(records)

    def finalize(
        self, battle_format: str, gamma: float, weights: RewardWeights
    ) -> tuple[TrajectoryDataset, Dataset | None]:
        """Stack the columns into the RL shard (+ winners-only BC shard)."""
        if not self.obs:
            raise RuntimeError(
                "No labelled turns reconstructed from any replay -- check the input format."
            )
        rl = TrajectoryDataset(
            obs=np.stack(self.obs).astype(np.float32),
            action=np.asarray(self.act, dtype=np.int64),
            mask=np.stack(self.mask).astype(bool),
            reward=np.asarray(self.rew, dtype=np.float32),
            done=np.asarray(self.done, dtype=bool),
            battle_format=battle_format,
            gamma=gamma,
            weights=weights,
        )
        bc = (
            Dataset(
                obs=np.stack(self.bc_obs).astype(np.float32),
                action=np.asarray(self.bc_act, dtype=np.int64),
                mask=np.stack(self.bc_mask).astype(bool),
                battle_format=battle_format,
            )
            if self.bc_obs
            else None
        )
        return rl, bc


def _gen_from_log(log: str) -> int:
    for line in log.split("\n"):
        if line.startswith("|gen|"):
            try:
                return int(line.split("|")[2])
            except (IndexError, ValueError):
                break
    return 9


def ingest_replays(
    replays: Iterable[Mapping[str, object]],
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
) -> tuple[TrajectoryDataset, Dataset | None, IngestStats]:
    """Reconstruct ``replays`` into an offline-RL shard (+ a winners-only BC shard).

    Returns ``(rl_dataset, bc_dataset_or_None, stats)``. The RL shard keeps every
    labelled turn of every finished POV; the BC shard keeps only the winning POVs'
    turns (the M2 reward filter). Both use the exact ``collect`` on-disk schema.
    """
    weights = weights or RewardWeights()
    builder = _ShardBuilder()
    stats = IngestStats()

    for replay in replays:
        stats.replays += 1
        log = replay.get("log")
        players = replay.get("players")
        if (
            not isinstance(log, str)
            or not log
            or not isinstance(players, list)
            or len(players) != 2
        ):
            stats.skipped_replays += 1
            continue
        tag = str(replay.get("id") or f"replay{stats.replays}")
        gen = _gen_from_log(log)
        lines = log.split("\n")

        try:
            for username in players:
                battle, records, values, dropped = _reconstruct_pov(
                    lines, str(username), f"{tag}-{username}", gen, weights
                )
                stats.dropped_turns += dropped
                kept = builder.add_pov(battle, records, values, weights)
                if kept:
                    stats.episodes += 1
                    stats.turns += kept
        except Exception:  # noqa: BLE001 -- a single bad replay shouldn't kill the run
            stats.skipped_replays += 1
            continue
        stats.parsed += 1

    rl, bc = builder.finalize(battle_format, gamma, weights)
    return rl, bc, stats


def ingest_replay_files(
    paths: Iterable[str | Path],
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
) -> tuple[TrajectoryDataset, Dataset | None, IngestStats]:
    """Load cached replay ``.json`` files and ingest them (see ``data.replays``)."""
    import json

    def _load() -> Iterable[Mapping[str, object]]:
        for p in paths:
            try:
                with open(p, encoding="utf-8") as fh:
                    yield json.load(fh)
            except (OSError, ValueError):
                continue

    return ingest_replays(_load(), weights, gamma, battle_format)


def ingest_and_save(
    paths: Iterable[str | Path],
    rl_out: str | Path,
    bc_out: str | Path | None = None,
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
) -> IngestStats:
    """Ingest cached replay files and write the RL shard (+ optional BC shard)."""
    rl, bc, stats = ingest_replay_files(paths, weights, gamma, battle_format)
    save_rl(rl, rl_out)
    if bc is not None and bc_out is not None:
        save(bc, bc_out)
    print(
        f"ingested {stats.parsed}/{stats.replays} replays "
        f"({stats.skipped_replays} skipped) -> {stats.turns} turns from "
        f"{stats.episodes} POV-episodes; drop rate {stats.drop_rate:.1%}"
    )
    return stats

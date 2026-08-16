"""Self-play data collection with reward filtering (plan.md, M2 / R-TRAIN).

We have no human replays for ``gen9randombattle`` (Metamon covers OU, not random
battles), so M2 bootstraps from *self-play*: a diverse pool of bots plays on the
local server, every turn is recorded as ``(obs, action, legal_mask)``, and we keep
**only the winning side's** turns -- reward-filtered behaviour cloning at its
simplest (reward == game outcome). Imitating the winners of a diverse pool, biased
toward the stronger demonstrators, is the bootstrap target.

This module is numpy-only (no torch): collection runs anywhere poke-env runs. The
on-disk format is a single compressed ``.npz`` stamped with the encoder version so
``data.dataset`` can refuse a stale encoding.

The data source is deliberately swappable: anything that yields ``(obs, action,
mask)`` arrays in this format trains the same policy. Scraped human replays (the
genuine PRD path) would drop in here without touching the model or training code.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, ForfeitBattleOrder, Player, cross_evaluate

from lategame.agents.turn_cap import TurnCap
from lategame.config import DEFAULT_FORMAT, LOCAL_SERVER, local_account
from lategame.data.reward import RewardWeights, state_value
from lategame.eval.arena import (
    _CHECKPOINT_AGENTS,
    _DOUBLES_LOOP_GUARD_AGENTS,
    _LOOP_GUARD_AGENTS,
    AGENTS,
    _unique_username,
)
from lategame.features.codec import codec_for, codec_for_battle

_Record = tuple[np.ndarray, int, np.ndarray]
_Episode = tuple[list[_Record], list[float]]  # one battle's records + per-turn rewards


class _RecordingMixin:
    """Records ``(obs, action, mask)`` each turn, keyed by battle tag.

    Mixed in *before* a concrete ``Player`` so its ``choose_move`` wraps the
    demonstrator's decision without changing it. Turns whose order can't be
    labelled (rare, e.g. an order not in ``valid_orders``) are dropped, not
    guessed, to keep labels clean.

    A scalar shaped ``state_value`` is captured alongside each record (parallel
    list keyed by tag) so offline-RL collection can reconstruct per-turn rewards
    by diffing successive values. BC collection ignores it.

    ``max_battle_turns`` caps decisions per battle and forfeits past the ceiling. It lives HERE,
    on the wrapper, rather than only on the learned agents, because the demonstrator pool is
    mostly poke-env baselines (``random``, ``maxbasepower``, ``simpleheuristics``) that have no
    cap of their own -- and it was their loops, not a learned policy's, that put 94.2% of
    ``data/vgc_rl.npz`` into 100 of 899 episodes. ``None`` is exact identity.
    """

    _reward_weights: RewardWeights = RewardWeights()

    def __init__(
        self, *args: object, max_battle_turns: int | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.records: dict[str, list[_Record]] = {}
        self.values: dict[str, list[float]] = {}
        # Keep a reference to each battle object: cross_evaluate calls
        # reset_battles() (clearing player.battles) after every pair, but the
        # battle objects -- with their final .won already set -- live on here.
        self.seen_battles: dict[str, AbstractBattle] = {}
        self.dropped = 0
        self.turn_cap = TurnCap(max_battle_turns)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Before the demonstrator is consulted: past the ceiling there is nothing to imitate,
        # only a re-requested state to stop paying for.
        if self.turn_cap.hit(battle):
            return ForfeitBattleOrder()
        order = super().choose_move(battle)  # type: ignore[misc]
        try:
            # Per-format codec, read off the battle itself: a doubles turn records a (2,)
            # per-slot action and a (2, 107) mask, a singles turn a scalar and (26,). The
            # shapes deliberately differ -- flattening doubles into one 11,449-way index would
            # erase the factoring that makes it tractable.
            codec = codec_for_battle(battle)
            obs = codec.embed(battle)
            action = codec.order_to_action(order, battle)
            mask = codec.action_mask(battle)
        except Exception:
            self.dropped += 1
            return order
        self.records.setdefault(battle.battle_tag, []).append((obs, action, mask))
        self.values.setdefault(battle.battle_tag, []).append(
            state_value(battle, self._reward_weights)
        )
        self.seen_battles[battle.battle_tag] = battle
        return order


_recording_cache: dict[type[Player], type[Player]] = {}


def _recording_class(cls: type[Player]) -> type[Player]:
    if cls not in _recording_cache:
        _recording_cache[cls] = type(f"Recording{cls.__name__}", (_RecordingMixin, cls), {})
    return _recording_cache[cls]


@dataclass(frozen=True)
class PlayerSpec:
    """A recordable player: an agent ``name`` plus optional checkpoint/sampling.

    ``checkpoint_path``/``sample`` are only meaningful for the checkpoint-backed
    agents (``bc``, ``offrl``) and are ignored for the fixed baselines.
    """

    name: str
    checkpoint_path: str | None = None
    sample: bool = False


def _build_recording_player(
    spec: PlayerSpec,
    battle_format: str,
    weights: RewardWeights | None = None,
    max_concurrent: int = 1,
    team: object | None = None,
    max_battle_turns: int | None = None,
    loop_penalty: float = 0.0,
) -> Player:
    if spec.name not in AGENTS:
        raise ValueError(f"Unknown agent '{spec.name}'. Choose from: {', '.join(AGENTS)}")
    cls = _recording_class(AGENTS[spec.name])
    extra: dict[str, object] = {}
    if spec.name in _CHECKPOINT_AGENTS:
        extra["sample"] = spec.sample
        if spec.checkpoint_path is not None:
            extra["checkpoint_path"] = spec.checkpoint_path
    # Same filtering as `eval.arena.build_player`: only the learned policies carry a guard, and
    # 0.0 is exact identity. Collection never forwarded this at all before B6f, so a demonstrator
    # pool could not be run WITH the guard even where the agent supported one.
    if spec.name in _LOOP_GUARD_AGENTS or spec.name in _DOUBLES_LOOP_GUARD_AGENTS:
        extra["loop_penalty"] = loop_penalty
    # B6f: the ceiling lives on the RECORDING WRAPPER, so it covers every demonstrator including
    # the poke-env baselines that have no cap of their own -- and it was their loops that put
    # 94.2% of the first VGC shard into 100 of 899 episodes. None is exact identity.
    if max_battle_turns is not None:
        extra["max_battle_turns"] = max_battle_turns
    # poke-env defaults to one battle at a time; the self-play loop passes a higher
    # value so cross_evaluate keeps many battles in flight (the local server is the
    # bottleneck). Default 1 keeps M2/M3 collection behaviour unchanged.
    if max_concurrent > 1:
        extra["max_concurrent_battles"] = max_concurrent
    # A teambuilt format (gen9ou, VGC) needs a team; Random Battles must NOT get one, since the
    # server supplies them there. Same convention as eval.arena.build_player.
    if team is not None:
        extra["team"] = team
    player = cls(
        account_configuration=local_account(_unique_username(f"rec{spec.name}")),
        battle_format=battle_format,
        server_configuration=LOCAL_SERVER,
        **extra,  # type: ignore[arg-type]
    )
    if weights is not None:
        player._reward_weights = weights  # type: ignore[attr-defined]
    return player


def _is_learnable(action: np.ndarray | int, mask: np.ndarray) -> bool:
    """Whether a recorded turn carries a decision worth training on.

    Two rejections, both measured on real VGC collection rather than assumed:

    * **The label must be legal under its own mask.** 2 rows in 51,650 were not, and each
      contributes ~1e9 to cross-entropy (the target sits at ``NEG_INF``), which alone drove BC's
      reported train loss to 43,024 while validation sat at 0.04.
    * **The turn must offer a choice.** A turn where every slot has exactly one legal action is
      not imitation, it is copying the only option; on a factored head it also hands the model a
      free correct prediction for that slot.

    CORRECTED (B6f). This docstring used to read "98.4% of recorded turns had exactly ONE legal
    action per slot ... against ~19 legal actions per slot on the 1.6% of turns that were real
    decisions", and cited that as a property of doubles. It was not: it measured the loop in
    ``agents.heuristic_agent._choose_doubles_move``, which emitted a WHOLE-order
    ``DefaultBattleOrder`` for an idle slot and so re-requested the same state thousands of times
    (see ``agents.turn_cap``). On a shard collected after that fix, VGC doubles is a normal,
    decision-DENSE format:

        decision density        0.923   (loop-contaminated shard: 0.571; gen9ou: 0.9998)
        legal actions per slot  14.0    (loop-contaminated shard: 2.62;  gen9ou: 7.67)

    So this filter rejects ~8% of doubles turns, not 98.4%, and doubles offers roughly TWICE OU's
    branching per slot rather than a fraction of it.
    """
    if mask.ndim == 1:  # singles: scalar action, (A,) mask
        a = int(action)
        return bool(0 <= a < mask.shape[0] and mask[a] and mask.sum() > 1)
    for i in range(mask.shape[0]):
        a = int(action[i])  # type: ignore[index]
        if not (0 <= a < mask.shape[1] and bool(mask[i][a])):
            return False
    return bool(mask.sum(axis=1).max() > 1)  # at least one slot faced a real choice


def _winning_records(player: Player) -> list[_Record]:
    """The recorded turns from battles this player won (reward filter)."""
    kept: list[_Record] = []
    for tag, recs in player.records.items():  # type: ignore[attr-defined]
        battle = player.seen_battles.get(tag)  # type: ignore[attr-defined]
        if battle is not None and battle.won:
            kept.extend(recs)
    return kept


@dataclass
class Dataset:
    obs: np.ndarray  # [K, OBS_DIM] float32
    action: np.ndarray  # [K] int64
    mask: np.ndarray  # [K, ACTION_SIZE] bool
    battle_format: str

    def __len__(self) -> int:
        return len(self.action)


def _team_factory(team_pool: str | None) -> Callable[[int], object | None]:
    """A fresh ``TeamPool`` per player, seeded distinctly so the two sides draw independently."""
    if not team_pool:
        return lambda _seed: None
    from lategame.teambuilding.pool import TeamPool

    teams = TeamPool.from_packed_file(team_pool).teams
    return lambda seed: TeamPool(teams, seed=seed)


async def collect(
    pool: list[str],
    n_per_pair: int,
    battle_format: str = DEFAULT_FORMAT,
    team_pool: str | None = None,
    max_battle_turns: int | None = None,
) -> Dataset:
    """Play every distinct pair in ``pool`` and return winners' ``(obs, action, mask)``."""
    if len(pool) < 2:
        raise ValueError("Need at least two agents in the pool to generate self-play games.")

    obs_chunks: list[np.ndarray] = []
    action_chunks: list[int] = []
    mask_chunks: list[np.ndarray] = []
    dropped = 0

    make_team = _team_factory(team_pool)
    for i, (a, b) in enumerate(itertools.combinations(pool, 2)):
        pa = _build_recording_player(
            PlayerSpec(a), battle_format, team=make_team(2 * i), max_battle_turns=max_battle_turns
        )
        pb = _build_recording_player(
            PlayerSpec(b),
            battle_format,
            team=make_team(2 * i + 1),
            max_battle_turns=max_battle_turns,
        )
        await cross_evaluate([pa, pb], n_challenges=n_per_pair)
        for player in (pa, pb):
            dropped += player.dropped  # type: ignore[attr-defined]
            for obs, action, mask in _winning_records(player):
                if not _is_learnable(action, mask):
                    dropped += 1
                    continue
                obs_chunks.append(obs)
                action_chunks.append(action)
                mask_chunks.append(mask)

    if not obs_chunks:
        raise RuntimeError("No winning trajectories collected -- is the local server running?")

    print(f"collected {len(obs_chunks)} winning turns ({dropped} unlabelable turns dropped)")
    return Dataset(
        obs=np.stack(obs_chunks).astype(np.float32),
        action=np.asarray(action_chunks, dtype=np.int64),  # [K] or [K, 2]
        mask=np.stack(mask_chunks).astype(bool),
        battle_format=battle_format,
    )


def save(dataset: Dataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=dataset.obs,
        action=dataset.action,
        mask=dataset.mask,
        obs_version=np.array(codec_for(dataset.battle_format).obs_version),
        obs_dim=np.array(codec_for(dataset.battle_format).obs_dim),
        battle_format=np.array(dataset.battle_format),
    )
    print(f"wrote {len(dataset)} samples to {path}")


# --------------------------------------------------------------------------- #
# M3: full-trajectory collection with shaped per-turn rewards (offline RL).
# Unlike BC's reward filter, this keeps *every* turn of *every* finished battle
# (wins and losses): the value function needs to see losing states, and the AWR
# actor decides which actions to imitate from the learned advantage, not the
# game outcome.
# --------------------------------------------------------------------------- #


@dataclass
class TrajectoryDataset:
    obs: np.ndarray  # [N, OBS_DIM] float32
    action: np.ndarray  # [N] int64
    mask: np.ndarray  # [N, ACTION_SIZE] bool
    reward: np.ndarray  # [N] float32 -- shaped per-turn reward (terminal incl. victory)
    done: np.ndarray  # [N] bool -- True on the last turn of each episode
    battle_format: str
    gamma: float
    weights: RewardWeights

    def __len__(self) -> int:
        return len(self.action)


def _battle_rewards(player: Player, weights: RewardWeights) -> list[_Episode]:
    """For each finished battle, return its turn records and per-turn rewards.

    ``reward[t] = sv[t+1] - sv[t]`` for interior turns and
    ``reward[last] = state_value(finished_battle) - sv[last]`` so the final entry
    carries the +/- victory jump. Unfinished battles are skipped.
    """
    out: list[tuple[list[_Record], list[float]]] = []
    for tag, recs in player.records.items():  # type: ignore[attr-defined]
        battle = player.seen_battles.get(tag)  # type: ignore[attr-defined]
        values = player.values.get(tag, [])  # type: ignore[attr-defined]
        if battle is None or not battle.finished or not recs:
            continue
        terminal = state_value(battle, weights)
        next_values = values[1:] + [terminal]
        rewards = [nv - v for v, nv in zip(values, next_values, strict=True)]
        out.append((recs, rewards))
    return out


def _append_episodes(
    player: Player,
    weights: RewardWeights,
    obs_chunks: list[np.ndarray],
    action_chunks: list[int],
    mask_chunks: list[np.ndarray],
    reward_chunks: list[float],
    done_chunks: list[bool],
) -> int:
    """Append every finished-battle turn from ``player`` to the chunk lists.

    Returns the player's count of unlabelable (dropped) turns.
    """
    for recs, rewards in _battle_rewards(player, weights):
        for i, ((obs, action, mask), r) in enumerate(zip(recs, rewards, strict=True)):
            obs_chunks.append(obs)
            action_chunks.append(action)
            mask_chunks.append(mask)
            reward_chunks.append(r)
            done_chunks.append(i == len(recs) - 1)
    return player.dropped  # type: ignore[attr-defined]


def _trajectory_dataset(
    obs_chunks: list[np.ndarray],
    action_chunks: list[int],
    mask_chunks: list[np.ndarray],
    reward_chunks: list[float],
    done_chunks: list[bool],
    battle_format: str,
    gamma: float,
    weights: RewardWeights,
) -> TrajectoryDataset:
    if not obs_chunks:
        raise RuntimeError("No finished trajectories collected -- is the local server running?")
    return TrajectoryDataset(
        obs=np.stack(obs_chunks).astype(np.float32),
        action=np.asarray(action_chunks, dtype=np.int64),  # [K] or [K, 2]
        mask=np.stack(mask_chunks).astype(bool),
        reward=np.asarray(reward_chunks, dtype=np.float32),
        done=np.asarray(done_chunks, dtype=bool),
        battle_format=battle_format,
        gamma=gamma,
        weights=weights,
    )


async def collect_trajectories(
    pool: list[str],
    n_per_pair: int,
    battle_format: str = DEFAULT_FORMAT,
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    team_pool: str | None = None,
    max_battle_turns: int | None = None,
    loop_penalty: float = 0.0,
) -> TrajectoryDataset:
    """Play every distinct pair in ``pool`` and return all turns + shaped rewards."""
    if len(pool) < 2:
        raise ValueError("Need at least two agents in the pool to generate self-play games.")
    weights = weights or RewardWeights()

    obs_chunks: list[np.ndarray] = []
    action_chunks: list[int] = []
    mask_chunks: list[np.ndarray] = []
    reward_chunks: list[float] = []
    done_chunks: list[bool] = []
    dropped = 0

    make_team = _team_factory(team_pool)
    for i, (a, b) in enumerate(itertools.combinations(pool, 2)):
        pa = _build_recording_player(
            PlayerSpec(a),
            battle_format,
            weights,
            team=make_team(2 * i),
            max_battle_turns=max_battle_turns,
            loop_penalty=loop_penalty,
        )
        pb = _build_recording_player(
            PlayerSpec(b),
            battle_format,
            weights,
            team=make_team(2 * i + 1),
            max_battle_turns=max_battle_turns,
            loop_penalty=loop_penalty,
        )
        await cross_evaluate([pa, pb], n_challenges=n_per_pair)
        for player in (pa, pb):
            dropped += _append_episodes(
                player, weights, obs_chunks, action_chunks, mask_chunks, reward_chunks, done_chunks
            )

    dataset = _trajectory_dataset(
        obs_chunks, action_chunks, mask_chunks, reward_chunks, done_chunks,
        battle_format, gamma, weights,
    )
    print(
        f"collected {len(dataset)} turns from {int(dataset.done.sum())} episodes "
        f"({dropped} unlabelable turns dropped)"
    )
    return dataset


async def collect_selfplay(
    learner: PlayerSpec,
    opponents: list[PlayerSpec],
    games_per_opp: int,
    battle_format: str = DEFAULT_FORMAT,
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    record_opponents: bool = True,
    max_concurrent: int = 1,
    max_battle_turns: int | None = None,
    team_pool: str | None = None,
) -> TrajectoryDataset:
    """Play ``learner`` against each opponent; return all turns + shaped rewards.

    The learner is rebuilt fresh per opponent (mirroring ``collect_trajectories``'s
    per-pair pattern). Both sides are recorded by default so the value head sees --
    and AWR can imitate -- the stronger anchor/opponent behaviour too, which is the
    whole point of keeping fixed anchors in the pool.

    ``team_pool`` is REQUIRED on a teambuilt format (gen9ou, VGC) and must be omitted on Random
    Battles, where the server supplies the teams. Each side of each pairing gets its own
    distinctly-seeded pool, so the two do not draw the same team in lockstep.

    REQUIRED is now enforced, because the failure without it is not an error. Showdown answers a
    teamless challenge on a teambuilt format with a popup -- "Your team was rejected ... This
    format requires you to use your own team" -- which poke-env logs as a WARNING. Collection then
    proceeds to gather nothing while every surface reads as running. A caller that forgot the pool
    should be told, in the same breath the docstring already used.
    """
    if "randombattle" not in battle_format and not team_pool:
        raise ValueError(
            f"collect_selfplay on the teambuilt format {battle_format!r} needs a `team_pool`; "
            f"without one Showdown rejects every challenge with a popup rather than an error and "
            f"the collection silently yields no episodes."
        )
    if not opponents:
        raise ValueError("Need at least one opponent to generate self-play games.")
    weights = weights or RewardWeights()

    obs_chunks: list[np.ndarray] = []
    action_chunks: list[int] = []
    mask_chunks: list[np.ndarray] = []
    reward_chunks: list[float] = []
    done_chunks: list[bool] = []
    dropped = 0

    make_team = _team_factory(team_pool)
    for i, opp in enumerate(opponents):
        pl = _build_recording_player(
            learner,
            battle_format,
            weights,
            max_concurrent,
            team=make_team(2 * i),
            max_battle_turns=max_battle_turns,
        )
        po = _build_recording_player(
            opp,
            battle_format,
            weights,
            max_concurrent,
            team=make_team(2 * i + 1),
            max_battle_turns=max_battle_turns,
        )
        await cross_evaluate([pl, po], n_challenges=games_per_opp)
        for player in (pl, po) if record_opponents else (pl,):
            dropped += _append_episodes(
                player, weights, obs_chunks, action_chunks, mask_chunks, reward_chunks, done_chunks
            )

    dataset = _trajectory_dataset(
        obs_chunks, action_chunks, mask_chunks, reward_chunks, done_chunks,
        battle_format, gamma, weights,
    )
    print(
        f"self-play: collected {len(dataset)} turns from {int(dataset.done.sum())} episodes "
        f"({dropped} unlabelable turns dropped)"
    )
    return dataset


def save_rl(dataset: TrajectoryDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = dataset.weights
    np.savez_compressed(
        path,
        obs=dataset.obs,
        action=dataset.action,
        mask=dataset.mask,
        reward=dataset.reward,
        done=dataset.done,
        obs_version=np.array(codec_for(dataset.battle_format).obs_version),
        obs_dim=np.array(codec_for(dataset.battle_format).obs_dim),
        battle_format=np.array(dataset.battle_format),
        gamma=np.array(dataset.gamma, dtype=np.float32),
        hp_value=np.array(w.hp_value, dtype=np.float32),
        fainted_value=np.array(w.fainted_value, dtype=np.float32),
        status_value=np.array(w.status_value, dtype=np.float32),
        victory_value=np.array(w.victory_value, dtype=np.float32),
    )
    print(f"wrote {len(dataset)} turns to {path}")


def concat_rl_shards(paths: Sequence[str | Path], out: str | Path) -> TrajectoryDataset:
    """Concatenate offline-RL shards into one shard and save it (self-play buffer).

    ``done`` flags are preserved end-to-end, so ``RLDataset``'s backward MC-return
    scan still resets at every episode boundary across the seam between shards.
    Metadata (format / gamma / shaping weights) is taken from the shards, which all
    share one self-play run's settings.
    """
    if not paths:
        raise ValueError("No shards to concatenate.")

    obs: list[np.ndarray] = []
    action: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    reward: list[np.ndarray] = []
    done: list[np.ndarray] = []
    battle_format = DEFAULT_FORMAT
    gamma = 0.99
    weights = RewardWeights()

    fingerprint: tuple[str, int] | None = None
    for p in paths:
        data = np.load(Path(p), allow_pickle=False)
        version, dim = str(data["obs_version"].item()), int(data["obs_dim"].item())
        # Shards must agree WITH EACH OTHER rather than with the singles constants: a doubles
        # buffer is legitimately (d1-, 888) and could otherwise never be concatenated. The check
        # still catches the case that matters -- mixing encoders inside one buffer.
        expected = codec_for(str(data["battle_format"].item())) if "battle_format" in data else None
        if expected is not None and (version, dim) != (expected.obs_version, expected.obs_dim):
            raise ValueError(
                f"Shard {p} encoder mismatch ({version}/{dim} vs "
                f"{expected.obs_version}/{expected.obs_dim} for {data['battle_format'].item()})."
            )
        if fingerprint is None:
            fingerprint = (version, dim)
        elif (version, dim) != fingerprint:
            raise ValueError(
                f"Shard {p} encoder ({version}/{dim}) differs from the first shard's "
                f"{fingerprint[0]}/{fingerprint[1]} -- refusing to mix encoders in one buffer."
            )
        obs.append(data["obs"])
        action.append(data["action"])
        mask.append(data["mask"])
        reward.append(data["reward"])
        done.append(data["done"])
        battle_format = str(data["battle_format"].item())
        gamma = float(data["gamma"].item())
        weights = RewardWeights(
            hp_value=float(data["hp_value"]),
            fainted_value=float(data["fainted_value"]),
            status_value=float(data["status_value"]),
            victory_value=float(data["victory_value"]),
        )

    dataset = TrajectoryDataset(
        obs=np.concatenate(obs).astype(np.float32),
        action=np.concatenate(action).astype(np.int64),
        mask=np.concatenate(mask).astype(bool),
        reward=np.concatenate(reward).astype(np.float32),
        done=np.concatenate(done).astype(bool),
        battle_format=battle_format,
        gamma=gamma,
        weights=weights,
    )
    save_rl(dataset, out)
    return dataset

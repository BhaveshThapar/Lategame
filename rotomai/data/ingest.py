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
* label the **move** a player used the instant it's revealed with ``label_action``
  (request-free codec) -- Build 5: both it and the encoder use the *canonical* move-slot
  order (sorted by move id), so the slot semantics match live play by construction;
* label a **switch** only when the target was already revealed earlier in the game (a
  pivot back), so the pre-switch obs and the switch index agree; first-reveal switches
  and forced/post-faint switches are *dropped*, not guessed;
* compute the shaped reward from public HP/faint/status (``data.reward.state_value``),
  diffing decision points exactly like ``collect._battle_rewards``.

Two POV gaps separate a public-log reconstruction from live play, and both are closed here:

* **Species presence** -- a player's own bench is revealed only as mons switch in, but live
  play sees all six from the request. Team-choice formats open with ``|poke|`` team-preview
  lines naming all six species per side, so ``_register_preview`` fills every own slot from
  turn 0 (poke-env keeps the opponent's preview separately; the encoder merges it). Random
  battles carry no preview, so this path is a no-op there.

* **Own-team detail** -- the log reveals the player's *own* item/ability/moves only progressively
  as they are used, but the live ``|request|`` hands over the full team from turn 1. Training on
  "my kit unknown" and playing on "my kit known" makes the identity-embedding channels
  (item/ability/move IDs in ``features.encoder``) out-of-distribution. We close this with
  **two-pass own-team completion**: ``_prescan_kits`` reads each own mon's full-game-revealed
  moves/item/ability (item recovered from the raw ``|-item|``/``|-enditem|`` lines, since
  poke-env resets a consumed item to ``None``), then ``_complete_own_team`` populates that kit
  onto the own team before every ``embed_battle`` so the training obs matches the live POV.
  Build 3 showed the log-only residual (kit a mon never revealed) still leaves those channels
  far from the live density -- and a *partial* POV fix is worthless. So ``_impute_kits`` fills
  each kit's still-unrevealed slots from the species' Smogon usage prior (``data.usage_prior``,
  usage-weighted + stably seeded per POV/mon, revealed truth always wins), taking every own mon
  to the full-kit detail the live request provides. For the own team this approximates the
  truth the player actually had.
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

from rotomai.config import DEFAULT_FORMAT
from rotomai.data.collect import Dataset, TrajectoryDataset, save, save_rl
from rotomai.data.reward import RewardWeights, state_value
from rotomai.data.usage_prior import UsagePrior, kit_seed, load_usage_prior, sample_kit
from rotomai.features.action_space import (
    GEN9_ACTION_SPACE_SIZE,
    label_action,
    synthesize_action_mask,
)
from rotomai.features.encoder import embed_battle

# poke-env logs a warning for every message it can't fully model on a request-less
# battle; that's expected here, so keep this quiet during bulk ingestion.
_LOGGER = logging.getLogger("rotomai.ingest")
_LOGGER.setLevel(logging.CRITICAL)

_Record = tuple[np.ndarray, int, np.ndarray]  # (obs, action, mask) -- matches collect._Record
_Sample = tuple[_Record, float]  # record + its shaped state-value

# poke-env's GenData.UNKNOWN_ITEM sentinel: an item slot that was never revealed. This is
# distinct from ``None`` (item consumed/knocked off) -- backfill fills the former, never the
# latter, so a consumed item stays ``None`` exactly as the live POV shows it post-consumption.
_UNKNOWN_ITEM = "unknown_item"


@dataclass
class _Kit:
    """One own mon's full-game-revealed kit, for two-pass own-team completion."""

    moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    species: str | None = None  # usage-prior lookup key (poke-env id-str)


_Kits = dict[str, _Kit]  # keyed by ``_team_key(ident)`` -> the mon's completed kit


@dataclass
class IngestStats:
    """Per-run counters; the drop rate is the key health signal for the pilot gate."""

    replays: int = 0  # replays seen
    parsed: int = 0  # replays reconstructed without a fatal error
    skipped_replays: int = 0  # replays dropped (no log / malformed / parse error)
    episodes: int = 0  # per-player trajectories kept
    turns: int = 0  # labelled decision turns kept
    dropped_turns: int = 0  # decisions seen but undecodable
    imputed_mons: int = 0  # own mons whose kit was usage-prior imputed
    usage_missing_mons: int = 0  # own mons with no usage-prior entry (left log-only)

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


def _register_preview(battle: AbstractBattle, parts: list[str]) -> None:
    """Seed an *own*-side previewed species into ``battle.team`` from a ``|poke|`` line.

    Team-choice formats reveal all six species per side at team preview. poke-env already
    keeps the *opponent's* preview mons (in ``teampreview_opponent_team``, which the encoder
    merges), but drops the player's *own* preview entirely -- so without this the ego bench is
    revealed only as mons switch in (the v1 gap). We inject each own previewed mon into
    ``battle.team`` via ``get_pokemon`` (the same path ``switch`` uses), so a later switch
    reconciles onto it by species -- nicknames included, since preview shows the species while a
    switch shows the nickname -- rather than creating a duplicate. Handling only the own side
    keeps the encoded POV identical to live play, where the request supplies the full own team and
    the opponent is seen via preview + reveals. Random battles carry no ``|poke|`` lines, so this
    is never called there.
    """
    if len(parts) < 4 or battle.player_role is None:
        return
    player, details = parts[2], parts[3]
    if player != battle.player_role:  # opponent preview is handled by the encoder merge
        return
    species = details.split(",")[0].strip()
    if not species:
        return
    try:
        battle.get_pokemon(f"{player}: {species}", details=details)
    except Exception:  # noqa: BLE001 -- a malformed preview line must not abort a replay
        pass


def _move_sample(
    battle: AbstractBattle,
    parts: list[str],
    tera: bool,
    weights: RewardWeights,
    kits: _Kits | None,
) -> _Sample | None:
    """Label the move just parsed (it is now in ``active.moves``) from this state."""
    active = battle.active_pokemon
    if active is None or len(parts) < 4:
        return None
    move = active.moves.get(to_id_str(parts[3]))
    if move is None:
        return None
    # Complete the own team before both labelling and encoding, so the canonical move slots
    # (label_action) and the obs move blocks read the same full moveset -- mutually consistent.
    if kits is not None:
        _complete_own_team(battle, kits)
    try:
        action = label_action(SingleBattleOrder(move, terastallize=tera), battle)
    except Exception:  # noqa: BLE001
        return None
    if not 6 <= action < GEN9_ACTION_SPACE_SIZE:  # must decode to a move slot
        return None
    obs = embed_battle(battle)
    return (obs, action, synthesize_action_mask(battle, action)), state_value(battle, weights)


def _switch_sample(
    battle: AbstractBattle, ident: str, weights: RewardWeights, kits: _Kits | None
) -> _Sample | None:
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
    # Backfill adds moves/item/ability to existing mons only, so the team roster and its
    # ``values()`` order (which the switch index depends on) are unchanged.
    if kits is not None:
        _complete_own_team(battle, kits)
    try:
        action = label_action(SingleBattleOrder(target), battle)
    except Exception:  # noqa: BLE001
        return None
    if not 0 <= action < 6:  # must decode to a switch slot
        return None
    obs = embed_battle(battle)
    return (obs, action, synthesize_action_mask(battle, action)), state_value(battle, weights)


def _complete_own_team(battle: AbstractBattle, kits: _Kits) -> None:
    """Populate each own mon with its full-game-revealed kit (two-pass completion).

    Idempotent and monotonic: adds only missing moves, fills an item only if the slot is
    still the unrevealed sentinel (never overwrites ``None``, which marks a consumed item the
    live POV also shows as ``None``), and fills an ability only if still unknown. Applied
    before every ``embed_battle`` so the training obs matches the live request-based POV.
    """
    for key, mon in battle.team.items():
        kit = kits.get(key)
        if kit is None or mon.transformed:  # a transformed mon's live request shows the copy
            continue
        # sorted() keeps shards byte-reproducible; slot semantics no longer depend on this
        # insertion order -- the codec/encoder canonicalize (Build 5, features.action_space).
        for move_id in sorted(kit.moves):
            if move_id not in mon.moves:
                mon._add_move(move_id)
        if kit.item is not None and mon.item == _UNKNOWN_ITEM:
            mon.item = kit.item
        if kit.ability is not None and mon.ability is None:
            mon.ability = kit.ability


def _prescan_kits(
    lines: list[str], username: str, battle_tag: str, gen: int, weights: RewardWeights
) -> _Kits:
    """Pass 1: reconstruct the full POV once and read each own mon's full-game-revealed kit.

    Moves and abilities are read off the fully-parsed ``battle.team``; the item is read there
    too, but poke-env resets a consumed/knocked-off item to ``None`` (``Pokemon.end_item``), so
    those are recovered from the raw ``|-item|``/``|-enditem|`` reveal lines. Kit a mon never
    revealed stays unknown -- the inherent public-log residual.
    """
    battle, _, _, _ = _reconstruct_pov(lines, username, battle_tag, gen, weights, kits=None)
    kits: _Kits = {}
    for key, mon in battle.team.items():
        item = mon.item if mon.item and mon.item != _UNKNOWN_ITEM else None
        kits[key] = _Kit(
            moves=set(mon.moves.keys()),
            item=item,
            ability=mon.ability or None,
            species=mon.species,
        )

    for raw in lines:
        if not raw.startswith("|"):
            continue
        parts = raw.split("|")
        if len(parts) < 4 or parts[1] not in ("-item", "-enditem") or not _own(parts, battle):
            continue
        kit = kits.get(_team_key(parts[2]))
        if kit is not None and kit.item is None:
            kit.item = to_id_str(parts[3])
    return kits


def _impute_kits(kits: _Kits, battle_tag: str, prior: UsagePrior) -> tuple[int, int]:
    """Fill each kit's still-unrevealed slots from the species' usage prior (Build 4).

    Runs once per POV between pass 1 and pass 2, so the sampled kit is constant across the
    replay's timesteps, and only ever touches kits -- revealed truth wins by construction
    (moves pad the never-used slots up to 4, so the labelled action is never imputed; item
    and ability fill only still-unknown slots; a drawn ``"nothing"`` item stays ``None``,
    encoder-identical to a live no-item mon). Returns ``(imputed_mons, missing_mons)`` where
    missing counts own mons whose species has no usage entry (left log-only, the fallback).
    """
    imputed = missing = 0
    for key, kit in kits.items():
        sampled = (
            sample_kit(
                prior,
                kit.species,
                kit.moves,
                need_item=kit.item is None,
                need_ability=kit.ability is None,
                seed=kit_seed(battle_tag, key),
            )
            if kit.species is not None
            else None
        )
        if sampled is None:
            missing += 1
            continue
        moves, item, ability = sampled
        kit.moves.update(moves)
        if item is not None and kit.item is None:
            kit.item = item
        if ability is not None and kit.ability is None:
            kit.ability = ability
        imputed += 1
    return imputed, missing


def _reconstruct_pov(
    lines: list[str],
    username: str,
    battle_tag: str,
    gen: int,
    weights: RewardWeights,
    kits: _Kits | None = None,
) -> tuple[AbstractBattle, list[_Record], list[float], int]:
    """Replay ``lines`` into one player's POV; return its battle + labelled decisions.

    A decision is the first ``|move|`` or qualifying voluntary ``|switch|`` the player
    makes each turn. Move obs is snapshotted *after* the move line (so the move is
    revealed and indexable); switch obs *before* (so the old active is still shown).

    When ``kits`` is provided (two-pass own-team completion), the own team is backfilled to its
    full-game kit before each obs snapshot; ``None`` reproduces the v1 progressive-reveal POV
    (used by Pass 1 and as the fidelity gate's negative control).
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

        if tag == "poke":
            # Team preview, before turn 1: seed both sides' full six-species rosters so
            # the encoder and switch mask see them from the start (team-choice formats).
            _safe_parse(battle, parts)
            _register_preview(battle, parts)
            continue

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
            sample = _switch_sample(battle, parts[2], weights, kits)
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
            sample = _move_sample(battle, parts, tera, weights, kits)
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
    complete_own_team: bool = True,
    impute_usage: bool = True,
) -> tuple[TrajectoryDataset, Dataset | None, IngestStats]:
    """Reconstruct ``replays`` into an offline-RL shard (+ a winners-only BC shard).

    Returns ``(rl_dataset, bc_dataset_or_None, stats)``. The RL shard keeps every
    labelled turn of every finished POV; the BC shard keeps only the winning POVs'
    turns (the M2 reward filter). Both use the exact ``collect`` on-disk schema.

    ``complete_own_team`` (default on) enables two-pass own-team completion so the training
    obs matches the live request-based POV; ``False`` reproduces the v1 progressive-reveal POV.
    ``impute_usage`` (default on) additionally fills the still-unrevealed kit slots from the
    format's committed usage prior; it is a no-op when no artifact exists for the format.
    """
    weights = weights or RewardWeights()
    builder = _ShardBuilder()
    stats = IngestStats()
    prior = load_usage_prior(battle_format) if (impute_usage and complete_own_team) else None

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
                pov_tag = f"{tag}-{username}"
                kits = (
                    _prescan_kits(lines, str(username), pov_tag, gen, weights)
                    if complete_own_team
                    else None
                )
                if kits is not None and prior is not None:
                    imputed, missing = _impute_kits(kits, pov_tag, prior)
                    stats.imputed_mons += imputed
                    stats.usage_missing_mons += missing
                battle, records, values, dropped = _reconstruct_pov(
                    lines, str(username), pov_tag, gen, weights, kits=kits
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
    complete_own_team: bool = True,
    impute_usage: bool = True,
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

    return ingest_replays(_load(), weights, gamma, battle_format, complete_own_team, impute_usage)


def ingest_and_save(
    paths: Iterable[str | Path],
    rl_out: str | Path,
    bc_out: str | Path | None = None,
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
    complete_own_team: bool = True,
    impute_usage: bool = True,
) -> IngestStats:
    """Ingest cached replay files and write the RL shard (+ optional BC shard)."""
    rl, bc, stats = ingest_replay_files(
        paths, weights, gamma, battle_format, complete_own_team, impute_usage
    )
    save_rl(rl, rl_out)
    if bc is not None and bc_out is not None:
        save(bc, bc_out)
    print(
        f"ingested {stats.parsed}/{stats.replays} replays "
        f"({stats.skipped_replays} skipped) -> {stats.turns} turns from "
        f"{stats.episodes} POV-episodes; drop rate {stats.drop_rate:.1%}; "
        f"usage-imputed {stats.imputed_mons} mons ({stats.usage_missing_mons} missing species)"
    )
    return stats

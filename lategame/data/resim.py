"""Re-simulated full-fidelity replay ingestion (plan.md, M6 v2).

M6 v1 (``data.ingest``) reconstructs each player's POV from a *public* spectator log,
which carries no ``|request|`` JSON. That left the agent's own bench hidden until a mon
switched in -- an observation the live agent never faces, since the request reveals all
six team members, every move, and exact HP. Training on that mismatched obs made the
policy *worse*, not better (the v1 pilot regressed below the heuristic).

This module closes that gap. The replay JSON also stores ``inputlog``: the PRNG seed plus
every choice both players made. Re-driving it through the vendored pokemon-showdown
simulator (``resim_driver.js``) deterministically reproduces the battle AND re-emits each
side's private ``|request|`` stream. We feed that stream to a poke-env ``Battle`` exactly
as the live ``Player`` does (``parse_request`` + ``parse_message``), so the reconstructed
POV is *identical in distribution to live play* -- and we can label with the real,
request-based strict codec (``order_to_action``/``action_mask``) instead of v1's
request-free approximations. Switches are no longer dropped: the full team is known.

Output schema is the exact ``collect`` ``(obs, action, mask, reward, done)`` shape, reusing
``ingest._ShardBuilder``, so ``train-rl``/``train`` consume the shard unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from poke_env import to_id_str
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import BattleOrder, SingleBattleOrder

from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import Dataset, TrajectoryDataset, save, save_rl
from lategame.data.ingest import IngestStats, _safe_parse, _ShardBuilder
from lategame.data.reward import RewardWeights, state_value
from lategame.features.action_space import (
    GEN9_ACTION_SPACE_SIZE,
    action_mask,
    order_to_action,
)
from lategame.features.encoder import embed_battle

# poke-env emits a warning for every message it can't fully model; expected here, so quiet.
_LOGGER = logging.getLogger("lategame.resim")
_LOGGER.setLevel(logging.CRITICAL)

# Mirror the live Player's ignore set (poke_env.player.Player.MESSAGES_TO_IGNORE).
_IGNORE = {"t:", "expire", "uhtmlchange"}

_DEFAULT_SHOWDOWN = "third_party/pokemon-showdown"
_DRIVER = Path(__file__).with_name("resim_driver.js")

_Record = tuple[np.ndarray, int, np.ndarray]  # (obs, action, mask) -- matches collect._Record
_Sample = tuple[_Record, float]
_Event = Mapping[str, str]  # {"t": "msg"|"choice", "d": ...}


# --------------------------------------------------------------------------- #
# Node driver subprocess.
# --------------------------------------------------------------------------- #


def run_driver(
    replays: Iterable[Mapping[str, str]],
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> Iterable[Mapping[str, object]]:
    """Re-simulate ``replays`` via the node driver; yield one result dict per replay.

    Each input is ``{"id", "inputlog"}``; each output is ``{"id", "p1", "p2"}`` on success
    or ``{"id", "error"}`` for a single bad replay. One long-lived node process handles the
    whole batch (NDJSON in, NDJSON out) so node startup is paid once.
    """
    if not _DRIVER.exists():
        raise RuntimeError(f"resim driver missing: {_DRIVER}")
    if not (Path(showdown_dir) / "dist").is_dir():
        raise RuntimeError(
            f"'{showdown_dir}/dist' not found -- build the vendored simulator first "
            "(run `bash scripts/setup_server.sh`)."
        )
    payload = "".join(
        json.dumps({"id": r.get("id"), "inputlog": r.get("inputlog", "")}) + "\n"
        for r in replays
    )
    proc = subprocess.run(
        [node, str(_DRIVER), str(showdown_dir)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"resim driver failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


# --------------------------------------------------------------------------- #
# Per-side POV reconstruction (mirrors poke_env.player.Player._handle_battle_message).
# --------------------------------------------------------------------------- #


def _feed_line(battle: AbstractBattle, line: str) -> Mapping[str, object] | None:
    """Dispatch one ``|...|`` line into ``battle``; return the request dict if any.

    Mirrors the live Player loop: ``|request|`` -> ``parse_request``, ``|win|``/``|tie|``
    drive the terminal helpers, ignored tags are skipped, everything else is parsed.
    """
    if not line.startswith("|"):
        return None
    parts = line.split("|")
    if len(parts) < 2:
        return None
    tag = parts[1]
    if tag == "request":
        if len(parts) > 2 and parts[2]:
            try:
                request = json.loads(parts[2])
            except ValueError:
                return None
            try:
                battle.parse_request(request)
            except Exception:  # noqa: BLE001 -- one odd request must not abort the POV
                return None
            return request if isinstance(request, dict) else None
        return None
    if tag in _IGNORE:
        return None
    if tag == "win":
        try:
            battle.won_by(parts[2] if len(parts) > 2 else "")
        except Exception:  # noqa: BLE001
            pass
        return None
    if tag == "tie":
        try:
            battle.tied()
        except Exception:  # noqa: BLE001
            pass
        return None
    _safe_parse(battle, parts)
    return None


def _resolve_move(battle: AbstractBattle, token: str):  # noqa: ANN201 -- poke-env Move
    """Find the Move an inputlog ``move <id|slot>`` token refers to (or None)."""
    active = battle.active_pokemon
    if active is None:
        return None
    if token.isdigit():
        idx = int(token) - 1
        moves = battle.available_moves
        return moves[idx] if 0 <= idx < len(moves) else None
    move_id = to_id_str(token)
    move = active.moves.get(move_id)
    if move is not None:
        return move
    for candidate in battle.available_moves:
        if candidate.id == move_id:
            return candidate
    return None


def _resolve_switch(
    battle: AbstractBattle, request: Mapping[str, object] | None, token: str
):  # noqa: ANN201 -- poke-env Pokemon
    """Find the Pokemon an inputlog ``switch <slot|name>`` token refers to (or None).

    A numeric slot indexes the request's ``side.pokemon`` party order (the canonical
    order the server uses); we map its ``ident`` to this POV's ``battle.team`` entry.
    """
    ident: str | None = None
    if token.isdigit():
        side = request.get("side") if isinstance(request, Mapping) else None
        pokemons = side.get("pokemon") if isinstance(side, Mapping) else None
        if isinstance(pokemons, list):
            idx = int(token) - 1
            if 0 <= idx < len(pokemons):
                entry = pokemons[idx]
                if isinstance(entry, Mapping):
                    raw = entry.get("ident")
                    ident = raw if isinstance(raw, str) else None
    else:
        ident = token
    if ident is None:
        return None
    target = battle.team.get(ident)
    if target is not None:
        return target
    species = to_id_str(ident.split(":")[-1])
    for mon in battle.team.values():
        if to_id_str(mon.species) == species or to_id_str(mon.base_species) == species:
            return mon
    return None


def _build_order(
    battle: AbstractBattle, request: Mapping[str, object] | None, choice: str
) -> BattleOrder | None:
    """Build a poke-env order from an inputlog choice string, or None if undecodable."""
    tokens = choice.split()
    if not tokens:
        return None
    if tokens[0] == "move" and len(tokens) >= 2:
        move = _resolve_move(battle, tokens[1])
        if move is None:
            return None
        return SingleBattleOrder(move, terastallize="terastallize" in tokens[2:])
    if tokens[0] == "switch" and len(tokens) >= 2:
        target = _resolve_switch(battle, request, tokens[1])
        if target is None:
            return None
        return SingleBattleOrder(target)
    return None  # default / pass / team / undecodable -> dropped


def _decision_sample(
    battle: AbstractBattle,
    request: Mapping[str, object] | None,
    choice: str,
    weights: RewardWeights,
) -> _Sample | None:
    """Label the inputlog ``choice`` made at this request with the strict codec."""
    if battle._wait or battle.teampreview or battle.active_pokemon is None:
        return None
    order = _build_order(battle, request, choice)
    if order is None:
        return None
    try:
        action = order_to_action(order, battle)  # strict: validates against the request
    except Exception:  # noqa: BLE001
        return None
    if not 0 <= action < GEN9_ACTION_SPACE_SIZE:
        return None
    obs = embed_battle(battle)
    return (obs, action, action_mask(battle)), state_value(battle, weights)


def _reconstruct_pov_resim(
    events: Iterable[_Event],
    username: str,
    battle_tag: str,
    gen: int,
    weights: RewardWeights,
) -> tuple[AbstractBattle, list[_Record], list[float], int]:
    """Replay one side's re-sim event stream into its POV + labelled decisions.

    Same signature as ``ingest._reconstruct_pov``. ``msg`` events carry public-log lines
    (channel-filtered to this POV by the driver); a ``choice`` event marks the decision
    answering the immediately-preceding request -- that is where a sample is recorded.
    """
    battle = Battle(battle_tag=battle_tag, username=username, gen=gen, logger=_LOGGER)
    records: list[_Record] = []
    values: list[float] = []
    dropped = 0
    last_request: Mapping[str, object] | None = None

    for event in events:
        if event.get("t") == "msg":
            for line in event.get("d", "").split("\n"):
                request = _feed_line(battle, line)
                if request is not None:
                    last_request = request
        elif event.get("t") == "choice":
            sample = _decision_sample(battle, last_request, event.get("d", ""), weights)
            if sample is None:
                dropped += 1
            else:
                records.append(sample[0])
                values.append(sample[1])

    return battle, records, values, dropped


def _parse_inputlog_meta(inputlog: str) -> tuple[str, str, int]:
    """Extract (p1_name, p2_name, gen) from an inputlog's ``>player``/``>start`` lines."""
    names = {"p1": "p1", "p2": "p2"}
    gen = 9
    for line in inputlog.split("\n"):
        if line.startswith(">player p1 ") or line.startswith(">player p2 "):
            side = line[8:10]
            try:
                spec = json.loads(line.split(" ", 2)[2])
            except (ValueError, IndexError):
                continue
            if isinstance(spec, dict) and spec.get("name"):
                names[side] = str(spec["name"])
        elif line.startswith(">start "):
            try:
                spec = json.loads(line[len(">start ") :])
            except ValueError:
                continue
            if not isinstance(spec, dict):
                continue
            match = re.match(r"gen(\d+)", str(spec.get("formatid", "")))
            if match:
                gen = int(match.group(1))
    return names["p1"], names["p2"], gen


# --------------------------------------------------------------------------- #
# Public API (mirrors data.ingest).
# --------------------------------------------------------------------------- #


def resim_replays(
    replays: Iterable[Mapping[str, object]],
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> tuple[TrajectoryDataset, Dataset | None, IngestStats]:
    """Re-simulate ``replays`` into an offline-RL shard (+ a winners-only BC shard).

    Each replay must carry ``inputlog`` (the cached replay JSON does). Returns
    ``(rl_dataset, bc_dataset_or_None, stats)`` in the exact ``collect`` schema.
    """
    weights = weights or RewardWeights()
    builder = _ShardBuilder()
    stats = IngestStats()

    prepared: list[tuple[str, str, int, str, str]] = []
    for replay in replays:
        stats.replays += 1
        inputlog = replay.get("inputlog")
        if not isinstance(inputlog, str) or not inputlog:
            stats.skipped_replays += 1
            continue
        rid = str(replay.get("id") or f"replay{stats.replays}")
        p1_name, p2_name, gen = _parse_inputlog_meta(inputlog)
        prepared.append((rid, inputlog, gen, p1_name, p2_name))

    results = {
        str(res.get("id")): res
        for res in run_driver(
            ({"id": rid, "inputlog": il} for rid, il, _, _, _ in prepared),
            showdown_dir,
            node,
        )
    }

    for rid, _inputlog, gen, p1_name, p2_name in prepared:
        res = results.get(rid)
        if res is None or "error" in res:
            stats.skipped_replays += 1
            continue
        try:
            for side, username in (("p1", p1_name), ("p2", p2_name)):
                raw = res.get(side)
                events = raw if isinstance(raw, list) else []
                battle, records, values, dropped = _reconstruct_pov_resim(
                    events, username, f"{rid}-{side}", gen, weights
                )
                stats.dropped_turns += dropped
                kept = builder.add_pov(battle, records, values, weights)
                if kept:
                    stats.episodes += 1
                    stats.turns += kept
        except Exception:  # noqa: BLE001 -- one bad replay shouldn't kill the run
            stats.skipped_replays += 1
            continue
        stats.parsed += 1

    rl, bc = builder.finalize(battle_format, gamma, weights)
    return rl, bc, stats


def resim_replay_files(
    paths: Iterable[str | Path],
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> tuple[TrajectoryDataset, Dataset | None, IngestStats]:
    """Load cached replay ``.json`` files and re-simulate them (see ``data.replays``)."""

    def _load() -> Iterable[Mapping[str, object]]:
        for path in paths:
            try:
                with open(path, encoding="utf-8") as fh:
                    yield json.load(fh)
            except (OSError, ValueError):
                continue

    return resim_replays(_load(), weights, gamma, battle_format, showdown_dir, node)


def resim_and_save(
    paths: Iterable[str | Path],
    rl_out: str | Path,
    bc_out: str | Path | None = None,
    weights: RewardWeights | None = None,
    gamma: float = 0.99,
    battle_format: str = DEFAULT_FORMAT,
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> IngestStats:
    """Re-simulate cached replay files and write the RL shard (+ optional BC shard)."""
    rl, bc, stats = resim_replay_files(paths, weights, gamma, battle_format, showdown_dir, node)
    save_rl(rl, rl_out)
    if bc is not None and bc_out is not None:
        save(bc, bc_out)
    print(
        f"resimulated {stats.parsed}/{stats.replays} replays "
        f"({stats.skipped_replays} skipped) -> {stats.turns} turns from "
        f"{stats.episodes} POV-episodes; drop rate {stats.drop_rate:.1%}"
    )
    return stats

"""R-EVAL seed: run battles between two agents and report win rates.

Wraps poke-env's ``cross_evaluate`` against the local server. Win rate is the
right phase-1 signal; bias-robust metrics (GXE / Glicko-1) come later -- see
plan.md, requirement R-EVAL.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    cross_evaluate,
)

from lategame.agents.bc_agent import BCAgent
from lategame.agents.heuristic_agent import HeuristicAgent
from lategame.agents.offline_rl_agent import OfflineRLAgent
from lategame.agents.ppo_agent import PPORecordingAgent
from lategame.config import DEFAULT_FORMAT, LOCAL_SERVER, local_account

# BCAgent / OfflineRLAgent / PPORecordingAgent are torch-free to import (torch loads
# lazily on instantiation), so registering them here does not pull torch into M0/M1.
AGENTS: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "maxbasepower": MaxBasePowerPlayer,
    "simpleheuristics": SimpleHeuristicsPlayer,
    "heuristic": HeuristicAgent,
    "bc": BCAgent,
    "offrl": OfflineRLAgent,
    "ppo": PPORecordingAgent,
}

# Agents backed by a trained checkpoint accept ``checkpoint_path``/``sample`` kwargs;
# the fixed baselines do not. Shared with ``data.collect`` so both build the same way.
_CHECKPOINT_AGENTS = {"bc", "offrl", "ppo"}


@dataclass
class EvalResult:
    p1_name: str
    p2_name: str
    n_battles: int
    battle_format: str
    p1_win_rate: float


def _unique_username(name: str) -> str:
    # Showdown usernames are short; keep a stable prefix + random suffix so
    # repeated runs never collide.
    return f"{name[:10]}-{secrets.token_hex(3)}"


def build_player(
    name: str,
    battle_format: str,
    checkpoint_path: str | None = None,
    sample: bool = False,
    max_concurrent_battles: int | None = None,
) -> Player:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Choose from: {', '.join(AGENTS)}")
    cls = AGENTS[name]
    extra: dict[str, object] = {}
    if name in _CHECKPOINT_AGENTS:
        extra["sample"] = sample
        if checkpoint_path is not None:
            extra["checkpoint_path"] = checkpoint_path
    # poke-env defaults to one battle at a time; PPO rollouts/eval pass a higher value
    # so cross_evaluate keeps many battles in flight (the local server is the bottleneck).
    if max_concurrent_battles is not None:
        extra["max_concurrent_battles"] = max_concurrent_battles
    return cls(
        account_configuration=local_account(_unique_username(name)),
        battle_format=battle_format,
        server_configuration=LOCAL_SERVER,
        **extra,  # type: ignore[arg-type]
    )


async def evaluate_built(p1: Player, p2: Player, n_battles: int) -> float:
    """Win rate of ``p1`` vs ``p2`` over ``n_battles`` on the local server.

    The low-level core shared by ``evaluate`` and the self-play loop, which needs to
    pit two already-built players (e.g. a specific checkpoint vs a baseline).
    """
    results = await cross_evaluate([p1, p2], n_challenges=n_battles)
    win_rate = results[p1.username][p2.username]
    return 0.0 if win_rate is None else float(win_rate)


async def evaluate(
    p1_name: str,
    p2_name: str,
    n_battles: int,
    battle_format: str = DEFAULT_FORMAT,
) -> EvalResult:
    p1 = build_player(p1_name, battle_format)
    p2 = build_player(p2_name, battle_format)
    win_rate = await evaluate_built(p1, p2, n_battles)
    return EvalResult(
        p1_name=p1_name,
        p2_name=p2_name,
        n_battles=n_battles,
        battle_format=battle_format,
        p1_win_rate=win_rate,
    )

"""R-EVAL: run battles between two agents and report win rate, Glicko-1 and GXE.

Wraps poke-env's ``cross_evaluate`` against the local server. Win rate remains the phase-1
signal and the authoritative build-vs-build statistic (``scripts/seed_strength_gate.py``);
the bias-robust metrics from ``lategame.eval.rating`` ride alongside it. Against the fixed
baselines used here they are a reparameterisation of the win rate and carry no extra
information -- they exist for M5 ladder play, where matchmaking pushes win rate toward 50%.
See plan.md 12, requirement R-EVAL.
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
from poke_env.teambuilder.teambuilder import Teambuilder

from lategame.agents.bc_agent import BCAgent
from lategame.agents.heuristic_agent import HeuristicAgent
from lategame.agents.offline_rl_agent import OfflineRLAgent
from lategame.agents.ppo_agent import PPORecordingAgent
from lategame.agents.search_agent import SearchAgent
from lategame.config import DEFAULT_FORMAT, LOCAL_SERVER, local_account
from lategame.eval.rating import gxe, rate_win_rate

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
    "search": SearchAgent,
}

# Agents backed by a trained checkpoint accept ``checkpoint_path``/``sample`` kwargs;
# the fixed baselines do not. Shared with ``data.collect`` so both build the same way.
_CHECKPOINT_AGENTS = {"bc", "offrl", "ppo", "search"}

# Learned-policy agents whose constructors accept a Build-14 ``loop_penalty`` (LoopGuard);
# ``search`` is checkpoint-backed but has no LoopGuard. ``LoopGuard(0.0)`` is exact identity,
# so forwarding the default 0.0 is a no-op for every existing caller.
_LOOP_GUARD_AGENTS = {"bc", "offrl", "ppo"}


@dataclass
class EvalResult:
    p1_name: str
    p2_name: str
    n_battles: int
    battle_format: str
    p1_win_rate: float
    # R-EVAL's bias-robust metrics. Against ONE fixed opponent these are a reparameterisation of
    # `p1_win_rate` and carry nothing new -- they exist for M5 ladder play, where the opponent
    # field varies and win rate is pushed toward 50% by matchmaking. See lategame/eval/rating.py.
    # Optional so every existing caller and gate is unchanged.
    p1_glicko: float | None = None
    p1_glicko_rd: float | None = None
    p1_gxe: float | None = None


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
    team: str | Teambuilder | None = None,
    loop_penalty: float = 0.0,
) -> Player:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Choose from: {', '.join(AGENTS)}")
    cls = AGENTS[name]
    extra: dict[str, object] = {}
    if name in _CHECKPOINT_AGENTS:
        extra["sample"] = sample
        if checkpoint_path is not None:
            extra["checkpoint_path"] = checkpoint_path
    # Build-14 decision-time anti-repetition. Only the learned policies carry a LoopGuard;
    # 0.0 is exact identity so this changes nothing unless a caller opts in.
    if name in _LOOP_GUARD_AGENTS:
        extra["loop_penalty"] = loop_penalty
    # poke-env defaults to one battle at a time; PPO rollouts/eval pass a higher value
    # so cross_evaluate keeps many battles in flight (the local server is the bottleneck).
    if max_concurrent_battles is not None:
        extra["max_concurrent_battles"] = max_concurrent_battles
    # Teambuilt formats (e.g. gen9ou) need a team; Random Battles leave this None and the
    # server supplies one. Accepts a packed/Showdown string or a Teambuilder (R-TEAM pool).
    if team is not None:
        extra["team"] = team
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
    # p2 is pinned at the reference rating rather than rated itself: with a single fixed opponent
    # there is nothing in the record to separate "p1 is strong" from "p2 is weak". That is a
    # convention, not a measurement -- rating.py's docstring says so at length.
    rated = rate_win_rate(win_rate, n_battles)
    return EvalResult(
        p1_name=p1_name,
        p2_name=p2_name,
        n_battles=n_battles,
        battle_format=battle_format,
        p1_win_rate=win_rate,
        p1_glicko=round(rated.rating, 1),
        p1_glicko_rd=round(rated.rd, 1),
        p1_gxe=round(gxe(rated), 4),
    )

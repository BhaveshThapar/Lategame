"""R-EVAL: run battles between two agents and report win rate, Glicko-1 and GXE.

Drives battles against the local server via ``Player.battle_against``. Win rate remains the phase-1
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
)
from poke_env.teambuilder.teambuilder import Teambuilder

from lategame.agents.bc_agent import BCAgent
from lategame.agents.doubles_agent import DoublesAgent
from lategame.agents.heuristic_agent import HeuristicAgent
from lategame.agents.offline_rl_agent import OfflineRLAgent
from lategame.agents.ppo_agent import PPORecordingAgent
from lategame.agents.search_agent import SearchAgent
from lategame.config import DEFAULT_FORMAT, LOCAL_SERVER, is_doubles_format, local_account
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
    # G4/M6: the learned DOUBLES policy. Separate from `offrl` rather than a mode of it,
    # because it reads a different encoder and a different action codec -- and the
    # checkpoint fingerprint (d1-/888) makes loading the wrong one an error, not a
    # silent mis-encode.
    "doubles": DoublesAgent,
}

# Agents backed by a trained checkpoint accept ``checkpoint_path``/``sample`` kwargs;
# the fixed baselines do not. Shared with ``data.collect`` so both build the same way.
_CHECKPOINT_AGENTS = {"bc", "offrl", "ppo", "search", "doubles"}

# Learned-policy agents whose constructors accept a Build-14 ``loop_penalty`` (LoopGuard);
# ``search`` is checkpoint-backed but has no LoopGuard. ``LoopGuard(0.0)`` is exact identity,
# so forwarding the default 0.0 is a no-op for every existing caller.
_LOOP_GUARD_AGENTS = {"bc", "offrl", "ppo"}

# SINGLES-ONLY, AND THE FAILURE ON DOUBLES IS SILENT RATHER THAN LOUD.
#
# The SINGLES learned agents assume one active Pokemon per side: the 761-d encoder's OBS_LAYOUT
# has one active slot per side, and `action_space` is built on `SinglesEnv.get_action_space_size(9)`
# (26 actions, against doubles' 107 per slot x 2 slots). On a `DoubleBattle` poke-env passes
# `active_pokemon` as a LIST, so a singles policy reaches `list.types` and raises AttributeError.
#
# Two agents are NOT in this set because they were made doubles-capable (G4/M6):
#   * `heuristic` -- a per-slot decision joined into a DoubleBattleOrder, built because the VGC
#     ceiling probe needs two agents near the top of the doubles gradient and poke-env supplies one.
#   * `doubles`   -- the learned per-slot policy over the 888-d doubles encoder and the 2x107
#     factored action codec.
#
# That exception never reaches a caller. poke-env dispatches every protocol message through
# `asyncio.create_task` and only calls `add_done_callback(discard)` -- nobody retrieves the result
# -- so the raise is logged and swallowed, exactly as `ShowdownException` is on a failed login
# (plan.md 13.1, M5/G1). The agent then simply never answers the request and the SERVER plays a
# default move for it on the timer. A ceiling probe anchored on an agent in that state would
# measure the timer, read it as enormous headroom, and be wrong in the most expensive direction.
#
# Refused at BUILD time, where the format string is in hand and the error is synchronous, rather
# than at choose-move time where it would vanish. poke-env's own baselines branch on `DoubleBattle`
# and are safe (`RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`).
_SINGLES_ONLY_AGENTS = {"bc", "offrl", "ppo", "search"}


def _singles_only_agents() -> set[str]:
    """``_SINGLES_ONLY_AGENTS``, minus ``search`` when it is running WITHOUT a trained model.

    ``search`` is singles-only because of what it evaluates leaves with, not because of the search
    itself: a ``PolicyValue`` leaf calls ``embed_battle``/``action_mask``, both singles-only. In
    shaped-only mode (``LATEGAME_SEARCH_SHAPED_ONLY=1``) the leaf is ``data.reward.state_value``,
    which reads the battle object directly, the driver enumerates JOINT doubles orders, and the
    fallback is the doubles-capable heuristic rule -- so no singles assumption survives anywhere on
    that path. That mode is exactly what the VGC M2 ceiling probe runs.
    """
    import os

    if os.environ.get("LATEGAME_SEARCH_SHAPED_ONLY") == "1":
        return _SINGLES_ONLY_AGENTS - {"search"}
    return _SINGLES_ONLY_AGENTS


_DOUBLES_SAFE_AGENTS = tuple(sorted(set(AGENTS) - _SINGLES_ONLY_AGENTS))


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
    if name in _singles_only_agents() and is_doubles_format(battle_format):
        raise ValueError(
            f"'{name}' is a SINGLES-ONLY agent and {battle_format!r} is a doubles format. It would "
            f"not fail loudly there: poke-env hands doubles agents `active_pokemon` as a list, the "
            f"resulting AttributeError is swallowed by poke-env's detached message task, and the "
            f"server plays default moves on the timer instead -- which reads as a very weak agent "
            f"rather than as a broken one.\n"
            f"Doubles-native agents: {', '.join(_DOUBLES_SAFE_AGENTS)}.\n"
            f"Making the learned agents doubles-capable is the G4/M6 build (plan.md 13): a "
            f"per-slot action head and a two-active encoder."
        )
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
    """Score rate of ``p1`` vs ``p2`` over ``n_battles`` on the local server.

    The low-level core shared by ``evaluate`` and the self-play loop, which needs to
    pit two already-built players (e.g. a specific checkpoint vs a baseline).

    **This no longer goes through ``cross_evaluate``, which cannot represent a tie.** For two
    players that helper is exactly ``battle_against`` followed by ``Player.win_rate``, and
    ``win_rate`` is ``n_won_battles / n_finished_battles`` -- so a tie sits in the denominator
    without contributing to the numerator and is scored as a *loss*. The correct score gives it a
    half, which is what ``eval/ladder.py``'s ``PairResult`` has always done and what the Glicko /
    Bradley-Terry fit downstream assumes. Booked in plan.md 13.1 (M5/G1) at the time and deferred
    only because ``arena`` was on the frozen path of the in-flight Build 26 jobs.

    **Identical to the old number on any tie-free record** (``(w + 0.5*0)/f == w/f``), which is
    every result this project has published -- singles ties are vanishingly rare on ``gen9ou``.
    Pinned by test rather than argued.

    Scoring is by DIFFERENCING the counters around the games, as ``ladder.run_round_robin`` does:
    poke-env's counters run over ``player._battles``, and ``cross_evaluate``'s ``reset_battles``
    *raises* while any battle is unfinished, so one stuck battle could take a whole gate down.
    """
    before = (p1.n_won_battles, p1.n_finished_battles, p1.n_tied_battles)
    await p1.battle_against(p2, n_battles=n_battles)
    won, finished, tied = (
        now - was
        for was, now in zip(
            before,
            (p1.n_won_battles, p1.n_finished_battles, p1.n_tied_battles),
            strict=True,
        )
    )
    return (won + 0.5 * tied) / finished if finished else 0.0


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

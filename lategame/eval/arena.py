"""R-EVAL: run battles between two agents and report win rate, Glicko-1 and GXE.

Drives battles against the local server via ``Player.battle_against``. Win rate remains the phase-1
signal and the authoritative build-vs-build statistic (``scripts/seed_strength_gate.py``);
the bias-robust metrics from ``lategame.eval.rating`` ride alongside it. Against the fixed
baselines used here they are a reparameterisation of the win rate and carry no extra
information -- they exist for M5 ladder play, where matchmaking pushes win rate toward 50%.
See plan.md 12, requirement R-EVAL.
"""

from __future__ import annotations

import os
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
from lategame.agents.doubles_ppo_agent import DoublesPPORecordingAgent
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
    # B6f: the doubles ROLLOUT agent -- `doubles` plus the on-policy signals (joint log-prob,
    # critic value, executed flag). Registered separately from `ppo` rather than as a mode of it,
    # for the same reason `doubles` is separate from `offrl`: a different encoder, a different
    # codec, and a fingerprint that makes loading the wrong checkpoint an error.
    "doubles_ppo": DoublesPPORecordingAgent,
}

# Agents backed by a trained checkpoint accept ``checkpoint_path``/``sample`` kwargs;
# the fixed baselines do not. Shared with ``data.collect`` so both build the same way.
_CHECKPOINT_AGENTS = {"bc", "offrl", "ppo", "search", "doubles", "doubles_ppo"}

# Learned-policy agents whose constructors accept a Build-14 ``loop_penalty`` (LoopGuard);
# ``search`` is checkpoint-backed but has no LoopGuard. ``LoopGuard(0.0)`` is exact identity,
# so forwarding the default 0.0 is a no-op for every existing caller.
#
# THE DOUBLES AGENTS ARE DELIBERATELY NOT IN THIS SET. They take a `loop_penalty` too, but they
# build a `DoublesLoopGuard` -- a `(2, 107)` per-slot penalty over the 1-6 switch range -- where
# the members below build the singles `LoopGuard`, a 26-wide vector penalizing indices 0-5. On
# the doubles layout index 0 is PASS, so a singles guard there penalizes passing and five of the
# six switches, which is not a weaker guard but a differently-wrong one. Kept as two sets so the
# mistake is a KeyError at build time rather than a plausible-looking penalty vector.
_LOOP_GUARD_AGENTS = {"bc", "offrl", "ppo"}
_DOUBLES_LOOP_GUARD_AGENTS = {"doubles", "doubles_ppo"}

# Agents accepting the B6f per-battle decision ceiling (`agents.turn_cap`). `None` is exact
# identity, so forwarding the default changes nothing for a caller that does not opt in.
_TURN_CAP_AGENTS = {"doubles", "doubles_ppo"}

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
# (docs/RESULTS.md, M5/G1). The agent then simply never answers the request and the SERVER plays a
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
    if os.environ.get("LATEGAME_SEARCH_SHAPED_ONLY") == "1":
        return _SINGLES_ONLY_AGENTS - {"search"}
    return _SINGLES_ONLY_AGENTS


_DOUBLES_SAFE_AGENTS = tuple(sorted(set(AGENTS) - _SINGLES_ONLY_AGENTS))


def policy_agent(battle_format: str) -> str:
    """Registry name of the GREEDY learned agent for ``battle_format``.

    Used for eval points and for frozen league opponents. The singles name is refused on a doubles
    format by ``build_player``, so the two PPO-side call sites (``train.ppo._eval_point`` and the
    league spec) have to ask rather than hardcode ``offrl``.

    Lives here, not on ``features.codec.FormatCodec``: that object's contract is "how to encode a
    POV, label an order, and mask legality" -- feature-layer and agent-free -- and it is what
    datasets and checkpoints fingerprint against. Registry names belong to the module that owns
    the registry.
    """
    return "doubles" if is_doubles_format(battle_format) else "offrl"


def rollout_agent(battle_format: str) -> str:
    """Registry name of the RECORDING learner for ``battle_format`` (on-policy rollouts)."""
    return "doubles_ppo" if is_doubles_format(battle_format) else "ppo"


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


# Team preview, applied to EVERY player on a doubles format rather than to our agents only.
#
# Set LATEGAME_TEAM_PREVIEW=0 to restore poke-env's `random_teampreview` on both sides, which is
# what every VGC number measured before this existed used. That is not a legacy switch: the
# preview-on/preview-off contrast is itself the measurement of how much of VGC skill lives in the
# bring-4 decision, and it needs both halves runnable from one build.
#
# Applied by SUBCLASSING rather than by editing the agents, because the three baselines a VGC
# ladder is measured against -- poke-env's RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
# -- are not ours to edit. Giving only our arm a real preview would let it win on bringing a better
# four and have that read as play strength.
_PREVIEW_CACHE: dict[type[Player], type[Player]] = {}


def team_preview_enabled() -> bool:
    return os.environ.get("LATEGAME_TEAM_PREVIEW", "1") != "0"


def _with_team_preview(cls: type[Player], battle_format: str) -> type[Player]:
    """``cls`` with ``TeamPreviewMixin`` ahead of it, on doubles formats with preview enabled.

    Singles formats are returned untouched: gen9ou has team preview too, but its selection is
    bring-6-play-6 (no subset to pick), and Random Battles have none at all. Cached so repeated
    builds of the same agent share one class -- poke-env checks `isinstance` in places, and a fresh
    subclass per player would make two builds of the same agent mutually non-identical.
    """
    if not (is_doubles_format(battle_format) and team_preview_enabled()):
        return cls
    # The registry is typed `dict[str, type[Player]]`, but tests substitute plain callables into it
    # to avoid constructing a real Player against a real server. Subclassing one of those raises a
    # metaclass conflict, so a non-class factory passes through unwrapped -- a double does not need
    # a preview policy, and this must not be the reason a test cannot substitute one.
    if not isinstance(cls, type):
        return cls
    if cls not in _PREVIEW_CACHE:
        from lategame.agents.teampreview import TeamPreviewMixin

        _PREVIEW_CACHE[cls] = type(f"Preview{cls.__name__}", (TeamPreviewMixin, cls), {})
    return _PREVIEW_CACHE[cls]


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
    max_battle_turns: int | None = None,
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
            f"Making the learned agents doubles-capable is the G4/M6 build (docs/RESULTS.md): a "
            f"per-slot action head and a two-active encoder."
        )
    cls = _with_team_preview(AGENTS[name], battle_format)
    extra: dict[str, object] = {}
    if name in _CHECKPOINT_AGENTS:
        extra["sample"] = sample
        if checkpoint_path is not None:
            extra["checkpoint_path"] = checkpoint_path
    # Build-14 decision-time anti-repetition. Only the learned policies carry a LoopGuard;
    # 0.0 is exact identity so this changes nothing unless a caller opts in. The doubles agents
    # take the same kwarg but build the per-slot guard (see _DOUBLES_LOOP_GUARD_AGENTS).
    if name in _LOOP_GUARD_AGENTS or name in _DOUBLES_LOOP_GUARD_AGENTS:
        extra["loop_penalty"] = loop_penalty
    # B6f per-battle decision ceiling; None is exact identity.
    if name in _TURN_CAP_AGENTS:
        extra["max_battle_turns"] = max_battle_turns
    # poke-env defaults to one battle at a time; PPO rollouts/eval pass a higher value
    # so cross_evaluate keeps many battles in flight (the local server is the bottleneck).
    if max_concurrent_battles is not None:
        extra["max_concurrent_battles"] = max_concurrent_battles
    # Teambuilt formats (e.g. gen9ou) need a team; Random Battles leave this None and the
    # server supplies one. Accepts a packed/Showdown string or a Teambuilder (R-TEAM pool).
    #
    # REFUSED HERE, because forgetting it is not an error anywhere else. Showdown answers a
    # teamless challenge on a teambuilt format with a popup -- "Your team was rejected ... This
    # format requires you to use your own team" -- which poke-env logs at WARNING and nothing
    # raises. The player then simply never battles, and the arm reads as a run that produced no
    # wins rather than as one that never started. Three separate call sites in
    # `scripts/curriculum_gate.py` had this defect simultaneously, each found only by watching a
    # cluster job's log; `build_player` is the one choke point where it can be found at all.
    if team is None and "randombattle" not in battle_format:
        raise ValueError(
            f"'{name}' on the teambuilt format {battle_format!r} needs a `team`; without one "
            f"Showdown rejects every challenge with a popup rather than an error, and the player "
            f"silently never battles. Pass a packed team or a TeamPool "
            f"(lategame/teambuilding/data/teams_gen9vgc.packed for VGC)."
        )
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
    Bradley-Terry fit downstream assumes. Booked in docs/RESULTS.md (M5/G1) at the time and deferred
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
    team_pool: str | None = None,
    p1_checkpoint: str | None = None,
    p2_checkpoint: str | None = None,
) -> EvalResult:
    """Score ``p1_name`` against ``p2_name``; the CLI's `evaluate` / `play` entry point.

    ``team_pool`` is required on a teambuilt format and refused by ``build_player`` when absent --
    the CLI defaults it per format rather than making every invocation pass one, since the pools
    are committed and there is exactly one sensible choice per format.

    Each side draws from its own distinctly-seeded pool, as the ceiling gate does, so the matchup
    varies rather than every battle being the same pairing.
    """
    def _team(seed: int) -> object | None:
        if not team_pool:
            return None
        from lategame.teambuilding.pool import TeamPool

        return TeamPool.from_packed_file(team_pool, seed=seed)

    p1 = build_player(
        p1_name, battle_format, checkpoint_path=p1_checkpoint, team=_team(0),  # type: ignore[arg-type]
    )
    p2 = build_player(
        p2_name, battle_format, checkpoint_path=p2_checkpoint, team=_team(1),  # type: ignore[arg-type]
    )
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

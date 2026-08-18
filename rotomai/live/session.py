"""Drive a live session: the three modes, plus a supervisor for a connection that cannot reconnect.

Two structural decisions are forced by poke-env 0.15 and are worth stating up front.

BATTLES ARE PLAYED IN CHUNKS, not with one ``ladder(n)`` call. ``ladder``/``accept_challenges``/
``send_challenges`` loop internally with no per-battle hook, so a single call is uninterruptible for
the whole session. Chunking gives a Ctrl-C boundary, per-battle telemetry checkpointing, per-battle
timeout granularity, and honest restart accounting. At the default concurrency of 1 it is
semantically identical to what poke-env would do anyway. Challenges arriving between calls are not
lost -- poke-env keeps filling its own challenge queue.

FAILURE IS DETECTED BY TIMEOUT, NOT BY EXCEPTION. poke-env dispatches each message in a detached
task whose result nobody retrieves, so ``ShowdownException`` (raised on ``|nametaken|``) never
reaches our ``await``: the coroutine simply hangs. And ``listen()`` swallows a closed socket, logs,
and returns -- there is no reconnect anywhere in the library. So the watchdog below samples forward
progress, and "recovery" means building a NEW player.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poke_env.player import Player

from rotomai.config import DEFAULT_FORMAT
from rotomai.eval.ladder import _TRAINING_ONLY_AGENTS, DEFAULT_LOOP_PENALTY
from rotomai.live.player import LoginError, LoginWatch, build_live_player, wait_for_login
from rotomai.live.policy import POLICY_NOTE, check_ladder_optin
from rotomai.live.server import live_account, live_server
from rotomai.live.telemetry import LiveLog, summarize, write_results

MODES = ("challenge", "accept", "ladder")

#: How often the watchdog samples forward progress.
_WATCHDOG_PERIOD = 5.0


@dataclass(frozen=True)
class LiveConfig:
    agent: str = "offrl"
    mode: str = "accept"
    battle_format: str = DEFAULT_FORMAT
    n: int = 1
    opponent: str | None = None
    checkpoint: str | None = None
    sample: bool = False
    #: The guarded policy, because that is the one every published gen9ou number describes
    #: (`eval-ladder`, `seed_strength_gate`, `results/*_v26*.json` are all at 4.0). At 0.0 a live
    #: session deploys a policy no measurement in this repo covers.
    loop_penalty: float = DEFAULT_LOOP_PENALTY
    team_pool: str | None = None
    team_seed: int = 0
    concurrency: int = 1
    start_timer: bool = True
    username: str | None = None
    avatar: str | None = None
    ws_url: str | None = None
    auth_url: str | None = None
    allow_guest: bool = False
    save_replays: str | None = None
    out: str = "results/live_session.json"
    #: Write to a fresh ``session_<NNN>.json`` under this directory instead of rewriting ``out``.
    #: ``_flush`` rebuilds ``out`` from an empty log on start, so a long-lived service restarting
    #: onto the same path erases its own history -- the failure `scripts/merge_live_sessions.py`
    #: exists to work around. With a directory, each process claims the next free index and the
    #: same merge script pools them.
    out_dir: str | None = None
    battle_timeout: float = 900.0
    #: Deadline for a battle that has stopped advancing turns.
    stall_timeout: float = 300.0
    #: Deadline while there is NO battle in progress -- queued for `/search`, or serving
    #: `battle_delay`. Separate from `stall_timeout` because the watchdog's two samples cannot
    #: move between battles, so one deadline would have to be set by the slower of the two
    #: situations and would then be useless for the other. See `_watchdog`.
    queue_timeout: float = 900.0
    login_timeout: float = 30.0
    max_restarts: int = 3
    #: Seconds before the first reconnect attempt; doubles per attempt, capped at 60. Configurable
    #: so a caller that already knows the server is up (or a test) need not sit through the ramp.
    backoff_base: float = 5.0
    #: Seconds to wait between battle chunks. 0.0 is exact identity, so nothing existing changes.
    #: poke-env's `ladder(k)` re-searches the instant a game ends, and back-to-back searches with no
    #: gap are the traffic pattern a ranked ladder reads as farming. Recorded in the results file
    #: rather than only in the command, so the etiquette claim travels with the number.
    battle_delay: float = 0.0
    ladder_ack: str | None = None
    log_level: int | None = None
    #: Rate opponents at their OBSERVED Showdown rating rather than pinning the field at the
    #: reference. Off by default, and a no-op outside rated ladder games -- but without it the
    #: session's own GXE is a reparameterisation of its score rate no matter what it played, which
    #: is precisely the number G2 cannot be settled with. See ``telemetry.summarize``.
    use_live_ratings: bool = False


@dataclass
class SessionOutcome:
    summary: dict[str, Any] = field(default_factory=dict)
    records: list[Any] = field(default_factory=list)
    restarts: int = 0
    stopped_early: bool = False
    #: Where the record was actually written -- with ``out_dir`` the caller does not choose it.
    out_path: str = ""


def _accept_list(opponent: str | None) -> str | list[str] | None:
    """``None`` accepts anyone; a comma list becomes the allowlist poke-env expects."""
    if opponent is None:
        return None
    names = [part.strip() for part in opponent.split(",") if part.strip()]
    return names or None


def is_unbounded(cfg: LiveConfig) -> bool:
    """``--n 0`` means "play until stopped" -- the standing-service mode."""
    return cfg.n <= 0


def remaining_battles(cfg: LiveConfig, finished: int) -> int:
    """Battles still owed. An unbounded session always owes another full chunk."""
    if is_unbounded(cfg):
        return max(1, cfg.concurrency)
    return cfg.n - finished


def next_segment_path(out_dir: str | Path) -> Path:
    """The next free ``session_<NNN>.json`` in ``out_dir``.

    Claimed once per process, before the first battle. Scanning for the highest existing index
    rather than using a timestamp keeps the merge order the play order, and keeps the filenames
    stable enough to name in a pre-registration.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    used = set()
    for existing in directory.glob("session_*.json"):
        stem = existing.stem.removeprefix("session_")
        if stem.isdigit():
            used.add(int(stem))
    index = 0
    while index in used:
        index += 1
    return directory / f"session_{index:03d}.json"


def _public_config(cfg: LiveConfig, username: str, ws_url: str) -> dict[str, Any]:
    """The credential-free subset written into the results file."""
    public: dict[str, Any] = {
        "agent": cfg.agent,
        "checkpoint": cfg.checkpoint,
        # The POLICY, not just the weights. `sample` and `loop_penalty` each select a measurably
        # different policy from the same checkpoint (0.675 vs 0.767 on v25b; every published gen9ou
        # number is at loop_penalty 4.0), and `team_pool`/`team_seed` decide what it played with.
        # Without them a session file cannot be checked against a pre-registration -- which is the
        # whole point of pre-registering a live run.
        "sample": cfg.sample,
        "loop_penalty": cfg.loop_penalty,
        "team_pool": cfg.team_pool,
        "team_seed": cfg.team_seed,
        "mode": cfg.mode,
        "battle_format": cfg.battle_format,
        "username": username,
        "server": ws_url,
        "concurrency": cfg.concurrency,
        "battle_delay": cfg.battle_delay,
        "requested_battles": cfg.n,
        # A standing service has no target n, so `requested_battles` alone cannot distinguish
        # "played 0 of 100" from "played 40 and was told to stop". Both readers of this file --
        # `merge_live_sessions` and a human -- need that distinction.
        "unbounded": is_unbounded(cfg),
        # Recorded for the same reason `ladder_ack` is: a published rating has to carry the basis
        # it was computed on, and "were opponents rated or pinned" is that basis.
        "use_live_ratings": cfg.use_live_ratings,
    }
    if cfg.mode == "ladder":
        public["policy"] = {"ranked": True, "ack": cfg.ladder_ack, "note": POLICY_NOTE}
    else:
        public["policy"] = {"ranked": False, "mode": cfg.mode}
    return public


def _default_factory(cfg: LiveConfig) -> Player:
    """Build the live player from the config (the real, non-test path)."""
    server = live_server(cfg.ws_url, cfg.auth_url)
    account = live_account(cfg.username, server=server, allow_guest=cfg.allow_guest)

    team: Any = None
    if cfg.team_pool:
        from rotomai.teambuilding.pool import TeamPool

        team = TeamPool.from_packed_file(cfg.team_pool, seed=cfg.team_seed)

    return build_live_player(
        cfg.agent,
        cfg.battle_format,
        account,
        server,
        checkpoint_path=cfg.checkpoint,
        sample=cfg.sample,
        loop_penalty=cfg.loop_penalty,
        team=team,
        max_concurrent_battles=cfg.concurrency,
        start_timer_on_battle_start=cfg.start_timer,
        save_replays=cfg.save_replays or False,
        avatar=cfg.avatar,
        log_level=cfg.log_level,
    )


async def _play_chunk(player: Player, cfg: LiveConfig, k: int) -> None:
    """One chunk of ``k`` battles in the configured mode."""
    if cfg.mode == "challenge":
        assert cfg.opponent is not None  # validated in run_session before any player is built
        await player.send_challenges(cfg.opponent, k)
    elif cfg.mode == "accept":
        await player.accept_challenges(_accept_list(cfg.opponent), k)
    elif cfg.mode == "ladder":
        await player.ladder(k)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown mode {cfg.mode!r}; choose from {', '.join(MODES)}")


async def _watchdog(player: Player, log: LiveLog, cfg: LiveConfig, tripped: asyncio.Event) -> None:
    """Trip ``tripped`` when the connection has stopped making forward progress.

    Samples ``(n_finished_battles, sum of turn counters)``. A live battle advances turns even when
    it is slow; a dead socket advances neither. That distinction is what a flat wall-clock timeout
    cannot make, and it is why a long legitimate game does not get killed.
    """
    last = None
    last_change = time.monotonic()
    while not tripped.is_set():
        await asyncio.sleep(_WATCHDOG_PERIOD)
        try:
            battles = dict(getattr(player, "battles", {}))
            sample = (
                int(getattr(player, "n_finished_battles", 0)),
                sum(int(getattr(b, "turn", 0) or 0) for b in battles.values()),
            )
            # BETWEEN battles both samples are frozen, so a healthy session looks exactly like a
            # dead one. On the ranked ladder that is not an edge case: `/search` can sit in the
            # queue for minutes, and `battle_delay` deliberately adds to the gap. Under a single
            # deadline a slow queue reads as a wedge, burns a restart, and reconnects-and-researches
            # -- the traffic pattern the pacing exists to avoid. So which deadline applies depends
            # on whether there is a battle to make progress in.
            waiting = not any(not getattr(b, "finished", False) for b in battles.values())
        except Exception:  # never let telemetry kill the session
            continue
        limit = cfg.queue_timeout if waiting else cfg.stall_timeout
        now = time.monotonic()
        if sample != last:
            last, last_change = sample, now
        elif now - last_change > limit:
            tripped.set()
            return


async def run_session(
    cfg: LiveConfig,
    player_factory: Callable[[LiveConfig], Player] | None = None,
    stop: asyncio.Event | None = None,
) -> SessionOutcome:
    """Play ``cfg.n`` battles live, restarting the client if the connection dies.

    ``player_factory`` is the seam the tests inject a fake player through; production passes None.
    """
    if cfg.mode not in MODES:
        raise ValueError(f"unknown mode {cfg.mode!r}; choose from {', '.join(MODES)}")
    if cfg.mode == "challenge" and not cfg.opponent:
        raise ValueError("--mode challenge requires --opponent (the username to challenge)")
    if cfg.checkpoint and cfg.agent in _TRAINING_ONLY_AGENTS:
        # The same refusal `eval.ladder.parse_field` makes, for the same reason, at the only other
        # place a checkpoint gets a policy attached to it. Refused rather than silently rewritten:
        # the results file names the agent it was told, and a swap would make the artifact disagree
        # with the command that produced it.
        instead = _TRAINING_ONLY_AGENTS[cfg.agent]
        raise ValueError(
            f"--agent {cfg.agent!r} with --checkpoint: {cfg.agent!r} is the TRAINING rollout agent "
            f"and forces sample=True, so it deploys a different -- and measurably weaker -- policy "
            f"than the same checkpoint read greedily (~9 points vs the heuristic on v25b). Every "
            f"published number in this project scores these checkpoints through {instead!r}. "
            f"Use --agent {instead}, and add --sample if you deliberately want the sampled policy."
        )
    if is_unbounded(cfg) and cfg.mode == "ladder":
        # An unbounded RATED run cannot be pre-registered: the stopping rule IS the
        # pre-registration (`results/live_ladder_gen9ou_prereg.json` freezes n=100, played once, no
        # top-up), and "play until I get bored" is the protocol that turns a measurement into a
        # selected sample.
        raise ValueError(
            "--n 0 (play until stopped) is not available for --mode ladder: a ranked run must "
            "declare its n before the first game. Use a fixed --n, or --mode accept."
        )
    if cfg.mode == "ladder":
        # Raises PolicyError unless BOTH opt-in channels are present.
        check_ladder_optin(cfg.ladder_ack)
        print(POLICY_NOTE, file=sys.stderr, flush=True)

    factory = player_factory or _default_factory
    log = LiveLog()
    outcome = SessionOutcome()
    attempt = 0
    out_path = str(next_segment_path(cfg.out_dir)) if cfg.out_dir else cfg.out

    while True:
        remaining = remaining_battles(cfg, sum(1 for r in log.records if r.finished))
        if remaining <= 0 or (stop is not None and stop.is_set()):
            break
        if attempt > cfg.max_restarts:
            break

        player = factory(cfg)
        watch = LoginWatch()
        player.logger.addHandler(watch)
        tripped = asyncio.Event()
        guard: asyncio.Task | None = None

        try:
            # A login failure is FATAL and never retried: hammering a bad password is how a live
            # account gets rate-limited or locked.
            await wait_for_login(player, cfg.login_timeout, watch)

            guard = asyncio.create_task(_watchdog(player, log, cfg, tripped))
            while remaining > 0 and not (stop is not None and stop.is_set()):
                k = min(cfg.concurrency, remaining)
                chunk = asyncio.create_task(_play_chunk(player, cfg, k))
                stall = asyncio.create_task(tripped.wait())
                done, _ = await asyncio.wait(
                    {chunk, stall},
                    timeout=cfg.battle_timeout * k,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if chunk not in done:
                    chunk.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await chunk
                    stall.cancel()
                    raise ConnectionError(
                        "no forward progress: the socket is dead or the client is wedged. "
                        "poke-env neither raises nor reconnects, so this is detected by timeout."
                    )
                stall.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stall
                chunk.result()  # surface a genuine error from the mode call

                log.harvest(player, cfg.mode, attempt, cfg.battle_format)
                _flush(cfg, log, player, outcome, out_path)
                remaining = remaining_battles(cfg, sum(1 for r in log.records if r.finished))

                # Pace the NEXT search. After the flush, so a Ctrl-C during the pause loses nothing,
                # and skipped on the last chunk, where it would only delay the summary.
                if cfg.battle_delay > 0 and remaining > 0:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            stop.wait() if stop is not None else asyncio.Event().wait(),
                            timeout=cfg.battle_delay,
                        )

            log.finalize(player, cfg.mode, attempt, cfg.battle_format)
            _flush(cfg, log, player, outcome, out_path)

        except LoginError:
            await _shutdown(player, watch, guard, tripped)
            raise
        except (ConnectionError, TimeoutError, OSError):
            log.harvest(player, cfg.mode, attempt, cfg.battle_format)
            outcome.restarts += 1
            attempt += 1
            await _shutdown(player, watch, guard, tripped)
            _flush(cfg, log, None, outcome, out_path)
            if attempt > cfg.max_restarts:
                break
            if cfg.backoff_base > 0:
                await asyncio.sleep(min(60.0, cfg.backoff_base * (2 ** (attempt - 1))))
            continue
        else:
            await _shutdown(player, watch, guard, tripped)
            break
        finally:
            with contextlib.suppress(Exception):
                player.logger.removeHandler(watch)

    # An unbounded session has no target to fall short of: being stopped IS how it finishes, and
    # flagging that as `stopped_early` would put a "this run was cut short" caveat on every service
    # shutdown.
    outcome.stopped_early = (not is_unbounded(cfg)) and (
        bool(stop is not None and stop.is_set())
        or sum(1 for r in log.records if r.finished) < cfg.n
    )
    outcome.records = log.records
    outcome.summary = _summarize(cfg, log)
    outcome.out_path = out_path
    _flush(cfg, log, None, outcome, out_path)
    return outcome


def _summarize(cfg: LiveConfig, log: LiveLog) -> dict[str, Any]:
    """The ONE place the summary options are applied.

    The incremental per-battle writes and the final summary must not be able to disagree: a run
    that flushed a pinned-field GXE 50 times and then reported a live-rated one at the end would
    look like a corrupted file rather than a wiring bug. Both paths come through here.
    """
    return summarize(
        log.records, use_live_ratings=cfg.use_live_ratings
    )


def _flush(
    cfg: LiveConfig,
    log: LiveLog,
    player: Player | None,
    outcome: SessionOutcome,
    out_path: str | None = None,
) -> None:
    """Rewrite the results file. Called after EVERY battle, so a SIGKILL loses at most one game."""
    username = getattr(player, "username", None) or (cfg.username or "")
    public = _public_config(cfg, username, cfg.ws_url or "wss://sim3.psim.us/showdown/websocket")
    public["restarts"] = outcome.restarts
    public["stopped_early"] = outcome.stopped_early
    with contextlib.suppress(OSError):
        write_results(out_path or cfg.out, public, _summarize(cfg, log), log.records)


async def _shutdown(
    player: Player, watch: LoginWatch, guard: asyncio.Task | None, tripped: asyncio.Event
) -> None:
    """Tear the client down without letting cleanup raise over the real error."""
    tripped.set()
    if guard is not None:
        guard.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await guard
    with contextlib.suppress(Exception):
        await player.ps_client.stop_listening()


def chunk_count(n: int, concurrency: int) -> int:
    """How many chunks ``n`` battles take. Exposed for the CLI's progress line."""
    return math.ceil(n / max(1, concurrency))

"""Construct an agent wired to a LIVE server, and diagnose the logins poke-env swallows.

``eval.arena.build_player`` hardcodes ``local_account(_unique_username(...))`` and ``LOCAL_SERVER``,
so it cannot express live play at all. This module mirrors its agent-selection logic exactly while
supplying the connection kwargs itself -- the same shape ``scripts/behavior_probe.py`` already uses
to work around that builder's fixed argument list.

The registry and the two agent-capability sets are IMPORTED from ``arena`` rather than copied. That
is a deliberate cross-module reach at a private name: copying them would rot silently the first time
an agent is registered, and ``tests/test_live_player.py`` carries a parity test that fails loudly if
the two builders ever disagree.
"""

from __future__ import annotations

import asyncio
import logging
import time

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder

from lategame.eval.arena import _CHECKPOINT_AGENTS, _LOOP_GUARD_AGENTS, AGENTS
from lategame.live.server import LIVE_SERVER


class LoginError(RuntimeError):
    """Raised when the account never logged in, with whatever diagnosis we could recover."""


def build_live_player(
    name: str,
    battle_format: str,
    account: AccountConfiguration,
    server: ServerConfiguration = LIVE_SERVER,
    *,
    checkpoint_path: str | None = None,
    sample: bool = False,
    loop_penalty: float = 0.0,
    team: str | Teambuilder | None = None,
    max_concurrent_battles: int = 1,
    start_timer_on_battle_start: bool = True,
    save_replays: bool | str = False,
    avatar: str | None = None,
    log_level: int | None = None,
) -> Player:
    """Build ``name`` against a live server under ``account``.

    ``account`` is a REQUIRED POSITIONAL on purpose. ``arena._unique_username`` appends
    ``secrets.token_hex(3)``, which on a live server is an unregistered name: password auth fails,
    poke-env silently degrades to a guest, and the session then hangs with no error. Making the
    caller supply the account means that line cannot be copied here by accident.
    """
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Choose from: {', '.join(AGENTS)}")
    if max_concurrent_battles < 1:
        # poke-env reads 0 as UNLIMITED, so a plausible "0 == default" misreading would uncap a
        # live account -- many simultaneous searches is the visible signature of farming.
        raise ValueError(
            f"max_concurrent_battles must be >= 1, got {max_concurrent_battles} "
            "(poke-env treats 0 as unlimited)"
        )

    cls = AGENTS[name]
    extra = live_agent_extras(
        name, checkpoint_path=checkpoint_path, sample=sample, loop_penalty=loop_penalty
    )
    if team is not None:
        extra["team"] = team
    if avatar is not None:
        extra["avatar"] = avatar
    if log_level is not None:
        extra["log_level"] = log_level
    if save_replays:
        extra["save_replays"] = save_replays

    return cls(
        account_configuration=account,
        battle_format=battle_format,
        server_configuration=server,
        max_concurrent_battles=max_concurrent_battles,
        # Nothing else in poke-env manages the live turn clock; without this the agent can be
        # timed out by an opponent who starts the timer.
        start_timer_on_battle_start=start_timer_on_battle_start,
        **extra,  # type: ignore[arg-type]
    )


def live_agent_extras(
    name: str,
    *,
    checkpoint_path: str | None = None,
    sample: bool = False,
    loop_penalty: float = 0.0,
) -> dict[str, object]:
    """The capability-gated kwargs, selected exactly as ``arena.build_player`` selects them.

    Factored out so the parity test can compare the two builders directly rather than by
    constructing players.
    """
    extra: dict[str, object] = {}
    if name in _CHECKPOINT_AGENTS:
        extra["sample"] = sample
        if checkpoint_path is not None:
            extra["checkpoint_path"] = checkpoint_path
    if name in _LOOP_GUARD_AGENTS:
        extra["loop_penalty"] = loop_penalty
    return extra


class LoginWatch(logging.Handler):
    """Turns poke-env's SWALLOWED login failures into something inspectable.

    poke-env dispatches each message in a detached ``asyncio.create_task`` and never retrieves the
    result, so the ``ShowdownException`` raised on ``|nametaken|`` NEVER reaches the caller's
    ``await`` -- the coroutine just hangs. The log is the only channel that carries the reason, so
    we listen on it. Never stores or emits the password.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.reason: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string must not break the session
            return
        lowered = message.lower()
        if "nametaken" in lowered:
            self.reason = (
                "the server rejected the username (|nametaken|): it is registered to someone "
                "else, or this account is already connected. poke-env raises ShowdownException "
                "inside a detached task, so it never surfaces -- read from the log instead."
            )
        elif "bypassing authentication" in lowered:
            self.reason = (
                "connected as a GUEST because the password was empty, so |updateuser| will never "
                "match the requested name and login can never complete."
            )
        elif "showdown returned" in lowered:
            self.reason = f"login rejected by the server: {message}"


async def wait_for_login(player: Player, timeout: float, watch: LoginWatch) -> None:
    """Block until the client is logged in, or raise ``LoginError``.

    POLLS rather than awaits: ``logged_in`` is an ``asyncio.Event`` bound to poke-env's own
    background loop (POKE_LOOP), and awaiting a foreign loop's Event from here would never wake.
    poke-env's own ``wait_for_login`` polls for exactly this reason.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if player.ps_client.logged_in.is_set():
            return
        if watch.reason is not None:
            raise LoginError(watch.reason)
        await asyncio.sleep(0.1)
    raise LoginError(
        watch.reason
        or (
            f"no |updateuser| match for {player.username!r} within {timeout:.0f}s: wrong password, "
            "or the server is unreachable. poke-env does not raise on a failed login."
        )
    )

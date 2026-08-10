"""The live builder must agree with arena's, and must never invent its own account.

Two regression guards matter most here:
  * the supplied AccountConfiguration is passed through UNTOUCHED -- copying arena's
    `_unique_username()` would produce an unregistered name on a live server, which poke-env
    silently downgrades to a guest login that then hangs forever;
  * the capability-gated kwargs stay identical to `arena.build_player`'s, since this module
    imports arena's two private sets and would otherwise rot the next time an agent is registered.
"""

import logging
from types import SimpleNamespace

import pytest
from poke_env import AccountConfiguration, ServerConfiguration

import lategame.eval.arena as arena
from lategame.live.player import (
    LoginError,
    LoginWatch,
    build_live_player,
    live_agent_extras,
    wait_for_login,
)
from lategame.live.server import LIVE_SERVER

_ACCOUNT = AccountConfiguration("lategamebot", "hunter2-do-not-leak")
_PRIVATE = ServerConfiguration("ws://localhost:8000/showdown/websocket", "https://x/action.php?")


@pytest.fixture
def captured(monkeypatch):
    """Swap every agent for a kwarg recorder -- the tests/test_arena.py pattern."""
    seen: dict[str, dict] = {}

    def _fake(tag):
        class _Fake:
            def __init__(self, **kwargs):
                seen[tag] = kwargs

        return _Fake

    for name in list(arena.AGENTS):
        monkeypatch.setitem(arena.AGENTS, name, _fake(name))
    return seen


def test_the_supplied_account_is_passed_through_untouched(captured):
    """THE guard against copying arena's random-suffix username onto a live server."""
    build_live_player("ppo", "gen9ou", _ACCOUNT)
    assert captured["ppo"]["account_configuration"] is _ACCOUNT
    assert captured["ppo"]["account_configuration"].username == "lategamebot"


def test_defaults_to_the_live_server_and_starts_the_turn_timer(captured):
    build_live_player("ppo", "gen9ou", _ACCOUNT)
    assert captured["ppo"]["server_configuration"] is LIVE_SERVER
    assert captured["ppo"]["start_timer_on_battle_start"] is True


def test_private_server_and_timer_are_overridable(captured):
    build_live_player("ppo", "gen9ou", _ACCOUNT, _PRIVATE, start_timer_on_battle_start=False)
    assert captured["ppo"]["server_configuration"] is _PRIVATE
    assert captured["ppo"]["start_timer_on_battle_start"] is False


def test_unknown_agent_is_rejected():
    with pytest.raises(ValueError, match="Unknown agent"):
        build_live_player("not-an-agent", "gen9ou", _ACCOUNT)


@pytest.mark.parametrize("bad", [0, -1])
def test_zero_concurrency_is_rejected_because_poke_env_reads_it_as_unlimited(bad):
    with pytest.raises(ValueError, match="unlimited"):
        build_live_player("ppo", "gen9ou", _ACCOUNT, max_concurrent_battles=bad)


def test_optional_kwargs_are_forwarded_only_when_set(captured):
    build_live_player("ppo", "gen9ou", _ACCOUNT)
    bare = captured["ppo"]
    for key in ("team", "avatar", "log_level", "save_replays"):
        assert key not in bare

    build_live_player(
        "ppo", "gen9ou", _ACCOUNT, team="packed", avatar="red",
        log_level=logging.DEBUG, save_replays="/tmp/replays",
    )
    rich = captured["ppo"]
    assert rich["team"] == "packed" and rich["avatar"] == "red"
    assert rich["log_level"] == logging.DEBUG and rich["save_replays"] == "/tmp/replays"


def test_checkpoint_and_loop_penalty_reach_only_the_agents_that_accept_them(captured):
    build_live_player("ppo", "gen9ou", _ACCOUNT, checkpoint_path="x.pt", sample=True,
                      loop_penalty=4.0)
    build_live_player("random", "gen9ou", _ACCOUNT, checkpoint_path="x.pt", loop_penalty=4.0)
    build_live_player("search", "gen9ou", _ACCOUNT, checkpoint_path="x.pt", loop_penalty=4.0)

    assert captured["ppo"]["checkpoint_path"] == "x.pt" and captured["ppo"]["sample"] is True
    assert captured["ppo"]["loop_penalty"] == 4.0
    # RandomPlayer accepts neither.
    assert "checkpoint_path" not in captured["random"] and "sample" not in captured["random"]
    assert "loop_penalty" not in captured["random"]
    # search is checkpoint-backed but has no LoopGuard.
    assert captured["search"]["checkpoint_path"] == "x.pt"
    assert "loop_penalty" not in captured["search"]


@pytest.mark.parametrize("name", sorted(arena.AGENTS))
def test_parity_with_arena_for_every_registered_agent(name):
    """Anti-rot guard for importing arena's two private capability sets.

    Mirrors arena.build_player's own selection logic; if a future agent is added to one set and
    not reflected here, this fails instead of silently dropping a kwarg on live play.
    """
    kwargs = {"checkpoint_path": "x.pt", "sample": True, "loop_penalty": 4.0}
    got = live_agent_extras(name, **kwargs)

    expected: dict[str, object] = {}
    if name in arena._CHECKPOINT_AGENTS:
        expected["sample"] = True
        expected["checkpoint_path"] = "x.pt"
    if name in arena._LOOP_GUARD_AGENTS:
        expected["loop_penalty"] = 4.0
    assert got == expected


# --------------------------------------------------------------------------- #
# LoginWatch -- the only channel that carries a swallowed login failure.
# --------------------------------------------------------------------------- #


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("poke-env", logging.WARNING, __file__, 1, message, None, None)


def test_login_watch_diagnoses_a_taken_name():
    watch = LoginWatch()
    watch.emit(_record("Error message received: |nametaken|lategamebot|already taken"))
    assert watch.reason and "registered to someone else" in watch.reason


def test_login_watch_diagnoses_the_silent_guest_downgrade():
    watch = LoginWatch()
    watch.emit(_record("Bypassing authentication request"))
    assert watch.reason and "GUEST" in watch.reason


def test_login_watch_diagnoses_a_rejected_login():
    watch = LoginWatch()
    watch.emit(_record("Trying to login as lategamebot, showdown returned Guest 42"))
    assert watch.reason and "rejected by the server" in watch.reason


def test_login_watch_ignores_ordinary_chatter_and_broken_records():
    watch = LoginWatch()
    watch.emit(_record("battle started"))
    assert watch.reason is None
    broken = logging.LogRecord("poke-env", logging.INFO, __file__, 1, "%d", ("not-an-int",), None)
    watch.emit(broken)  # must not raise
    assert watch.reason is None


def _fake_player(logged_in: bool):
    """Only the surface wait_for_login touches. `logged_in` is POLLED, never awaited."""
    return SimpleNamespace(
        ps_client=SimpleNamespace(logged_in=SimpleNamespace(is_set=lambda: logged_in)),
        username="bot",
    )


async def test_wait_for_login_returns_once_logged_in():
    await wait_for_login(_fake_player(True), 1.0, LoginWatch())


async def test_wait_for_login_raises_the_diagnosis_rather_than_hanging():
    """A taken name never reaches an await in poke-env; the log handler is the only signal."""
    watch = LoginWatch()
    watch.emit(_record("|nametaken|"))
    with pytest.raises(LoginError, match="registered to someone else"):
        await wait_for_login(_fake_player(False), 5.0, watch)


async def test_wait_for_login_times_out_with_a_useful_message():
    with pytest.raises(LoginError, match="does not raise on a failed login"):
        await wait_for_login(_fake_player(False), 0.2, LoginWatch())

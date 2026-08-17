"""Live server/account config: the failure modes here are SILENT, so they get pinned.

An empty password does not raise in poke-env -- it connects as a Guest and hangs. A too-long
username is truncated and then never matches the login. Both are diagnosed up front, and both
error paths must keep the secret out of the message.
"""

import pytest
from poke_env import LocalhostServerConfiguration, ServerConfiguration, ShowdownServerConfiguration

from rotomai.config import LOCAL_SERVER
from rotomai.live.policy import PASSWORD_ENV, USERNAME_ENV
from rotomai.live.server import (
    LIVE_SERVER,
    CredentialsError,
    is_public,
    live_account,
    live_server,
)

_SECRET = "hunter2-do-not-leak"
_LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket", "https://x/action.php?")


def test_live_server_is_the_upstream_constant_not_a_copy():
    """An alias, so an upstream host change follows automatically."""
    assert LIVE_SERVER is ShowdownServerConfiguration
    assert "psim.us" in LIVE_SERVER.websocket_url


def test_the_training_path_is_untouched():
    """tests/test_config.py pins this too; asserted from the live side as well, because the whole
    point of new symbols was to leave the frozen local config alone."""
    assert LOCAL_SERVER == LocalhostServerConfiguration
    assert LIVE_SERVER != LOCAL_SERVER


def test_live_server_override_builds_a_private_eval_target():
    got = live_server("ws://127.0.0.1:9000/showdown/websocket")
    assert got.websocket_url == "ws://127.0.0.1:9000/showdown/websocket"
    assert got.authentication_url == LIVE_SERVER.authentication_url  # inherited by default
    assert live_server("ws://h/x", "https://auth/").authentication_url == "https://auth/"
    assert live_server() is LIVE_SERVER


@pytest.mark.parametrize(
    "url,public",
    [
        ("wss://sim3.psim.us/showdown/websocket", True),
        ("wss://sim.smogon.com/showdown/websocket", True),
        ("ws://localhost:8000/showdown/websocket", False),
        ("ws://127.0.0.1:8100/showdown/websocket", False),
        ("ws://tron62:8200/showdown/websocket", False),
    ],
)
def test_is_public_classification(url, public):
    assert is_public(ServerConfiguration(url, "https://x")) is public


def test_reads_credentials_from_the_environment():
    env = {USERNAME_ENV: "rotomaibot", PASSWORD_ENV: _SECRET}
    account = live_account(server=LIVE_SERVER, env=env)
    assert account.username == "rotomaibot"
    assert account.password == _SECRET


def test_explicit_username_beats_the_environment():
    env = {USERNAME_ENV: "fromenv", PASSWORD_ENV: _SECRET}
    assert live_account("explicit", server=LIVE_SERVER, env=env).username == "explicit"


def test_empty_password_against_the_public_server_is_a_hard_error():
    """poke-env would bypass auth, become a Guest, and hang with no exception."""
    with pytest.raises(CredentialsError, match="hang"):
        live_account(server=LIVE_SERVER, env={USERNAME_ENV: "bot"})


def test_guest_is_allowed_only_off_the_public_server_and_only_when_asked():
    env = {USERNAME_ENV: "bot"}
    with pytest.raises(CredentialsError, match="allow-guest"):
        live_account(server=_LOCAL, env=env)
    assert live_account(server=_LOCAL, allow_guest=True, env=env).password is None
    # --allow-guest must NOT unlock the public server.
    with pytest.raises(CredentialsError, match="PUBLIC"):
        live_account(server=LIVE_SERVER, allow_guest=True, env=env)


def test_missing_username_names_the_env_var_and_the_bot_account_rule():
    with pytest.raises(CredentialsError) as exc:
        live_account(env={PASSWORD_ENV: _SECRET})
    assert USERNAME_ENV in str(exc.value)
    assert "BOT ACCOUNT" in str(exc.value)


def test_username_length_and_delimiters_are_rejected_before_login():
    env = {PASSWORD_ENV: _SECRET}
    with pytest.raises(CredentialsError, match="18"):
        live_account("x" * 19, env=env)
    for bad in ("bot|name", "bot,name"):
        with pytest.raises(CredentialsError, match="delimiters"):
            live_account(bad, env=env)


@pytest.mark.parametrize(
    "username,env,server,guest",
    [
        (None, {PASSWORD_ENV: _SECRET}, LIVE_SERVER, False),          # missing username
        ("x" * 19, {PASSWORD_ENV: _SECRET}, LIVE_SERVER, False),      # too long
        ("bot|x", {PASSWORD_ENV: _SECRET}, LIVE_SERVER, False),       # delimiter
        ("bot", {}, LIVE_SERVER, False),                              # empty password, public
        ("bot", {}, _LOCAL, False),                                   # empty password, no guest
    ],
)
def test_no_error_path_ever_echoes_the_password(username, env, server, guest):
    """The message names the ENV VAR to set, never its value."""
    with pytest.raises(CredentialsError) as exc:
        live_account(username, server=server, allow_guest=guest, env=env)
    assert _SECRET not in str(exc.value)

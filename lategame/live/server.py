"""Server and account configuration for LIVE play (plan.md section 7.1 / M5).

Everything here is NEW rather than an extension of ``lategame/config.py``. Two reasons: that module
is on the frozen path of in-flight cluster jobs, and ``tests/test_config.py`` pins
``LOCAL_SERVER == LocalhostServerConfiguration`` including the auth URL -- a deliberate guard that
the training path can never be pointed at a public server by editing a shared constant. Live play
gets its own symbols so that guard keeps holding.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse

from poke_env import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration

from lategame.live.policy import PASSWORD_ENV, USERNAME_ENV

#: The public Showdown sim. An ALIAS of poke-env's constant, never a re-built tuple: if upstream
#: moves the websocket host, we follow automatically instead of silently connecting nowhere.
LIVE_SERVER: ServerConfiguration = ShowdownServerConfiguration

#: Hosts treated as "the public ladder" for policy purposes.
_PUBLIC_HOSTS = frozenset({"sim.smogon.com", "sim3.psim.us", "psim.us", "play.pokemonshowdown.com"})

#: poke-env truncates usernames to 18 characters (AccountConfiguration.generate); Showdown itself
#: rejects longer ones, so catch it here rather than after a failed login.
_MAX_USERNAME = 18


class CredentialsError(RuntimeError):
    """Raised when the account configuration cannot produce a real login.

    Never carries the password value -- only the name of the environment variable to set.
    """


def live_server(ws_url: str | None = None, auth_url: str | None = None) -> ServerConfiguration:
    """The public sim by default; a private eval server when ``ws_url`` is given.

    plan.md section 15 prefers "a private eval server" over public play, so this override is the
    supported path for that -- and it is also what lets the whole live stack be exercised against
    the local ``--no-security`` server in tests, with no account and no policy exposure.
    """
    if ws_url is None:
        return LIVE_SERVER
    return ServerConfiguration(ws_url, auth_url or LIVE_SERVER.authentication_url)


def is_public(server: ServerConfiguration) -> bool:
    """True when this configuration points at the public ladder."""
    host = (urlparse(server.websocket_url).hostname or "").lower()
    return host in _PUBLIC_HOSTS


def live_account(
    username: str | None = None,
    server: ServerConfiguration = LIVE_SERVER,
    allow_guest: bool = False,
    env: Mapping[str, str] | None = None,
) -> AccountConfiguration:
    """Build the bot account, refusing the configurations that fail silently.

    The password comes from the environment and nowhere else, so it cannot appear in ``ps`` output,
    shell history, or a results file.

    THE EMPTY-PASSWORD CASE IS A HARD ERROR, not a warning. poke-env treats a falsy password as
    "bypass authentication": it sends ``/trn <name>,0,`` and the server answers ``|updateuser| Guest
    N``, which matches neither the requested name nor a failure branch -- so ``logged_in`` never
    sets, nothing is raised, and the session HANGS. Diagnosing that from the outside is miserable,
    so it is rejected up front.
    """
    if env is None:
        env = os.environ

    name = (username or env.get(USERNAME_ENV, "")).strip()
    if not name:
        raise CredentialsError(
            f"no live username: pass --username or set ${USERNAME_ENV}. "
            "plan.md section 15 requires a DEDICATED BOT ACCOUNT, never a primary account."
        )
    if len(name) > _MAX_USERNAME:
        raise CredentialsError(
            f"username {name!r} is {len(name)} characters; Showdown allows {_MAX_USERNAME}. "
            "poke-env would truncate it and the login would not match."
        )
    if any(c in name for c in "|,"):
        raise CredentialsError(
            f"username {name!r} contains '|' or ',', which are protocol delimiters in the "
            "Showdown message format and would corrupt the login command."
        )

    password = env.get(PASSWORD_ENV) or None
    if password is None:
        if is_public(server):
            raise CredentialsError(
                f"${PASSWORD_ENV} is empty and {server.websocket_url} is the PUBLIC server. "
                "poke-env would bypass authentication, connect as a Guest, and then hang forever "
                "with no error -- it never raises and never sets logged_in. Set the password, or "
                "use --server to point at a private eval server."
            )
        if not allow_guest:
            raise CredentialsError(
                f"${PASSWORD_ENV} is empty. Pass --allow-guest to connect anonymously to "
                f"{server.websocket_url} (only permitted off the public server)."
            )
    return AccountConfiguration(name, password)

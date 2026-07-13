"""The Showdown port must be overridable, and its default must not move.

poke-env's LocalhostServerConfiguration pins ws://localhost:8000. Two runs sharing a host
therefore share one server and silently see each other's battles -- which is what forces seeds
to run one at a time. LATEGAME_SHOWDOWN_PORT lets each job bring its own server (scripts/cluster/),
turning a sequential sweep into an sbatch array.

The default is load-bearing in the other direction: every existing run, checkpoint and result in
this repo was produced against :8000, so a drifting default would silently repoint them.
"""

import importlib

from poke_env import LocalhostServerConfiguration

import lategame.config


def _reload(monkeypatch, port: str | None):
    if port is None:
        monkeypatch.delenv("LATEGAME_SHOWDOWN_PORT", raising=False)
    else:
        monkeypatch.setenv("LATEGAME_SHOWDOWN_PORT", port)
    return importlib.reload(lategame.config)


def test_default_is_identical_to_poke_env_localhost(monkeypatch):
    """Unset env must reproduce poke-env's config exactly -- not merely 'also port 8000'."""
    cfg = _reload(monkeypatch, None)
    assert cfg.SHOWDOWN_PORT == 8000
    assert cfg.LOCAL_SERVER == LocalhostServerConfiguration


def test_env_var_overrides_the_port(monkeypatch):
    cfg = _reload(monkeypatch, "8123")
    assert cfg.SHOWDOWN_PORT == 8123
    assert cfg.LOCAL_SERVER.websocket_url == "ws://localhost:8123/showdown/websocket"


def test_auth_url_is_untouched_by_the_override(monkeypatch):
    """Only the websocket port moves; the (unused, --no-security) auth URL must not."""
    cfg = _reload(monkeypatch, "9001")
    assert cfg.LOCAL_SERVER.authentication_url == LocalhostServerConfiguration.authentication_url


def test_restores_default_after_reload(monkeypatch):
    """Guard the reload fixture itself: a leaked env var would silently repoint later tests."""
    _reload(monkeypatch, "8123")
    cfg = _reload(monkeypatch, None)
    assert cfg.LOCAL_SERVER == LocalhostServerConfiguration

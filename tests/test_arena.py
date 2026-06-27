"""Arena smoke test. The end-to-end battle test needs a local Showdown server
on :8000; it is skipped automatically when the server is not running."""

import socket

import pytest

from lategame.eval.arena import build_player, evaluate


def _server_up(host: str = "localhost", port: int = 8000, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_build_player_rejects_unknown_agent():
    with pytest.raises(ValueError):
        build_player("not-an-agent", "gen9randombattle")


@pytest.mark.skipif(not _server_up(), reason="local Showdown server not running on :8000")
async def test_random_vs_random_completes_a_battle():
    result = await evaluate("random", "random", n_battles=1)
    assert result.n_battles == 1
    assert 0.0 <= result.p1_win_rate <= 1.0

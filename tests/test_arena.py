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


def test_build_player_forwards_loop_penalty_only_to_learned_agents(monkeypatch):
    """loop_penalty must reach bc/offrl/ppo (they carry a LoopGuard) but never a baseline
    like random (RandomPlayer has no such kwarg)."""
    import lategame.eval.arena as arena

    captured: dict[str, dict] = {}

    def _fake(tag: str):
        class _Fake:
            def __init__(self, **kwargs):
                captured[tag] = kwargs

        return _Fake

    monkeypatch.setitem(arena.AGENTS, "bc", _fake("bc"))
    monkeypatch.setitem(arena.AGENTS, "random", _fake("random"))

    build_player("bc", "gen9ou", checkpoint_path="x.pt", loop_penalty=4.0)
    build_player("random", "gen9ou", loop_penalty=4.0)

    assert captured["bc"]["loop_penalty"] == 4.0
    assert "loop_penalty" not in captured["random"]


@pytest.mark.skipif(not _server_up(), reason="local Showdown server not running on :8000")
async def test_random_vs_random_completes_a_battle():
    result = await evaluate("random", "random", n_battles=1)
    assert result.n_battles == 1
    assert 0.0 <= result.p1_win_rate <= 1.0

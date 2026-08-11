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


class _FakePlayer:
    """A player whose counters advance by a scripted (won, lost, tied) on ``battle_against``.

    Counters are CUMULATIVE in poke-env (they run over ``player._battles``), so the fake is
    cumulative too -- ``evaluate_built`` differences around the games and a fake that reset would
    hide exactly the bug the differencing exists to avoid.
    """

    def __init__(self, won: int, lost: int, tied: int) -> None:
        self._script = (won, lost, tied)
        self.n_won_battles = 7  # nonzero starting offsets: a read of the ABSOLUTE
        self.n_lost_battles = 3  # counter instead of the delta must not pass
        self.n_tied_battles = 1
        self.n_finished_battles = 11

    async def battle_against(self, other: "_FakePlayer", n_battles: int) -> None:
        won, lost, tied = self._script
        self.n_won_battles += won
        self.n_lost_battles += lost
        self.n_tied_battles += tied
        self.n_finished_battles += won + lost + tied


async def test_a_tie_scores_a_half_not_a_loss():
    """``cross_evaluate`` reports ``n_won / n_finished``, so a tie sits in the denominator without
    contributing to the numerator -- scored as a loss. 6W/2L/2T is 0.70, not 0.60."""
    from lategame.eval.arena import evaluate_built

    p1 = _FakePlayer(won=6, lost=2, tied=2)
    assert await evaluate_built(p1, _FakePlayer(2, 6, 2), n_battles=10) == pytest.approx(0.70)


async def test_identical_to_the_old_win_rate_on_a_tie_free_record():
    """Every result this project has published is tie-free (singles ties are vanishingly rare on
    gen9ou), so the fix must not move a single one of them."""
    from lategame.eval.arena import evaluate_built

    p1 = _FakePlayer(won=7, lost=3, tied=0)
    assert await evaluate_built(p1, _FakePlayer(3, 7, 0), n_battles=10) == pytest.approx(7 / 10)


async def test_a_pair_where_nothing_finished_scores_zero_rather_than_dividing_by_zero():
    from lategame.eval.arena import evaluate_built

    p1 = _FakePlayer(won=0, lost=0, tied=0)
    assert await evaluate_built(p1, _FakePlayer(0, 0, 0), n_battles=10) == 0.0


@pytest.mark.skipif(not _server_up(), reason="local Showdown server not running on :8000")
async def test_random_vs_random_completes_a_battle():
    result = await evaluate("random", "random", n_battles=1)
    assert result.n_battles == 1
    assert 0.0 <= result.p1_win_rate <= 1.0

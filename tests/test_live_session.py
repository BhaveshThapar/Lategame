"""Session supervisor: modes, the policy gate, and recovery from a connection that cannot reconnect.

Everything here runs against a fake Player injected through `run_session(player_factory=...)`.
That seam is the reason `build_live_player` is a separate function -- no socket, no torch, no
checkpoint is touched by these tests.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from lategame.live.policy import ALLOW_LADDER_ENV, LADDER_ACK, PolicyError
from lategame.live.session import LiveConfig, _accept_list, chunk_count, run_session


class FakePlayer:
    """Only the surface session.py touches."""

    def __init__(self, *, logged_in=True, hang=False, finish=True, opp_rating=None):
        self.battles = {}
        self._opp_rating = opp_rating
        self.n_finished_battles = 0
        self.username = "lategamebot"
        self.plays: list[tuple] = []  # (mode, k, opponent) -- play calls only
        self.stopped = False
        self._hang = hang
        self._finish = finish
        self._n = 0
        self.logger = logging.getLogger(f"fake-{id(self)}")
        self.ps_client = SimpleNamespace(
            logged_in=SimpleNamespace(is_set=lambda: logged_in),
            stop_listening=self._stop,
        )

    async def _stop(self):
        self.stopped = True

    async def _play(self, kind, k, opponent=None):
        self.plays.append((kind, k, opponent))
        if self._hang:
            await asyncio.sleep(3600)
        if self._finish:
            for _ in range(k):
                self._n += 1
                tag = f"battle-{self._n}"
                self.battles[tag] = SimpleNamespace(
                    battle_tag=tag, finished=True, won=True, turn=15,
                    opponent_username="rival", rating=None,
                    opponent_rating=self._opp_rating,
                )
                self.n_finished_battles += 1

    async def ladder(self, k):
        await self._play("ladder", k)

    async def accept_challenges(self, opponent, k, packed_team=None):
        await self._play("accept", k, opponent)

    async def send_challenges(self, opponent, k, to_wait=None):
        await self._play("challenge", k, opponent)


def _cfg(tmp_path, **kw):
    return LiveConfig(out=str(tmp_path / "live.json"), **kw)


# --------------------------------------------------------------------------- #
# Validation + policy
# --------------------------------------------------------------------------- #


async def test_challenge_mode_requires_an_opponent(tmp_path):
    with pytest.raises(ValueError, match="requires --opponent"):
        await run_session(_cfg(tmp_path, mode="challenge"), lambda c: FakePlayer())


async def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        await run_session(_cfg(tmp_path, mode="nope"), lambda c: FakePlayer())


async def test_ladder_refuses_without_both_opt_in_channels(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_LADDER_ENV, raising=False)
    with pytest.raises(PolicyError):
        await run_session(_cfg(tmp_path, mode="ladder"), lambda c: FakePlayer())
    # right ack, missing env
    with pytest.raises(PolicyError):
        await run_session(_cfg(tmp_path, mode="ladder", ladder_ack=LADDER_ACK),
                          lambda c: FakePlayer())


async def test_ladder_runs_with_both_channels_and_records_the_ack(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ALLOW_LADDER_ENV, "1")
    player = FakePlayer()
    cfg = _cfg(tmp_path, mode="ladder", ladder_ack=LADDER_ACK, n=2)
    out = await run_session(cfg, lambda c: player)

    assert [c[0] for c in player.plays] == ["ladder", "ladder"]
    assert out.summary["finished"] == 2
    assert "NG3" in capsys.readouterr().err  # the policy note is surfaced, not just recorded
    written = json.loads((tmp_path / "live.json").read_text())
    assert written["policy"]["ranked"] is True
    assert written["policy"]["ack"] == LADDER_ACK
    assert "NG3" in written["policy"]["note"]  # provenance travels with the number


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


async def test_accept_mode_passes_none_for_an_open_allowlist(tmp_path):
    player = FakePlayer()
    await run_session(_cfg(tmp_path, mode="accept", n=1), lambda c: player)
    assert player.plays[0] == ("accept", 1, None)


async def test_accept_mode_normalises_a_comma_allowlist(tmp_path):
    player = FakePlayer()
    await run_session(_cfg(tmp_path, mode="accept", opponent="alice, bob", n=1), lambda c: player)
    assert player.plays[0] == ("accept", 1, ["alice", "bob"])


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("", None), (" , ", None), ("alice", ["alice"]), ("a,b , c", ["a", "b", "c"])],
)
def test_accept_list_normalisation(raw, expected):
    assert _accept_list(raw) == expected


async def test_challenge_mode_targets_the_named_user(tmp_path):
    player = FakePlayer()
    await run_session(_cfg(tmp_path, mode="challenge", opponent="rival", n=3), lambda c: player)
    assert player.plays == [("challenge", 1, "rival")] * 3


async def test_battles_are_chunked_by_concurrency(tmp_path):
    """Chunking is what buys a Ctrl-C boundary and per-battle telemetry."""
    player = FakePlayer()
    await run_session(_cfg(tmp_path, mode="accept", n=5, concurrency=2), lambda c: player)
    assert [c[1] for c in player.plays] == [2, 2, 1]


@pytest.mark.parametrize("n,c,expected", [(1, 1, 1), (5, 2, 3), (6, 3, 2), (3, 10, 1)])
def test_chunk_count(n, c, expected):
    assert chunk_count(n, c) == expected


# --------------------------------------------------------------------------- #
# Telemetry + recovery
# --------------------------------------------------------------------------- #


async def test_results_file_is_rewritten_after_every_battle(tmp_path):
    """A SIGKILL must lose at most one game."""
    seen = []
    out = tmp_path / "live.json"

    class _Watching(FakePlayer):
        async def accept_challenges(self, opponent, k, packed_team=None):
            await super().accept_challenges(opponent, k)
            if out.exists():
                seen.append(json.loads(out.read_text())["finished"])

    player = _Watching()
    await run_session(_cfg(tmp_path, mode="accept", n=3), lambda c: player)
    assert json.loads(out.read_text())["finished"] == 3
    assert len(seen) >= 2  # written mid-run, not only at the end


async def test_a_wedged_connection_trips_the_watchdog_and_restarts(tmp_path, monkeypatch):
    """poke-env neither raises nor reconnects, so this must be caught by timeout."""
    monkeypatch.setattr("lategame.live.session._WATCHDOG_PERIOD", 0.05)
    built = []

    def factory(cfg):
        # first player hangs forever; the replacement works
        player = FakePlayer(hang=not built)
        built.append(player)
        return player

    cfg = _cfg(tmp_path, mode="accept", n=1, stall_timeout=0.15, battle_timeout=30.0,
               max_restarts=2, backoff_base=0.0)
    out = await run_session(cfg, factory)

    assert len(built) == 2  # rebuilt, because a dead socket cannot be reconnected
    assert out.restarts == 1
    assert out.summary["finished"] == 1
    assert built[0].stopped is True  # the wedged client was torn down


async def test_restarts_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("lategame.live.session._WATCHDOG_PERIOD", 0.05)
    built = []

    def factory(cfg):
        built.append(FakePlayer(hang=True))
        return built[-1]

    cfg = _cfg(tmp_path, mode="accept", n=1, stall_timeout=0.1, max_restarts=2,
               backoff_base=0.0)
    out = await run_session(cfg, factory)

    assert out.restarts == 3  # attempts 0,1,2 then the bound stops it
    assert out.stopped_early is True
    assert out.summary["finished"] == 0


async def test_a_failed_login_is_fatal_and_never_retried(tmp_path):
    """Hammering a bad password is how a live account gets locked."""
    from lategame.live.player import LoginError

    built = []

    def factory(cfg):
        built.append(FakePlayer(logged_in=False))
        return built[-1]

    cfg = _cfg(tmp_path, mode="accept", n=1, login_timeout=0.2)
    with pytest.raises(LoginError):
        await run_session(cfg, factory)
    assert len(built) == 1  # exactly one attempt -- no retry storm


async def test_stop_event_drains_cleanly(tmp_path):
    stop = asyncio.Event()

    class _Stopping(FakePlayer):
        async def accept_challenges(self, opponent, k, packed_team=None):
            await FakePlayer.accept_challenges(self, opponent, k)
            if self.n_finished_battles >= 2:
                stop.set()

    player = _Stopping()
    out = await run_session(_cfg(tmp_path, mode="accept", n=10), lambda c: player, stop=stop)

    assert out.summary["finished"] == 2
    assert out.stopped_early is True


async def test_credentials_never_reach_the_results_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LATEGAME_PS_PASSWORD", "hunter2-do-not-leak")
    await run_session(_cfg(tmp_path, mode="accept", n=1), lambda c: FakePlayer())
    raw = (tmp_path / "live.json").read_bytes()
    assert b"hunter2-do-not-leak" not in raw


# --------------------------------------------------------------------------- #
# The varied-field option: reachable, and applied in BOTH places
# --------------------------------------------------------------------------- #


async def test_use_live_ratings_reaches_the_summary(tmp_path):
    """`summarize(use_live_ratings=True)` had a test but no way to get there from a LiveConfig, so
    a real ladder session would capture every opponent rating and then report the pinned-field GXE
    anyway. This pins the wiring."""
    cfg = _cfg(tmp_path, mode="accept", n=2, use_live_ratings=True)
    out = await run_session(cfg, lambda c: FakePlayer(opp_rating=1800))

    assert out.summary["opponent_rating_source"] == "showdown_elo_approximated_as_glicko"
    assert "units are not identical" in out.summary["gxe_basis"]


async def test_default_still_pins_the_field(tmp_path):
    out = await run_session(
        _cfg(tmp_path, mode="accept", n=2), lambda c: FakePlayer(opp_rating=1800)
    )
    assert "opponent_rating_source" not in out.summary
    assert "pinned at rating.REFERENCE" in out.summary["gxe_basis"]


async def test_incremental_writes_agree_with_the_final_summary(tmp_path):
    """The per-battle flush and the final summary must apply the SAME options. If only one call
    site were threaded, a session would flush a pinned-field GXE all the way through and then
    report a live-rated one -- which reads as a corrupt file, not as a wiring bug."""
    cfg = _cfg(tmp_path, mode="accept", n=3, use_live_ratings=True)
    out = await run_session(cfg, lambda c: FakePlayer(opp_rating=1800))

    written = json.loads((tmp_path / "live.json").read_text())
    assert written["gxe"] == out.summary["gxe"]
    assert written["gxe_basis"] == out.summary["gxe_basis"]
    assert written["use_live_ratings"] is True


async def test_opponent_rd_is_threaded(tmp_path):
    """A tighter deviation on the opponent makes its rating count for more, so the same record
    must not produce the same GXE at both settings."""
    loose = await run_session(
        _cfg(tmp_path, mode="accept", n=3, use_live_ratings=True, opponent_rd=350.0),
        lambda c: FakePlayer(opp_rating=1900),
    )
    tight = await run_session(
        _cfg(tmp_path, mode="accept", n=3, use_live_ratings=True, opponent_rd=30.0),
        lambda c: FakePlayer(opp_rating=1900),
    )
    assert loose.summary["gxe"] != tight.summary["gxe"]

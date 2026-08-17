"""End-to-end live session against the LOCAL server -- the real Player, socket, and rating sweep.

Deliberately runs against ``localhost``, never the public sim: it exercises every line the public
path would (websocket, login, the finished-battle callback, the finalize sweep, the results file)
with zero account and zero policy exposure. That is also exactly the "private eval server" plan.md
section 15 prefers.

Opt-in via ROTOMAI_LIVE_TEST=1, because it needs a running `pokemon-showdown --no-security` on
ROTOMAI_SHOWDOWN_PORT (default 8000) and takes a few seconds.
"""

import json
import os

import pytest

from rotomai.config import SHOWDOWN_PORT
from tests.conftest import server_up

_WS = f"ws://localhost:{SHOWDOWN_PORT}/showdown/websocket"




requires_live = pytest.mark.skipif(
    not os.environ.get("ROTOMAI_LIVE_TEST") or not server_up(port=SHOWDOWN_PORT),
    reason="opt-in: set ROTOMAI_LIVE_TEST=1 with a local Showdown server running",
)


@requires_live
async def test_two_local_players_complete_a_live_session(tmp_path):
    """A real challenge/accept pair, driven entirely through run_session."""
    import asyncio

    from rotomai.live.player import build_live_player
    from rotomai.live.server import live_account, live_server
    from rotomai.live.session import LiveConfig, run_session

    server = live_server(_WS)
    out = tmp_path / "live.json"

    # The opponent: a plain local player that accepts whatever arrives.
    opponent = build_live_player(
        "random", "gen9randombattle",
        live_account("smoke-opp", server=server, allow_guest=True, env={}),
        server,
        start_timer_on_battle_start=False,
    )
    accepting = asyncio.create_task(opponent.accept_challenges(None, 1))
    await asyncio.sleep(1.0)  # let the acceptor register before the challenge lands

    cfg = LiveConfig(
        agent="random", mode="challenge", opponent=opponent.username,
        battle_format="gen9randombattle", n=1,
        ws_url=_WS, allow_guest=True, username="smoke-bot",
        start_timer=False, out=str(out), login_timeout=20.0, stall_timeout=60.0,
        battle_timeout=120.0, max_restarts=0,
    )
    outcome = await run_session(cfg)
    await asyncio.wait_for(accepting, timeout=30)

    assert outcome.summary["finished"] == 1
    assert outcome.summary["unfinished"] == 0
    assert outcome.records[0].turns > 0
    assert outcome.records[0].result in {"win", "loss", "tie"}

    written = json.loads(out.read_text())
    assert written["schema"] == "rotomai.live.telemetry/1"
    assert written["policy"]["ranked"] is False  # challenge mode is never ranked
    assert written["gxe"] is not None and 0.0 <= written["gxe"] <= 1.0
    # Unrated local play never emits a |raw| rating line; None everywhere is the correct outcome.
    assert written["showdown_elo"]["n_reported"] == 0

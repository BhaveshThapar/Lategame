"""Live Showdown play (plan.md M5 / G1).

Everything through Build 26 plays a FIXED baseline on a local server, where win rate is the
sufficient statistic. G2 is stated in GXE/Glicko instead, and those are ladder metrics: they need a
varied opponent field, which only live play provides. This package is that missing half.

Import cost note: ``policy`` is dependency-free and is the only submodule ``lategame/cli.py``
imports at module scope, so building ``--help`` does not drag in poke-env or torch.
"""

from lategame.live.player import LoginError, build_live_player
from lategame.live.policy import (
    ALLOW_LADDER_ENV,
    LADDER_ACK,
    PASSWORD_ENV,
    POLICY_NOTE,
    USERNAME_ENV,
    PolicyError,
)
from lategame.live.server import LIVE_SERVER, CredentialsError, live_account, live_server
from lategame.live.session import MODES, LiveConfig, SessionOutcome, run_session
from lategame.live.telemetry import BattleRecord, LiveLog, summarize, write_results

__all__ = [
    "ALLOW_LADDER_ENV",
    "LADDER_ACK",
    "LIVE_SERVER",
    "MODES",
    "PASSWORD_ENV",
    "POLICY_NOTE",
    "USERNAME_ENV",
    "BattleRecord",
    "CredentialsError",
    "LiveConfig",
    "LiveLog",
    "LoginError",
    "PolicyError",
    "SessionOutcome",
    "build_live_player",
    "live_account",
    "live_server",
    "run_session",
    "summarize",
    "write_results",
]

"""Server configuration and format constants for local play/evaluation."""

from __future__ import annotations

import os

from poke_env import AccountConfiguration, ServerConfiguration

# Battle formats. Phase 1 (MVP) targets Gen 9 Random Battles; the others are
# placeholders for later phases (see plan.md, Section 6).
DEFAULT_FORMAT = "gen9randombattle"
OU_FORMAT = "gen9ou"
VGC_FORMAT = "gen9vgc2024regh"

# Local `pokemon-showdown --no-security` server. The port is overridable via
# LATEGAME_SHOWDOWN_PORT so that several runs can share one machine: poke-env's
# LocalhostServerConfiguration pins ws://localhost:8000, which makes two concurrent
# jobs on the same host silently fight over one server (they see each other's
# battles). On a cluster that is the difference between an sbatch array and one
# seed at a time -- so the port has to come from the environment, and the job
# script starts its own Showdown on the matching port.
SHOWDOWN_PORT: int = int(os.environ.get("LATEGAME_SHOWDOWN_PORT", "8000"))

LOCAL_SERVER: ServerConfiguration = ServerConfiguration(
    f"ws://localhost:{SHOWDOWN_PORT}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


def local_account(username: str) -> AccountConfiguration:
    """Account config for the local server, which needs no password."""
    return AccountConfiguration(username, None)

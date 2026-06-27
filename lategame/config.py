"""Server configuration and format constants for local play/evaluation."""

from __future__ import annotations

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration

# Battle formats. Phase 1 (MVP) targets Gen 9 Random Battles; the others are
# placeholders for later phases (see plan.md, Section 6).
DEFAULT_FORMAT = "gen9randombattle"
OU_FORMAT = "gen9ou"
VGC_FORMAT = "gen9vgc2024regh"

# Local `pokemon-showdown --no-security` server on ws://localhost:8000.
LOCAL_SERVER: ServerConfiguration = LocalhostServerConfiguration


def local_account(username: str) -> AccountConfiguration:
    """Account config for the local server, which needs no password."""
    return AccountConfiguration(username, None)

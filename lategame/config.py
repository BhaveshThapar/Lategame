"""Server configuration and format constants for local play/evaluation."""

from __future__ import annotations

import os

from poke_env import AccountConfiguration, ServerConfiguration

# Battle formats. Phase 1 (MVP) targets Gen 9 Random Battles; Phase 2 is gen9ou (the amended G2
# target); Phase 3 is VGC doubles (see plan.md, Section 6).
DEFAULT_FORMAT = "gen9randombattle"
OU_FORMAT = "gen9ou"

# CORRECTED 2026-08-11: this was `gen9vgc2024regh`, WHICH THE VENDORED SIMULATOR DOES NOT HAVE.
# It was written as a Phase-3 placeholder and never exercised, so nothing failed. Asking the
# pinned simulator (third_party/pokemon-showdown, rev 393d5c8) for its gen 9 doubles formats
# returns 2023 Reg C/D, 2024 Reg G, 2025 Reg I, and the [Gen 9 Champions] 2026 Reg M-A/M-B mod --
# no Reg H. `gen9vgc2025regi` is the current standard `[Gen 9]` VGC format there.
#
# The Bo3 variants (`gen9championsvgc2026regma bo3`, ...) are deliberately NOT the default: a
# "battle" in those is a best-of-three SERIES, which every win/loss counter in `eval/` would
# miscount without noticing.
VGC_FORMAT = "gen9vgc2025regi"

#: Substrings that mark a format as doubles. Same shape as the `"randombattle" in fmt` test the
#: ceiling gate already uses to decide whether a team pool is needed -- the simulator's own format
#: table is a node artifact and not reachable from Python.
_DOUBLES_MARKERS = ("doubles", "vgc", "2v2", "triples")


def is_doubles_format(battle_format: str) -> bool:
    """Whether ``battle_format`` puts TWO Pokemon on each side.

    Load-bearing rather than cosmetic: on a doubles battle poke-env hands an agent
    ``active_pokemon`` as a *list* and ``available_moves`` as a list-of-lists, and expects a
    ``DoubleBattleOrder`` back. A singles-only policy handed one of those does not degrade -- see
    ``eval.arena.build_player``.
    """
    fmt = battle_format.lower()
    return any(marker in fmt for marker in _DOUBLES_MARKERS)

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

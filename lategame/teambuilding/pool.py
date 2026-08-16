"""A ``Teambuilder`` that draws each battle's team at random from a fixed pool.

poke-env's ``ConstantTeambuilder`` yields one fixed team; for the ceiling probe (and later
OU eval / self-play) we want variety, so every bot draws from a shared pool of pre-validated,
format-legal packed teams. Both sides sampling the same pool keeps the mirror match fair in
expectation. Packing/parsing reuses poke-env's ``Teambuilder`` helpers -- we do not reimplement
the packed format.
"""

from __future__ import annotations

import random
from pathlib import Path

from poke_env.teambuilder.teambuilder import Teambuilder

# Curated, validator-checked pool written by scripts/build_ou_teampool.py (one packed team per
# line). Committed under the package (like features/data/) since /data/ holds gitignored shards.
DEFAULT_OU_POOL = Path("lategame/teambuilding/data/teams_gen9ou.packed")
# The VGC twin. Named here rather than spelled out at each call site: a teambuilt format that
# gets the wrong pool does not fail, it plays illegal-for-the-format teams the server rejects
# one battle at a time, and a typo'd path is a FileNotFoundError only if you are lucky.
DEFAULT_VGC_POOL = Path("lategame/teambuilding/data/teams_gen9vgc.packed")


class TeamPool(Teambuilder):
    """Yields a uniformly-random packed team from ``teams`` on each ``yield_team`` call."""

    def __init__(self, teams: list[str], seed: int = 0) -> None:
        if not teams:
            raise ValueError("TeamPool requires at least one team")
        self._teams = list(teams)
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        return self._rng.choice(self._teams)

    @property
    def teams(self) -> list[str]:
        return list(self._teams)

    @classmethod
    def from_packed_file(cls, path: str | Path = DEFAULT_OU_POOL, seed: int = 0) -> TeamPool:
        """Load a pool from a file with one packed team per line (blank lines ignored)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"team pool {p} not found -- run scripts/build_ou_teampool.py first"
            )
        teams = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if not teams:
            raise ValueError(f"team pool {p} is empty")
        return cls(teams, seed=seed)


def pack_showdown_team(paste: str) -> str:
    """Convert a Showdown-paste team (one blank line between mons) to packed format."""
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(paste))

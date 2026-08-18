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
DEFAULT_OU_POOL = Path("rotomai/teambuilding/data/teams_gen9ou.packed")
# The VGC twin. Named here rather than spelled out at each call site: a teambuilt format that
# gets the wrong pool does not fail, it plays illegal-for-the-format teams the server rejects
# one battle at a time, and a typo'd path is a FileNotFoundError only if you are lucky.
DEFAULT_VGC_POOL = Path("rotomai/teambuilding/data/teams_gen9vgc.packed")

#: Where the pools actually live once the package is imported, wherever the process was started.
_PACKAGE_DATA = Path(__file__).resolve().parent / "data"


def resolve_pool_path(path: str | Path) -> Path:
    """Find a pool file whether or not the process was started from the repo root.

    The ``DEFAULT_*_POOL`` constants are deliberately REPO-RELATIVE and stay that way: their string
    form is written verbatim into every results file and pre-registration as ``team_pool``, and
    ``merge_live_sessions.py`` refuses to pool segments whose arm fields disagree. Rewriting them to
    absolute paths would make two runs of the same arm on two machines look like two experiments.

    So the constant stays relative and resolution happens here instead: if the literal path does not
    exist, fall back to the same basename inside the installed package data. That is what makes a
    long-lived ``accept``-mode service work -- it is launched by systemd or a container from ``/``,
    not from a checkout, and before this the pool lookup died with a bare ``FileNotFoundError``.
    """
    p = Path(path)
    if p.exists():
        return p
    candidate = _PACKAGE_DATA / p.name
    return candidate if candidate.exists() else p


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
        p = resolve_pool_path(path)
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

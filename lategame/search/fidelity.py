"""Forward-model fidelity check (Lever 11 / R-PREDICT, Gate A -- the cheap KILL gate).

Search needs a forward model -- step a hypothetical (my-move, opp-move) -> next state --
and we build it on the vendored simulator's ``State.serializeBattle``/``deserializeBattle``
(fork) + ``battle.choose`` (step). This module validates that primitive is *faithful*:
``search/fidelity_driver.js`` replays real replays turn-by-turn and, at every decision
round, forks the battle, steps the fork with the same choices, and checks the result
matches stepping the battle directly. The serialized PRNG seed makes a faithful fork an
*exact* match (identical damage rolls / crits / accuracy), so a high mismatch rate is a
genuine infidelity, not noise. If the primitive isn't faithful, search would be GIGO and
the lever pivots to the curriculum fallback -- hence Gate A runs before any search code.

Mirrors ``data.resim.run_driver``: one long-lived node process, NDJSON in/out.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_SHOWDOWN = "third_party/pokemon-showdown"
_DRIVER = Path(__file__).with_name("fidelity_driver.js")
_MAX_GLOBAL_SAMPLES = 5  # keep a handful of mismatch diffs for the report


@dataclass
class FidelityStats:
    """Aggregate forward-model fidelity over a batch of replays."""

    replays: int = 0
    errored: int = 0
    transitions: int = 0
    core_mismatch: int = 0
    full_mismatch: int = 0
    drive_errors: int = 0
    samples: list[Mapping[str, object]] = field(default_factory=list)

    @property
    def core_match_rate(self) -> float:
        """Fraction of transitions whose hp/status/faint/field/hazards/active match exactly."""
        return 1.0 if self.transitions == 0 else 1.0 - self.core_mismatch / self.transitions

    @property
    def full_match_rate(self) -> float:
        """Fraction matching on the stricter superset (+ boosts/item/ability/pp/tera)."""
        return 1.0 if self.transitions == 0 else 1.0 - self.full_mismatch / self.transitions

    def to_dict(self) -> dict[str, object]:
        return {
            "replays": self.replays,
            "errored": self.errored,
            "transitions": self.transitions,
            "core_mismatch": self.core_mismatch,
            "full_mismatch": self.full_mismatch,
            "drive_errors": self.drive_errors,
            "core_match_rate": round(self.core_match_rate, 6),
            "full_match_rate": round(self.full_match_rate, 6),
            "samples": self.samples,
        }


def run_fidelity(
    replays: Iterable[Mapping[str, object]],
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> FidelityStats:
    """Fork/step every decision round of each replay; aggregate the match rates.

    Each input is ``{"id", "inputlog"}``. One node process handles the whole batch.
    """
    if not _DRIVER.exists():
        raise RuntimeError(f"fidelity driver missing: {_DRIVER}")
    if not (Path(showdown_dir) / "dist").is_dir():
        raise RuntimeError(
            f"'{showdown_dir}/dist' not found -- build the vendored simulator first "
            "(run `bash scripts/setup_server.sh`)."
        )
    payload = "".join(
        json.dumps({"id": r.get("id"), "inputlog": r.get("inputlog", "")}) + "\n"
        for r in replays
    )
    proc = subprocess.run(
        [node, str(_DRIVER), str(showdown_dir)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fidelity driver failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}"
        )

    stats = FidelityStats()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        res = json.loads(line)
        stats.replays += 1
        if "error" in res:
            stats.errored += 1
            continue
        stats.transitions += int(res.get("transitions", 0))
        stats.core_mismatch += int(res.get("core_mismatch", 0))
        stats.full_mismatch += int(res.get("full_mismatch", 0))
        stats.drive_errors += int(res.get("drive_errors", 0))
        for sample in res.get("samples", []):
            if len(stats.samples) < _MAX_GLOBAL_SAMPLES:
                stats.samples.append({"id": res.get("id"), **sample})
    return stats


def run_fidelity_files(
    paths: Iterable[str | Path],
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> FidelityStats:
    """Load cached replay ``.json`` files (must carry ``inputlog``) and fork/step them."""

    def _load() -> Iterable[Mapping[str, object]]:
        for path in paths:
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("inputlog"):
                yield {"id": data.get("id"), "inputlog": data["inputlog"]}

    return run_fidelity(_load(), showdown_dir, node)

"""Reconstruction fidelity check (Lever 11 / R-PREDICT) -- the determinization mini-gate.

Gate A proved the forward *step* is faithful. This checks the other half of the forward
model: does ``determinize.battle_to_spec`` -> ``forward_driver`` reconstruct preserve the
*observable* state of a live POV? We get real POV battles by re-simulating replays (reusing
``data.resim``), snapshot each decision turn, determinize it, and compare the reconstructed
battle's observable digest to poke-env's. Hidden opponent details (filled slots) are not
checked -- only what the live agent actually observes (our full team, revealed opponent mons,
field, hazards). Cheap: no live server, no training -- just node reconstructs over replays.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from poke_env.battle import AbstractBattle, Battle

from lategame.data.resim import (
    _DEFAULT_SHOWDOWN,
    _LOGGER,
    _feed_line,
    _parse_inputlog_meta,
    run_driver,
)
from lategame.search.determinize import battle_to_spec, pokeenv_digest
from lategame.search.forward import ForwardModel

_HP_TOL = 0.02  # opponent hp is observed only to ~1%, so allow a one-step rounding band


def _decision_snapshots(
    events: Iterable[Mapping[str, str]], username: str, tag: str, gen: int
) -> Iterator[AbstractBattle]:
    """Yield a POV battle snapshot at each actionable decision turn (mirrors resim's loop)."""
    battle = Battle(battle_tag=tag, username=username, gen=gen, logger=_LOGGER)
    for event in events:
        if event.get("t") == "msg":
            for line in event.get("d", "").split("\n"):
                _feed_line(battle, line)
        elif event.get("t") == "choice":
            if not battle._wait and not battle.teampreview and battle.active_pokemon is not None:
                yield copy.deepcopy(battle)


@dataclass
class ReconStats:
    snapshots: int = 0
    errored: int = 0
    checks: int = 0
    mismatches: int = 0
    by_field: dict[str, list[int]] = field(default_factory=dict)  # field -> [mismatch, total]
    samples: list[Mapping[str, Any]] = field(default_factory=list)

    def _bump(self, name: str, ok: bool) -> None:
        slot = self.by_field.setdefault(name, [0, 0])
        slot[1] += 1
        self.checks += 1
        if not ok:
            slot[0] += 1
            self.mismatches += 1

    @property
    def match_rate(self) -> float:
        return 1.0 if self.checks == 0 else 1.0 - self.mismatches / self.checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": self.snapshots,
            "errored": self.errored,
            "checks": self.checks,
            "mismatches": self.mismatches,
            "match_rate": round(self.match_rate, 6),
            "by_field": {
                k: {
                    "mismatch": v[0],
                    "total": v[1],
                    "rate": round(1 - v[0] / v[1], 4) if v[1] else 1.0,
                }
                for k, v in sorted(self.by_field.items())
            },
            "samples": self.samples,
        }


def _compare_mon(
    stats: ReconStats, side: str, sp: str, pe: Mapping, drv: Mapping, sample: list
) -> None:
    if abs(float(pe["hp"]) - float(drv.get("hp", -9))) > _HP_TOL:
        sample.append(f"{side}.{sp}.hp pe={pe['hp']} drv={drv.get('hp')}")
        stats._bump("hp", False)
    else:
        stats._bump("hp", True)
    for key in ("status", "fainted", "active"):
        ok = pe[key] == drv.get(key)
        if not ok:
            sample.append(f"{side}.{sp}.{key} pe={pe[key]} drv={drv.get(key)}")
        stats._bump(key, ok)
    ok = dict(pe.get("boosts", {})) == dict(drv.get("boosts", {}))
    if not ok:
        sample.append(f"{side}.{sp}.boosts pe={pe.get('boosts')} drv={drv.get('boosts')}")
    stats._bump("boosts", ok)


def _compare(stats: ReconStats, pe: Mapping, drv: Mapping) -> None:
    sample: list[str] = []
    for k in ("weather", "terrain"):
        ok = pe[k] == drv.get(k)
        if not ok:
            sample.append(f"{k} pe={pe[k]} drv={drv.get(k)}")
        stats._bump(k, ok)
    # p1 = our full team (every mon must be reproduced); p2 = revealed opponent mons only.
    for side, check_all in (("p1", True), ("p2", False)):
        drv_side = drv.get(side, {})
        for sp, mon in pe[side].items():
            if sp not in drv_side:
                if check_all:
                    sample.append(f"{side}.{sp} MISSING in reconstruction")
                    stats._bump("present", False)
                continue
            if check_all:
                stats._bump("present", True)
            _compare_mon(stats, side, sp, mon, drv_side[sp], sample)
    for side in ("p1", "p2"):
        drv_hz = drv.get("hazards", {}).get(side, {})
        ok = dict(pe["hazards"][side]) == dict(drv_hz)
        if not ok:
            sample.append(f"hazards.{side} pe={pe['hazards'][side]} drv={drv_hz}")
        stats._bump("hazards", ok)
    if sample and len(stats.samples) < 8:
        stats.samples.append({"diffs": sample[:6]})


def run_recon_check(
    replays: Iterable[Mapping[str, object]],
    showdown_dir: str = _DEFAULT_SHOWDOWN,
    node: str = "node",
    max_snapshots_per_replay: int = 0,
) -> ReconStats:
    """Re-simulate ``replays``, snapshot decision turns, and score reconstruction fidelity."""
    prepared: list[tuple[str, str, int, str, str]] = []
    for rep in replays:
        il = rep.get("inputlog")
        if isinstance(il, str) and il:
            rid = str(rep.get("id") or f"r{len(prepared)}")
            p1, p2, gen = _parse_inputlog_meta(il)
            prepared.append((rid, il, gen, p1, p2))

    driver_in = ({"id": rid, "inputlog": il} for rid, il, _, _, _ in prepared)
    results = {str(r.get("id")): r for r in run_driver(driver_in, showdown_dir, node)}

    stats = ReconStats()
    fm = ForwardModel(showdown_dir=showdown_dir, node=node)
    try:
        for rid, _il, gen, p1n, p2n in prepared:
            res = results.get(rid)
            if not res or "error" in res:
                continue
            for side, user in (("p1", p1n), ("p2", p2n)):
                raw = res.get(side)
                events = raw if isinstance(raw, list) else []
                seen = 0
                for snap in _decision_snapshots(events, user, f"{rid}-{side}", gen):
                    if max_snapshots_per_replay and seen >= max_snapshots_per_replay:
                        break
                    seen += 1
                    stats.snapshots += 1
                    try:
                        drv = fm.reconstruct(battle_to_spec(snap, seed=stats.snapshots))["digest"]
                    except Exception:  # noqa: BLE001 -- a bad reconstruction is a data point, not a crash
                        stats.errored += 1
                        continue
                    _compare(stats, pokeenv_digest(snap), drv)
    finally:
        fm.close()
    return stats

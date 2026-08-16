"""Pool the sharded search-gate runs into one authoritative read (Build 27 Gate B).

`scripts/cluster/search_gate.slurm` splits the pre-registered n across an array because search is
serial in its node driver (~29 s/battle measured, so n=2500/arm is ~40 h in one process). Each
shard is a complete, independent gate at n/shards; this sums them.

WINS, NOT MEAN-OF-RATES. Averaging shard rates is only equal to the pooled rate when every shard
finished the same number of battles, and a shard that timed out mid-run would silently get equal
weight. Each shard's wins are reconstructed as `round(rate * n)` -- exact here because gen9ou
singles ties are vanishingly rare and `evaluate_built` now scores a tie as a half either way --
and the pooled rate is `sum(wins) / sum(n)`.

The pooled contrast is `search_vs_heuristic - base_vs_heuristic`, tested with the same
two-proportion z used by `seed_strength_gate.py`, so it is commensurable with Build 26's doses.

    python scripts/merge_search_shards.py --glob 'results/rpredict_search_ou_shard*.json' \
        --out results/rpredict_search_ou.json
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
from pathlib import Path
from typing import Any

#: Pre-registered bar. Build 26's contrasts resolved ~0.025; the same bar makes a NULL here as
#: strong a claim as the update-count doses, which is the point of running at that n.
ALPHA = 0.05
MDE_NOTE = "pre-registered: WIN iff the pooled delta is significant at alpha and positive"


def _z_test(w1: int, n1: int, w2: int, n2: int) -> dict[str, float]:
    """Two-proportion z, candidate (1) vs baseline (2) -- mirrors seed_strength_gate."""
    p1, p2 = w1 / n1, w2 / n2
    pool = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return {"diff": round(p1 - p2, 4), "z": round(z, 3), "p_value": round(p, 6), "se": round(se, 4)}


def _one(key: str, shards: list[dict[str, Any]]) -> Any:
    """The single value of ``key`` across shards, or None if absent; refuse a disagreement.

    Pooling shards that were run differently produces one number describing two experiments. The
    fields that define the run -- format, init, depth -- are taken from shard 0 by long-standing
    habit; this is for the ones where silently taking the first would hide a real split.
    """
    seen = {s[key] for s in shards if s.get(key) is not None}
    if len(seen) > 1:
        raise SystemExit(f"shards disagree on {key}: {sorted(seen)} -- these are not one run")
    return seen.pop() if seen else None


def merge(paths: list[str]) -> dict[str, Any]:
    shards = [json.loads(Path(p).read_text()) for p in sorted(paths)]
    if not shards:
        raise SystemExit("no shards matched")

    arms = sorted({a for s in shards for a in s.get("arms", {})})
    totals: dict[str, dict[str, list[int]]] = {}  # metric -> arm -> [wins, n]

    def _add(metric: str, arm: str, rate: float | None, n: int) -> None:
        if rate is None:
            return
        slot = totals.setdefault(metric, {}).setdefault(arm, [0, 0])
        slot[0] += int(round(rate * n))
        slot[1] += n

    for s in shards:
        n, sanity = int(s["n"]), int(s.get("sanity_n", 0) or 0)
        _add("base_vs_heuristic", "base", s.get("base_vs_heuristic"), n)
        for arm, r in (s.get("arms") or {}).items():
            _add("search_vs_heuristic", arm, r.get("search_vs_heuristic"), n)
            _add("search_vs_base", arm, r.get("search_vs_base"), n)
            if sanity:
                _add("search_vs_random", arm, r.get("search_vs_random"), sanity)

    base_w, base_n = totals["base_vs_heuristic"]["base"]
    out: dict[str, Any] = {
        "gate": "rpredict_search_pooled",
        "shards": len(shards),
        "format": shards[0].get("format"),
        "init": shards[0].get("init"),
        "depth": shards[0].get("depth"),
        # Which leaf evaluated the search, carried up from the shards. `ShapedOnlyPolicy` is a
        # weaker instrument than a trained value head and `expectimax.ShapedOnlyPolicy`
        # pre-registers the consequence -- a WIN is decisive, a NULL suggestive only -- so a pooled
        # record that dropped it would launder that caveat away. Absent when the shards predate the
        # field; a run whose shards DISAGREE is not poolable and says so rather than picking one.
        "search_leaf": _one("search_leaf", shards),
        "alpha": ALPHA,
        "note": MDE_NOTE,
        "base_vs_heuristic": {"wins": base_w, "n": base_n, "rate": round(base_w / base_n, 4)},
        "arms": {},
    }
    for arm in arms:
        sw, sn = totals["search_vs_heuristic"][arm]
        hw, hn = totals["search_vs_base"][arm]
        test = _z_test(sw, sn, base_w, base_n)
        significant = test["p_value"] < ALPHA
        out["arms"][arm] = {
            "search_vs_heuristic": {"wins": sw, "n": sn, "rate": round(sw / sn, 4)},
            "search_vs_base": {"wins": hw, "n": hn, "rate": round(hw / hn, 4)},
            "search_vs_random": (
                {"rate": round(totals["search_vs_random"][arm][0]
                               / totals["search_vs_random"][arm][1], 4)}
                if arm in totals.get("search_vs_random", {}) else None
            ),
            "contrast_vs_base": test,
            "verdict": (
                "WIN" if significant and test["diff"] > 0
                else "REGRESSION" if significant and test["diff"] < 0
                else "NULL"
            ),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/rpredict_search_ou_shard*.json")
    p.add_argument("--out", default="results/rpredict_search_ou.json")
    args = p.parse_args(argv)

    out = merge(globlib.glob(args.glob))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

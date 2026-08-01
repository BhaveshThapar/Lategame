"""Build the committed Smogon usage-prior artifact (Build 4, plan.md 13.1).

Fetches one month's chaos-stats JSON from smogon.com (cached raw under gitignored
``replays/usage/``), distills it to per-species top-K item/ability/move usage lists via
``data.usage_prior``, and freezes the small artifact into ``lategame/features/data/``.
A species count collapse means silent id-normalization drift, so this is a loud kill-gate:
it exits non-zero if too few species survive distillation.

    python scripts/build_usage_prior.py --month 2026-06                # gen9ou, 1500 cutoff
    python scripts/build_usage_prior.py --month 2026-06 --cutoff 1695
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from lategame.data.usage_prior import _usage_path, write_usage_prior

_UA = "lategame-research/0.1 (offline RL replay study)"
_TIMEOUT = 60.0
_MIN_SPECIES = 350  # gen9ou-1500 carries ~403; far fewer means normalization drift


def _fetch_chaos(url: str, cache: Path) -> dict:
    if cache.exists():
        print(f"[usage-prior] using cached raw {cache}")
        with cache.open() as f:
            return json.load(f)
    print(f"[usage-prior] fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 -- fixed https host
        raw = resp.read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw, encoding="utf-8")
    return json.loads(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", required=True, help="stats month, e.g. 2026-06")
    ap.add_argument("--format", dest="battle_format", default="gen9ou")
    ap.add_argument("--cutoff", type=int, default=1500, help="rating cutoff file to use")
    ap.add_argument("--url", default=None, help="override the chaos JSON url")
    ap.add_argument("--raw-cache", default=None, help="override the raw chaos cache path")
    ap.add_argument("--top-moves", type=int, default=10)
    ap.add_argument("--top-items", type=int, default=5)
    ap.add_argument("--top-abilities", type=int, default=3)
    ap.add_argument("--min-share", type=float, default=0.01)
    args = ap.parse_args()

    url = args.url or (
        f"https://www.smogon.com/stats/{args.month}/chaos/{args.battle_format}-{args.cutoff}.json"
    )
    cache = Path(
        args.raw_cache or f"replays/usage/{args.battle_format}-{args.cutoff}-{args.month}.json"
    )
    chaos = _fetch_chaos(url, cache)

    version, report = write_usage_prior(
        chaos,
        args.battle_format,
        month=args.month,
        cutoff=args.cutoff,
        source=url,
        top_moves=args.top_moves,
        top_items=args.top_items,
        top_abilities=args.top_abilities,
        min_share=args.min_share,
    )
    out = _usage_path(args.battle_format)
    kept = report["kept_species"]
    print(f"[usage-prior] kept {kept} species; skipped {len(report['skipped_species'])}")
    if report["skipped_species"]:
        print(f"  skipped: {', '.join(report['skipped_species'])}")
    print(f"  dropped out-of-vocab ids: {report['dropped_ids']}")
    print(f"  -> wrote {out} ({out.stat().st_size / 1024:.0f} KB, version {version})")

    if kept < _MIN_SPECIES:
        sys.exit(f"only {kept} species survived (< {_MIN_SPECIES}) -- id normalization drift?")


if __name__ == "__main__":
    main()

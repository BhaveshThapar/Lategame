"""Fetch + cache public Pokemon Showdown replays (plan.md, M6 data plan).

Thin, polite client over replay.pokemonshowdown.com's public JSON endpoints. It
pulls recent ``gen9randombattle`` ladder replays, keeps only high-skill ones
(rating >= a threshold), and caches each replay's raw JSON to disk so a re-run
never re-fetches. Stdlib only -- no new dependency -- and rate-limited by a small
sleep between requests to stay a good citizen of a free community service.

The downloaded JSON (``{id, format, players, rating, log, ...}``) is exactly what
``data.ingest`` consumes; fetching and ingesting are deliberately separate steps so
a large scrape happens once and ingestion can be re-run as the encoder evolves.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from lategame.config import DEFAULT_FORMAT

_BASE = "https://replay.pokemonshowdown.com"
_UA = "lategame-research/0.1 (offline RL replay study)"
_TIMEOUT = 20.0


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 -- fixed https host
        return json.loads(resp.read().decode("utf-8"))


def _rating_of(entry: object) -> int | None:
    """Best-effort rating from a search entry or a full replay dict."""
    if isinstance(entry, dict):
        rating = entry.get("rating")
        if isinstance(rating, (int, float)):
            return int(rating)
    return None


def search_ids(battle_format: str, page: int) -> list[dict]:
    """One page of the replay search index for ``battle_format`` (most recent first)."""
    url = f"{_BASE}/search.json?format={battle_format}&page={page}"
    result = _get_json(url)
    return [e for e in result if isinstance(e, dict)] if isinstance(result, list) else []


def fetch_replay(replay_id: str, cache_dir: Path) -> dict | None:
    """Return a replay's JSON, fetching + caching it if not already on disk."""
    path = cache_dir / f"{replay_id}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass  # corrupt cache -> re-fetch
    try:
        data = _get_json(f"{_BASE}/{replay_id}.json")
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None
    if not isinstance(data, dict) or "log" not in data:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_replays(
    battle_format: str = DEFAULT_FORMAT,
    min_rating: int = 1200,
    limit: int = 200,
    cache_dir: str | Path = "replays/gen9randombattle",
    max_pages: int = 80,
    sleep: float = 0.4,
    require_rating: bool = True,
) -> list[Path]:
    """Fetch up to ``limit`` rated replays, caching each; return the cached paths.

    Walks the search index page by page (the index is 1-indexed). The skill filter is
    what makes this *human demonstrations* rather than noise: rating-1000 floor games
    are mostly bots / brand-new accounts, so ``require_rating`` (default) keeps only
    replays whose ladder rating is known and ``>= min_rating``. Entries with no
    search-level rating are skipped without a network hit (polite); a present-but-low
    rating is skipped too. Already-cached ids are reused for free.

    Set ``require_rating=False`` to also keep unrated replays (rating unknown), useful
    only if the rated yield is too thin to assemble a batch.
    """
    cache = Path(cache_dir)
    kept: list[Path] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        if len(kept) >= limit:
            break
        try:
            entries = search_ids(battle_format, page)
        except (urllib.error.URLError, ValueError, TimeoutError):
            break
        if not entries:
            break
        for entry in entries:
            if len(kept) >= limit:
                break
            replay_id = entry.get("id")
            if not isinstance(replay_id, str) or replay_id in seen:
                continue
            seen.add(replay_id)
            search_rating = _rating_of(entry)
            if search_rating is not None and search_rating < min_rating:
                continue  # known-low: skip cheaply
            if require_rating and search_rating is None:
                continue  # unknown rating + require_rating: don't even fetch
            cached = (cache / f"{replay_id}.json").exists()
            data = fetch_replay(replay_id, cache)
            if not cached:
                time.sleep(sleep)  # only rate-limit actual network hits
            if data is None:
                continue
            rating = _rating_of(data)
            if rating is not None and rating < min_rating:
                continue
            if require_rating and rating is None:
                continue
            kept.append(cache / f"{replay_id}.json")
        print(f"page {page}: {len(kept)}/{limit} rated replays cached")

    print(f"fetched {len(kept)} replays (>= {min_rating}) to {cache}")
    return kept


def cached_replay_paths(cache_dir: str | Path = "replays/gen9randombattle") -> list[Path]:
    """All cached replay ``.json`` paths under ``cache_dir`` (for ingestion)."""
    return sorted(Path(cache_dir).glob("*.json"))


def iter_cached_replays(cache_dir: str | Path = "replays/gen9randombattle") -> Iterator[dict]:
    """Yield each cached replay's parsed JSON, skipping unreadable files."""
    for path in cached_replay_paths(cache_dir):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            yield data

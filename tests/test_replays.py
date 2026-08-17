"""No-network tests for the replay cache layer (M6).

The fetch path itself hits an external service, so it stays out of CI (an optional
manual smoke covers it). These exercise only the on-disk cache behaviour with
pre-written files, so they never touch the network.
"""

from __future__ import annotations

import json

from rotomai.data.replays import (
    cached_replay_paths,
    fetch_replay,
    iter_cached_replays,
)

_REPLAY = {
    "id": "gen9randombattle-123",
    "format": "gen9randombattle",
    "players": ["A", "B"],
    "log": "|gen|9",
}


def _write(cache_dir, replay) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{replay['id']}.json").write_text(json.dumps(replay), encoding="utf-8")


def test_fetch_replay_returns_cached_without_network(tmp_path) -> None:
    _write(tmp_path, _REPLAY)
    # If this tried the network it would fail offline; a cache hit returns instantly.
    data = fetch_replay("gen9randombattle-123", tmp_path)
    assert data is not None and data["id"] == "gen9randombattle-123"


def test_cached_paths_and_iter_skip_corrupt(tmp_path) -> None:
    _write(tmp_path, _REPLAY)
    _write(tmp_path, {**_REPLAY, "id": "gen9randombattle-456"})
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    paths = cached_replay_paths(tmp_path)
    assert len(paths) == 3  # includes broken.json on disk

    loaded = list(iter_cached_replays(tmp_path))
    ids = {d["id"] for d in loaded}
    assert ids == {"gen9randombattle-123", "gen9randombattle-456"}  # corrupt one skipped

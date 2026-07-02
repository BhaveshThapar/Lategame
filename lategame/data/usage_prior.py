"""Smogon usage-prior imputation of own-team kit (Build 4, plan.md 13.1).

Two-pass own-team completion (``data.ingest``) closes only the *revealed* part of the
own-team POV gap: public logs top out around item 0.30 / ability 0.47 / moves 2.78 per
own active mon, while the live ``|request|`` shows 0.89 / 1.00 / 4.00 -- and Build 3
proved a partial fix leaves the identity channels out-of-distribution and the agent
non-functional. This module supplies the missing detail: each species' standard
competitive kit from Smogon's published usage statistics, so ingestion can fill every
still-unrevealed own item/ability/move slot to eval-full density. For the *own* team
this approximates the truth the player actually had.

Mirrors ``features.embed_prior``: built once from a monthly chaos-stats JSON
(maintenance path, ``scripts/build_usage_prior.py``), distilled to per-species top-K
usage lists, frozen into a committed ``features/data/usage_<format>.json`` stamped with
``vocab_version`` (drift guard), and read back at ingest time with no network dependency.

Kit choice is **usage-weighted sampling, stably seeded** per (replay-POV, mon) -- not
deterministic top-1: the item slot is ~70% imputed, so top-1 would make (species -> item)
near-constant in training, re-introducing a distribution mismatch against eval teams that
run standard-but-not-always-modal kits. Sampling reproduces the ladder marginals the
replay corpus itself was drawn from, while staying byte-reproducible across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from poke_env import to_id_str

from lategame.features import vocab

_LOGGER = logging.getLogger("lategame.usage_prior")

# Chaos-stats id for an empty item slot. Kept in the artifact (it is real usage mass) but
# imputed as ``None``: the encoder maps both the unrevealed sentinel and a live no-item
# ``None`` to the same UNK id, so "imputed nothing" and "live no item" match exactly.
NO_ITEM = "nothing"

_USAGE_DIR = Path(vocab.__file__).with_name("data")
_FIELDS = ("moves", "items", "abilities")
_CHAOS_KEYS = {"moves": "Moves", "items": "Items", "abilities": "Abilities"}


def _usage_path(battle_format: str) -> Path:
    return _USAGE_DIR / f"usage_{to_id_str(battle_format)}.json"


@dataclass(frozen=True)
class SpeciesUsage:
    """One species' distilled usage distributions, usage-descending, weights summing to 1."""

    moves: tuple[tuple[str, float], ...]
    items: tuple[tuple[str, float], ...]
    abilities: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class UsagePrior:
    """A frozen usage-prior artifact keyed by ``to_id_str`` species."""

    version: str
    vocab_version: str
    source: str
    month: str
    cutoff: int
    battle_format: str
    species: dict[str, SpeciesUsage]


# --------------------------------------------------------------------------- #
# Build / regenerate (maintenance tool -- scripts/build_usage_prior.py).
# --------------------------------------------------------------------------- #
def _distill_field(
    raw: dict, field: str, table: dict[str, int], top_k: int, min_share: float
) -> tuple[list[list[str | float]], int]:
    """Distill one usage dict to a normalized top-K list; returns (pairs, dropped_ids)."""
    dropped = 0
    pairs: list[tuple[str, float]] = []
    for key, weight in raw.items():
        if not key or not isinstance(weight, (int, float)) or weight <= 0:
            continue  # "" marks an empty slot in chaos Moves; not a vocab miss
        kid = to_id_str(key)
        if kid not in table and not (field == "items" and kid == NO_ITEM):
            dropped += 1
            continue
        pairs.append((kid, float(weight)))
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return [], dropped
    kept = [(k, w / total) for k, w in pairs[:top_k] if w / total >= min_share]
    norm = sum(w for _, w in kept)
    out: list[list[str | float]] = [[k, round(w / norm, 6)] for k, w in kept]
    return out, dropped


def build_usage_prior(
    chaos: dict,
    battle_format: str = "gen9ou",
    *,
    month: str = "",
    cutoff: int = 0,
    source: str = "",
    top_moves: int = 10,
    top_items: int = 5,
    top_abilities: int = 3,
    min_share: float = 0.01,
) -> tuple[dict, dict]:
    """Distill a raw chaos-stats dict into the committed artifact; returns (artifact, report).

    Species keys are display-form in chaos ("Iron Valiant") -> normalized with ``to_id_str``
    to match poke-env ``mon.species``. Any id absent from the frozen vocab is dropped and
    counted (it would only ever encode to UNK); species absent from the vocab are skipped
    and reported -- both are the tripwires for silent normalization drift.
    """
    v = vocab.load_vocab()
    top_k = {"moves": top_moves, "items": top_items, "abilities": top_abilities}
    species_out: dict[str, dict] = {}
    skipped: list[str] = []
    dropped = dict.fromkeys(_FIELDS, 0)
    for name, entry in chaos["data"].items():
        sid = to_id_str(name)
        if sid not in v.tables["species"]:
            skipped.append(name)
            continue
        rec: dict[str, list] = {}
        for field in _FIELDS:
            pairs, n_dropped = _distill_field(
                entry.get(_CHAOS_KEYS[field], {}), field, v.tables[field],
                top_k[field], min_share,
            )
            rec[field] = pairs
            dropped[field] += n_dropped
        species_out[sid] = rec
    artifact = {
        "version": _content_hash(species_out),
        "vocab_version": v.version,
        "source": source,
        "month": month,
        "cutoff": cutoff,
        "battle_format": battle_format,
        "species": species_out,
    }
    report = {
        "kept_species": len(species_out),
        "skipped_species": skipped,
        "dropped_ids": dropped,
    }
    return artifact, report


def write_usage_prior(
    chaos: dict,
    battle_format: str = "gen9ou",
    path: Path | None = None,
    *,
    month: str = "",
    cutoff: int = 0,
    source: str = "",
    top_moves: int = 10,
    top_items: int = 5,
    top_abilities: int = 3,
    min_share: float = 0.01,
) -> tuple[str, dict]:
    """Build and freeze the artifact JSON; returns (version, report)."""
    artifact, report = build_usage_prior(
        chaos,
        battle_format,
        month=month,
        cutoff=cutoff,
        source=source,
        top_moves=top_moves,
        top_items=top_items,
        top_abilities=top_abilities,
        min_share=min_share,
    )
    out = path or _usage_path(battle_format)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(artifact, f)
        f.write("\n")
    load_usage_prior.cache_clear()
    return artifact["version"], report


def _content_hash(species: dict) -> str:
    return hashlib.sha256(json.dumps(species, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Load + sample (ingest path -- committed JSON only, no network).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_usage_prior(battle_format: str = "gen9ou", path: Path | None = None) -> UsagePrior | None:
    """Load the committed artifact for ``battle_format``, or ``None`` if none exists.

    Missing-artifact is a warning, not an error, so imputation can default ON without
    breaking formats that have no usage stats (gen9randombattle). A ``vocab_version``
    mismatch *is* an error: ids in the artifact would silently stop matching the encoder.
    """
    src = path or _usage_path(battle_format)
    if not src.exists():
        _LOGGER.warning(
            "no usage-prior artifact for %s (%s) -- imputation disabled", battle_format, src
        )
        return None
    with src.open() as f:
        raw = json.load(f)
    if str(raw["vocab_version"]) != vocab.vocab_version():
        raise ValueError(
            f"usage prior vocab_version {raw['vocab_version']} != live {vocab.vocab_version()} "
            "-- rebuild via scripts/build_usage_prior.py"
        )
    species = {
        sid: SpeciesUsage(
            moves=tuple((str(k), float(w)) for k, w in rec["moves"]),
            items=tuple((str(k), float(w)) for k, w in rec["items"]),
            abilities=tuple((str(k), float(w)) for k, w in rec["abilities"]),
        )
        for sid, rec in raw["species"].items()
    }
    return UsagePrior(
        version=str(raw["version"]),
        vocab_version=str(raw["vocab_version"]),
        source=str(raw["source"]),
        month=str(raw["month"]),
        cutoff=int(raw["cutoff"]),
        battle_format=str(raw["battle_format"]),
        species=species,
    )


def kit_seed(battle_tag: str, team_key: str) -> int:
    """Stable per-(replay-POV, mon) sampling seed -- blake2s, not ``hash()`` (PYTHONHASHSEED)."""
    digest = hashlib.blake2s(f"{battle_tag}|{team_key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _draw(rng: np.random.Generator, dist: tuple[tuple[str, float], ...]) -> str:
    weights = np.asarray([w for _, w in dist], dtype=np.float64)
    return dist[int(rng.choice(len(dist), p=weights / weights.sum()))][0]


def sample_kit(
    prior: UsagePrior,
    species: str,
    known_moves: set[str],
    need_item: bool,
    need_ability: bool,
    seed: int,
) -> tuple[list[str], str | None, str | None] | None:
    """Sample the unrevealed remainder of one mon's kit; ``None`` if the species is unknown.

    Moves: usage-weighted draws without replacement (renormalizing each draw) from the
    species' top list minus ``known_moves``, until the kit reaches 4 or candidates run out.
    Item/ability: a single weighted draw each, only when still needed; a drawn ``"nothing"``
    item comes back as ``None`` (see ``NO_ITEM``). Revealed truth is never touched here --
    the caller only fills what the log left unknown.
    """
    usage = prior.species.get(species)
    if usage is None:
        return None
    rng = np.random.default_rng(seed)
    moves: list[str] = []
    candidates = [(m, w) for m, w in usage.moves if m not in known_moves]
    while len(known_moves) + len(moves) < 4 and candidates:
        picked = _draw(rng, tuple(candidates))
        moves.append(picked)
        candidates = [(m, w) for m, w in candidates if m != picked]
    item: str | None = None
    if need_item and usage.items:
        drawn = _draw(rng, usage.items)
        item = None if drawn == NO_ITEM else drawn
    ability: str | None = None
    if need_ability and usage.abilities:
        ability = _draw(rng, usage.abilities)
    return moves, item, ability

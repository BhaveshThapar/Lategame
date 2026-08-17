"""Stable integer-ID vocabularies for species / move / item / ability (R-ENCODE).

The hand-crafted encoder (``features.encoder``) tells the policy a Pokemon's stats
and a move's power, but never *which* species/move/item/ability it is -- so the policy
cannot reason about identity (the ~32% imitation-accuracy bottleneck). These tables map
each poke-env id-str to a stable integer index for the model's learned embedding tables.

**Index 0 is reserved for ``<none/unk>``** in every table: an absent slot, an unrevealed
opponent detail, or an out-of-vocab id all map to 0. The vocab is built once from poke-env
Gen 9 ``GenData`` (species/moves/abilities) plus the vendored Showdown items dex, then
frozen into a committed JSON (``data/vocab_gen9.json``) and stamped (``vocab_version``)
into shards/checkpoints so the mapping can never silently drift. The runtime path only
loads the JSON -- no ``GenData`` or node dependency.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from poke_env import to_id_str
from poke_env.battle import Move, Pokemon

UNK = 0  # reserved index: <none> / unrevealed / out-of-vocab.
_UNKNOWN_ITEM = "unknown_item"  # == poke_env.data.GenData.UNKNOWN_ITEM

CATEGORIES = ("species", "moves", "items", "abilities")
_VOCAB_PATH = Path(__file__).with_name("data") / "vocab_gen9.json"
_DEFAULT_SHOWDOWN = "third_party/pokemon-showdown"


@dataclass(frozen=True)
class Vocab:
    """Frozen id-str -> index maps (index >= 1; 0 is ``UNK``) plus a content version."""

    version: str
    tables: dict[str, dict[str, int]]

    def index(self, category: str, key: str | None) -> int:
        if not key:
            return UNK
        return self.tables[category].get(key, UNK)

    def size(self, category: str) -> int:
        """Embedding rows for ``category`` (entries + 1 for the shared ``UNK`` row)."""
        return len(self.tables[category]) + 1


@lru_cache(maxsize=1)
def load_vocab() -> Vocab:
    """Load the committed vocab JSON once. Position ``i`` in each list -> index ``i + 1``."""
    with _VOCAB_PATH.open() as f:
        raw = json.load(f)
    tables = {cat: {k: i + 1 for i, k in enumerate(raw[cat])} for cat in CATEGORIES}
    return Vocab(version=str(raw["vocab_version"]), tables=tables)


def vocab_version() -> str:
    return load_vocab().version


def vocab_sizes() -> dict[str, int]:
    """Embedding-table sizes keyed by category, for the model's ``nn.Embedding`` rows."""
    vocab = load_vocab()
    return {cat: vocab.size(cat) for cat in CATEGORIES}


# --------------------------------------------------------------------------- #
# Per-entity lookups (used by the encoder; hot path -> cached dict lookups).
# --------------------------------------------------------------------------- #
def species_id(mon: Pokemon | None) -> int:
    if mon is None:
        return UNK
    return load_vocab().index("species", mon.species)


def move_id(move: Move | None) -> int:
    if move is None:
        return UNK
    return load_vocab().index("moves", move.id)


def item_id(mon: Pokemon | None) -> int:
    if mon is None or not mon.item or mon.item == _UNKNOWN_ITEM:
        return UNK
    return load_vocab().index("items", mon.item)


def ability_id(mon: Pokemon | None) -> int:
    if mon is None or not mon.ability:
        return UNK
    return load_vocab().index("abilities", mon.ability)


# --------------------------------------------------------------------------- #
# Build / regenerate (maintenance tool -- needs GenData + node + the built dist).
# --------------------------------------------------------------------------- #
def build_vocab(showdown_dir: str | Path = _DEFAULT_SHOWDOWN, node: str = "node") -> dict:
    """Rebuild the vocab dict from poke-env GenData + the vendored items dex.

    Not on any runtime path: ``load_vocab`` reads the committed JSON this produces.
    """
    from poke_env.data import GenData

    gd = GenData.from_gen(9)
    species = sorted(gd.pokedex.keys())
    moves = sorted(gd.moves.keys())
    abilities = sorted(
        {
            to_id_str(a)
            for entry in gd.pokedex.values()
            for a in (entry.get("abilities") or {}).values()
            if a
        }
    )
    items = sorted(_extract_items(showdown_dir, node))
    return {
        "vocab_version": _content_hash(species, moves, items, abilities),
        "species": species,
        "moves": moves,
        "items": items,
        "abilities": abilities,
    }


def write_vocab(
    path: Path = _VOCAB_PATH,
    showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
    node: str = "node",
) -> str:
    """Build and write the vocab JSON; returns the new ``vocab_version``."""
    vocab = build_vocab(showdown_dir=showdown_dir, node=node)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(vocab, f)
        f.write("\n")
    load_vocab.cache_clear()
    return vocab["vocab_version"]


def _extract_items(showdown_dir: str | Path, node: str) -> list[str]:
    dist = Path(showdown_dir) / "dist" / "data" / "items.js"
    if not dist.exists():
        raise RuntimeError(f"items dex missing: {dist} -- run `bash scripts/setup_server.sh`")
    script = (
        "const {Items}=require(process.argv[1]);"
        "console.log(JSON.stringify(Object.keys(Items).filter(k=>Items[k]&&Items[k].name)));"
    )
    proc = subprocess.run(
        [node, "-e", script, str(dist.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    return [to_id_str(k) for k in json.loads(proc.stdout)]


def _content_hash(*lists: list[str]) -> str:
    digest = hashlib.sha256(json.dumps(lists, sort_keys=True).encode()).hexdigest()
    return digest[:12]

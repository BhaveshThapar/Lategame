"""Vocab tables: stable id->index maps with index 0 reserved for <none/unk>.

Node-free: exercises only the committed ``vocab_gen9.json`` via ``load_vocab`` and the
per-entity lookups. The (re)build path (``build_vocab``) needs node + the Showdown dist
and is not covered here.
"""

from __future__ import annotations

import json

from poke_env.battle import Move, Pokemon

from rotomai.features import vocab


def test_load_vocab_version_matches_content_hash() -> None:
    with vocab._VOCAB_PATH.open() as f:
        raw = json.load(f)
    recomputed = vocab._content_hash(
        raw["species"], raw["moves"], raw["items"], raw["abilities"]
    )
    assert raw["vocab_version"] == recomputed
    assert vocab.vocab_version() == recomputed


def test_indices_are_dense_unique_and_skip_zero() -> None:
    v = vocab.load_vocab()
    for cat in vocab.CATEGORIES:
        idxs = sorted(v.tables[cat].values())
        assert idxs[0] == 1, f"{cat} must reserve index 0 for UNK"
        assert idxs == list(range(1, len(idxs) + 1)), f"{cat} indices not dense"
        assert v.size(cat) == len(idxs) + 1


def test_unknown_and_empty_map_to_unk() -> None:
    v = vocab.load_vocab()
    assert vocab.UNK == 0
    assert v.index("species", "definitely-not-a-mon") == 0
    assert v.index("moves", None) == 0
    assert v.index("items", "") == 0


def test_entity_lookups_handle_none_and_unknown_item() -> None:
    assert vocab.species_id(None) == 0
    assert vocab.move_id(None) == 0
    assert vocab.item_id(None) == 0
    assert vocab.ability_id(None) == 0

    mon = Pokemon(gen=9, species="Great Tusk")
    assert vocab.species_id(mon) == vocab.load_vocab().index("species", "greattusk") > 0
    # Freshly built mon has an unrevealed item sentinel -> UNK, not a real index.
    assert mon.item == vocab._UNKNOWN_ITEM
    assert vocab.item_id(mon) == 0

    assert vocab.move_id(Move("uturn", gen=9)) == vocab.load_vocab().index("moves", "uturn") > 0

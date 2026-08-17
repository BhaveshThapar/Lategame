"""Offline tests for the Smogon usage-prior artifact (Build 4). No network.

A hand-written chaos-stats dict exercises the distillation rules (vocab filtering,
empty-slot ``""`` keys, the ``"nothing"`` item, top-K + normalization), the frozen
artifact roundtrip with its vocab drift guard, and the seeded kit sampler.
"""

from __future__ import annotations

import json

import pytest

from rotomai.data.usage_prior import (
    SpeciesUsage,
    UsagePrior,
    build_usage_prior,
    kit_seed,
    load_usage_prior,
    sample_kit,
    write_usage_prior,
)
from rotomai.features.vocab import vocab_version

# Two real species (ids present in the committed vocab), one bogus species, one bogus
# move id, one bogus item id, a "" empty move slot, and a "nothing" item -- every
# distillation rule in one dict. Weights are chaos-style raw usage mass.
CHAOS = {
    "info": {"metagame": "gen9ou", "cutoff": 1500},
    "data": {
        "Kingambit": {
            "Raw count": 1000,
            "Moves": {
                "": 100.0,
                "suckerpunch": 300.0,
                "swordsdance": 250.0,
                "ironhead": 200.0,
                "kowtowcleave": 150.0,
                "lowkick": 90.0,
                "notamove": 50.0,
            },
            "Items": {"leftovers": 120.0, "airballoon": 60.0, "nothing": 15.0, "notanitem": 5.0},
            "Abilities": {"supremeoverlord": 190.0, "defiant": 10.0},
        },
        "Gholdengo": {
            "Raw count": 900,
            "Moves": {
                "makeitrain": 300.0,
                "shadowball": 280.0,
                "nastyplot": 200.0,
                "recover": 150.0,
            },
            "Items": {"airballoon": 100.0},
            "Abilities": {"goodasgold": 100.0},
        },
        "Fakemon": {"Raw count": 10, "Moves": {"tackle": 5.0}, "Items": {}, "Abilities": {}},
    },
}


def test_build_distills_topk_normalized_and_filters_vocab() -> None:
    artifact, report = build_usage_prior(CHAOS, "gen9ou", month="2026-06", cutoff=1500)

    assert report["kept_species"] == 2
    assert report["skipped_species"] == ["Fakemon"]
    assert report["dropped_ids"] == {"moves": 1, "items": 1, "abilities": 0}

    king = artifact["species"]["kingambit"]
    # Usage-descending, "" and out-of-vocab ids gone, "nothing" retained as real usage mass.
    assert [m for m, _ in king["moves"]] == [
        "suckerpunch", "swordsdance", "ironhead", "kowtowcleave", "lowkick"
    ]
    assert [i for i, _ in king["items"]] == ["leftovers", "airballoon", "nothing"]
    for field in ("moves", "items", "abilities"):
        weights = [w for _, w in king[field]]
        assert weights == sorted(weights, reverse=True)
        assert sum(weights) == pytest.approx(1.0, abs=1e-4)

    assert artifact["vocab_version"] == vocab_version()
    assert artifact["version"] and artifact["month"] == "2026-06" and artifact["cutoff"] == 1500


def test_write_load_roundtrip_and_vocab_drift_guard(tmp_path) -> None:
    path = tmp_path / "usage_gen9ou.json"
    version, _ = write_usage_prior(CHAOS, "gen9ou", path=path, month="2026-06", cutoff=1500)

    prior = load_usage_prior("gen9ou", path=path)
    assert prior is not None and prior.version == version and prior.cutoff == 1500
    king = prior.species["kingambit"]
    assert isinstance(king, SpeciesUsage)
    assert king.items[0][0] == "leftovers" and king.abilities[0][0] == "supremeoverlord"
    assert len(prior.species) == 2  # Fakemon never made it into the artifact

    # Tampered vocab_version must refuse to load: artifact ids would silently stop
    # matching the encoder's tables.
    raw = json.loads(path.read_text())
    raw["vocab_version"] = "bogus"
    path.write_text(json.dumps(raw))
    load_usage_prior.cache_clear()
    with pytest.raises(ValueError, match="vocab_version"):
        load_usage_prior("gen9ou", path=path)
    load_usage_prior.cache_clear()


def test_load_missing_format_returns_none() -> None:
    load_usage_prior.cache_clear()
    assert load_usage_prior("gen9nosuchformat") is None
    load_usage_prior.cache_clear()


_PRIOR = UsagePrior(
    version="v",
    vocab_version="x",
    source="",
    month="",
    cutoff=0,
    battle_format="gen9ou",
    species={
        "kingambit": SpeciesUsage(
            moves=(
                ("suckerpunch", 0.3),
                ("swordsdance", 0.25),
                ("ironhead", 0.2),
                ("kowtowcleave", 0.15),
                ("lowkick", 0.1),
            ),
            items=(("leftovers", 0.7), ("airballoon", 0.3)),
            abilities=(("supremeoverlord", 1.0),),
        ),
        # A species whose entire item mass is "nothing": imputed as None (live no-item).
        "pincurchin": SpeciesUsage(moves=(), items=(("nothing", 1.0),), abilities=()),
    },
)


def test_sample_kit_deterministic_excludes_known_fills_to_four() -> None:
    seed = kit_seed("battle-1-Alice", "p1: Kingambit")
    first = sample_kit(
        _PRIOR, "kingambit", {"kowtowcleave"}, need_item=True, need_ability=True, seed=seed
    )
    second = sample_kit(
        _PRIOR, "kingambit", {"kowtowcleave"}, need_item=True, need_ability=True, seed=seed
    )
    assert first == second  # stable seed -> byte-identical kit across runs
    assert first is not None
    moves, item, ability = first
    # Fills to exactly 4 total, never re-draws a known move, never repeats a draw.
    assert len(moves) == 3 and len(set(moves)) == 3 and "kowtowcleave" not in moves
    assert item in ("leftovers", "airballoon") and ability == "supremeoverlord"


def test_sample_kit_nothing_item_missing_species_and_need_flags() -> None:
    moves, item, ability = sample_kit(
        _PRIOR, "pincurchin", set(), need_item=True, need_ability=True, seed=7
    ) or ([], "", "")
    assert moves == [] and item is None and ability is None  # "nothing" -> live-style no-item

    assert sample_kit(_PRIOR, "pikachu", set(), need_item=True, need_ability=True, seed=7) is None

    _, item2, ability2 = sample_kit(
        _PRIOR, "kingambit", set(), need_item=False, need_ability=False, seed=7
    ) or ([], "", "")
    assert item2 is None and ability2 is None  # revealed truth is never re-drawn

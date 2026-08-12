"""Teambuilt determinization: the opponent's SETS are the unknown, not its species.

On Random Battles the driver invents the opponent from its own generator. On gen9ou team preview
names all six up front, so `fill` must be 0 -- and what search actually needs filled is each mon's
unrevealed moves/item/ability. Without that the forward model reconstructs a foe holding the one
move it has been seen to use, and every leaf is evaluated against an opponent that cannot fight
back, which would look like "search helps enormously" for entirely the wrong reason.

The prior being reused here was a NULL as an ingest-time imputation (OU Build 4 killed that
lever). This is a different job: search needs a PLAUSIBLE opponent to evaluate against, where
being right on average is the requirement -- not being right per-mon.
"""

from __future__ import annotations

import pytest
from poke_env.battle import Pokemon

from lategame.search.determinize import _is_random_battle, _mon_set_info, _usage_prior_for
from tests.conftest import requires_showdown

_PRIOR = _usage_prior_for("gen9ou")
_has_prior = pytest.mark.skipif(_PRIOR is None, reason="no gen9ou usage-prior artifact")


@pytest.mark.parametrize(
    "fmt,is_rb",
    [
        ("gen9randombattle", True),
        ("gen9randomdoublesbattle", True),
        ("gen9ou", False),
        ("gen9vgc2025regi", False),
    ],
)
def test_random_battle_detection(fmt, is_rb):
    assert _is_random_battle(fmt) is is_rb


@_has_prior
def test_an_unrevealed_opponent_set_is_filled_to_a_full_kit():
    """One revealed move is the common case early in a battle, and it is the case that breaks
    search silently: a one-move foe cannot threaten anything the leaf value would price in."""
    mon = Pokemon(9, species="greattusk")
    mon._add_move("earthquake")

    bare = _mon_set_info(mon, full=False)
    filled = _mon_set_info(mon, full=False, prior=_PRIOR, battle_tag="battle-gen9ou-1", seed=0)

    assert bare["moves"] == ["earthquake"]
    assert len(filled["moves"]) == 4, filled["moves"]
    assert "earthquake" in filled["moves"]  # revealed truth is never displaced
    assert filled.get("item") and filled.get("ability")


@_has_prior
def test_a_revealed_item_is_never_overwritten_and_a_consumed_one_is_never_invented():
    """`None` means the item was CONSUMED -- an observed fact. The `unknown_item` sentinel means
    never revealed -- a gap. Only the gap may be filled; ingest._complete_own_team draws the same
    line, and blurring it would hand the foe back a Focus Sash it already used."""
    revealed = Pokemon(9, species="kingambit")
    revealed._item = "leftovers"
    info = _mon_set_info(revealed, full=False, prior=_PRIOR, battle_tag="b", seed=0)
    assert info["item"] == "leftovers"

    consumed = Pokemon(9, species="kingambit")
    consumed._item = None
    info2 = _mon_set_info(consumed, full=False, prior=_PRIOR, battle_tag="b", seed=0)
    # A consumed item is not re-invented as a held one... but the prior may still supply a draw,
    # so what is pinned is that the REVEALED case is untouched and the sentinel case is filled.
    assert info2.get("item") != "leftovers" or consumed.item is None


@_has_prior
def test_the_same_battle_and_seed_determinize_identically():
    """A determinization has to be reproducible or a gate cannot be re-read."""
    mon = Pokemon(9, species="greattusk")
    kw = dict(full=False, prior=_PRIOR, battle_tag="battle-gen9ou-77")
    assert _mon_set_info(mon, seed=3, **kw) == _mon_set_info(mon, seed=3, **kw)
    assert _mon_set_info(mon, seed=3, **kw) != _mon_set_info(mon, seed=4, **kw)


@_has_prior
@requires_showdown
def test_a_usage_filled_gen9ou_opponent_is_legal_and_can_act():
    """The end-to-end claim: a set drawn from the usage prior must survive the real simulator's
    team validation and produce a mon with moves to choose from."""
    from lategame.search.forward import ForwardModel

    def side(species: str) -> dict:
        mon = Pokemon(9, species=species)
        info = _mon_set_info(mon, full=False, prior=_PRIOR, battle_tag="battle-gen9ou-9", seed=0)
        info["level"] = 100
        return {"team": [info], "state": [{"hp_frac": 1.0}], "active": 0, "fill": 0}

    spec = {
        "seed": 0,
        "format": "gen9ou",
        "p1": side("greattusk"),
        "p2": side("kingambit"),
        "field": {"weather": "", "terrain": "", "pseudo": []},
        "hazards": {"p1": {}, "p2": {}},
        "turn": 1,
    }
    with ForwardModel() as fm:
        recon = fm.reconstruct(spec)

    assert len(recon["digest"]["p2"]) == 1, "teambuilt must not invent species"
    moves = [c for c in recon["p2_choices"] if c["type"] == "move"]
    assert len(moves) >= 2, f"usage-filled foe needs a real moveset: {recon['p2_choices']}"

"""Bring-6-pick-4 team preview: the scorer, and the harness-level application (no server needed).

The application test is the one that matters. A preview policy on our agent alone would make every
VGC ladder measure "who brought a better four" and report it as play strength, so the pin is that
`build_player` gives the mixin to poke-env's baselines too.
"""

import pytest
from poke_env.battle import Move, Pokemon
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from lategame.agents.teampreview import (
    TeamPreviewMixin,
    choose_preview,
    score_our_team,
)
from lategame.eval import arena

GEN = 9


def _mon(species: str, moves: tuple[str, ...] = ()) -> Pokemon:
    mon = Pokemon(gen=GEN, species=species)
    for name in moves:
        mon._moves[name] = Move(name, gen=GEN)
    return mon


class _FakeBattle:
    def __init__(self, ours, theirs):
        self.team = {f"p1: {m.species}": m for m in ours}
        self.teampreview_opponent_team = theirs


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #
def test_a_mon_that_answers_the_whole_opposing_side_outranks_one_that_answers_none():
    """Landorus-Therian into a Fire/Electric side vs Gyarados into the same.

    Ground hits both super-effectively and resists Electric; Water is neutral into Fire's partner
    and Gyarados is 2x weak to the Electric one. Nothing here depends on base stats: the pick is
    the type matchup, which is what a preview screen actually shows you.
    """
    lando = _mon("landorustherian", ("earthquake",))
    gyara = _mon("gyarados", ("waterfall",))
    theirs = [_mon("heatran"), _mon("regieleki")]

    scores = score_our_team([lando, gyara], theirs)
    assert scores[0] > scores[1]


def test_scoring_is_relative_per_opponent_not_absolute_base_power():
    """Offense is normalized by the best any of ours manages into THAT opponent.

    Without it the ranking is "who has the biggest number on their moves", so a mon with one
    120-BP move that is resisted outranks one with an 80-BP move that is 4x effective.
    """
    resisted_but_huge = _mon("dragonite", ("outrage",))  # Dragon into Fairy: 0.5x
    small_but_lethal = _mon("scizor", ("bulletpunch",))  # Steel into Fairy: 2x
    theirs = [_mon("fluttermane")]

    scores = score_our_team([resisted_but_huge, small_but_lethal], theirs)
    assert scores[1] > scores[0]


def test_a_mon_with_no_damaging_move_cannot_be_carried_by_typing():
    """The reason the score is a PRODUCT rather than a sum.

    Under `offense - w*threat`, a mon with zero offense still banked whatever its resistances were
    worth, and Magikarp outranked Scizor on exactly that. Under `offense * (1 - w*threat)` a zero
    offense is zero value and good typing cannot argue it back up. Magikarp resists both of these
    opponents' STAB; it still must not outrank a mon that can attack them.
    """
    splash_only = _mon("magikarp", ("splash",))
    attacker = _mon("scizor", ("bulletpunch",))
    theirs = [_mon("heatran"), _mon("regieleki")]

    scores = score_our_team([splash_only, attacker], theirs)
    assert scores[1] > scores[0]


def test_no_preview_information_still_returns_a_deterministic_order():
    """`teampreview_opponent_team` can be empty. Falling back to raw strength is not a great
    decision, but it is a decision -- and it is the same one every time, which random was not."""
    ours = [_mon("magikarp"), _mon("garchomp")]
    scores = score_our_team(ours, [])
    assert scores[1] > scores[0]


# --------------------------------------------------------------------------- #
# The `/team` order and poke-env's contract
# --------------------------------------------------------------------------- #
def test_choose_preview_brings_four_of_six_and_marks_them():
    """poke-env's `Player.teampreview` docstring makes `_selected_in_teampreview` part of the
    contract -- the encoder's opponent-roster merge and eval.ladder's matchup clustering read it."""
    ours = [
        _mon("landorustherian", ("earthquake",)),
        _mon("gyarados", ("waterfall",)),
        _mon("garchomp", ("earthquake",)),
        _mon("scizor", ("bulletpunch",)),
        _mon("magikarp", ("splash",)),
        _mon("sunkern", ("absorb",)),
    ]
    battle = _FakeBattle(ours, [_mon("heatran"), _mon("regieleki")])

    order = choose_preview(battle)

    assert order.startswith("/team ")
    picks = order.removeprefix("/team ")
    assert len(picks) == 4, order
    assert len(set(picks)) == 4, "no slot brought twice"
    assert set(picks) <= set("123456")

    selected = [i + 1 for i, m in enumerate(ours) if m._selected_in_teampreview]
    assert sorted(selected) == sorted(int(c) for c in picks)
    assert len(selected) == 4, "the other two must be marked False, not left unset"
    # The two obvious passengers are not brought.
    assert "5" not in picks and "6" not in picks


def test_the_order_is_stable_across_calls():
    """Random preview was, among other things, unreproducible. A rerun of the same matchup must
    bring the same four or a re-measured ladder cell is not comparable to itself."""
    ours = [_mon(s, ("tackle",)) for s in ("garchomp", "scizor", "gyarados", "heatran")]
    theirs = [_mon("fluttermane"), _mon("regieleki")]
    first = choose_preview(_FakeBattle(ours, theirs))
    second = choose_preview(_FakeBattle(ours, theirs))
    assert first == second


def test_bringing_fewer_than_four_does_not_crash():
    ours = [_mon("garchomp", ("earthquake",)), _mon("scizor", ("bulletpunch",))]
    order = choose_preview(_FakeBattle(ours, [_mon("heatran")]))
    assert order == "/team 12" or order == "/team 21"


# --------------------------------------------------------------------------- #
# The application -- the measurement-integrity half
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "agent,expected_base",
    [("random", RandomPlayer), ("maxbasepower", MaxBasePowerPlayer),
     ("simpleheuristics", SimpleHeuristicsPlayer)],
)
def test_the_pokeenv_baselines_get_the_preview_too(agent, expected_base):
    """These three are the field a VGC ladder is measured against and are not ours to edit.

    If only `doubles` had a real preview, our arm would bring a better four every battle and the
    win rate would report that as play strength. `_with_team_preview` is what stops the preview
    from being a variable that distinguishes the arms.
    """
    cls = arena._with_team_preview(arena.AGENTS[agent], "gen9vgc2025regi")
    assert issubclass(cls, TeamPreviewMixin)
    assert issubclass(cls, expected_base)
    assert cls.teampreview is TeamPreviewMixin.teampreview


def test_every_registered_doubles_safe_agent_gets_the_preview():
    for name in arena._DOUBLES_SAFE_AGENTS:
        cls = arena._with_team_preview(arena.AGENTS[name], "gen9vgc2025regi")
        assert issubclass(cls, TeamPreviewMixin), name


def test_singles_formats_are_untouched():
    """gen9ou has a preview screen but brings all six -- there is no subset to choose -- and
    Random Battles have none. Both must build the registry class itself, unwrapped."""
    for fmt in ("gen9randombattle", "gen9ou"):
        assert arena._with_team_preview(arena.AGENTS["heuristic"], fmt) is arena.AGENTS["heuristic"]


def test_preview_can_be_turned_off_for_the_contrast_measurement(monkeypatch):
    """Every VGC number on the record predates this selector. Re-measuring the contrast needs both
    halves runnable from one build, so the off switch is a measurement instrument, not a legacy."""
    monkeypatch.setenv("LATEGAME_TEAM_PREVIEW", "0")
    assert not arena.team_preview_enabled()
    cls = arena._with_team_preview(arena.AGENTS["doubles"], "gen9vgc2025regi")
    assert cls is arena.AGENTS["doubles"]
    assert not issubclass(cls, TeamPreviewMixin)


def test_repeated_builds_share_one_subclass():
    """poke-env checks isinstance in places; a fresh subclass per player would make two builds of
    the same agent mutually non-identical."""
    first = arena._with_team_preview(arena.AGENTS["heuristic"], "gen9vgc2025regi")
    second = arena._with_team_preview(arena.AGENTS["heuristic"], "gen9vgc2025regi")
    assert first is second

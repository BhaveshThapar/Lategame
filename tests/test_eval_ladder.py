"""The eval ladder: the joint fit, and the three ways it could be silently wrong.

Every test here guards a property that a plausible-looking implementation gets wrong.

  * The fit is a MAXIMUM-LIKELIHOOD fit, not a Glicko period. Its defining property is that the
    Bradley-Terry score equation holds at the solution, which for two agents has a closed form to
    check against. Checking it against `rate_win_rate` instead would be checking it against a
    DIFFERENT estimator -- see `test_two_agent_fit_is_not_rate_win_rate`.
  * The undamped iteration OSCILLATES rather than converging, and does so worst at two agents.
    Nothing about the code looks wrong when it happens; the ratings just walk to the clamp.
  * A round-robin identifies rating DIFFERENCES only, so the anchor is load-bearing. Without a
    gauge fix there is no unique answer to converge to.

No server is needed anywhere in this file: `rate_field` takes a matrix, and the matrix is a
dataclass. The one test that exercises `run_round_robin` drives it with fakes.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from lategame.eval import ladder as L
from lategame.eval.ladder import (
    DEFAULT_FIELD,
    FieldEntry,
    LadderError,
    PairResult,
    format_table,
    parse_field,
    rate_field,
    summarize_ladder,
    write_results,
)
from lategame.eval.rating import DEFAULT_RATING, MAX_RD, Rating, expected_score, g


def _round_robin(true_ratings: dict[str, float], n: int) -> list[PairResult]:
    """A noiseless round-robin generated FROM known ratings, for recovery tests."""
    labels = list(true_ratings)
    pairs = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            e = expected_score(Rating(true_ratings[a], MAX_RD), Rating(true_ratings[b], MAX_RD))
            wins = round(e * n)
            pairs.append(PairResult(a=a, b=b, wins=wins, losses=n - wins, ties=0, games=n))
    return pairs


# --------------------------------------------------------------------------- #
# Parsing the field
# --------------------------------------------------------------------------- #
def test_parses_names_and_checkpoints():
    entries = parse_field("heuristic, offrl@checkpoints/ppo_v25b_s0/iter_160.pt")
    assert entries == [
        FieldEntry("heuristic", None),
        FieldEntry("offrl", "checkpoints/ppo_v25b_s0/iter_160.pt"),
    ]


@pytest.mark.parametrize(
    "spec,label",
    [
        ("heuristic", "heuristic"),
        ("offrl@checkpoints/ppo_v25b_s0/iter_160.pt", "offrl@ppo_v25b_s0/iter_160"),
        # A checkpoint sitting directly in checkpoints/ has no meaningful parent to name.
        ("offrl@checkpoints/offrl_gen9ou_wide_s0.pt", "offrl@offrl_gen9ou_wide_s0"),
    ],
)
def test_label_identifies_the_checkpoint_not_just_the_agent(spec, label):
    assert parse_field(f"{spec},random")[0].label == label


def test_checkpoints_of_one_agent_do_not_collide():
    """The default field holds four `offrl` entries; if labels collapsed to the agent name the
    matrix would silently overwrite itself and the fit would rate a phantom."""
    labels = [e.label for e in parse_field(",".join(DEFAULT_FIELD))]
    assert len(labels) == len(set(labels)) == len(DEFAULT_FIELD)


def test_ppo_checkpoint_is_refused_because_it_forces_sampling():
    """`ppo` is the training rollout agent and always samples, so `ppo@ckpt` measures a weaker
    policy than the one every published number is about -- while looking exactly like the right
    thing to type. Refused, not documented."""
    with pytest.raises(LadderError, match="TRAINING rollout agent"):
        parse_field("heuristic,ppo@checkpoints/ppo_v25b_s0/iter_160.pt")
    # The bare name is still fine: with no checkpoint it loads its own default, and the refusal is
    # about silently rating a NAMED checkpoint through the wrong policy.
    assert parse_field("heuristic,ppo")[1].agent == "ppo"


def test_default_field_scores_ppo_checkpoints_through_offrl():
    """The counterpart of the refusal above: the shipped default must itself take the advice."""
    entries = parse_field(",".join(DEFAULT_FIELD))
    ppo_builds = [e for e in entries if e.checkpoint and "ppo_v" in e.checkpoint]
    assert ppo_builds and all(e.agent == "offrl" for e in ppo_builds)


def test_rejects_duplicates_and_undersized_fields():
    with pytest.raises(LadderError, match="duplicate"):
        parse_field("heuristic,random,heuristic")
    with pytest.raises(LadderError, match="at least 2"):
        parse_field("heuristic")
    with pytest.raises(LadderError, match="no checkpoint"):
        parse_field("ppo@,random")


def test_default_field_is_parseable_and_holds_the_anchor():
    entries = parse_field(",".join(DEFAULT_FIELD))
    assert L.DEFAULT_ANCHOR in {e.label for e in entries}


def test_default_field_excludes_in_flight_v26():
    """Build 26's verdict comes from the pre-registered gate, not from a rating table. A v26 arm
    in the DEFAULT field would put a mid-flight number in front of readers by accident."""
    assert not any("v26" in entry for entry in DEFAULT_FIELD)


# --------------------------------------------------------------------------- #
# The fit: what makes it a fit
# --------------------------------------------------------------------------- #
def test_two_agent_fit_matches_the_closed_form():
    """With two agents the score equation inverts exactly: the rating gap that reproduces a score
    of s is -400/g(350) * log10(1/s - 1). This is the fit's definition, checked in closed form."""
    fit = rate_field([PairResult("a", "b", wins=75, losses=25, ties=0, games=100)], anchor="b")
    expected = -400.0 / g(MAX_RD) * math.log10(1.0 / 0.75 - 1.0)
    assert fit.ratings["a"].rating - fit.ratings["b"].rating == pytest.approx(expected, abs=1e-6)
    assert fit.converged


def test_score_equation_holds_for_every_agent():
    """The general form of the property above: at the solution each agent's observed score equals
    its expected score summed over the field."""
    pairs = _round_robin({"w": 1300.0, "x": 1450.0, "y": 1500.0, "z": 1650.0, "q": 1800.0}, 4000)
    fit = rate_field(pairs, anchor="y")
    games = L._games_by_label(pairs)
    for label, played in games.items():
        residual = sum(
            score
            - expected_score(
                Rating(fit.ratings[label].rating, MAX_RD),
                Rating(fit.ratings[opponent].rating, MAX_RD),
            )
            for opponent, score in played
        )
        assert residual == pytest.approx(0.0, abs=1e-3), label


def test_recovers_known_ratings():
    true = {"w": 1300.0, "x": 1450.0, "y": 1500.0, "z": 1650.0, "q": 1800.0}
    fit = rate_field(_round_robin(true, 4000), anchor="y")
    assert fit.converged
    for label, value in true.items():
        assert fit.ratings[label].rating == pytest.approx(value, abs=1.0), label


def test_rating_reflects_opposition_quality_not_just_score_rate():
    """The reason this module iterates at all, and the property one Glicko period cannot have.

    Hold every opponent at the reference -- what a single period over a fresh field does -- and each
    rating becomes a monotone function of that agent's own score rate, carrying nothing the score
    rate did not. Here `x` and `z` post the SAME 0.900 score over the same number of games, but x's
    came against the weak agent and z's against the strong one, so the fit must separate them.

    Note the schedule is not a complete round-robin: `rate_field` takes any connected set of pairs.
    """
    pairs = [
        PairResult("x", "weak", wins=90, losses=10, ties=0, games=100),
        PairResult("z", "strong", wins=90, losses=10, ties=0, games=100),
        PairResult("weak", "strong", wins=10, losses=90, ties=0, games=100),
    ]
    fit = rate_field(pairs, anchor="weak")
    games = L._games_by_label(pairs)
    rate = {k: sum(s for _, s in v) / len(v) for k, v in games.items()}
    assert rate["x"] == pytest.approx(rate["z"]) == pytest.approx(0.9)
    assert fit.ratings["z"].rating > fit.ratings["x"].rating


def test_a_pure_cycle_rates_flat():
    """A stated limitation, pinned so it is never mistaken for a bug.

    Bradley-Terry models a single latent strength per agent, so it cannot represent
    NON-TRANSITIVITY: in a perfect rock-paper-scissors cycle every agent is equally good by the
    model's own lights, and the fit says so by rating them all equal. Pokemon matchups carry some
    genuine non-transitivity (team archetypes counter each other), so a cluster of near-equal
    ratings can mean "cyclic", not only "equal in strength" -- read the matrix, not just the table.
    """
    pairs = [
        PairResult("rock", "scissors", wins=100, losses=0, ties=0, games=100),
        PairResult("scissors", "paper", wins=100, losses=0, ties=0, games=100),
        PairResult("paper", "rock", wins=100, losses=0, ties=0, games=100),
    ]
    fit = rate_field(pairs, anchor="rock")
    values = [rating.rating for rating in fit.ratings.values()]
    assert max(values) - min(values) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Damping and the gauge: the two ways the iteration fails
# --------------------------------------------------------------------------- #
def test_undamped_iteration_oscillates(monkeypatch):
    """Pins the reason DAMPING exists. Undamped, a difference is corrected from both ends at once,
    so the error flips sign and keeps its magnitude -- forever. Two agents is the worst case."""
    pairs = [PairResult("a", "b", wins=75, losses=25, ties=0, games=100)]
    monkeypatch.setattr(L, "DAMPING", 1.0)
    assert not rate_field(pairs, anchor="b", max_sweeps=200).converged
    monkeypatch.setattr(L, "DAMPING", 0.5)
    assert rate_field(pairs, anchor="b", max_sweeps=200).converged


def test_anchor_shifts_every_rating_by_one_constant():
    """Gauge invariance: a round-robin identifies differences only, so changing the anchor may move
    the whole scale but must not reorder or restretch it."""
    pairs = _round_robin({"w": 1300.0, "x": 1450.0, "y": 1500.0, "z": 1650.0}, 2000)
    a = rate_field(pairs, anchor="y").ratings
    b = rate_field(pairs, anchor="w").ratings
    shifts = {label: a[label].rating - b[label].rating for label in a}
    assert max(shifts.values()) - min(shifts.values()) == pytest.approx(0.0, abs=1e-3)


def test_anchor_sits_exactly_at_the_reference():
    fit = rate_field(_round_robin({"a": 1400.0, "b": 1600.0, "c": 1500.0}, 500), anchor="c")
    assert fit.ratings["c"].rating == pytest.approx(DEFAULT_RATING, abs=1e-9)


def test_unknown_anchor_is_refused():
    with pytest.raises(LadderError, match="anchor"):
        rate_field(_round_robin({"a": 1400.0, "b": 1600.0}, 100), anchor="nobody")


# --------------------------------------------------------------------------- #
# Ties, which win rate cannot represent
# --------------------------------------------------------------------------- #
def test_ties_score_a_half_and_win_rate_would_not():
    """The failure this module avoids by not using `cross_evaluate`, which reports
    n_won / n_finished -- so a tie drags the number down exactly as a loss would."""
    pair = PairResult("a", "b", wins=40, losses=40, ties=20, games=100)
    assert pair.score == pytest.approx(0.5)
    win_rate_only = pair.wins / pair.games
    assert win_rate_only == pytest.approx(0.4)  # what cross_evaluate would have reported

    fit = rate_field([pair], anchor="b")
    # A genuinely even record must rate as even; under win-rate accounting `a` would sit below `b`.
    assert fit.ratings["a"].rating == pytest.approx(fit.ratings["b"].rating, abs=1e-6)


def test_an_all_tie_field_rates_flat():
    pairs = [PairResult("a", "b", wins=0, losses=0, ties=50, games=50)]
    fit = rate_field(pairs, anchor="a")
    assert fit.ratings["a"].rating == pytest.approx(fit.ratings["b"].rating, abs=1e-6)


# --------------------------------------------------------------------------- #
# RD: computed once, never accumulated
# --------------------------------------------------------------------------- #
def test_rd_does_not_shrink_with_sweep_count():
    """RD must describe the GAMES, not the solver. Running a Glicko period per sweep would shrink
    it every iteration over the same games and manufacture certainty."""
    pairs = _round_robin({"w": 1300.0, "x": 1500.0, "z": 1700.0}, 400)
    lo = rate_field(pairs, anchor="x", tol=1e-2)
    hi = rate_field(pairs, anchor="x", tol=1e-12)
    assert hi.sweeps > lo.sweeps
    for label in lo.ratings:
        assert hi.ratings[label].rd == pytest.approx(lo.ratings[label].rd, abs=0.5), label


def test_more_games_narrows_rd():
    few = rate_field(_round_robin({"a": 1400.0, "b": 1600.0}, 50), anchor="a")
    many = rate_field(_round_robin({"a": 1400.0, "b": 1600.0}, 5000), anchor="a")
    assert many.ratings["b"].rd < few.ratings["b"].rd


def test_two_agent_fit_is_not_rate_win_rate():
    """A deliberate NON-equivalence, pinned so nobody "fixes" the ladder to match.

    `rate_win_rate` takes ONE Glicko period from the (1500, 350) prior, so its answer is shrunk
    toward 1500. This module solves the score equation instead, which is unshrunk. They agree in
    sign and ordering and must not be expected to agree in value.
    """
    from lategame.eval.rating import rate_win_rate

    single_period = rate_win_rate(0.75, 100)
    fit = rate_field([PairResult("a", "b", wins=75, losses=25, ties=0, games=100)], anchor="b")
    mle = fit.ratings["a"]
    assert mle.rating > single_period.rating > DEFAULT_RATING
    assert mle.rating - single_period.rating > 10.0


# --------------------------------------------------------------------------- #
# The unbounded case
# --------------------------------------------------------------------------- #
def test_agent_that_never_scores_is_flagged_and_clamped():
    """A 0-for-everything record has no finite maximum-likelihood rating. Clamping silently would
    publish a made-up number, so it is clamped AND flagged."""
    pairs = [
        PairResult("hopeless", "a", wins=0, losses=100, ties=0, games=100),
        PairResult("hopeless", "b", wins=0, losses=100, ties=0, games=100),
        PairResult("a", "b", wins=60, losses=40, ties=0, games=100),
    ]
    fit = rate_field(pairs, anchor="a")
    assert fit.unbounded == ("hopeless",)
    assert fit.ratings["hopeless"].rating >= DEFAULT_RATING - L.RATING_CLAMP - 1e-6

    summary = summarize_ladder(
        [FieldEntry(name) for name in ("hopeless", "a", "b")],
        pairs, fit, battle_format="gen9ou", n_battles=100,
    )
    assert "unbounded_note" in summary
    assert next(r for r in summary["agents"] if r["label"] == "hopeless")["unbounded"] is True


def test_losing_every_game_to_one_opponent_is_not_unbounded():
    """Only a 0-against-the-WHOLE-field record is unbounded. Losing every game to the best agent
    while beating others is ordinary, and must still be rated."""
    pairs = [
        PairResult("mid", "best", wins=0, losses=100, ties=0, games=100),
        PairResult("mid", "worst", wins=90, losses=10, ties=0, games=100),
        PairResult("best", "worst", wins=100, losses=0, ties=0, games=100),
    ]
    fit = rate_field(pairs, anchor="mid")
    assert "mid" not in fit.unbounded


# --------------------------------------------------------------------------- #
# The payload
# --------------------------------------------------------------------------- #
def _small_summary():
    true = {"weak": 1300.0, "heuristic": 1500.0, "strong": 1700.0}
    pairs = _round_robin(true, 200)
    entries = [FieldEntry(name) for name in true]
    fit = rate_field(pairs, anchor="heuristic")
    return entries, pairs, fit, summarize_ladder(
        entries, pairs, fit, battle_format="gen9ou", n_battles=200
    )


def test_summary_orders_by_rating_and_carries_its_basis():
    _, _, _, summary = _small_summary()
    assert [row["label"] for row in summary["agents"]] == ["strong", "heuristic", "weak"]
    assert summary["anchor"] == "heuristic" and summary["converged"]
    # The basis is what stops the number being read as a Showdown ladder GXE.
    assert "AGENT-ONLY" in summary["gxe_basis"]
    assert "seed_strength_gate" in summary["gxe_basis"]
    assert "warning" not in summary


def test_matrix_is_symmetric_and_scores_complement():
    _, _, _, summary = _small_summary()
    matrix = summary["matrix"]
    for label, row in matrix.items():
        for opponent, cell in row.items():
            mirror = matrix[opponent][label]
            assert cell["wins"] == mirror["losses"]
            assert cell["score"] + mirror["score"] == pytest.approx(1.0)


def test_gxe_tracks_rating_order():
    _, _, _, summary = _small_summary()
    gxes = [row["gxe"] for row in summary["agents"]]
    assert gxes == sorted(gxes, reverse=True)
    # The anchor is pinned at the reference, so its GXE is 50% by construction.
    anchor_row = next(r for r in summary["agents"] if r["label"] == "heuristic")
    assert anchor_row["gxe"] == pytest.approx(0.5, abs=1e-3)


def test_results_file_round_trips(tmp_path):
    _, _, _, summary = _small_summary()
    out = tmp_path / "nested" / "ladder.json"
    write_results(out, {"field_spec": "weak,heuristic,strong"}, summary)
    payload = json.loads(out.read_text())
    assert payload["schema"] == L.SCHEMA
    assert payload["field_spec"] == "weak,heuristic,strong"
    assert payload["agents"][0]["label"] == "strong"
    assert not list(out.parent.glob("*.tmp"))


def test_table_renders_every_agent():
    _, _, _, summary = _small_summary()
    table = format_table(summary)
    assert all(row["label"] in table for row in summary["agents"])
    assert "Glicko" in table and "GXE" in table


# --------------------------------------------------------------------------- #
# run_round_robin, against fakes
# --------------------------------------------------------------------------- #
class _FakePlayer:
    """Enough of poke-env's Player for the driver: counters, battle_against, reset, teardown."""

    def __init__(self, name, script):
        self.username = name
        self._script = script  # opponent -> (wins, losses, ties)
        self.n_won_battles = self.n_lost_battles = self.n_finished_battles = 0
        self.resets = 0
        self.stopped = False
        self.ps_client = type("_C", (), {"stop_listening": self._stop})()

    async def _stop(self):
        self.stopped = True

    async def battle_against(self, other, n_battles):
        # ACCUMULATES, as poke-env's counters do -- they run over `player._battles` and are only
        # zeroed by `reset_battles`. A fake that assigned instead would hide the bug the delta
        # accounting exists to prevent.
        for player, opponent in ((self, other), (other, self)):
            wins, losses, ties = player._script[opponent.username]
            player.n_won_battles += wins
            player.n_lost_battles += losses
            player.n_finished_battles += wins + losses + ties

    def reset_battles(self):
        if getattr(self, "refuse_reset", False):
            raise OSError("Can not reset player's battles while they are still running")
        self.resets += 1
        self.n_won_battles = self.n_lost_battles = self.n_finished_battles = 0


def test_round_robin_scores_every_pair_from_the_counters(monkeypatch):
    script = {
        "a": {"b": (60, 30, 10), "c": (80, 20, 0)},
        "b": {"a": (30, 60, 10), "c": (55, 45, 0)},
        "c": {"a": (20, 80, 0), "b": (45, 55, 0)},
    }
    built = {}

    def _fake_build_player(name, battle_format, **kwargs):
        built[name] = _FakePlayer(name, script[name])
        return built[name]

    import lategame.eval.arena as arena

    monkeypatch.setattr(arena, "build_player", _fake_build_player)
    monkeypatch.setattr(arena, "AGENTS", {"a": object, "b": object, "c": object})

    entries = [FieldEntry(name) for name in ("a", "b", "c")]
    pairs = asyncio.run(L.run_round_robin(entries, "gen9ou", 100))

    assert len(pairs) == 3  # 3 agents -> 3 unordered pairs, each played once
    ab = next(p for p in pairs if {p.a, p.b} == {"a", "b"})
    assert (ab.wins, ab.losses, ab.ties, ab.games) == (60, 30, 10, 100)
    assert ab.score == pytest.approx(0.65)  # ties count a half
    # Players are built ONCE and reset between pairs, not rebuilt per pair.
    assert len(built) == 3
    assert all(player.stopped for player in built.values())


def test_round_robin_refuses_an_unknown_agent(monkeypatch):
    import lategame.eval.arena as arena

    monkeypatch.setattr(arena, "AGENTS", {"heuristic": object})
    with pytest.raises(LadderError, match="unknown agent"):
        asyncio.run(L.run_round_robin([FieldEntry("nope"), FieldEntry("heuristic")], "gen9ou", 1))


def test_pairs_are_scored_by_difference_when_reset_is_refused(monkeypatch):
    """poke-env's `reset_battles` REFUSES while any battle is still running, and a stuck battle must
    not take the tournament down. Suppressing that error alone would roll the previous pair's games
    into the next pair's totals -- so pairs are measured as deltas and this pins it."""
    script = {
        "a": {"b": (60, 30, 10), "c": (80, 20, 0)},
        "b": {"a": (30, 60, 10), "c": (55, 45, 0)},
        "c": {"a": (20, 80, 0), "b": (45, 55, 0)},
    }

    def _fake_build_player(name, battle_format, **kwargs):
        player = _FakePlayer(name, script[name])
        player.refuse_reset = True  # every reset fails, so counters never zero
        return player

    import lategame.eval.arena as arena

    monkeypatch.setattr(arena, "build_player", _fake_build_player)
    monkeypatch.setattr(arena, "AGENTS", {"a": object, "b": object, "c": object})

    entries = [FieldEntry(name) for name in ("a", "b", "c")]
    pairs = asyncio.run(L.run_round_robin(entries, "gen9ou", 100))

    # `a` plays b then c. Absolute counters would report a's SECOND pair as 140-50-10 over 200
    # games; the delta is the pair that was actually played.
    ac = next(p for p in pairs if {p.a, p.b} == {"a", "c"})
    assert (ac.wins, ac.losses, ac.ties, ac.games) == (80, 20, 0, 100)
    assert all(p.games == 100 for p in pairs)


def test_a_finite_but_terrible_rating_is_flagged_if_it_reaches_the_bound(monkeypatch):
    """The case the `unbounded` flag alone MISSED on the first real ladder.

    `offrl_gen9ou_wide_s0` scored 19/1050 against the gen9ou field. That is a finite MLE -- it won
    games, so `unbounded` was correctly False -- but it sat ~1140 points below the anchor, which the
    then-1000-point clamp truncated to exactly 500.0. The fit still reported `converged`, because a
    clamped value stops moving. So a bound was published as a rating with nothing marking it.
    """
    monkeypatch.setattr(L, "RATING_CLAMP", 300.0)  # force the bound to bind
    pairs = [
        PairResult("awful", "a", wins=1, losses=149, ties=0, games=150),
        PairResult("awful", "b", wins=1, losses=149, ties=0, games=150),
        PairResult("a", "b", wins=75, losses=75, ties=0, games=150),
    ]
    fit = rate_field(pairs, anchor="a")

    assert fit.unbounded == ()          # it won games, so an MLE exists
    assert "awful" in fit.clamped       # ...but the guard truncated it, and that must show
    assert fit.converged                # and convergence alone would NOT have told you

    summary = summarize_ladder(
        [FieldEntry(n) for n in ("awful", "a", "b")], pairs, fit,
        battle_format="gen9ou", n_battles=150,
    )
    assert "clamped_note" in summary
    assert next(r for r in summary["agents"] if r["label"] == "awful")["clamped"] is True
    assert "CLAMPED" in format_table(summary)


def test_nothing_is_clamped_in_a_normal_field():
    pairs = _round_robin({"w": 1300.0, "x": 1500.0, "z": 1700.0}, 300)
    fit = rate_field(pairs, anchor="x")
    assert fit.clamped == () and fit.unbounded == ()

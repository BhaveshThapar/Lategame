"""Glicko-1 and GXE, pinned against published answers rather than against themselves.

Two external anchors, because a rating implementation that is only self-consistent is worthless:
Glickman's own worked example for the update, and Showdown's published GXE formula for the
prediction. Everything else here guards a property the arena will depend on at M5.
"""

import math

import pytest

from lategame.eval.rating import (
    DEFAULT_RATING,
    MAX_RD,
    REFERENCE,
    Rating,
    decay_rd,
    expected_score,
    g,
    gxe,
    rate_win_rate,
    update,
)


def _showdown_gxe(rating: float, rd: float) -> float:
    """Showdown's published GXE, transcribed. The fraction, not the rounded percentage."""
    ln10 = math.log(10.0)
    denom = math.sqrt(3.0 * ln10 * ln10 * rd * rd + 2500.0 * (64.0 * math.pi**2 + 147.0 * ln10**2))
    return 1.0 / (1.0 + 10.0 ** ((1500.0 - rating) * math.pi / denom))


# --- the update, against Glickman's worked example ------------------------------------------


def test_glickman_worked_example():
    """The example from the Glicko paper: 1500/200 plays 1400/30 (W), 1550/100 (L), 1700/300 (L),
    and comes out at 1464 / 151.4."""
    player = Rating(1500.0, 200.0)
    new = update(
        player,
        [(Rating(1400.0, 30.0), 1.0), (Rating(1550.0, 100.0), 0.0), (Rating(1700.0, 300.0), 0.0)],
    )
    assert new.rating == pytest.approx(1464.0, abs=0.5)
    assert new.rd == pytest.approx(151.4, abs=0.1)


def test_glickman_example_g_and_e_terms():
    """The intermediate quantities the paper tabulates, so a failure localises."""
    assert g(30.0) == pytest.approx(0.9955, abs=1e-4)
    assert g(100.0) == pytest.approx(0.9531, abs=1e-4)
    assert g(300.0) == pytest.approx(0.7242, abs=1e-4)

    player = Rating(1500.0, 200.0)
    assert expected_score(player, Rating(1400.0, 30.0)) == pytest.approx(0.639, abs=1e-3)
    assert expected_score(player, Rating(1550.0, 100.0)) == pytest.approx(0.432, abs=1e-3)
    assert expected_score(player, Rating(1700.0, 300.0)) == pytest.approx(0.303, abs=1e-3)


def test_a_rating_period_is_order_independent():
    """All games in a period are simultaneous -- that is the property that makes Glicko-1 a
    RATING PERIOD method rather than an incremental one."""
    games = [
        (Rating(1400.0, 30.0), 1.0),
        (Rating(1550.0, 100.0), 0.0),
        (Rating(1700.0, 300.0), 0.0),
    ]
    forward = update(Rating(1500.0, 200.0), games)
    backward = update(Rating(1500.0, 200.0), list(reversed(games)))
    assert forward.rating == pytest.approx(backward.rating)
    assert forward.rd == pytest.approx(backward.rd)


def test_games_always_shrink_the_deviation():
    player = Rating(1500.0, 350.0)
    after = update(player, [(REFERENCE, 1.0), (REFERENCE, 0.0)])
    assert after.rd < player.rd


def test_beating_a_stronger_opponent_moves_the_rating_further():
    weak = update(Rating(), [(Rating(1200.0, 50.0), 1.0)])
    strong = update(Rating(), [(Rating(1800.0, 50.0), 1.0)])
    assert strong.rating > weak.rating > DEFAULT_RATING


def test_an_empty_period_changes_nothing():
    """Nothing observed is not an error, and must not be confused with a 50% period."""
    player = Rating(1600.0, 120.0)
    assert update(player, []) == player


def test_rejects_a_score_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
        update(Rating(), [(REFERENCE, 1.5)])


def test_rejects_a_non_positive_deviation():
    with pytest.raises(ValueError, match="rd must be positive"):
        Rating(1500.0, 0.0)


# --- GXE, against Showdown's published formula -----------------------------------------------


@pytest.mark.parametrize("rating", [1000.0, 1300.0, 1500.0, 1700.0, 2100.0])
@pytest.mark.parametrize("rd", [25.0, 50.0, 130.0, 350.0])
def test_gxe_matches_showdowns_published_formula(rating, rd):
    assert gxe(Rating(rating, rd)) == pytest.approx(_showdown_gxe(rating, rd), abs=1e-9)


def test_gxe_of_the_reference_player_is_one_half():
    assert gxe(REFERENCE) == pytest.approx(0.5)


def test_gxe_is_monotone_in_rating():
    rates = [gxe(Rating(r, 100.0)) for r in (1200.0, 1400.0, 1600.0, 1800.0)]
    assert rates == sorted(rates)


def test_uncertainty_pulls_gxe_toward_one_half():
    """The property that makes GXE bias-robust: an unproven 1800 is not credited as an 1800."""
    certain = gxe(Rating(1800.0, 30.0))
    unproven = gxe(Rating(1800.0, 350.0))
    assert 0.5 < unproven < certain


# --- RD decay ---------------------------------------------------------------------------------


def test_rd_decay_grows_and_caps_at_the_ceiling():
    assert decay_rd(50.0, c=30.0, periods=1.0) == pytest.approx(math.sqrt(50.0**2 + 30.0**2))
    assert decay_rd(340.0, c=100.0, periods=10.0) == MAX_RD


def test_rd_decay_rejects_negative_periods():
    with pytest.raises(ValueError, match="periods must be non-negative"):
        decay_rd(50.0, c=30.0, periods=-1.0)


# --- the arena bridge -------------------------------------------------------------------------


def test_rate_win_rate_is_monotone_and_centred():
    """Against ONE pinned opponent this is a reparameterised win rate -- pinning that, because it
    is exactly the limitation the module docstring warns not to over-read."""
    lost = rate_win_rate(0.2, 300)
    even = rate_win_rate(0.5, 300)
    won = rate_win_rate(0.8, 300)
    assert lost.rating < even.rating < won.rating
    assert even.rating == pytest.approx(DEFAULT_RATING, abs=1e-6)


def test_more_battles_shrink_the_deviation():
    assert rate_win_rate(0.6, 1800).rd < rate_win_rate(0.6, 300).rd < Rating().rd


def test_rate_win_rate_carries_the_prior_forward():
    """M5 accumulates across sessions; a fresh Rating() every period would discard the history.

    A 1700 that only breaks even against the 1500 reference is pulled DOWN toward it -- the prior
    is a starting point, not an anchor. The check is that the prior was used at all: the result
    must sit strictly between the prior and what the same record would score from scratch.
    """
    prior = Rating(1700.0, 80.0)
    from_prior = rate_win_rate(0.5, 100, prior=prior)
    from_scratch = rate_win_rate(0.5, 100)
    assert from_scratch.rating < from_prior.rating < prior.rating


def test_rate_win_rate_rejects_impossible_inputs():
    with pytest.raises(ValueError, match=r"win_rate must be in \[0, 1\]"):
        rate_win_rate(1.4, 100)
    with pytest.raises(ValueError, match="n_battles must be non-negative"):
        rate_win_rate(0.5, -1)


def test_zero_battles_leaves_the_prior_untouched():
    assert rate_win_rate(0.0, 0, prior=Rating(1650.0, 90.0)) == Rating(1650.0, 90.0)

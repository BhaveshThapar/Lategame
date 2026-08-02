"""R-EVAL: Glicko-1 and GXE, the bias-robust metrics plan.md 12 makes primary.

Through Build 25 every verdict in this project is a win rate against a FIXED baseline, and that
was the right Phase-1 signal: with one opponent held constant, win rate is the sufficient
statistic and `scripts/seed_strength_gate.py` remains the authoritative build-vs-build protocol.
Nothing here replaces it.

What win rate cannot do is score LADDER play, where the opponents vary and matchmaking pushes
everyone toward ~50% -- the exact bias 12 cites. That is M5's regime, and it needs a rating.

Two honest limitations, stated here because the numbers look authoritative either way:

  * On a fixed-baseline arena this DEGENERATES. Rate a policy against one opponent pinned at the
    reference and the Glicko rating is a monotone reparameterisation of the win rate, plus an
    uncertainty term driven by ``n``. It carries no information the win rate did not. It becomes
    meaningful only when fed a varied opponent field -- i.e. live/ladder play.
  * Glicko-1, not Glicko-2. Glicko-1 tracks (rating, RD); the volatility parameter is Glicko-2's.
    Showdown itself ladders on Glicko-1, which is what makes GXE below comparable to the number
    shown on the site.

References: Glickman's Glicko paper (the worked example is pinned in tests/test_rating.py), and
Showdown's own GXE, reproduced exactly by ``gxe`` below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Showdown's ladder conventions, and Glicko's own defaults.
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
MAX_RD = 350.0

# Glickman's q = ln(10)/400: the logistic scale that makes 400 points a 10:1 odds ratio.
Q = math.log(10.0) / 400.0


@dataclass(frozen=True)
class Rating:
    """A Glicko-1 rating: a point estimate and its deviation.

    ``rd`` is a standard deviation in rating points, not a confidence interval -- a fresh player
    at (1500, 350) is roughly "somewhere in 800-2200", which is why an unrated agent's GXE sits
    near 50% no matter how it plays until games accumulate.
    """

    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD

    def __post_init__(self) -> None:
        if self.rd <= 0.0:
            raise ValueError(f"rd must be positive, got {self.rd}")


#: The average ladder player GXE is defined against, and the default opponent for a fixed-baseline
#: arena. Pinning the baseline here (rather than rating it too) is a CONVENTION, not a
#: measurement -- see the module docstring.
REFERENCE = Rating(DEFAULT_RATING, MAX_RD)


def g(rd: float) -> float:
    """Glicko's attenuation: how much an opponent's uncertainty flattens the expected score."""
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi * math.pi))


def expected_score(player: Rating, opponent: Rating) -> float:
    """P(player scores) against ``opponent``, attenuated by the OPPONENT's deviation only.

    Glicko-1's within-period expectation deliberately ignores the player's own RD: it is the
    quantity the update equations take a derivative of. ``gxe`` below is the other case -- a
    forward-looking prediction, where the player's own uncertainty does matter.
    """
    return 1.0 / (1.0 + 10.0 ** (-g(opponent.rd) * (player.rating - opponent.rating) / 400.0))


def decay_rd(rd: float, c: float, periods: float = 1.0) -> float:
    """RD growth over ``periods`` of inactivity, capped at ``MAX_RD``.

    ``c`` sets how fast certainty decays; Glickman picks it so a typical player returns to the
    350 ceiling over the time you would consider them unrated again.
    """
    if periods < 0.0:
        raise ValueError(f"periods must be non-negative, got {periods}")
    return min(math.sqrt(rd * rd + c * c * periods), MAX_RD)


def update(player: Rating, results: list[tuple[Rating, float]]) -> Rating:
    """Glicko-1 update over ONE rating period.

    ``results`` is (opponent, score) with score in {0, 0.5, 1}. All games in a period are treated
    as simultaneous -- the opponent ratings are the ones held at the START of the period, which is
    what makes the result independent of the order they were played in.

    An empty period is not an error: nothing was observed, so the rating is unchanged. Callers
    that also want inactivity decay apply ``decay_rd`` themselves, since the period length is
    theirs to define.
    """
    if not results:
        return player

    d_inv = 0.0
    delta = 0.0
    for opponent, score in results:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        g_j = g(opponent.rd)
        e_j = expected_score(player, opponent)
        d_inv += g_j * g_j * e_j * (1.0 - e_j)
        delta += g_j * (score - e_j)
    d_inv *= Q * Q

    # d_inv == 0 only when every expected score saturated to 0 or 1 (a mismatch so extreme the
    # games carry no information about the rating). Guard rather than divide by zero.
    if d_inv <= 0.0:
        return player

    denom = 1.0 / (player.rd * player.rd) + d_inv
    return Rating(rating=player.rating + (Q / denom) * delta, rd=math.sqrt(1.0 / denom))


def gxe(player: Rating, opponent: Rating = REFERENCE) -> float:
    """Expected win rate vs an average player, as a FRACTION in [0, 1].

    Unlike ``expected_score`` this is a forward-looking prediction, so BOTH deviations enter: an
    agent at 1800/350 has a lower GXE than one at 1800/50, because we are less sure it is 1800.

    This reproduces Showdown's published GXE exactly. Showdown writes the same quantity as

        10000 / (1 + 10 ^ ((1500 - r) * pi / sqrt(3*ln(10)^2*rd^2 + 2500*(64*pi^2 + 147*ln(10)^2))))

    whose constants are this formula's in disguise: 2500*64*pi^2 = 400^2*pi^2 and
    2500*147*ln(10)^2 = 3*ln(10)^2*350^2, i.e. an opponent pinned at RD 350. Pinned in
    tests/test_rating.py against that form.
    """
    combined_rd = math.sqrt(player.rd * player.rd + opponent.rd * opponent.rd)
    return 1.0 / (1.0 + 10.0 ** (-g(combined_rd) * (player.rating - opponent.rating) / 400.0))


def rate_win_rate(
    win_rate: float,
    n_battles: int,
    opponent: Rating = REFERENCE,
    prior: Rating | None = None,
) -> Rating:
    """Rating implied by ``win_rate`` over ``n_battles`` against a single fixed ``opponent``.

    The bridge from this project's arena to R-EVAL's metrics. Read the module docstring before
    reporting the result: against ONE pinned opponent this is a reparameterised win rate, and the
    only genuinely new information it carries is how much ``n_battles`` narrowed the deviation.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
    if n_battles < 0:
        raise ValueError(f"n_battles must be non-negative, got {n_battles}")

    wins = round(win_rate * n_battles)
    results = [(opponent, 1.0)] * wins + [(opponent, 0.0)] * (n_battles - wins)
    return update(prior if prior is not None else Rating(), results)

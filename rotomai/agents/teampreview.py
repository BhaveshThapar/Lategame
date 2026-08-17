"""Bring-6-pick-4: a matchup-scored team-preview selector for doubles formats.

Until now nothing in this repository overrode ``Player.teampreview``, so every VGC battle ever run
here -- every ceiling-probe cell, every BC/AWR ladder row, every self-play episode -- opened with
poke-env's ``random_teampreview`` picking 4 of 6 uniformly at random on BOTH sides. plan.md has
flagged that as "a large part of VGC skill the policy cannot express, adding variance to every
number above"; this is the part of it that does not need a new action space.

WHY THIS IS HARNESS-LEVEL AND NOT AGENT-LEVEL. The obvious shape is a method on ``DoublesAgent``.
That would be a measurement bug. Three of the five arms in every VGC ladder -- ``random``,
``maxbasepower``, ``simpleheuristics`` -- are poke-env's own built-ins, and they would have kept
picking at random. Our arm would then win partly on bringing a better four, and the win rate would
read as *play* strength. So the selector is mixed into whatever class the registry hands back
(``eval.arena.build_player``), the baselines included, and preview stops being a variable that
distinguishes the arms at all.

WHAT IT SCORES. At preview we know our own six completely (species, types, moves) and the
opponent's six only as species -- so their types and base stats, and nothing about their moves.
The score is therefore asymmetric by necessity:

  * offense(m, o): the best expected-damage proxy over m's KNOWN damaging moves into o, via
    ``engine.damage.score_move``, normalized by the best any of our mons manages into that o --
    so it lands in [0, 1].
  * threat(o, m): the best type multiplier of o's OWN types into m -- the STAB assumption, which
    is the strongest thing you can say about a moveset you cannot see -- expressed in EFFECTIVENESS
    STEPS, log2 of the multiplier over 2, so it also lands in [-1, 1] (4x = 1.0, 2x = 0.5, neutral
    = 0, resisted = -0.5, immune floors at -1).

  matchup(m, o) = offense(m, o) * (1 - THREAT_WEIGHT * threat(o, m))

  MULTIPLICATIVE, not additive, and the difference is not cosmetic. The first draft added the two
  terms, and a mon with no damaging moves at all then scored on typing alone -- a test caught it
  bringing Magikarp and Sunkern over Scizor. What a bring is worth is how much it can do, scaled by
  how well it holds up; if it can do nothing, holding up well does not rescue it. Zero offense is
  zero value under a product and cannot be argued back up.

and a mon's score is its mean matchup over the opponent's six, plus a small base-stat-total term to
break ties toward the stronger Pokemon. Top four are brought.

WHAT IT IS NOT. Not speed control, not Protect/redirection cores, not weather or Tera planning,
not a learned policy over the 6-choose-4 space -- the codec has no slot for preview and the model
has no head for it. This is the proportionate version: it makes both sides express *some* preview
skill instead of none, and it is a fixed rule, so it adds no free parameters to fit.
"""

from __future__ import annotations

import math

from poke_env.battle import AbstractBattle, MoveCategory, Pokemon

from rotomai.engine.damage import score_move

# How much a mon being threatened counts against bringing it, relative to what it threatens. Both
# terms are already on a [0, 1] / [-1, 1] scale, so this is a genuine weight rather than a unit
# conversion. Below 1.0 on purpose: offense is computed from moves we can actually see, threat from
# a STAB assumption over a moveset we cannot, so the offensive term is the better-evidenced one.
THREAT_WEIGHT = 0.6

# Tie-break only. A base-stat total runs ~200-700, so dividing by 600 and weighting this low keeps
# it from outvoting a real type matchup while still preferring the stronger of two even mons.
BST_WEIGHT = 0.15

VGC_BRING = 4


def _offense(attacker: Pokemon, defender: Pokemon) -> float:
    """Best expected-damage proxy from ``attacker``'s known damaging moves into ``defender``."""
    scores = [
        score_move(move, attacker, defender)
        for move in attacker.moves.values()
        if move.category != MoveCategory.STATUS
    ]
    return max(scores, default=0.0)


def _threat(attacker: Pokemon, defender: Pokemon) -> float:
    """``attacker``'s best STAB into ``defender``, in effectiveness steps scaled to [-1, 1].

    The opponent's moves are not visible at preview, so their types stand in for their coverage.
    That understates a mon carrying off-type coverage and overstates one whose STAB is dead weight,
    but it is the only claim the preview state supports.

    Steps, not the raw multiplier: log2(4x) = 2 and log2(0.25x) = -2, so halving puts a 4x weakness
    at exactly +1 against a normalized offense that maxes at 1. An immunity is log2(0) = -inf and
    is floored at -1 rather than dominating the mean of six columns on its own.
    """
    best = max(
        (defender.damage_multiplier(t) for t in attacker.types if t is not None),
        default=1.0,
    )
    if best <= 0.0:
        return -1.0
    return max(-1.0, min(1.0, math.log2(best) / 2.0))


def _bst(mon: Pokemon) -> float:
    stats = mon.base_stats or {}
    return float(sum(stats.values()))


def score_our_team(ours: list[Pokemon], theirs: list[Pokemon]) -> list[float]:
    """Per-mon bring scores for ``ours`` against the previewed ``theirs``, index-aligned to ours.

    Offense is normalized PER OPPONENT MON: raw ``score_move`` values scale with base power, so an
    unnormalized mean would rank a team by how hard it hits in absolute terms rather than by who
    handles this opponent best. Dividing by the best any of our six manages into that opponent
    makes each column a relative-answer score in [0, 1].
    """
    if not ours:
        return []
    if not theirs:
        # No preview information (the opponent's roster never arrived). Fall back to raw strength
        # rather than to an arbitrary order -- still deterministic, still better than random.
        return [BST_WEIGHT * _bst(m) / 600.0 for m in ours]

    scores = []
    columns = [[_offense(m, o) for m in ours] for o in theirs]
    for i, mon in enumerate(ours):
        total = 0.0
        for o, column in zip(theirs, columns, strict=True):
            best = max(column)
            offense = column[i] / best if best > 0 else 0.0
            total += offense * (1.0 - THREAT_WEIGHT * _threat(o, mon))
        scores.append(total / len(theirs) + BST_WEIGHT * _bst(mon) / 600.0)
    return scores


def choose_preview(battle: AbstractBattle, bring: int = VGC_BRING) -> str:
    """The ``/team`` order for ``battle``, plus the ``_selected_in_teampreview`` marks with it.

    poke-env's ``Player.teampreview`` docstring makes the marking part of the contract -- the
    encoder's opponent-roster merge and ``eval.ladder``'s matchup clustering both read it -- so
    setting it is not optional bookkeeping.
    """
    ours = list(battle.team.values())
    theirs = list(battle.teampreview_opponent_team or [])
    scores = score_our_team(ours, theirs)

    # Rank by score, and break exact ties by original slot so the order is reproducible across runs
    # rather than dependent on sort stability of equal floats from a different battle.
    order = sorted(range(len(ours)), key=lambda i: (-scores[i], i))
    picked = order[: min(bring, len(ours))]

    for i, mon in enumerate(ours):
        mon._selected_in_teampreview = i in picked
    return "/team " + "".join(str(i + 1) for i in picked)


class TeamPreviewMixin:
    """Mixed in ahead of a ``Player`` subclass so ``teampreview`` resolves here first.

    Applied by ``eval.arena.build_player`` on doubles formats, to every agent it builds. Kept as a
    mixin rather than an edit to each agent class because three of the agents that need it --
    poke-env's ``RandomPlayer``, ``MaxBasePowerPlayer``, ``SimpleHeuristicsPlayer`` -- are not ours
    to edit, and those are exactly the baselines a ladder is measured against.
    """

    def teampreview(self, battle: AbstractBattle) -> str:
        return choose_preview(battle)

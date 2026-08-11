"""R-EVAL part 2: an agent-only eval ladder -- the varied opponent field G2 is defined against.

plan.md 12 states G2 in GXE and Glicko-1, and section 12's last line says to evaluate "on a
private/agent-only server or eval ladder wherever possible". This module is that eval ladder: a
round-robin over a heterogeneous field of agents on the local server, rated jointly.

WHY THIS EXISTS AT ALL. ``eval/arena.py`` and every build gate score one agent against ONE fixed
baseline, where win rate is the sufficient statistic and the rating is a monotone reparameterisation
of it -- ``eval/rating.py``'s docstring says so at length, and nothing here replaces
``scripts/seed_strength_gate.py`` as the authoritative build-vs-build protocol. What a fixed
baseline cannot produce is a *varied field*, and without one GXE carries no information the win rate
did not. A round-robin over agents of genuinely different strengths is a varied field.

Three things here are load-bearing and non-obvious.

ONE GLICKO PERIOD IS NOT ENOUGH -- IT IS DEGENERATE. Every agent starts at (1500, 350). Update each
against opponents held at their priors and every opponent IS the reference, so each rating collapses
to a monotone function of that agent's own score rate: the same degeneracy, reached by a longer
road. The fit has to iterate, using the current estimates as the opponent ratings. At the fixed
point every agent satisfies ``sum_j g(RD_j) * (s_j - E_j) = 0`` -- the Bradley-Terry score equation
-- so what this computes is the BT/Elo maximum-likelihood fit over the round-robin matrix, reported
on the Glicko scale. The iteration is only how we get there.

THE FIT NEEDS A GAUGE FIX. A round-robin matrix identifies rating *differences* and nothing else:
add 100 to every agent and the likelihood is unchanged, so the fixed point is not unique. One agent
is therefore pinned. ``heuristic`` is the right anchor for this project rather than an arbitrary
choice -- it is the fixed baseline every build from M1 onward is already reported against, so
pinning it at ``DEFAULT_RATING`` makes the ladder's Glicko directly comparable to the win-rate
history, and makes GXE read as "expected score vs a heuristic-strength average player".

RD IS COMPUTED ONCE, AT CONVERGENCE. Running a Glicko period per sweep would shrink RD on every
sweep over the *same* games and manufacture certainty -- an agent's deviation would depend on how
many iterations the solver happened to take. So the sweeps move means only, and the deviation comes
from a single period against the converged field, which is the honest one-period quantity.

ONE LIMITATION, STATED BECAUSE IT IS EASY TO MISREAD AS A BUG. Bradley-Terry gives each agent a
single latent strength, so it cannot represent NON-TRANSITIVITY: in a perfect rock-paper-scissors
cycle the fit rates every agent equal, because by the model's own lights they are. Pokemon carries
real non-transitivity (team archetypes counter each other), so a cluster of near-equal ratings can
mean "cyclic" rather than "equally strong". The full win matrix is written alongside the table for
exactly this reason -- ``tests/test_eval_ladder.py`` pins the behaviour.

Finally, note what this module does NOT use. ``poke_env.player.cross_evaluate`` reports
``Player.win_rate`` = ``n_won_battles / n_finished_battles``, so A TIE DRAGS WIN RATE DOWN WITHOUT
COUNTING AS A LOSS -- the same failure that stops ``rating.rate_win_rate`` from representing a tie,
one layer down. Pairs are therefore played through ``battle_against`` directly and scored off the
win/loss/tie counters. ``arena.evaluate_built`` wrapped ``cross_evaluate`` and inherited the bug
when this was written; it was fixed to score the same way once Build 26 released the frozen path.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lategame.eval.rating import DEFAULT_RATING, MAX_RD, Rating, gxe, update

SCHEMA = "lategame.eval.ladder/1"

#: The default anchor. See the module docstring: this is the project's fixed baseline, so pinning
#: it is what makes the ladder's numbers commensurable with every win rate already published.
DEFAULT_ANCHOR = "heuristic"

#: The default field. Deliberately spans the whole measured strength range rather than clustering
#: near the top -- the PPO dose-response curve (80 -> 120 -> 160 updates) doubles as the opponent
#: population, which is what gives the fit something to separate.
#:
#: ``v26a`` is NOT here on purpose. Its two completed arms exist, but Build 26's verdict comes from
#: the pre-registered ``scripts/seed_strength_gate.py`` protocol, and a mid-flight arm in a rating
#: table invites exactly the reading pre-registration exists to prevent. Add it with --field once
#: Build 26 is called.
#:
#: Note every learned entry is ``offrl@``, INCLUDING the PPO builds. That is not a typo -- see
#: ``_TRAINING_ONLY_AGENTS``.
DEFAULT_FIELD = (
    "random",
    "maxbasepower",
    "simpleheuristics",
    "heuristic",
    "offrl@checkpoints/offrl_gen9ou_wide_s0.pt",
    "offrl@checkpoints/ppo_v23b_s0/iter_80.pt",
    "offrl@checkpoints/ppo_v25a_s0/iter_120.pt",
    "offrl@checkpoints/ppo_v25b_s0/iter_160.pt",
)

#: Agents that exist to GENERATE training data, not to be scored, mapped to what to use instead.
#:
#: ``PPORecordingAgent`` forces ``sample=True`` -- its own docstring says sampling is mandatory,
#: because PPO needs the on-policy action distribution rather than greedy arg-max. So a PPO
#: checkpoint played through ``ppo`` is a MATERIALLY DIFFERENT and weaker policy than the same
#: checkpoint played through ``offrl``: measured head-to-head vs the heuristic on ``v25b``'s
#: terminal checkpoint, 0.675 sampled against 0.767 greedy, a ~9-point gap.
#:
#: Every published number in this project reads a PPO checkpoint through ``offrl``
#: (``scripts/seed_strength_gate.py``, ``scripts/ppo_continue_gate.py``), because greedy is the
#: deployed policy. ``ppo@some_checkpoint`` therefore looks exactly like "rate the PPO build" and
#: silently rates something else, which is why it is refused rather than documented. To score the
#: sampled policy deliberately, use ``offrl@`` with ``--sample``.
_TRAINING_ONLY_AGENTS = {"ppo": "offrl"}

#: Divergence guard ONLY, and deliberately generous. An agent that scores 0 or N against the ENTIRE
#: field has no finite maximum-likelihood rating and the iteration would walk off to infinity; that
#: case is detected up front and flagged ``unbounded``, so this bound exists purely to keep the
#: arithmetic finite while it happens.
#:
#: It is 2000 rather than something tighter because a merely-terrible agent still has a FINITE MLE
#: that can sit a very long way down, and truncating it would publish a bound as though it were a
#: fit. Measured on the first real gen9ou ladder: `offrl_gen9ou_wide_s0` scored 19/1050 against the
#: field -- finite, but ~1140 points below the anchor, which a 1000-point clamp silently truncated
#: to exactly 500.0 while the fit still reported `converged`. Any rating that reaches this bound is
#: now flagged ``clamped`` whatever the cause.
RATING_CLAMP = 2000.0

#: Build-14 LoopGuard strength, defaulted to match the AUTHORITATIVE gate rather than to zero.
#:
#: ``scripts/cluster/strength_gate.slurm`` sets ``LOOP_PENALTY=4`` and every published seed-strength
#: read records ``loop_penalty: 4.0`` -- including ``v25b``'s 0.6807 terminal read. Since the whole
#: point of anchoring on ``heuristic`` is to make these ratings commensurable with that win-rate
#: history, the ladder has to score the same policy the history scored. Defaulting to 0.0 would
#: quietly rate a DIFFERENT (unguarded) agent and invite a direct comparison anyway.
#:
#: Only the learned agents carry a LoopGuard (``arena._LOOP_GUARD_AGENTS``); it is a no-op for the
#: scripted baselines, so this does not touch their side of the matrix.
DEFAULT_LOOP_PENALTY = 4.0

#: Iteration controls. ``TOL`` is in rating points; 1e-6 is far below anything reportable.
MAX_SWEEPS = 1000
TOL = 1e-6

#: THE UNDAMPED ITERATION DOES NOT CONVERGE, AND THE TWO-AGENT CASE IS WHERE IT IS WORST.
#:
#: Every agent is updated against the field held at the *previous* sweep (Jacobi, not Gauss-Seidel),
#: which keeps the result independent of the order agents are walked in. The cost is that a rating
#: DIFFERENCE gets corrected from both ends at once: with two agents, ``a`` moves to put the
#: difference at its target relative to ``b``'s old rating while ``b`` does the mirror, so the new
#: difference overshoots to ``2*target - current`` -- an error that flips sign and keeps its
#: magnitude forever. Pure oscillation, never a fit.
#:
#: The Glicko step is exactly a Newton step on the Bradley-Terry score equation (its ``q/denom``
#: factor is ``1 / (q * sum g^2 E (1-E))``, the reciprocal Hessian), so the iteration matrix is
#: ``I - D^-1 H``. ``H`` is the BT Hessian, whose rows sum to zero, giving ``D^-1 H`` eigenvalues in
#: ``[0, 2]`` -- and the 2 is attained exactly at two agents. Damping by ``a`` maps those to
#: ``1 - a*lambda``, so **a = 0.5 puts every eigenvalue in [0, 1]**; the residual 1 is the gauge
#: direction, which the anchor shift below removes. Halving is therefore not a fudge factor, it is
#: the value that exactly cancels the double correction.
DAMPING = 0.5

_GXE_BASIS = (
    "AGENT-ONLY eval ladder: a round-robin over the field below, fitted jointly (Bradley-Terry "
    "score equation, reported on the Glicko-1 scale) with {anchor!r} pinned at {rating:.0f} to fix "
    "the additive gauge. Because the field VARIES, GXE here is a measurement rather than a "
    "reparameterisation of a win rate -- which is what it degenerates to against a single fixed "
    "baseline. Two things it is NOT: (1) comparable to a Showdown GXE, which is measured against "
    "HUMANS on the public ladder, not against bots; (2) a replacement for "
    "scripts/seed_strength_gate.py, which remains the authoritative build-vs-build statistic. "
    "RD IS A LOWER BOUND ON THE UNCERTAINTY, not an interval: it is derived from binomial noise "
    "alone, but on a teambuilt format the battles are CLUSTERED BY TEAM MATCHUP (a small pool plus "
    "archetype counters), so the effective sample is well under n. Measured across two independent "
    "n=150 gen9ou runs, one pairing moved 0.127 -- about 3.1 binomial SE -- while each agent's own "
    "GXE moved under 0.004. Read the standings as separating BANDS, not as ordering neighbours."
)


class LadderError(RuntimeError):
    """Raised when a field or a fit cannot produce a meaningful rating."""


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldEntry:
    """One competitor: a registered agent name, optionally backed by a checkpoint."""

    agent: str
    checkpoint: str | None = None

    @property
    def label(self) -> str:
        """Short unique name used as the key everywhere downstream.

        Checkpoint-backed entries need more than the agent name -- a field can hold four ``ppo``
        entries at different iteration counts, and ``ppo`` four times over would collide.
        """
        if self.checkpoint is None:
            return self.agent
        path = Path(self.checkpoint)
        # checkpoints/ppo_v25b_s0/iter_160.pt -> ppo@ppo_v25b_s0/iter_160
        # checkpoints/offrl_gen9ou_wide_s0.pt -> offrl@offrl_gen9ou_wide_s0
        parent = path.parent.name
        stem = path.stem
        tail = stem if parent in {"", ".", "checkpoints"} else f"{parent}/{stem}"
        return f"{self.agent}@{tail}"


def parse_field(spec: str | Sequence[str]) -> list[FieldEntry]:
    """Parse ``"heuristic,ppo@checkpoints/ppo_v25b_s0/iter_160.pt"`` into entries.

    The ``name@path`` form is the only place a checkpoint is named, so a field is one flag and one
    string -- which is also what makes the field reproducible from the results file.
    """
    parts = list(spec) if not isinstance(spec, str) else [p.strip() for p in spec.split(",")]
    entries: list[FieldEntry] = []
    for part in parts:
        if not part.strip():
            continue
        agent, sep, checkpoint = part.strip().partition("@")
        if sep and not checkpoint:
            raise LadderError(f"field entry {part!r} has '@' but no checkpoint path")
        if checkpoint and agent in _TRAINING_ONLY_AGENTS:
            instead = _TRAINING_ONLY_AGENTS[agent]
            raise LadderError(
                f"{part!r}: {agent!r} is the TRAINING rollout agent and forces sample=True, so it "
                f"plays a different -- and measurably weaker -- policy than the same checkpoint "
                f"deployed greedily (~9 points vs the heuristic on v25b). Every published number "
                f"in this project scores these checkpoints through {instead!r}. "
                f"Use {instead}@{checkpoint} instead, and add --sample if you deliberately want "
                f"the sampled policy."
            )
        entries.append(FieldEntry(agent=agent, checkpoint=checkpoint or None))

    if len(entries) < 2:
        raise LadderError(f"a ladder needs at least 2 agents, got {len(entries)}")

    labels = [e.label for e in entries]
    duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
    if duplicates:
        raise LadderError(
            f"duplicate field entries: {', '.join(duplicates)}. Two entries that differ only by a "
            "checkpoint directory name collide; rename or drop one."
        )
    return entries


# --------------------------------------------------------------------------- #
# Playing the round-robin
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairResult:
    """One unordered pair, scored from ``a``'s side. ``b``'s side is the mirror."""

    a: str
    b: str
    wins: int  # a's wins
    losses: int  # a's losses
    ties: int
    games: int  # finished games; may be < requested if a battle never completed

    @property
    def score(self) -> float:
        """``a``'s score rate. Ties count a half, which is the whole reason for this dataclass."""
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0


def _counters(player: Any) -> tuple[int, int, int]:
    """(wins, losses, finished) as poke-env currently sees them.

    These are cumulative over ``player._battles``, so a pair is scored by DIFFERENCING them around
    the games rather than reading them absolutely -- see ``run_round_robin``.
    """
    return (
        int(player.n_won_battles),
        int(player.n_lost_battles),
        int(player.n_finished_battles),
    )


async def run_round_robin(
    entries: Sequence[FieldEntry],
    battle_format: str,
    n_battles: int,
    *,
    team_pool: str | None = None,
    concurrency: int = 20,
    sample: bool = False,
    loop_penalty: float = DEFAULT_LOOP_PENALTY,
    on_pair: Any = None,
) -> list[PairResult]:
    """Play every unordered pair ``n_battles`` times on the local server.

    Players are built ONCE for the whole tournament and reset between pairs, as poke-env's own
    ``cross_evaluate`` does: a checkpoint-backed agent loads torch weights at construction, so
    rebuilding per pair would pay that cost O(k^2) times instead of O(k).
    """
    # Imported here, not at module scope: `arena` pulls poke-env and the agent registry, and this
    # module is imported by the CLI to build --help.
    from lategame.eval.arena import AGENTS, build_player

    for entry in entries:
        if entry.agent not in AGENTS:
            raise LadderError(
                f"unknown agent {entry.agent!r} in the field. Choose from: {', '.join(AGENTS)}"
            )

    teams: list[str] | None = None
    if team_pool is not None:
        from lategame.teambuilding.pool import TeamPool

        teams = TeamPool.from_packed_file(team_pool).teams

    def _pool(seed: int) -> Any:
        """A fresh pool per player, seeded distinctly, so two players draw independently."""
        if teams is None:
            return None
        from lategame.teambuilding.pool import TeamPool

        return TeamPool(teams, seed=seed)

    players: list[Any] = [
        build_player(
            entry.agent,
            battle_format,
            checkpoint_path=entry.checkpoint,
            sample=sample,
            max_concurrent_battles=concurrency,
            team=_pool(i),
            loop_penalty=loop_penalty,
        )
        for i, entry in enumerate(entries)
    ]

    results: list[PairResult] = []
    try:
        for i, p1 in enumerate(players):
            for j in range(i + 1, len(players)):
                p2 = players[j]

                # SCORED BY DIFFERENCE, not by absolute counter. poke-env's counters run over
                # `player._battles`, which `reset_battles` clears -- but it REFUSES to clear while
                # any battle is unfinished, and a stuck battle must not be allowed to take the
                # tournament down. Suppressing that error alone would silently carry the previous
                # pair's games into this one's totals, so the pair is measured as a delta and the
                # reset becomes hygiene rather than a correctness dependency.
                before = _counters(p1)
                await p1.battle_against(p2, n_battles=n_battles)
                after = _counters(p1)
                wins, losses, finished = (now - was for was, now in zip(before, after, strict=True))

                pair = PairResult(
                    a=entries[i].label,
                    b=entries[j].label,
                    wins=wins,
                    losses=losses,
                    ties=finished - wins - losses,
                    games=finished,
                )
                results.append(pair)
                if on_pair is not None:
                    on_pair(pair)
                for player in (p1, p2):
                    with contextlib.suppress(EnvironmentError):
                        player.reset_battles()
    finally:
        for player in players:
            with contextlib.suppress(Exception):
                await player.ps_client.stop_listening()

    return results


# --------------------------------------------------------------------------- #
# The joint fit
# --------------------------------------------------------------------------- #
def _games_by_label(pairs: Iterable[PairResult]) -> dict[str, list[tuple[str, float]]]:
    """Expand pair totals into one (opponent, score) per GAME, from both sides.

    Glicko's period update is defined over individual games, and expanding is what lets a tie enter
    as a genuine 0.5 rather than being rounded into a win or a loss.
    """
    games: dict[str, list[tuple[str, float]]] = {}
    for pair in pairs:
        games.setdefault(pair.a, [])
        games.setdefault(pair.b, [])
        for label, opponent, wins, losses in (
            (pair.a, pair.b, pair.wins, pair.losses),
            (pair.b, pair.a, pair.losses, pair.wins),
        ):
            games[label].extend([(opponent, 1.0)] * wins)
            games[label].extend([(opponent, 0.0)] * losses)
            games[label].extend([(opponent, 0.5)] * pair.ties)
    return games


def _clamp(rating: float) -> float:
    return max(DEFAULT_RATING - RATING_CLAMP, min(DEFAULT_RATING + RATING_CLAMP, rating))


@dataclass(frozen=True)
class FieldFit:
    """The fitted field, plus enough diagnostics to tell a converged fit from a stopped one."""

    ratings: dict[str, Rating]
    anchor: str
    sweeps: int
    max_delta: float
    converged: bool
    #: No finite MLE exists (scored 0 or 1 against the WHOLE field).
    unbounded: tuple[str, ...]
    #: Rating sits at ``RATING_CLAMP``. A superset of ``unbounded``: a merely-terrible agent can
    #: have a finite MLE that still reaches the bound, and ``converged`` stays True when it does --
    #: so this is the flag that says "read this row as a bound, not a rating".
    clamped: tuple[str, ...]


def rate_field(
    pairs: Iterable[PairResult],
    *,
    anchor: str = DEFAULT_ANCHOR,
    max_sweeps: int = MAX_SWEEPS,
    tol: float = TOL,
) -> FieldFit:
    """Fit every agent's rating jointly from the round-robin matrix.

    See the module docstring for why this iterates, why it needs an anchor, and why RD is computed
    exactly once at the end.
    """
    games = _games_by_label(pairs)
    labels = sorted(games)
    if len(labels) < 2:
        raise LadderError(f"a ladder needs at least 2 rated agents, got {len(labels)}")
    if anchor not in games:
        raise LadderError(
            f"anchor {anchor!r} is not in the field ({', '.join(labels)}). The anchor fixes the "
            "additive gauge, so it must be an agent that actually played."
        )

    # An agent that never scored (or never dropped a point) against the WHOLE field has no finite
    # maximum-likelihood rating. Detect it up front so the flag describes the field, not whichever
    # sweep the iteration happened to stop on.
    unbounded = tuple(
        lab
        for lab in labels
        if games[lab]
        and (
            all(score == 0.0 for _, score in games[lab])
            or all(score == 1.0 for _, score in games[lab])
        )
    )

    current = {lab: DEFAULT_RATING for lab in labels}
    sweeps = 0
    max_delta = float("inf")
    for sweep in range(1, max_sweeps + 1):
        sweeps = sweep
        previous = dict(current)
        updated: dict[str, float] = {}
        for lab in labels:
            # Opponents are held at the PREVIOUS sweep's estimates, so every agent in a sweep sees
            # the same field and the result does not depend on the order `labels` is walked in.
            results = [
                (Rating(previous[opponent], MAX_RD), score) for opponent, score in games[lab]
            ]
            step = update(Rating(previous[lab], MAX_RD), results).rating - previous[lab]
            # Damped, because the undamped step double-corrects every difference. See DAMPING.
            updated[lab] = previous[lab] + DAMPING * step

        # Gauge fix: only differences are identified, so pin the anchor and shift the rest with it.
        shift = DEFAULT_RATING - updated[anchor]
        current = {lab: _clamp(value + shift) for lab, value in updated.items()}

        max_delta = max(abs(current[lab] - previous[lab]) for lab in labels)
        if max_delta < tol:
            break

    # RD once, from a SINGLE period against the converged field -- never accumulated over sweeps.
    ratings = {
        lab: Rating(
            current[lab],
            update(
                Rating(current[lab], MAX_RD),
                [(Rating(current[opponent], MAX_RD), score) for opponent, score in games[lab]],
            ).rd,
        )
        for lab in labels
    }
    return FieldFit(
        ratings=ratings,
        anchor=anchor,
        sweeps=sweeps,
        max_delta=max_delta,
        converged=max_delta < tol,
        unbounded=unbounded,
        clamped=tuple(
            lab for lab in labels if abs(current[lab] - DEFAULT_RATING) >= RATING_CLAMP - 1e-6
        ),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarize_ladder(
    entries: Sequence[FieldEntry],
    pairs: Sequence[PairResult],
    fit: FieldFit,
    *,
    battle_format: str,
    n_battles: int,
) -> dict[str, Any]:
    """The payload: per-agent ratings, the full matrix, and the basis they mean nothing without."""
    by_label = {entry.label: entry for entry in entries}
    games = _games_by_label(pairs)

    agents: list[dict[str, Any]] = []
    for label, rating in fit.ratings.items():
        entry = by_label.get(label)
        played = games.get(label, [])
        wins = sum(1 for _, score in played if score == 1.0)
        losses = sum(1 for _, score in played if score == 0.0)
        ties = sum(1 for _, score in played if score == 0.5)
        n = len(played)
        agents.append(
            {
                "label": label,
                "agent": entry.agent if entry else label,
                "checkpoint": entry.checkpoint if entry else None,
                "games": n,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "score_rate": round((wins + 0.5 * ties) / n, 4) if n else None,
                "glicko": round(rating.rating, 1),
                "glicko_rd": round(rating.rd, 1),
                "gxe": round(gxe(rating), 4),
                "unbounded": label in fit.unbounded,
                "clamped": label in fit.clamped,
            }
        )
    agents.sort(key=lambda row: row["glicko"], reverse=True)

    matrix: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        for label, opponent, wins, losses in (
            (pair.a, pair.b, pair.wins, pair.losses),
            (pair.b, pair.a, pair.losses, pair.wins),
        ):
            total = pair.games
            matrix.setdefault(label, {})[opponent] = {
                "wins": wins,
                "losses": losses,
                "ties": pair.ties,
                "games": total,
                "score": round((wins + 0.5 * pair.ties) / total, 4) if total else None,
            }

    summary: dict[str, Any] = {
        "battle_format": battle_format,
        "n_battles_per_pair": n_battles,
        "field": [entry.label for entry in entries],
        "pairs": len(pairs),
        "anchor": fit.anchor,
        "converged": fit.converged,
        "sweeps": fit.sweeps,
        "max_delta": round(fit.max_delta, 9),
        "unbounded": list(fit.unbounded),
        "clamped": list(fit.clamped),
        "agents": agents,
        "matrix": matrix,
        "gxe_basis": _GXE_BASIS.format(anchor=fit.anchor, rating=DEFAULT_RATING),
    }
    if not fit.converged:
        summary["warning"] = (
            f"the joint fit did NOT converge in {fit.sweeps} sweeps (max_delta "
            f"{fit.max_delta:.6f}); the ratings below are where the iteration stopped, not a fit."
        )
    if fit.unbounded:
        summary["unbounded_note"] = (
            f"{', '.join(fit.unbounded)} scored 0 or 1 against the ENTIRE field, so no finite "
            "maximum-likelihood rating exists."
        )
    if fit.clamped:
        summary["clamped_note"] = (
            f"{', '.join(fit.clamped)} reached the {RATING_CLAMP:.0f}-point divergence guard "
            f"around {DEFAULT_RATING:.0f}. READ THOSE ROWS AS A BOUND, NOT A RATING -- note that "
            "`converged` stays true when this happens, because a clamped value stops moving."
        )
    return summary


def write_results(path: str | Path, cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    """Write the ladder file atomically, mirroring ``live.telemetry.write_results``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, **dict(cfg), **dict(summary)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def format_table(summary: Mapping[str, Any]) -> str:
    """The human-readable standings, for the CLI."""
    rows = summary["agents"]
    width = max([len(str(row["label"])) for row in rows] + [5])
    lines = [
        f"  {'agent'.ljust(width)}  {'W-L-T':>13}  {'score':>6}  "
        f"{'Glicko':>8}  {'RD':>6}  {'GXE':>7}"
    ]
    for row in rows:
        record = f"{row['wins']}-{row['losses']}-{row['ties']}"
        score = "  n/a " if row["score_rate"] is None else f"{row['score_rate']:.3f}"
        flag = "  (CLAMPED -- a bound, not a rating)" if row["clamped"] else ""
        lines.append(
            f"  {str(row['label']).ljust(width)}  {record:>13}  {score:>6}  "
            f"{row['glicko']:>8.1f}  {row['glicko_rd']:>6.1f}  {row['gxe']:>7.4f}{flag}"
        )
    return "\n".join(lines)


__all__ = [
    "DAMPING",
    "DEFAULT_ANCHOR",
    "DEFAULT_FIELD",
    "DEFAULT_LOOP_PENALTY",
    "MAX_SWEEPS",
    "RATING_CLAMP",
    "SCHEMA",
    "TOL",
    "FieldEntry",
    "FieldFit",
    "LadderError",
    "PairResult",
    "format_table",
    "parse_field",
    "rate_field",
    "run_round_robin",
    "summarize_ladder",
    "write_results",
]

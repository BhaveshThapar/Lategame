"""Differential selection bias between PPO arms of different length (plan.md §13).

WHY THIS EXISTS. Each seed reports the ``argmax`` over its curve of noisy ``eval_n``-battle
evaluations, then that one checkpoint is re-scored at large N. Re-scoring kills the winner's curse
on the curve *value*, but not on the *selection*: an arm with more checkpoints gets more draws at
the argmax and lands on a truly-better one more often, so a longer arm beats a shorter one by a
margin that is partly procedural rather than real. Build 25 established that this does NOT cancel
once arms differ in length, and pre-registered that the bias be SUBTRACTED before any dose is
declared -- a contrast must clear both ``p < alpha`` and ``diff - bias > 0``.

Build 25 simulated this ad hoc and the code was never committed, so the numbers in its
pre-registration cannot be re-derived. This script is that simulation, committed.

TWO ESTIMATES, and they differ because the model has to be fit to the arm's schedule:

``estimate``  -- fits sigma_b, the true checkpoint-to-checkpoint sd of strength within an arm's
                 plateau, from real curves. The plateau is the FROZEN-schedule window
                 (``iter > anneal_iters``); an arm whose anneal runs the full length has no
                 plateau at all and must not be used, because its "late window" variance is
                 dominated by the still-moving lr/entropy schedule, i.e. by trend rather than by
                 dispersion. Build 25 estimated sigma_b = 0.0428 from Build 23/24 curves, which
                 were exactly that un-pinned shape; see the note in ``estimate_sigma_b``.

``simulate``  -- Monte-Carlos the argmax under the null that the arms are EQUALLY STRONG, so any
                 difference in the selected checkpoint's true strength is pure selection. The
                 trend is deliberately excluded from the null: growth in true strength with more
                 updates is the effect under test, not the bias being corrected for.

Usage:
    python scripts/selection_bias_sim.py estimate --gate results/ppo_ou_gate_v25a_s0.json ...
    python scripts/selection_bias_sim.py simulate --arm v25b:160 --arm v26a:240 --arm v26b:320 \
        --anneal-iters 80 --sigma-b 0.025 --out results/selection_bias_v26.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Selection happens over the checkpoints that can plausibly win. Empirically every anneal-pinned
# Build 25 arm's `best_iter` landed in the frozen window (v25a 118/105/114, v25b 132/125/159 --
# all > 80), because the pre-anneal checkpoints are genuinely weaker and never take the argmax.
# The effective draw count is therefore `iters - anneal_iters`, NOT `iters`.
_DEFAULT_TRIALS = 200_000


def _norm_ppf(p: float) -> float:
    """Standard-normal quantile, by bisection on ``math.erf``.

    scipy is deliberately not in this env (see the note in ``scripts/seed_strength_gate.py``), and
    this is the only distributional call here, so it is cheaper to invert the error function than
    to take the dependency.
    """
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _chi2_ppf(p: float, dof: int) -> float:
    """Chi-square quantile via Wilson-Hilferty; exact enough at the dof this script sees (100s)."""
    z = _norm_ppf(p)
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


# --------------------------------------------------------------------------- #
# sigma_b estimation from real curves.
# --------------------------------------------------------------------------- #


@dataclass
class SigmaEstimate:
    """Pooled within-plateau strength dispersion, net of the binomial eval floor."""

    sigma_b: float  # point estimate (clamped at 0)
    sigma_b_upper: float  # conservative upper confidence bound
    resid_var: float
    binom_var: float
    dof: int
    slopes: list[float]
    detrended: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sigma_b": round(self.sigma_b, 5),
            "sigma_b_upper": round(self.sigma_b_upper, 5),
            "pooled_resid_var": round(self.resid_var, 6),
            "pooled_binom_var": round(self.binom_var, 6),
            "dof": self.dof,
            "slope_per_iter": [round(s, 5) for s in self.slopes],
            "detrended": self.detrended,
        }


def estimate_sigma_b(
    gates: list[str | Path],
    eval_n: int = 100,
    detrend: bool = True,
    conf: float = 0.95,
) -> SigmaEstimate:
    """Fit sigma_b from the frozen-schedule plateau of each seed's curve.

    Observed curve variance is ``sigma_b^2 + p(1-p)/eval_n``; subtracting the binomial floor
    leaves the real dispersion. ``detrend`` first removes a linear drift across the plateau --
    without it, an arm that is still improving (which, per Build 25, they all are) books its own
    climb as dispersion and inflates the bias correction.
    """
    resid_ss = 0.0
    binom_terms: list[float] = []
    dof = 0
    slopes: list[float] = []

    for path in gates:
        gate = json.loads(Path(path).read_text())
        anneal = gate.get("anneal_iters")
        if anneal is None:
            raise SystemExit(
                f"{path}: anneal_iters is null -- the lr/entropy schedule runs the whole arm, so "
                "there is no frozen plateau to estimate dispersion from. Build 25's sigma_b was "
                "estimated from exactly this shape and is inflated by schedule trend as a result. "
                "Pass only anneal-pinned arms."
            )
        for rec in gate["records"]:
            pts = [p for p in rec["curve"] if p["iter"] > anneal]
            if len(pts) < 10:
                continue
            y = np.array([p["vs_heuristic"] for p in pts], dtype=float)
            x = np.array([p["iter"] for p in pts], dtype=float)
            n = len(y)

            if detrend:
                design = np.vstack([np.ones(n), x - x.mean()]).T
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                resid = y - design @ beta
                k = 2
                slopes.append(float(beta[1]))
            else:
                resid = y - y.mean()
                k = 1
                slopes.append(float("nan"))

            resid_ss += float((resid**2).sum())
            dof += n - k
            mean = float(y.mean())
            binom_terms.append(mean * (1 - mean) / eval_n)

    if dof <= 0:
        raise SystemExit("no usable plateau points across the supplied gates")

    resid_var = resid_ss / dof
    binom_var = float(np.mean(binom_terms))
    point = max(resid_var - binom_var, 0.0)
    # Upper bound on the residual variance (chi-square, lower tail), then net off the binomial
    # floor. sigma_b sits at the eval floor in practice, so the BOUND is what the correction
    # should be built on -- "not resolvable" is not the same as "zero".
    upper_resid = dof * resid_var / _chi2_ppf(1 - conf, dof)
    upper = max(upper_resid - binom_var, 0.0)

    return SigmaEstimate(
        sigma_b=point**0.5,
        sigma_b_upper=upper**0.5,
        resid_var=resid_var,
        binom_var=binom_var,
        dof=dof,
        slopes=slopes,
        detrended=detrend,
    )


# --------------------------------------------------------------------------- #
# Monte-Carlo of the argmax selection.
# --------------------------------------------------------------------------- #


def arm_bias(
    draws: int,
    sigma_b: float,
    mu: float,
    eval_n: int,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """E[true strength of the argmax checkpoint] - mu, under equally-strong checkpoints.

    ``draws`` noisy evaluations per trial; the truth is iid N(mu, sigma_b) across checkpoints and
    the observation is Binomial(eval_n, truth)/eval_n. Returns the inflation the argmax carries
    into the re-scored gate.
    """
    if sigma_b <= 0:
        return 0.0
    total = 0.0
    done = 0
    # Chunked so a 200k x 320 draw matrix never materialises at once.
    chunk = max(1, min(trials, 4_000_000 // max(draws, 1)))
    while done < trials:
        m = min(chunk, trials - done)
        truth = rng.normal(mu, sigma_b, size=(m, draws))
        np.clip(truth, 1e-6, 1 - 1e-6, out=truth)
        obs = rng.binomial(eval_n, truth) / eval_n
        picked = truth[np.arange(m), obs.argmax(axis=1)]
        total += float(picked.sum())
        done += m
    return total / trials - mu


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    est = sub.add_parser("estimate", help="fit sigma_b from anneal-pinned gate curves")
    est.add_argument("--gate", action="append", required=True, help="gate JSON; pass 2+ times")
    est.add_argument("--eval-n", type=int, default=100)
    est.add_argument("--no-detrend", action="store_true", help="book plateau drift as dispersion")
    est.add_argument("--out")

    sim = sub.add_parser("simulate", help="differential bias for a set of arm lengths")
    sim.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL:ITERS",
        help="arm label and its iteration count; pass 2+ times -- EVERY PAIR is reported",
    )
    sim.add_argument("--anneal-iters", type=int, required=True, help="frozen-schedule start")
    sim.add_argument(
        "--sigma-b",
        type=float,
        action="append",
        required=True,
        help="pass 2+ times for a sensitivity sweep -- the correction should be reported at the "
        "MOST CONSERVATIVE value it survives, not at the best-fitting one",
    )
    sim.add_argument("--mu", type=float, default=0.62, help="plateau win rate the arms share")
    sim.add_argument("--eval-n", type=int, default=100)
    sim.add_argument("--trials", type=int, default=_DEFAULT_TRIALS)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--out")

    args = ap.parse_args()

    if args.cmd == "estimate":
        got = estimate_sigma_b(args.gate, args.eval_n, detrend=not args.no_detrend)
        payload = {"gates": args.gate, "eval_n": args.eval_n, **got.to_dict()}
        print(json.dumps(payload, indent=2))
        if got.sigma_b == 0.0:
            print(
                "\nNOTE: the point estimate is at or below the binomial eval floor -- within the "
                "frozen window the checkpoints are not resolvably different in true strength. "
                f"Use the upper bound ({got.sigma_b_upper:.4f}) for the correction.",
            )
    else:
        arms = []
        for spec in args.arm:
            label, _, iters = spec.partition(":")
            if not iters.isdigit():
                raise SystemExit(f"--arm expects LABEL:ITERS, got {spec!r}")
            arms.append((label, int(iters)))
        rng = np.random.default_rng(args.seed)
        for label, iters in arms:
            if iters - args.anneal_iters < 1:
                raise SystemExit(f"{label}: iters {iters} <= anneal_iters {args.anneal_iters}")

        sweep = []
        for sigma_b in args.sigma_b:
            per_arm = {
                label: {
                    "iters": iters,
                    "draws": iters - args.anneal_iters,
                    "bias": round(
                        arm_bias(
                            iters - args.anneal_iters,
                            sigma_b,
                            args.mu,
                            args.eval_n,
                            args.trials,
                            rng,
                        ),
                        5,
                    ),
                }
                for label, iters in arms
            }
            pairs = [
                {
                    "contrast": f"{a} -> {b}",
                    "differential_bias": round(per_arm[b]["bias"] - per_arm[a]["bias"], 5),
                }
                for i, (a, _) in enumerate(arms)
                for b, _ in arms[i + 1 :]
            ]
            sweep.append({"sigma_b": sigma_b, "arms": per_arm, "contrasts": pairs})

        payload = {
            "mu": args.mu,
            "eval_n": args.eval_n,
            "anneal_iters": args.anneal_iters,
            "trials": args.trials,
            "seed": args.seed,
            "sweep": sweep,
        }
        print(json.dumps(payload, indent=2))

        contrasts = [p["contrast"] for p in sweep[0]["contrasts"]]
        head = " | ".join(contrasts)
        print(f"\n| sigma_b | {head} |")
        print("|---|" + "---|" * len(contrasts))
        for entry in sweep:
            cells = " | ".join(f"{p['differential_bias']:+.4f}" for p in entry["contrasts"])
            print(f"| {entry['sigma_b']:.4f} | {cells} |")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

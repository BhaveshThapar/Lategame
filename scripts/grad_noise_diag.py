"""Build 20 (docs/RESULTS.md): is the PPO plateau a per-iteration SAMPLE-BUDGET ceiling?

Build 18 plateaued PPO self-play at ``vs_heuristic`` ~0.45 (n=300) and Build 19's ent/lr
schedule did not move it (NULL). The remaining cheap suspect is the rollout budget:
``games_per_opp=16`` x 3 opponents = 48 battles ~ 2K transitions per iteration.

The naive framing -- "a 3x buffer buys 3x the gradient steps" -- is already falsified by the
code, so do not spend seeds on it un-qualified:

    advantages are normalized PER-BUFFER   (ppo.py: adv = (adv - adv.mean()) / adv.std())
    the epoch loop KL-early-stops          (ppo.py: mean epoch approx_kl > 1.5 * target_kl)

Per-iteration policy *displacement* is therefore governed by the trust region, not by the step
count: a 3x buffer gives ~3x minibatches per epoch, the policy travels further per epoch, and
the KL guard just cuts the epochs short. What a bigger budget actually buys is a LOWER-VARIANCE
ESTIMATE OF THE GRADIENT DIRECTION at a fixed trust-region step. That is measurable offline,
before spending ~45 min/seed.

Build 19's lesson was that a gap can be a COST rather than HEADROOM (the +0.140 greedy-sampled
gap was the price of sampling, not slack in the argmax). This probe exists to avoid repeating
that: it asks whether the update direction is actually noise-dominated at the current budget.

Three measurements on rollouts collected from a SHIPPED v19 checkpoint (no training, no change to
rotomai/):

  B1 cross-rollout cosine -- cos(g_i, g_j) between gradients from INDEPENDENT rollouts at the
                   run's real budget, plus a scaling curve at smaller batch sizes. Near 0 at the
                   budget => two PPO iterations would step in unrelated directions, so more
                   samples straighten it. Run in TWO arms (see below).
  B2 noise scale  -- B_simple = tr(Sigma) / |G|^2 (McCandlish et al. 2018), the batch size at
                   which gradient noise stops dominating. B_simple >> buffer => a bigger budget
                   pays; B_simple << buffer => we are already past the useful batch size.
  B3 value EV     -- explained variance of the critic on the buffer. EV ~ 0 means advantages
                   collapse toward Monte-Carlo returns (very high variance) and the CRITIC, not
                   the budget, is the lever. ``vmae`` is logged per iter today; EV is not.

TWO ARMS, because ``--games-per-opp`` can only reduce ONE of the two variances in an iteration:

  same_mix  -- every rollout faces the SAME league draw, so the only thing that varies is what a
               bigger budget buys: more battles against the opponents the iteration already drew.
               THE VERDICT READS THIS ARM.
  fresh_mix -- every rollout re-draws the league (``pop_size=2`` uniform samples + the anchor), as
               successive real iterations do. The extra disagreement here is opponent-SELECTION
               variance, which no per-opponent budget can touch.

If same_mix is clean while fresh_mix is not, the binding variance is WHICH opponents got drawn --
a bigger budget is then predicted NULL for a reason a single-mix cosine could never reveal, and
the lever is ``pop_size`` / league averaging instead.

Why INDEPENDENT rollouts and not two halves of one buffer: a single rollout is ONE draw of league
opponents, teams and episodes. Two halves of it inherit that same draw, so they see only turn-level
noise and agree far more than two real PPO iterations ever would -- the split-half cosine is biased
toward AGREEMENT, which is the dangerous direction (it would overstate how well the direction is
pinned down and could manufacture a SIGNAL_LIMITED verdict, wrongly skipping Stage B). PPO's
per-buffer advantage centering (ppo.py:169) couples the rows on top of that. Separate rollouts
re-draw everything the learner actually faces, and pose the operative question directly: each PPO
iteration draws a fresh buffer, so do two of them agree on which way to step?

Gradients are evaluated AT THE ROLLOUT POLICY (theta_old), where the PPO ratio is 1 and the
clipped surrogate reduces to the vanilla policy gradient -- that is exactly the first update the
iteration would take, and it makes the measured gradient unambiguous.

Pre-registered read (docs/RESULTS.md, Build 20), on the POLICY-term gradient:
  NOISE_LIMITED  -- cos < 0.30 at the budget and B_simple > 2x the buffer: the direction is
                    noise-bound. Run Stage B (--games-per-opp 48) with the mechanism confirmed.
  SIGNAL_LIMITED -- cos > 0.50 and B_simple < the buffer: the direction is already well
                    estimated; a bigger budget is predicted NULL. Skip Stage B, go to capacity.
  AMBIGUOUS      -- otherwise. Run Stage B anyway (it is only ~45 min/seed).

Caveat stated in the output: Adam preconditions the raw gradient, so cosine similarity is a
proxy for the true update direction. It is the standard measure -- treat it as directional
evidence, not proof.

    python scripts/grad_noise_diag.py \
      --policy checkpoints/ppo_ou_sched_s0/iter_47.pt \
      --reference checkpoints/ppo_ou_sched_s0/iter_10.pt \
      --reference checkpoints/ppo_ou_sched_s0/iter_50.pt \
      --init checkpoints/offrl_gen9ou_v7_s0.pt \
      --league-dir checkpoints/ppo_ou_sched_s0 \
      --games-per-opp 16 --loop-penalty 4 --rollouts 3 --splits 20
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rotomai.data.collect import PlayerSpec
from rotomai.data.rollout import RolloutBuffer, collect_rollout
from rotomai.features.encoder import OBS_DIM, OBS_VERSION
from rotomai.model.actor_critic import (
    hl_gauss_target,
    load_actor_critic_weights,
    value_support,
)
from rotomai.model.factory import build_model
from rotomai.teambuilding.pool import TeamPool
from rotomai.train.offline_rl import _value_ce
from rotomai.train.ppo import PPOConfig, _policy_stats, compute_gae
from rotomai.train.selfplay import _sample_league

# Pre-registered thresholds (docs/RESULTS.md, Build 20), read on the POLICY-term split-half cosine.
NOISY_COS = 0.30  # below this, the update direction is noise-dominated at this batch size
CLEAN_COS = 0.50  # above this, the direction is already well estimated
# B_simple is compared against the buffer size N: > NOISE_SCALE_MULT * N argues for more samples.
NOISE_SCALE_MULT = 2.0


@dataclass
class GradSpec:
    """The loss terms a gradient is taken through."""

    clip_eps: float
    ent_coef: float
    value_coef: float


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two flat gradient vectors; 0.0 if either is degenerate."""
    na = float(a.norm())
    nb = float(b.norm())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def explained_variance(value: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var(returns - value) / Var(returns). 1.0 = perfect critic, <=0 = no better than mean."""
    var_ret = float(returns.var(unbiased=False))
    if var_ret <= 0.0:
        return 0.0
    return float(1.0 - (returns - value).var(unbiased=False) / var_ret)


def noise_scale(sq_small: float, b_small: int, sq_big: float, b_big: int) -> dict[str, Any]:
    """Gradient noise scale from mean squared grad norms at two batch sizes.

    E|g_B|^2 = |G|^2 + tr(Sigma)/B  (unbiased minibatch gradient), so two batch sizes pin both
    terms (McCandlish et al. 2018, "An Empirical Model of Large-Batch Training"):

        |G|^2     = (B_big*E|g_big|^2 - B_small*E|g_small|^2) / (B_big - B_small)
        tr(Sigma) = (E|g_small|^2 - E|g_big|^2) / (1/B_small - 1/B_big)
        B_simple  = tr(Sigma) / |G|^2

    ``B_simple`` is the batch size at which gradient noise stops dominating the true gradient.
    A ``|G|^2`` estimate at or below 0 means noise swamps the signal so hard that the true
    gradient is not resolvable from these two batch sizes -- itself strong noise-limited
    evidence, reported as an infinite ``b_simple`` rather than a silent NaN.
    """
    if b_big <= b_small:
        raise ValueError(f"need b_big > b_small, got {b_small} and {b_big}")
    g2 = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
    trace = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
    resolvable = g2 > 0.0 and trace > 0.0
    b_simple = trace / g2 if resolvable else math.inf
    return {
        "g_norm_sq": round(g2, 8),
        "trace_sigma": round(trace, 6),
        "b_simple": b_simple if math.isinf(b_simple) else round(b_simple, 1),
        "resolvable": resolvable,
        "b_small": b_small,
        "b_big": b_big,
    }


def _flat_grad(
    model: nn.Module, loss: torch.Tensor, params: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Flattened dL/dtheta over every trainable parameter (zeros where the loss is independent)."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params, strict=True)
        ]
    )


def _losses(
    model: nn.Module,
    buffer: RolloutBuffer,
    adv: torch.Tensor,
    target: torch.Tensor,
    idx: torch.Tensor,
    spec: GradSpec,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(policy_loss, total_loss, mean|ratio-1|)`` over rows ``idx``, mirroring ``ppo_update``.

    At theta_old the ratio is 1, the clip is inactive, and this reduces to the vanilla policy
    gradient -- the first update the iteration would actually take. ``mean|ratio-1|`` is returned
    as a sanity check that we really are at theta_old.
    """
    new_log_prob, entropy, value_logits = _policy_stats(
        model, buffer.obs[idx], buffer.mask[idx], buffer.action[idx]
    )
    ratio = (new_log_prob - buffer.old_log_prob[idx]).exp()
    a = adv[idx]
    surr1 = ratio * a
    surr2 = ratio.clamp(1.0 - spec.clip_eps, 1.0 + spec.clip_eps) * a
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = _value_ce(value_logits, target[idx])
    total = policy_loss + spec.value_coef * value_loss - spec.ent_coef * entropy.mean()
    return policy_loss, total, float((ratio - 1.0).abs().mean().detach())


def gradient_stats(
    model: nn.Module,
    rollouts: list[tuple[RolloutBuffer, torch.Tensor, torch.Tensor]],
    sizes: list[int],
    splits: int,
    spec: GradSpec,
    seed: int,
) -> dict[str, Any]:
    """Cross-rollout gradient agreement + noise scale, on the policy-term and total-loss gradients.

    The two gradients in each cosine come from INDEPENDENT rollouts, never from two halves of one
    buffer. A single rollout fixes the league draw, the teams and the episodes; its halves inherit
    all of that and so agree more than two real PPO iterations would. That bias runs toward
    AGREEMENT -- it would overstate how well the direction is pinned down. Separate rollouts
    re-draw every source of variance the learner actually faces.
    """
    model.eval()  # dropout off, exactly as ppo_update runs it
    params = [p for p in model.parameters() if p.requires_grad]
    n_min = min(len(buf) for buf, _, _ in rollouts)
    sizes = sorted({s for s in sizes if s < n_min} | {n_min})
    generator = torch.Generator().manual_seed(seed)
    pairs = list(itertools.combinations(range(len(rollouts)), 2))

    terms = ("policy", "total")
    cos: dict[str, dict[int, list[float]]] = {t: {s: [] for s in sizes} for t in terms}
    sq: dict[str, dict[int, list[float]]] = {t: {s: [] for s in sizes} for t in terms}
    ratio_devs: list[float] = []

    for size in sizes:
        for _ in range(splits):
            grads: dict[str, list[torch.Tensor]] = {t: [] for t in terms}
            for buffer, adv, target in rollouts:
                idx = torch.randperm(len(buffer), generator=generator)[:size]
                policy_loss, total_loss, ratio_dev = _losses(model, buffer, adv, target, idx, spec)
                ratio_devs.append(ratio_dev)
                grads["policy"].append(_flat_grad(model, policy_loss, params))
                grads["total"].append(_flat_grad(model, total_loss, params))
            for term in terms:
                for i, j in pairs:
                    cos[term][size].append(cos_sim(grads[term][i], grads[term][j]))
                sq[term][size].extend(float(g.norm() ** 2) for g in grads[term])

    out: dict[str, Any] = {
        "n_transitions": [len(buf) for buf, _, _ in rollouts],
        "n_rollouts": len(rollouts),
        "budget_size": n_min,
        "sizes": sizes,
        "splits": splits,
        # Sanity: we should be AT theta_old, where the PPO ratio is exactly 1.
        "mean_abs_ratio_minus_1": round(sum(ratio_devs) / len(ratio_devs), 6),
    }
    for term in terms:
        curve = {
            str(size): {
                "cos_mean": round(sum(cos[term][size]) / len(cos[term][size]), 4),
                "cos_std": round(_std(cos[term][size]), 4),
                "grad_norm_sq_mean": round(sum(sq[term][size]) / len(sq[term][size]), 8),
            }
            for size in sizes
        }
        small, big = sizes[0], sizes[-1]
        scale = noise_scale(
            sum(sq[term][small]) / len(sq[term][small]),
            small,
            sum(sq[term][big]) / len(sq[term][big]),
            big,
        )
        out[term] = {
            "cos_by_size": curve,
            # The headline: do two INDEPENDENT rollouts at the current budget agree on a direction?
            "cos_at_budget": curve[str(n_min)]["cos_mean"],
            "noise_scale": scale,
            "signal_fraction_at_budget": signal_fraction(scale["b_simple"], n_min),
        }
    return out


def signal_fraction(b_simple: float, batch: int) -> float:
    """Share of a batch gradient's squared norm that is signal: |G|^2 / E|g_B|^2 = 1/(1 + B*/B).

    Equals the expected cosine between two independent batch gradients, so it is the model-based
    twin of the measured ``cos_at_budget`` -- agreement between them corroborates both.
    """
    if math.isinf(b_simple):
        return 0.0
    return round(1.0 / (1.0 + b_simple / batch), 4)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def verdict(policy_stats: dict[str, Any], n: int) -> tuple[str, str]:
    """NOISE_LIMITED / SIGNAL_LIMITED, pre-registered (docs/RESULTS.md, Build 20), on the
    policy gradient.

    Read on the SAME_MIX arm on purpose: ``--games-per-opp`` buys more battles against the opponent
    set an iteration has already drawn, so within-mix sampling noise is the only variance it can
    reduce. The fresh_mix arm is reported alongside as the ceiling that budget cannot touch.
    """
    cos = float(policy_stats["cos_at_budget"])
    raw = policy_stats["noise_scale"]["b_simple"]
    b_simple = math.inf if raw in (None, "inf") else float(raw)
    noisy = cos < NOISY_COS and b_simple > NOISE_SCALE_MULT * n
    clean = cos > CLEAN_COS and b_simple < n
    scale = "unresolvable (noise swamps the signal)" if math.isinf(b_simple) else f"{b_simple:.0f}"

    if noisy:
        return "NOISE_LIMITED", (
            f"same-mix cos {cos:.3f} < {NOISY_COS} and B_simple {scale} > {NOISE_SCALE_MULT:g}x "
            f"the {n}-transition budget: two rollouts against the SAME opponents still disagree on "
            "the update direction, so within-mix sampling noise dominates it -- exactly the "
            "variance more battles reduce. Run Stage B (--games-per-opp 48)."
        )
    if clean:
        return "SIGNAL_LIMITED", (
            f"same-mix cos {cos:.3f} > {CLEAN_COS} and B_simple {scale} < the {n}-transition "
            "budget: rollouts against the same opponents already agree on the direction, so more "
            "battles per opponent buy little -- Stage B is predicted NULL. Skip it; pivot to "
            "capacity."
        )
    return "AMBIGUOUS", (
        f"same-mix cos {cos:.3f} and B_simple {scale} against a {n}-transition budget match "
        "neither pre-registered arm -- no clean call. Run Stage B anyway (~45 min/seed)."
    )


def _iter_of(path: str) -> int:
    """Iteration number from an ``iter_NN.pt`` checkpoint name; 0 for a warm-start."""
    m = re.search(r"iter_(\d+)\.pt$", path)
    return int(m.group(1)) if m else 0


def _league_for(
    policy: str, init: str, league_dir: str | None, pop_size: int, seed: int
) -> list[str]:
    """The league the real run would have sampled from at this policy's iteration.

    ``run_ppo`` seeds a league with the warm-start and appends each ``iter_NN.pt`` as it is
    written, then draws ``pop_size`` uniformly per iteration (``_sample_league``). Rebuild that
    pool up to (but excluding) the policy's own iteration and draw from it the same way.
    """
    league = [init]
    k = _iter_of(policy)
    if league_dir:
        league += [
            str(p) for p in sorted(Path(league_dir).glob("iter_*.pt")) if 0 < _iter_of(str(p)) < k
        ]
    return _sample_league(league, pop_size, random.Random(seed))


async def buffer_for(
    policy: str,
    init: str,
    league_dir: str | None,
    config: PPOConfig,
    seed: int,
) -> RolloutBuffer:
    """One on-policy rollout from ``policy`` against the mix its own iteration faced."""
    opponents = [
        PlayerSpec("offrl", checkpoint_path=p, sample=True)
        for p in _league_for(policy, init, league_dir, config.pop_size, seed)
    ]
    opponents += [PlayerSpec(a) for a in config.anchors]
    print(f"  opponents: {[o.checkpoint_path or o.name for o in opponents]}")
    team = TeamPool.from_packed_file(config.team_pool) if config.team_pool else None
    return await collect_rollout(
        policy,
        opponents,
        config.games_per_opp,
        config.battle_format,
        config.weights,
        config.max_concurrent,
        team=team,
        loop_penalty=config.loop_penalty,
    )


def load_policy(path: str) -> tuple[nn.Module, torch.Tensor, float]:
    """``(model, value_support_centers, hl_gauss_sigma)`` from a PPO/offrl checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("obs_version") != OBS_VERSION or ckpt.get("input_dim") != OBS_DIM:
        raise ValueError(f"encoder mismatch in '{path}' -- expected {OBS_VERSION}/{OBS_DIM}.")
    model = build_model(ckpt)
    load_actor_critic_weights(model, ckpt["state_dict"])
    v_min, v_max, n_bins = float(ckpt["v_min"]), float(ckpt["v_max"]), int(ckpt["n_bins"])
    sigma = PPOConfig.hl_gauss_sigma_bins * (v_max - v_min) / (n_bins - 1)
    return model, value_support(v_min, v_max, n_bins), sigma


def probe(
    policy: str,
    init: str,
    league_dir: str | None,
    config: PPOConfig,
    sizes: list[int],
    splits: int,
    n_rollouts: int,
    seed: int,
) -> dict[str, Any]:
    """Collect INDEPENDENT rollouts from ``policy`` and measure the gradient's signal-to-noise.

    Each rollout is a separate draw at the run's real budget -- exactly what one PPO iteration
    gets -- so cosines across them answer the operative question directly: at this budget, do two
    iterations agree on which way to step?
    """
    print(f"\n{policy}")
    model, centers, sigma = load_policy(policy)
    spec = GradSpec(config.clip_eps, config.ent_coef, config.value_coef)

    arms: dict[str, Any] = {}
    for arm in ("same_mix", "fresh_mix"):
        rollouts: list[tuple[RolloutBuffer, torch.Tensor, torch.Tensor]] = []
        evs: list[float] = []
        episodes = 0
        for r in range(n_rollouts):
            # same_mix pins the league draw across rollouts, so the ONLY thing that varies is what
            # --games-per-opp buys: more battles against a fixed opponent set. fresh_mix re-draws
            # the league each time, adding the opponent-selection variance a real iteration also
            # faces -- which more games_per_opp CANNOT reduce.
            league_seed = seed if arm == "same_mix" else seed + r
            buffer = asyncio.run(buffer_for(policy, init, league_dir, config, league_seed))
            adv, returns = compute_gae(
                buffer.reward, buffer.value, buffer.done, config.gamma, config.gae_lambda
            )
            # Per-buffer normalization, exactly as ppo_update does it.
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            target = hl_gauss_target(returns.clamp(centers[0], centers[-1]), centers, sigma)
            rollouts.append((buffer, adv, target))
            evs.append(explained_variance(buffer.value, returns))
            episodes += int(buffer.done.sum())

        n = min(len(buf) for buf, _, _ in rollouts)
        if min(sizes) >= n:
            raise RuntimeError(
                f"smallest rollout is {n} transitions, under every requested size {sizes}; "
                "raise --games-per-opp."
            )
        stats = gradient_stats(model, rollouts, sizes, splits, spec, seed)
        stats["explained_variance"] = round(sum(evs) / len(evs), 4)
        stats["n_episodes"] = episodes
        arms[arm] = stats

        p = stats["policy"]
        print(
            f"  [{arm}] {n_rollouts} rollouts, {stats['n_transitions']} transitions / "
            f"{episodes} episodes   EV={stats['explained_variance']:.3f}   "
            f"|ratio-1|={stats['mean_abs_ratio_minus_1']:.2e}"
        )
        for size in stats["sizes"]:
            c = p["cos_by_size"][str(size)]
            tag = "  <- budget" if size == n else ""
            print(f"      B={size:>5}  cos {c['cos_mean']:+.3f} +/- {c['cos_std']:.3f}{tag}")
        print(
            f"      B_simple {p['noise_scale']['b_simple']}  signal_frac "
            f"{p['signal_fraction_at_budget']:.3f}"
        )

    budget = arms["same_mix"]["budget_size"]
    call, why = verdict(arms["same_mix"]["policy"], budget)
    out: dict[str, Any] = {"arms": arms, "verdict": call, "why": why}
    out["opponent_draw_dominates"] = _opponent_draw_dominates(arms)
    print(f"  VERDICT {call}")
    return out


def _opponent_draw_dominates(arms: dict[str, Any]) -> bool:
    """True if a fixed opponent set agrees but a re-drawn one does not.

    ``--games-per-opp`` buys more battles against the mix an iteration already drew; it cannot make
    two DIFFERENT league draws agree. So if same_mix is clean while fresh_mix is not, the binding
    variance is opponent SELECTION (pop_size / league averaging), and a bigger budget is predicted
    NULL for a reason no cosine on a single mix would reveal.
    """
    same = float(arms["same_mix"]["policy"]["cos_at_budget"])
    fresh = float(arms["fresh_mix"]["policy"]["cos_at_budget"])
    return same > CLEAN_COS and fresh < NOISY_COS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="checkpoint under test (v19 best iter)")
    ap.add_argument(
        "--reference",
        action="append",
        default=[],
        help="extra checkpoints to probe for the trend (iter_10, iter_50); repeatable",
    )
    ap.add_argument("--init", required=True, help="warm-start (the league's iter-0 member)")
    ap.add_argument("--league-dir", default=None, help="run dir holding iter_NN.pt league members")
    ap.add_argument("--out", default="results/grad_noise_diag.json")
    ap.add_argument("--games-per-opp", type=int, default=16, help="MUST match the run under test")
    ap.add_argument(
        "--rollouts",
        type=int,
        default=3,
        help="independent rollouts per checkpoint; cosines are taken across them (3 => 3 pairs)",
    )
    ap.add_argument("--splits", type=int, default=20, help="random subsamples per batch size")
    ap.add_argument(
        "--sizes",
        default="128,256,512",
        help="sub-budget batch sizes for the scaling curve; the full budget is always appended",
    )
    ap.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--team-pool", default="rotomai/teambuilding/data/teams_gen9ou.packed")
    ap.add_argument("--loop-penalty", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = PPOConfig(
        init=args.init,
        battle_format=args.format,
        team_pool=args.team_pool,
        loop_penalty=args.loop_penalty,
        games_per_opp=args.games_per_opp,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )
    sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})

    probes: dict[str, Any] = {}
    for path in [args.policy, *args.reference]:
        probes[path] = probe(
            path,
            args.init,
            args.league_dir,
            config,
            sizes,
            args.splits,
            args.rollouts,
            args.seed,
        )

    primary = probes[args.policy]
    result: dict[str, Any] = {
        "policy": args.policy,
        "init": args.init,
        "league_dir": args.league_dir,
        "games_per_opp": args.games_per_opp,
        "rollouts_per_checkpoint": args.rollouts,
        "loop_penalty": args.loop_penalty,
        "format": args.format,
        "thresholds": {
            "noisy_cos": NOISY_COS,
            "clean_cos": CLEAN_COS,
            "noise_scale_mult": NOISE_SCALE_MULT,
        },
        "probes": probes,
        "verdict": primary["verdict"],
        "why": primary["why"],
        "opponent_draw_dominates": primary["opponent_draw_dominates"],
        "note": (
            "Gradients are taken at theta_old (PPO ratio 1, clip inactive), so this is the vanilla "
            "policy gradient -- the first update the iteration would take. cos is the similarity "
            "between gradients from INDEPENDENT rollouts, never two halves of one buffer: halves "
            "share that rollout's league/team/episode draw, so they agree more than two real PPO "
            "iterations would, biasing the cosine toward AGREEMENT and risking a false "
            "SIGNAL_LIMITED. TWO ARMS: same_mix pins the league draw, so its cosine isolates the "
            "within-mix sampling noise that --games-per-opp actually reduces -- the VERDICT reads "
            "it. fresh_mix re-draws the league per rollout, adding opponent-selection variance "
            "that no per-opponent budget can touch; if same_mix is clean while fresh_mix is not "
            "(opponent_draw_dominates), the binding variance is WHICH opponents got drawn "
            "(pop_size / league averaging), not how many battles were played against them. "
            "CAVEAT: Adam preconditions the raw gradient, so cosine similarity is a proxy for the "
            "true update direction -- directional evidence, not proof. The verdict reads the "
            "POLICY-term gradient (ent_coef-independent)."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nVERDICT {primary['verdict']}: {primary['why']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

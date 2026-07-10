"""On-policy PPO self-play loop (plan.md, M5 Phase 2).

Why: M5 Phase 1 fixed the critic (offline value MAE ~2.3x better) but win-rate did
not move -- the binding constraint is the offline AWR objective, which can only
*re-weight actions already present* in fixed self-play data. PPO is on-policy: it uses
the improved critic as a low-variance GAE baseline AND lets the policy explore actions
outside the demonstrator distribution, attacking that ceiling directly.

The model + optimizer persist in-process across iterations (a true continual on-policy
update). Each iteration writes the current policy to a checkpoint so the separate acting
agent (``PPORecordingAgent``) reloads it to roll out the next batch; we then run GAE +
clipped-surrogate PPO on those turns and overwrite the checkpoint with the updated
policy (used for eval and as a frozen league opponent later).

Value support is pinned to the warm-start checkpoint exactly as the M4 self-play loop
does, so the carried value head stays calibrated and the categorical critic loss
(``_value_ce`` against an HL-Gauss soft label) is reused unchanged. Opponents are a
uniform league sample + a light fixed anchor (fictitious play): a frozen opponent keeps
the environment stationary within a rollout, which the advantage estimates assume.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from poke_env.teambuilder.teambuilder import Teambuilder
from torch import nn

from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import PlayerSpec
from lategame.data.reward import RewardWeights
from lategame.data.rollout import RolloutBuffer, collect_rollout
from lategame.eval.arena import build_player, evaluate_built
from lategame.features.encoder import OBS_DIM, OBS_VERSION
from lategame.model.actor_critic import (
    hl_gauss_target,
    load_actor_critic_weights,
    value_from_logits,
    value_support,
)
from lategame.model.factory import build_model, model_metadata
from lategame.model.policy import masked_logits
from lategame.teambuilding.pool import TeamPool
from lategame.train.bc import select_device
from lategame.train.offline_rl import _value_ce
from lategame.train.selfplay import (
    CurvePoint,
    _print_curve_row,
    _sample_league,
    _write_curve,
)


@dataclass
class PPOConfig:
    init: str = "checkpoints/offrl_gen9randombattle.pt"
    out_dir: str = "checkpoints/ppo"
    battle_format: str = DEFAULT_FORMAT
    team_pool: str | None = None  # packed-team pool for teambuilt formats (gen9ou); None for RB
    loop_penalty: float = 0.0  # Build-14 LoopGuard on the learner/learned opponents (0 = off)
    iters: int = 8
    games_per_opp: int = 8
    pop_size: int = 2  # league checkpoints sampled per iteration (fictitious play)
    max_concurrent: int = 20  # concurrent battles during rollout / eval
    anchors: tuple[str, ...] = ("simpleheuristics",)  # lighter than M4 on purpose
    eval_baselines: tuple[str, ...] = ("random", "simpleheuristics", "heuristic")
    eval_n: int = 100
    # PPO update.
    epochs: int = 4
    minibatch: int = 256
    lr: float = 2.5e-4  # ~10x below offline; PPO from a warm start collapses easily
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # early-stop the epoch loop past 1.5x this
    hl_gauss_sigma_bins: float = 0.75
    device: str = "auto"
    seed: int = 0
    weights: RewardWeights = field(default_factory=RewardWeights)


def compute_gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda) advantages + value targets over a flat multi-episode buffer.

    ``done[t]`` marks episode terminals (every collected episode ends ``done=True``, so
    the bootstrap is masked there -- no separate truncation value is needed). Backward
    scan, resetting the advantage recursion at each terminal.
    """
    n = int(reward.shape[0])
    adv = torch.zeros(n, dtype=torch.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 0.0 if bool(done[t]) else 1.0
        next_value = float(value[t + 1]) if t + 1 < n else 0.0
        delta = float(reward[t]) + gamma * next_value * nonterminal - float(value[t])
        last = delta + gamma * gae_lambda * nonterminal * last
        adv[t] = last
    returns = adv + value
    return adv, returns


def _policy_stats(
    model: nn.Module, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(new_log_prob[B], entropy[B], value_logits[B, n_bins])`` over the legal set."""
    logits, value_logits = model(obs)
    log_probs = torch.log_softmax(masked_logits(logits, mask), dim=1)
    new_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=1)
    return new_log_prob, entropy, value_logits


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    centers: torch.Tensor,
    sigma: float,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """Clipped-surrogate PPO over the rollout; categorical value loss; KL early-stop.

    Runs in ``eval`` mode (dropout off) for consistency with the acting agent that
    produced ``old_log_prob`` -- gradients still flow.
    """
    model.eval()
    obs = buffer.obs.to(device)
    action = buffer.action.to(device)
    mask = buffer.mask.to(device)
    old_log_prob = buffer.old_log_prob.to(device)
    adv = advantages.to(device)
    ret = returns.to(device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    target = hl_gauss_target(ret.clamp(centers[0], centers[-1]), centers, sigma)

    n = int(obs.shape[0])
    sums = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0, "vmae": 0.0}
    steps = 0
    epochs_run = 0
    for _ in range(config.epochs):
        epochs_run += 1
        perm = torch.randperm(n, device=device)
        epoch_kls: list[float] = []
        for start in range(0, n, config.minibatch):
            mb = perm[start : start + config.minibatch]
            new_log_prob, entropy, value_logits = _policy_stats(
                model, obs[mb], mask[mb], action[mb]
            )
            ratio = (new_log_prob - old_log_prob[mb]).exp()
            a = adv[mb]
            surr1 = ratio * a
            surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = _value_ce(value_logits, target[mb])
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                logratio = new_log_prob - old_log_prob[mb]
                approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                clip_frac = float((ratio - 1.0).abs().gt(config.clip_eps).float().mean().item())
                vmae = float((value_from_logits(value_logits, centers) - ret[mb]).abs().mean())
            sums["policy"] += float(policy_loss.item())
            sums["value"] += float(value_loss.item())
            sums["entropy"] += float(entropy_mean.item())
            sums["kl"] += approx_kl
            sums["clip"] += clip_frac
            sums["vmae"] += vmae
            steps += 1
            epoch_kls.append(approx_kl)
        if epoch_kls and sum(epoch_kls) / len(epoch_kls) > 1.5 * config.target_kl:
            break  # KL early-stop: the policy has moved far enough this rollout

    steps = max(steps, 1)
    return {
        "policy_loss": sums["policy"] / steps,
        "value_loss": sums["value"] / steps,
        "entropy": sums["entropy"] / steps,
        "approx_kl": sums["kl"] / steps,
        "clip_frac": sums["clip"] / steps,
        "vmae": sums["vmae"] / steps,
        "ret_min": float(ret.min().item()),
        "ret_max": float(ret.max().item()),
        "epochs_run": float(epochs_run),
    }


def _save_checkpoint(
    model: nn.Module, path: str, battle_format: str, v_min: float, v_max: float, n_bins: int
) -> None:
    meta = model_metadata(model)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": meta["model_type"],
            "arch": meta["arch"],
            "input_dim": OBS_DIM,
            "hidden_dim": meta["hidden_dim"],
            "n_actions": meta["n_actions"],
            "n_bins": n_bins,
            "v_min": v_min,
            "v_max": v_max,
            "obs_version": OBS_VERSION,
            "battle_format": battle_format,
            "metrics": {},
        },
        Path(path),
    )


def _print_stats(iteration: int, n_turns: int, stats: dict[str, float]) -> None:
    print(
        f"  ppo iter {iteration:>3}  turns {n_turns}  "
        f"pi_loss {stats['policy_loss']:.4f}  v_loss {stats['value_loss']:.4f}  "
        f"entropy {stats['entropy']:.3f}  approx_kl {stats['approx_kl']:.4f}  "
        f"clip {stats['clip_frac']:.3f}  R[{stats['ret_min']:.2f},{stats['ret_max']:.2f}]  "
        f"vmae {stats['vmae']:.3f}  epochs {int(stats['epochs_run'])}"
    )


async def _eval_point(
    iteration: int,
    ckpt_path: str,
    config: PPOConfig,
    team: str | Teambuilder | None = None,
) -> CurvePoint:
    """Win-rate of the iteration's checkpoint (greedy) vs each baseline + vs iter 0.

    The PPO checkpoint is a standard actor-critic checkpoint, so the M3 ``offrl`` agent
    loads it directly -- no separate eval agent needed. ``team`` (a shared pool) is passed
    to every player for teambuilt formats; the learner arms carry ``config.loop_penalty`` so
    eval matches the acting policy (baselines ignore it via ``build_player``).
    """
    point: CurvePoint = {"iter": iteration}
    for base in config.eval_baselines:
        learner = build_player(
            "offrl",
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
        )
        opponent = build_player(
            base, config.battle_format, max_concurrent_battles=config.max_concurrent, team=team
        )
        point[f"vs_{base}"] = round(await evaluate_built(learner, opponent, config.eval_n), 4)
    if iteration > 0:
        learner = build_player(
            "offrl",
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
        )
        iter0 = build_player(
            "offrl",
            config.battle_format,
            checkpoint_path=config.init,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
        )
        point["vs_iter0"] = round(await evaluate_built(learner, iter0, config.eval_n), 4)
    return point


async def run_ppo(config: PPOConfig) -> list[CurvePoint]:
    """Run the on-policy PPO self-play loop; return (and persist) the per-iter curve."""
    init_path = Path(config.init)
    if not init_path.exists():
        raise FileNotFoundError(
            f"Warm-start checkpoint not found at '{init_path}'. Train M3 first "
            f"(`lategame train-rl`) or pass --init."
        )
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    print(f"training on {device}")

    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    if ckpt.get("obs_version") != OBS_VERSION or ckpt.get("input_dim") != OBS_DIM:
        raise ValueError("init checkpoint encoder mismatch; cannot warm-start PPO. Retrain.")
    # Rollout runs in the checkpoint's format; eval (_eval_point) runs config.battle_format.
    # If they disagree the loop would roll out one format and score another -- fail loudly.
    battle_format = str(ckpt.get("battle_format", config.battle_format))
    if battle_format != config.battle_format:
        raise ValueError(
            f"format mismatch: init checkpoint is '{battle_format}' but config is "
            f"'{config.battle_format}'; rollout/eval would differ -- pass --format {battle_format}"
        )
    v_min, v_max, n_bins = float(ckpt["v_min"]), float(ckpt["v_max"]), int(ckpt["n_bins"])
    centers = value_support(v_min, v_max, n_bins).to(device)
    sigma = config.hl_gauss_sigma_bins * (v_max - v_min) / (n_bins - 1)
    team = TeamPool.from_packed_file(config.team_pool) if config.team_pool else None
    print(f"value support [{v_min:.3f}, {v_max:.3f}] over {n_bins} bins (sigma={sigma:.3f})")

    model = build_model(ckpt).to(device)
    load_actor_critic_weights(model, ckpt["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)

    latest = str(init_path)  # checkpoint the acting agent reads (iter 0 = init)
    league = [str(init_path)]

    curve: list[CurvePoint] = [await _eval_point(0, latest, config, team)]
    _print_curve_row(curve[0])
    _write_curve(out_dir / "curve.json", curve)

    for k in range(1, config.iters + 1):
        opponents = [
            PlayerSpec("offrl", checkpoint_path=p, sample=True)
            for p in _sample_league(league, config.pop_size, rng)
        ]
        opponents += [PlayerSpec(a) for a in config.anchors]

        # Write the CURRENT in-process policy so the acting agent rolls out from it.
        ckpt_path = str(out_dir / f"iter_{k:02d}.pt")
        _save_checkpoint(model, ckpt_path, battle_format, v_min, v_max, n_bins)

        buffer = await collect_rollout(
            ckpt_path,
            opponents,
            config.games_per_opp,
            battle_format,
            config.weights,
            config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
        )
        advantages, returns = compute_gae(
            buffer.reward, buffer.value, buffer.done, config.gamma, config.gae_lambda
        )
        stats = ppo_update(
            model, optimizer, buffer, advantages, returns, centers, sigma, config, device
        )

        # Overwrite with the UPDATED policy -- used for eval and as a league opponent.
        _save_checkpoint(model, ckpt_path, battle_format, v_min, v_max, n_bins)
        latest = ckpt_path
        league.append(ckpt_path)

        _print_stats(k, len(buffer), stats)
        point = await _eval_point(k, latest, config, team)
        curve.append(point)
        _print_curve_row(point)
        _write_curve(out_dir / "curve.json", curve)

    print(f"wrote curve ({len(curve)} points) to {out_dir / 'curve.json'}")
    return curve

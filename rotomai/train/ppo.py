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
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from poke_env.teambuilder.teambuilder import Teambuilder
from torch import nn

from rotomai.config import DEFAULT_FORMAT, is_doubles_format
from rotomai.data.collect import PlayerSpec
from rotomai.data.reward import RewardWeights
from rotomai.data.rollout import RolloutBuffer, collect_rollout
from rotomai.eval.arena import build_player, evaluate_built, policy_agent
from rotomai.features.codec import codec_for
from rotomai.model.actor_critic import (
    hl_gauss_target,
    load_actor_critic_weights,
    value_from_logits,
    value_support,
)
from rotomai.model.factory import build_model, model_metadata
from rotomai.model.policy import (
    factored_entropy,
    factored_has_choice,
    factored_log_prob,
    factored_logits,
    factored_masked_logits,
    masked_logits,
    restrict_slot1_mask,
)
from rotomai.teambuilding.pool import TeamPool
from rotomai.train.bc import select_device
from rotomai.train.offline_rl import _value_ce
from rotomai.train.selfplay import (
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
    # Build 19 (docs/RESULTS.md): linear schedules over the run. ``None`` holds the value constant,
    # which is exactly the Build 16-18 behavior. Rollout SAMPLES but eval takes the ARGMAX, so a
    # fixed entropy bonus optimizes a policy we never deploy: on the v18 plateau the sampled
    # policy scored 0.347 vs the argmax's 0.487 (scripts/policy_sharpness_diag.py). Annealing
    # ent_coef -> 0 collapses that train/eval gap by letting the distribution sharpen onto its
    # own argmax.
    ent_coef_final: float | None = None
    lr_final: float | None = None
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # early-stop the epoch loop past 1.5x this
    # Build 24: the anneal HORIZON, decoupled from ``iters``. ``None`` == ``iters``, which is the
    # Build 19-23 behavior. This exists because raising ``iters`` alone silently rescales BOTH
    # schedules: at iteration 40, v22 (50-budget) ran lr 9.08e-05 / ent 0.0020 while v23b
    # (80-budget) ran lr 1.51e-04 / ent 0.0051, and at iteration 50 v22 was frozen at its finals
    # while v23b was still at lr 1.26e-04. So "more iters" is really three changes at once, and
    # a strength difference between budgets cannot be attributed without pinning this.
    anneal_iters: int | None = None
    hl_gauss_sigma_bins: float = 0.75
    device: str = "auto"
    seed: int = 0
    weights: RewardWeights = field(default_factory=RewardWeights)
    # Build 26: restart an interrupted run from ``out_dir``'s last completed iteration. OFF by
    # default, and when off this file behaves exactly as it did through Build 25 -- no resume
    # state is written and no existing state is read.
    #
    # This exists because arm length is bounded by MEMORY, not by walltime. RSS grows ~0.23
    # GiB/iter (40.7 GiB at 160 iters, 59.3 at 240), and the medium QoS caps a job at 64 GB --
    # a per-job MaxTRES, so a larger --mem is rejected at submit rather than queued. v26b's
    # first attempt was OUT_OF_MEMORY at 304/320. A fresh process resets RSS, so resuming is
    # what makes an arm past ~260 iters runnable at all; it is not merely crash insurance.
    resume: bool = False
    # B6f: per-battle decision ceiling, forfeiting past it. ``None`` is exact identity and is what
    # every OU build ran with. It exists because a soft loop guard cannot break the absorbing
    # forced-replacement cycle that put 94.2% of the first VGC shard into 100 of 899 episodes --
    # see ``agents.turn_cap``.
    max_battle_turns: int | None = None

    @property
    def anneal_horizon(self) -> int:
        """Iterations the lr/ent schedules span -- ``anneal_iters`` if pinned, else ``iters``."""
        return self.anneal_iters or self.iters


def anneal(start: float, final: float | None, iteration: int, iters: int) -> float:
    """Linear interpolation from ``start`` (iteration 1) to ``final`` (iteration ``iters``).

    ``final is None`` holds ``start`` constant -- the pre-Build-19 behavior, so an unscheduled
    config is bit-identical to Build 16-18. ``iteration`` is clamped into ``[1, iters]``.
    """
    if final is None or iters <= 1:
        return start
    k = min(max(iteration, 1), iters)
    frac = (k - 1) / (iters - 1)
    return start + (final - start) * frac


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
    """``(new_log_prob[B], entropy[B], value_logits[B, n_bins])`` over the legal set.

    Dispatches on the mask's rank, exactly as ``train.offline_rl._run_epoch`` does: ``(B, S, A)``
    is a doubles turn and ``(B, A)`` a singles one. On the factored branch the joint log-prob is
    the SUM of the two slots' log-probs, and slot 1's distribution is conditioned on slot 0's
    choice through ``restrict_slot1_mask`` -- recomputed here from ``mask`` and ``action[:, 0]``
    rather than stored, so there is exactly one definition of the restriction and act-time and
    train-time cannot disagree about what the behaviour policy was.
    """
    logits, value_logits = model(obs)
    if mask.dim() == 3:
        n_slots = mask.shape[1]
        a = action.clamp(min=0)
        restricted = restrict_slot1_mask(mask, a[:, 0])
        log_probs = torch.log_softmax(
            factored_masked_logits(factored_logits(logits, n_slots), restricted), dim=-1
        )
        return factored_log_prob(log_probs, a), factored_entropy(log_probs), value_logits
    log_probs = torch.log_softmax(masked_logits(logits, mask), dim=1)
    new_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=1)
    return new_log_prob, entropy, value_logits


def _surrogate_weights(buffer: RolloutBuffer) -> torch.Tensor | None:
    """Per-row policy-gradient weight for a FACTORED buffer, or ``None`` on singles.

    THREE kinds of row carry no usable policy gradient and must not sit in the denominator of the
    reported policy loss, entropy, approx-KL or clip fraction:

    * **No decision.** Every slot had exactly one legal action, so ``log pi`` is identically 0 and
      the surrogate is a constant in the parameters. On the first VGC shard these were 43% of all
      turns; a plain ``.mean()`` over them divides the policy gradient, scales ``ent_coef`` down
      by the same factor, and -- worst -- shrinks ``approx_kl`` so ``target_kl`` never binds while
      ``scripts/ppo_telemetry`` still certifies "trust region clean, 100% full epochs".
    * **Not executed as recorded.** The doubles decoder fell back to a random legal order, so the
      played action is not the recorded one and the ratio's denominator describes a different
      action.
    * **Illegal under its own (restricted) mask.** Its logit sits at ``NEG_INF``, so the ratio
      explodes; the AWR path hit the same rows for 719,296 of actor loss (``train.offline_rl``).

    Zeroed, never dropped: ``compute_gae`` scans a flat contiguous buffer and resets on ``done``,
    so deleting rows would splice unrelated turns together. Singles returns ``None`` and keeps the
    plain means it has always reported -- applying this there would silently rescale every
    published OU number.
    """
    if buffer.mask.dim() != 3:
        return None
    a = buffer.action.clamp(min=0)
    restricted = restrict_slot1_mask(buffer.mask, a[:, 0])
    legal = torch.gather(restricted, 2, a.unsqueeze(-1)).squeeze(-1).all(dim=1)
    ok = factored_has_choice(buffer.mask) & buffer.executed_mask() & legal
    return ok.to(torch.float32)


def _wmean(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """``x.mean()``, or the weighted mean over ``w`` when a weight vector is in play."""
    if w is None:
        return x.mean()
    return (x * w).sum() / w.sum().clamp(min=1.0)


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
    weights = _surrogate_weights(buffer)
    if weights is not None:
        weights = weights.to(device)

    n = int(obs.shape[0])
    # THE CANARY. `ppo_update` runs in eval mode and the acting agent read the very checkpoint
    # these weights were saved from, so recomputing log pi over the buffer BEFORE any gradient
    # step must reproduce what the agent recorded to floating-point noise. Any real drift means
    # the acting distribution and the training distribution are not the same function -- a
    # different mask convention, a post-hoc action rewrite, a device mismatch -- and every ratio
    # in the update is then wrong in a way no loss curve would show.
    lp_drift = 0.0
    with torch.no_grad():
        check = weights > 0 if weights is not None else torch.ones(n, dtype=torch.bool)
        if bool(check.any()):
            idx = check.nonzero(as_tuple=True)[0]
            lp_now, _, _ = _policy_stats(model, obs[idx], mask[idx], action[idx])
            lp_drift = float((lp_now - old_log_prob[idx]).abs().max().item())
    if lp_drift > _MAX_LP_DRIFT:
        raise RuntimeError(
            f"acting/training log-prob mismatch: max|new - old| = {lp_drift:.4g} before any "
            f"gradient step (ceiling {_MAX_LP_DRIFT:g}). The behaviour policy and the policy "
            f"being updated are not the same function, so every importance ratio in this update "
            f"is wrong. Check the masking convention and the action recorded vs. executed."
        )

    sums = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0, "vmae": 0.0}
    steps = 0
    epochs_run = 0
    for _ in range(config.epochs):
        epochs_run += 1
        perm = torch.randperm(n, device=device)
        epoch_kls: list[float] = []
        for start in range(0, n, config.minibatch):
            mb = perm[start : start + config.minibatch]
            w = None if weights is None else weights[mb]
            new_log_prob, entropy, value_logits = _policy_stats(
                model, obs[mb], mask[mb], action[mb]
            )
            ratio = (new_log_prob - old_log_prob[mb]).exp()
            a = adv[mb]
            surr1 = ratio * a
            surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * a
            policy_loss = -_wmean(torch.min(surr1, surr2), w)
            # The critic wants EVERY state, including the ones that offered no choice -- they are
            # real positions with real returns. Only the actor's reductions are weighted.
            value_loss = _value_ce(value_logits, target[mb])
            entropy_mean = _wmean(entropy, w)
            loss = policy_loss + config.value_coef * value_loss - config.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                logratio = new_log_prob - old_log_prob[mb]
                approx_kl = float(_wmean((ratio - 1.0) - logratio, w).item())
                clip_frac = float(
                    _wmean((ratio - 1.0).abs().gt(config.clip_eps).float(), w).item()
                )
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
    stats = {
        "policy_loss": sums["policy"] / steps,
        "value_loss": sums["value"] / steps,
        "entropy": sums["entropy"] / steps,
        "approx_kl": sums["kl"] / steps,
        "clip_frac": sums["clip"] / steps,
        "vmae": sums["vmae"] / steps,
        "ret_min": float(ret.min().item()),
        "ret_max": float(ret.max().item()),
        "epochs_run": float(epochs_run),
        # The values this update actually ran at -- the schedule is auditable per iteration.
        "ent_coef": config.ent_coef,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    if weights is not None:
        # Factored-only extras, APPENDED to the dict (and to the log line) so an OU run's stats
        # and its telemetry regex are byte-identical to every build before this one.
        stats["n_decision"] = float(weights.sum().item())
        stats["dec_frac"] = float(weights.mean().item())
        stats["invalid_frac"] = float((~buffer.executed_mask()).float().mean().item())
        stats["lp_drift"] = lp_drift
    return stats


def _save_checkpoint(
    model: nn.Module, path: str, battle_format: str, v_min: float, v_max: float, n_bins: int
) -> None:
    meta = model_metadata(model)
    # Dims come from the FORMAT'S codec, not from the singles encoder's module constants. Stamping
    # `v5-`/761 onto a doubles checkpoint would not fail here -- it fails later, in `DoublesAgent`'s
    # fingerprint check, after an iteration of rollout has already been paid for.
    codec = codec_for(battle_format)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": meta["model_type"],
            "arch": meta["arch"],
            "input_dim": codec.obs_dim,
            "hidden_dim": meta["hidden_dim"],
            "n_actions": meta["n_actions"],
            "n_bins": n_bins,
            "v_min": v_min,
            "v_max": v_max,
            "obs_version": codec.obs_version,
            "battle_format": battle_format,
            "metrics": {},
        },
        Path(path),
    )


#: Written beside the per-iteration checkpoints when ``PPOConfig.resume`` is on. Kept SEPARATE
#: from ``iter_NN.pt`` on purpose: those are loaded by the agents and by every gate, so widening
#: their schema to carry optimizer state would change an artifact the whole project reads.
_RESUME_FILE = "resume_state.pt"

#: Ceiling on ``max|log pi_new - log pi_old|`` measured before any gradient step. The acting agent
#: read the checkpoint these weights were saved from and ``ppo_update`` runs in eval mode, so the
#: honest value is floating-point noise (~1e-7 on CPU). Set well above that but far below any real
#: policy difference, so it catches a convention mismatch without tripping on device arithmetic.
_MAX_LP_DRIFT = 1e-2


def _save_resume_state(
    path: Path,
    iteration: int,
    optimizer: torch.optim.Optimizer,
    curve: list[CurvePoint],
    rng: random.Random,
) -> None:
    """Persist everything a restart needs that ``iter_NN.pt`` does not carry.

    The model weights are already on disk; what is NOT is Adam's moment estimates and the two
    RNG streams. Dropping the moments would restart the optimizer cold mid-run -- a real
    discontinuity in training dynamics, not a bookkeeping detail -- and dropping the RNG state
    would re-draw the league from the top of the stream. Written atomically: a job killed
    mid-write must not leave a truncated state that poisons the retry.
    """
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "curve": curve,
            "py_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


def _load_resume_state(out_dir: Path, iters: int) -> dict | None:
    """Return the resume state if this ``out_dir`` holds a usable partial run, else None.

    Returns None (start fresh) when there is nothing to resume from, and raises when the state
    is present but unusable -- a silent fresh start would burn the whole walltime re-running
    work that already exists.
    """
    path = out_dir / _RESUME_FILE
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    done = int(state["iteration"])
    if done >= iters:
        raise ValueError(
            f"{out_dir} already holds {done} iterations against a target of {iters}; nothing to "
            "resume. Raise --iters or point --ckpt-prefix somewhere else."
        )
    ckpt = out_dir / f"iter_{done:02d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"resume state says iteration {done} but {ckpt} is missing -- the checkpoint it "
            "names must exist to restart from."
        )
    state["checkpoint"] = str(ckpt)
    return state


def _print_stats(iteration: int, n_turns: int, stats: dict[str, float]) -> None:
    line = (
        f"  ppo iter {iteration:>3}  turns {n_turns}  "
        f"pi_loss {stats['policy_loss']:.4f}  v_loss {stats['value_loss']:.4f}  "
        f"entropy {stats['entropy']:.3f}  approx_kl {stats['approx_kl']:.4f}  "
        f"clip {stats['clip_frac']:.3f}  R[{stats['ret_min']:.2f},{stats['ret_max']:.2f}]  "
        f"vmae {stats['vmae']:.3f}  epochs {int(stats['epochs_run'])}  "
        f"ent_coef {stats['ent_coef']:.4f}  lr {stats['lr']:.2e}"
    )
    # APPENDED, and only on the factored path. `scripts/ppo_telemetry.STATS` matches up to `lr`
    # with `.search`, so an OU log line stays byte-identical and every historical certificate
    # still parses; a VGC line simply carries four more fields after it.
    if "dec_frac" in stats:
        line += (
            f"  dec_frac {stats['dec_frac']:.4f}  n_decision {int(stats['n_decision'])}  "
            f"invalid_frac {stats['invalid_frac']:.4f}  lp_drift {stats['lp_drift']:.2e}"
        )
    print(line)


async def _eval_point(
    iteration: int,
    ckpt_path: str,
    config: PPOConfig,
    team: str | Teambuilder | None = None,
) -> CurvePoint:
    """Win-rate of the iteration's checkpoint (greedy) vs each baseline + vs iter 0.

    The PPO checkpoint is a standard actor-critic checkpoint, so the format's own greedy agent
    loads it directly -- ``offrl`` on singles, ``doubles`` on doubles (``arena.policy_agent``);
    no separate eval agent needed. ``team`` (a shared pool) is passed to every player for
    teambuilt formats; the learner arms carry ``config.loop_penalty`` so eval matches the acting
    policy (baselines ignore it via ``build_player``).
    """
    point: CurvePoint = {"iter": iteration}
    for base in config.eval_baselines:
        learner = build_player(
            policy_agent(config.battle_format),
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
            max_battle_turns=config.max_battle_turns,
        )
        opponent = build_player(
            base, config.battle_format, max_concurrent_battles=config.max_concurrent, team=team
        )
        point[f"vs_{base}"] = round(await evaluate_built(learner, opponent, config.eval_n), 4)
    if iteration > 0:
        learner = build_player(
            policy_agent(config.battle_format),
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
            max_battle_turns=config.max_battle_turns,
        )
        iter0 = build_player(
            policy_agent(config.battle_format),
            config.battle_format,
            checkpoint_path=config.init,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            loop_penalty=config.loop_penalty,
            max_battle_turns=config.max_battle_turns,
        )
        point["vs_iter0"] = round(await evaluate_built(learner, iter0, config.eval_n), 4)
    return point


async def run_ppo(config: PPOConfig) -> list[CurvePoint]:
    """Run the on-policy PPO self-play loop; return (and persist) the per-iter curve."""
    # Validated FIRST, before any checkpoint is read or model built: a config that cannot produce
    # a coherent arm should fail in milliseconds, not after loading a warm start.
    if config.resume and config.anneal_iters is None:
        # ``anneal_horizon`` falls back to ``iters``, and a chunked run raises ``iters`` between
        # chunks -- so an unpinned schedule would anneal over a DIFFERENT span in each chunk and
        # the arm would silently stop being one experiment. This is precisely the confound Build
        # 24 introduced ``anneal_iters`` to remove, so refuse rather than reintroduce it.
        raise ValueError(
            "--resume requires --anneal-iters to be pinned: the lr/entropy horizon otherwise "
            "defaults to --iters, which differs between chunks and silently rescales both "
            "schedules mid-arm."
        )
    init_path = Path(config.init)
    if not init_path.exists():
        raise FileNotFoundError(
            f"Warm-start checkpoint not found at '{init_path}'. Train M3 first "
            f"(`rotomai train-rl`) or pass --init."
        )
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    print(f"training on {device}")

    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    # Rollout runs in the checkpoint's format; eval (_eval_point) runs config.battle_format.
    # If they disagree the loop would roll out one format and score another -- fail loudly.
    # Checked FIRST, because the encoder check below is only meaningful once we know which
    # format's codec to check against.
    battle_format = str(ckpt.get("battle_format", config.battle_format))
    if battle_format != config.battle_format:
        raise ValueError(
            f"format mismatch: init checkpoint is '{battle_format}' but config is "
            f"'{config.battle_format}'; rollout/eval would differ -- pass --format {battle_format}"
        )
    codec = codec_for(battle_format)
    if (
        ckpt.get("obs_version") != codec.obs_version
        or ckpt.get("input_dim") != codec.obs_dim
        or ckpt.get("n_actions") != codec.n_actions
    ):
        raise ValueError(
            f"init checkpoint encoder mismatch (ckpt {ckpt.get('obs_version')}/"
            f"{ckpt.get('input_dim')}/{ckpt.get('n_actions')} vs {codec.name} codec "
            f"{codec.obs_version}/{codec.obs_dim}/{codec.n_actions}); cannot warm-start PPO. "
            f"Retrain."
        )
    if is_doubles_format(battle_format) and config.loop_penalty != 0.0:
        # `--loop-penalty 4.0` is on every OU invocation, so it copy-pastes into a VGC one. The
        # doubles agents DO carry a guard, but it is `DoublesLoopGuard` with its own tuning; a
        # value chosen against the singles 26-action layout would be recorded in the gate JSON as
        # if it meant the same thing. Refuse rather than let an arm mislabel itself.
        raise ValueError(
            f"--loop-penalty {config.loop_penalty} was passed with the doubles format "
            f"'{battle_format}'. The doubles guard (agents.loop_guard.DoublesLoopGuard) is a "
            f"per-slot penalty over a different action layout, so an OU-tuned value does not "
            f"carry over. Pass 0 and tune it as its own arm."
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

    # ``resume`` off => this whole block is skipped and the run is identical to Build 25's.
    state = _load_resume_state(out_dir, config.iters) if config.resume else None
    if state is None:
        start_iter = 1
        curve: list[CurvePoint] = [await _eval_point(0, latest, config, team)]
        _print_curve_row(curve[0])
        _write_curve(out_dir / "curve.json", curve)
    else:
        done = int(state["iteration"])
        start_iter = done + 1
        latest = str(state["checkpoint"])
        # Reload the POLICY from the last completed iteration, and Adam's moments with it.
        resumed = torch.load(latest, map_location="cpu", weights_only=False)
        load_actor_critic_weights(model, resumed["state_dict"])
        model.to(device)
        optimizer.load_state_dict(state["optimizer"])
        # Restore both RNG streams so the league draw continues the sequence instead of
        # replaying it from the seed.
        rng.setstate(state["py_rng"])
        torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        curve = list(state["curve"])
        # The league is the init plus every checkpoint written so far -- reconstructed rather
        # than stored, so it cannot disagree with what is actually on disk.
        league = [str(init_path)] + [
            str(out_dir / f"iter_{i:02d}.pt")
            for i in range(1, done + 1)
            if (out_dir / f"iter_{i:02d}.pt").exists()
        ]
        print(
            f"RESUMING from iteration {done} ({latest}) | league {len(league)} | "
            f"curve {len(curve)} points | -> {config.iters}"
        )

    league_agent = policy_agent(battle_format)
    for k in range(start_iter, config.iters + 1):
        opponents = [
            PlayerSpec(league_agent, checkpoint_path=p, sample=True)
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
            max_battle_turns=config.max_battle_turns,
        )
        advantages, returns = compute_gae(
            buffer.reward, buffer.value, buffer.done, config.gamma, config.gae_lambda
        )
        # Build 19: linear ent_coef / lr schedules. Both default to None (constant), so an
        # unscheduled run is identical to Build 18. ``replace`` keeps ppo_update pure -- it
        # reads config.ent_coef, so hand it a config that already carries this iteration's value.
        # Build 24: horizon defaults to ``iters`` (Build 19-23 behavior). Pinning it SHORTER holds
        # both schedules at their finals past that point -- ``anneal`` clamps the iteration -- which
        # is what separates "more updates" from "a slower anneal".
        horizon = config.anneal_horizon
        for group in optimizer.param_groups:
            group["lr"] = anneal(config.lr, config.lr_final, k, horizon)
        iter_config = replace(
            config, ent_coef=anneal(config.ent_coef, config.ent_coef_final, k, horizon)
        )
        stats = ppo_update(
            model, optimizer, buffer, advantages, returns, centers, sigma, iter_config, device
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
        # Last thing in the iteration, so the state can only ever name a FULLY completed one:
        # the checkpoint is written, the curve point is scored, and the curve file is on disk.
        if config.resume:
            _save_resume_state(out_dir / _RESUME_FILE, k, optimizer, curve, rng)

    print(f"wrote curve ({len(curve)} points) to {out_dir / 'curve.json'}")
    return curve

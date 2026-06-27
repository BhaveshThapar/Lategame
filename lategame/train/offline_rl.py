"""Offline-RL training: value-classification critic + AWR actor (plan.md, M3).

One pass over a collected trajectory shard. For each turn we have the discounted
Monte-Carlo return ``ret``. We train:

* the **critic** by value classification -- cross-entropy of the value-bin logits
  against an HL-Gauss soft label of ``ret``;
* the **actor** by advantage-weighted behaviour cloning -- the masked action CE,
  weighted per sample by ``clamp(exp((ret - V)/beta), max=clip)``. Where the taken
  action did better than the state's estimated value it is imitated harder; where
  it did worse it is suppressed. This is what lets the policy exceed the average
  self-play behaviour BC could only match.

The actor + policy head warm-start from the M2 BC checkpoint; the critic starts
fresh. Device handling mirrors ``train.bc`` (cpu / mps / cuda).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from lategame.data.rl_dataset import RLDataset
from lategame.features.encoder import OBS_DIM, OBS_VERSION
from lategame.model.actor_critic import (
    ActorCritic,
    hl_gauss_target,
    load_bc_weights,
    value_from_logits,
    value_support,
)
from lategame.model.policy import masked_logits
from lategame.train.bc import select_device


@dataclass
class OfflineRLConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 256
    dropout: float = 0.1
    n_bins: int = 51
    beta: float = 1.0  # AWR temperature
    value_coef: float = 0.5
    awr_weight_clip: float = 20.0
    hl_gauss_sigma_bins: float = 0.75  # smoothing width, in bin-widths
    val_frac: float = 0.1
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "auto"
    bc_init: str | None = None


def _support_from_returns(ret: torch.Tensor, margin: float = 0.1) -> tuple[float, float]:
    v_min, v_max = float(ret.min()), float(ret.max())
    span = v_max - v_min
    if span < 1e-6:
        return v_min - 1.0, v_max + 1.0
    return v_min - margin * span, v_max + margin * span


def _value_ce(value_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft (HL-Gauss) label."""
    return -(target * F.log_softmax(value_logits, dim=-1)).sum(dim=1).mean()


def _run_epoch(
    model: ActorCritic,
    loader: DataLoader,
    device: torch.device,
    centers: torch.Tensor,
    config: OfflineRLConfig,
    sigma: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"actor": 0.0, "value": 0.0, "acc": 0.0, "value_mae": 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for obs, action, mask, ret in loader:
            obs, action, mask, ret = (
                obs.to(device),
                action.to(device),
                mask.to(device),
                ret.to(device),
            )
            logits, value_logits = model(obs)
            logits_masked = masked_logits(logits, mask)

            target = hl_gauss_target(ret, centers, sigma)
            value_loss = _value_ce(value_logits, target)

            v = value_from_logits(value_logits, centers)
            adv = ret - v.detach()
            weight = torch.clamp(torch.exp(adv / config.beta), max=config.awr_weight_clip)
            ce = F.cross_entropy(logits_masked, action, reduction="none")
            actor_loss = (weight * ce).mean()

            loss = actor_loss + config.value_coef * value_loss
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = obs.shape[0]
            totals["actor"] += actor_loss.item() * n
            totals["value"] += value_loss.item() * n
            totals["acc"] += int((logits_masked.argmax(dim=1) == action).sum().item())
            totals["value_mae"] += float((v - ret).abs().sum().item())
            count += n
    return {k: v / count for k, v in totals.items()}


def train_offline_rl(data_path: str | Path, out_path: str | Path, config: OfflineRLConfig) -> dict:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    print(f"training on {device}")

    dataset = RLDataset(data_path)
    v_min, v_max = _support_from_returns(dataset.ret)
    centers = value_support(v_min, v_max, config.n_bins).to(device)
    bin_width = (v_max - v_min) / (config.n_bins - 1)
    sigma = config.hl_gauss_sigma_bins * bin_width
    print(f"value support [{v_min:.3f}, {v_max:.3f}] over {config.n_bins} bins (sigma={sigma:.3f})")

    n_val = max(1, int(len(dataset) * config.val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    hidden_dim = config.hidden_dim
    bc_state = None
    if config.bc_init:
        bc_ckpt = torch.load(config.bc_init, map_location="cpu", weights_only=False)
        if bc_ckpt.get("obs_version") != OBS_VERSION or bc_ckpt.get("input_dim") != OBS_DIM:
            raise ValueError("BC checkpoint encoder mismatch; cannot warm-start. Retrain BC.")
        hidden_dim = int(bc_ckpt["hidden_dim"])  # trunk shapes must match for warm-start
        bc_state = bc_ckpt["state_dict"]

    model = ActorCritic(
        OBS_DIM, hidden_dim=hidden_dim, n_bins=config.n_bins, dropout=config.dropout
    ).to(device)
    if bc_state is not None:
        load_bc_weights(model, bc_state)
        print(f"warm-started trunk + policy head from {config.bc_init}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_metrics: dict = {}
    for epoch in range(1, config.epochs + 1):
        tr = _run_epoch(model, train_loader, device, centers, config, sigma, optimizer)
        va = _run_epoch(model, val_loader, device, centers, config, sigma, None)
        print(
            f"epoch {epoch:>3}/{config.epochs}  "
            f"train actor {tr['actor']:.4f} value {tr['value']:.4f} acc {tr['acc']:.3f}  "
            f"val actor {va['actor']:.4f} value {va['value']:.4f} "
            f"acc {va['acc']:.3f} vmae {va['value_mae']:.3f}"
        )
        val_total = va["actor"] + config.value_coef * va["value"]
        if val_total < best_val:
            best_val = val_total
            best_metrics = {"epoch": epoch, "val_total": val_total, **va}
            _save_checkpoint(
                model, out_path, dataset.battle_format, config, v_min, v_max, best_metrics
            )

    print(f"best: {best_metrics}")
    return best_metrics


def _save_checkpoint(
    model: ActorCritic,
    out_path: str | Path,
    battle_format: str,
    config: OfflineRLConfig,
    v_min: float,
    v_max: float,
    metrics: dict,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "actor_critic",
            "input_dim": OBS_DIM,
            "hidden_dim": model.hidden_dim,
            "n_actions": model.n_actions,
            "n_bins": model.n_bins,
            "v_min": v_min,
            "v_max": v_max,
            "obs_version": OBS_VERSION,
            "battle_format": battle_format,
            "metrics": metrics,
        },
        out_path,
    )

"""Behaviour-cloning training loop (plan.md, M2).

Supervised classification: predict the demonstrated action from the encoded
state. Illegal actions are masked before the softmax so the loss only competes
among legal choices. Best-by-validation-loss checkpoint is written with enough
metadata (encoder version, dims, format) for the ``bc`` agent to reload it safely.

Device-agnostic (cpu / mps / cuda): small enough to train on this Mac, a no-code
change to run on a remote GPU later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from lategame.data.dataset import BCDataset
from lategame.features.encoder import OBS_DIM, OBS_LAYOUT, OBS_VERSION
from lategame.model.factory import build_model, model_metadata
from lategame.model.policy import BCPolicy, masked_logits, policy_logits
from lategame.train.augment import augment_pp_full

BC_POLICY = "bc_policy"  # model_type sentinel for the flat MLP (factory has no BCPolicy).


def select_device(prefer: str = "auto") -> torch.device:
    if prefer not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(f"Unknown device '{prefer}'")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 256
    dropout: float = 0.1
    val_frac: float = 0.1
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "auto"
    model_type: str = BC_POLICY  # BC_POLICY (flat MLP) | "entity_transformer" | "actor_critic"
    id_embed: bool = True  # entity_transformer only: learned species/move/item/ability ids
    id_embed_dim: int = 32
    id_embed_init: str = "random"  # entity_transformer only: "random" | "prior" (dex warm-start)
    max_samples: int | None = None  # data-scaling sweep: train on a nested subset of N samples
    # Build 10 (plan.md 13): train-time pp augmentation. On this fraction of attack-labeled,
    # deep-turn rows, force the active mon's pp channels to full -- making "full pp deep in a
    # game -> attack" in-distribution. 0.0 disables it (train-time only; validation stays clean).
    pp_aug_frac: float = 0.0
    pp_aug_turn_threshold: float = 0.15  # normalized turn (~turn 15) above which a row is "deep"


# Fixed seed for the data-subset permutation -- deliberately independent of TrainConfig.seed
# so that (a) every model seed at a given N sees the *same* subset and (b) larger N nests the
# smaller (size-N subset is a prefix of size-2N). config.seed varies only init + train/val split.
_SUBSET_SEED = 1234


def _subset(dataset: BCDataset, max_samples: int | None) -> BCDataset | Subset:
    """Deterministic nested subset of the first ``max_samples`` shuffled indices, or the
    full dataset when ``max_samples`` is None or already covers it."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(_SUBSET_SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def _build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    """BC_POLICY -> flat MLP (legacy default); anything else -> factory architecture."""
    if config.model_type == BC_POLICY:
        return BCPolicy(OBS_DIM, hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    meta = {
        "model_type": config.model_type,
        "input_dim": OBS_DIM,
        "dropout": config.dropout,
        "hidden_dim": config.hidden_dim,
        "arch": {
            "id_embed": config.id_embed,
            "id_embed_dim": config.id_embed_dim,
            "id_embed_init": config.id_embed_init,
        },
    }
    return build_model(meta).to(device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
) -> tuple[float, float]:
    """Returns (mean loss, accuracy). Trains when ``optimizer`` is given, else evals."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(training):
        for obs, action, mask in loader:
            obs, action, mask = obs.to(device), action.to(device), mask.to(device)
            if training and config.pp_aug_frac > 0.0:
                obs = augment_pp_full(
                    obs,
                    action,
                    OBS_LAYOUT,
                    frac=config.pp_aug_frac,
                    turn_threshold=config.pp_aug_turn_threshold,
                )
            logits = masked_logits(policy_logits(model(obs)), mask)
            loss = F.cross_entropy(logits, action)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * obs.shape[0]
            correct += int((logits.argmax(dim=1) == action).sum().item())
            count += obs.shape[0]
    return total_loss / count, correct / count


def train_bc(data_path: str | Path, out_path: str | Path, config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    print(f"training on {device}")

    dataset = BCDataset(data_path)
    battle_format = dataset.battle_format
    data = _subset(dataset, config.max_samples)
    print(f"training on {len(data)}/{len(dataset)} samples")
    n_val = max(1, int(len(data) * config.val_frac))
    n_train = len(data) - n_val
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(data, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    model = _build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_metrics: dict = {}
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer, config)
        val_loss, val_acc = _run_epoch(model, val_loader, device, None, config)
        print(
            f"epoch {epoch:>3}/{config.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_metrics = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
            _save_checkpoint(model, out_path, battle_format, config, best_metrics)

    print(f"best: {best_metrics}")
    return best_metrics


def _save_checkpoint(
    model: nn.Module,
    out_path: str | Path,
    battle_format: str,
    config: TrainConfig,
    metrics: dict,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, BCPolicy):
        meta = {
            "model_type": BC_POLICY,
            "hidden_dim": config.hidden_dim,
            "n_actions": model.n_actions,
        }
    else:
        meta = model_metadata(model)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": OBS_DIM,
            "obs_version": OBS_VERSION,
            "battle_format": battle_format,
            "metrics": metrics,
            "pp_aug": {"frac": config.pp_aug_frac, "turn_threshold": config.pp_aug_turn_threshold},
            **meta,
        },
        out_path,
    )

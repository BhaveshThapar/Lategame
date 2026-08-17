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
from torch import nn
from torch.utils.data import DataLoader, random_split

from rotomai.data.rl_dataset import RLDataset
from rotomai.features.codec import codec_for
from rotomai.model.actor_critic import (
    hl_gauss_target,
    load_actor_critic_weights,
    load_bc_weights,
    value_from_logits,
    value_support,
)
from rotomai.model.factory import MODEL_ACTOR_CRITIC, build_model, model_metadata
from rotomai.model.policy import (
    factored_accuracy,
    factored_cross_entropy_none,
    factored_logits,
    factored_masked_logits,
    masked_logits,
)
from rotomai.train.bc import BC_POLICY, select_device

# A flat-MLP BC checkpoint warm-starts the actor-critic trunk + policy head. Legacy
# checkpoints carry no ``model_type`` (default "bc"); ``train_bc`` now stamps BC_POLICY.
_BC_WARM_START_KINDS = ("bc", BC_POLICY)

# The arch fields that fix the transformer's parameter shapes. An AC->AC warm-start
# cannot honour a request that differs on any of these.
_ARCH_SHAPE_KEYS = ("d_model", "n_layers", "n_heads", "ff_dim")


def _arch_conflicts(requested: dict[str, object], init: dict[str, object]) -> list[str]:
    """Shape keys on which an explicit arch request disagrees with a warm-start ckpt."""
    return [
        f"{k}: requested {requested[k]!r} != checkpoint {init[k]!r}"
        for k in _ARCH_SHAPE_KEYS
        if k in init and requested[k] != init[k]
    ]


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
    # Architecture: "actor_critic" (flat MLP, M3) or "entity_transformer" (M5).
    # The transformer arch fields are ignored for the MLP.
    model_type: str = MODEL_ACTOR_CRITIC
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 256
    # True when the caller explicitly asked for the shape above (i.e. passed the CLI
    # flags). An AC->AC warm-start must adopt the init checkpoint's arch -- the strict
    # state-dict load demands it -- which would SILENTLY discard that request, so we
    # raise instead. The self-play loop leaves this False: it defers to the checkpoint.
    arch_explicit: bool = False
    # Learned species/move/item/ability identity embeddings (transformer only).
    # The R-ENCODE gate adopted prior-init as the BC default; offrl must pass it
    # through too, else the factory silently falls back to random-init embeddings.
    id_embed: bool = True
    id_embed_dim: int = 32
    id_embed_init: str = "random"  # "random" | "prior"
    # Fixed value support: when both set, skip deriving it from the shard's returns.
    # The self-play loop pins these to the warm-start checkpoint's support so the
    # carried-over value head stays calibrated across iterations.
    v_min: float | None = None
    v_max: float | None = None

    def arch(self) -> dict[str, object]:
        """Transformer architecture block (used when ``model_type`` is the transformer)."""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "id_embed": self.id_embed,
            "id_embed_dim": self.id_embed_dim,
            "id_embed_init": self.id_embed_init,
        }


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
    model: nn.Module,
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
            factored = mask.dim() == 3  # (B, slots, A) on doubles; (B, A) on singles
            logits_masked = (
                factored_masked_logits(factored_logits(logits, mask.shape[1]), mask)
                if factored
                else masked_logits(logits, mask)
            )

            target = hl_gauss_target(ret, centers, sigma)
            value_loss = _value_ce(value_logits, target)

            v = value_from_logits(value_logits, centers)
            adv = ret - v.detach()
            weight = torch.clamp(torch.exp(adv / config.beta), max=config.awr_weight_clip)
            # A turn the demonstrator answered with a whole DefaultBattleOrder carries no action
            # to imitate -- but it DOES carry a state the value head should see, and dropping it
            # would break the episode's return scan. So it is excluded from the ACTOR loss by
            # zeroing its weight, and its label is clamped only so cross-entropy can be evaluated
            # at all (`Target -2 is out of bounds`). Unlike BC, the RL path keeps every turn.
            unlearnable = (action < 0).any(dim=1) if action.dim() > 1 else action < 0
            # ALSO exclude a label that is illegal under its OWN mask: its logit sits at NEG_INF,
            # so cross-entropy returns ~1e9 for that row and a handful of them dominate the whole
            # actor loss -- which made the reported figure read 7e5 while the value head trained
            # fine beside it, and made checkpoint selection pick epoch 1 on noise.
            action = action.clamp(min=0)
            gathered = (
                torch.gather(mask, 2, action.unsqueeze(-1)).squeeze(-1).all(dim=1)
                if factored
                else torch.gather(mask, 1, action.unsqueeze(-1)).squeeze(-1)
            )
            unlearnable = unlearnable | ~gathered
            if unlearnable.any():
                weight = weight * (~unlearnable).to(weight.dtype)
            ce = (
                factored_cross_entropy_none(logits_masked, action)
                if factored
                else F.cross_entropy(logits_masked, action, reduction="none")
            )
            actor_loss = (weight * ce).mean()

            loss = actor_loss + config.value_coef * value_loss
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = obs.shape[0]
            totals["actor"] += actor_loss.item() * n
            totals["value"] += value_loss.item() * n
            if factored:
                # Strict: a turn counts only if BOTH slots match the demonstrator.
                totals["acc"] += factored_accuracy(logits_masked, action)[0]
            else:
                totals["acc"] += int((logits_masked.argmax(dim=1) == action).sum().item())
            totals["value_mae"] += float((v - ret).abs().sum().item())
            count += n
    return {k: v / count for k, v in totals.items()}


def train_offline_rl(data_path: str | Path, out_path: str | Path, config: OfflineRLConfig) -> dict:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    print(f"training on {device}")

    dataset = RLDataset(data_path)
    codec = codec_for(dataset.battle_format)
    print(f"codec: {codec.name} (obs {codec.obs_dim}, {codec.n_actions} logits)")
    if config.v_min is not None and config.v_max is not None:
        v_min, v_max = config.v_min, config.v_max
    else:
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

    # Model spec defaults to the config; an AC->AC warm-start overrides it with the
    # init checkpoint's architecture so the carried-over weights load exactly.
    model_meta: dict = {
        "model_type": config.model_type,
        "input_dim": codec.obs_dim,
        "n_actions": codec.n_actions,
        "n_bins": config.n_bins,
        "dropout": config.dropout,
        "hidden_dim": config.hidden_dim,
        "arch": config.arch(),
    }
    init_state = None
    init_kind = "bc"
    if config.bc_init:
        init_ckpt = torch.load(config.bc_init, map_location="cpu", weights_only=False)
        if (
            init_ckpt.get("obs_version") != codec.obs_version
            or init_ckpt.get("input_dim") != codec.obs_dim
        ):
            raise ValueError("Warm-start checkpoint encoder mismatch; cannot warm-start. Retrain.")
        init_state = init_ckpt["state_dict"]
        init_kind = str(init_ckpt.get("model_type", "bc"))
        if init_kind in _BC_WARM_START_KINDS:
            # BC->AC warm-start copies the MLP trunk + policy head; the transformer
            # has no compatible trunk, so it must train from scratch instead.
            if config.model_type != MODEL_ACTOR_CRITIC:
                raise ValueError(
                    f"BC warm-start only supports the MLP actor-critic, not "
                    f"{config.model_type!r}; train it from scratch (omit bc_init)."
                )
            model_meta["hidden_dim"] = int(init_ckpt["hidden_dim"])  # trunk shapes must match
        else:
            # AC->AC warm-start (self-play continuation): load the full state dict,
            # so the architecture must match the checkpoint exactly.
            if int(init_ckpt["n_bins"]) != config.n_bins:
                raise ValueError(
                    f"Warm-start n_bins {init_ckpt['n_bins']} != config n_bins {config.n_bins}; "
                    "match them to continue an actor-critic checkpoint."
                )
            init_arch = init_ckpt.get("arch") or {}
            conflicts = _arch_conflicts(config.arch(), init_arch) if config.arch_explicit else []
            if conflicts:
                raise ValueError(
                    "Warm-start checkpoint arch conflicts with the requested architecture, "
                    "which an AC->AC warm-start cannot honour (the full state dict must "
                    f"load). {'; '.join(conflicts)}. To train the requested arch, drop the "
                    "warm-start (bc_init='') and train from scratch."
                )
            model_meta = {
                "model_type": init_kind,
                "input_dim": codec.obs_dim,
                "n_actions": int(init_ckpt.get("n_actions", codec.n_actions)),
                "n_bins": int(init_ckpt["n_bins"]),
                "dropout": config.dropout,
                "hidden_dim": int(init_ckpt.get("hidden_dim", config.hidden_dim)),
                "arch": init_ckpt.get("arch") or config.arch(),
            }

    model = build_model(model_meta).to(device)
    if init_state is not None:
        if init_kind in _BC_WARM_START_KINDS:
            load_bc_weights(model, init_state)
            print(f"warm-started trunk + policy head from {config.bc_init}")
        else:
            load_actor_critic_weights(model, init_state)
            print(f"warm-started full {init_kind} from {config.bc_init}")
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
    model: nn.Module,
    out_path: str | Path,
    battle_format: str,
    config: OfflineRLConfig,
    v_min: float,
    v_max: float,
    metrics: dict,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec = codec_for(battle_format)
    # model_type / arch / dims come from the actual model (which an AC->AC
    # warm-start may have built from the init checkpoint, not from config).
    meta = model_metadata(model)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": meta["model_type"],
            "arch": meta["arch"],
            "input_dim": codec.obs_dim,
            "hidden_dim": meta["hidden_dim"],
            "n_actions": meta["n_actions"],
            "n_bins": meta["n_bins"],
            "v_min": v_min,
            "v_max": v_max,
            "obs_version": codec.obs_version,
            "battle_format": battle_format,
            "metrics": metrics,
        },
        out_path,
    )

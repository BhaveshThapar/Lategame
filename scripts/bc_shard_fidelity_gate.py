"""Decide whether the reconstructed BC shard may be trained on.

`scripts/rebuild_bc_shard.py` recovers the winners-only rows from the surviving offline-RL shard and
lands within 43 rows (0.07%) of the 61,723 the build log records. Row-count agreement is NECESSARY
and NOT SUFFICIENT: a reconstruction that took the wrong POV of every pair would also have the right
number of rows. The rows have to teach the same thing.

THE READOUT is BC validation accuracy, because that number exists in the record for the exact shard
being reconstructed: `docs/RESULTS.md` books the v5 BC gate at **0.647 +/- 0.002** over three seeds
(0.647 / 0.646 / 0.649), entity transformer with dex-prior identity embeddings.

KILL CRITERIA, written before running:

    PASS   |mean val-acc - 0.647| <= 0.020. The shard teaches what the recorded one taught.
    FAIL   otherwise. Do not train the history arm on it; the reconstruction is not the shard.

WHY 0.020 AND NOT 0.002. The recorded spread is the seed-to-seed spread of ONE configuration, and
this run cannot reproduce that configuration exactly -- the epoch count of the recorded run is not
in the record. A band tight enough to catch a schedule difference would fail for reasons that have
nothing to do with the reconstruction. 0.020 is ten times the recorded seed spread and still an
order of magnitude smaller than the gap to the 0.636 reveal-order baseline the same build measured,
so it discriminates the thing at issue: whether these are the winners' turns or somebody else's.

A control arm is scored alongside it: the same training on the shard the SIGN test would have
produced (702 extra rows, ~1.1% of them losing POVs). If both arms pass, this gate has no teeth on
this dataset and says so rather than claiming a result -- the same negative-control discipline
`scripts/ou_ingest_gate.py` applies to the ingest fidelity check.

    python scripts/bc_shard_fidelity_gate.py --epochs 6 --seeds 0,1,2
    python scripts/bc_shard_fidelity_gate.py --skip-control    # the arm only, half the compute
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_bc_shard import (  # noqa: E402
    RECORDED_BC_ROWS,
    Rebuild,
    rebuild,
    save,
    segment_episodes,
)

#: `docs/RESULTS.md`, the v5 BC gate: "BC ET+prior val-acc 0.647 / 0.646 / 0.649".
RECORDED_VAL_ACC = 0.647
RECORDED_SEED_SPREAD = 0.002
#: Pre-registered band. See the module docstring for why it is not the seed spread.
BAND = 0.020
DEFAULT_OUT = Path("results/bc_shard_fidelity_gate.json")


@dataclass
class ArmResult:
    name: str
    rows: int
    accuracies: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.accuracies))

    @property
    def spread(self) -> float:
        return float(np.std(self.accuracies, ddof=1)) if len(self.accuracies) > 1 else 0.0

    @property
    def delta(self) -> float:
        return self.mean - RECORDED_VAL_ACC

    @property
    def passes(self) -> bool:
        return abs(self.delta) <= BAND

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.name,
            "rows": self.rows,
            "val_acc": self.accuracies,
            "mean": self.mean,
            "seed_spread": self.spread,
            "delta_vs_recorded": self.delta,
            "verdict": "PASS" if self.passes else "FAIL",
        }


def _by_sign(rl_shard: Path, out: Path) -> int:
    """Write the control shard the SIGN test would have produced. Returns its row count."""
    data = np.load(rl_shard, allow_pickle=False)
    done = np.asarray(data["done"], dtype=bool)
    reward = np.asarray(data["reward"], dtype=np.float64)
    episodes = segment_episodes(done)
    won = reward[episodes.ends] > 0
    rows = np.concatenate(
        [np.arange(s, e + 1) for s, e in zip(episodes.starts[won], episodes.ends[won], strict=True)]
    )
    done_out = np.zeros(len(rows), dtype=bool)
    done_out[np.cumsum(episodes.lengths[won]) - 1] = True
    save(
        Rebuild(
            obs=np.asarray(data["obs"])[rows],
            action=np.asarray(data["action"])[rows],
            mask=np.asarray(data["mask"])[rows],
            done=done_out,
            battle_format=str(data["battle_format"].item()),
            obs_version=str(data["obs_version"].item()),
            obs_dim=int(data["obs_dim"].item()),
            n_episodes=len(episodes),
            n_winner_episodes=int(won.sum()),
            rows=len(rows),
            rows_by_sign=len(rows),
            disagreeing_pairs=0,
        ),
        out,
    )
    return len(rows)


def score_arm(name: str, shard: Path, rows: int, seeds: list[int], epochs: int,
              workdir: Path) -> ArmResult:
    """Train one BC model per seed on ``shard`` and collect the best validation accuracy."""
    import torch

    from rotomai.train.bc import TrainConfig, train_bc

    accuracies: list[float] = []
    for seed in seeds:
        out = workdir / f"{name}_s{seed}.pt"
        train_bc(
            shard,
            out,
            TrainConfig(
                epochs=epochs,
                device="cpu",
                model_type="entity_transformer",
                id_embed=True,
                id_embed_init="prior",
                seed=seed,
            ),
        )
        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        accuracies.append(float(ckpt["metrics"]["val_acc"]))
        print(f"  {name} seed {seed}: val_acc {accuracies[-1]:.4f}", flush=True)
    return ArmResult(name=name, rows=rows, accuracies=accuracies)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rl-shard", type=Path, default=Path("data/gen9ou_v7_rl.npz"))
    parser.add_argument("--workdir", type=Path, default=Path("checkpoints/bc_shard_fidelity"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--skip-control", action="store_true")
    args = parser.parse_args(argv)

    if not args.rl_shard.exists():
        print(f"{args.rl_shard} not found -- stage it first.", file=sys.stderr)
        return 1

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.workdir.mkdir(parents=True, exist_ok=True)

    result = rebuild(args.rl_shard)
    arm_shard = args.workdir / "bc_pairwise.npz"
    save(result, arm_shard)
    print(f"arm shard: {result.rows} rows (recorded {RECORDED_BC_ROWS})", flush=True)

    arms = [score_arm("pairwise", arm_shard, result.rows, seeds, args.epochs, args.workdir)]
    if not args.skip_control:
        control_shard = args.workdir / "bc_by_sign.npz"
        control_rows = _by_sign(args.rl_shard, control_shard)
        print(f"control shard: {control_rows} rows (the sign test's answer)", flush=True)
        arms.append(
            score_arm("by_sign", control_shard, control_rows, seeds, args.epochs, args.workdir)
        )

    arm = arms[0]
    report: dict[str, object] = {
        "schema": "rotomai.gate.bc_shard_fidelity/1",
        "gate": "Does the reconstructed BC shard teach what the lost one taught?",
        "recorded": {
            "val_acc": RECORDED_VAL_ACC,
            "seed_spread": RECORDED_SEED_SPREAD,
            "rows": RECORDED_BC_ROWS,
            "source": "docs/RESULTS.md, the v5 BC gate (ET + dex prior, 3 seeds)",
        },
        "protocol": {
            "band": BAND,
            "seeds": seeds,
            "epochs": args.epochs,
            "model": "entity_transformer, id_embed_init=prior, trainer defaults otherwise",
        },
        "arms": [a.as_dict() for a in arms],
        "verdict": "PASS" if arm.passes else "FAIL",
    }
    if len(arms) > 1:
        # Stated whether or not it is convenient: a control that also passes means this gate did
        # not discriminate, and reporting the arm alone would overclaim.
        report["control_has_teeth"] = not arms[1].passes
        report["separation"] = arm.mean - arms[1].mean

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{arm.name}: mean {arm.mean:.4f} +/- {arm.spread:.4f}  "
          f"(recorded {RECORDED_VAL_ACC}, delta {arm.delta:+.4f}, band +/-{BAND})")
    if len(arms) > 1:
        print(f"{arms[1].name}: mean {arms[1].mean:.4f}  "
              f"separation {report['separation']:+.4f}  "
              f"control has teeth: {report['control_has_teeth']}")
    print(f"VERDICT {report['verdict']} -> {args.out}")
    return 0 if arm.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())

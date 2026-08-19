"""Reconstruct the winners-only BC shard from the offline-RL shard, without re-fetching replays.

THE PROBLEM. `data/gen9ou_v7_bc.npz` is gone from disk, and so are the 2,760 raw replay JSONs under
`replays/gen9ou/`. Only `data/gen9ou_v7_rl.npz` survived. Re-scraping is not an option and
`scripts/cluster/README.md` says so in terms: `fetch-replays` walks Showdown's LIVE search index, so
a re-fetch months later returns a different replay set, and any build comparing against an older
checkpoint is then confounded by data rather than by the lever under test.

THE OBSERVATION. The two shards were emitted by one pass of `_ShardBuilder`. The RL shard keeps
every labelled turn of every finished POV; the BC shard is the same rows filtered to the POVs that
WON (`data/ingest.py:451`, the M2 reward filter). The rows are therefore already present -- what was
lost is only the per-POV win/loss label. Recovering that label recovers the shard.

RECOVERING THE LABEL. `done` marks the last turn of each POV episode, so the episodes segment
exactly. Two things then suggest themselves, and only one of them works:

    SIGN OF THE TERMINAL REWARD -- WRONG. It reconstructs 62,425 rows against the recorded 61,723,
    a 702-row (1.1%) error. The terminal reward is `state_value(terminal) - state_value(last)`,
    which mixes the +/- victory jump with that step's HP and faint deltas, so a POV that lost while
    landing a big final hit can still diff positive, and vice versa.

    HIGHER TERMINAL REWARD OF THE POV PAIR -- RIGHT. Both POVs of one replay are appended
    back-to-back, and exactly one of them won. Comparing the two makes the shared shaping terms
    cancel and leaves the victory jump as the discriminator. This reconstructs 61,766 rows against
    61,723: 43 rows, 0.07%.

WHAT THE RESIDUAL IS. 5,503 episodes is odd and short of 2 x 2,760, so 17 POVs were dropped as
unusable at ingest and their partners are unpaired. Those, plus any true tie, are the residue.
They are reported, not hidden: `--verify` prints the count and fails past `--tolerance`.

WHAT THIS DOES NOT CLAIM. Row-count agreement is necessary, not sufficient. The real check is that a
BC run on the rebuilt shard reproduces the recorded validation accuracy (0.647 +/- 0.002 over three
seeds, `docs/RESULTS.md`), and that gate is `scripts/bc_shard_fidelity_gate.py`. Use this script to
build the shard; use that one to decide whether to believe it.

    python scripts/rebuild_bc_shard.py --verify            # report only, writes nothing
    python scripts/rebuild_bc_shard.py --out data/gen9ou_v7_bc_rebuilt.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_RL_SHARD = Path("data/gen9ou_v7_rl.npz")
DEFAULT_OUT = Path("data/gen9ou_v7_bc_rebuilt.npz")

#: The BC row count `docs/RESULTS.md` records for the v5 re-ingest that produced these shards
#: ("v5 = 119,996 turns / 61,723 BC"). The reconstruction is measured against it.
RECORDED_BC_ROWS = 61_723
RECORDED_RL_ROWS = 119_996
#: Fraction of `RECORDED_BC_ROWS` the reconstruction may differ by before it is rejected.
DEFAULT_TOLERANCE = 0.005


@dataclass(frozen=True)
class Episodes:
    """Episode boundaries of a trajectory shard, derived from its ``done`` column."""

    starts: np.ndarray
    ends: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return self.ends - self.starts + 1

    def __len__(self) -> int:
        return len(self.ends)


def segment_episodes(done: np.ndarray) -> Episodes:
    """Split a shard into POV episodes on its ``done`` flags.

    A shard whose last row is not ``done`` has a truncated trailing episode; it is dropped rather
    than treated as complete, because its terminal reward would not carry a victory jump and would
    therefore lose every pairwise comparison it entered.
    """
    ends = np.flatnonzero(done)
    if ends.size == 0:
        raise ValueError("no episode boundaries: the shard's `done` column is all False")
    starts = np.concatenate(([0], ends[:-1] + 1))
    return Episodes(starts=starts, ends=ends)


def winners_by_pair(terminal_reward: np.ndarray) -> np.ndarray:
    """Boolean mask over episodes: True where that POV is the winner of its pair.

    Episodes arrive as consecutive (p1, p2) pairs from one replay, so the winner is the higher
    terminal reward of the two. The shaping terms are shared between the pair and cancel; the
    victory jump does not. An odd trailing episode has no partner and falls back to the sign test.
    """
    n = len(terminal_reward)
    won = np.zeros(n, dtype=bool)
    for first in range(0, n - 1, 2):
        second = first + 1
        won[first if terminal_reward[first] >= terminal_reward[second] else second] = True
    if n % 2:
        won[-1] = bool(terminal_reward[-1] > 0)
    return won


def pairs_disagreeing(terminal_reward: np.ndarray) -> int:
    """Pairs where the two POVs do NOT straddle zero -- the sign test's error surface.

    Reported as a diagnostic: it is the population on which the sign test and the pairwise test
    give different answers, and it is why the sign test is off by 702 rows.
    """
    n = len(terminal_reward) - (len(terminal_reward) % 2)
    positive = terminal_reward[:n] > 0
    return int(sum(1 for i in range(0, n, 2) if positive[i] == positive[i + 1]))


@dataclass
class Rebuild:
    """The reconstructed shard plus everything needed to judge it."""

    obs: np.ndarray
    action: np.ndarray
    mask: np.ndarray
    done: np.ndarray
    battle_format: str
    obs_version: str
    obs_dim: int
    n_episodes: int
    n_winner_episodes: int
    rows: int
    rows_by_sign: int
    disagreeing_pairs: int

    def report(self, expected: int = RECORDED_BC_ROWS) -> dict[str, object]:
        delta = self.rows - expected
        return {
            "schema": "rotomai.data.bc_rebuild/1",
            "obs_version": self.obs_version,
            "obs_dim": self.obs_dim,
            "battle_format": self.battle_format,
            "episodes": self.n_episodes,
            "winner_episodes": self.n_winner_episodes,
            "rows_pairwise": self.rows,
            "rows_by_sign": self.rows_by_sign,
            "recorded_rows": expected,
            "delta_rows": delta,
            "delta_fraction": abs(delta) / expected,
            "disagreeing_pairs": self.disagreeing_pairs,
        }


def rebuild(rl_shard: str | Path) -> Rebuild:
    """Reconstruct the winners-only BC rows from an offline-RL shard."""
    data = np.load(Path(rl_shard), allow_pickle=False)
    for key in ("obs", "action", "mask", "reward", "done"):
        if key not in data:
            raise ValueError(f"{rl_shard} is not an offline-RL shard: no {key!r} column")

    done = np.asarray(data["done"], dtype=bool)
    reward = np.asarray(data["reward"], dtype=np.float64)
    episodes = segment_episodes(done)
    terminal = reward[episodes.ends]

    won = winners_by_pair(terminal)
    rows = np.concatenate(
        [np.arange(s, e + 1) for s, e in zip(episodes.starts[won], episodes.ends[won], strict=True)]
    )

    # The BC shard has no `done` column of its own -- `Dataset` never carried one. It is written
    # here anyway, because history-conditioned training has to know where an episode starts or it
    # stacks the tail of the previous battle onto the head of the next one. Extra keys are ignored
    # by `BCDataset`, so the shard stays loadable by everything that exists today.
    rebuilt_done = np.zeros(len(rows), dtype=bool)
    rebuilt_done[np.cumsum(episodes.lengths[won]) - 1] = True

    return Rebuild(
        obs=np.asarray(data["obs"])[rows],
        action=np.asarray(data["action"])[rows],
        mask=np.asarray(data["mask"])[rows],
        done=rebuilt_done,
        battle_format=str(data["battle_format"].item()),
        obs_version=str(data["obs_version"].item()),
        obs_dim=int(data["obs_dim"].item()),
        n_episodes=len(episodes),
        n_winner_episodes=int(won.sum()),
        rows=len(rows),
        rows_by_sign=int(episodes.lengths[terminal > 0].sum()),
        disagreeing_pairs=pairs_disagreeing(terminal),
    )


def save(result: Rebuild, path: str | Path) -> None:
    """Write the shard in the exact `collect.save` schema, plus `done`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=result.obs.astype(np.float32),
        action=result.action.astype(np.int64),
        mask=result.mask.astype(bool),
        done=result.done,
        obs_version=np.array(result.obs_version),
        obs_dim=np.array(result.obs_dim),
        battle_format=np.array(result.battle_format),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rl-shard", type=Path, default=DEFAULT_RL_SHARD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expect-rows", type=int, default=RECORDED_BC_ROWS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--verify", action="store_true", help="report only; write nothing")
    parser.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    args = parser.parse_args(argv)

    if not args.rl_shard.exists():
        print(
            f"{args.rl_shard} not found. `data/` is gitignored, so a clone has no shards -- "
            "stage it from the machine that produced it (scripts/cluster/README.md).",
            file=sys.stderr,
        )
        return 1

    result = rebuild(args.rl_shard)
    report = result.report(args.expect_rows)
    delta_fraction = abs(result.rows - args.expect_rows) / args.expect_rows
    within = delta_fraction <= args.tolerance
    report["verdict"] = "PASS" if within else "FAIL"

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"rl shard      {args.rl_shard} ({result.obs_version}, {result.battle_format})")
        print(f"episodes      {result.n_episodes}  ({result.n_winner_episodes} winners)")
        print(f"rows pairwise {result.rows}   vs recorded {args.expect_rows}  "
              f"delta {report['delta_rows']:+d} ({report['delta_fraction']:.4%})")
        print(f"rows by sign  {result.rows_by_sign}   "
              f"delta {result.rows_by_sign - args.expect_rows:+d}  -- the method NOT used")
        print(f"pairs whose two POVs share a reward sign: {result.disagreeing_pairs}")
        print(f"verdict       {report['verdict']} (tolerance {args.tolerance:.3%})")

    if not within:
        print(
            "\nRejected: the reconstruction does not match the recorded row count. Do NOT train "
            "on this shard -- a silently different dataset is the one confound the whole "
            "no-re-fetch argument exists to avoid.",
            file=sys.stderr,
        )
        return 1

    if not args.verify:
        save(result, args.out)
        print(f"\nwrote {result.rows} rows to {args.out}")
        print(
            "Row count agreeing is necessary, not sufficient. Run "
            "scripts/bc_shard_fidelity_gate.py before any claim rests on this shard."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

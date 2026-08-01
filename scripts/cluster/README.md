# Running on a Slurm cluster (UMIACS)

## Why the cluster, and what to ask for

Build 21 measured the actual constraints, and they are **not** what you would guess:

| resource | verdict |
|---|---|
| **GPU** | **Do not request one.** The net is 4.56M params. A GPU does nothing for it and only lengthens the queue. |
| **RAM** | **The binding constraint.** `grad_noise_diag` drove a 16 GB machine to 14.8 GB swap and stalled **48 min** in `UN` (uninterruptible I/O wait) on a single gradient phase. Ask for **64 GB+**. |
| **CPU** | Modest per job (~2 cores are actually busy). The parallelism you want is **across** jobs, not within one. |
| **Job durability** | A sleeping laptop reaped a probe run and its unwritten JSON. `grad_noise_diag` serializes only at the end, so an interrupted run loses **everything**. Slurm fixes this. |

The win is that seeds run **in parallel** rather than sequentially: 3 seeds went 4.6 h serial on the
laptop; as an `sbatch` array they are one seed's wall-clock.

**But the parallelism stops at 4 tasks** (measured, Build 24 planning). `tron` caps a *user* at
`cpu=32,mem=256G` (`sacctmgr show qos where name=tron format=MaxTRESPU`), and every script here asks
8 CPU / 64 GB — so **exactly 4 tasks run concurrently**, with CPU and memory binding at the same
time. A 3-seed array fits; two of them do not. Build 23 submitted 6 tasks and `sacct` shows four
starting at once and the rest starting *the second* a slot freed (`7181042_1` at 00:51:10 against
`7181042_0`'s 00:51:09 end). Slurm backfills perfectly, so **submit everything and never stage waves
by hand** — but estimate in **task-hours ÷ 4**, not "the arms run concurrently."

## The one thing that made this possible

poke-env's `LocalhostServerConfiguration` **hardcodes `ws://localhost:8000`**. Two jobs on one host
would therefore share a single Showdown server and silently see each other's battles — a data
corruption that no gate here would catch. `lategame/config.py` now reads **`LATEGAME_SHOWDOWN_PORT`**
(default 8000, byte-identical to poke-env's config — see `tests/test_config.py`), and each job starts
its own server on its own port.

## Setup (once)

```bash
bash scripts/cluster/setup_umiacs.sh
```

**Measured on UMIACS Nexus login nodes: there is no conda anywhere** — no module (`module avail`
comes up empty), no system install at any common path. So this script does not try to find one; it
bootstraps a private Miniforge3 under `$REPO_DIR/.miniforge3` (no root needed, gitignored) and
creates the `lategame` env from `environment.yml` inside it. `node` is more forgiving: `module load
nodejs` alone already satisfies Showdown's own `>=16` requirement (measured: v16.20.2); the conda
env's newer `nodejs` (environment.yml wants `>=18`) simply takes over on PATH once activated, and
nothing depends on which one wins.

Idempotent — reruns reuse an existing `.miniforge3` or `lategame` env rather than recreating them.

`_job_common.sh`'s `activate_env()` finds this same bootstrap at job time: `sbatch` does **not**
inherit an interactive shell's `conda activate`, so every job script sources it before doing
anything else. If a job fails with "no conda found," rerun this setup script first.

## Run

**Always pass `-p tron --qos=medium`.** The association default QoS has a **max of 4 CPUs per job**,
so every script here — all of which ask for 8 — is rejected outright at submit time with
`CPU count specification invalid`. There is no way to discover this except by hitting it. `medium`
allows the 8-CPU/64 GB spec and a 2-day wall clock (`sacctmgr show qos`); `nexus` is the account.

```bash
# Build 22 Stage A: probe |G|^2 at iter_0 for both warm starts (the pre-registered discriminator)
sbatch -p tron --qos=medium scripts/cluster/stage_a.slurm

# Build 22 Stage B-0: the reduced-epoch offline dose-response qualifier (train + probe per task).
# Needs the RL shard staged first -- see below.
sbatch -p tron --qos=medium --array=0-2 scripts/cluster/stage_b0.slurm

# A 3-seed PPO sweep, in parallel
sbatch -p tron --qos=medium --array=0-2 scripts/cluster/ppo_seed.slurm
```

Each task picks a port from its array index, starts a private Showdown on it, waits for the port to
accept connections, runs, and tears the server down on exit (including on failure — see the `trap`).
Array tasks routinely land on the **same node** (measured: 7141999 tasks 1 and 2 both on tron64),
which is precisely the collision the per-task port exists to prevent.

## Staging data

`/data/` is gitignored, so **no shard arrives with a clone** — a fresh node has checkpoints but no
training data. Copy the original from the machine that produced it:

```bash
rsync -avP <laptop>:<repo>/data/gen9ou_v7_rl.npz $REPO_DIR/data/
```

**Do not regenerate a shard with `fetch-replays`.** It walks Showdown's *live* search index, so a
re-fetch months later returns a different replay set, and any build comparing against an older
checkpoint is then confounded by data rather than by the lever under test.

To check a shard **is** the one that trained a given checkpoint: `train_offline_rl` derives the value
support from the shard's own returns and stamps it into the checkpoint, so `v_min`/`v_max` act as a
fingerprint of the training data. `stage_b0.slurm`'s preflight asserts exactly this (and `RLDataset`
independently raises on an `obs_version`/dim mismatch, so a stale-encoder shard cannot slip through).

## Before you trust a result

Everything in `plan.md` §13.1 still applies, in particular:

- Run `scripts/ppo_telemetry.py` **first** after a run. The stdout log is gitignored; that JSON is the
  durable copy of the trust-region certificate, and without it a NULL cannot be attributed to the lever.
- Pool seeds with `scripts/merge_gate_seeds.py`. A dropped seed silently halves the strength gate's
  power — it still prints a verdict.
- `grad_noise_diag`'s `NOISE_LIMITED` verdict answers **Build 20's** question. `B_simple` is a ratio,
  `tr(Σ)/|G|²`; when `|G|²` → 0 it explodes and "collect more samples" is exactly wrong.

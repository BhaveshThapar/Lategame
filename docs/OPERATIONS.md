# Operations

Everything needed to *run* RotomAI: environment, simulator, training and evaluation pipelines, the
cluster, live play, and the traps that cost time. The [README](../README.md) is the entry point and
carries the results; this file is the manual. The complete experimental record — every build in the
order it ran, with its pre-registration and verdict — is [RESULTS.md](RESULTS.md).

- [Setup](#setup)
- [Running and evaluating](#running-and-evaluating)
- [The eval ladder](#the-eval-ladder)
- [Live play](#live-play)
- [Data & training pipelines](#data--training-pipelines)
- [Cluster](#cluster)
- [Reading CI without `gh`](#reading-ci-without-gh)
- [Checkpoint retention](#checkpoint-retention)

---

## Setup

```bash
# 1. Python env (Python 3.11, isolated) — environment.yml runs `pip install -e ".[dev,ml]"`,
#    torch included: train / grad_noise_diag / the bc agent all die at the first `import torch`
conda env create -f environment.yml
conda activate rotomai

# 2. On a CPU-only box, install the CPU torch wheel FIRST to avoid pulling ~2.5 GB of CUDA:
#    pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

# 3. Local Showdown server + vendored simulator (fetches smogon/pokemon-showdown into
#    third_party/ and builds dist/ — dist/ is also used by replay re-simulation)
bash scripts/setup_server.sh
```

**The vendored simulator is pinned** (`SHOWDOWN_REV` in `scripts/setup_server.sh`), and bumping the
pin is a deliberate act. gen9randombattle sets change upstream, so the same PRNG seed rolls
different teams under a different rev and every recorded inputlog goes illegal partway through —
which is how the R-PREDICT fidelity and resim end-to-end tests silently broke against a simulator
that had moved on. Bump the pin and the inputlog fixture in `tests/conftest.py` must be regenerated
in the same commit:

```bash
node scripts/gen_inputlog_fixture.js third_party/pokemon-showdown
```

**`node` must be on PATH**, not merely installed. `scripts/run_server.sh` execs `node` directly, so
running it outside the activated env fails with `exec: node: not found`; and without `node` six
simulator tests self-skip even when `dist/` is built.

---

## Running and evaluating

```bash
# Start the local server (ws://localhost:8000) — required for any battle/eval
bash scripts/run_server.sh

# Evaluate agents head-to-head (the heuristic is the baseline to beat)
python -m rotomai.cli evaluate --p1 heuristic --p2 random --n 100
python -m rotomai.cli evaluate --p1 offrl --p2 heuristic --n 100
```

Agent names: `random`, `maxbasepower`, `simpleheuristics`, `heuristic` (baselines);
`bc`, `offrl`, `ppo` (learned — load their default checkpoint); `search` (R-PREDICT
depth-limited lookahead on the GREEN checkpoint — config via `ROTOMAI_SEARCH_*` env vars).
`simpleheuristics` is poke-env's `SimpleHeuristicsPlayer` imported unmodified; `heuristic` is this
repo's own.

**Score a PPO checkpoint through `offrl`, never through `ppo`.** `PPORecordingAgent` is the
training rollout agent and forces `sample=True`; the same `v25b` terminal checkpoint measures
**0.675 sampled vs 0.767 greedy** against the heuristic. Every published number here reads these
checkpoints greedily through `offrl`, which is the deployed policy. Both `eval-ladder` and `live`
refuse a training-rollout agent with a checkpoint outright for this reason.

**Teambuilt formats are defaulted, not guessed.** `evaluate` and `live` both fill in the committed
pool for the format (`rotomai/teambuilding/data/`), and both refuse a teamless build on a teambuilt
format. Random Battles must *not* get a pool — the server supplies the teams there.

---

## The eval ladder

G2's metric. A VARIED opponent field, which is the only condition under which GXE/Glicko-1 carry
information a win rate did not: round-robin plus one joint Bradley-Terry fit on the Glicko scale,
with `heuristic` pinned at 1500 to fix the gauge.

```bash
python -m rotomai.cli eval-ladder --format gen9ou \
    --team-pool rotomai/teambuilding/data/teams_gen9ou.packed \
    --n 150 --out results/eval_ladder_gen9ou.json
```

NOT a replacement for `scripts/seed_strength_gate.py`, and **NOT comparable to a Showdown GXE**,
which is measured against humans. The RD it reports is a LOWER bound: on a teambuilt format battles
cluster by team matchup, so the effective sample is under `n`.

**A single ladder cell is not a measurement.** Build 28 found a `v26b` s0 vs `simpleheuristics` cell
at 0.8167 (n=300) that re-measured at 0.741 [0.713, 0.767] at n=1000 — non-overlapping. Use the
ladder for the *field* fit; use `seed_strength_gate.py` for any number you intend to publish.

---

## Live play

M5 deploy / G1. Default mode is `accept` — opt-in opponents only. Credentials come from
`$ROTOMAI_PS_USERNAME` / `$ROTOMAI_PS_PASSWORD`; the password is never a flag, never logged, and
never reaches `--out`.

```bash
python -m rotomai.cli live --mode accept --n 5 --format gen9ou \
    --checkpoint checkpoints/ppo_v26b_s0/iter_320.pt
python -m rotomai.cli live --mode challenge --opponent <user> --n 3   # opt-in opponent
python -m rotomai.cli live --server ws://localhost:8000/showdown/websocket --allow-guest --n 1
```

**Defaults are the measured policy.** `--agent` defaults to `offrl` (the greedy read) and
`--loop-penalty` to 4.0, because that pair is what every published gen9ou number describes. At
`--loop-penalty 0` a live session deploys a policy no measurement in this repo covers.

**An empty password on a public host is a hard error, on purpose.** poke-env would otherwise send
`/trn <name>,0,` , get back `|updateuser| Guest N`, never set `logged_in`, never raise, and hang
forever. `--allow-guest` is for private servers only.

### Timeouts, pacing and the watchdog

| flag | default | what it bounds |
|---|---|---|
| `--battle-timeout` | 900 s | one battle chunk |
| `--stall-timeout` | 300 s | a battle that has stopped advancing turns |
| `--queue-timeout` | 900 s | **no battle in progress** — queued for `/search`, or serving `--battle-delay` |
| `--battle-delay` | 0.0 s | pause between battles; set it on the ladder |
| `--login-timeout` | 30 s | awaiting `|updateuser|` |
| `--max-restarts` | 3 | client rebuilds on a drop |

The last two rows of the timeout group are one bug's worth of design. The watchdog samples
`(finished battles, Σ turns)` and calls a frozen sample a dead socket — but *between* battles both
are frozen anyway, so a ladder queue was indistinguishable from a wedge. Under one deadline a slow
queue tripped the detector, burned a restart, rebuilt the client and re-searched, which is both a
spurious failure and exactly the reconnect-and-search traffic pattern good etiquette avoids. Hence
two deadlines. A login failure is fatal and never retried: hammering a bad password is how a live
account gets locked.

### The ranked ladder

`--mode ladder` is the PUBLIC RANKED ladder. There is no unranked ladder — `/search` on the public
server IS the rated one. It needs BOTH opt-in channels, so neither a stale export nor a recalled
command can start ranked play on its own:

```bash
export ROTOMAI_LIVE_ALLOW_LADDER=1
python -m rotomai.cli live --mode ladder --ladder-ack i-have-read-plan-md-section-15 \
    --agent offrl --checkpoint checkpoints/ppo_v26b_s0/iter_320.pt \
    --format gen9ou --concurrency 1 --battle-delay 20 --use-live-ratings --n 25
```

`--use-live-ratings` rates each opponent at its OBSERVED rating instead of pinning the field;
without it the session's GXE is a reparameterisation of its own score rate. Read `plan.md` §15 and
NG3 first: a dedicated, clearly-labeled bot account is mandatory, and automated *farming* is out of
scope regardless of the gate.

**`--out` is rewritten, not appended.** The results file is re-serialised after every battle, so a
SIGKILL loses at most one game — but pointing a second run at the same path erases the first. Give
each run its own file.

---

## Data & training pipelines

```bash
# M2/M3 — collect self-play trajectories, train BC then offline RL
python -m rotomai.cli collect-rl --n 50
python -m rotomai.cli train-rl   --data data/gen9rb_rl.npz

# M4/M5 — self-play league / on-policy PPO improvement loops
python -m rotomai.cli selfplay --init checkpoints/offrl_gen9randombattle.pt --iters 8
python -m rotomai.cli ppo      --init checkpoints/offrl_gen9randombattle.pt --iters 8

# Lever experiment gates (win-rate vs the heuristic; write results/*.json)
python scripts/offrl_scale_gate.py     --out results/offrl_scale_gate.json   # Lever 9: AWR @ 82k (GREEN)
python scripts/ppo_continue_gate.py    --out results/ppo_continue_gate.json  # Lever 10: PPO continuation (AMBER)
python scripts/rpredict_fidelity_gate.py --out results/rpredict_fidelity.json  # Lever 11 Gate A: forward-model fidelity (PASS)
python scripts/rpredict_recon_gate.py    --out results/rpredict_recon.json     # Lever 11: reconstruction mini-gate (PASS)
python scripts/rpredict_gate.py          --out results/rpredict_gate_b.json    # Lever 11 Gate B: depth-1 search (AMBER)
python scripts/rpredict_gate.py --depth 2 --opp-cap 4 --out results/rpredict_gate_b2_mean.json  # Lever 12: depth-2 search (AMBER)
python scripts/curriculum_gate.py        --out results/curriculum_gate.json   # Lever 13: tougher-opponent AWR self-play (AMBER)
python scripts/rpredict_oppmodel_gate.py --gate a   # Lever 14 Gate A: opponent-model fidelity (PASS)
python scripts/rpredict_oppmodel_gate.py --gate b --arms whitebox,learned --concurrency 6  # Lever 14 Gate B: real-opponent-model search (AMBER)

# M6 — human replays: fetch, then reconstruct each player's POV either from the public
# spectator log (v1) or by re-simulating the inputlog for the private |request| (v2)
python -m rotomai.cli fetch-replays  --min-rating 1200 --limit 200
python -m rotomai.cli ingest-replays --out data/ingest_gen9rb_rl.npz   # v1 (public-log POV)
python -m rotomai.cli resim-replays  --out data/resim_gen9rb_rl.npz    # v2 (needs node + dist/)
```

### The gen9ou PPO build toolchain, in order

```bash
# On the cluster the seeds are an array job (scripts/cluster/ppo_seed.slurm); each writes its own _s{N}.json.
python scripts/ppo_telemetry.py --log 0 logs/ppo/v21/ppo_v21_s0.log --kl-bar 0.045 --out results/ppo_ou_telemetry_v21.json
#   ^ the TRUST-REGION CERTIFICATE. Run FIRST: the run log is gitignored, this JSON is the durable copy.
#     Job logs live under logs/ — logs/ppo/<build>/ (run stdout), logs/slurm/ (sbatch --output),
#     logs/showdown/<bucket>/ (per-job server). ppo_seed.slurm writes its own; only logs/slurm/.gitkeep
#     and logs/MANIFEST.tsv are tracked. sbatch does NOT create its --output dir, hence the .gitkeep.
#     A NULL is only attributable to the lever if the trust region did not bind (Build 22: it bound 59/150,
#     which cost that build its verdict). --kl-bar MUST be 1.5 * the run's --target-kl, or the certificate
#     names a bar the optimizer never enforced. ppo_seed.slurm derives both from TARGET_KL so they cannot drift.
python scripts/merge_gate_seeds.py --seed-json results/ppo_ou_gate_v21_s{0,1,2}.json \
    --ladder-source ... --note ... --out results/ppo_ou_gate_v21.json
#   ^ pools the per-seed runs. Drop a seed here and the z-test below silently loses its power.
python scripts/seed_strength_gate.py --build v20 ... --build v21 ... --out results/seed_strength_gate_v21.json
#   ^ THE AUTHORITATIVE verdict: every seed's best ckpt, n=300 each, pooled to 900/arm, two-proportion z.
#     Resolves ~+0.07 at z~3. The training curve does NOT decide WIN vs NULL. Absolute rates move
#     between runs (winner's curse); only the within-run DIFFERENCE is trustworthy.
#     ALWAYS pass BOTH arms to ONE invocation (cluster: scripts/cluster/strength_gate.slurm, which
#     preflights that each arm's best checkpoints are staged and NAMES the missing ones). Build 22
#     measured v20's IDENTICAL checkpoints at 0.472 and 0.499 in two runs -- 0.027 apart, at the 0.023
#     SE -- so a cross-run difference under ~0.03 is not a result.
python scripts/grad_noise_diag.py --policy <best> --init <warm-start> --league-dir <run> \
    --games-per-opp 48 --rollouts 6 --splits 5 --out results/grad_noise_diag_v21.json
#   ^ the EXPLAINER: reads |G|^2 (probes[*].arms.same_mix.policy.noise_scale.g_norm_sq).
#     --games-per-opp MUST match the run under test; --splits defaults to 20 (v20 used 5).
#     WARNING: its NOISE_LIMITED verdict answers Build 20's question, not yours. B_simple is a RATIO
#     (tr(Sigma)/|G|^2) — when |G|^2 -> 0 it explodes and "collect more samples" is exactly WRONG.
#     RETIRED for small effects (Build 22): a FIXED checkpoint swung 2.7x on the --seed alone, larger
#     than the 2.2x effect the gate was built to detect. Any gate reading a |G|^2 difference under ~3x
#     MUST run >=2 probe seeds (scripts/cluster/probe_replicate.slurm) and report both, or it is
#     reporting noise. The seed is NOT recorded in the output JSON -- provenance is the filename only.
#     The COSINE from the same runs did replicate (0.312 -> 0.317) and remains usable.
```

**Scoring one arm against a fixed baseline** (Build 28's rebase is this shape — one build, one
opponent, a level estimate with a Wilson interval and no z-test):

```bash
BUILD_A=v26b GATE_A=results/ppo_ou_gate_v26b_terminal.json \
OPPONENT=simpleheuristics N=1000 \
OUT=results/seed_strength_gate_v26b_simpleheuristics.json \
  sbatch -p tron --qos=medium scripts/cluster/strength_gate.slurm
```

Pair any such run with a **cross-run control** — the same arm against a baseline this project has
already published — in the same protocol at the same n. Without it a new number cannot be separated
from a shifted run. Build 28's control landed 0.0013 from the published 0.7513, which is what
licensed reading its simpleheuristics rate as a property of the arm.

---

## Cluster

```bash
# Build 26's whole analysis as ONE submission (server + merge + both strength gates + bias table).
#   The preflight refuses a MISSING or SHORT arm before starting anything: v26b runs in two chunks
#   that write the same per-seed JSON, so between them the file exists and is 160 iters long.
#   N is pinned to Build 25's pre-registered 3000 / 1800 -- the gate's own default is 300, whose
#   MDE (0.079) is wider than any dose this build can produce, so it would book a NULL by design.
sbatch -p tron --qos=medium scripts/cluster/build26_analysis.slurm

# R-EVAL — the agent-only eval ladder.
OUT=results/eval_ladder_gen9ou.json sbatch -p tron --qos=medium scripts/cluster/eval_ladder.slurm
```

**THE PORT COLLISION.** poke-env hardcodes `ws://localhost:8000`, so two jobs on one host silently
share a Showdown server and battle into each other's games — no error, corrupted comparisons.
`scripts/cluster/_job_common.sh` computes `8100 + TASK_ID`; a non-array job takes `TASK_ID=0` and
therefore 8100, which collides with PPO array task 0. Each wrapper sets a base clear of the others
(`eval_ladder.slurm` → 8300, `build26_analysis.slurm` → 8400, the second PPO array → 8200). Set
`ROTOMAI_SHOWDOWN_PORT_BASE` for anything new.

**No GPU.** Do not request one — the net is 4.56M parameters and the binding constraints are RAM
(64 GB) and the local server. The QoS cap on `tron` is `cpu=32,mem=256G`, i.e. exactly four
concurrent 8-CPU/64 GB tasks, so the accounting unit is task-hours ÷ 4. Measured throughput is
209–242 battles/min at concurrency 20. Resource verdicts per job are in
[`scripts/cluster/README.md`](../scripts/cluster/README.md).

**Kill your servers.** `scripts/run_server.sh` execs `node`, so killing a wrapper leaves the server
orphaned and still listening. Verify by the `node` PID, not the shell that started it.

**Give every concurrent run its own `--out`.** A gate's output path is a global. Two overlapping
jobs launched with the same `OUT` both write it and the later one wins — no error, no warning, and
the earlier run's JSON simply ceases to exist while its number lives on only in
`logs/slurm/slurm-<name>-<jobid>.out`. Build 28 lost two of six reads this way. It is the `results/`
twin of the port collision above: a default that is correct for one job and silently wrong for two.
When in doubt the job log, not the JSON, is the primary record of what a run measured.

---

## Reading CI without `gh`

There is no `gh` on the dev host, which is a large part of why a red CI went unnoticed for eleven
commits. It does not need one — the runs are public and the REST API is unauthenticated:

```bash
# last 5 runs: which branch, which commit, red or green
curl -s "https://api.github.com/repos/BhaveshThapar/RotomAI/actions/runs?per_page=5" \
  | python -m json.tool | grep -E '"(run_number|head_branch|head_sha|conclusion)"'

# per-step conclusions for one run (is it `pytest -q` that failed, or `mypy`?)
curl -s "https://api.github.com/repos/BhaveshThapar/RotomAI/actions/runs/<RUN_ID>/jobs" \
  | python -m json.tool | grep -E '"(name|conclusion)"'
```

Only the log *text* needs a token (the download 403s anonymously), so the pass/skip split is not
readable remotely — the step conclusion is. Check this after every push. A local run of the same
command is a reproduction, not an observation, and the distinction is exactly what went wrong.

---

## Producing the README demo GIF

`--save-replays <dir>` writes a self-contained `<username> - <battle_tag>.html` per battle
(poke-env `player.py:682-703`). It renders by loading `replay-embed.js` from
play.pokemonshowdown.com, so it needs **a browser with network** — there is no headless path on the
cluster box, which has no browser, no `ffmpeg` and no image tooling.

**Pick the battle from the log, not by opening 40 replays.** Each file embeds the full battle log
in a `battle-log-data` script tag, so the watchable ones can be found programmatically: count
`|faint|p1` (their KOs) against `|faint|p2` (ours), and treat a `|win|` with 0 turns or a handful of
turns as an opponent forfeit rather than a win worth showing.

The one selected from the first ladder block:

| | |
|---|---|
| file | `replays/live_ladder_gen9ou/RotomLover12 - battle-gen9ou-2666751310.html` |
| result | win, 6-3 on KOs in 17 turns, 10 super-effective hits, opponent at 1004 Elo |
| segment to record | **turns 13-17** — Kingambit comes in and Sucker Punches through Jirachi, Abomasnow and Dragapult for three consecutive KOs to close the game |

Then, on a machine with a browser and `ffmpeg`:

```bash
# 1. open the .html, seek to turn 13, screen-record ~15s through the win
# 2. two-pass palette conversion -- one pass gives visible banding on Showdown's flat colours
ffmpeg -ss T -t 15 -i clip.mp4 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen" /tmp/pal.png
ffmpeg -ss T -t 15 -i clip.mp4 -i /tmp/pal.png \
  -lavfi "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" assets/rotomai_gen9ou.gif
```

Budget ≤3 MB. Commit under `assets/`; nothing in `.gitignore` matches it. **Do not add the
`![...](assets/...)` line to the README until the file exists** — a broken image on the first screen
is worse than no image.

`/replays/` is gitignored, so the source HTML is not part of the committed record. If a linkable
replay is ever wanted, note that poke-env never sends `/savereplay`, so these battles have no
`replay.pokemonshowdown.com` permalink; adding one is a single `ps_client.send_message` in the live
player, and it publishes the opponent's game too, so it should be opt-in.

---

## Checkpoint retention

**What pruning is allowed to take.** 234 intermediate checkpoints (0.657 GiB) were deleted by
`scripts/prune_checkpoints.py`, from the three `ppo_b6f_*` arms. An arm directory keeps four things —
any checkpoint a committed `results/*.json` names, the arm's terminal iteration whether or not a
result names it, `curve.json`, and any file it does not recognise. Every top-level `checkpoints/*.pt`
is out of scope entirely. What goes is the rest: intermediates that existed so `argmax` had something
to range over. Dry run is the default and `--apply` is required, because an arm is ~40 h of
wall-clock and cannot be re-derived from its curve. The rules live in that script's docstring;
`python scripts/prune_checkpoints.py` re-derives them rather than trusting this paragraph.

**Verify before you load.** A checkpoint is a pickle and `torch.load` executes code from it, so
"trust the file on the release page" is not a security posture. `results/release_assets.json` carries
the sha256 and byte length of each asset, is committed, and is derived from `check_artifacts.HEADLINE`
rather than hand-kept, so it cannot drift from the claims those weights back.

```bash
python scripts/release_assets.py --manifest   # re-derive after changing HEADLINE
python scripts/release_assets.py --verify     # re-hash local files against it
python scripts/check_artifacts.py             # every headline's checkpoints still on disk?
```

Adding a `HEADLINE` entry without regenerating the manifest turns `tests/test_release_assets.py` red.

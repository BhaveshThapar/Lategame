# Lategame

A competitive Pokémon Showdown battle agent, trained and evaluated end to end against a pinned local
simulator. Behaviour cloning → offline RL → PPO self-play, across three formats — `gen9randombattle`,
`gen9ou` and `gen9vgc2025regi` — with **every published number gated on a pre-registered evaluation**
and the negative results kept.

[**Quickstart**](#quickstart) · [**Results**](#results) · [**Build log**](docs/RESULTS.md) ·
[**Design doc**](plan.md) · [**Releases**](https://github.com/BhaveshThapar/Lategame/releases)

---

## Quickstart

```bash
conda env create -f environment.yml && conda activate lategame
bash scripts/setup_server.sh          # vendored Showdown at a pinned rev, builds dist/
bash scripts/run_server.sh &          # ws://localhost:8000

python -m lategame.cli evaluate --p1 heuristic --p2 random --n 20 --format gen9ou
```

The rule-based baseline against `random` on a local server, printing a win rate. **No weights, no
GPU, no network** — team pools for the teambuilt formats are committed and defaulted per format.
`lategame --help` lists the rest; [Run](#run) covers training and evaluation, [Develop](#develop)
the test / lint / type bar.

To play the *trained* agents, fetch the release weights — see
[Using the released weights](#using-the-released-weights).

---

## Results

| goal | status | headline |
|---|---|---|
| **G1** live play | built + verified | `lategame/live/`, behind an explicit opt-in |
| **G2** strong-human on one format | **met, both halves** | gen9ou **0.7513** vs the heuristic; agent-only ladder **Glicko 1776.3 / GXE 0.7434** |
| **G3** continual improvement | **booked** | five-dose self-play curve, monotone on both reads (80 → 320 updates) |
| **G4** ≥3 formats through one core | **met** | `gen9randombattle`, `gen9ou`, `gen9vgc2025regi` end to end |
| **G5** skill stack as testable capability | **met** | four capabilities, each on its own gate's criterion |

**Two of the three formats measure as ceiling-bound, and that is a result rather than a shortfall.**
gen9-RB and VGC both return `FORMAT_BOUND` from the three-leg ceiling probe — nothing competent
beats the fixed heuristic there, near-optimal search included — so the project's own instruments say
not to spend more on their strength axes. gen9ou is where headroom was proven and where the strength
result lives.

**Not claimed:** no public *ranked* ladder play (NG3), so the Glicko figure is measured against a
bot field and is **not** comparable to a Showdown GXE against humans; and only the self-play axis of
G3 has a dose-response curve, not the replay-data axis.

Full experimental record, every build in the order it ran with its pre-registration and verdict:
**[docs/RESULTS.md](docs/RESULTS.md)**. The design document, requirements and milestone roadmap are
in [plan.md](plan.md).

---

## Setup

```bash
# 1. Python env (Python 3.11, isolated) — environment.yml runs `pip install -e ".[dev,ml]"`,
#    torch included: train / grad_noise_diag / the bc agent all die at the first `import torch`
conda env create -f environment.yml
conda activate lategame

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

---

## Run

```bash
# Start the local server (ws://localhost:8000) — required for any battle/eval
bash scripts/run_server.sh

# Evaluate agents head-to-head (the heuristic is the baseline to beat)
python -m lategame.cli evaluate --p1 heuristic --p2 random --n 100
python -m lategame.cli evaluate --p1 offrl --p2 heuristic --n 100
```

Agent names: `random`, `maxbasepower`, `simpleheuristics`, `heuristic` (baselines);
`bc`, `offrl`, `ppo` (learned — load their default checkpoint); `search` (R-PREDICT
depth-limited lookahead on the GREEN checkpoint — config via `LATEGAME_SEARCH_*` env vars).

**Score a PPO checkpoint through `offrl`, never through `ppo`.** `PPORecordingAgent` is the
training rollout agent and forces `sample=True`; the same `v25b` terminal checkpoint measures
**0.675 sampled vs 0.767 greedy** against the heuristic. Every published number here reads these
checkpoints greedily through `offrl`, which is the deployed policy. `eval-ladder` refuses
`ppo@<checkpoint>` outright for this reason.

### The eval ladder (G2's metric) and live play

```bash
# R-EVAL: a VARIED opponent field, which is the only condition under which GXE/Glicko-1 carry
# information a win rate did not. Round-robin + one joint Bradley-Terry fit on the Glicko scale,
# with `heuristic` pinned at 1500 to fix the gauge. NOT a replacement for seed_strength_gate.py,
# and NOT comparable to a Showdown GXE (which is measured against humans).
python -m lategame.cli eval-ladder --format gen9ou \
    --team-pool lategame/teambuilding/data/teams_gen9ou.packed \
    --n 150 --out results/eval_ladder_gen9ou.json

# M5 deploy / G1 — live server. Default mode is `accept` (opt-in opponents only). Credentials come
# from $LATEGAME_PS_USERNAME / $LATEGAME_PS_PASSWORD and never reach --out.
python -m lategame.cli live --agent ppo --mode accept --n 5 --format gen9ou
python -m lategame.cli live --mode challenge --opponent <user> --n 3   # opt-in opponent
python -m lategame.cli live --server ws://localhost:8000/showdown/websocket --allow-guest --n 1

# `--mode ladder` is the PUBLIC RANKED ladder, which plan.md NG3 puts out of scope. There is no
# unranked ladder: `/search` on the public server IS the rated one. It needs BOTH opt-in channels,
# so neither a stale export nor a recalled command can start ranked play on its own:
#   export LATEGAME_LIVE_ALLOW_LADDER=1
#   python -m lategame.cli live --mode ladder --ladder-ack i-have-read-plan-md-section-15 \
#       --use-live-ratings --n 50
# --use-live-ratings rates each opponent at its OBSERVED rating instead of pinning the field;
# without it the session's GXE is a reparameterisation of its own score rate. Read plan.md 15 first.
```

### Data & training pipelines

```bash
# M2/M3 — collect self-play trajectories, train BC then offline RL
python -m lategame.cli collect-rl --n 50
python -m lategame.cli train-rl   --data data/gen9rb_rl.npz

# M4/M5 — self-play league / on-policy PPO improvement loops
python -m lategame.cli selfplay --init checkpoints/offrl_gen9randombattle.pt --iters 8
python -m lategame.cli ppo      --init checkpoints/offrl_gen9randombattle.pt --iters 8

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

# Gen 9 OU PPO builds (19-23) — the build-vs-build toolchain. Run in this order.
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

# Build 26's whole analysis as ONE submission (server + merge + both strength gates + bias table).
#   The preflight refuses a MISSING or SHORT arm before starting anything: v26b runs in two chunks
#   that write the same per-seed JSON, so between them the file exists and is 160 iters long.
#   N is pinned to Build 25's pre-registered 3000 / 1800 -- the gate's own default is 300, whose
#   MDE (0.079) is wider than any dose this build can produce, so it would book a NULL by design.
sbatch -p tron --qos=medium scripts/cluster/build26_analysis.slurm

# R-EVAL — the agent-only eval ladder. On the cluster, scripts/cluster/eval_ladder.slurm.
#   MUST set a port base clear of any in-flight PPO build: _job_common.sh computes 8100 + TASK_ID,
#   a non-array job takes TASK_ID=0 -> 8100, and colliding jobs SILENTLY SHARE one Showdown server.
#   The slurm wrapper defaults LATEGAME_SHOWDOWN_PORT_BASE=8300 for exactly this reason.
OUT=results/eval_ladder_gen9ou.json sbatch -p tron --qos=medium scripts/cluster/eval_ladder.slurm

# M6 — human replays: fetch, then reconstruct each player's POV either from the public
# spectator log (v1) or by re-simulating the inputlog for the private |request| (v2)
python -m lategame.cli fetch-replays  --min-rating 1200 --limit 200
python -m lategame.cli ingest-replays --out data/ingest_gen9rb_rl.npz   # v1 (public-log POV)
python -m lategame.cli resim-replays  --out data/resim_gen9rb_rl.npz    # v2 (needs node + dist/)
```

### Using the released weights

**The weights are on the release.** Three checkpoints, 23 MiB, one playable policy per teambuilt
format:
[v1.0.0](https://github.com/BhaveshThapar/Lategame/releases/tag/v1.0.0).
**Restore the paths as you download** — the manifest keys on them, and the gen9ou asset is attached
under the bare name `iter_320.pt`, which does not say which arm it came from:

```bash
B=https://github.com/BhaveshThapar/Lategame/releases/download/v1.0.0
mkdir -p checkpoints/ppo_v26b_s0
curl -sL -o checkpoints/ppo_v26b_s0/iter_320.pt  $B/iter_320.pt          # gen9ou, 17.4 MiB
curl -sL -o checkpoints/doubles_offrl_vgc_v2.pt  $B/doubles_offrl_vgc_v2.pt
curl -sL -o checkpoints/doubles_bc_vgc_v2.pt     $B/doubles_bc_vgc_v2.pt

python scripts/release_assets.py --verify        # want: 3/3 release assets present and verified

python -m lategame.cli evaluate --p1 offrl --p1-checkpoint checkpoints/ppo_v26b_s0/iter_320.pt \
  --p2 heuristic --n 100 --format gen9ou         # the gen9ou ladder top vs the fixed baseline
```

Measured by doing exactly the above into a clean `v1.0.0` clone: `3/3` verified, then **0.825 over
40 battles** on gen9ou and **0.450 over 20** on VGC — the published 0.7513 and 0.467 at small n.

**Verify before you load.** A checkpoint is a pickle and `torch.load` executes code from it, so
"trust the file on the release page" is not a security posture. `results/release_assets.json`
carries the sha256 and byte length of each asset, is committed, and is derived from
`check_artifacts.HEADLINE` rather than hand-kept, so it cannot drift from the claims those weights
back. `--verify` gates on the release assets only; the other 15 checkpoints the manifest lists are
provenance for published curves and were never going to be on your machine, so it says so rather
than reporting them as a failed download.

---

## Develop

```bash
pytest            # 777 tests. With the env active (node ON PATH) + a local server up: 776 pass,
                  #   1 skip -- the opt-in live-client smoke, which LATEGAME_LIVE_TEST=1 enables
                  #   for 777/0. `node` on PATH is load-bearing: without it six simulator tests
                  #   self-skip even with dist/ built.
                  #   On a bare clone -- no server, no built dist/, no checkpoints/ -- 16 self-skip
                  #   and 761 pass. That is what CI runs; a 17th skip is a regression, not noise.
                  #   Run it as `pytest`, not `python -m pytest`: pyproject sets pythonpath so the
                  #   two agree, and CI runs the bare form.
ruff check .
mypy lategame scripts   # both trees; CI runs the same two arguments
```

**Reading the CI result without `gh`.** There is no `gh` on the dev host, which is a large part of
why a red CI went unnoticed for eleven commits. It does not need one — the runs are public and the
REST API is unauthenticated:

```bash
# last 5 runs: which branch, which commit, red or green
curl -s "https://api.github.com/repos/BhaveshThapar/Lategame/actions/runs?per_page=5" \
  | python -m json.tool | grep -E '"(run_number|head_branch|head_sha|conclusion)"'

# per-step conclusions for one run (is it `pytest -q` that failed, or `mypy`?)
curl -s "https://api.github.com/repos/BhaveshThapar/Lategame/actions/runs/<RUN_ID>/jobs" \
  | python -m json.tool | grep -E '"(name|conclusion)"'
```

Only the log *text* needs a token (the download 403s anonymously), so the pass/skip split is not
readable remotely — the step conclusion is. Check this after every push. A local run of the same
command is a reproduction, not an observation, and the distinction is exactly what went wrong.

---

## Layout

| Path | Role |
|---|---|
| `lategame/config.py` | Local server config + format constants |
| `lategame/engine/damage.py` | Expected-damage / move-value estimator |
| `lategame/features/` | `embed_battle` 720-d encoder (`OBS_LAYOUT`) + action-space codec |
| `lategame/model/` | MLP actor-critic + entity transformer + build factory |
| `lategame/agents/` | `heuristic`, `bc`, `offrl`, `ppo` agents |
| `lategame/data/` | self-play collection, reward, replay fetch / ingest (v1) / resim (v2) |
| `lategame/train/` | BC, offline RL, self-play, PPO training loops |
| `lategame/search/` | R-PREDICT: forward model, determinization, expectimax, opponent model |
| `lategame/teambuilding/` | R-TEAM: validator-checked packed team pools for teambuilt formats |
| `lategame/eval/arena.py` | run N battles vs one fixed baseline, report win rates |
| `lategame/eval/rating.py` | R-EVAL: Glicko-1 + GXE (degenerate against a fixed baseline — read the docstring) |
| `lategame/eval/ladder.py` | agent-only eval ladder: round-robin + joint rating fit (the varied field G2 needs) |
| `lategame/live/` | M5 deploy / G1: live-server client, policy gate, session supervisor, telemetry |
| `lategame/cli.py` | all subcommands (eval / collect / train / data / live) |
| `scripts/` | local Showdown server + simulator setup/run, and every experiment gate |

---

## Artifacts & reproducibility

**A clone of this repo contains no weights and no training shards.** `checkpoints/` and `data/` are
gitignored, so every published number was produced from files that exist only on the machine that
ran them. What *is* committed, and is the durable record, is the evidence rather than the artifacts: 236
`results/*.json` gate summaries, each arm's per-iteration `curve.json`, the validator-checked packed
team pools (`lategame/teambuilding/data/`), the encoder vocab and the gen9ou usage prior
(`lategame/features/data/`), and the pinned simulator rev.

**Which record backs which headline.** These name result files rather than checkpoint paths, because
a gate can be re-pinned and a result file cannot:

| headline | record |
|---|---|
| gen9ou **0.7513** selection-free terminal | `results/ppo_ou_gate_v26b_terminal.json`, `results/seed_strength_gate_v26_terminal.json` |
| Glicko **1776.3** / GXE **0.7434** | `results/eval_ladder_gen9ou_v26.json` |
| VGC B6f C1 **0.530** | `results/ppo_vgc_gate_b6f{,_s0,_s1,_s2}.json`, `results/seed_strength_gate_b6f_c1.json`, `results/awr_vgc_arm_b6f.json` |
| VGC corrected ladder **BC 0.453** / **AWR 0.467** | `results/format_ceiling_gate_vgc_v2.json` |
| **G5 MET** — four capabilities, each on its own gate's criterion | `results/g5_capability_gate.json` |

Every checkpoint those records name is present on the machine that produced them, and none of them
ship. `python scripts/check_artifacts.py` re-derives that statement rather than trusting this table.

**58 of the 122 checkpoint paths named across `results/**.json` no longer exist**, cited by 52 of the
236 result files. 27 are top-level warm starts (`bc_gen9ou_v*.pt`, `offrl_scale_*.pt`); the rest are
whole absent arm directories (`ppo_ou_*`, `ppo_scale_*`, `curriculum_*`). The cause is scratch
teardown, not pruning: `scripts/prune_checkpoints.py` iterates directories only
(`plan_prune`, `p.is_dir()`) and only ever `unlink()`s files, so top-level `*.pt` and whole arm dirs
were never candidates. **No headline claim is among the 58** — they back superseded intermediate
builds whose measured numbers remain in the JSON. A reader following an older record to a file will
find nothing; that is a known state, recorded here rather than left to be discovered.

**What pruning is allowed to take.** Distinct from the above, and now load-bearing: 234 intermediate
checkpoints (0.657 GiB) *were* deleted by that script, from the three `ppo_b6f_*` arms. An arm
directory keeps four things — any checkpoint a committed `results/*.json` names, the arm's terminal
iteration whether or not a result names it, `curve.json`, and any file it does not recognise. Every
top-level `checkpoints/*.pt` is out of scope entirely. What goes is the rest: intermediates that
existed so `argmax` had something to range over. Dry run is the default and `--apply` is required,
because an arm is ~40 h of wall-clock and cannot be re-derived from its curve. The rules themselves
live in `scripts/prune_checkpoints.py`'s docstring; `python scripts/prune_checkpoints.py` re-derives
them rather than trusting this paragraph.

**What a clean clone can and cannot reproduce.** It can run the whole pipeline end to end — setup,
collect, train, gate — against a pinned simulator, with the pools, vocab and prior it needs already
committed. It cannot bit-exactly re-derive a published build: an arm is ~40 h of wall-clock and the
shards it trained on are gone. The gate scripts are the reproduction path, not the checkpoints.

---

## License & acknowledgements

This project is MIT-licensed — see [`LICENSE`](LICENSE). `CITATION.cff` carries the citation record.

- **[poke-env](https://github.com/hsahovic/poke-env)** (Haris Sahovic), MIT — the battle-client and
  baseline-player layer every agent here is built on. A pip dependency, declared in `pyproject.toml`.
- **[pokemon-showdown](https://github.com/smogon/pokemon-showdown)** (© 2011–2026 Guangcong Luo and
  other contributors), MIT — the simulator. It is **cloned, not vendored**: `scripts/setup_server.sh`
  fetches it into `third_party/` at the pinned rev `393d5c86`, and `.gitignore` keeps it out of this
  tree entirely. Nothing from it is redistributed here, and its license is its own — read it at
  `third_party/pokemon-showdown/LICENSE` after running the setup script.

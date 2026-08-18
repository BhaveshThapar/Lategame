# RotomAI

A competitive Pokémon Showdown battle agent, trained and evaluated end to end against a pinned local
simulator. Behaviour cloning → offline RL → PPO self-play, across three formats — `gen9randombattle`,
`gen9ou` and `gen9vgc2025regi` — with **every published number gated on a pre-registered evaluation**
and the negative results kept.

[**Results**](#results) · [**How it works**](#how-it-works) · [**Operations**](docs/OPERATIONS.md) ·
[**Build log**](docs/RESULTS.md) · [**Design doc**](plan.md) · [**Releases**](https://github.com/BhaveshThapar/RotomAI/releases)

---

## Quickstart

```bash
conda env create -f environment.yml && conda activate rotomai
bash scripts/setup_server.sh          # vendored Showdown at a pinned rev, builds dist/
bash scripts/run_server.sh &          # ws://localhost:8000

python -m rotomai.cli evaluate --p1 heuristic --p2 random --n 20 --format gen9ou
```

The rule-based baseline against `random` on a local server, printing a win rate. **No weights, no
GPU, no network** — team pools for the teambuilt formats are committed and defaulted per format.
`rotomai --help` lists the rest; [docs/OPERATIONS.md](docs/OPERATIONS.md) covers training,
evaluation, the cluster and live play, and [Develop](#develop) is the test / lint / type bar.

To play the *trained* agents, fetch the release weights — see
[Using the released weights](#using-the-released-weights).

---

## Results

| goal | status | headline |
|---|---|---|
| **G1** live play | **run on the public ranked ladder** | 100 pre-registered rated gen9ou games: Elo **1004** / GXE **30%** — pre-registered **WEAK** |
| **G2** strong-human on one format | **met, both halves** | gen9ou **0.7303** vs poke-env's `SimpleHeuristicsPlayer` (n=9000), **0.7513** vs the in-repo heuristic; agent-only ladder **Glicko 1776.3 / GXE 0.7434** |
| **G3** continual improvement | **booked** | five-dose self-play curve, monotone on both reads (80 → 320 updates) |
| **G4** ≥3 formats through one core | **met** | `gen9randombattle`, `gen9ou`, `gen9vgc2025regi` end to end |
| **G5** skill stack as testable capability | **met** | four capabilities, each on its own gate's criterion |

**The strength number, against a baseline that is not ours.** `simpleheuristics` here *is* poke-env's
`SimpleHeuristicsPlayer`, imported unmodified — the rule-based baseline the field inherits — so this
row is the one an outside reader can situate:

| arm | opponent | task | n | score rate (Wilson 95%) |
|---|---|---|---|---|
| v26b terminal, greedy, LoopGuard 4 | poke-env `SimpleHeuristicsPlayer` | **gen9ou**, 12-team committed pool | **9000** | **0.7303** [0.7211, 0.7394] |
| v26b terminal, same protocol | in-repo `heuristic` — *control* | gen9ou, same pool | 9000 | 0.7504 [0.7414, 0.7593] |
| v26b terminal | in-repo `heuristic` — *published* | gen9ou, same pool | 5400 | 0.7513 [0.7396, 0.7626] |

Every seed's best checkpoint is scored and pooled, so these are selection-free terminal reads. Each
row is **three independent runs** in separate Slurm allocations across two nodes, pooled — because
this project has measured identical checkpoints at 0.472 and 0.499 in two runs, and one read cannot
tell "this arm" from "this run" apart:

| run | simpleheuristics | heuristic |
|---|---|---|
| tron62, 01:00 / 01:46 | 0.7293 | 0.7500 |
| tron65, 02:14 | 0.7357 | 0.7563 |
| tron62, 02:27 | 0.7260 | 0.7450 |
| **pooled, n=9000** | **0.7303** | **0.7504** |

Two things settle it. The three runs span **0.0097** on the headline opponent and **0.0113** on the
control — both well inside a single run's own interval. And the pooled control re-reads the
*published* baseline at **0.7504** against 0.7513, **0.0009** apart. The opponent gap holds at
**0.0201**: poke-env's baseline is consistently the harder of the two, so the rebase moved the
headline down.

**The replication is itself a result.** Individual checkpoints swung up to **0.028** across runs —
same weights, same opponent, same settings — while the seed-pooled read moved **0.0097**. Pooling
over seeds is what buys run-to-run stability, and a single n=300 cell buys none: the `eval-ladder`
record already held a v26b-s0 vs `simpleheuristics` cell at **0.8167** (n=300), against 0.741, 0.769
and 0.761 for that same checkpoint at n=1000. Small-n cells on a twelve-team pool cluster by team
matchup and cannot carry a comparison.

Numbers commonly cited elsewhere against `SimpleHeuristicsPlayer` (~85%) are on **gen8randombattle**,
where the server supplies both teams. That is a different task from gen9ou with a committed team
pool, where battles cluster by team matchup. The rows sit side by side; they are not differenced.

**Two of the three formats measure as ceiling-bound, and that is a result rather than a shortfall.**
gen9-RB and VGC both return `FORMAT_BOUND` from the three-leg ceiling probe — nothing competent
beats the fixed heuristic there, near-optimal search included — so the project's own instruments say
not to spend more on their strength axes. gen9ou is where headroom was proven and where the strength
result lives.

**Not claimed:** the agent-only Glicko/GXE figure is measured against a **bot field** and is **not**
comparable to a Showdown GXE against humans; and only the self-play axis of G3 has a dose-response
curve, not the replay-data axis.

**Public ranked ladder: run, and the result is WEAK.** One bounded, pre-registered measurement run
on gen9ou — bands, n and the exact command frozen in
[`results/live_ladder_gen9ou_prereg.json`](results/live_ladder_gen9ou_prereg.json) **before the
first rated game**, including the statement that a weak result would be published too. It was.

| | |
|---|---|
| games | **100** (the pre-registered n, played once, no early stop, no top-up) |
| record | **35–65**, score rate **0.350** [0.264, 0.448] |
| **Elo / GXE** | **1004** / **30%** — pre-registered WEAK is Elo < 1200 or GXE < 45% |
| Glicko | 1339.5 ± 36.0 |
| clean-run check | `ladder_games_delta` **0** — the ladder charged exactly the games observed; 0 restarts |

**This is the most useful number here, precisely because it is bad.** The same checkpoint scores
0.7303 against poke-env's shared baseline and 0.350 against ~1055-Elo humans. Both are correct;
they measure different things. **A bot-baseline win rate does not predict human-ladder
performance** — and no amount of further self-play against a fixed heuristic could have shown that,
because every instrument this project had pointed at an agent field.

The diagnosis is specific rather than vague: the policy never saw the human metagame. It trained on
self-play plus a fixed heuristic, brings a committed 12-team pool against opponents who bring
anything, and decides from a 761-d **single-turn** observation. That points at the replay-data axis
of G3, which this repo already flagged as measured once and never scaled — so the next training
work is BC on human replays → offline RL, not more self-play.

Full experimental record, every build in the order it ran with its pre-registration and verdict:
**[docs/RESULTS.md](docs/RESULTS.md)**. The design document, requirements and milestone roadmap are
in [plan.md](plan.md).

---

## How it works

```mermaid
flowchart LR
  subgraph T["Training"]
    direction LR
    RP["Human replays<br/><code>data/replays.py</code>"]:::dim
    POV["POV reconstruction<br/>ingest v1 / resim v2"]:::dim
    BC["Behaviour cloning<br/><code>train/bc.py</code>"]
    ORL["Offline RL — AWR<br/><code>train/offline_rl.py</code>"]
    PPO["PPO self-play<br/><code>train/ppo.py</code>"]
    RP --> POV --> BC --> ORL --> PPO
  end
  SP["Self-play collection<br/><code>data/collect.py</code>"] --> ORL
  PPO --> CK[("checkpoints/*.pt")]
  CK --> SRCH["Depth-limited expectimax<br/><code>search/</code>"]
  CK --> EV["Gates + eval ladder<br/><code>eval/</code>, <code>scripts/*_gate.py</code>"]
  CK --> LV["Live client<br/><code>live/</code>"]
  classDef dim stroke-dasharray: 4 3;
```

A 761-d single-turn observation (`features/encoder.py`; 888-d for doubles) through a shared
actor-critic or entity-transformer trunk, deployed greedily. The dashed arm is honest rather than
decorative: the replay pipeline is built and tested, but the replay-data axis was measured once and
never scaled, so no claim rests on it. Self-play is the axis with a dose-response curve.

**Operational manual — environment, cluster, pipelines, live play:
[docs/OPERATIONS.md](docs/OPERATIONS.md).**

---

## Using the released weights

**The weights are on the release.** Three checkpoints, 23 MiB, one playable policy per teambuilt
format:
[v1.0.0](https://github.com/BhaveshThapar/RotomAI/releases/tag/v1.0.0).
**Restore the paths as you download** — the manifest keys on them, and the gen9ou asset is attached
under the bare name `iter_320.pt`, which does not say which arm it came from:

```bash
B=https://github.com/BhaveshThapar/RotomAI/releases/download/v1.0.0
mkdir -p checkpoints/ppo_v26b_s0
curl -sL -o checkpoints/ppo_v26b_s0/iter_320.pt  $B/iter_320.pt          # gen9ou, 17.4 MiB
curl -sL -o checkpoints/doubles_offrl_vgc_v2.pt  $B/doubles_offrl_vgc_v2.pt
curl -sL -o checkpoints/doubles_bc_vgc_v2.pt     $B/doubles_bc_vgc_v2.pt

python scripts/release_assets.py --verify        # want: 3/3 release assets present and verified

python -m rotomai.cli evaluate --p1 offrl --p1-checkpoint checkpoints/ppo_v26b_s0/iter_320.pt \
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
pytest            # 792 tests. With the env active (node ON PATH) + a local server up: 791 pass,
                  #   1 skip -- the opt-in live-client smoke, which ROTOMAI_LIVE_TEST=1 enables
                  #   for 792/0. `node` on PATH is load-bearing: without it six simulator tests
                  #   self-skip even with dist/ built; with dist/ but no server, 785 pass / 7 skip.
                  #   On a bare clone -- no server, no built dist/, no checkpoints/ -- 16 self-skip
                  #   and 776 pass. That is what CI runs; a 17th skip is a regression, not noise.
                  #   Run it as `pytest`, not `python -m pytest`: pyproject sets pythonpath so the
                  #   two agree, and CI runs the bare form.
ruff check .
mypy rotomai scripts   # both trees; CI runs the same two arguments
```

Reading the CI result without `gh` (there is none on the dev host, which is why a red CI once went
unnoticed for eleven commits): [docs/OPERATIONS.md](docs/OPERATIONS.md#reading-ci-without-gh).

---

## Layout

| Path | Role |
|---|---|
| `rotomai/config.py` | Local server config + format constants |
| `rotomai/engine/damage.py` | Expected-damage / move-value estimator |
| `rotomai/features/` | `embed_battle` 761-d singles encoder (`OBS_LAYOUT`) / 888-d doubles encoder + action-space codec |
| `rotomai/model/` | MLP actor-critic + entity transformer + build factory |
| `rotomai/agents/` | `heuristic`, `bc`, `offrl`, `ppo` agents |
| `rotomai/data/` | self-play collection, reward, replay fetch / ingest (v1) / resim (v2) |
| `rotomai/train/` | BC, offline RL, self-play, PPO training loops |
| `rotomai/search/` | R-PREDICT: forward model, determinization, expectimax, opponent model |
| `rotomai/teambuilding/` | R-TEAM: validator-checked packed team pools for teambuilt formats |
| `rotomai/eval/arena.py` | run N battles vs one fixed baseline, report win rates |
| `rotomai/eval/rating.py` | R-EVAL: Glicko-1 + GXE (degenerate against a fixed baseline — read the docstring) |
| `rotomai/eval/ladder.py` | agent-only eval ladder: round-robin + joint rating fit (the varied field G2 needs) |
| `rotomai/live/` | M5 deploy / G1: live-server client, policy gate, session supervisor, telemetry |
| `rotomai/cli.py` | all subcommands (eval / collect / train / data / live) |
| `scripts/` | local Showdown server + simulator setup/run, and every experiment gate |

---

## Artifacts & reproducibility

**A clone of this repo contains no weights and no training shards.** `checkpoints/` and `data/` are
gitignored, so every published number was produced from files that exist only on the machine that
ran them. What *is* committed, and is the durable record, is the evidence rather than the artifacts: 240
`results/*.json` gate summaries, each arm's per-iteration `curve.json`, the validator-checked packed
team pools (`rotomai/teambuilding/data/`), the encoder vocab and the gen9ou usage prior
(`rotomai/features/data/`), and the pinned simulator rev.

**Which record backs which headline.** These name result files rather than checkpoint paths, because
a gate can be re-pinned and a result file cannot:

| headline | record |
|---|---|
| gen9ou **0.7513** selection-free terminal | `results/ppo_ou_gate_v26b_terminal.json`, `results/seed_strength_gate_v26_terminal.json` |
| ranked ladder **Elo 1004 / GXE 30%** (n=100, WEAK) | `results/live_ladder_gen9ou.json`, `results/live_ladder_gen9ou_prereg.json`, `results/live_ladder_gen9ou_seg{0,1,2,3}.json` |
| gen9ou **0.7303** vs poke-env `SimpleHeuristicsPlayer` (n=9000) | `results/seed_strength_gate_v26b_simpleheuristics{,_paired}.json`, `results/seed_strength_gate_v26b_heuristic_{control,paired}.json` |
| Glicko **1776.3** / GXE **0.7434** | `results/eval_ladder_gen9ou_v26.json` |
| VGC B6f C1 **0.530** | `results/ppo_vgc_gate_b6f{,_s0,_s1,_s2}.json`, `results/seed_strength_gate_b6f_c1.json`, `results/awr_vgc_arm_b6f.json` |
| VGC corrected ladder **BC 0.453** / **AWR 0.467** | `results/format_ceiling_gate_vgc_v2.json` |
| **G5 MET** — four capabilities, each on its own gate's criterion | `results/g5_capability_gate.json` |

Every checkpoint those records name is present on the machine that produced them, and none of them
ship. `python scripts/check_artifacts.py` re-derives that statement rather than trusting this table.

**58 of the 122 checkpoint paths named across `results/**.json` no longer exist**, cited by 52 of the
240 result files. 27 are top-level warm starts (`bc_gen9ou_v*.pt`, `offrl_scale_*.pt`); the rest are
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

# Lategame

A competitive Pokémon Showdown ML battle agent for **Gen 9 Random Battles**, played on a
local Showdown server. See [plan.md](plan.md) for the full PRD and roadmap.

## Status

A complete experimental pipeline is built and verified — rule-based baseline → behavior
cloning → offline RL → self-play → entity transformer + on-policy PPO → human-replay
ingestion (public-log **and** full-fidelity re-simulation).

**Key finding: a learned method finally clears the heuristic plateau.** For eight levers
every approach stalled at ~27–34% win rate vs the heuristic (human-replay imitation did
worse, ~5–11%), and winners-only behavior cloning saturated at ~0.42 imitation accuracy
*regardless of encoder*. The breakthrough (lever 9) is **value-RL at data scale**:
advantage-weighted regression over all 82k re-simulated turns (winners **and** losers) with
the **EntityTransformer + dex-prior critic** reaches **~45–48% vs the heuristic** (3 seeds,
n=200). The plateau was never one constraint — it breaks only at the *conjunction* of
value-RL **×** a critic that can fit the value function (the transformer two-tower; value-MAE
0.28 vs the MLP's 2.59, which collapses to ~4%) **×** data scale.

**Three follow-ups all find the GREEN policy near a local ceiling (AMBER).** Lever 10 —
on-policy PPO warm-started from the GREEN checkpoint — is stable (no collapse) but does not
compound. Lever 11 (**R-PREDICT**) builds the infrastructure the project never had: a
*faithful Showdown forward model* (serialize/fork/step, validated bit-for-bit, **0/9,734**
transition mismatches) + *determinization* of the hidden opponent (live POV → full battle,
**~100%** observable-faithful). But depth-1 test-time search on the frozen GREEN policy doesn't
beat it (head-to-head ~0.36 at n=120). Lever 12 deepens it to **depth-2** (same forward model,
recursion + policy-prior pruning): the over-switch is gone (vs random **1.000**) and search now
reaches **parity** with its base (h2h **0.500**) but still does not exceed it (vs heuristic
−0.025; the minimax arm regresses to h2h 0.275 — the determinized opponent model is too weak for
worst-case search). Three independent mechanisms — gradient (L10), depth-1 (L11), depth-2 (L12) —
confirm the local ceiling; the search direction is retired and the next lever is the
tougher-opponent **curriculum** (change the signal, not the inference). See `plan.md` §13.1.

## Setup

```bash
# 1. Python env (Python 3.11, isolated) — environment.yml runs `pip install -e ".[dev]"`
conda env create -f environment.yml
conda activate lategame

# 2. Torch (needed for the learned agents / all training)
pip install -e ".[ml]"

# 3. Local Showdown server + vendored simulator (clones smogon/pokemon-showdown into
#    third_party/ and builds dist/ — dist/ is also used by replay re-simulation)
bash scripts/setup_server.sh
```

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

# M6 — human replays: fetch, then reconstruct each player's POV either from the public
# spectator log (v1) or by re-simulating the inputlog for the private |request| (v2)
python -m lategame.cli fetch-replays  --min-rating 1200 --limit 200
python -m lategame.cli ingest-replays --out data/ingest_gen9rb_rl.npz   # v1 (public-log POV)
python -m lategame.cli resim-replays  --out data/resim_gen9rb_rl.npz    # v2 (needs node + dist/)
```

## Develop

```bash
pytest            # 112 tests; server-gated smokes run when the local server is up
ruff check .
mypy lategame
```

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
| `lategame/eval/arena.py` | run N battles, report win rates |
| `lategame/cli.py` | all subcommands (eval / collect / train / data) |
| `scripts/` | local Showdown server + simulator setup/run |

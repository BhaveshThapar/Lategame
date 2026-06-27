# Lategame

A competitive Pokémon Showdown ML battle agent for **Gen 9 Random Battles**, played on a
local Showdown server. See [plan.md](plan.md) for the full PRD and roadmap.

## Status

A complete experimental pipeline is built and verified — rule-based baseline → behavior
cloning → offline RL → self-play → entity transformer + on-policy PPO → human-replay
ingestion (public-log **and** full-fidelity re-simulation).

**Key finding so far: no learned method yet beats the M1 heuristic.** Every RL approach
plateaus at ~27–34% win rate vs the heuristic; human-replay imitation does worse (~5–11%).
The evidence across six methods points at the hand-crafted observation encoder (no
species/move/item identity) as the remaining lever — not the algorithm, critic, or
trajectory data. The `heuristic` agent stays the strongest player and the fixed eval baseline.

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
`bc`, `offrl`, `ppo` (learned — load their default checkpoint).

### Data & training pipelines

```bash
# M2/M3 — collect self-play trajectories, train BC then offline RL
python -m lategame.cli collect-rl --n 50
python -m lategame.cli train-rl   --data data/gen9rb_rl.npz

# M4/M5 — self-play league / on-policy PPO improvement loops
python -m lategame.cli selfplay --init checkpoints/offrl_gen9randombattle.pt --iters 8
python -m lategame.cli ppo      --init checkpoints/offrl_gen9randombattle.pt --iters 8

# M6 — human replays: fetch, then reconstruct each player's POV either from the public
# spectator log (v1) or by re-simulating the inputlog for the private |request| (v2)
python -m lategame.cli fetch-replays  --min-rating 1200 --limit 200
python -m lategame.cli ingest-replays --out data/ingest_gen9rb_rl.npz   # v1 (public-log POV)
python -m lategame.cli resim-replays  --out data/resim_gen9rb_rl.npz    # v2 (needs node + dist/)
```

## Develop

```bash
pytest            # 80 tests; server-gated smokes run when the local server is up
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

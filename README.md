# Lategame

A competitive Pokémon Showdown ML battle agent. See [plan.md](plan.md) for the full
PRD and roadmap (milestones M0→M7).

**This build covers M0 (infra) + M1 (rule-based baseline):** a runnable agent that
plays full **Gen 9 Random Battles** end-to-end on a *local* Showdown server, plus a
damage/speed/type-aware heuristic that beats a random policy. The heuristic is the
baseline that later learned agents (BC → offline RL → self-play) are measured against.

## Setup

```bash
# 1. Python env (Python 3.11, isolated)
conda env create -f environment.yml
conda activate lategame
# (environment.yml already does `pip install -e ".[dev]"`)

# 2. Local Showdown server (clones smogon/pokemon-showdown into third_party/)
bash scripts/setup_server.sh
```

> `torch` is **not** installed yet — M0/M1 need no model. When you reach M2,
> `pip install -e ".[ml]"`.

## Run

```bash
# Terminal A: start the local server (ws://localhost:8000)
bash scripts/run_server.sh

# Terminal B: evaluate agents against each other
# M0 sanity: random vs random completes full battles
python -m lategame.cli evaluate --p1 random --p2 random --n 5

# M1: heuristic should clearly beat random (target >= ~65% win rate)
python -m lategame.cli evaluate --p1 heuristic --p2 random --n 100

# Sanity-check against poke-env's built-in baselines
python -m lategame.cli evaluate --p1 heuristic --p2 maxbasepower --n 100
python -m lategame.cli evaluate --p1 heuristic --p2 simpleheuristics --n 100
```

Available agent names: `random`, `maxbasepower`, `simpleheuristics`, `heuristic`.

## Develop

```bash
pytest            # unit tests (engine.damage) + arena smoke test
ruff check .
mypy lategame
```

## Layout

| Path | Role |
|---|---|
| `lategame/config.py` | Local server config + format constants |
| `lategame/engine/damage.py` | R-CALC seed: expected-damage / move-value estimator |
| `lategame/agents/heuristic_agent.py` | M1 rule-based agent |
| `lategame/eval/arena.py` | R-EVAL seed: run N battles, report win rates |
| `lategame/cli.py` | `evaluate` / `play` entrypoint |
| `scripts/` | Local Showdown server setup/run |

# PRD — Competitive Pokémon Showdown ML Battle Agent

**Working name:** Lategame (placeholder)
**Author:** Bhavesh
**Date:** June 26, 2026
**Version:** 0.1 (draft)
**Status:** In progress — see §13.1 for build status & findings (updated 2026-07-01).

---

## 1. Summary

Build a machine-learning agent that plays competitive Pokémon on the live Pokémon Showdown website at a strong-human level. The agent connects directly to Showdown's server over its websocket protocol (the same channel the browser uses), reads the text-based battle state, and returns move/switch decisions. It is **bootstrapped from a large dataset of high-skill human replays**, then **continually improved via self-play and offline-to-online reinforcement learning**, so its strength increases over time rather than being fixed at release.

The design target is **general use across formats** (Random Battles, Gen 9 OU, VGC doubles, older gens), achieved through a shared architecture and pipeline that is *instantiated and trained per format* — not a single universal model, which does not yet exist and is a research frontier.

This document is grounded in an explicit decomposition of what makes a strong competitive battler (Section 4); every capability requirement traces back to a real competitive skill.

---

## 2. Problem statement & motivation

Competitive Pokémon Singles/Doubles is a two-player, simultaneous-move, **imperfect-information** game with a very large state space, stochastic outcomes (damage rolls, accuracy, crits, secondary effects), and long planning horizons. It is a genuinely hard RL/decision-making problem and an excellent testbed for sequence models and offline RL.

Existing public work proves the core capability is achievable:
- **poke-env** — the standard Python interface to Showdown (protocol handling + RL/Gym API), format-agnostic via a format string.
- **Metamon** (UT Austin, RLC 2025) — offline RL + transformers trained on 5M+ reconstructed human trajectories; agents have reached #1 on the human ladder in Gen 1 OU and the 90–99th percentile elsewhere. The current state of the art for singles.
- **PokéChamp** (2025) — LLM + minimax search; ~1300–1500 projected Elo in Gen 9 OU.
- **VGC-Bench** (2025) — doubles/VGC baselines: BC, self-play, fictitious play, double oracle, and BC-initialized RL variants.
- **foul-play** — a ready-to-run rule-based bot connecting to the live server.

The opportunity is to combine the **offline-bootstrap-then-self-play** recipe (Metamon's strength) with optional **test-time search and opponent modeling** (PokéChamp's strength) into one system that runs on the live site and improves continually — and to make it work across multiple formats.

---

## 3. Goals & non-goals

### 3.1 Goals
- **G1.** Connect to the live Showdown server and play complete battles autonomously (challenge and unranked ladder play) within the turn timer.
- **G2.** Reach **strong-human performance** in the first target format (Gen 9 Random Battles), measured by matchmaking-bias-robust metrics (GXE / Glicko-1), competitive with the foul-play heuristic and Metamon baselines.
- **G3.** Demonstrate **continual improvement**: model strength measurably increases as self-play volume and replay data grow.
- **G4.** Generalize the *system* to ≥3 formats spanning singles random, singles teambuilt (OU), and doubles (VGC) by plugging in per-format data, action heads, and team sources — without rewriting the core.
- **G5.** Encode the competitive skill stack of Section 4 as explicit, testable capabilities (state estimation, prediction/opponent modeling, win-condition planning, precise damage math).

### 3.2 Non-goals (for v1)
- **NG1.** A single model that plays *all* formats well simultaneously. Out of scope; each format is trained separately.
- **NG2.** Superhuman / top-1-ladder dominance. Aspirational, not a v1 commitment.
- **NG3.** Automated farming of the public *ranked* ladder under a primary account (see Section 15 — policy-sensitive).
- **NG4.** Screen-scraping or GUI automation (clicking buttons / reading pixels). The protocol is open; we use it directly.
- **NG5.** A browser extension as the runtime. The model lives in Python/PyTorch; a websocket client is the correct deployment.
- **NG6.** Full automated teambuilding from scratch as a learned generative model (v1 uses curated team pools; learned teambuilding is a stretch goal).

---

## 4. Background — What makes a strong competitive battler (and the requirement it implies)

This is the design backbone. Strong human play decomposes into five capability layers. Each maps directly onto a system requirement.

### 4.1 Mechanical & numerical foundation (the math that is never guessed)
A competitive player computes, not estimates, the deterministic core of the game.

- **Stats.** Each Pokémon's real stat derives from its species **base stat**, **IVs** (0–31 per stat), **EVs** (≤252 per stat, ≤510 total, ~every 8 EVs = +1 stat at level 50), and **nature** (±10% to two stats, applied multiplicatively). Move order and damage are functions of these.
- **Speed tiers.** Turn order decides games more than any single factor. Players reason in *speed tiers* — who outspeeds whom after EVs/nature/items/boosts — and EV-invest specifically to win key speed benchmarks (e.g., one point above a base-100 threat).
- **Damage.** Damage follows the damage formula and lands in a **range (the "rolls," ~85–100%)**, modified by STAB (×1.5), type effectiveness (×0/¼/½/1/2/4), crits, weather, items, and abilities. Outcomes are expressed as **OHKO / 2HKO probabilities** (e.g., "68.8% to OHKO").
- **Stat stages.** In-battle boosts/drops operate on a 13-stage multiplier ladder (e.g., +1 Atk ≈ ×1.5), affecting both damage and speed.
- **Stochasticity.** Accuracy, crit chance, secondary-effect chance, and damage rolls make the game probabilistic; correct play maximizes **expected** outcome, not best-case.

> **Requirement → R-CALC:** The agent must include a precise damage/speed calculator and stat engine. Decisions must be evaluated in expectation over damage rolls, accuracy, and crits — never assuming best-case.

### 4.2 State estimation under imperfect information
You never see the opponent's full team, items, EV spreads, or movesets. Strong players continuously **acquire and track information** and fill gaps with **metagame priors**.

- Track everything revealed: Pokémon seen, moves used, items/abilities inferred, current HP, boosts, status, hazards, weather, screens, field effects, remaining team slots.
- Infer hidden sets from **usage statistics** (most-likely moves/items/spreads for a revealed species) and update as evidence arrives. "See one move or one Pokémon and you can practically fill in the blanks."

> **Requirement → R-STATE / R-PRIORS:** Maintain a structured, fully-observed-from-our-POV battle state plus a **belief distribution over the opponent's hidden information**, seeded by usage priors and Bayesian-updated on observations.

### 4.3 Prediction & opponent modeling (the central skill)
Because the opponent switches and plays to win, you cannot simply click the super-effective move. **Prediction — "intelligent guessing based on collected information"** — is the only tool that overcomes a poor matchup.

- **Risk vs. reward.** Most turns are decided by the value of each Pokémon and the risk/reward of the line, *not* by deep reads. Deep prediction is reserved for genuine 50/50s or when you are losing unless you take a risk.
- **Over-prediction is a classic failure mode.** Reading too many levels deep loses games; the agent must calibrate prediction depth to the situation and opponent skill.
- **Early vs. late game.** Early game = scouting and obvious plays; late game = precise reads once the board is known.
- **Exploitation & anti-exploitation.** Strong players exploit predictable tendencies and *avoid being predictable themselves* (mixing lines, occasionally feeding false information).

> **Requirement → R-PREDICT:** The decision engine must evaluate actions against a **model of the opponent's likely response** (learned from human replays and self-play), choosing the action with the best expected outcome — and must regulate prediction depth and avoid exploitable determinism.

### 4.4 Win-condition reasoning & long-horizon planning
Pokémon is frequently compared to chess: you must plan around short- and long-term threats while executing your own gameplan.

- **Team Preview threat assessment.** Before turn 1, identify your win path and the opponent's key threats; recognize which of your Pokémon counter what.
- **Win conditions.** Identify the Pokémon/line that wins the game, **preserve it**, and remove its checks; be willing to **sacrifice** lesser Pokémon to position it.
- **Tempo / momentum.** Controlling tempo is repeatedly cited as the most important macro skill — forcing favorable switches, keeping initiative, denying the opponent setup time.
- **Hazards & chip.** Entry hazards (Stealth Rock, Spikes, Toxic Spikes) inflict permanent damage on switch-ins and are central to wearing down offense over time; **hazard control** (setting vs. removing via Defog/Rapid Spin) is a strategic axis. **Pivoting** (U-turn / Volt Switch / Flip Turn) maintains momentum by bringing in favorable matchups safely.
- **Direct vs. indirect damage.** Offensive Pokémon (no recovery) are worn down by chip/hazards; defensive Pokémon are removed by direct pressure. Choosing the right pressure type matters.

> **Requirement → R-PLAN:** Decisions must be made over a multi-turn horizon with an implicit value of game state (not greedy per-turn), capturing win-condition preservation, hazard/tempo dynamics, and sacrifice logic. This is naturally expressed by an RL value function and/or test-time search.

### 4.5 Team archetypes & roles (teambuilt formats)
For non-random formats, *what you bring* is half the skill. Teams cohere around **archetypes** — hyper offense, bulky offense, balance, stall — built from **roles**: sweepers, wallbreakers, setup sweepers, walls/pivots, and utility/support, organized into synergistic **cores**, with EV spreads tuned to **survive specific threats and outspeed specific benchmarks** via a damage calculator.

> **Requirement → R-TEAM:** For teambuilt formats, the agent must be supplied with format-legal, meta-relevant teams (v1: curated pools; stretch: learned/generated teams), and its play must respect each Pokémon's intended role.

### Skill-stack → requirement summary

| Competitive layer | Core idea | System requirement |
|---|---|---|
| Numerical foundation | Stats, speed tiers, damage rolls, KO odds computed exactly | R-CALC: damage/speed engine; expected-value decisions |
| State estimation | Track revealed info; infer hidden sets from usage priors | R-STATE / R-PRIORS: full POV state + Bayesian belief |
| Prediction | Intelligent guessing vs. opponent model; calibrated depth | R-PREDICT: opponent-response model; anti-exploitation |
| Win-condition planning | Preserve win con, control tempo, hazards, sacrifice | R-PLAN: long-horizon value / search |
| Archetypes & roles | Team cohesion, role-appropriate play (teambuilt formats) | R-TEAM: team provisioning + role-aware policy |

---

## 5. Users & personas

- **P1 — Operator/Developer (primary):** runs, trains, and iterates the agent; needs clean control (start/stop, format select), logging, and reproducible training.
- **P2 — The Agent (autonomous actor):** connects under a dedicated bot account, queues battles, and plays within timer constraints with high reliability.
- **P3 — Spectator/Reviewer:** watches the agent's games (by spectating its account in a browser) and reviews replays/telemetry to debug behavior.

---

## 6. Scope & phased rollout

"General use, any format" is delivered **incrementally**, hardest-last. Build the entire pipeline end-to-end on the simplest format first; only then generalize.

| Phase | Format | Why first/next | Added complexity |
|---|---|---|---|
| 1 (MVP) | Gen 9 Random Battles | No teambuilding; large dataset; simplest action space | Baseline everything |
| 2 | Gen 9 OU (singles) | Most popular competitive singles | Teambuilding / team pools; richer metagame |
| 3 | VGC (Gen 9 Doubles) | Official format; high value | **Doubles action space** (2 slots + targeting), much larger |
| 4+ | Older gens / other tiers | Breadth; Metamon data exists for early gens | Per-format data + tuning |

**MVP definition (Phase 1):** an agent that logs into the live server, plays Gen 9 Random Battles end-to-end within the timer, and beats the rule-based baseline, using a model trained via BC + self-play.

---

## 7. Functional requirements

### 7.1 Connectivity & client (R-NET)
- Connect to the live server websocket (`wss://sim3.psim.us/showdown/websocket`) and to a local server for training.
- Authenticate with configurable account credentials; support **challenge**, **accept-challenge**, and **ladder** modes; configurable battle format string and concurrency.
- Parse the Showdown protocol into structured state; emit `/choose` commands. (Use poke-env to wrap this.)
- Robust reconnect/resume; handle timer, forced switches, team preview, mid-battle disconnects, and illegal-move rejection gracefully.

### 7.2 State representation & feature encoder (R-STATE, R-ENCODE)
- A **format-agnostic** encoder producing a tensor representation of the full battle from our POV: both active Pokémon, both benches (revealed + unrevealed slots), moves/items/abilities (embedded), HP, boosts, status, hazards (per side), weather, terrain, screens, and turn history.
- Standard backbone: embed each Pokémon's moves/items/abilities → aggregation token + **Transformer encoder over the 12 Pokémon** (6 ego + 6 opponent) → **temporal Transformer** over stacked frames with causal masking for history.
- **Action head is format-conditioned:** singles = {move ×4, switch ×5}; doubles = per-slot {move × targets} with interdependent-action constraints (e.g., two Pokémon cannot switch into the same slot). **Invalid actions masked to −∞.**

### 7.3 Knowledge / priors module (R-PRIORS, R-CALC)
- **Usage-stat priors:** for each revealed species, a prior over likely moves/items/abilities/spreads, updated as moves are observed.
- **Damage & speed engine:** exact calculator (damage rolls, type chart, STAB, items, abilities, weather, boosts) exposed to the decision engine for expected-value/KO-probability reasoning. Integrate or port an existing damage engine (e.g., the Rust `poke-engine` used by foul-play).

### 7.4 Decision engine (R-PREDICT, R-PLAN)
- **Core policy** π(state) → action distribution, trained per format (Section 11).
- **Opponent model** for prediction: an estimate of the opponent's response distribution (from human replays / self-play population), used to evaluate expected outcomes.
- **Optional test-time search:** depth-limited minimax / one-ply expectimax over predicted opponent responses, using the policy as prior and the value head (and/or damage engine) as leaf evaluation. Toggle for "Extended Timer" formats where deliberation is allowed.
- **Prediction-depth regulation & anti-exploitation:** avoid deterministic, easily-countered lines; sample from the action distribution where appropriate.

### 7.5 Team provisioning (R-TEAM)
- For random formats: none (server supplies teams).
- For teambuilt formats: a **Teambuilder** (poke-env `yield_team`) drawing from curated, format-legal team pools (importable from Showdown paste / packed format). Stretch: usage-informed team generation; learned teambuilding.

### 7.6 Training pipeline (R-TRAIN)
- **Replay ingestion + POV reconstruction:** download Showdown replays per format; reconstruct each player's partially-observed POV from spectator logs; filter to high-skill demonstrations (ladder rating ≥ ~1200).
- **Self-play infrastructure:** run massive self-play on a local `--no-security` server (CPU-parallel battle generation).
- **RL training loop:** offline RL + online fine-tuning (Section 11), checkpointing, and a population of past/diverse opponents.

### 7.7 Evaluation & telemetry (R-EVAL)
- Compute **GXE**, **Glicko-1**, and win rate vs. fixed baselines; log per-battle decisions, predicted vs. actual opponent actions, and KO-probability calibration.
- Evaluate primarily on a **private/agent-only server or eval ladder** to avoid matchmaking bias and human-ladder disruption.

### 7.8 Control surface (R-CTRL)
- CLI (and optional thin local GUI) for start/stop, format/account selection, mode (challenge/ladder/self-play), and live logs. The GUI is a wrapper over the same engine — optional polish, not core.

---

## 8. Non-functional requirements

- **Latency:** decide within Showdown's per-turn timer with comfortable margin; search depth must be bounded to stay in budget on the deployment machine (CPU inference is sufficient for these model sizes).
- **Reliability:** auto-reconnect; never hang a battle; recover from protocol edge cases; idempotent move submission.
- **Reproducibility:** seeded training; versioned datasets, configs, and checkpoints; deterministic eval harness.
- **Compute budget:** training models are small (≤~200M params) and run on a single consumer GPU or university cluster; self-play (CPU-bound) is the heavier resource. Inference/deployment is cheap. No per-battle API cost (unlike LLM-agent approaches).
- **Account isolation & safety:** run only under a dedicated bot account, never a primary account.

---

## 9. System architecture

```
                ┌─────────────────────────────────────────────┐
                │            Showdown Server                    │
                │  (live: sim3.psim.us  |  local: --no-security)│
                └───────────────▲───────────────┬───────────────┘
                  websocket      │ protocol      │ state
                  /choose        │               ▼
        ┌──────────────────────────────────────────────────────┐
        │                  Agent Runtime (Python)               │
        │  ┌────────────┐   ┌──────────────┐   ┌──────────────┐ │
        │  │  Client    │──▶│ State Encoder │──▶│  Decision    │ │
        │  │ (poke-env) │   │ (transformer) │   │  Engine      │ │
        │  └────────────┘   └──────────────┘   │  π + value   │ │
        │        ▲                              │ + opp. model │ │
        │        │          ┌──────────────┐    │ + opt. search│ │
        │        └──────────│ Priors /     │◀──▶│              │ │
        │                   │ Damage engine │   └──────────────┘ │
        │                   └──────────────┘                     │
        │                   ┌──────────────┐                     │
        │                   │ Teambuilder  │ (teambuilt formats) │
        │                   └──────────────┘                     │
        └──────────────────────────────────────────────────────┘
                                  ▲
                                  │ checkpoints
        ┌──────────────────────────────────────────────────────┐
        │              Offline Training Pipeline                 │
        │  Replays → POV reconstruction → filter(≥1200) → BC →   │
        │  Offline RL → Self-play (local) → Offline→Online RL    │
        │  Eval harness (GXE / Glicko / vs baselines)           │
        └──────────────────────────────────────────────────────┘
```

**Train where it's fast (local server, GPU/cluster); deploy where it matters (live site).** The same model artifact serves both.

---

## 10. Data plan

- **Sources:** public Showdown replays (per format); reuse Metamon's released dataset + reconstruction pipeline for early gens; gather and grow new replays for Gen 9 OU / VGC.
- **POV reconstruction:** convert spectator-POV replays into per-player partially-observed trajectories (essential — raw replays leak hidden info).
- **Quality filtering:** restrict BC demonstrations to ladder rating ≥ ~1200.
- **Priors:** maintain usage-statistics tables per format (move/item/ability/spread frequencies) for the priors module and (optionally) team generation.
- **Team sets:** curated, format-legal team pools (Showdown paste / packed format) for teambuilt formats.

---

## 11. ML approach & training

Three-stage recipe (the proven path), with optional test-time search layered on top.

1. **Offline bootstrap — reward-filtered Behavior Cloning.** Train the transformer policy to imitate high-skill human actions (rating ≥ ~1200). Establishes competent baseline play.
2. **Offline RL.** Continue with offline RL (actor-critic with value classification, à la Metamon) over the human dataset to push past pure imitation and learn a value of game state — the substrate for win-condition/tempo reasoning (R-PLAN).
3. **Self-play improvement.** Initialize from the offline policy and improve via self-play on the local server. **Use population methods (fictitious play / double oracle) over a diverse opponent pool** to prevent the classic self-play collapse into exploitable strategies. This is where strength compounds.

**"Learns from each battle" — the honest mechanism.** Online learning is real but *secondary*: the agent improves through (a) periodic **offline-to-online fine-tuning** against fresh self-play experience, and (b) periodically **folding new human replays back into the dataset and retraining**. The *live ladder is for evaluation*, not the primary gradient source — single-game online RL on a human-paced ladder is far too sample-inefficient to drive learning.

**Prediction & opponent modeling (R-PREDICT).** The value function + an explicit opponent-response model (learned from replays / the self-play population) let the engine choose actions by expected outcome against likely opponent play. Optional **depth-limited search** (PokéChamp-style) sharpens this where the timer allows.

**Reward shaping.** Sparse win/loss is the true objective; shape with intermediate signals (e.g., + for opponent HP/faints, − for own HP/faints, hazard/chip and momentum proxies) to densify learning, taking care not to distort the win objective.

**Architecture recap.** Per-Pokémon move/item/ability embeddings → aggregation token + Transformer over 12 Pokémon → temporal Transformer (causal) → format-conditioned, masked action head; separate value head.

---

## 12. Evaluation & success metrics

Raw ladder win rate is misleading (matchmaking pushes everyone toward ~50%). Use bias-robust metrics and fixed baselines.

| Metric | Measures | Phase-1 target |
|---|---|---|
| **GXE** (Glicko X-Act Estimate) | Expected win % vs. a random opponent | Beat rule-based baseline; approach strong-human band |
| **Glicko-1** | Skill rating with uncertainty | Steady climb across training |
| **Win rate vs. baselines** | Head-to-head vs. foul-play heuristic & Metamon baselines | > 50% vs. heuristic |
| **Ladder percentile** (eval) | Standing vs. humans (anonymized, non-ranked) | Top ~30% as a milestone |
| **Decision quality** | KO-probability calibration; predicted-vs-actual opponent action accuracy | Tracked over time |
| **Continual-improvement curve** | Metric vs. self-play/data volume | Monotonic improvement (G3) |

Evaluate on a **private/agent-only server or eval ladder** wherever possible.

---

## 13. Milestones / roadmap

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| **M0 — Infra** | Local server + poke-env; live connection on bot account | Plays a full game with a random policy end-to-end |
| **M1 — Rule-based baseline** | Heuristic agent (damage/speed engine + simple logic) | Beats random; serves as eval baseline |
| **M2 — BC** | Reward-filtered BC policy (Gen 9 Random Battles) | Beats heuristic baseline |
| **M3 — Offline RL** | Value-augmented offline RL model | Improves on BC in head-to-head |
| **M4 — Self-play** | Self-play loop + population methods | Continual-improvement curve demonstrated (G3) |
| **M5 — Live deploy + eval** | Live-server play within timer; GXE/Glicko telemetry | Strong-human band in Random Battles (G2) |
| **M6 — Multi-format** | OU (team pools) + VGC (doubles head) instantiated | ≥3 formats playable end-to-end (G4) |
| **M7 — Search (optional)** | Test-time depth-limited search toggle | Measurable win-rate lift on Extended-Timer formats |

### 13.1 Build status & findings (as of 2026-07-02)

> The build milestones below track the *actual* implementation sequence and differ from the
> roadmap table above (which numbers M5+ as deploy/multi-format). All work so far targets
> **Gen 9 Random Battles** on the local server; live deploy and multi-format are not started.

- **M0 infra + M1 heuristic — done & verified.** Heuristic: 96% vs random, 84% vs
  maxbasepower, 51% vs simpleheuristics. It is the fixed baseline every learned agent must beat.
- **M2 BC → M3 offline RL (value-classification critic + AWR) → M4 self-play league →
  M5 entity transformer + on-policy PPO — all built and verified; none beat the baseline.**
  Learned policies plateau at **~27–34% vs heuristic** regardless of algorithm/architecture.
  M5 fixed the critic ~2.3× (offline value MAE 1.35→0.58) **without** moving win-rate, and PPO
  with a strong GAE baseline + exploration outside the demonstrator distribution was flat →
  the objective/algorithm is not the bottleneck.
- **M6 human-replay ingestion — built, two POV reconstructions, decisive negative.** v1 (public
  spectator logs) regressed the policy and was attributed to a train/inference observation
  distribution shift (public logs carry no `|request|`, so the agent's own bench is hidden at
  train time but full at live inference). **v2** re-simulates each replay's `inputlog` (PRNG seed
  + choices) through the vendored simulator to recover the private `|request|` per player →
  full 6-mon, in-distribution POV + exact labels (**0% decision drop** vs v1's 9.4%). **Fixing
  the shift did NOT break the plateau** (controlled, n=100 vs heuristic: v2 offrl 11% / bc 6% vs
  v1 offrl 8% / bc 5% — within noise). The shift was real but **not the binding constraint**.
- **R-ENCODE (learned species/move/item/ability ID embeddings) — built, and its cheap BC gate
  is a decisive negative.** The v2 encoder appends identity ID channels and the EntityTransformer
  learns an embedding per id; a controlled BC ablation (embeddings ON vs OFF, 3 seeds, same
  faithful v2 shard) shows identity **hurts** imitation accuracy — **ON 26.9% vs OFF 36.0%**
  (best-by-val-loss val_acc), with the ON arm overfitting instantly (early-stop epoch 2–3, train
  acc barely 62%). At 4196 winners-only samples over a ~1600-species vocab the embeddings are
  un-learnable (most rows see <5 examples). Kill criterion triggered → did not proceed to the
  offrl/PPO retrain. (Aside: transformer-numerics OFF ~36% edges the historical MLP ~32% — a
  small architecture lift, not identity.)
- **Conclusion across seven methods (BC, offline AWR, self-play, transformer+critic, PPO,
  human-replay v1+v2, identity embeddings): the binding constraint is DATA QUANTITY**, not the
  algorithm / objective / critic / architecture / encoder. Identity is not a cheap win at this
  data scale; pursuing it would require a much larger replay scrape OR pre-initialized embeddings
  (usage stats / species2vec), not learned-from-scratch on a few thousand samples. This sharpens
  Data Plan §10 and ML Approach §11 toward **data scale** as the next lever.
- **R-ENCODE data-scaling sweep — built + run; data-starvation CONFIRMED, but lands just short
  (AMBER).** Tested the "embeddings are starved, not useless" hypothesis directly. Scaled the
  scrape 150→**1424** replays (≥1200; the public index for recent gen9randombattle was exhausted
  at page 100, so this is the achievable ceiling), re-simulated at **0% drop** → **40,496
  winners-only BC samples** (9.65×) + 82,751 RL turns (`data/resim_v3_gen9rb_*.npz`). Ran a BC
  imitation-accuracy sweep (EntityTransformer, id_embed ON vs OFF, 3 seeds) over nested subsets
  N ∈ {1k, 2k, 4196, 8k, 16k, 32k, 40496} via a new `--max-samples` train flag +
  `scripts/embed_scaling_sweep.py`. **Gap (OFF−ON) val-acc:** +0.04 (1k) → −0.02 (4196) →
  **+0.088 (8k) → +0.125 (16k)** → **+0.025 (32k) → +0.017 (40k)**. Reading: (i) absolute
  imitation rises with data for both arms (OFF 0.28→0.42, ON 0.24→0.40), (ii) the R-ENCODE
  "identity hurts" negative is a **small-data artifact** — the gap is worst in the overfitting
  regime (8k–16k) and then **collapses** as data grows, (iii) a phase transition between 16k→32k
  where ON jumps **+0.118** (0.274→0.392) vs OFF's +0.017 — embeddings learn ~7× faster once fed,
  the signature of emergence with scale. **But** at the data ceiling ON (0.401±0.010) still trails
  OFF (0.418±0.004) by a small, seed-resolved margin (non-overlapping bands) — ON did not reach
  parity. **Verdict: AMBER** — starvation was real and is resolving with scale, yet the achievable
  replay ceiling (~40k samples) lands just short of identity paying off. Per the decision gate, did
  **not** greenlight an offrl/PPO retrain on this half-signal; and since the scrape is now maxed,
  the indicated next lever is **pre-initialized embeddings** (Smogon usage stats / species2vec) —
  the precise remedy for "identity is real but data-starved," and BC-gate testable before any
  expensive retrain. Full grid in `results/embed_scaling_sweep.json`.

- **R-ENCODE identity-prior gate — built + run; priors reach parity, do not clear OFF (AMBER).**
  Acted on the indicated next lever from the scaling sweep. Built dex-feature **pre-initialized
  embeddings** (`lategame/features/embed_prior.py` + committed `data/id_priors_gen9.npz`, stamped
  with `vocab_version`): species warm-started from base stats + types, moves from
  power/accuracy/type/category/priority/pp — all from local poke-env `GenData` (no network; the
  named "Smogon usage / species2vec" alternatives were dropped — species2vec is meaningless on
  *random* battle teams, and dex features are the cheapest prior that captures "what this entity
  is"). Items/abilities lack structured features → stay random. Threaded a new
  `id_embed_init={random,prior}` through `EntityTransformer`→factory→`TrainConfig`→CLI (warm-start
  applied in `_reset_parameters`, overwritten by `load_state_dict`, so backward compatible). Ran a
  **3-arm** BC gate (OFF / random-init / prior-init, 3 seeds, 20 epochs) over N ∈ {8k,16k,32k,40496}
  via `scripts/embed_prior_sweep.py`. **Val-acc:** 8k OFF .363 / rand .276 / prior .282; 16k .400 /
  .274 / **.378**; 32k .417 / .392 / .407; 40496 OFF **.418±.004** / rand **.401±.010** / prior
  **.420±.005**. Reading: (i) priors **decisively beat random-init** at every rung and
  **accelerate emergence** — the headline is 16k, where prior is already .378 while random is still
  stuck at .274 (**+0.104**); (ii) priors **close the random→OFF deficit** — random trailed OFF by
  −1.7pp at the ceiling, prior reaches **+0.001 (parity)**; (iii) **but parity is not a win** —
  prior (.420±.005) vs OFF (.418±.004) bands overlap heavily, and all three arms converge to ~0.42
  while OFF itself is flat 32k→40k. **Verdict: AMBER** — priors make identity embeddings "free"
  (worth keeping as the default ON config: faster, no downside) but are **not the unlock**; the
  ~0.42 winners-only imitation-accuracy ceiling holds regardless of identity encoding. Per the gate
  (GREEN required ON-prior to clear OFF with non-overlapping bands), did **not** greenlight an
  offrl/PPO retrain. **8th lever to land at the same wall** → the binding constraint is no longer
  plausibly the encoder/embedding at all; it points at the imitation *signal itself* (winners-only
  BC saturates ~0.42), i.e. the next lever is the learning target/data composition (losers' turns,
  value-based RL beyond pure imitation), not the representation. Full grid in
  `results/embed_prior_sweep.json`.

- **Lever 9 — value-RL at data scale (AWR on all turns) — built + run; CLEARS THE WALL (GREEN).**
  Acted on the indicated next lever: the *learning target / data composition*, not the encoder. Key
  realization — this lever was already built and the data already existed but had **never been run
  at scale**: `data/resim_v3_gen9rb_rl.npz` holds **82,751 turns (winners AND losers)** with
  shaped Monte-Carlo returns (losers carry a `−victory_value` terminal), and `train/offline_rl.py`
  already does advantage-weighted regression over all turns; yet every offrl checkpoint on disk
  predated the 9.65×-scaled shard (AWR had only ever seen the small ~8k-turn shards). Pre-flight
  confirmed the signal is informative: winner vs loser start-return gap **+5.48**. Built a win-rate
  gate (`scripts/offrl_scale_gate.py`, gated on win-rate not val-acc — val-acc is the wrong metric
  for AWR, which deliberately deviates from imitation), threaded `id_embed`/`id_embed_init` through
  `OfflineRLConfig.arch()`+`train-rl` CLI (offrl previously couldn't pass the dex-prior embeddings —
  it silently used random init), and fixed a latent warm-start bug (the BC→AC path only matched
  `model_type=="bc"` but `train_bc` now stamps `"bc_policy"` → `KeyError`). **2 arms × 3 seeds, 30
  epochs on the 82k shard, eval n=200 vs heuristic:**
  - **MLP actor-critic (warm-started): 4.2% ± 0.6% — COLLAPSE.** Its critic cannot fit a value
    function over 82k diverse states (**value-MAE 2.59** over a ±10 support ≈ uninformative), so the
    AWR advantage weights are noise and the actor degrades *below* even the old plateau.
  - **EntityTransformer + dex-prior: 47.7% ± 2.5% (min 45%, seed-0 51%) — CLEARS 27–34%.** Its
    decoupled two-tower critic fits the value function (**value-MAE 0.28, ~9× sharper**), so
    advantage-weighting is meaningful and the actor exceeds winners-only imitation. Confirmatory
    ladder (best ckpt): vs random **98.5%** (matches the heuristic's own dominance), maxbasepower
    82.5%, simpleheuristics 42.5%, **heuristic 41.7% (n=300)** — a monotonic, sensible placement, not
    a harness artifact (the MLP arm fails in the *same* harness).
  - **The binding insight: the plateau was never one constraint.** Each ingredient failed in
    isolation across levers 1–8 — winners-only BC caps ~0.42 *regardless of encoder*; AWR on the MLP
    collapses because its critic underfits; data scale alone (MLP arm) does nothing. The wall breaks
    only at the **conjunction**: value-RL (advantage-weighting over losers' turns too) **×** a critic
    with the capacity to fit the value function (transformer two-tower) **×** data scale (82k). The
    AMBER dex-priors are part of the winning config — "free" turned out to matter here.
  **Verdict: GREEN** — first method in the whole arc to beat the heuristic baseline band. Per the
  gate, this greenlights a **self-play / PPO continuation** warm-started from the best
  `checkpoints/offrl_scale_et_prior_s0.pt`. Full grid in `results/offrl_scale_gate.json`.

- **Lever 10 — on-policy PPO continuation from the GREEN checkpoint — built + run; STABLE BUT
  DOES NOT COMPOUND (AMBER).** Acted on the Lever 9 greenlight. The on-policy PPO loop
  (`train/ppo.py`) needed **no code change** — it builds straight from the checkpoint meta
  (`build_model(ckpt)` + full-state-dict `load_actor_critic_weights`), so the EntityTransformer
  + dex-prior arch loads as-is; added a server-free ET warm-start + `ppo_update` regression test
  (`tests/test_ppo.py`) and a gate runner (`scripts/ppo_continue_gate.py`). Thesis: every prior
  method was either off-policy AWR (re-weights actions *already in the data* → caps at the
  demonstrator ceiling) or PPO on a *broken* MLP critic (M5 cratered 0.34→0.20); for the first
  time we have both a >47% start **and** a working low-variance GAE baseline (transformer critic,
  value-MAE 0.28), so on-policy PPO can push probability toward actions outside the demonstrator
  distribution with a critic that can score them. `heuristic` held **out** of training (anchors =
  simpleheuristics + the self-play league); win-rate reported vs the held-out heuristic. **3 seeds
  × 10 iters, games_per_opp 16, eval n=200/iter:**
  - **No collapse — the real positive vs M5.** vs_heuristic stays 0.43–0.56 across all
    iters/seeds (vs_random 0.98–1.0); the carried critic's value-MAE settles ~1.0 (disturbed from
    0.28 by the on-policy return distribution but stable — nothing like the MLP's 2.59 or the
    smoke run's transient blow-up). The decoupled two-tower critic is what keeps PPO stable from a
    strong warm-start where the M5 MLP cratered.
  - **No durable improvement — fails GREEN.** best-iter mean vs_heuristic **0.530±0.023** looks
    up, but it is the *max over 10 noisy n=200 evals* (per-iter σ≈0.035 ⇒ max-of-10 upward bias
    ≈ +0.05 ≈ the observed +0.065–0.090 per-seed deltas). Two **unbiased** checks say it's noise:
    the n=300 confirmatory ladder on the apparent-best checkpoint (seed-2 iter-9, 0.56@n=200)
    regresses to **heuristic 0.517** (≈ the ~0.45–0.48 start), and **final vs_iter0 = 0.442±0.024
    < 0.50** — the end-state policy marginally *loses* the head-to-head to the start it descended
    from. The per-iter curve wanders around the start with no trend. Confirmatory ladder (n=300):
    random 0.993, maxbasepower 0.843, simpleheuristics 0.410, **heuristic 0.517**.
  - **Insight:** with a strong warm-start + working critic, vanilla on-policy PPO against this
    opponent set is *stable but flat* — the GREEN offline policy is near a local ceiling for gen9
    random-battle vs the heuristic; on-policy gradient on the same opponents gives the policy
    nothing new to climb toward.
  **Verdict: AMBER** — did **not** trigger the RED conservative-retry (no collapse) and did
  **not** clear GREEN (no durable gain), so per the gate it was recorded without re-tuning. The
  next lever is no longer more on-policy gradient: it is the **R-PREDICT** direction
  (depth-limited search / opponent-modeling on the strong base policy + working value function) or
  a tougher opponent curriculum. Full grid in `results/ppo_continue_gate.json`.
- **Lever 11 — R-PREDICT (depth-1 search on the frozen GREEN base) — built + run; AMBER.**
  Acting on the Lever 10 verdict: layer test-time search on the frozen GREEN checkpoint (no
  retrain). The binding obstacle is that there is **no forward model** — the net gives V(s) and
  π(a|s) but not Q(s,a), and the vendored simulator was wired only for replay re-simulation. The
  forward model is built on the simulator's `State.serializeBattle`/`deserializeBattle` (fork) +
  `battle.choose` (step); hidden opponent info will be **determinized** from the gen9
  random-battle pool. Gated cheap-first.
  - **Gate A — forward-model fidelity (the cheap KILL gate) — PASS.** Before building search,
    prove the fork/step primitive is faithful: `search/fidelity_driver.js` replays real replays
    turn-by-turn and, at each decision round, forks the battle, steps the fork with the *same*
    choices, and checks the result matches stepping the battle directly. The serialized PRNG seed
    makes a faithful fork an **exact** match (same damage rolls / crits / accuracy). Over **300
    replays = 9,734 transitions: 0 mismatches** (core *and* the stricter full digest incl.
    boosts/pp/item/ability), 0 drive errors, match rate **1.0000** (≥0.99 ⇒ build search). A
    negative control confirms the check has teeth (a fork stepped with a *swapped* choice does
    **not** match). `lategame/search/{fidelity_driver.js,fidelity.py}`,
    `scripts/rpredict_fidelity_gate.py`, `results/rpredict_fidelity.json`, `tests/test_fidelity.py`
    (server-free, env-gated like `test_resim`).
  - **Forward model + reconstruction mini-gate — built, PASS.** Persistent
    `search/forward_driver.js` (reconstruct a full battle from a determinization spec; fork+step
    a serialized state, emitting clean fog-of-war per-side delta + new request) + `search/forward.py`
    wrapper (one long-lived node process per agent). `search/determinize.py` transcribes a live
    poke-env POV into the spec: our full team (known) + revealed opponent mons + field/hazards,
    with hidden opponent slots filled from the gen9 RB pool via the simulator's own team generator
    (`Teams.getGenerator().randomSet`). The mini-gate (`search/recon_check.py`,
    `scripts/rpredict_recon_gate.py`) re-simulates replays, snapshots each decision turn,
    determinizes it, and compares the reconstruction's *observable* digest to poke-env's. Over
    **40 replays = 2,437 snapshots / 145,400 field-checks: every dynamic field matches 100%**
    (hp, status, fainted, active, boosts, hazards, weather, terrain); `present` 99.95% (the only
    gap is Minior-Orange, a cosmetic forme the dex normalizes to base `minior` — a digest-naming
    artifact, not a reconstruction defect). Two reconstruction fixes found by the gate: reset
    active-mon boosts to observed (clear Intimidate-at-`>start`) and clear weather/terrain a
    determinized lead's ability set at start. Leaf-eval path validated piecewise: deepcopy the live
    poke-env root + feed the step delta (resim's `_feed_line`) → `embed_battle`. `results/rpredict_recon.json`.
  - **Gate B — does depth-1 search help? — AMBER (does not compound).** Built depth-1
    expectimax (`search/expectimax.py`) + `SearchAgent` (registered `search` in the arena),
    test-time only on the FROZEN GREEN checkpoint. The same checkpoint powers both the `search`
    agent and the greedy `offrl` base, so any gap is the search, not the weights. Two findings:
    - **Pure value-head leaves over-switch (a real property of V, not a bug).** Diagnosed
      directly: at one ply every leaf scores ~4.2 and V rates "my strong mon active" *above*
      "opponent at 21% HP" (it prefers switching to Salamence over a move that chips the foe to
      21%). The GREEN value head is a coarse *outcome* estimator, near-flat at one ply, so
      greedy-over-V switches almost every turn (offline 12/13 switches) → ~0.19 vs random, 0.0
      vs base at n=16. A **shaped tactical leaf** (`leaf = V + c·state_value`, c=3) adds the
      immediate chip/HP signal V lacks and restores sane play (1/13 switches; 1.000 vs random).
    - **Even with the tactical leaf, search does NOT beat the base.** Confirmatory n=120, two
      opponent-aggregation arms: **mean** — search vs base (h2h) **0.383**, search vs heuristic
      0.425 vs base 0.458 (−0.033); **min** (worst-case) — h2h **0.342**, 0.408 vs 0.442, and
      0.833 vs random (over-conservative). Search is *slightly net-negative* vs the base both
      ways. **Crucially this is NOT GIGO** — the forward model is validated (Gate A exact, recon
      ~100%) and the agent plays sanely; the cause is the value head being too flat at one ply,
      not broken machinery. `results/rpredict_gate_b_{mean,min}.json` (the script's raw
      threshold auto-labels "RED" at h2h<0.45; the *lever* verdict is AMBER — no collapse, the
      policy stays in the GREEN band ~0.41–0.43 vs heuristic, search just fails to add).
    **Verdict: AMBER** — structurally identical to Lever 10 ("stable, does not compound";
    L10's final-vs-iter0 was likewise 0.442<0.5). Two *independent* mechanisms — on-policy
    gradient (L10) and test-time depth-1 search (L11) — now both fail to compound, strongly
    reaffirming that the GREEN policy is near a **local ceiling** for gen9-RB vs the heuristic.
    Gate B negative ⇒ the n=300×4×3 Gate C ladder was **not** run, per cheap-gate-first. **Lasting
    asset:** a *faithful Showdown forward model + determinization* for gen9-RB (infra the project
    never had), reusable for any future lookahead. **Next-lever candidates:** depth-2+ search (V
    may discriminate deeper — compute-heavy, uncertain); a learned one-ply *afterstate/Q* leaf
    trained to be tactically discriminative; an explicit learned opponent model (uniform/min are
    weak); or the tougher-opponent **curriculum** (the Lever-10 fallback). Did NOT re-tune to
    chase GREEN.

- **Lever 12 — R-PREDICT depth-2 search on the frozen GREEN base — built + run; AMBER (does not
  compound).** Acted on the Lever 11 verdict's lead candidate: L11 condemned *depth-1*, the
  weakest lookahead (provably near-equivalent to the policy when the leaf is the value the policy
  trained against). Depth-2 is where the shaped term (HP/faints) should bite — it can see 2-ply
  sequences (switch → they KO my switch-in → revenge range) the 1-ply reactive policy cannot
  represent. The build was the cheapest possible: **reuses ~100% of the L11 forward model +
  determinization**; only the recursion is new. Generalized `search/expectimax.py` depth-1 into a
  depth-limited `_node_value` (refactored `_leaf_value` → `_node_battle` + `_leaf_eval`; added
  policy-prior pruning `_top_k_by_prior`), added one driver field (`step` now returns
  `p1_choices`/`p2_choices` via the existing `legalChoices`, so recursion needs no request
  re-parse), threaded `depth`/`top_k_my`/`opp_cap_deep` through `SearchConfig`→agent env→gate CLI.
  Root expands all our actions; deeper plies prune to the top-k by policy prior and cap opponent
  branching (≈ `opp_cap`×`top_k_my`×`opp_cap_deep` = 4×3×3 steps/decision), all serialized through
  the one node subprocess per agent — so n=40 is ~75 min/arm. **Gate B2 (cheap), 2 arms, n=40:**
  - **mean / expectimax — AMBER.** search vs random **1.000** (sanity; the depth-1 over-switch is
    *gone* — depth-2 plays cleanly), **h2h vs base 0.500** (exact parity), search vs heuristic
    **0.550** vs base **0.575** (**−0.025**, within noise). Depth-2 reaches **parity** with its
    base — a clear improvement over depth-1's slightly-net-negative h2h 0.383 — but does **not**
    exceed it. PROMISING needs h2h>0.52 **and** delta>+0.03; neither met.
  - **min / minimax — RED.** sanity **0.925**, **h2h vs base 0.275** (search *loses* to its base),
    vs heuristic **0.450** = base 0.450 (+0.000). Worst-case aggregation over the **weak uniform-
    fill determinized opponent** is over-conservative (assumes the foe always plays the single
    most-damaging line) — the same failure as L11's min arm (0.342), amplified one ply deeper.
  - **Insight:** with one more ply of *ground-truth* dynamics + a tactical leaf, search ≈ the base
    policy (expectimax) or worse (minimax). The limiter is not lookahead depth: the V/shaped leaf
    encodes nothing the policy lacks, and the determinized opponent model is too weak for adversarial
    search. This is the **third** independent mechanism — gradient (L10), depth-1 search (L11),
    depth-2 search (L12) — to fail to compound on GREEN ⇒ a very strong **local-ceiling**
    confirmation for gen9-RB vs the heuristic. `results/rpredict_gate_b2_{mean,min}.json`.
  **Verdict: AMBER** (primary expectimax arm parity; minimax RED). Per cheap-gate-first, the
  n=300×4×3 Gate C ladder was **not** run and GREEN was **not** re-tuned. **The search direction is
  now retired on this base** — both the depth axis (L11→L12) and the aggregation axis (mean/min)
  are exhausted, and the residual weakness is the *opponent model*, not the search. **Next lever =
  the tougher-opponent curriculum (Lever 13):** change the learning *signal*, not the inference
  machinery — consistent with the whole arc (every method that moved the needle was about data/
  signal, not cleverness on fixed data). The forward model + determinization remain a reusable
  asset for any future model-based opponent modeling.

- **Lever 13 — tougher-opponent AWR self-play curriculum from the GREEN base — built + run;
  AMBER (does not compound).** Acted on the Lever 12 verdict's lead: the search direction was
  retired, so change the learning *signal*. The only mechanism that ever cleared the wall was L9
  (AWR over all turns, winners + losers, at scale); GREEN still **loses** to two fast bots
  stronger than itself (`simpleheuristics` ~42.5%, `heuristic` ~41.7%). Thesis: re-run the
  wall-clearer on a tougher, *self-generated* opponent distribution, powered up. The M4 AWR
  self-play loop (`train/selfplay.py`) was the right machine but had **never been run from GREEN**
  (it defaulted to the old weak `offrl_gen9randombattle.pt`) and was underpowered — 4 epochs, and
  the real bottleneck, collection/eval running **one battle at a time**. The build was minimal:
  thread `max_concurrent` through `SelfPlayConfig`→`collect_selfplay`/`_build_recording_player`/
  `_eval_point` (mirroring `ppo.py`; the AC→AC warm-start already auto-carries the ET+dex-prior
  arch from the init ckpt, so no model change), expose `--max-concurrent`, and add
  `scripts/curriculum_gate.py`. `heuristic` held **OUT** of training (anchor = `simpleheuristics`
  + the fictitious-play league); win-rate reported vs the held-out heuristic — comparable to L9/L10.
  Gated cheap-first.
  - **Gate A — cheap KILL pre-flight (no training) — PASS.** Collect a small GREEN-vs-tough shard
    (40 games/opp vs `simpleheuristics` + the iter-0 league member, both sides), confirm the AWR
    signal before paying for the run: winner/loser **start-return gap 6.30** (≥1.0; L9 saw +5.48)
    and **loser fraction 0.50** (GREEN and the tough opponents split games evenly, so the data is
    full of tough winning lines to weight toward). 160 episodes / 4,392 turns.
  - **Gate B — the powered-up run (3 seeds × 12 iters, eval n=200) — flat.** vs the held-out
    heuristic the curve **wanders around the ~0.462 start with no trend**: best-iter mean
    **0.472±0.013** (max single point 0.49 — max-over-13-noisy-evals inflation, +0.010 over start),
    and the unbiased compounding check **final vs_iter0 = 0.443±0.006 < 0.50** — the end policy
    marginally *loses* the head-to-head to its own start (≈ L10's 0.442). The sharpest diagnostic:
    win-rate vs **`simpleheuristics` — the opponent it trained against — also did not rise**
    (wanders 0.35–0.45, mostly at/below the ~0.44–0.47 start). The critic stayed healthy throughout
    (value-MAE ~0.18–0.26, no blow-up), so this is *stable but flat*, not a training failure.
  - **Gate C — not run.** Per cheap-gate-first the gate auto-skips the n=300 confirmatory ladder
    when Gate B is AMBER-flat (mirrors L11/L12). GREEN was **not** re-tuned.
  **Verdict: AMBER** (`results/curriculum_gate.json`). This is the **fourth** independent
  mechanism — gradient (L10), depth-1 search (L11), depth-2 search (L12), tougher-opponent
  curriculum (L13) — to fail to compound on GREEN. The local-ceiling conclusion is now very
  strongly confirmed, and **the limiter is neither the inference machinery nor the opponent
  distribution**: even fresh data dominated by stronger opponents, fed through the only mechanism
  that ever worked, does not move the needle vs the heuristic on gen9-RB. **Implication:** the
  remaining headroom is unlikely to live in the *training loop* on this format/data — candidate
  directions shift to the *substrate*: a stronger/larger encoder+model trained on far more data, a
  genuinely stronger expert to distill from (e.g. search-as-teacher via the reusable L11/L12
  forward model, accepting its cost), or a different/harder **format** where the heuristic ceiling
  is higher. Lasting assets from L13: a non-underpowered, concurrency-parallel self-play loop and a
  reusable AWR-signal pre-flight gate.

- **Lever 14 — R-PREDICT with a real opponent model — built + run; AMBER (does not compound).**
  Acted on the axis the search direction was retired on but never tested. L11/L12's own verdict
  named the residual — *"the opponent model was too weak"* (uniform-mean / worst-case-min over the
  determinized foe) — then L13 pivoted to curriculum and skipped it. Yet the eval opponent is a
  **fixed, white-box, deterministic** rule (`HeuristicAgent`), so model it *exactly* and re-run the
  depth-2 gate with **probability-weighted expectimax** (`agg = Σ_oc p(oc)·v(oc)`). Reuses the
  entire validated L11/L12 forward model; only the opponent branch changes. Build: refactored the
  eval rule into a pure `agents.heuristic_agent.heuristic_pick` (DRY); new `search/opponent_model.py`
  with `WhiteBoxHeuristicOpponent` (one-hot on the exact heuristic move, computed from our-POV +
  the driver's determinized `p2_choices` — **no** opponent POV battle, byte-faithful to the eval
  opponent) and `LearnedOpponent` (the frozen GREEN policy on a reconstructed opponent POV, built
  from new `p2_log`/`p2_request` emission in the driver's `reconstruct` + `build_opp_pov`); wove
  `opp_aggregation="model"` through root + deep plies (`mean`/`min` untouched, L11/L12 reproducible);
  new `scripts/rpredict_oppmodel_gate.py` + 16 tests (134 total). ruff + mypy clean.
  - **Gate A — opponent-model fidelity (no battles) — PASS (decisive).** Over 239 determinized
    nodes from re-simulated replays: opponent-POV team fidelity **1.000** (the learned arm's
    prerequisite), white-box decodable **1.000**, and white-box agrees with the real
    `HeuristicAgent` on the reconstructed opponent POV **0.958**. The opponent model faithfully
    reproduces the exact eval opponent — the L11/L12 residual is genuinely *closed*, not GIGO.
    `results/rpredict_oppmodel_gate_a.json`.
  - **Gate B — depth-2 `model`-aggregation search (server) — AMBER.** *White-box* (the decisive
    **upper bound** — a near-perfect model of the eval opponent): n=40 gave a **false-positive
    spike** (search-vs-heuristic 0.600 vs base 0.375, delta **+0.225**) driven entirely by base's
    unlucky low draw; the **n=120 confirmation regresses to PARITY** — base 0.483, search **0.500**,
    delta **+0.017** (< the +0.03 bar), search-vs-random 0.800 (sanity; the depth-1 over-switch is
    gone). The h2h-vs-base 0.317 is **confounded** — the white-box model assumes the heuristic, so
    in the h2h (opponent = the *base*, not the heuristic) it mispredicts; the *valid* metric is
    vs-heuristic, and it is parity. *Learned* (GREEN-as-opponent, generalizable, n=40): 0.550 vs
    base 0.425 (delta +0.125) — the **same** base-low/search-high small-n inflation as white-box
    n=40, and **bounded above** by white-box (a *less* accurate model of the heuristic can't beat
    the exact one), so parity by transitivity. `results/rpredict_oppmodel_gate_b.json`.
  - **Gate C — not run.** Gate B AMBER, and the white-box **upper bound is parity**, so there is no
    stronger-than-base teacher to distill (the search-as-teacher premise fails at its root).
  **Verdict: AMBER.** This is the **fifth** independent mechanism — gradient (L10), depth-1 (L11),
  depth-2 (L12), curriculum (L13), and now depth-2 with a *validated real opponent model* (L14) —
  to fail to compound on GREEN, and the sharpest: it **closes the exact axis the search direction
  was retired on**. The residual was *not* the opponent model. **The search/inference family is now
  exhausted on every axis** — depth (1→2), aggregation (mean/min/model), *and* opponent-model
  quality (uniform → worst-case → near-perfect white-box). The gen9-RB-vs-heuristic local ceiling
  lives in **neither the training loop nor the inference machinery**. **Next = Lever 15: the
  substrate/format pivot** (bigger encoder+model on far more data, or a genuinely harder format
  where the heuristic ceiling is higher) — now justified on evidence, not assumed. Lasting assets:
  a Gate-A-validated (1.000 team fidelity) opponent-POV reconstruction (driver `p2_log`/`p2_request`
  + `build_opp_pov`) and probability-weighted expectimax, reusable for any future model-based
  opponent modeling.

- **Lever 15 — format-ceiling diagnostic (is the wall the *format* or the *model*?) — built + run;
  verdict FORMAT_BOUND → OU pivot.** Levers 10–14 all showed only that *our* model + inference can't
  beat the heuristic on gen9-RB; **none directly measured the *achievable* ceiling.** Before paying
  for either expensive branch of the "substrate/format pivot" — (A) scale the model, (B) pivot to a
  harder format — measure the ceiling directly. Cheap, no-training, decisive; new
  `scripts/format_ceiling_gate.py` (+ 8 tests → 137 pass / 5 skip; ruff/mypy clean) reusing `eval.arena`,
  `data.resim`, `features.embed_prior`, and the L14 white-box result. `results/format_ceiling_gate.json`.
  - **M1 — bot-skill-gradient sweep vs `heuristic` (server, n=300, Wilson CIs).** How far can *play*
    move the needle? `heuristic`-mirror **0.513** (sanity ≈0.50 ✓), `simpleheuristics` (poke-env's
    strongest built-in) **0.523** [0.467, 0.579], `maxbasepower` **0.107**, `random` **0.007**, GREEN
    `offrl` **0.430** [0.375, 0.487]. The heuristic *crushes* the naive bots (99.3% vs random, 89.3%
    vs maxbp) but the **strongest competent bot is at statistical parity** (0.523, CI spans 0.50) and
    **GREEN — the product of 9 levers — loses to it (0.430)**.
  - **M2 — strongest-inference upper bound (reuse L14).** Depth-2 expectimax with a *near-perfect
    white-box opponent model* + faithful forward model = **0.500** vs heuristic. Near-optimal inference
    is at parity — the ceiling holds *from above*.
  - **M3 — team-RNG variance decomposition (node re-sim, no live server, n=500).** AUC of
    (team-strength difference → winner): effective level-adjusted stats (`mon.stats`) **0.495**
    [0.444, 0.546], level-blind base-stat z-sum **0.469** (0 OOV). **Team strength does *not* predict
    the winner** — gen9-RB is balanced by design (the RB generator equalizes gross power via level:
    Jirachi lvl 80, Breloom lvl 83…). So the wall is *not* gross team RNG; it is a genuinely balanced,
    competitive format where a good heuristic sits near the achievable skill ceiling.
  - **Decision rule (M1/M2 primary, M3 corroborating — a hand-crafted strength proxy can't *prove*
    play-dominance, so it must not gate the branch).** Best competent agent = max(0.523, 0.500, 0.430)
    = **0.523 ≤ 0.53** (band top), below the 0.58 headroom bar ⇒ **FORMAT_BOUND**.
  **Verdict: FORMAT_BOUND → OU pivot.** Nothing we can field — poke-env's best bot, near-optimal
  search, or 9 levers of learned RL — meaningfully beats this heuristic on gen9-RB; the achievable
  ceiling vs a competent heuristic is ~parity. **Honest nuance:** GREEN (0.430) sits *below* that
  ceiling (~0.52), so ~0.09 of *model* headroom exists — but closing it only reaches **parity**, never
  a *decisive* win, so **G2 (clearly beat the heuristic / strong-human) is unreachable in gen9-RB no
  matter the model** ⇒ scaling the model on this format is unjustified. This is the **first direct
  ceiling measurement** in the project (levers 1–14 measured *our methods'* failure; this measures
  what *any* agent can achieve) and it selects **format over substrate**: pivot to **Gen 9 OU** (PRD
  G4/M6) — teambuilt, higher skill ceiling, abundant high-ladder human data, and the encoder/action
  head are already singles-native (`config.OU_FORMAT`; `--format` threaded everywhere). The only new
  build is **R-TEAM** team provisioning (a poke-env `Teambuilder` over a curated packed-team pool),
  needed for live play/eval/self-play; replay *training* data reconstructs teams from logs. **Lasting
  asset:** a reusable, gate-validated format-ceiling probe (skill band + inference upper bound +
  team-RNG AUC) to re-run on OU and confirm the new ceiling is genuinely higher before committing the
  full pipeline. **Two lessons:** (1) the mirror-match sanity (0.513 ≈ 0.50) is a cheap harness check
  worth keeping; (2) raw BST is the wrong strength proxy for a level-balanced format — use realized
  `mon.stats`, and don't let a caveated proxy hard-gate a strategic decision.

- **OU pivot — Build 1: R-TEAM + OU ceiling re-run — built + run; GREEN (the OU ceiling is genuinely
  higher).** The cheap gate the L15 pivot demanded *before* committing the full OU pipeline. **R-TEAM
  built:** `lategame/teambuilding/pool.py` — `TeamPool` (a poke-env `Teambuilder` yielding a
  seeded-random team per battle from a curated, pre-validated packed pool); `scripts/build_ou_teampool.py`
  — a Gate-A legality preflight that packs Showdown-paste teams via poke-env and validates each against
  the bundled Showdown `validate-team gen9ou` (kill-gate if < 8 pass). 12/12 curated teams legal →
  `lategame/teambuilding/data/teams_gen9ou.packed`; the validator caught real illegalities while authoring (Zamazenta can't
  learn Roost; Gouging Fire and Baxcalibur are Uber-banned in this build). `team=` threaded through
  `eval.arena.build_player` (backward-compatible; RB leaves it `None`); `format_ceiling_gate.py`
  parameterized (`--format`/`--team-pool`/`--out`) — the teambuilt path runs an **M1-only smoke** to a
  separate `results/format_ceiling_gate_ou.json` and does **not** apply the RB FORMAT/MODEL thresholds
  (`assess_ou`). Suite 137→**145 pass / 5 skip** (+8 tests), ruff/mypy clean.
  - **M1 skill band on gen9ou (server, n=300, Wilson CIs).** `heuristic`-mirror **0.487** [0.431, 0.543]
    (sanity ≈0.50 ✓), `simpleheuristics` **0.633** [0.577, 0.686], `maxbasepower` **0.040**, `random`
    **0.023**, GREEN `offrl` (RB-trained, **OOD** on OU) **0.383** [0.330, 0.439]. Harness clean (mirror
    ~0.50; monotone gradient 0.023 < 0.040 < 0.633; 12/12 legal teams); band width **0.610 > RB 0.516**.
  - **Decisive read.** The *exact quantity* that forced FORMAT_BOUND on gen9-RB — the strongest competent
    bot (`simpleheuristics`) vs the heuristic — was **0.523 (parity, CI spans 0.50)** on RB but is
    **0.633 on OU with the whole CI above the 0.58 headroom bar**. So **OU is *not* capped at parity**:
    real, capturable headroom exists above the heuristic, confirming the "higher skill ceiling" premise
    of the pivot by direct measurement. **PASS ⇒ greenlight the OU data/training build.**
  - **Honest scope.** This is an M1-only smoke: it removes the FORMAT_BOUND objection but does **not**
    itself measure how high a *strong learned* agent can reach — **M2** (port the L14 white-box depth-2
    search to OU) and **M3** (needs scraped gen9ou replays) are deferred to the next gate. GREEN's 0.383
    is out-of-distribution transfer (RB-trained, never saw OU teams/mons), which motivates OU-specific
    training — not a format verdict. **Next:** scrape gen9ou replays → reconstruct teams from logs
    (`data.resim` is already format-agnostic) → train OU checkpoints, then re-run this probe + M2 on the
    trained agent. **Lesson:** the `assess_ou` mirror-sanity flag needs a real `n` — at n=30 the mirror
    read 0.633 (CI [0.455, 0.781]) and tripped a false "harness NOT clean"; at n=300 it settled to 0.487.

- **OU pivot — Build 2: OU human-replay ingestion → first OU checkpoint — built + run; AMBER (pipeline
  works, agent is non-functional; a deeper POV gap than Build 1 assumed is precisely diagnosed).**
  The full data→train→probe chain, gated on reconstruction fidelity. **Empirical premise correction:**
  gen9ou public replays carry **no `inputlog`** (keys: format/id/log/players/rating/… — confirmed by
  direct fetch), so the RB `resim` path (re-simulate from the random-battle PRNG *seed*, fill opponents
  from randbats) **cannot** be used — Build 1's "`data.resim` is format-agnostic" note was wrong. But OU
  logs open with `|poke|` **team-preview** lines naming all six species per side, so the v1 log-based
  `data.ingest` (seed-free) is the right reconstructor once it seeds those species.
  - **Built.** `ingest._register_preview` (own-side team-preview → `battle.team` via `get_pokemon`, so a
    later switch reconciles by species, nicknames included, no duplicate); `encoder._opponent_mons`
    (merge revealed + `teampreview_opponent_team`, revealed-first, so the opponent roster is identical at
    offline reconstruction and live play — no-op for RB); `scripts/ou_ingest_gate.py` (Gate-A fidelity
    KILL gate: species coverage, drop rate, reward-sign, + a strip-`|poke|` negative control with teeth);
    `--seed` threaded into the `train-rl` CLI (was hardwired to 0); `--offrl-checkpoint` override on
    `format_ceiling_gate.py`. Suite 145→**154 pass**, ruff/mypy clean.
  - **Data (Gate A PASS).** Scraped **2,760** rated gen9ou replays (≥1200; index exhausts ~page 100, ~2×
    the RB 1,424) → ingested **120,012** all-turns turns + 61,740 BC samples (0 skipped, drop 2.8%,
    >RB's 82k). Gate A on n=500: parse 1.000, **species coverage 1.000** (stripped-`|poke|` control
    0.685, **lift +0.315** = teeth), drop 0.024, winner−loser return gap +6.5. `results/ou_ingest_gate.json`.
  - **Trained** BC (val-acc 0.71) + 3 AWR seeds (EntityTransformer + dex-prior; value-MAE **~0.47–0.52**,
    healthy — the two-tower critic fits fine, not the M4 MLP crater).
  - **Gate B (M1, n=300, seed 0): RED for the agent.** offrl vs heuristic **0.007** ≈ random 0.013
    (harness fine: `simpleheuristics` 0.573, monotone gradient; mirror 0.437 is n=300 noise). offrl vs
    **random 0.495** (no signal), vs maxbasepower 0.150. `results/format_ceiling_gate_ou_trained.json`.
  - **Diagnosis (airtight, two obs mismatches; the project's train==eval lesson).** (1) *Opponent roster*
    — training injected all 6 opp species but live `opponent_team` is revealed-only (mean 2.95, 9% full);
    **fixed** by the `_opponent_mons` encoder merge (verified 6/6 at eval), but alone insufficient.
    (2) *Own-team detail* — the log reveals the player's **own** item/ability/moves only progressively,
    while the live `|request|` supplies the full team from turn 1: own-active **item known 0.18→0.82,
    ability 0.45→1.00, moves present 2.18→4.00** (train→eval). The policy trains on "my kit unknown" and
    plays on "my kit known" → OOD on the identity-embedding channels → random-quality despite 0.71
    imitation accuracy. Same log-vs-request POV gap that sank RB v1 and needed resim — which OU cannot
    use. Team preview closes *species*, not *detail*.
  - **Verdict AMBER; next lever = two-pass own-team completion.** Lasting assets: OU reconstruction, the
    fidelity gate, the encoder opp-roster fix, CLI seed. **Next:** pre-scan each replay for every own mon's
    full-game-revealed moves/item/ability and populate them at every timestep so training obs matches the
    live POV, then re-ingest + re-train + re-run Gate B (its own gated build). **Lesson:** Gate A measured
    *species* coverage but not *detail* coverage — a fidelity gate must check the channels the encoder
    actually feeds the model (item/ability/move IDs), not just presence.

- **OU pivot — Build 3: two-pass own-team completion — built + run; AMBER/negative (log-only completion
  cannot close the POV gap; confirmed no bug).**
  - **Built.** `ingest._prescan_kits` (Pass 1: reconstruct the full POV once, read each own mon's
    full-game-revealed moves/item/ability off `battle.team`; the item is recovered from the raw
    `|-item|`/`|-enditem|` lines because poke-env's `Pokemon.end_item` resets a consumed item to `None`) +
    `_complete_own_team` (backfill before every `embed_battle`: `_add_move` missing moves; fill the item only
    when it is still the `unknown_item` sentinel, **never** overwriting `None` so a consumed item stays `None`
    as the live POV shows post-consumption; fill the ability only when unknown). Threaded via
    `_reconstruct_pov(kits=…)` with a default-on `complete_own_team` toggle (`ingest_replays`/`ingest_and_save`
    /CLI `--no-complete-own-team`). Upgraded `ou_ingest_gate.py` to read the **encoder ID channels**
    (own-active item/ability at each Pokémon block's trailing channels, move-count at each move block's ID
    channel) WITH two-pass and WITH it OFF (v1 control): teeth = ON must beat OFF on item + moves; reports the
    absolute ceiling and the residual vs the live 1.0/1.0/4.0.
  - **Gate A PASS** (n=200): parse 1.0, species 1.0, drop 0.023, reward gap +6.68; item ON 0.297 / OFF 0.183
    (**+0.114**), moves ON 2.62 / OFF 2.02 (**+0.604**), ability ON 0.466 / OFF 0.450 (**+0.016, negligible**).
    **Ability is irreducible from public logs:** poke-env's `_update_from_pokedex` already auto-assigns
    single-option abilities, so the ~55% unknown are multi-ability species whose ability never triggered.
    Residual vs live stays large: item 0.70, ability 0.53, moves 1.38.
  - **Retrained & Gate B still dead.** Re-ingested 120,001 turns / 61,731 BC (same scale — two-pass changes
    channel *values*, not turn count); BC val-acc 0.651 (lower than v1's 0.71, as expected for a richer/more
    honest obs); 3 AWR seeds, value-MAE 0.54–0.62 (healthy). **Gate B (n=300):** offrl **0.020** vs heuristic
    (≈ Build-2 0.007); offrl **0.16 / 0.45** vs random across seeds. Harness valid (mirror 0.493, gradient
    random 0.013 < maxbp 0.090 < simpleheuristics 0.630).
  - **Regression investigation → no bug.** A controlled OFF-vs-ON eval across all three seeds: vs heuristic
    OFF ≈ ON (both ~0.02, dead); vs random OFF is tight 0.45–0.52 while ON is seed-dependent (s1 0.455, but
    s0/s2 0.13–0.16). Seed-dependence rules out a deterministic obs bug; a live-encoder probe (2,918
    decisions) confirmed the **eval obs is full** (own-active item 0.89 / ability 1.00 / moves 4.00) while
    two-pass training reaches only 0.30/0.47/2.62 — a real, large OOD gap. The apparent v3<v1 dip is a minor
    perturbation of an already-dead agent (amplified by the log codec labelling moves in reveal/set order
    while the live request uses declaration order), not the binding failure.
  - **Verdict AMBER/negative; next lever = usage-prior imputation.** A **partial** POV fix is functionally
    neutral (OFF ≈ ON) — it lands in a third distribution matching neither train-sparse nor eval-full. The fix
    must reach **eval-full**: fill each own mon's *unrevealed* item/ability/moves from the species' standard
    competitive set (Smogon usage / a sets DB), which for the own team approximates the truth the live request
    provides. Lasting assets: two-pass completion, the channel-measuring Gate A, the tests (a multi-ability
    Kingambit fixture with a knocked-off item exercises every backfill path incl. consumed-item recovery).
    **Lesson:** a partial fix on an OOD channel can be *worse* than none — verify the fix reaches the eval
    distribution, not merely closer to it. Suite **152 pass / 5 skip**; `results/ou_ingest_gate.json`,
    `results/format_ceiling_gate_ou_v3.json`.

---

## 14. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Sample inefficiency** of online RL | Slow/failed learning | Offline-first; self-play for volume; ladder only for eval |
| **Self-play strategy collapse / nonstationarity** | Exploitable, brittle agent | Fictitious play / double oracle; diverse opponent population |
| **Partial observability** mishandled | Bad decisions, info leaks | Rigorous POV reconstruction; explicit belief over hidden info |
| **Teambuilding for non-random formats** | Weak in OU/VGC despite good play | Curated meta team pools v1; learned teambuilding later |
| **Doubles action-space explosion** | Hard to train Phase 3 | Format-conditioned head; defer to after singles works |
| **Scope creep ("any format" at once)** | Project stalls | Strict format-by-format sequencing; Random Battles MVP first |
| **Ladder/ToS friction** | Account action; ethics | Bot account; unranked/challenge testing; private eval server |
| **Compute for self-play** | Cost/time | University cluster / free tiers; CPU-parallel battle gen |

---

## 15. Ethical & policy considerations

- **Account isolation:** operate only under a dedicated bot account, never a primary/identity account.
- **Ranked ladder is policy-sensitive.** Showdown's rules prohibit "gaming the system," with moderators exercising discretion; the research community discourages botting the human ranked ladder and runs separate agent-only servers to avoid disrupting humans. **Default to challenge/unranked play and a private eval server.** Treat automated ranked farming as out of scope (NG3).
- **Transparency:** prefer testing against opt-in opponents (challenges) and clearly-labeled environments.

---

## 16. Open questions

1. **First-format confirmation:** Gen 9 Random Battles as MVP — agreed? (Recommended.)
2. **Search vs. pure policy:** ship Phase 1 as policy-only and add search in M7, or invest in search earlier for prediction quality?
3. **Team generation:** how far to push beyond curated pools toward learned teambuilding for OU/VGC?
4. **Reuse vs. rebuild:** fork Metamon (dataset + baselines + reconstruction) as the foundation, or build the pipeline fresh for full control? (Forking is faster; rebuilding is more educational.)
5. **Eval environment:** stand up a private Showdown server for clean evaluation, or use anonymized non-ranked live play?
6. **Reward shaping specifics:** which intermediate signals densify learning without distorting the win objective?

---

## Appendix A — Key references

- **poke-env** — Python interface for Showdown bots (protocol + Gym/RL API): https://github.com/hsahovic/poke-env
- **Metamon** (UT Austin) — offline RL + transformers, human-level singles, open dataset/baselines: https://github.com/UT-Austin-RPL/metamon
- **PokéChamp** — LLM + minimax expert-level agent (Gen 9 OU): https://github.com/sethkarten/pokechamp
- **VGC-Bench** — doubles/VGC baselines (BC, SP, FP, DO, BC-init RL)
- **foul-play** — rule-based bot + `poke-engine` damage engine: https://github.com/pmariglia/foul-play
- **PokéAgent Challenge** — leaderboard, formats, FH-BT/Glicko/GXE eval: https://pokeagent.github.io
- **Smogon University** — competitive theory: getting started, prediction, battle conditions; **VGC Guide** — stats/EV/IV/speed-tier mechanics.

## Appendix B — Glossary (quick)

- **STAB:** Same-Type Attack Bonus (×1.5). **OHKO/2HKO:** knockout in 1/2 hits. **Hazards:** Stealth Rock/Spikes/Toxic Spikes (chip on switch-in). **Pivoting:** U-turn/Volt Switch/Flip Turn (momentum). **GXE:** Glicko X-Act Estimate (win % vs. random). **Speed tier:** ordering of Pokémon by effective Speed. **Win condition:** the Pokémon/line that closes the game.
# PRD — Competitive Pokémon Showdown ML Battle Agent

**Working name:** Lategame (placeholder)
**Author:** Bhavesh
**Date:** June 26, 2026
**Version:** 0.1 (draft)
**Status:** In progress — see §13.1 for build status & findings (updated 2026-06-27).

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

### 13.1 Build status & findings (as of 2026-06-27)

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
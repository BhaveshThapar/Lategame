# PRD — Competitive Pokémon Showdown ML Battle Agent

**Working name:** Lategame (placeholder)
**Author:** Bhavesh
**Date:** June 26, 2026
**Version:** 0.1 (draft)
**Status:** In progress — see §13.1 for build status & findings (updated 2026-07-10).

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
- **G2.** Reach **strong-human performance** in the first target format, measured by matchmaking-bias-robust metrics (GXE / Glicko-1), competitive with the foul-play heuristic and Metamon baselines.
  - **AMENDED 2026-08-10: the target format is `gen9ou`, not Gen 9 Random Battles, and the change was forced by measurement rather than preference.** Lever 15 measured the *achievable* ceiling on gen9-RB directly (see §13.1): the strongest competent bot reaches **0.523** vs the heuristic with its CI spanning 0.50, near-optimal depth-2 search with a white-box opponent model reaches **0.500**, and team strength does not predict the winner (AUC 0.495) — RB is balanced by design, and a good heuristic already sits at the achievable skill ceiling. Verdict **FORMAT_BOUND**: G2 is unreachable in gen9-RB *no matter the model*, so pursuing it there was not a difficulty problem but an impossibility one. On `gen9ou` the same measurement shows real headroom — `simpleheuristics` **0.633** [0.577, 0.686], heuristic mirror **0.487** — and the band is wider (0.610 vs RB's 0.516).
  - **Status against the two halves of G2.** The *fixed-baseline* half is met: §12's "> 50% vs the heuristic" bar is cleared on `gen9ou`, with `v25b`'s selection-free terminal read at **0.6807** [0.6682, 0.6930], above `simpleheuristics`' 0.633.
  - **The *ladder* half is now COMPUTED (2026-08-10), on an agent-only eval ladder rather than the human ladder.** GXE and Glicko-1 need a varied opponent field, which no amount of further training can supply — but §12's own last line prefers "a private/agent-only server or eval ladder", and §16 Q5 is answered: there is *no* unranked public ladder, so that route is the only policy-clean one. `lategame/eval/ladder.py` fits a whole field jointly (Bradley-Terry on the Glicko scale, `heuristic` pinned at 1500). Over 8 agents × 28 pairs × 300 battles on `gen9ou`, **`v25b` reaches Glicko 1696.9 ± 14.9, GXE 0.6809**, above `simpleheuristics` (1579.2 / 0.5756) and the `heuristic` anchor (1500 / 0.5000) — see §13.1. Two caveats travel with it: the RD is a **lower bound** (battles cluster by team matchup, so the effective sample is under `n`), and this is an **agent-only** field, so the number is *not* comparable to a Showdown GXE measured against humans. The human-ladder reading remains out of scope under NG3; `lategame/live/` (**G1**) is built and gated for it should that ever change.
- **G3.** Demonstrate **continual improvement**: model strength measurably increases as self-play volume and replay data grow.
  - **MET (booked 2026-08-11), and the evidence was already on the record before it was claimed.** §12's continual-improvement row asks for "metric vs self-play/data volume → monotonic improvement". Builds 24–26 are exactly that curve and nobody had written it up as one. Five doses of PPO self-play at a pinned schedule (`anneal_iters` 80, `target_kl` 0.06, from the converged offline init), 3 seeds each, pooled **n = 9000/arm** against the fixed heuristic, chained through the shared `v25b` anchor because absolute rates move between runs (the anchor re-calibrated **+0.0157**, inside the ±0.027 cross-run band) and only within-run differences are trustworthy:

    | updates | 80 | 120 | 160 | 240 | 320 |
    |---|---|---|---|---|---|
    | seed-best read | 0.5607 | 0.6301 | 0.6691 | 0.7250 | 0.7420 |
    | terminal (selection-free) read | 0.5778 | 0.6189 | 0.6883 | 0.7354 | 0.7513 |

    **Monotone at every step on both reads**, which is the claim G3 makes. What it does *not* claim is that the improvement continues: the marginal return per update decays **1.62 → 0.91 → 0.64 → 0.18** (×10⁻³, bias-corrected; the raw chained figures are 1.73/0.98/0.70/0.21 and agree in shape), a ~9× decay that is itself monotone. G3 is met **and** the axis that met it is saturating — see Build 26.
  - **The curve comes from `seed_strength_gate.py`, NOT from the eval ladder, and that is a measured decision rather than a stylistic one.** The obvious improvement — plot the curve in the Glicko the goal is stated in — was tried and refused by the data. On the 2026-08-11 ladder the Glicko ordering is not even monotone in update count (120 → 1755.0 sits *above* 160 → 1710.7, against the gate's seed-robust move the other way), and the cluster bootstrap shows **no adjacent learned pair separates at 95%**. The ladder is a single-seed field: it separates bands, and cannot speak to build-vs-build at all. It is therefore not evidence against the gate, and not a substitute for it.
  - **Not demonstrated on the other half of G3's wording.** The goal says "self-play volume *and replay data* grow". Only the self-play axis has a dose-response curve; the replay-data axis was measured once (OU Builds 2–4) and never scaled, so no claim is made for it.
- **G4.** Generalize the *system* to ≥3 formats spanning singles random, singles teambuilt (OU), and doubles (VGC) by plugging in per-format data, action heads, and team sources — without rewriting the core.
  - **EXIT CRITERION MET (2026-08-12): three formats play end to end through one core.** Verified against a live local server, 6/6 battles completed on each:

    | format | agent | vs `random` | finished |
    |---|---|---|---|
    | `gen9randombattle` (singles random) | `heuristic` | 1.000 | 6/6 |
    | `gen9ou` (singles teambuilt) | `offrl` @ `v26b` | 1.000 | 6/6 |
    | `gen9vgc2025regi` (**doubles**) | `doubles` @ init | 0.667 | 6/6 |

  - **"Without rewriting the core" is the substantive half of the claim, and it held.** Doubles plugged in as a per-slot action codec (`features/doubles_action_space`, wrapping `DoublesEnv` exactly as the singles codec wraps `SinglesEnv`), a second encoder (`features/doubles_encoder`, importing every per-mon and per-move block builder from the singles one), and an agent. **The model factory needed no change at all** — `build_model` already took `input_dim`/`n_actions` from checkpoint metadata, so the doubles network is the same architecture at a different width. The one genuine edit to shared code was making `EntityTransformer` take its token layout as a parameter instead of importing the singles constant; it resolves the layout from `input_dim`, so every checkpoint written before doubles existed still builds the model it always did.
  - **The action space is FACTORED, and that is a choice with one known cost.** Doubles is **107 actions per slot** and a turn commits both, so a joint head would be 107² = **11,449** outputs. The head is instead two independent 107-way distributions (**214 logits**). What factoring cannot represent is a constraint that *couples* the slots, and exactly one matters: both slots switching to the same benched Pokemon. Showdown answers that order with a **default move rather than an error**, so it is a silently lost turn — handled by an explicit post-sampling resolution step, not by the mask.
  - **Singles is frozen and pinned by test**: `OBS_DIM` 761 / `OBS_VERSION` `v5-` / `GEN9_ACTION_SPACE_SIZE` 26 unchanged. Doubles is separately versioned (`d1-`, 888-d) on **both** fields, so `data.collect`'s fingerprint check rejects a cross-format shard on either one alone.
  - **What this does NOT claim.** The doubles checkpoint is randomly initialised — G4's exit is "playable end to end", and strength on VGC is a separate question. The 0.667 is 6 battles of a random-init policy and is not a result.
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

### 13.1 Build status & findings (as of 2026-07-10)

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

- **OU pivot — Build 4: usage-prior imputation — built + run; RED (train obs verified at eval-full
  density and the agent is still dead → the binding failure is NOT own-kit detail density; lever killed).**
  - **Built.** `lategame/data/usage_prior.py` (mirrors `embed_prior`'s build/write/load split): distills a
    monthly Smogon chaos-stats JSON (gen9ou-1500, 2026-06; fetched by `scripts/build_usage_prior.py`, raw
    cached under gitignored `replays/usage/`) into a committed per-species top-K artifact
    (`features/data/usage_gen9ou.json`, 181 KB, **402 species kept / 0 skipped / 0 out-of-vocab ids** —
    the vocab covers the whole metagame) stamped with `vocab_version` (drift guard), plus `sample_kit` =
    **usage-weighted sampling without replacement, stably seeded** per (replay-POV, mon) via blake2s (not
    top-1: the item slot is ~70% imputed, so a modal pick would make species→item near-constant in training
    — mode-collapse on exactly the channel under repair). `ingest._impute_kits` runs **once per POV between
    prescan and reconstruction**, fills kit-level only, and revealed truth always wins (moves pad never-used
    slots up to 4 so the labelled action can never be imputed; item only if still unknown, drawn `"nothing"`
    stays `None` = encoder-identical to live no-item; ability only if unknown; consumed items never touched).
    Default-on `impute_usage` toggle (`--no-impute-usage`), no-op for formats without an artifact; `sorted()`
    backfill + transform guard harden determinism. 10 new tests (5 sampler/artifact, 5 ingest).
  - **Gate A PASS — with an honest metric correction.** The 4th (imputation) arm first **KILLed** on the
    planned absolute item bar (0.760 < 0.85). A per-decision decomposition (new `_ItemStateProbe`, reading
    battle state at each labelled decision) proved the residual is **0.233 consumed-`None`** — Knock Off /
    Booster Energy / berry states the live POV also shows as `None`, ~2.4× more frequent in human ladder
    games than in eval-vs-heuristic — with only **0.0069 actually-unfilled sentinel**. The gate now kills on
    the *unfilled rate* (≤ 0.02, the real failure mode: a broken species lookup would push it to Build-3's
    ~0.70) and reports the known/consumed split + live reference. Final arm: item unfilled **0.0069**,
    ability **0.999** (≥ 0.99), moves **3.990** (≥ 3.95), missing-species **0.0083** (≤ 0.02). Prior-vs-truth
    advisory on the reveal-biased truth subset: item top-K 0.903, ability top-1 0.972, moves containment 0.938.
  - **Retrained & Gate B RED.** Re-ingested **120,001 turns / 61,729 BC** (identical turn count to v3 —
    imputation changes channel values, not counts; 32,650 mons imputed, 1.4% missing species). BC val-acc
    **0.636** (v3 0.651, expected with richer channels); 3 AWR seeds value-MAE **0.507–0.613** (healthy).
    **Gate B (n=300, harness clean: mirror 0.520, gradient random 0.020 < maxbp 0.080 < simpleheuristics
    0.660):** offrl **0.003** vs heuristic — and vs random **0.160 / 0.270 / 0.050** across seeds (all
    LOSE to random; same `bigerror` stall signature as v3).
  - **Verdict RED — the POV-density hypothesis chain is closed.** Three builds chased one hypothesis: the
    agent is dead because training obs under-fills the own-team identity channels vs the live request. Build 4
    **verified the train obs reaches eval-full density** (Gate A) and the agent got *worse*, completing a
    monotone pattern vs random: v1 sparse **0.495** → v3 two-pass **0.13–0.45** → v4 eval-full **0.05–0.27**.
    More own-kit detail at train time consistently makes the live agent worse, so detail *density* was never
    the binding failure. The strongest remaining candidate fits that monotonicity: **positional move-slot
    ORDER** — training obs/labels use reveal-order + sorted-backfill slots while the live `|request|` uses
    declaration order, the action space indexes moves *positionally* (slots 6–9), and every kit-completion
    step scrambles more slots relative to live (v1's sparse kits were mostly the moves actually used, so
    slot semantics were closest to self-consistent). Other candidates: imputed-vs-curated kit composition,
    and human-vs-bot state distribution drift. **Next lever = canonical move-slot ordering** applied
    identically at ingest and live encode (cheap, BC-gateable before any retrain) — its own gated build; OU
    ceiling re-probe + OU PPO stay gated OFF.
  - **Lessons.** (1) A fidelity gate must separate the *failure* residual from the *live-faithful* residual
    (unfilled sentinel vs consumed-`None`) or it kills on truth. (2) Obs-density parity ≠ obs parity: the
    ORDER/structure of positional channels is part of the distribution, and an action space indexed by
    position makes slot order a *label* semantics issue, not just an obs one. Suite **162 pass / 5 skip**;
    `results/ou_ingest_gate.json`, `results/format_ceiling_gate_ou_v4.json`, `results/gateb_v4_vs_random.json`.

- **OU pivot — Build 5: canonical move-slot ordering — built + run; RED (slot order fixed and *verified*,
  agent still dead; pure BC craters too → the stall is upstream of RL; lever killed).**
  - **Gate A (scramble measurement, before any code change) PASS.** New `scripts/slot_order_gate.py` probed
    `ingest._move_sample` down the exact v4 path over 200 replays / 7,340 move decisions: **25.9%** of
    training labels change slot under canonicalization (30.7% of decisions have a scrambled 4-list), and
    **91.7%** of the eval teampool's 72 mons declare moves non-canonically (mean displacement 0.97 slots) —
    the divergence was real at magnitude on both sides. Out-of-4 truncation drops: **zero** either direction
    (no >4-move sets in practice), so canonical `[:4]` is free.
  - **Built (unit 2).** `features/action_space.py` reimplements the codec locally (mirroring
    `SinglesEnv`'s structure incl. both lone-available-move fallbacks — a permutation shim can't handle the
    slot-6 fallback or >4-move subset selection): `canonical_moves(mon)` = known moves sorted by id, first
    four; used by `label_action` / `order_to_action` / `action_to_order` / `action_mask` /
    `synthesize_action_mask` AND `encoder.embed_battle`'s move blocks — train and live slot semantics agree
    **by construction**. **`OBS_VERSION` v2→v3**: the existing shard/checkpoint guards hard-reject every
    v2-era artifact (incl. the retired RB GREEN — reproducible at old commits) so orderings can never
    silently mix. Every codec consumer (ingest, resim, collect, agents, PPO, search) flows through this one
    module — zero changes elsewhere. 14 new tests: insertion-order invariance of labels and obs,
    request-backed round-trip on every masked action, disabled-move mask alignment at the canonical slot,
    lone-move fallback, obs-block↔action-slot vocab-id alignment, backfilled-move canonical slot.
  - **Gate B1 (BC, before AWR spend) PASS — and the hypothesis's cheap prediction held.** Re-ingest needed
    ZERO ingest changes: v5 = 119,996 turns / 61,723 BC (−0.005% vs v4, the predicted ~zero new drops);
    `ou_ingest_gate` regression PASS with channel metrics identical to the Build-4 record (order-only change
    verified). BC ET+prior val-acc **0.647 / 0.646 / 0.649** (mean 0.647) — above the pre-registered 0.63
    bar AND the v4 reveal-order baseline 0.636: canonical labels are consistent across replays, hence more
    learnable.
  - **Gate B2 RED.** 3 AWR seeds healthy (value-MAE 0.452/0.476/0.554), but vs-random **0.02 / 0.11 / 0.06**
    (mean 0.063 ≪ the 0.55 RED bar; at/below v4's 0.05–0.27) and vs-heuristic **0.010** at n=300 (v4 0.003).
    Harness clean (mirror 0.493; gradient random 0.030 < maxbp 0.060 < simpleheuristics 0.643). Recorded
    without re-tuning per the gate.
  - **Post-hoc localization (cheap, decisive): pure BC also loses to random** — bc_v5 s0 **0.06** / s1
    **0.03** (n=100). With 0.647 imitation accuracy, an agent that loses ≥94% to RANDOM is not "weak", it is
    systematically broken at eval — and identically so WITHOUT value-weighted RL. Four causes now eliminated
    by direct measurement: the harness (sanity clean), AWR (BC craters equally), own-kit obs *density*
    (Build 4 verified eval-full), move-slot *order* (this build fixed + verified it). The v1→v5 vs-random
    story re-reads as: v1's 0.495 was never "closest to working" — it likely played fallback-random-like;
    v3+ agents act *confidently* on the training distribution and that confident play loses live.
  - **Remaining candidates for the stall (next levers, cheapest first).** (1) **Behavioral probe** — log the
    orders + request/error stream of ~2 live games vs random (what does it DO? switch loops? one-move spam?
    server-rejected choices?); hours, zero training, and it discriminates the remaining hypotheses. (2)
    **Switch-slot/team-order semantics** — actions 0–5 and the six own-mon obs blocks index
    `battle.team.values()` insertion order: train = `|poke|` preview (upload) order, live = first-request
    order; believed equal (both upload order) but never probed — same gate pattern as this build. (3)
    **Human-replay → bot-eval distribution drift** (opponents, teams, game phases). OU ceiling re-probe +
    OU PPO stay gated OFF.
  - **Lessons.** (1) A verified fix that doesn't move the outcome metric is still progress — but only if the
    gate *measures the mechanism* (Gate A) so the negative eliminates the cause rather than the
    implementation. (2) When an agent loses to random, check the *simplest* policy (BC) before blaming the
    RL objective — one n=100 eval localized the failure upstream of AWR. Suite **181 pass / 5 skip**;
    `results/slot_order_gate.json` (a/b1/b2 blocks), `results/format_ceiling_gate_ou_v5.json`,
    `results/gateb_v5_vs_random.json`.

- **OU pivot — Build 6: live behavioral probe — built + run; CONCLUSIVE (cause isolated: the agents play
  legal, confident, pathological OVER-SWITCHING — absorbing two-mon switch loops; decode/mask and
  team-order causes eliminated by direct measurement).**
  - **Built (zero library changes).** `scripts/behavior_probe.py`: observe-only monkeypatch hooks in the
    `slot_order_gate` pattern — a `_FallbackSpy` rebinds `action_space.Player` so the codec's *silent*
    random fallback (the `action_to_order` except branch) becomes visible; the agent modules'
    `action_to_order` is wrapped to record every decision (turn, action, decoded order, `force_switch`,
    actives/HP, live `team.values()` order); `masked_logits` is wrapped *before agent construction* (the
    agents bind it in `__init__`) for legal-count / all-False / top-3 masked probs; probed players get
    `log_level=25` + a handler capturing poke-env's `[Invalid choice]`/`[Unavailable choice]` (logged at
    level 25 < WARNING — dropped by default, previously invisible); `teampreview` is wrapped to log the
    random lead order. Near-free candidate-(b) live half in the same games: `battle._teambuilder_team`
    (packed upload order) vs live `team.values()` per battle. Pre-registered decision tree with precedence
    a(decode/mask) > b(team order) > c(pathological-legal) > d(drift); random-mirror control arm;
    INCONCLUSIVE-only exit. 17 new pure-logic tests. Artifacts: `results/behavior_probe.json`, per-decision
    JSONL + 40 poke-env HTML replays (gitignored), **40 human-readable per-turn transcripts (committed —
    the primary evidence)**. Gotcha found: `cross_evaluate` calls `reset_battles()` on exit, wiping
    per-battle results — the probe uses `battle_against` directly.
  - **Findings (n=20/arm, bc_v5_s0 + offrl_v5_s0 vs random; control 0.45 ∈ [0.3,0.7] PASS; 1,807 + 1,946
    decisions):**
    - **(a) ELIMINATED.** Fallback rate **0.000** — not one of 3,753 decisions hit the codec's random
      fallback; **zero** server rejections; **zero** all-False masks; zero retry repeats. The live
      decode/mask path is clean — Build 5's codec works, and the "invisible plumbing failure" hypothesis
      is dead.
    - **(b) live half ELIMINATED.** Packed upload order == live `team.values()` in **40/40** battles
      (match 1.00, stable 1.00). Live order is confirmed = upload order; train `|poke|` lines are upload
      order by protocol, so the train-side gate is deprioritized to negligible.
    - **(c) CONFIRMED — the failure is visible and stereotyped.** Both agents open *sanely* (turn-1 tera +
      attack; move spam while an attacker is active), then fall into **absorbing two-mon switch loops**
      the moment a defensive mon is in (transcripts: Gholdengo↔Corviknight for 70+ consecutive turns,
      Kingambit↔Corviknight): voluntary switch fraction **0.77 / 0.70** vs train base **0.184** (bar
      0.50), max consecutive-switch runs **117 / 129**, ping-pong rate **0.77 / 0.84**, mean top-1 prob
      0.62 — confident, legal, and losing. Games last 84–91 turns and end only when random happens to KO
      through; win 0.00 / 0.05. NOT one-move spam (top-share 0.16/0.21, entropy 3.2/3.1 bits) — the
      collapse is specifically *switch mass*.
    - **The loop is in the policy mass, not argmax brittleness:** in loop states switches carry ~0.9 of
      the masked probability (top-3 all switches at ~0.4/0.3/0.2), so sampling would still switch ~90% of
      the time. Echoes the L11 depth-1 finding — the RB *value head* over-switched ("my strong mon
      active" rated above "opp at 21% HP"); the same signature now appears in the OU *imitation policy*.
  - **Why a loop can be absorbing at all:** the obs is memoryless (no last-action/recency channel), so
    "wall A active vs opp X" looks identical whether we just switched in or not — a 2-cycle is a fixed
    point for a deterministic policy whose per-state argmax is "switch".
  - **Next levers (cheapest first, each its own gated build).** (1) **Train-side switch-mass diagnostic**
    — on the existing v5 shard, measure the human switch rate in the states the loop lives in (own
    defensive mon active, healthy) and the trained policy's switch mass on those *training* states:
    discriminates "faithful imitation of a pivot-heavy human prior that composes into a loop under
    self-play" vs "OOD generalization artifact"; offline, zero training. (2) Depending on (1):
    anti-loop *learning* signal (history/recency feature + retrain, or switch-damping at inference as a
    diagnostic-only counterfactual), vs distribution-drift work (d). OU ceiling re-probe + OU PPO stay
    gated OFF.
  - **Lessons.** (1) Win-rate-only evals hid a two-mon switch loop for five builds — a per-decision
    transcript of TWO games would have shown it in Build 2; make the behavioral probe a standing tool for
    any "agent inexplicably weak" state. (2) Observe-only hooks cheaply *falsify* plumbing causes before
    touching semantics: both invisible-failure channels (fallback, rejections) measured exactly zero.
    Suite **198 pass / 5 skip** (203 with the local server up), ruff + mypy clean;
    `results/behavior_probe.json` + `results/behavior_probe_transcripts/` committed.

- **OU pivot — Build 7: train-side switch-mass diagnostic — built + run; H2 UNANIMOUS (the switch loop
  is an OOD generalization artifact: the live ~0.9 switch mass never appears on training states;
  imitated-pivot-prior and training-amplification causes eliminated by direct measurement).**
  - **Built (offline, zero training, no server).** `scripts/switch_mass_gate.py`: decodes own/opp-active
    species + HP straight from the v5 shard obs via `OBS_LAYOUT`-derived offsets (block idx 1 active /
    3 hp / 44 species; `force_switch` at global −3 measured ≡ 0 on every row — all shard rows are
    voluntary decisions); loop states extracted from the Build-6 decisions JSONL (runs ≥ 10 → 8 loop
    species, 27 (own, opp) pairs; frozen as fallback constants since the JSONL is gitignored); a
    dex-defensiveness secondary lens from the raw `id_priors` z-stats (needed because the empirical loop
    set is NOT purely walls — Gholdengo/Great Tusk score ~0.2). Per (arm × shard × conditioning) it
    reports **H** (human switch rate, `action<6`), **P** (masked-softmax switch mass on actions 0–5),
    **T** (top-1-is-switch), and **U** (uniform-over-legal anchor, ~0.44–0.47 here — "high" must beat
    this, not 0). Pre-registered H1(pivot prior)/H2(OOD)/H3(amplification) tree + controls with teeth:
    zero-logits harness identity (err 1e-16), random-init ET band (0.53–0.54 ∈ [0.25, 0.65]), taken-action
    mask invariant, sample floors, and an attacker-specificity control (mass on `dex_attacker` states must
    stay ≤ 0.30 or the wall-conditioning story collapses to H3-GLOBAL). Secondary replay-log pass over the
    2,760 raw logs bounds the ingest undercount. 23 pure-logic tests.
  - **Findings (6 ckpts × both full shards, 61,723 + 119,996 rows; loop-species n = 9,728 / 19,077):**
    - **H2 on all 6 seeds, both families.** On the exact loop-species states: H **0.205/0.210**, P
      **0.161–0.229** (matched — max |P−H| 0.049), T **0.034–0.056** — in-distribution the argmax picks a
      switch < 6% of the time, vs the live 0.77 voluntary-switch fraction and ~0.9 mass. Even on the
      exact live loop (own, opp) pairs (n 1,534/3,055) P_pair is only **0.165–0.234**. `in_dist_live`
      never fires (bar 0.60); `argmax_amp` never fires.
    - **H1 dead:** humans are NOT pivot-heavy in wall states — H 0.205 ≈ overall 0.184 (dex_wall H
      0.21–0.22, far below the 0.40 bar). The replay-log true voluntary rate is **0.211** vs shard 0.189:
      the ingest undercount is only +0.02 (43% of human switches are first-reveals ingest drops, but the
      rate barely moves), so the shard H is honest.
    - **H3 dead:** no amplification anywhere (max P−H **+0.024**); attacker control clean (0.13–0.20,
      wall-conditioned story intact); BC-vs-AWR delta on the same shard **−0.008** — AWR adds zero
      switch mass over pure BC.
  - **Implication.** The three candidate mechanisms are now all measured: imitated prior dead,
    amplification dead, leaving **distribution shift confirmed by elimination + direct contrast** — the
    live states (full-kit `|request|` detail, curated-teampool teams, self-play loop states) sit off the
    human-shard manifold, and off-manifold the policy's mass collapses onto switches. Prime suspect stays
    the *measured* Build-2/3 train≠live obs-detail gap. Mapped next lever per the pre-registered tree =
    **drift-side work**: a live-state-vs-shard distance/causal probe to localize which channels carry the
    shift. A history/recency feature is NOT the indicated fix — there is nothing to damp in-distribution.
    **[Corrected by Build 8]** this bullet also named "usage-prior imputation to bring train own-team
    detail to live-FULL" as the next lever — that was **stale**: Build 4 already did exactly that (v5
    shard verified at eval-full item/ability/move density) and it was RED. Build 8 ran the localization
    probe instead and found the carrier is **not** own-kit detail at all but the move **pp-fraction**
    channel (see the Build 8 entry).
  - **Lessons.** (1) The pathology lives entirely OFF-shard: no shard-side metric (BC val-acc, value-MAE,
    this gate's P) could have seen it — pair any train-side mass gate with a live behavioral probe, and
    treat "healthy in-distribution + insane live" as a drift signature, not a modeling bug. (2) When a
    gate's conditioning comes from a gitignored artifact, freeze the extraction output as committed
    fallback constants or the gate is unreproducible from a clone.
    Suite **221 pass / 5 skip**, ruff + mypy clean; `results/switch_mass_gate.json` committed.

- **OU Pivot — Build 8: live-vs-shard drift localization — built + run; CARRIER ISOLATED to a single
  encoder channel: the move `pp-fraction` (the loop is self-sustained OOD via "all my moves are at full
  pp because I never attack").** Build 7 proved the ~0.9 live switch mass is an OOD artifact (H2) but did
  not say *which* channels carry the drift; its recorded "next lever = usage-prior imputation" was stale
  (Build 4 did that → RED). This build localizes the drift offline, zero training, via a **causal swap
  bisection**, and the answer is one channel — not own-kit detail, not team composition, not the opponent.
  - **Phase A (obs capture) — extended `behavior_probe.py`.** Wrapped the agents' `embed_battle` (like the
    existing `masked_logits` hook) to dump the exact per-decision obs/mask the model scored to a gitignored
    npz aligned with the decisions JSONL, and added a **held-out-eval-opponent arm** ({bc,offrl}×{random,
    heuristic}, n=20). The pathology reproduces vs the real eval opponent too (vol-switch 0.67–0.85, runs to
    194). Validated the Phase-A→B path offline: the captured live loop obs, re-scored through the frozen
    checkpoints, reproduce **P ≈ 0.74** switch mass (matching the live ~0.74 vol-switch fraction) — so the
    offline swap operates on genuinely-live states. 5,885 loop-rich obs rows over 4 arms.
  - **Phase B (`scripts/drift_probe.py`) — the causal swap.** Pair-match each live loop state to shard rows
    with the same (own-active, opp-active) species, then per channel group paste that group's values across
    the manifold both ways and re-score the frozen policy: **deletion** (live base ← shard donor: does the
    switch mass FALL to the shard's ~0.2?) and **insertion** (shard base ← live donor: does it RISE to
    ~0.9?). ``frac_explained`` = |ΔP| / |P_live − P_shard|; the swap holds action legality fixed (hybrid
    keeps the base mask), isolating obs *features* from *legality*. Reuses the Build-7 gate machinery
    (`decode_actives`, `load_policy`, `score_shard`, `masked_softmax_np`, the frozen loop constants).
    Groups from `OBS_LAYOUT`: own-active (ids vs numeric), own-bench, opp-active, opp-bench, moves, global.
  - **Carrier = `moves`, +0.875 of the gap (bc +0.880 / offrl +0.870), every other group ≈ 0.** Both
    directions agree on every one of the 6 checkpoints (deletion +0.89–0.97, insertion +0.75–0.87) — the
    gold-standard remove-cause-and-effect-vanishes / add-cause-and-effect-appears pattern, not an asymmetry
    artifact. Controls all pass: **C-self** (donor==base identity swap) err **0.0**, **C-full** (ALL-channel
    swap must move ≥0.5 of the gap) **0.98**, **C-harness** (torch vs numpy masked softmax) **1.7e-7**,
    floors met (2,805 matched live ≥ 300, 3,776 shard ≥ 500).
  - **Sub-split pins it to ONE channel — `pp_fraction`, +0.875 (the whole gap); the R-CALC expected-damage
    `score` is inert at +0.016.** Progressive within-move-block splits: `move_ids` +0.193 (identity minor),
    `move_numeric` +0.688, `move_context` (pp+score) +0.869, then **`move_score` +0.016 vs `move_pp`
    +0.875**. So it is *not* the move identity, *not* the damage proxy, *not* base-power/type — it is the
    single pp-fraction channel of the active mon's move blocks.
  - **Mechanism (triangulated 3 ways; unifies Builds 6+7).** Direct distribution: live loop states have
    **93.6%** of moves at full pp vs the shard's **55%** — because the looping agent *never attacks*, so its
    active mon's pp stays maxed, and "all four moves simultaneously at full pp deep in a game" is
    off-manifold vs human data (humans attack, depleting pp). The policy has learned a sharp full-pp→switch
    correlate; off-manifold it collapses onto switching, which keeps pp full → the loop is **self-sustaining
    through the pp channel**. This is the concrete carrier of Build-6's "memoryless obs → absorbing 2-cycle"
    (pp *is* the hidden "I haven't acted" variable) and Build-7's H2 OOD. Onset stratification corroborates
    the compounding: switch mass on the **first** loop-state decision per battle is 0.305, rising to 0.752
    later. Lesson-in-passing: the distance screen ranked `own_bench` (0.30) highest while the *causal*
    carrier was `moves`/pp — **raw obs distance ≠ causal importance**; the swap is what discriminates.
  - **Phase C (corpus-teampool live A/B) — DEFERRED (would test a hypothesis Phase B decisively excluded).**
    The planned confirmatory arm tests own-team *composition* (G2). Phase B ruled that out (own_bench ≈
    0.02) and localized to a single encoder channel; moreover `pp_fraction` depends on move *usage during
    the game*, not the team, so a corpus team cannot change the pp mechanism. Per the project's cheap-gate-
    first pattern (skip the expensive confirmatory when the decisive gate answers), it is deferred; the
    replay→packed-team exporter remains a Build-9 asset candidate only if a composition angle ever
    resurfaces.
  - **Verdict + next lever.** The carrier is `pp_fraction` — a low-information, OOD-brittle channel. Per the
    pre-registered tree this maps to an **encoder/robustness** fix, not more data: candidate Build-9 levers,
    cheapest first — (1) ablate/robustify the pp channel at train (drop it, or noise/dropout-augment it, or
    synthesize full-pp deep-game states) and re-run the behavior probe: does the loop break? (2) add an
    explicit last-action/recency feature so "just switched, haven't attacked" is *represented* rather than
    leaking through pp (Build 7 deprioritized this for lacking an in-distribution target; the pp mechanism
    revives it — the loop is absorbing precisely because pp is the only trace of "I keep switching"). Each
    is BC-gateable before any AWR/PPO spend. OU ceiling re-probe + OU PPO stay gated OFF.
  - **Lessons.** (1) When a behavior is OOD, localize the *channel* with a causal swap before proposing a
    fix — three prior builds (2/3/4) chased own-kit item/ability/move *detail* and one (5) chased slot
    order; the actual carrier was a channel none of them touched. (2) Distance is a screen, not a verdict —
    only the causal swap (with a self-swap identity control and an ALL-swap positive control) separates
    "different" from "responsible." Suite **244 pass** (+23: drift-probe pure logic + obs-capture),
    ruff + `mypy lategame` clean (the gate scripts carry the same 2 pre-existing "assign to a type"
    monkeypatch notes as Build 6); `results/drift_probe.json` + `results/behavior_probe8.json` committed
    (obs/decisions/transcripts gitignored).
- **OU Pivot — Build 9: the pp-channel fix (two gates: drop pp / add first_turn) — built + run; BOTH gates
  triangulate that the fix must ROBUSTIFY pp, not drop it or out-vote it.** Build 8 isolated the switch-loop
  carrier to the move `pp_fraction` channel (OOD "all moves at full pp because I never attack"). Build 9
  runs the two pre-registered, BC-gateable levers cheapest-first. Implementation note: the pp ablation is
  realized by zeroing the channel in place (dim/layout unchanged, so `drift_probe` + shape tests stay
  intact) rather than deleting the slot — scientifically identical for the policy (a constant input weight
  is dead), and each gate re-ingests under a bumped `OBS_VERSION` so train and live agree.
  - **Gate A (drop pp) — BC RED: pp is load-bearing for imitation.** Ablate `pp_fraction` to a constant 0
    (`OBS_VERSION` v3→**v4**), re-ingest (v6 shards: 61,723 BC / 119,996 turns — identical counts, and the
    v6 obs is **byte-identical to v5 on every non-pp column**, actions identical, so the change is surgical),
    3-seed BC ET+prior. Val-acc collapses **0.647 → 0.390** (−0.257), far below the 0.63 bar. Removing one
    "low-information" channel tanks imitation → pp is *not* droppable: it is the encoder's implicit trace of
    own move-usage/recency and feeds the dominant switch-vs-attack axis. Kill-gate stops before a (confounded)
    live probe on a 0.39 agent. `results/bc_gate9a.json`.
  - **Gate B (keep pp + add explicit `first_turn` recency) — BC PASS but the live loop PERSISTS.** Append one
    global scalar `float(active.first_turn)` (poke-env "first action since switch-in"; driven by the same
    protocol messages offline+live, so **drift-free**, unlike pp), pp retained (`OBS_VERSION` v4→**v5**, 761-d).
    v7 shard is byte-identical to v5 on the first 760 cols (surgical add). The shard signal looks perfect —
    `first_turn=1` → switch **9.8%** vs `first_turn=0` → 23.9% (humans just-switched-in switch 2.4× *less*),
    and BC val-acc **0.654** clears 0.63 *and beats* the pp-only 0.647 (first_turn is genuinely informative).
    But live (bc_v7_s0, n=20 vs random+heuristic) the loop **persists**: both arms `c_pathological`
    (max_switch_run 110/77, ping-pong 0.73/0.65, win 0.05/0.0) — attenuated vs Build 8 (vol-switch ~0.49–0.64
    vs ~0.75) but not broken.
  - **Why Gate B fails — causal counterfactual on the v7 BC policy (which HAS first_turn).** `first_turn` is
    NOT drifted (fires live, 56% of decisions), yet the policy **inverts** its shard prior under OOD: live
    `first_turn=1` → **87%** switch (vs the shard's 9.8%), because pp stays full (`frac_full` 0.86 live / 0.51
    shard; means near-identical 0.946/0.943 — the OOD signal is *exactly-full* pp, per Build 8). Neutralizing
    pp to in-distribution (drawing shard pp, keeping first_turn) drops live switch mass **0.604 → 0.244
    (ΔP −0.36)**; flipping `first_turn` moves it only **±0.03–0.04**. So with pp present, pp out-weighs the
    honest recency feature **~10×** — a parallel in-distribution feature cannot rescue a policy from an OOD
    carrier that is still there. `results/first_turn_gate9b.json` + `results/behavior_probe9b.json`.
  - **Verdict + next lever.** `first_turn` is **kept** (drift-free, +val-acc) but is **insufficient** as a
    standalone loop-fix. The two gates close the pincer: pp can't be **dropped** (Gate A, load-bearing) and
    can't be **out-voted** (Gate B, ΔP −0.36 ≫ 0.03) → the only remaining branch is to **robustify pp** while
    keeping it — **Build 10**: train-time augmentation of the pp channel (noise/dropout, or synthesize
    full-pp deep-game states so "full pp deep in a game → attack" is in-distribution), then re-run the
    behavior probe. BC-gateable; OU ceiling re-probe + OU PPO stay gated OFF.
  - **Lessons.** (1) A channel's *information value for imitation* and its *OOD brittleness* are separate axes:
    pp is simultaneously load-bearing (drop = −0.26 val-acc) and the loop's OOD carrier — you cannot fix the
    second by removing the channel. (2) Adding an honest, in-distribution feature *alongside* an OOD carrier
    does not neutralize the carrier; the policy keeps weighting the (present) OOD channel. The carrier itself
    must be made robust. (3) The offline causal counterfactual (ΔP from a targeted channel edit) diagnosed
    the live failure with zero extra training — reuse it before proposing the next fix. Suite **245 pass**
    (+1: the first_turn encoder test), ruff + `mypy lategame` clean; `results/bc_gate9a.json` +
    `results/first_turn_gate9b.json` + `results/behavior_probe9b.json` committed (obs/decisions/transcripts
    gitignored).

- **OU Pivot — Build 10: robustify pp by SYNTHESIZING full-pp deep-turn states — built + run; BC PASS but
  the loop PERSISTS, and the diagnostic proves the fix is region-local (can't reach the loop).** Build 9's
  pincer left "robustify pp" as the only branch, with two pre-registered mechanisms (noise/dropout, or
  synthesize full-pp deep-game states); this build runs **synthesize**. New train-time augmentation
  `lategame/train/augment.py::augment_pp_full` (+`TrainConfig.pp_aug_frac/pp_aug_turn_threshold`, CLI
  `--pp-aug-frac`, +8 unit tests): on a random `frac` of **attack-labeled** (`action ≥ team_size`),
  **deep-turn** (normalized turn ≥ threshold) rows, force the active mon's pp channels to full (present-guarded,
  offsets from `OBS_LAYOUT`), so "full pp deep in a game → attack" is in-distribution. Applied **train-time
  only** (in `_run_epoch`, gated on `optimizer is not None`), so validation stays clean **and the encoder +
  shards are unchanged — no re-ingest, no `OBS_VERSION` bump** (HEAD stays v5/761).
  - **BC gate — PASS.** 3-seed ET+prior on `data/gen9ou_v7_bc.npz`, frac 0.5 / turn ≥ 0.15: val-acc
    **[0.648, 0.654, 0.658], mean 0.653 ≥ 0.63** — imitation preserved (matches the first_turn 0.654).
    `results/bc_gate10.json`.
  - **Live probe — LOOP PERSISTS.** `bc_v10_s0` vs random+heuristic (n=20): both arms `c_pathological`
    (max-switch-run **108 / 989**, ping-pong 0.62/0.88, win 0.0/0.0) — unbroken vs Build 9 (110/77).
    `results/behavior_probe10.json`.
  - **Why (pp-reliance diagnostic, `results/pp_reliance_diag10.json`).** Self-contained pp-neutralization
    (draw in-distribution pp from the v7 BC shard pool, `frac_full` 0.50): on the **identical** frozen v7 loop
    states, v10 baseline switch mass **0.597 ≈ v7 0.575** and v10 pp-**ΔP 0.390 ≈ v7 0.397** — the augmentation
    did **not** reduce pp-reliance at all. The synthesized examples are *real mid-game attack states with pp
    maxed* → they populate the **attack-context** region, not the **loop (repeated-switch) context** region;
    the policy fits both "full pp + attack ctx → attack" and "full pp + loop ctx → switch" at once because pp
    is not the sole discriminator. **Region-local label augmentation of real states cannot reach the loop
    corner — increasing `frac` will not help.**
  - **Verdict + next lever.** Synthesize is **empirically ruled out**. The remaining pre-registered candidate
    is **Build 11 (BC-gateable): noise/dropout on the pp channel** — a *global* regularizer that blunts the
    `exactly-1.0 → switch` extrapolation in **all** contexts (including the unseen loop region), which
    region-local synthesis provably cannot. The `augment.py` hook + gate harness generalize directly (add a
    pp-noise transform beside `augment_pp_full`). OU ceiling re-probe + OU PPO stay gated OFF.
  - **Lessons.** (1) A train-time data augmentation of **real** states only robustifies the regions those
    states occupy; a failure mode that lives in an *unreachable* region of feature space (the loop context)
    can't be fixed by relabeling real examples — you need a mechanism that acts on the feature *globally*
    (noise/dropout) or that can synthesize the failure region itself. (2) Passing the BC gate proves the
    augmentation didn't break imitation, but says nothing about the loop — the live probe + the ΔP diagnostic
    are what adjudicate, and here they agree the mechanism is untouched. Suite **253 pass** (+8 augment
    tests), ruff + `mypy lategame` clean; `results/bc_gate10.json` + `results/behavior_probe10.json` +
    `results/pp_reliance_diag10.json` committed (obs/decisions/transcripts + checkpoints gitignored).

- **OU Pivot — Build 11: robustify pp by GLOBAL regularization (noise + resample), decide by gates —
  built + run; PARTIAL WIN: the first mechanism to actually MOVE the live loop, but BC-passing strength
  attenuates (loop depth 108 → 29, ~4×) without fully breaking it.** Build 10 ruled out synthesize; this
  build runs the *other* pre-registered candidate — a **global** pp regularizer applied in **every** context
  (no attack/deep gate), so it can reach the loop corner region-local synthesis could not. Two flavors added
  beside `augment_pp_full` in `lategame/train/augment.py`: **`augment_pp_noise`** (Gaussian jitter on pp,
  `[0,1]`-clamped; `--pp-noise-std`) and **`augment_pp_resample`** (a random `frac` of present pp cells
  resampled from the batch's own pp pool — an empirical draw from the shard's ~50%-full pp distribution,
  mirroring the proven neutralization counterfactual; `--pp-resample-frac`). +`TrainConfig.pp_noise_std/
  pp_resample_frac`, +13 unit tests (pin the *global* reach: switch rows AND shallow turns are perturbed).
  The pp-reliance diagnostic was **promoted from the Build-10 inline snippet to a committed script**
  `scripts/pp_reliance_diag.py` (reuses `switch_mass_gate.load_policy/score_shard`; re-baselines confirm
  v7_s0 ΔP **0.377** ≈ v10_s0 **0.369**, reproducing the Build-10 finding). All **train-time only — no
  `OBS_VERSION` bump** (HEAD stays v5/761).
  - **Stage 1+2 screen (seed 0, 6 configs; `results/bc_gate11_screen.json`).** **Gaussian noise RULED OUT** —
    it collapses BC at *every* strength (σ 0.05 → val-acc **0.575**, 0.10 → 0.501, 0.20 → 0.442; all < 0.63):
    pp is far too load-bearing for additive jitter (the model can't fit clean val pp after training on
    always-noised pp). **Resample** shows a clean monotone frontier — p 0.10 → val **0.648 PASS**, ΔP 0.203;
    p 0.25 → val 0.627 (just misses), ΔP 0.113; p 0.50 → val 0.551, ΔP 0.039 (more resampling → lower
    pp-reliance but lower BC). **Winner = resample p 0.10** (the only BC-pass; ~halves ΔP vs v10).
  - **Stage 3 confirm (3-seed + live probe; `results/bc_gate11.json`).** **BC PASS** — val-acc
    **[0.6444, 0.6434, 0.6442], mean 0.644 ≥ 0.63** (on par with the v7 baseline 0.647). **pp-reliance HALVED**
    — on the identical frozen v7 states, `bc_v11_s0` baseline switch mass **0.404** (vs v10 0.597 / v7 0.575)
    and **ΔP 0.187** (vs v10 0.369 / v7 0.377), the first mechanism in the investigation to reduce it at all.
  - **Live probe — LOOP ATTENUATED ~4× BUT NOT BROKEN** (`bc_v11_s0`, n=20, random+heuristic). `bc_vs_heuristic`
    `c_pathological` but **max-switch-run 29 (vs Build 10's 108)**, voluntary-switch **0.33 now < the 0.5 bar**
    (the switch-fraction flag no longer trips; only switch-run ≥ 6 and ping-pong 0.52 > 0.25 do), win 0.0.
    `bc_vs_random` flips to `b_team_order` — a separate team-order/decode issue that takes precedence and
    surfaces once the loop is attenuated (its c-flags still show switch-run/ping-pong, so the loop is present
    but no longer the *primary* cause on that arm).
  - **Verdict + next lever. PARTIAL SUCCESS — mechanism validated, strength-capped.** pp is confirmed the
    causal loop carrier, and a global regularizer that reaches all contexts *does* move the live loop (unlike
    region-local synthesize) — cutting pp-reliance ~50% and loop depth ~4× while preserving imitation. But the
    **BC-passing frontier caps ΔP reduction at ~0.19**: going lower (p ≥ 0.25 → ΔP 0.11) fails the 0.63 gate,
    so at survivable strength the residual pp signal still sustains shorter ping-pong loops (win 0). **Build 12
    candidates (BC-gateable):** (a) push past the frontier with an **encoder-level pp intervention** (bins /
    monotone transform — *would* need an `OBS_VERSION` bump + re-ingest) or a **hybrid** global-resample +
    deep-turn-targeted resample; (b) **isolate the `b_team_order` signal** on the random arm as a possible
    independent decode confound; (c) resample p 0.25 (ΔP 0.11) is a near-miss (val 0.627) worth a relaxed-bar
    probe if the loop, not BC, is the binding constraint. OU ceiling re-probe + OU PPO stay gated OFF. Suite
    **266 pass** (+13 augment tests), ruff + `mypy` clean; `results/bc_gate11_screen.json` + `bc_gate11.json` +
    `behavior_probe_v11.json` committed (obs sidecar + checkpoints gitignored).

- **OU Pivot — Build 12: resample p 0.25 relaxed-bar probe + close `b_team_order` (candidate c, decided) —
  FRONTIER-CONFIRMED, loop still not broken; the residual ping-pong is pp-INDEPENDENT → pivot off pp.**
  - **Cheap, runs-only** (no code, no `OBS_VERSION` bump, no re-ingest): the Build-11 CLI already exposes
    `--pp-resample-frac`, and `pp_reliance_diag.py` + `behavior_probe.py` are committed. Mirrored the Build-11
    confirm protocol exactly (`entity_transformer`, d_model 128, `id_embed_init prior`, 20 epochs), swapping
    resample `0.10→0.25` and `v11→v12`. Pre-registered a **relaxed bar** (judge on the live loop; BC must stay
    ≥ 0.625 within noise, void < 0.62) because the question is whether the loop or BC is the binding constraint.
  - **BC — MARGINAL.** 3-seed val-acc **[0.6298, 0.6218, 0.6191], mean 0.6236** — misses the 0.63 hard bar and
    sits ~within seed noise (std ~0.005) just below the 0.625 relaxed line, above the 0.62 void line. vs p 0.10's
    0.644, p 0.25 costs **~2pts val-acc**: the imitation frontier is real and monotone.
  - **pp-reliance — continues to drop** (offline, identical frozen v7 states, so directly comparable): ΔP **0.122**
    (v11 0.187 / v10 0.369 / v7 0.377) and baseline switch mass **0.343** (v11 0.404 / v7 0.575). More resample →
    lower pp-reliance, as designed.
  - **Live probe (n=20, both arms) — LOOP SHORTER BUT NOT BROKEN.** `bc_vs_heuristic` still **`c_pathological`**:
    max-switch-run **21** (v11 29), voluntary-switch 0.221 (v11 0.335), but **ping-pong 0.55 ≈ v11 0.52** and
    **win 0.0**. `bc_vs_random` flips **back to `c_pathological`** (v11 was `b_team_order`) with max-switch-run
    **14** (v11 73) and **win 0.9** (v11 0.3). The pp-CARRIED long-run component keeps shrinking (depth down,
    random win up), but the short **ping-pong (A→B→A) is pp-INDEPENDENT** — its rate does not move even as ΔP
    halves again — and it still costs every game vs the competent heuristic.
  - **`b_team_order` — RESOLVED as noise, not a decode/packing bug (candidate b closed, no separate build).**
    v11's flag came from `stable_rate` 0.95 (**1/20** battles unstable) with `match_rate` already a perfect 1.0
    (no packed-vs-live mismatch); v12 shows `match_rate` 1.0 **and** `stable_rate` 1.0 on **both** arms — the
    single-battle instability did not reproduce. Team order is set pre-decision so BC weights can't cause it.
  - **Verdict + next lever. Pre-registered Outcome 2 (resample capping on the core loop), refined:** halving
    pp-reliance again (0.187→0.122) still loses 100% vs the heuristic with a flat ping-pong → **pushing pp-reliance
    lower — more resample OR the Build-13 encoder pp transform — will not break the ping-pong, because it is not
    pp-carried.** So **DEPRIORITIZE the encoder pp transform as a loop fix** (it would only shrink the
    already-shrinking pp-carried runs at more BC cost) and **PIVOT to the pp-independent ping-pong**: (a) localize
    its carrier the way Build 8 localized pp (drift/counterfactual on the 2-cycle states), or (b) a decision-time
    anti-repetition intervention / RL fine-tune with a loop penalty (imitation structurally cannot fix an OOD
    2-cycle absent from winning human play). BC 0.6236 rules out shipping p 0.25 as a milestone — **p 0.10 (v11)
    stays the best BC-passing resample point.** OU ceiling re-probe + OU PPO stay gated OFF. Suite unchanged
    (no code touched); `results/bc_gate12.json` + `pp_reliance_diag12.json` + `behavior_probe_v12.json` committed
    (obs sidecar + checkpoints gitignored).

- **OU Pivot — Build 13: localize the ping-pong carrier (clone Build 8's causal-swap on the 2-cycle states) —
  VERDICT `PP_CARRIED` (3-seed): the pp-independent *rate* is a sticky-argmax phenomenon; the 2-cycle decision
  is ~100% pp-carried → reverses the Build-12 deprioritization of the encoder pp transform.**
  - **Runs-only** (no `OBS_VERSION` bump, no re-ingest, no retrain): new `scripts/pingpong_probe.py` +
    `behavior_probe.two_cycle_rows` (single-sources the ping-pong definition; `_ping_pong_rate` refactored to
    reuse it) + tests, reusing `drift_probe.swap_group`/`carrier_verdict`/controls and
    `pp_reliance_diag.neutralize_pp`. Fresh aligned capture from the shipped v11 winner
    (`bc_gen9ou_v11_s0`, n=50; `results/behavior_probe_obs_v13.npz` + `_decisions_v13.jsonl`, gitignored).
  - **The naive metric just re-finds pp** (the smoke on v12 showed `move_pp` +0.89): a 2-cycle active mon just
    switched in → full pp, Build 8's exact OOD cue. So the gate is **two-stage** — Stage 1 neutralizes pp
    (`neutralize_pp`, frac_full 0.99→0.51) and measures its grip; Stage 2 localizes the *residual* on the
    pp-neutralized base. Metric = **P(return action)** (softmax prob on the recorded switch-back index),
    conditioned on **own active species** (own-only matching keeps all rows; (own,opp) discards ~70%). Verdict
    read on the **generating** seed (s0); s1/s2 are robustness only (they need not reproduce the loop).
  - **Data + controls (all pass):** 337 A→B→A rows on `bc_vs_heuristic`, **322 matched-live** across 9 species,
    10 041 shard donors; C-self 0.0, C-gather-harness 6.5e-8, floors clear.
  - **Result — `PP_CARRIED`, all 3 seeds agree.** Primary s0: baseline P(return) **0.306** → **pp-neutralized
    0.073** → full-neutralized 0.076, so **pp_share 1.01** (pp explains ~100% of the drop) and **residual
    0.237 < 0.40**. s1/s2 residual 0.203 / 0.276 → both `PP_CARRIED`. Switch-mass cross-check: **0.611 → 0.152**
    on pp-neut (dP **0.46**, >> `pp_reliance`'s full-capture 0.187 — the 2-cycle is a pp-saturated fresh-switch
    region). There is **no residual to localize** (Stage-2 fracs 0.000, `move_pp` sanity 0.000).
  - **The decisive disambiguation — `P(return|switch)` is pp-INVARIANT: 0.501 → 0.479.** So pp carries the
    **switch-vs-stay** decision; conditional on switching, bouncing back to A is ~50% *regardless of pp* — a
    structural "few viable targets" effect (uniform floor ~0.157 over 6.37 legal actions; pp-neut return 0.073
    sits *below* uniform). The ping-pong = "pp says switch, and a switch has a coin-flip chance of going back."
  - **Reconciliation with Build 12:** the pp-independent **rate** is real but is a *sticky-argmax* artifact —
    resample shrank the pp probability margin (switch mass, ΔP 0.187→0.122) yet could not flip the discrete
    argmax, so the rate stayed flat while the mechanism remained pp. "pp-independent rate" ≠ "pp-independent
    mechanism."
  - **Verdict + next lever. `PP_CARRIED` reverses the Build-12 deprioritization of the encoder pp transform.**
    The 2-cycle is pp-driven at the decision level; the **resample lever is exhausted** (continuous margin
    reduction plateaued at the BC frontier) and pp is **load-bearing** (Build 8 v4 ablation collapsed BC, so it
    can't be dropped). **Build 14 levers:** (a) **encoder pp-transform** — change pp's *representation*
    (bucketize / drop the exactly-full spike / replace with an honest "turns since this mon attacked" recency)
    to break the argmax lock; stays in imitation but costs an `OBS_VERSION` bump + re-ingest and risks BC. Or
    (b) **decision-time anti-repetition** — penalize the return action directly; this attacks the pp-INVARIANT
    structural bounce-back, is cheap, and needs no retrain (RL loop-penalty is the heavier sibling).
    **Recommendation: (b) first** (the pp-invariant `return|switch` says the bounce-back is structural, not a
    feature the encoder can robustify away), with (a) the principled-but-costly imitation alternative. OU
    ceiling re-probe + OU PPO stay gated OFF. Suite 271 pass, ruff + mypy(lategame) clean;
    `results/pingpong_probe.json` + `behavior_probe_v13.json` committed (obs/decisions/transcripts + checkpoints
    gitignored).

- **OU Pivot — Build 14: decision-time anti-repetition (loop guard) — the absorbing switch loop is BROKEN
  (`max_switch_run` 58/26 → 2/2), with a milder interleaved ping-pong residual persisting (rate 0.30-0.37 > 0.25).**
  - **Runs + small code, no `OBS_VERSION` bump / re-ingest / retrain.** New `lategame/agents/loop_guard.py`
    (`LoopGuard`, torch-free) wired into `BCAgent`/`OfflineRLAgent.choose_move` between `masked_logits` and the
    argmax; a `--loop-penalty` flag on `behavior_probe` threads it through `_build_probe_player` for a clean A/B on
    identical machinery. `LoopGuard(0)` is exact identity. Chosen mechanism (per user): a **soft escalating logit
    penalty**, not a hard mask (a finite penalty can never make the only legal action unreachable → no
    forced-switch hang, and it leaves attacks untouched so the argmax is pushed toward attacking).
  - **Two mechanism iterations (CLAUDE.md re-plan discipline).** (1) *return-only* — penalize just the switch-back
    action (mirror `two_cycle_rows`). Live it merely converted the tight A→B→A into a longer roster-cycle via
    **fresh-mon escape** (ping_pong 0.44→0.13 but `max_run` 17→26, switch mass flat, win 0) → **ruled out**
    (fresh-mon escape is structural to a return-only penalty; no window/magnitude fixes it). (2) *streak* —
    penalize **every** voluntary switch by `penalty·max(0, run − free_switches)` once a consecutive-switch run
    forms (`free_switches=1`: a lone scout and a double-switch pivot stay free, the penalty first bites on the 3rd
    consecutive switch where the 2-cycle forms), directly pressuring the pp-driven switch mass toward attacking.
    Dropped the species/window tracking — a run counter is all the streak penalty needs.
  - **Result (n=50, `bc_gen9ou_v11_s0`, p=4).** vs random: `max_run` **58→2**, vol_switch 0.171→0.107, win
    **0.54→0.58**; vs heuristic: `max_run` **26→2**, vol_switch 0.337→0.169, win 0.00→0.02. Fallback/rejection
    clean both arms. **The absorbing consecutive switch loop — the acute pathology since Build 8 — is broken and
    the agent is functional vs random.** Smoke: p8 is no better than p4 (over-penalizes, win 0) → **p4 is the
    pick**.
  - **Residual (honest caveat).** `ping_pong_rate` fell ~0.55→0.30-0.37 but stays **> 0.25** (the `ping_pong`
    c-flag still trips): the run resets on every attack, so the guard kills *consecutive* loops but not a **slower
    interleaved oscillation** (switch→attack→switch→attack, 2-periodic across turns). It is much milder (win vs
    random 0.58), and the still-~0 heuristic win reflects the OU policy's **general weakness (FORMAT_BOUND, gated
    off)**, not the loop.
  - **Verdict: loop BROKEN, committed as-is.** Suite **276 pass** (271 + 5 loop-guard), ruff + mypy(lategame)
    clean; `results/behavior_probe_v14_{off,on}.json` committed (obs/decisions/transcripts + checkpoints
    gitignored). **Open next:** the interleaved residual would need a persists-across-attacks penalty (risks
    over-penalizing legit pivots and won't lift heuristic win); the heuristic-win frontier is the separate
    FORMAT_BOUND strength problem. OU ceiling re-probe + OU PPO stay gated OFF.

- **OU Pivot — Build 15: OU ceiling re-probe (loop-fixed) — the OU FORMAT_BOUND label was inherited from RB and
  is now measured WRONG: OU is `MODEL_BOUND`, FORMAT_BOUND rejected.**
  - **Motivation.** Build 14 produced the first loop-free OU agent, but the "OU heuristic-loss is FORMAT_BOUND"
    posture was never *computed on OU* — `scripts/format_ceiling_gate.py::assess_ou` deliberately withholds the
    FORMAT/MODEL verdict for teambuilt formats (no OU near-optimal reference), so the label was inherited from the
    RB run. And no win-rate harness scored the loop-fixed agent: `arena.build_player` / CLI `evaluate` /
    `format_ceiling_gate` never threaded `loop_penalty` and targeted the `offrl` arm, while the shipped winner is
    the `bc` checkpoint. The 0.58/0.02 figures came only from the serial n=50 `behavior_probe`.
  - **Runs + small eval wiring, no `OBS_VERSION` bump / re-ingest / retrain.** (1) `loop_penalty` now threads
    through `arena.build_player` via a new `_LOOP_GUARD_AGENTS = {bc, offrl, ppo}` set (`LoopGuard(0)` is exact
    identity → every existing caller unchanged). (2) `format_ceiling_gate` gains `--bc-checkpoint` /
    `--loop-penalty`, a pure `_build_matchups(bc_ckpt, include_offrl_green)` helper appending a loop-fixed
    `bc_v11` M1 arm, and an OU FORMAT-vs-MODEL verdict in `assess_ou` that applies the Lever-15 `HEADROOM=0.58`
    threshold to the competent-bot reference (`simpleheuristics`). The stale RB `offrl_green` arm (checkpoint
    pinned to encoder **v2/760**, un-loadable since the **v5/761** bump) is dropped on OU — the `bc_v11` arm is the
    meaningful learned arm there.
  - **Result (M1, n=300, `bc_gen9ou_v11_s0` + `LoopGuard(4)`).** Harness clean: mirror **0.510** (sanity within
    `MIRROR_TOL`), gradient monotone **random 0.027 < maxbasepower 0.060 < simpleheuristics 0.620** [0.564, 0.673];
    band width **0.593 > RB 0.516**. A *simple competent bot* (simpleheuristics) clears the heuristic by
    **0.62 ≥ HEADROOM 0.58** — the very quantity that forced FORMAT_BOUND on RB (where even the strongest bot only
    tied the heuristic) shows **wide headroom** on OU ⇒ OU rewards skill, the format is **not** the ceiling →
    **`MODEL_BOUND`**. Our loop-fixed winner sits at **bc_v11 0.053** [0.033, 0.085] (consistent with the n=50
    behavior_probe 0.02), near the random/maxbasepower floor, **model_gap 0.567** below the competent bot: the ~0
    heuristic win is a *model* gap, not a format cap. (bc_v11 0.053 ≈ maxbasepower 0.060 — the loop-fixed policy
    plays roughly at "always max-base-power" level vs the heuristic.)
  - **Verdict: OU is MODEL_BOUND — the project posture flips.** OU has real, uncaptured headroom → **OU strength
    (PPO self-play / better BC data) is now the justified next build.** The interleaved ping-pong residual
    deprioritizes (won't lift the heuristic win). Suite **281 pass** (276 + 5), ruff + mypy(lategame) clean;
    `results/format_ceiling_gate_ou_v15.json` committed. **Open next:** OU strength push (Option C) — turn the
    heavier machinery on to close the 0.567 model gap; M2 (OU near-optimal search) / M3 (OU replays) remain
    deferred (the wide-band simpleheuristics evidence already rejects FORMAT_BOUND without them).

- **OU Pivot — Build 16: PPO self-play on OU — the first method to move OU vs-heuristic with CI-clean significance;
  `AMBER` (mechanism validated, gap dented not closed).**
  - **Motivation.** Build 15 → `MODEL_BOUND` named PPO self-play as the justified strength build. The RB PPO
    (Lever 10) was `AMBER` ("stable, no collapse, but doesn't beat the AWR ceiling"), but RB was *format-capped*
    (no headroom), so that flatness plausibly wouldn't transfer to OU. Build 16 is the honest test of whether
    on-policy PPO compounds past the demonstrator ceiling where headroom provably exists.
  - **Preflight re-plan (blocker caught before coding).** No existing OU offrl checkpoint was on the current
    encoder — the filename "v5" is a *build* number, not the encoder version (`offrl_gen9ou_v5_s0` = encoder
    **v3/760**, `_et_prior` = **v2/760**; live is **v5/761**). The offrl lineage never re-ran after the v6→v7
    (encoder v4→v5) bumps, and the v11 BC winner carries no *fitted* critic (warm-starting PPO from it = the M5
    random-critic collapse mode). Resolution (reuses built M3 machinery, a run not new code): retrained OU
    offline-RL on the v5/761 RL shard, BC-init from v11 → **`checkpoints/offrl_gen9ou_v7_s0.pt`** (val actor-acc
    0.648, value-MAE **0.53**, `v_min/v_max/n_bins` stamped → passes `run_ppo`'s guard).
  - **Wiring (runs + team/format plumbing, no `OBS_VERSION` bump / re-ingest / BC-retrain).** (1) `team` +
    `loop_penalty` thread through `data/rollout.py::collect_rollout` (learner + opponents). (2) `PPOConfig` gains
    `team_pool` / `loop_penalty`; `run_ppo` builds one shared `TeamPool` and adds a **format-consistency guard**
    (rollout uses the checkpoint's format, eval uses `config.battle_format` — mismatch now fails loudly);
    `_eval_point` threads both. (3) `ppo_continue_gate.py` gains `--team-pool` / `--loop-penalty` / `--ckpt-prefix`
    (non-clobbering `checkpoints/ppo_ou_et_prior_s{seed}/`). (4) `format_ceiling_gate.py` gains a dedicated
    **`offrl_ou`** learned arm (`_build_matchups(offrl_ckpt=…)`) and `assess_ou` computes `model_gap` against the
    strongest learned arm present (PPO `offrl_ou` over `bc_v11`).
  - **Loop-guard decided by smoke, not assumed.** At `loop_penalty=0` the `offrl` agent loops to the 1000-turn
    auto-tie and scores **0.000 vs random** (the pre-Build-14 pathology); at `loop_penalty=4` it's functional →
    **lp=4** (matches how `bc_v11` was scored). On-policy correctness: `PPORecordingAgent.choose_move` records
    `old_log_prob` from the *un-penalized* masked logits (the learner acts on its true policy), and the guard only
    keeps `offrl`/`ppo` **opponents/eval-arms** from stalling — so lp=4 is correct with no code change. The looping
    learner simply loses to a progressing opponent → PPO's reward teaches it to stop.
  - **Result — full gate (3 seeds × 10 iters, warm-start `offrl_gen9ou_v7_s0`, lp=4).** PPO **works**: `vs_iter0`
    **0.849 ± 0.017** (decisive self-improvement, no collapse) and `vs_random` climbs monotonically **0.40 → 0.78**;
    `best_vs_heuristic` **0.124 ± 0.046**. Authoritative M1 (n=300, best ckpt `ppo_ou_et_prior_s1/iter_10` +
    `LoopGuard(4)`) is harness-clean (mirror **0.480**, gradient monotone **random 0.010 < maxbasepower 0.080 <
    simpleheuristics 0.643**, band 0.633 > RB 0.516): **`offrl_ou` 0.133 [0.099, 0.176]** vs **`bc_v11`
    0.057 [0.036, 0.089]** — the CIs are **disjoint** (0.099 > 0.089), a statistically significant **~2.3×** gain
    over the BC winner. **`model_gap` 0.567 → 0.510.**
  - **Verdict: `AMBER` (positive) — the mechanism is validated, the magnitude is modest.** On-policy PPO self-play
    is the **first method to move OU vs-heuristic with CI-clean significance** (0.057 → 0.133), with decisive
    self-improvement and no collapse ⇒ **the RB AMBER did NOT transfer** (it was a format artifact). But 0.133 is
    still far below the competent bot 0.643 — the gap is **dented (~10%), not closed**. `MODEL_BOUND` reconfirmed.
    Suite **289 pass** (281 + 8), ruff + mypy(lategame) clean; `results/ppo_ou_gate_v16.json` +
    `results/format_ceiling_gate_ou_v16.json` + `checkpoints/offrl_gen9ou_v7_s0.pt` +
    `checkpoints/ppo_ou_et_prior_s{0,1,2}/`. **Open next (AMBER follow-ups):** the vs_random curve was still
    climbing at iter 10 and self-play ran on only a **12-team pool** — the prime ceiling suspect. Candidates:
    expand the team pool (`build_ou_teampool.py`), more PPO iters, a stronger/larger warm-start or more BC data.

- **OU Pivot — Build 17: extend PPO self-play iterations (10 → 25) — the run was cut short, not plateaued;
  extending it ~2.3×'d OU vs-heuristic. Stronger `AMBER`.**
  - **Motivation (which follow-up first).** Build 16 left three AMBER candidates (team pool / more iters /
    stronger warm-start). Re-reading the v16 curves settled the order: **every** metric was monotone-climbing at
    iter 10 and `best_iter` was the *final* iter for 2/3 seeds — the run never plateaued. You cannot diagnose a
    "12-team pool ceiling" or "weak warm-start ceiling" from a run that never plateaued (it confounds "the lever
    helped" with "we just trained longer"). So Build 17 isolates the cheapest, only directly-evidenced lever —
    **more iterations** — which also *tells us which* expensive lever to spend on next.
  - **Approach (runs-only, no code change).** Both gate scripts already expose every flag. A fresh 25-iter run
    from the **same** warm-start (`offrl_gen9ou_v7_s0`) + **same** 12-team pool + **lp=4** as Build 16 (isolates
    iterations as the one changed lever), seeds 0/1/2, `--eval-n 100` (per-iter eval is diagnostic; the n=300
    ladder/harness is authoritative), `--ckpt-prefix ppo_ou_long` (preserves the v16 `ppo_ou_et_prior_s*` dirs).
    Fresh restart (not a continue-from-iter10) keeps `vs_iter0` anchored to the true warm-start → apples-to-apples
    curves.
  - **Result — extended gate (3 seeds × 25 iters).** Every metric kept climbing well past iter 10 with **no
    collapse**: `best_vs_heuristic` **0.307 ± 0.065** (per-seed 0.23 / 0.30 / 0.39 at iters 19 / 25 / 22 — s1 peaks
    at the *final* iter, so the plateau is still not reached), `final_vs_iter0` **0.947 ± 0.012**, `vs_random` →
    0.91–0.98, `vs_simpleheuristics` → 0.25–0.26. Authoritative M1 (n=300, best ckpt `ppo_ou_long_s2/iter_22` +
    `LoopGuard(4)`) is harness-clean (mirror **0.473**, gradient monotone **random 0.013 < maxbasepower 0.047 <
    simpleheuristics 0.620**, band 0.607 > RB 0.516): **`offrl_ou` 0.303 [0.254, 0.358]** vs v16's
    **0.133 [0.099, 0.176]** — the CIs are **disjoint** (0.254 > 0.176), a real **~2.3×** gain from iterations
    alone, and **~5.3×** over the unchanged `bc_v11` 0.057 [0.036, 0.089]. **`model_gap` 0.510 → 0.317** (now
    **44%** below Build 15's 0.567).
  - **Verdict: stronger `AMBER` — the "still-climbing" outcome.** Iterations were the binding lever, not the pool:
    the same 12 teams and warm-start went from 0.133 → 0.303 vs heuristic purely on more training, and the curve is
    *still rising* at iter 25 (plateau not yet found). `MODEL_BOUND` reconfirmed; the gap is dented ~1/3 more but
    not closed. Suite **289 pass** (no code change this build), ruff + mypy(lategame) clean;
    `results/ppo_ou_gate_v17.json` + `results/format_ceiling_gate_ou_v17.json` + `checkpoints/ppo_ou_long_s{0,1,2}/`
    (gitignored). **Open next:** the plateau is *still* not reached (s1 best = final iter) → **extend iters again**
    is the cheapest, still-evidenced move (watch for the flatten); team-pool expansion (`build_ou_teampool.py`) and
    a stronger/larger warm-start become the diagnosable levers only *once* vs_heuristic flattens while vs_random
    stays high. Watch for late instability from the stale uniform league / fixed lr over longer runs.

- **OU Pivot — Build 18: extend PPO self-play iterations (25 → 50) — the curve PLATEAUED, and at a *higher* level.
  Outcome 1 (plateau found) + stronger `AMBER` in one build.**
  - **Motivation.** Build 17's pre-registered decision tree said extend iters once more to resolve the vs-heuristic
    asymptote (s1's best was still the final iter). Single-lever isolation, exactly as Build 17 isolated 10→25:
    same warm-start (`offrl_gen9ou_v7_s0`), same 12-team pool, same lp=4, same eval protocol — only `iters` 25→50,
    fresh `--ckpt-prefix ppo_ou_x50`. Runs-only, no code change.
  - **Infra recovery (per-seed chunking).** The full 3-seed run kept dying at session teardowns, which reap the whole
    process tree **and wipe scratchpad** — even `start_new_session`/setsid-detached daemons (PPID→launchd) did **not**
    survive. Only the repo disk persists (the per-iter `iter_XX.pt` + `curve.json` the gate writes each iteration). So
    the run was completed **one seed at a time** (each a standalone `ppo_continue_gate --seeds N`, identical config,
    throwaway `--ladder-n 20`): seed 0 was salvaged complete from the first attempt; seeds 1,2 re-run standalone. This
    bounds teardown loss to a single partial seed. Consolidated into `results/ppo_ou_gate_v18.json` (provenance noted
    in the file); the authoritative n=300 eval is a single clean `format_ceiling_gate` run.
  - **Result — UNANIMOUS PLATEAU (all 3 seeds).** `best_iter` = **41 / 44 / 46** — every seed peaks in the low-40s,
    *interior* (vs v17's 19/25/22 with s1 at the final iter). Each tail (iters 35–50) is **flat within eval noise and
    *above* the mid-run (20–34) mean** (s0 0.475 vs 0.415; s1 0.396 vs 0.269; s2 0.377 vs 0.325 — climb-then-flatten,
    not decline), and `vs_iter0` never drops below **0.90** (final mean **0.977**) → **no collapse, the *destabilized*
    branch of the tree is ruled out** (no fixed-lr / stale-league instability at 50 iters). `best_vs_heuristic` per-seed
    **0.503 ± 0.048** (0.57 / 0.46 / 0.48) — *higher* than v17's 0.307, so 25→50 both **lifted** the peak and **revealed**
    the ceiling.
  - **Authoritative M1 (n=300, best ckpt `ppo_ou_x50_s0/iter_41` + `LoopGuard(4)`, harness clean — mirror 0.520,
    monotone gradient random 0.013 < maxbasepower 0.073 < simpleheuristics 0.633, band 0.62 > RB 0.516):**
    **`offrl_ou` 0.453 [0.398, 0.510]** vs v17's **0.303 [0.254, 0.358]** — **disjoint CIs** (0.398 > 0.358), a real
    **~1.5×** gain from the extra iterations; `bc_v11` unchanged **0.043 [0.025, 0.073]**. **`model_gap` 0.317 → 0.18**
    (43% below v17, **68% below Build 15's 0.567**) — PPO is now near parity with the competent heuristic (0.633).
    `MODEL_BOUND` reconfirmed, `format_bound_rejected`.
  - **Verdict: PLATEAU FOUND — the iterations lever is exhausted at 50, and it paid off (0.303 → 0.453, gap halved).**
    Suite **289 pass** (no code change), ruff + mypy clean; `results/ppo_ou_gate_v18.json` +
    `results/format_ceiling_gate_ou_v18.json`; `checkpoints/ppo_ou_x50_s{0,1,2}/` gitignored. **Open next:** the plateau
    finally makes the *expensive* levers diagnosable — **team-pool expansion** (`build_ou_teampool.py`, 12 → ~24) and a
    **stronger/larger warm-start** are now the evidenced Build-19 candidates (each isolated). The *destabilized* branch
    did not trigger, so lr-decay / PFSP-league hardening is **not** indicated. More iterations is retired (curve flat).

- **OU Pivot — Build 19: the PPO train/eval objective mismatch — MEASURED, FIXED, and *ruled out* as the plateau's
  cause. `NULL` on strength; the schedule lever is RETIRED.**
  - **Re-reading v18 overrode its own decision tree.** Build 18 pre-registered *team-pool* or *warm-start*. But the
    v18 curves show the agent **trains directly against `simpleheuristics`** (it is the PPO anchor, `ppo.py`
    `anchors=("simpleheuristics",)`, injected every iteration) **and after 50 iters still wins only 26–45% against
    it** (final per-seed 0.45 / 0.34 / 0.26). A *fixed, scripted, non-adapting* opponent already in the training mix
    is not beaten by more opponent variety ⇒ the plateau is in the **learner**, not the opponents. That **refutes
    team-pool expansion as the binding lever** (and it is unmeasurable as specified anyway: the same pool feeds
    rollouts *and* `_eval_point`, so expanding it moves the metric out from under v18's CI — it is a *generalization*
    question needing a held-out `--eval-team-pool` first). **Capacity** is also weaker than assumed: the net is
    already over-parameterized for imitation (**0.72M params on 61,723 BC rows** = 0.086 rows/param; d256/4L would be
    4.56M = 0.014), so the 0.63 BC gate would likely reject a bigger teacher for *overfitting* — a false negative that
    says nothing about what PPO (unbounded self-play data) wants. Lever chosen: the **PPO optimization schedule**
    (`ent_coef`/`lr` were **fixed for all 50 iters** and not exposed by the gate) — cheapest, reuses the v7 warm-start,
    no BC/offrl retrain, no `OBS_VERSION` bump, no re-ingest.
  - **Stage A — the probe that qualified the spend (`scripts/policy_sharpness_diag.py`, new, committed).** Reading the
    code **falsified the naive story before it cost a run**: **eval is already greedy** (`_eval_point` builds the
    learner `sample=False` → argmax) while **rollout samples** (`sample=True`; `PPORecordingAgent`), so residual
    entropy **cannot directly cost eval win-rate**. It can only hurt *indirectly* — by holding the distribution soft so
    the **argmax lags** the distribution PPO optimizes. Probe (v18 best ckpt, 1266 frozen live states from a fresh
    `behavior_probe --obs-out` capture + a greedy/sampled win-rate A/B at n=300): **(A1)** entropy sharpens early then
    **stalls exactly when the win-rate stalls** — `h_ratio` (mean H / mean uniform-over-legal H) 0.571 (warm-start) →
    0.493 (it10) → 0.472 (it25) → **0.465 (it41)**, `max_prob` flat at 0.682 from iter 10, only 27% of decisions
    near-deterministic. **(A2)** the mismatch is real and large: **greedy 0.487 vs sampled 0.347 → gap +0.140**. PPO
    was maximizing the return of a distribution that plays **14 points worse** than the one we deploy. → **LIVE.**
  - **Stage B — the schedule (code, back-compatible).** `PPOConfig.ent_coef_final` / `lr_final` (`None` ⇒ constant ⇒
    **bit-identical to Build 16–18**), a pure `anneal(start, final, k, iters)`, per-iteration `optimizer.param_groups`
    lr + `replace(config, ent_coef=…)` into `ppo_update`, and both values echoed into the per-iter stats so a run is
    auditable. `--ent-coef/--ent-coef-final/--lr/--lr-final` on `ppo_continue_gate`. Smoke proved both directions:
    scheduled → ent 0.0100/0.0050/0.0000 + lr 2.50e-4/1.50e-4/5.00e-5 (exact endpoints); flags omitted → constant.
    Run identical to v18 in every other respect (init `offrl_gen9ou_v7_s0`, 12-team pool, lp=4, 50 iters, 3 seeds,
    `games_per_opp` 16, `eval_n` 100), schedule `ent_coef 0.01 → 0.0`, `lr 2.5e-4 → 5e-5`.
  - **Result — the mechanism ENGAGED and the metric DID NOT MOVE.** Sharpening worked: `h_ratio` **0.465 → 0.336**
    (0.512 @10 → 0.370 @30 → 0.336 @50 — it keeps falling instead of stalling), `max_prob` 0.682 → **0.788**,
    near-deterministic decisions 27% → **44%**. And the objective mismatch **closed to zero**: on the scheduled policy
    **greedy 0.433 vs sampled 0.437 → gap −0.003** (was **+0.140**). But strength is flat: 3-seed `best_vs_heuristic`
    **0.493 ± 0.063** vs v18's 0.503 ± 0.048, and **authoritative M1 (n=300, best ckpt `ppo_ou_sched_s2/iter_49` +
    `LoopGuard(4)`, harness clean — mirror 0.487, gradient 0.030 < 0.067 < 0.643, band 0.613 > RB 0.516):**
    **`offrl_ou` 0.490 [0.434, 0.546]** vs v18's **0.453 [0.398, 0.510]** — **CIs overlap heavily** (pre-registration
    demanded *disjoint above 0.510*) ⇒ **NULL**. `model_gap` 0.18 → 0.153 (not significant). `bc_v11` 0.040.
    `MODEL_BOUND` reconfirmed. Sanity guards clean (`vs_iter0` ≥ 0.96, `vs_random` ≥ 0.97 — the lr decay did **not**
    destabilize).
  - **What the null actually teaches (the load-bearing correction).** The +0.140 gap was **the cost of *sampling*, not
    headroom in the *argmax***. Annealing entropy pulled the **sampled** policy **up to** the argmax (0.347 → 0.437,
    +0.09) — it did **not** push the **argmax** higher (0.487 → ~0.45–0.49, flat), and **the argmax is what we score**.
    PPO was already extracting the argmax's value; the entropy bonus was taxing only the *rollout* policy, a train-time
    cost eval never paid. **So the objective mismatch was real, is now fixed, and is NOT the plateau's cause.**
  - **Verdict: `NULL` — the schedule lever is RETIRED.** Keep the schedule as the OU default anyway (it costs nothing —
    0.490 vs 0.453, if anything nominally higher — and it removes a real confound, making future levers cleaner to
    read), but claim **no strength win** from it. The flags stay **opt-in** (`*_final=None`), so RB and every prior
    build are untouched. Suite **296 pass** (289 + 7 new), ruff + mypy clean; `scripts/policy_sharpness_diag.py`,
    `results/ppo_ou_gate_v19.json`, `results/format_ceiling_gate_ou_v19.json`,
    `results/policy_sharpness_diag_v19{,_sched}.json`; `checkpoints/ppo_ou_sched_s{0,1,2}/` gitignored.
  - **Open next (Build 20), now that entropy/lr AND opponent-variety are both eliminated.** The learner-side suspects,
    in cost order: **(1) per-iteration sample budget** — `games_per_opp=16` × (pop_size 2 + 1 anchor) = **48 battles ≈
    ~2K transitions ≈ ~32 gradient steps per iteration**, on a *normalized* advantage estimate; ~2,400 battles for the
    whole run. That is a very small RL budget and a plausible **noise-floor** plateau. `--games-per-opp 48` is already a
    flag ⇒ **zero code change**, ~3× wall-clock. **(2) capacity** — a bigger warm-start; `factory.py` already reads
    `d_model/n_layers/n_heads/ff_dim` from `arch` and `train-rl --bc-init` auto-inherits it, so the only gap is that
    `bc.py::_build_model` never populates them (4 `TrainConfig` fields + 4 CLI flags). Team-pool stays **deprioritized**
    (refuted as binding). **Methodology note for every future build: `best_vs_heuristic` is a max over 50 noisy n=100
    evals — an optimistically biased statistic (winner's curse; v18 shrank 0.503 → 0.453 at n=300). Compare builds on
    the authoritative n=300 CI, never on `best_vs_heuristic`.**

- **OU Pivot — Build 20: the per-iteration sample budget — the plateau is a STATIONARY POINT, not a sampling-noise
  floor. `NULL` on strength; the whole optimization/sampling family of levers is RETIRED.**
  - **Lever:** 3× the rollout, `--games-per-opp 16 → 48` (already a flag ⇒ **zero code change**), everything else
    byte-identical to v19. Reading the code first corrected two things. **(a) The cost model was wrong in our favour:**
    each iteration plays **48 rollout battles but 400 eval battles** (`_eval_point` runs `eval_n=100` vs 3 baselines
    *plus* iter0), so **~89% of every PPO run's battle budget is measurement, not learning** (2,384 rollout vs 20,300
    eval battles over 50 iters) and tripling the rollout costs **~+21% battles**, not the 3× wall-clock v19 assumed.
    **(b) "More gradient steps" was the wrong mechanism:** advantages are normalized **per-buffer** (`ppo.py:169`) and
    the epoch loop **KL-early-stops** (`ppo.py:212`), so per-iteration *displacement* is governed by the trust region,
    not the step count. A bigger buffer buys a **lower-variance estimate of the gradient direction** — nothing else.
  - **Stage A — the probe reframed the build before it was paid for** (`scripts/grad_noise_diag.py`, new, committed).
    On shipped v19 checkpoints it takes the gradient at **θ_old** (PPO ratio 1, clip inactive ⇒ the vanilla policy
    gradient — the first step the iteration would take) and compares it across **independent rollouts**.
    **Result: the noise is CONSTANT and the signal VANISHES.** At iters 10 / 47 / 50: `tr(Σ)` (noise) **3284 → 2741 →
    3295** (flat), `|G|²` (signal) **4.54 → 0.93 → −0.41** — a *negative* estimate, i.e. indistinguishable from zero;
    `B_simple` **723 → 2945 → ∞**, exploding only because it is the **ratio**. The plateau lands exactly where the
    ~1.6K buffer crosses the noise scale. Verdict `NOISE_LIMITED` (pre-registered ⇒ run Stage B) — **but flagged at the
    time as the Build-19 trap in a new costume: `B_simple → ∞` is the signature of a growing noise floor AND of a
    vanishing gradient (a stationary point). A bigger batch estimates a near-zero gradient more precisely; it cannot
    manufacture one.** Prior on a win lowered *before* the seeds ran.
  - **Two method fixes, both load-bearing** (each caught by its own failing test): **(1)** compare **independent
    rollouts**, never two halves of one buffer — halves share that rollout's league/team/episode draw, so they agree
    more than two real PPO iterations do; the bias runs *toward agreement* and could have manufactured a false
    `SIGNAL_LIMITED` and wrongly cancelled Stage B. **(2) Two arms**, because `--games-per-opp` only buys battles
    against the mix an iteration **already drew**: `same_mix` (league pinned) decides the verdict; `fresh_mix` isolates
    opponent-**selection** variance that no per-opponent budget can touch. Measured `opponent_draw_dominates=False`
    everywhere ⇒ that alternative is refuted and the budget is the right knob.
  - **Stage B — the lever landed and the metric did not move.** Telemetry is unambiguous: gradient steps/iter **24 →
    71** (2.96×), `epochs` held at **4 in 40/40** late iters, `approx_kl` 0.008 → 0.014 against a 0.045 bar ⇒ **the
    trust region never bound** (so a NULL cannot be blamed on it), and the critic even fit better (`vmae` 1.44 → 1.09).
    3-seed `best_vs_heuristic` **0.493 → 0.567 ± 0.029** (per-seed 0.570/0.600/0.530, best iters 46/47/45 all interior
    ⇒ plateaued) — and it was **almost entirely winner's curse**.
  - **The MEASUREMENT itself had to be fixed** (`scripts/seed_strength_gate.py`, new, committed). The authoritative
    protocol used through Build 19 — score the **single best checkpoint** at n=300 — is **not fit for build-vs-build**:
    **UNDERPOWERED** (a difference has SE ≈ 0.041 ⇒ resolves only gaps **> 0.08**; the candidate effect was 0.074 —
    *under its own detection floor*) and **SELECTION-BIASED** (that checkpoint is the argmax over ~150 noisy curve
    evals, then re-scored ⇒ regression to the mean, and build-dependent: v20's best fell **0.600 → 0.480**, v19's only
    0.493 → 0.490). Fix, applied **symmetrically to both builds**: score **every seed's best** checkpoint and pool
    (SE 0.041 → **0.024**, resolving +0.07 at z ≈ 3), and read a **z-test** alongside CI-disjointness — **CI overlap is
    a CONSERVATIVE test** (at +0.05 over 900/arm the intervals overlap while p = 0.034). Per-seed "best iter" is still a
    max over 50 evals, so absolute rates stay optimistic; the protocol is identical across builds, so that bias
    **cancels in the DIFFERENCE**, which is the quantity under test.
  - **Corrected verdict: v19 `0.448 [0.416, 0.480]` → v20 `0.472 [0.440, 0.505]`; diff +0.024, z = 1.04, p = 0.30 ⇒
    NULL.** (The old single-ckpt gate said 0.490 → 0.480, p = 0.81 — same call, but it could not have seen the effect
    either way.) Harness clean (mirror 0.493, gradient 0.013 < 0.073 < 0.613, band 0.600 > RB 0.516); MODEL_BOUND
    reconfirmed.
  - **What the null teaches — the part worth keeping: 3× the samples collapsed seed-to-seed variance ~7× (std 0.074 →
    0.010) WITHOUT moving the mean.** That is exactly the signature of estimating a **near-zero** gradient more
    precisely: a far more *reproducible* policy that converges to the same place. Stage A's reframing was right — **the
    plateau is a stationary point of the PPO objective, not a sampling-noise floor.** With Build 19 (entropy/lr, NULL)
    and opponent-variety (refuted), the **entire optimization/sampling family is now exhausted**; the binding constraint
    is the **model class**. Suite **328 pass** (296 + 25 + 7), ruff + mypy clean;
    `results/{grad_noise_diag,ppo_ou_gate,format_ceiling_gate_ou,seed_strength_gate}_v20.json`;
    `checkpoints/ppo_ou_budget_s{0,1,2}/` gitignored.
  - **Open next (Build 21) — CAPACITY, now the sole indicated lever.** A bigger model changes the loss landscape and
    can carry a nonzero gradient where the current **0.72M-param** net has none. **Zero-code path:** `train-rl --model
    entity_transformer --d-model 256 --n-layers 4 --n-heads 8 --id-embed-init prior` trains a wide net **from scratch**
    on the 120K-row RL shard; PPO's `build_model(ckpt)` reads `arch` and fine-tunes **all** of it (`ppo.py:346-348`).
    This **bypasses BC's 0.63 val-acc gate entirely** — which matters, because at 0.086 rows/param that gate would
    likely reject a bigger teacher for *overfitting*: a false negative w.r.t. what PPO wants. **Confound:** from-scratch
    offline-RL loses the BC warm-start, so the clean version still needs the 4 edits to `bc.py`/`cli.py` (`d_model`/
    `n_layers`/`n_heads`/`ff_dim` → `TrainConfig` → `_build_model`'s `arch` dict → 4 CLI flags). **Footgun found:**
    `--d-model` is **silently ignored** when `--bc-init` is passed (`offline_rl.py:215-223` overwrites `model_meta` from
    the checkpoint's `arch`; it must, or the strict `load_state_dict` would explode) — worth a guard. **Secondary
    lever:** the **critic** (EV ≈ 0.30 — a better critic shrinks `tr(Σ)` directly, with **no extra samples**). Reuse
    `grad_noise_diag.py` as a cheap **pre-filter**: a candidate that fixes the plateau must show `|G|²` *recovering*.
    **Methodology, updated: compare builds with `seed_strength_gate.py` (pooled seed-bests + z-test). The
    single-checkpoint n=300 gate is RETIRED for build-vs-build — it cannot see effects below ~0.08.**

- **Build 21 — CAPACITY (0.72M → 4.56M params): NULL, and a REGRESSION. The model-side story is CLOSED.**
  - **Lever:** warm-start widened to a 4.56M-param entity transformer (`--ff-dim` exposed, Stage A0), PPO self-play
    otherwise **identical to v20** (50 iters × 3 seeds, `games_per_opp` 48, ent/lr schedules unchanged).
  - **Verdict (`seed_strength_gate.py`, the authoritative protocol): v20 `0.499 [0.466, 0.531]` → v21
    `0.452 [0.420, 0.485]`; diff **−0.047**, z = −1.98, p = 0.047 ⇒ NULL.** 6.3× the parameters made the agent
    **slightly worse**, not better. (Note v20's arm re-scored at 0.499 here vs 0.472 in its own build — same
    checkpoints, same protocol, a 0.027 swing between independent n=900 evals. **Only the within-run difference is
    trustworthy**; this gate's absolute rates move.)
  - **Trust region CERTIFIED CLEAN, so the NULL is unconfounded** (`scripts/ppo_telemetry.py`,
    `results/ppo_ou_telemetry_v{19,20,21}.json`): bound in **4/150 iters** (s0 [1], s1 [1,31], s2 [1]), 96–98%
    full-epoch, `approx_kl` max **0.0675** vs **v20's 0.1602**. **Every build in this lineage binds at iteration 1** —
    v19 [1,2]/[1,2]/[1], v20 [1]/[1]/[1] — so the opening KL overshoot is the pipeline's standard transient, **not** a
    capacity artifact. The wide net takes **more consistent** steps, not bigger ones.
  - **The explainer, and the whole point of the build (`grad_noise_diag_v21.json`, gpo=48/rollouts 6/splits 5 to match
    v20): `|G|²` does NOT recover — it was NEVER THERE.** Same-mix, by iterate:

    | | `\|G\|²` (signal) | `tr(Σ)` (noise) | `B_simple` | cos |
    |---|---|---|---|---|
    | v20 (0.72M) iter_10 | **4.540** | 3284 | 723 *(< budget)* | +0.585 |
    | v20 iter_47 → iter_50 | 0.931 → −0.408 | 2741 → 3295 | 2945 → ∞ | +0.053 → −0.037 |
    | v21 (4.56M) iter_10 | **0.057** | **580** | 10188 | +0.429 |
    | v21 iter_40 → iter_50 | 0.004 → −0.109 | 1698 → 2642 | 395113 → ∞ | +0.017 → +0.143 |

    The 0.72M net had **real gradient signal early (4.540) and lost it**. The 4.56M net **enters PPO already at a
    stationary point** — its iter_10 `|G|²` (0.057) is where the narrow net *ended up* after fully converging. §13.1's
    pre-registered pre-filter ("a candidate that fixes the plateau must show `|G|²` **recovering**") is **failed
    outright**. `opponent_draw_dominates: False` — the noise is not the league draw.
  - **TRAP, and it cost a wrong read mid-build: `B_simple` is a RATIO (`tr(Σ)/|G|²`). v21's is huge (10k–395k vs a
    ~4–6k budget), and the script's own `NOISE_LIMITED` verdict duly says "run with more battles."** That is **WRONG
    here**: `B_simple` exploded because the **denominator collapsed**, not because noise grew — v21's `tr(Σ)` is
    *lower* than v20's (580 vs 3284). **More samples buy nothing against a vanished gradient.** This is the *same*
    ratio-as-headroom trap the project already booked; `grad_noise_diag`'s verdict answers **Build 20's** question and
    must not be read as the capacity finding. (Caveats: absolute `|G|²` is **not** strictly comparable across
    architectures — quote the scale-free `cos`/`B_simple`; and `cos` at iter_10 read 0.182 on a first, killed probe vs
    0.429 on the rerun — **the cos estimate is itself noisy across probe runs**.)
  - **Banked regardless of the verdict.** (1) The wide net is a **much better critic** — offline value MAE 0.531 →
    0.370, in-run `vmae` 0.78–0.84 vs v20's 1.20–1.31. It still **degrades** over training (→ 1.12–1.22), just from a
    far better start; the "wide critic holds while others degrade" reading was an artifact of the first 13 iters.
    (2) **The BC warm-start contributes NOTHING** — a from-scratch control matched it at **0.6485 accuracy exactly**
    (`stage_a_confound_v21.json`), retiring a whole pipeline stage as decorative.
  - **Correction to the record:** commit `f77acd7`'s message and `ppo_telemetry.py`'s docstring assert v20's telemetry
    was *unrecoverable*. **False** — `results/ppo_ou_budget_s{0,1,2}.log` were on disk the whole time.
  - **Open next (Build 22) — CRITIC BIAS, per the pre-registered exit.** Capacity is refuted as the lever: it did not
    restore `|G|²`, and it *cost* 0.047. With optimization/sampling (Builds 19–20) and now the model class exhausted,
    the remaining model-side suspect is the **value function biasing the advantage** — EV ≈ 0.23–0.33 in-probe. The
    wide net proves a **better critic is achievable** (MAE 0.370) *without* improving the policy gradient, which is
    itself the clue: the critic is fit well but the **advantage estimator** is what feeds `|G|²`.
  - **Infra:** this workload is **memory-bound, not compute-bound** — the probe drove a 16 GB M2 Pro to 14.8 GB swap
    and stalled 48 min in `UN` (uninterruptible I/O wait) on one gradient phase, at ~2 of 10 cores. Build 22 belongs on
    a **high-RAM CPU cluster node (no GPU** — the net is 4.56M params**)**, where seeds run in parallel rather than
    sequentially and a sleeping laptop cannot reap the job (which killed one probe run and its unwritten JSON).

- **Build 22 — PRE-REGISTERED (written 2026-07-13, BEFORE running). The exit "critic bias" is WITHDRAWN.**
  - **Why the pre-registered exit is dead.** Build 21 slated Build 22 as *critic bias*, on the theory that "a better
    critic shrinks `tr(Σ)` directly, with no extra samples." **Build 21 incidentally ran that experiment and refuted
    both halves.** Its wide net *is* a much better critic (offline value MAE 0.531 → 0.370), its `tr(Σ)` *did* fall as
    predicted (**3284 → 580, 5.7×**) — **and the policy got WORSE** (−0.047). Worse for the theory: the critic's
    **explained variance is FLAT across the two builds** (EV@iter_10 **0.112** for v20 vs **0.074** for v21; EV@late
    0.27–0.30 vs 0.24–0.25) while `|G|²` moves **~80×**. **EV does not predict `|G|²`** — it is the one variable that
    did *not* change between the build that worked and the build that didn't. Spending Build 22 there would be
    spending it on the known-non-discriminating variable.
  - **What Build 21's data actually says.** Nothing measurable differs except capacity and the warm start. The wide net
    is **not stuck**: entropy@10 **1.071** (vs 1.064 — *higher*), `|pi_loss|`@10 **0.0215** (vs 0.0122 — *larger*),
    return span 14.5 (vs 13.9 — same), KL normal. **It takes full-sized steps in NOISE directions** (cos ≈ 0,
    `|G|²` ≈ 0) and random-walks away from a good initialization — which is *why* it regressed rather than merely
    plateauing.
  - **HYPOTHESIS (H22): the OFFLINE warm start manufactures the stationary point.** The wide net enters PPO already
    flat — `|G|²` = 0.057 at iter_10, where the narrow net was at 4.540 (its own iter_50 value). The one thing that
    improved dramatically is the *offline fit*. H22: **fitting the offline objective harder lands the policy in a
    region that is flat under the ON-POLICY objective** — better offline, no gradient left to improve from. Consistent
    with the BC ablation (warm start contributes **nothing** to final strength, 0.6485 either way): the offline stage
    may be **actively harmful**, not merely decorative.
    **HONESTY FLAG: H22 is POST-HOC** — derived from Build 21's data after the fact. It is a hypothesis, not a result,
    and is pre-registered here *before* the test precisely so it cannot be quietly reshaped into one (HARKing).
  - **STAGE A — the cheap discriminator (~1–1.5 h, run FIRST; a full build is NOT authorized until it passes).**
    Probe `|G|²` at **iter_0 — the warm start itself, before any PPO** — for both nets, at `--games-per-opp 48` (the
    budget both v20 and v21 actually trained at), `--rollouts 6 --splits 5`. `_league_for` handles `k=0`: the league is
    `[init]` + anchors, exactly the mix iteration 0 faced.

    ```
    scripts/grad_noise_diag.py --policy checkpoints/offrl_gen9ou_wide_s0.pt \
        --init checkpoints/offrl_gen9ou_wide_s0.pt --games-per-opp 48 --rollouts 6 --splits 5
    scripts/grad_noise_diag.py --policy checkpoints/offrl_gen9ou_v7_s0.pt   \
        --init checkpoints/offrl_gen9ou_v7_s0.pt   --games-per-opp 48 --rollouts 6 --splits 5
    ```

    **Read the WITHIN-architecture trend, not the cross-architecture ratio** — absolute `|G|²` is *not* comparable
    across parameterizations (gradient norms scale with width); only `cos` and `B_simple` are scale-free. Pre-registered
    outcomes:

    | Stage A result | reading | Build 22 becomes |
    |---|---|---|
    | wide `\|G\|²`(0) ≈ 0.057 (its iter_10 value) | **BORN FLAT** — H22 CONFIRMED; the offline stage creates it | weaken/shorten offline training, or PPO from a less-converged init (BC already shown unnecessary) |
    | wide `\|G\|²`(0) ≫ 0.057 | signal **COLLAPSED during PPO iters 1–10** — H22 REFUTED; a PPO×capacity interaction | probe iters 1,2,3,5 to localize the death; H22 is retired |
    | narrow `\|G\|²`(0) ≉ its 4.540 @ iter_10 | the probe is **not measuring what we think** at k=0 | fix the instrument before trusting either arm |

    The narrow arm is the **control**: it must show real signal at iter_0 (it had 4.540 by iter_10). If it does not,
    Stage A is uninterpretable and nothing else may be concluded from it.
  - **Infra prerequisite.** Run Stage A on **UMIACS**, not the laptop — see `scripts/cluster/`. High-RAM CPU nodes, **no
    GPU** (4.56M params; the GPU is irrelevant and only lengthens the queue). The binding constraints measured in Build
    21 are **RAM** (16 GB → 14.8 GB swap → a 48-min `UN` stall on one gradient phase) and **job durability** (a sleeping
    laptop reaped a probe and its unwritten JSON — `grad_noise_diag` serializes only at the end).
  - **STAGE A — RESULT (ran on UMIACS 2026-07-20, s0 per the spec). H22 CONFIRMED: the wide net is BORN FLAT.**
    `same_mix.policy.noise_scale.g_norm_sq` at **iter_0 — the warm start itself, before a single PPO step**
    (`results/grad_noise_diag_b22_stageA_{narrow,wide}.json`, `--games-per-opp 48 --rollouts 6 --splits 5`):

    | arm | `\|G\|²`(0) iter_0 | (Build 21 iter_10) | cos@budget | tr(Σ) |
    |---|---|---|---|---|
    | narrow (control, v7) | 3.014 | 4.540 | 0.451 | 8825 |
    | wide (under test) | 0.064 | 0.057 | 0.186 | 1701 |

    - **Control PASSES — the instrument is sound.** The narrow net shows real, resolvable signal at the warm
      start: `|G|²`(0) = 3.014, cos@budget 0.451 (well above the 0.3 noise floor — independent rollouts genuinely
      agree in direction), the same real-signal regime as its own iter_10 (4.540). The pre-registered requirement
      ("must show real signal at iter_0") is met, so Stage A is interpretable.
    - **Wide is BORN FLAT** (within-architecture trend, per the constraint — no cross-arch ratio): `|G|²`(0) =
      0.064 ≈ its Build-21 iter_10 value 0.057 — **not ≫ it**. The wide net enters PPO already at the stationary
      point it later sits in; PPO iters 1–10 did not manufacture the flatness. Scale-free corroboration: cos@budget
      0.186 is *below* the 0.3 floor — the iter_0 gradient direction is noise-dominated, exactly what a vanished
      `|G|²` looks like (contrast narrow's 0.451). This is the pre-registered **BORN FLAT → H22 CONFIRMED** row.
      `opponent_draw_dominates: False` in both arms — not the league draw.
    - **Ignore the scripts' self-verdicts** (narrow `AMBIGUOUS`, wide `NOISE_LIMITED`): that is `grad_noise_diag`'s
      Build-20 sample-budget logic, which §13.1 says must not be read as the capacity finding. Wide's
      `NOISE_LIMITED` is the `B_simple` ratio trap — it exploded only because the denominator `|G|²` collapsed to
      0.064, while its `tr(Σ)` = 1701 is *lower* than narrow's 8825: **noise did not grow, signal vanished**.
      "Collect more samples" is exactly wrong against a vanished gradient.
    - **HONESTY.** H22 was flagged POST-HOC (from Build 21's data); this Stage A is the pre-registered test that
      confirms it, so the discipline held. Clean result, but a **single seed per arm (s0)**, as Stage A specified.
    - **Triggers (pre-registered):** Build 22 becomes **weaken/shorten offline training / PPO from a less-converged
      init** — BC already shown unnecessary (0.6485 either way), so the offline stage may be *actively harmful*.
  - **STAGE B — PRE-REGISTERED (written 2026-07-20, BEFORE running). Reduced-epoch offline dose-response,
    cheap-probe-gated.** Claim: the wide net is born flat *because* offline AWR is fit to convergence; a
    less-converged wide init should (i) still have real `|G|²` at iter_0 and (ii) let PPO improve. Wide arch read
    from the checkpoint (do **not** guess): `entity_transformer d_model=256 n_layers=4 n_heads=8 ff_dim=512
    id_embed_init=prior n_bins=51`, support pinned `v_min=-9.5695 v_max=10.2248`. `train_offline_rl` saves
    best-by-val-loss (default 30 epochs), so a reduced `--epochs` caps convergence depth.
    - **Blocking prereq:** stage the *same* `gen9ou` RL shard that built `offrl_gen9ou_wide_s0.pt` (`data/` is empty
      on the current node); a different shard confounds the comparison.
    - **B-0 (cheap qualifier, ~1–1.5 h): retrain wide @ `--epochs {3,10}`, then probe `|G|²` at iter_0** on each
      (Stage-A protocol). Ignore the script's `NOISE_LIMITED`/`AMBIGUOUS` verdict; quote scale-free `cos` + the
      within-arch `|G|²` trend only. Gate:

      | B-0 result (vs converged wide's 0.064) | reading | action |
      |---|---|---|
      | weakened `\|G\|²`(0) ≫ 0.064 AND cos > 0.3 | offline fit is what flattens it — mechanism **ACTIONABLE** | proceed to B (PPO) |
      | weakened `\|G\|²`(0) ≈ 0.064, cos < 0.3 | offline convergence is **not** the flattener | **CANCEL PPO**; the flat point is architectural — revisit |

    - **B (gated, only if B-0 passes): `BUILD=v22 INIT=<weakened> sbatch --array=0-2 scripts/cluster/ppo_seed.slurm`**
      (~45 min/seed ×3), then `ppo_telemetry` (certify the trust region did not bind) → `merge_gate_seeds` (pool 3
      seeds; never hand a single-seed file to the strength gate) → `seed_strength_gate` vs the converged wide **v21
      (0.452)** — same arch, only offline convergence differs, so it isolates H22 — and vs the narrow champion
      (v20/v7 ~0.499) for the north-star win-rate. Gate:

      | strength v22 vs v21 | conclusion |
      |---|---|
      | v22 > v21 (significant) | offline over-fit was **actively harmful**; weakening it is the first lever that moved win-rate — set Build 23 dose direction |
      | v22 ≈ v21 (NULL) | flatness is real but **not** the binding constraint on strength; offline convergence is a red herring for win-rate — retire the lever, pivot |

  - **STAGE B-0 — RESULT (ran on UMIACS 2026-07-26, job 7141999, one array task per dose).** The
    shard was staged and **fingerprint-verified** before training: `train_offline_rl` derives the value
    support from the shard's own returns and stamps it into the checkpoint, so `v_min`/`v_max` identify
    the training data. `data/gen9ou_v7_rl.npz` reproduced the wide net's stamp to **delta 0.000e+00**
    (119,996 rows, `gen9ou`) — same shard, so the dose-response is not confounded by data.
    Arch was **read** from the checkpoint per the pre-registration (`d_model=256 n_layers=4 n_heads=8
    ff_dim=512 n_bins=51`), and `--seed 0` made the three arms **bit-identical trajectories truncated at
    different depths** (task 2's epoch-10 val line matched task 1's saved best to every printed digit,
    from a separate process on a separate node).

    | arm | offline fit (acc / vMAE) | `\|G\|²`(0) | cos | tr(Σ) | budget | EV |
    |---|---|---|---|---|---|---|
    | narrow ctrl (Stage A) | — | 3.014 | 0.451 | 8825 | 5244 | −0.010 |
    | wide converged (ref) | 0.635 / 0.370 | 0.064 | 0.186 | 1701 | 7462 | 0.166 |
    | wide e3 | 0.287 / 1.910 | **unresolvable** | 0.186 | 251 | 3591 | −0.022 |
    | wide e10 | 0.613 / 0.703 | 0.142 | **0.312** | 1514 | 8226 | 0.133 |
    | wide e30 (on-node) | 0.646 / 0.322 | 0.076 | 0.185 | 954 | 7369 | 0.135 |

    - **e3 is uninterpretable, by its own numbers.** `|G|²` came back **≤ 0** — which the McCandlish
      two-batch estimator returns when noise swamps the signal (`grad_noise_diag.py:146-171`), NOT a
      negative gradient. But `tr(Σ)` fell 1701 → 251 and **EV went to −0.022**: the critic explains less
      than predicting the mean, so its advantages are noise and the small gradient follows from a broken
      critic rather than from the landscape. Signal *and* noise collapsed together. At acc 0.287 this arm
      is barely trained; it fails the gate under H22 and under its negation alike, so it discriminates
      nothing. **The dose was set too aggressively.**
    - **e10 passed the gate as written** — `|G|²` 0.142 (2.2× ref) and cos 0.312 > 0.3, with EV 0.133
      confirming a functioning critic. Dose ladder is an **inverted U** (e3 ≈ 0, e10 0.142, e30 0.076),
      coherent with "too broken → functional-but-unconverged → converged and flat."
    - **On-node control did its job.** e30 retrained here gives 0.076 / cos 0.185 vs the laptop-trained
      reference's 0.064 / cos 0.186 — same regime, so **device numerics are not confounding** the
      dose-response. It did *not* reproduce the reference's training exactly (best epoch 29, acc 0.646,
      vMAE 0.322 vs epoch 25, 0.635, 0.370): same recipe, different run.

  - **STAGE B-0 REPLICATION — the finding that matters (job 7172540, e10 re-probed at `--seed 1`,
    protocol otherwise identical).** §13.1 already warned the cosine is noisy across probe runs
    (0.182 vs 0.429 on one checkpoint), so the marginal cos 0.312 was re-run before spending PPO.
    **The opposite of the expected result:**

    | e10, same_mix | `\|G\|²`(0) | cos |
    |---|---|---|
    | seed 0 (the gate) | 0.142 | 0.312 |
    | seed 1 (replication) | **0.052** | **0.317** |

    - **cos REPLICATED tightly (0.312 → 0.317)**; across all four e10 measurements (2 seeds × 2 arms) it
      spans a narrow 0.286–0.317.
    - **`\|G\|²` DID NOT (0.142 → 0.052, a 2.7× swing from the probe seed alone)**, and seed 1 lands
      *below* the converged reference's 0.064 rather than 2.2× above it.
    - **Therefore: `\|G\|²` at this sample budget CANNOT RESOLVE the effect the gate was built to detect
      — its probe-to-probe noise (2.7×) exceeds the signal (2.2×).** The "2.2× ref" that passed the gate
      was within measurement noise. Any future gate reading `|G|²` differences of this size needs
      multiple probe seeds, or it is reading noise. **This supersedes the `|G|²` half of the B-0 gate.**
    - **What survives is the scale-free statistic the pre-registration told us to prefer:** e10 sits at
      cos ≈ 0.31 while both converged nets sit at ≈ 0.185 (`same_mix`, replicated). Stage B rests on
      that separation, not on `|G|²`. **Blemish, recorded not buried:** e30's `fresh_mix` read cos 0.546
      with `|G|²` 0.018 — high agreement, near-zero gradient, internally incoherent and the one point
      that does not fit.
    - **Stage A is UNAFFECTED.** Its conclusion rested on narrow 3.014 vs wide 0.064 — a **47× gap**,
      far outside this noise. "The wide net is born flat" stands.
    - **HONESTY.** The pre-registered gate returned PASS on e10 and Stage B was launched on it. The
      replication then removed one of the two conditions that produced that PASS. Stage B's strength
      verdict is therefore the **load-bearing** test, not a confirmation of a settled mechanism.

  - **STAGE B — RESULT (ran on UMIACS 2026-07-30/31, jobs 7172508 + 7180385). NULL on strength, and
    the NULL is NOT ATTRIBUTABLE.** PPO warm-started from the reduced-epoch `offrl_gen9ou_wide_e10_s0`
    (acc 0.613 / vMAE 0.703) instead of v21's converged wide net (0.635 / 0.370). Same arch, same
    shard, same PPO config — **offline convergence is the only changed variable**, which is what makes
    it a test of H22. 3 seeds × 50 iters, pooled with `merge_gate_seeds.py` (900 battles/arm, never a
    single-seed file).

    | build | pooled | CI95 | diff vs v22 | z | p | verdict |
    |---|---|---|---|---|---|---|
    | v22 (wide e10) | **0.4622** (416/900) | [0.430, 0.495] | — | — | — | — |
    | v21 (wide converged) | 0.4522 (407/900) | [0.420, 0.485] | +0.010 | +0.43 | 0.67 | **NULL** |
    | v20 (narrow champ) | 0.4722 (425/900) | [0.440, 0.505] | −0.010 | −0.43 | 0.67 | **NULL** |

    Per-checkpoint: s0 `iter_41` 0.4567, s1 **`iter_50` 0.5033**, s2 `iter_48` 0.4267. On the strength
    axis, weakening the offline fit did **nothing**: +0.010 against SE 0.024.
    - **WHY THE NULL DOES NOT INDICT THE LEVER — the trust region BOUND.** §13.1's own rule is that a
      NULL is attributable only if the trust region did not bind. It bound in **59/150 iters at 56–64%
      full-epoch** (s0 18, s1 22, s2 19), vs **v21's 4/150** and **v20's 3/150** at 96–98%;
      `approx_kl_mean_late` 0.031 vs 0.023 / 0.013. **v22 was THROTTLED through 36–44% of its
      optimization.** This caveat was written into the merged gate's `_note` BEFORE the comparison ran,
      so it is not a post-hoc rescue. A NULL under a binding trust region cannot separate "the lever
      does nothing" from "the lever worked and the optimizer would not follow it."
    - **Two independent signs the build was CUT OFF, not converged:** seed 1's best checkpoint is
      `iter_50` — the **final** iteration — and it is also the strongest single checkpoint (0.5033).
    - **The trust-region telemetry INDEPENDENTLY corroborates B-0's cosine.** A 20× difference in
      binding frequency (59 vs 4) is nowhere near measurement noise, unlike the 2.2× `|G|²` claim the
      replication destroyed. The e10 init keeps taking steps large enough to hit the KL ceiling —
      which is what an init with real gradient looks like, measured from the PPO run itself.
    - **CALIBRATION FINDING — build-vs-build differences of ~0.03 are NOT resolvable in one run.**
      v20's *identical* checkpoints scored **425/900 (0.472)** in `seed_strength_gate_v20.json` and
      **449/900 (0.499)** in `seed_strength_gate_v21.json` — 0.027 apart, right at the 0.023 SE.
      Consistent with sampling noise, not a bug. **But it makes Build 21's headline regression
      (−0.047, p = 0.047) fragile**: against v20's *other* scoring the same v21 sits only −0.020 away,
      nowhere near significance. Build 21's "capacity is a REGRESSION" rests on which v20 scoring it
      drew. The *direction* (capacity did not help) survives; the *regression* claim does not.
  - **Open next (Build 23) — RAISE THE KL BUDGET, indicated by telemetry rather than theory.**
    `kl_bar` is **0.045** and v22's `approx_kl_max` reached **0.074**: for the first time in this
    project the trust region is the **active constraint**, not a vanished gradient. Re-running v22
    with a larger trust region is the one experiment that converts this NULL into an attributable
    result, and it is cheap — same init, same config, one changed number. Extending iterations past 50
    is indicated by the same data (s1 peaked at the cap). **State of the evidence for H22:** cos 0.31
    vs 0.185 (replicated) and a 20×-more-binding trust region support it; a flat win-rate from a
    throttled run neither confirms nor refutes it. Those are consistent, not contradictory — an init
    with real gradient that PPO was not permitted to follow.

- **STANDING MEASUREMENT RULE (from Build 22 Stage B-0) — `|G|²` is RETIRED as a small-effect
  instrument.** Recorded here as a rule rather than left inside Build 22's narrative, because it
  constrains every future build and the project has leaned on `|G|²` since Build 20.
  - **The rule.** A `|G|²` difference **under ~3×** is **not measurable at this budget**
    (`--games-per-opp 48 --rollouts 6 --splits 5`). Any gate whose verdict turns on one **must** run
    **≥2 probe seeds** (`scripts/cluster/probe_replicate.slurm`, which varies only `--seed`) and
    report both. A single-seed reading below that threshold is noise, whatever it says.
  - **The evidence.** On a **fixed** checkpoint, changing only the probe seed swung `|G|²` by
    **2.7×** — larger than the **2.2×** effect Stage B-0's gate was pre-registered to detect. The
    gate returned PASS on e10 and Stage B was launched on it; the replication then removed one of
    the two conditions that produced that PASS.
  - **What survives.** The **cosine** from the same probes replicated tightly (0.312 → 0.317) and
    separates e10 (~0.31) from both converged nets (~0.185). Cosine remains usable; `|G|²` at these
    effect sizes does not. **Stage A is unaffected** — its narrow/wide gap is **47×**, an order of
    magnitude outside this noise.
  - **Footgun in the tool.** `scripts/grad_noise_diag.py` takes a scalar `--seed` and **does not
    record it in the output JSON** (top-level keys are policy/init/league_dir/…/verdict — no `seed`,
    no `splits`). Provenance lives **only in the filename**, and the naming is asymmetric:
    `stage_b0.slurm` writes seed 0 with no suffix while replicates get `_seed{N}`. Anyone comparing
    two `grad_noise_diag_*.json` files must confirm from the filenames that the seeds differ.
  - **Corollary already booked twice.** `B_simple` is a **ratio** `tr(Σ)/|G|²`; when the denominator
    is unmeasurable the ratio is too, and its `NOISE_LIMITED` "collect more samples" advice is
    actively misleading (Build 21's TRAP, `plan.md` above).

- **STANDING MEASUREMENT RULE (from Build 22's CALIBRATION FINDING) — score both arms in ONE
  strength-gate run.** v20's *identical* checkpoints scored 0.472 and 0.499 in two separate runs,
  0.027 apart against a 0.023 SE. Cross-run differences below ~0.03 are therefore not results.
  `scripts/cluster/strength_gate.slurm` takes `BUILD_A`/`BUILD_B` and preflights that **both** arms'
  best checkpoints are on disk, naming the missing files rather than failing an hour into battles —
  `/checkpoints/` is gitignored, so an arm trained elsewhere must be staged first.

- **Build 23 — PRE-REGISTERED (written 2026-07-31, BEFORE running). OPEN THE TRUST REGION, and
  re-run H22's comparison at a budget PPO can actually follow.**
  - **Lever: unchanged from Build 22** — the offline convergence of the PPO init (reduced-epoch
    `e10` vs converged). Build 23 does not test a new idea; it re-tests H22 with the throttle removed.
  - **Budget, raised IDENTICALLY IN BOTH ARMS** (so it is not a second lever on the contrast):
    `target_kl` 0.03 → **0.06** (`kl_bar` 0.045 → **0.090**) and `iters` 50 → **80**.
    - **Why 0.06 and not less.** v22's `approx_kl_max` was **0.0737**. A bar at 0.0675 (`target_kl`
      0.045) would still sit *under* the observed max and keep binding at the peaks. 0.090 clears it,
      so the constraint goes **genuinely inactive** rather than merely looser. Not more, because
      `ppo.py:74` records that PPO from a warm start collapses easily.
    - **Why 80 iters.** v22's s1 peaked at `iter_50` — the cap — and s2 at `iter_48`. Two of three
      seeds were still climbing when the run ended.
  - **HYPOTHESIS (H23):** with the trust region inactive, the `e10` init converts its measured
    gradient advantage (cosine 0.31 vs 0.185, replicated; trust region 20× more binding) into
    **strength** over the converged init.
  - **DESIGN — both arms are trained FRESH at the new budget.** `v23a` = `e10` init (treatment),
    `v23b` = converged wide init (control), 3 seeds each, everything else identical. **The tempting
    cheaper design is wrong:** scoring `v23a` against v21's staged checkpoints would compare arms
    differing in **init AND budget**, which is exactly the two-variable ambiguity that cost Build 22
    its verdict. One could argue v21's trust region was inactive anyway (4/150, `mean_late` 0.023),
    so raising its bar would be a no-op — but that is an **assumption**, and this project has just
    spent a build discovering what assumptions cost. Both arms fresh also removes the staging
    dependency entirely: both inits are already on the node.
  - **PRE-REGISTERED GATE — read the certificate BEFORE the win rate.**

    | trust-region certificate | strength (`v23a` − `v23b`, one run, 1800 battles) | verdict |
    |---|---|---|
    | both arms bound < 10% of iters | diff > 0, p < 0.05 | **H22 CONFIRMED** — weakening the offline fit helps, once PPO is permitted to follow it |
    | both arms bound < 10% of iters | p ≥ 0.05 | **H22 REFUTED** — an *attributable* NULL. The mechanism is real but does not reach strength; the init-quality axis closes |
    | `v23a` still bound > 25% of iters | any | **INCONCLUSIVE AGAIN** — the raise was insufficient. This is not a result; do not report a verdict |
    | entropy → 0, or `vs_iter0` < 0.5 late | any | **COLLAPSE** — the warm-start failure `ppo.py:74` warns of. The budget is too loose; report as collapse, never as a NULL |

    The COLLAPSE row is pre-registered because at `kl_bar` 0.090 it is a live outcome, and it must
    not be rationalized afterwards into evidence about the init. v22's endpoint for reference:
    entropy 0.778, `vs_iter0` 1.000.
  - **What Build 23 deliberately gives up.** Moving KL and iters together means `v23` vs `v22` is
    **not** a clean read on the KL raise in isolation. That is the accepted cost of keeping the
    *contrast under test* (init) single-levered, which is the comparison that matters.
  - **Plumbing this needed (`a8e72d6`, landed before the pre-registration, per the house rule that
    harnesses ship before results).** "One changed number" was not achievable: **no trust-region knob
    was reachable from a v20/v21/v22 run** — `ppo_seed.slurm` passed no KL flag and `_ppo_config()`
    never forwarded one, so the whole lineage ran at the frozen `PPOConfig` default. `--target-kl` is
    now threaded through to the result JSON and guarded in `ARM_FIELDS`; `ppo_telemetry.py --kl-bar`
    stops the certificate from naming a bar the optimizer never enforced; and
    `LATEGAME_SHOWDOWN_PORT_BASE` keeps two concurrent arrays off each other's Showdown servers —
    without it both arms compute ports 8100-8102 and silently battle into each other's games.
  - **Cost:** ~9–11 h wall-clock with the arms concurrent; 6 seeds × 80 × 17.4 MiB = **8.4 GB** of
    new checkpoints (85 GB free). The two-arm strength gate is ~11 min.

  - **RESULT (ran on UMIACS 2026-08-01, jobs 7181039 + 7181042 + 7182357). H22 is REFUTED — and the
    refutation is ATTRIBUTABLE. Weakening the offline fit does not fail to help; it actively HURTS.**
    - **TRUST REGION CERTIFIED CLEAN, both arms.** `v23a` bound **6/240** iters (s0 `[21]`,
      s1 `[42,48]`, s2 `[51,52,53]`) at 96–99% full-epoch; `v23b` bound **0/240**, 100% full-epoch.
      Against v22's **59/150** at 56–64%. `approx_kl_max` 0.0785 / 0.0797 against the 0.090 bar.
      **The throttle that cost Build 22 its verdict is gone**, and the pre-registered precondition for
      attribution is met. No collapse: final entropy 0.63–0.68, `vs_iter0` 0.99–1.00.

      | build | bound | full-epoch | `approx_kl_mean_late` |
      |---|---|---|---|
      | **v23a** (e10) | **6/240** | 96–99% | **0.056–0.058** |
      | **v23b** (converged) | **0/240** | 100% | **0.032–0.038** |
      | v22 (e10, old budget) | 59/150 | 56–64% | 0.031 |
      | v21 (converged, old budget) | 4/150 | 96–98% | 0.023 |

    - **STRENGTH — one run, both arms, 1800 battles (`results/seed_strength_gate_v23.json`).**

      | arm | init | pooled | CI95 | per-checkpoint |
      |---|---|---|---|---|
      | **v23b** (control) | converged wide | **0.5578** (502/900) | [0.525, 0.590] | s0 `iter_71` 0.597, s1 `iter_80` 0.470, s2 `iter_75` 0.607 |
      | **v23a** (treatment) | reduced-epoch e10 | **0.4733** (426/900) | [0.441, 0.506] | s0 `iter_66` 0.433, s1 `iter_78` 0.447, s2 `iter_78` 0.540 |

      **diff −0.0844, z −3.58, p = 0.0003, CIs DISJOINT.** By the pre-registered table both arms
      bound < 10%, so this is attributable. H22 predicted diff **> 0**; the measured effect is
      **negative and significant at p = 0.0003** — the strongest signal this project has produced on
      the strength axis, pointing the opposite way from the hypothesis.
    - **THE OUTCOME FELL OUTSIDE THE PRE-REGISTERED ROWS, and that is recorded rather than
      smoothed over.** The table anticipated `diff > 0, p < 0.05` (CONFIRMED) or `p ≥ 0.05`
      (REFUTED, an attributable NULL). It did **not** anticipate a *significant reversal*. The honest
      reading is that H22 is refuted **more strongly** than the "attributable NULL" row contemplated:
      not "the mechanism is real but does not reach strength," but "the mechanism is real and reaches
      strength **with the wrong sign**."
    - **THE MECHANISM EVIDENCE WAS RIGHT; THE INFERENCE FROM IT WAS WRONG.** Every measurement Build
      22 banked replicates here. The e10 init *does* carry more gradient: unthrottled, its
      `approx_kl_mean_late` is **0.056–0.058 against v23b's 0.032–0.038** — ~1.6× larger steps, from
      the run itself, exactly as the cosine (0.31 vs 0.185) and the 20×-more-binding trust region
      predicted. **It takes bigger steps, and they go somewhere worse.** `|G|²`-style "is there
      gradient here" reasoning cannot distinguish a *useful* gradient from a merely *large* one, and
      this build is the counterexample: **real gradient ≠ useful gradient.** That retires the
      pre-filter Build 21 pre-registered ("a candidate that fixes the plateau must show `|G|²`
      recovering") as *necessary but nowhere near sufficient*.
    - **TOOLING DEFECT, recorded not buried.** `seed_strength_gate.py:164-176` computes
      `verdict = "WIN" if significant and diff > 0 else "NULL"`, so it stamped this result **`NULL`**
      — a significant, CI-disjoint, p = 0.0003 reversal labelled identically to a p = 0.67 nothing.
      The verdict field is **not trustworthy for negative effects**; read `diff`/`z`/`p`. A three-way
      verdict (WIN / NULL / REGRESSION) is owed.
    - **UNEXPECTED, CONFOUNDED, AND THE MOST INTERESTING NUMBER HERE: `v23b` = 0.5578 is the highest
      pooled rate in the project's history**, against a previous best of v20's 0.4989 and a v19–v22
      band of 0.448–0.499. Its CI [0.525, 0.590] barely overlaps that of the best prior build. This
      is **NOT attributable** and must not be reported as a win: `v23b` differs from v21 in **both**
      `target_kl` (0.03 → 0.06) **and** `iters` (50 → 80), *and* the comparison is cross-run, which
      §13.1's own calibration finding puts at ±0.027. But +0.059 over the best prior scoring is
      **above** that noise floor, and it is the first thing in five builds to move this far. **This is
      the lever Build 24 should test**, one variable at a time.
    - **The iteration budget may STILL bind.** `v23b`'s seed 1 peaked at `iter_80` — the cap, again —
      and is also that arm's weakest checkpoint (0.470). Two of three v22 seeds peaked at or beside
      the 50-cap; one of three does so at 80.
  - **Open next (Build 24) — SEPARATE THE BUDGET LEVERS, on the converged init.** The init-quality
    axis is **closed**: weakening the offline fit is refuted with the sign against it, and Build 21
    closed capacity, 19–20 closed optimization/sampling. What is left standing is the accidental
    finding — that the raised budget moved strength further than any deliberate lever since Build 16.
    Run `target_kl` 0.06 at `iters` 50, and `target_kl` 0.03 at `iters` 80, both from
    `offrl_gen9ou_wide_s0`, scored against `v23b` in one gate. That resolves which half of the raise
    did the work, or whether it needs both.

- **Build 24 — PRE-REGISTERED (written 2026-08-01, BEFORE running). WHICH HALF OF THE BUDGET RAISE
  DID THE WORK?**
  - **Lever: the budget raise Build 23 made accidentally**, decomposed. Build 23 moved `target_kl`
    0.03 → 0.06 **and** `iters` 50 → 80 together, and `v23b` came out at **0.5578**, +0.059 over the
    best prior scoring and the first thing in five builds to clear the ±0.027 cross-run noise floor.
    That total is not attributable — two levers, and a cross-run comparison. Build 24 makes it
    attributable and splits it.
  - **DESIGN — a 2×2 factorial plus a decomposition arm, all five scored in ONE gate run.** Every
    arm warm-starts from `offrl_gen9ou_wide_s0` (converged wide), seeds 0/1/2, everything else at
    the v17–v23 values. `v23b` is reused as-is; the other four are trained fresh.

    | arm | `target_kl` | `iters` | `anneal_iters` | role |
    |---|---|---|---|---|
    | `v23b` | 0.06 | 80 | — (= 80) | the accidental finding, reused |
    | `v24a` | 0.06 | 50 | — (= 50) | 2×2 cell |
    | `v24b` | 0.03 | 80 | — (= 80) | 2×2 cell |
    | `v24c` | 0.03 | 50 | — (= 50) | 2×2 cell — v21's configuration, retrained fresh |
    | `v24d` | 0.06 | 80 | **50** | splits the `iters` effect |

    **`v24c` is trained fresh rather than read off v21**, both because v21's checkpoints are no
    longer on disk and because Build 23's own rule requires it: comparing against staged
    old-budget checkpoints is the two-variable ambiguity that cost Build 22 its verdict.
  - **WHY `anneal_iters` HAD TO EXIST FIRST.** `iters` silently doubled as the lr/ent anneal
    horizon, so "50 → 80" was never one change. At iteration 40, v22 (50-budget) ran lr 9.08e-05 /
    ent 0.0020 while v23b (80-budget) ran lr 1.51e-04 / ent 0.0051; at iteration 50 v22 was frozen
    at its finals while v23b was still at 1.26e-04. `v24d` pins the horizon to 50 at `iters` 80, so
    it holds both schedules at their finals past 50. **`config.iters` is used in exactly two places**
    — the loop range (`ppo.py:373`) and `anneal_horizon` (`ppo.py:103-105`) — so with the horizon
    pinned, `v24d`'s iterations 1–50 run the *identical code path* to `v24a`. `v24d` − `v24a`
    therefore differs in **nothing but whether the loop kept going past 50**.
  - **PRE-REGISTERED CONTRASTS — four, at α = 0.0125 (Bonferroni).**

    | # | contrast | isolates |
    |---|---|---|
    | 1 | `v24c` → `v23b` | the **total** — does +0.059 reproduce *inside one scoring run*? |
    | 2 | `v24b` → `v23b` | the **KL half**, holding `iters` 80 |
    | 3 | `v24a` → `v24d` | pure **update count** |
    | 4 | `v24d` → `v23b` | pure **anneal horizon** — both 80 updates, horizons 50 vs 80 |

    Contrasts 3 + 4 sum to the `iters` half (`v24a` → `v23b`) **by construction**; that sum is
    reported as a descriptive consistency check, not a fifth test. The gate prints all
    5·4/2 = **10** pairs — **the other six are NOT pre-registered** and are descriptive only.
  - **PRE-REGISTERED GATE.**

    | total (#1) | KL half (#2) | `iters` half (#3+#4) | verdict |
    |---|---|---|---|
    | not significant | — | — | **NOT REPLICATED** — the +0.059 was cross-run noise. Do not decompose a total that is not there; the budget axis returns to the pool |
    | sig, > 0 | sig, > 0 | not sig | **KL DID THE WORK** |
    | sig, > 0 | not sig | sig, > 0 | **ITERS DID THE WORK** → read #3 against #4 |
    | sig, > 0 | both sig | both sig | **BOTH HALVES CONTRIBUTE** |
    | sig, > 0 | neither sig | neither sig | **UNDERPOWERED SPLIT** — the total is real but each half sits under the MDE. This is *not* a finding that neither matters, and must not be reported as one |
    | sig, < 0 | — | — | **REVERSAL** — book it outside the anticipated rows, as Build 23's outcome was booked |
    | entropy → 0, or `vs_iter0` < 0.5 late | any | any | **COLLAPSE** — report as collapse, never as a NULL |

  - **TRUST-REGION BINDING IN `v24b`/`v24c` IS THE TREATMENT, NOT A DEFECT — a deliberate departure
    from Build 23's rule.** Build 23 pre-registered an INCONCLUSIVE row that voided the comparison if
    the trust region bound, because there it was a *nuisance* throttling the lever under test. Here
    `target_kl` **is** the lever, so binding in the 0.03 arms is the mechanism being measured. The
    certificate is still recorded, to characterise the dose rather than to invalidate the arm.
    Expect it to be modest: v21, the same init at the same config, bound 4/150.
  - **SCOPE OF EVERY VERDICT: THESE CHECKPOINTS, NOT THE TRAINING PROCEDURE.** The pooled z-test
    treats 900+ battles as iid and **the seeds are not**. Build 23's per-seed rates were 0.597 /
    0.470 / 0.607 (`v23b`) against a within-seed binomial sd of 0.0287 at n=300 — so the
    **between-seed sd is ~0.0706, 2.5× the within-seed noise**. Read at seed level, Build 23's own
    p = 0.0003 is a paired **t = 2.04** (3/3 seeds positive, sign-test p = 0.25). The direction
    replicates; the *procedure-level* evidence is far weaker than the pooled p reads. This is the
    inference model the protocol has used since Build 20, not a defect in Build 23's execution, and
    **no feasible seed count fixes it** — detecting +0.059 at 80% power at seed level needs ~24
    seeds/arm ≈ 150 task-hours/arm. So: the pooled test stays the pre-registered verdict, every
    contrast now also carries a `seed_level` block (`seed_strength_gate.py`), and every Build 24
    claim is scoped to the checkpoints it scored.
  - **WHY N=1800 AND NOT THE USUAL 300.** Under the checkpoint-level model, battles are the cheapest
    power in the build — Build 23 ran 1800 battles in 9:14. The halves are ~0.030 if the total
    splits evenly:

    | N per checkpoint | SE(diff) | MDE @ 80%, α=0.0125 | power at a 0.030 half |
    |---|---|---|---|
    | 300 | 0.0236 | 0.079 | ~7% |
    | 900 | 0.0136 | 0.045 | ~24% |
    | **1800** | **0.0096** | **0.032** | **~73%** |

    At N=900 the overwhelmingly likely outcome is "total significant, both halves NULL" — the
    UNDERPOWERED SPLIT row, which answers nothing. N=1800 is 27,000 battles ≈ 2.3 h against the
    gate's 8 h limit. Note this buys power for the **pooled** test only: the seed-level SE moves
    0.0622 → 0.0584 across the same range, because that variance is *between* seeds.
  - **Selection bias, checked and cleared.** Comparing 50-iter against 80-iter arms means the longer
    arm gets more draws at the noisy `argmax` that picks each seed's best checkpoint. Simulated at
    `eval_n`=100, that is worth **+0.0014** — negligible against a 0.030 effect. Not a confound.
  - **Cost, and the QOS fact that sets it.** `tron` carries `MaxTRESPU cpu=32,mem=256G` and each
    task asks 8 CPU / 64 GB, so **exactly 4 tasks run concurrently** — measured on Build 23's own
    `sacct`, where task `7181042_1` started at 00:51:10 against `7181042_0`'s 00:51:09 end, and
    `7181042_2` at 03:00:38 against `7181039_1`'s 03:00:38 end. Slurm backfills perfectly, so all 12
    tasks are submitted at once and no manual wave staging is needed — but the estimate must be
    built from **task-hours, not "arms run concurrently"**: ~60 task-hours ÷ 4 ≈ **15–18 h**
    wall-clock (v23b measured 4.65 min/iter). Plus **13.3 GB** of checkpoints (77 GB free) and a
    ~2.3 h gate. Four concurrent 3-task arrays would *not* have run; assuming they would was the
    open operational question, and the answer is no.

  - **RESULT (ran on UMIACS 2026-08-01/02, jobs 7183404–7183407 + 7186606). ITERS DID THE WORK —
    the pre-registered row, hit exactly. The KL raise Build 23 built all its plumbing for did
    essentially NOTHING; the iteration budget did all of it, and ~70% of that is raw update count.**
    - **CERTIFICATES, all four arms, written before the comparison.** No collapse anywhere
      (`vs_iter0` 1.00 on all 12 seeds; final entropy 0.56–0.80).

      | arm | `kl_bar` | bound | full-epoch | `approx_kl_mean_late` |
      |---|---|---|---|---|
      | `v24a` (0.06/50) | 0.09 | **0/150** | 100% | 0.0242–0.0262 |
      | `v24b` (0.03/80) | 0.045 | **56/240** | 70–86% | 0.0298–0.0305 |
      | `v24c` (0.03/50) | 0.045 | **2/150** | 96–100% | 0.0201–0.0247 |
      | `v24d` (0.06/80, pinned) | 0.09 | **2/240** | 99–100% | 0.0187–0.0210 |

      The KL lever therefore had a **real dose** — `v24b` bound 56/240 against `v24a`'s 0/150 — so
      the null on it below is a null on a treatment that was actually administered. Note the
      asymmetry inside the 0.03 row: `v24b` bound 56/240 but `v24c` only 2/150, concentrated in
      `v24b`'s iters ~22–45, where its lr is still high because its anneal spans 80. **The KL
      contrast has more bite at 80 iters than at 50** — a property of the 2×2 as designed
      (anneal = iters in every cell), recorded because it bears on how the interaction reads.
    - **THE 2×2 (one run, N=1800/checkpoint, 5400/arm, 27,000 battles).**

      | | `iters` 50 | `iters` 80 | **iters effect** |
      |---|---|---|---|
      | `target_kl` 0.03 | `v24c` **0.449** | `v24b` **0.533** | **+0.084** |
      | `target_kl` 0.06 | `v24a` **0.444** | `v23b` **0.550** | **+0.106** |
      | **KL effect** | **−0.005** | **+0.017** | |

      **Both KL contrasts are null; both `iters` contrasts are large and significant.** The
      interaction is +0.022, small.
    - **THE FOUR PRE-REGISTERED CONTRASTS at α = 0.0125.**

      | # | contrast | isolates | diff | p | sig | seed-level `t` | seeds agreeing |
      |---|---|---|---|---|---|---|---|
      | 1 | `v24c`→`v23b` | **total** | **+0.102** | <0.0001 | ✔ | +2.11 | 3/3 |
      | 2 | `v24b`→`v23b` | **KL half** | +0.017 | 0.0725 | ✘ | +0.89 | 2/3 |
      | 3 | `v24a`→`v24d` | **update count** | **+0.075** | <0.0001 | ✔ | **+9.76** | **3/3** |
      | 4 | `v24d`→`v23b` | **anneal** | **+0.031** | 0.0011 | ✔ | +0.72 | 2/3 |

      Consistency check passes: #3 + #4 = **+0.107** against the `iters` half (`v24a`→`v23b`) of
      **+0.106**, as it must by construction.
    - **BY THE PRE-REGISTERED TABLE: total significant and positive, KL half not significant,
      `iters` half significant and positive ⇒ ITERS DID THE WORK.** Unlike Build 23, this outcome
      fell *inside* the anticipated rows.
    - **THE ACCIDENTAL FINDING REPLICATES, AND IS NOW ATTRIBUTABLE.** Build 23's +0.059 was
      cross-run and confounded across two levers. Measured within one run against a freshly trained
      v21-configuration baseline it is **+0.102** — larger, not smaller. And `v23b` itself scored
      **0.550** here against **0.5578** in the Build 23 gate: 0.008 apart, comfortably inside the
      §13.1 calibration band of ±0.027. **The highest pooled rate in the project's history is
      confirmed, and its cause is identified.**
    - **READING #3 AGAINST #4 — update count is the driver, and it is the FIRST LEVER IN THIS
      PROJECT THAT IS ROBUST AT BOTH INFERENCE LEVELS.** Raw update count carries +0.075 of the
      +0.106, with per-seed diffs +0.068 / +0.067 / +0.091 — **3/3 seeds, sd 0.0136, seed-level
      t = +9.76.** Every previous "significant" result in this project has been checkpoint-level
      only (Build 23's p = 0.0003 is a seed-level t of 2.04). This one is not.
      **Why the paired design earned that:** `v24a` and `v24d` share init, seed, and — because
      `config.iters` is used *only* at the loop range and in `anneal_horizon` — the identical code
      path over iterations 1–50. The between-seed variance that swamps every other contrast here
      largely cancels in this one. Pinning the horizon did not merely disambiguate the lever; it
      bought a **7× better seed-level statistic** than any unpaired contrast in the same run.
    - **The anneal contribution is real but NOT seed-robust, and must not be reported as though it
      were.** +0.031 at p = 0.0011 pooled, but seed-level t = +0.72 with only **2/3** seeds agreeing.
      This is exactly the checkpoint-vs-procedure divergence the pre-registration scoped for, showing
      up on the very first build that instrumented it.
    - **THE TRUST REGION WAS NEVER THE CONSTRAINT.** `v24b` spent 56/240 iterations throttled and
      still scored 0.533 against `v23b`'s 0.550 — a −0.017 cost that does not clear α. Build 23
      spent an entire build's plumbing making `target_kl` reachable on the theory that the trust
      region was binding the lever. It was binding; **it just did not matter.** Booked as plainly
      as Build 23's own reversal was: the KL axis is now **closed**.
  - **Open next (Build 25) — PUSH THE ITERATION BUDGET UNTIL THE RETURN FLATTENS.** For the first
    time since Build 16 there is a lever that is significant, replicated, seed-robust, and
    *mechanistically identified*: **more updates**. Every other axis is closed — init quality (23),
    capacity (21), optimization/sampling (19–20), and now the trust region (24). The budget is not
    saturated: `v24c` s0 peaked at `iter_50` (its cap) and Build 23's `v23b` s1 peaked at `iter_80`
    (its cap). Run `iters` 80 → 120 → 160 at `target_kl` 0.06 from `offrl_gen9ou_wide_s0`, scored
    against `v23b` in one gate, and **pin the anneal horizon across arms** so the contrast stays on
    update count — the one thing measured to carry the effect. Cost scales linearly and the QOS caps
    throughput at 4 tasks, so budget in task-hours ÷ 4 (§ above): a 3-arm × 3-seed sweep at 120/160
    is ~100 task-hours ≈ 25 h.

- **Build 25 — PRE-REGISTERED (written 2026-08-02, BEFORE running). WHERE DOES THE UPDATE-COUNT
  RETURN FLATTEN?**
  - **Lever: the one axis Build 24 left open.** Update count is the only lever in this project that
    is significant, replicated, seed-robust (`v24a`→`v24d` +0.075, seed-level t = +9.76, 3/3 seeds)
    *and* mechanistically identified. It is also not saturated: `v24c` s0 peaked at its `iter_50`
    cap, `v23b` s1 at its `iter_80` cap, and `v24d` — schedule frozen at its finals past iteration
    50 — still peaked at 73/75/54. Build 25 extends the dose 80 → 120 → 160 and finds the knee.
  - **DESIGN — three fresh arms against the reused `v23b` anchor, `target_kl` 0.06 throughout,
    warm-started from `offrl_gen9ou_wide_s0`, seeds 0/1/2.**

    | arm | `iters` | `anneal_iters` | role |
    |---|---|---|---|
    | `v23b` | 80 | — (= 80) | anchor, reused; its schedule over 1–80 is identical to every new arm's |
    | `v25a` | 120 | **80** | update-count dose 1 |
    | `v25b` | 160 | **80** | update-count dose 2 |
    | `v25c` | 160 | — (= 160) | schedule scaled to the long budget |

    Pinning the horizon at **80** — not 50 — is what makes `v23b` a legitimate anchor rather than a
    fourth configuration: with `anneal_iters` 80, arms `v25a`/`v25b` run `v23b`'s exact schedule
    over iterations 1–80 and then continue at the finals, so `v23b` → `v25a` inherits Build 24's
    pairing (shared init, shared seed, identical code path over the whole anchor's length).
  - **WHY `v25c` EXISTS.** With the horizon pinned at 80, iterations 81+ run at lr 5e-5 / ent 0.
    A flat `v25a` → `v25b` would then be ambiguous in exactly the way Build 23's KL story was:
    "updates saturate" and "the schedule froze" predict the same null. `v25c` (160 updates, horizon
    160) separates them, and doubles as a re-test of Build 24's anneal contribution (+0.031, but
    seed-level t = +0.72 on 2/3 seeds — booked as **not** seed-robust) at a budget with more room.
  - **PRE-REGISTERED CONTRASTS — four, at α = 0.0125 (Bonferroni).**

    | # | contrast | isolates |
    |---|---|---|
    | 1 | `v23b` → `v25a` | update count, 80 → 120 |
    | 2 | `v25a` → `v25b` | update count, 120 → 160 |
    | 3 | `v23b` → `v25b` | **total** update count, 80 → 160 |
    | 4 | `v25b` → `v25c` | anneal horizon at the 160 budget |

    #1 + #2 = #3 by construction — a descriptive consistency check, not a fifth test. The gate
    prints all 4·3/2 = 6 pairs; the other two are **not** pre-registered.
  - **THE SELECTION BIAS DOES *NOT* CANCEL HERE, AND IT IS NOT NEGLIGIBLE — the one methodological
    departure from Build 24.** Each seed's reported checkpoint is the `argmax` of ~`iters` noisy
    `eval_n`=100 points. Re-scoring at N kills the winner's curse on the curve *value*, but not the
    *selection*: more draws over checkpoints of genuinely different strength land on a truly better
    one more often, and Build 25's arms differ in length by **2×** where Build 24's differed by 1.6×.
    Simulated (true late-window p ~ N(µ, σ_b), observed ~ Bin(100, p)/100, select `argmax`, 200k
    trials), with **σ_b = 0.0428 estimated from the Build 23/24 curves themselves** (observed
    late-window variance minus the binomial component):

    | contrast | differential selection bias |
    |---|---|
    | #1 `v23b` → `v25a` | **+0.0045** |
    | #2 `v25a` → `v25b` | **+0.0028** |
    | #3 `v23b` → `v25b` | **+0.0073** |

    Build 24 measured +0.0014 for its 50-vs-80 comparison and cleared it as negligible. **+0.0073
    against an MDE of 0.025 is not negligible** — it is ~30% of the smallest dose worth calling
    real. Two pre-registered consequences: (a) the biases above are **subtracted before a dose is
    declared**, i.e. a contrast must clear both `p < α` *and* `diff − bias > 0`; (b) a second,
    **selection-free** gate scores each arm's **terminal** checkpoint (`iter_80` / `iter_120` /
    `iter_160` / `iter_160`) via `scripts/pin_gate_checkpoint.py`, which rewrites a merged gate
    JSON's `best_checkpoint` to a pinned iteration and hands it to the *unmodified* authoritative
    gate. The terminal read carries zero selection bias and is reported as a descriptive check on
    the same four contrasts — **not** α-corrected, because it is not a second family of tests.
    **A sign disagreement between the seed-best and terminal reads is itself the finding**, and is
    to be booked as one rather than resolved in favour of whichever agrees with the hypothesis.
  - **PRE-REGISTERED GATE.**

    | #1 (80→120) | #2 (120→160) | #4 (anneal) | verdict |
    |---|---|---|---|
    | sig, > 0 | sig, > 0 | — | **STILL CLIMBING** — the axis is not saturated; Build 26 extends again and this becomes a dose-response curve |
    | sig, > 0 | not sig | not sig | **KNEE BETWEEN 120 AND 160** — book the saturation point and stop extending |
    | not sig | not sig | not sig | **ALREADY SATURATED AT 80** — the update-count axis closes with every other PPO-side axis |
    | — | not sig | **sig, > 0** | **FROZEN-SCHEDULE ARTIFACT** — the flattening was the schedule, not the updates; the dose-response must be re-run with the anneal scaled, and no saturation claim may be made from this build |
    | sig, < 0 | — | — | **REVERSAL** — more updates cost strength; book outside the anticipated rows, as Build 23's outcome was |
    | entropy → 0, or `vs_iter0` < 0.5 late | any | any | **COLLAPSE** — report as collapse, never as a NULL |

  - **A SECOND, INDEPENDENT SATURATION READ — from the curves, not the gate.** `best_iter` per seed.
    If `v25b`'s `best_iter` lands ≤ 120 for ≥2/3 seeds, that is saturation evidence that does not
    depend on the gate at all, and it is pre-registered *because* it can contradict the gate: a
    significant #2 with every `best_iter` below 120 would mean the contrast is being carried by
    something other than the extra iterations.
  - **WHY N = 3000 FOR THE PRIMARY GATE.** Battles remain the cheapest power in a build whose
    training is ~107 task-hours (Build 24's gate: 27,000 battles in 2:09:31, i.e. ~209/min).
    Build 24's update-count dose was +0.075 across a 1.6× budget raise; the per-step doses here are
    plausibly ~0.02–0.04, which N = 1800 cannot resolve:

    | N per checkpoint | pooled/arm | SE(diff) | MDE @ 80%, α=0.0125 | power at a 0.025 dose |
    |---|---|---|---|---|
    | 1800 | 5400 | 0.0096 | 0.032 | ~48% |
    | **3000** | **9000** | **0.0075** | **0.025** | **~80%** |

    4 arms × 3 seeds × 3000 = **36,000 battles ≈ 2:52** against the gate's 8 h limit. The terminal
    (selection-free) gate runs separately at N = 1800 — 21,600 battles ≈ 1:44 — because it is a
    descriptive check, and because two self-contained invocations keep each comparison internally
    calibrated. **Arms are never compared across the two gates.** As in Build 24, raising N buys
    power for the **pooled** test only; the seed-level SE is between-seed variance and barely moves.
  - **SCOPE, UNCHANGED FROM BUILD 24: every verdict is scoped to the checkpoints it scored.** The
    pooled z-test treats 9000 battles as iid and the seeds are not. Every contrast carries its
    `seed_level` block, and a pooled-significant / seed-null result is reported as what it is — the
    outcome Build 24's anneal half produced (+0.031 pooled, t = +0.72, 2/3 seeds) and booked as not
    seed-robust. Expect #1 to be the strong seed-level statistic, for the same pairing reason
    `v24a`→`v24d` was.
  - **COST.** 9 tasks — 3 × 120 iters (~9.7 h) + 6 × 160 iters (~13 h) ≈ **107 task-hours**. The
    `tron` QOS caps a user at `cpu=32,mem=256G` against 8 CPU / 64 GB per task, so **4 run
    concurrently** ⇒ ≈ **32 h wall-clock**, plus ~2:52 + ~1:44 of gate. Per-iteration cost measured
    on Build 24's `sacct`: 50 iters 4:08–4:44, 80 iters 6:21–6:41 ⇒ ~4.8–5.3 min/iter. **Disk is the
    binding operational constraint, not time**: ~17.5 MB/checkpoint ⇒ 3 × 2.1 GB + 6 × 2.8 GB ≈
    **23 GB**, against 64 GB free with 25 GB already in `checkpoints/`. The 160-iter arms are
    submitted with `--time=24:00:00` (the script's 20 h default leaves only 7 h of margin at 13 h
    expected; `medium` allows 2 days and headroom is free).

- **Build 25 — STILL CLIMBING: the update-count axis is NOT saturated at 160, and scaling the
  anneal to match COSTS strength.** 9 training runs (`v25a`/`v25b`/`v25c` × 3 seeds, 2026-08-03/04),
  then both pre-registered gates on 2026-08-06: primary N = 3000 (36,000 battles, 2:28:51) and the
  selection-free terminal read at N = 1800 (21,600 battles, 1:54:09). Both `COMPLETED 0:0`.
  - **POOLED RATES vs the heuristic (n = 9000/arm, seed-best).**

    | arm | `iters` | `anneal_iters` | rate | CI95 |
    |---|---|---|---|---|
    | `v23b` | 80 | — (= 80) | 0.5450 | [0.5347, 0.5553] |
    | `v25a` | 120 | 80 | 0.6144 | [0.6043, 0.6244] |
    | **`v25b`** | **160** | **80** | **0.6534** | **[0.6435, 0.6632]** |
    | `v25c` | 160 | — (= 160) | 0.5604 | [0.5502, 0.5707] |

    **0.6534 is the highest pooled rate in the project's history**, against `v23b`'s 0.550 — the
    previous record, set by Build 24. The anchor re-calibrates: `v23b` scored **0.5504** in Build
    24's gate and **0.5450** here, **0.0054** apart, comfortably inside the §13.1 band of ±0.027.
  - **THE FOUR PRE-REGISTERED CONTRASTS at α = 0.0125, with the bias subtracted before any dose is
    declared.**

    | # | contrast | isolates | diff | − bias | = adj | p | sig | seed `t` | seeds |
    |---|---|---|---|---|---|---|---|---|---|
    | 1 | `v23b`→`v25a` | 80 → 120 | +0.0694 | 0.0045 | **+0.0649** | <0.0001 | ✔ | +2.12 | **3/3** |
    | 2 | `v25a`→`v25b` | 120 → 160 | +0.0390 | 0.0028 | **+0.0362** | <0.0001 | ✔ | +1.03 | 2/3 |
    | 3 | `v23b`→`v25b` | **total** 80 → 160 | +0.1084 | 0.0073 | **+0.1011** | <0.0001 | ✔ | **+5.36** | **3/3** |
    | 4 | `v25b`→`v25c` | anneal horizon @ 160 | **−0.0930** | — | — | <0.0001 | ✔ | −2.17 | **0/3** |

    Consistency check passes: #1 + #2 = **+0.1084** against #3's **+0.1084**, as it must by
    construction. #4 carries no bias: both its arms run 160 iterations, so the draws at the `argmax`
    are equal in number and the selection cancels exactly as it did through Build 24.
  - **BY THE PRE-REGISTERED TABLE: #1 significant and positive, #2 significant and positive ⇒
    STILL CLIMBING.** The axis is not saturated; Build 26 extends again and this becomes a
    dose-response curve. The outcome fell inside the anticipated rows.
  - **THE SELECTION-BIAS CORRECTION WAS APPLIED AND DID NOT BIND.** All three doses clear their
    bias by an order of magnitude (+0.0649 / +0.0362 / +0.1011 against 0.0045 / 0.0028 / 0.0073).
    Booked explicitly because the pre-registration committed to it before the sign was known: it is
    reported as *applied and immaterial*, not quietly dropped because it did not change a verdict.
  - **THE SECOND, SELECTION-FREE READ AGREES IN SIGN ON ALL FOUR CONTRASTS** (terminal, n = 5400/arm:
    #1 **+0.0411**, #2 **+0.0694**, #3 **+0.1106**, #4 **−0.0907**). There is **no sign disagreement
    to book.** #3 is near-identical across the two reads (+0.1084 vs +0.1106), which is the strongest
    evidence in this build that the total effect is not a selection artifact.
  - **#2 IS POOLED-SIGNIFICANT BUT NOT SEED-ROBUST ON THE SEED-BEST READ, AND THIS MUST NOT BE
    REPORTED AS THOUGH IT WERE.** Its per-seed diffs are **+0.003 / −0.001 / +0.115** — t = +1.03,
    2/3 agreeing, i.e. the 120 → 160 dose is carried almost entirely by seed 2. **The terminal read
    is the stronger one here** (+0.0694, t = +2.34, **3/3** seeds), which is the reverse of the usual
    direction and is itself worth carrying into Build 26: the 120 → 160 dose is real at the
    procedure level but the seed-best estimate of it is noisy. #1 and #3 are 3/3 on both reads, and
    #3's t = +5.36 makes the **total** 80 → 160 dose the second lever in this project robust at both
    inference levels.
  - **#4 IS A SEED-ROBUST REGRESSION, AND IT IS NOT THE ROW THE PRE-REGISTRATION ANTICIPATED.** The
    FROZEN-SCHEDULE ARTIFACT row required #4 **significant and positive** — the story where the
    flattening was the schedule rather than the updates. #4 came back **significant and negative**:
    −0.093 pooled, **0/3** seeds positive, t = −2.17 seed-best and **−4.54** terminal. So the frozen
    schedule is not hiding the effect; **the frozen schedule is actively better.** Holding lr at
    5e-5 / ent at 0 from iteration 81 to 160 beats annealing across the full 160 by ~9 points. This
    also settles Build 24's anneal half — booked there as +0.031 pooled but **not** seed-robust
    (t = +0.72, 2/3) — in the opposite direction and with authority at the longer budget.
  - **THE CURVE-SIDE SATURATION READ, PRE-REGISTERED AS ABLE TO CONTRADICT THE GATE, DOES NOT.**
    `v25b`'s `best_iter` is **132 / 125 / 159** — **0/3** seeds at or below 120, where ≥2/3 was the
    saturation signal. Curve and gate agree that 160 is not the ceiling.
  - **THE TRUST REGION DID NOT BIND, SO THE VERDICT IS ATTRIBUTABLE.** `v25a` bound in 1/360 iters,
    `v25b` in **0/480** (100% full-epoch on all three seeds), `v25c` in 19/480 (95–98% full-epoch,
    the only arm where a live schedule engaged it at all). `approx_kl_max` 0.0868 / 0.0551 / 0.0816
    all under the 0.09 bar. Nowhere near Build 22's verdict-costing 59/150. No collapse: `vs_iter0`
    0.99–1.00 on all nine runs, final entropy 0.52–0.60.
  - **Open next (Build 26) — EXTEND THE DOSE AGAIN, AND PIN THE HORIZON AT 80.** Two arms are now
    specified by the data rather than guessed: `iters` **240** and **320** at `anneal_iters` **80**,
    `target_kl` 0.06, from `offrl_gen9ou_wide_s0`, scored against `v25b` in one gate. The horizon
    question is **closed by #4** — scaling it is worse, seed-robustly — so Build 26 spends no arm on
    it and the contrast stays purely on update count. Two open risks to pre-register against:
    (a) the differential selection bias grows with the length ratio and must be re-simulated for
    240/320 vs 160, not reused from this build; (b) #2's seed-best/terminal split says the per-step
    dose is now near the resolution of the seed-best read, so Build 26 should treat the **terminal**
    read as co-primary rather than descriptive. Cost scales linearly: 6 tasks at ~4.8–5.3 min/iter is
    ~150 task-hours ÷ 4 concurrent ≈ **38 h wall-clock**, and ~29 GB of checkpoints against 146 GB
    free — disk is no longer the binding constraint it was in Build 25.

- **Build 26 — PRE-REGISTERED (written 2026-08-06, BEFORE running).**
  - **Lever: the same axis, extended.** Build 25 returned STILL CLIMBING on the pre-registered
    table — #1 (80→120) **+0.0649 adj**, #2 (120→160) **+0.0362 adj**, both significant and
    positive — so the update-count axis is not saturated at 160 and the anticipated row says
    extend. `v25b`'s `best_iter` was **132 / 125 / 159**, 0/3 seeds at or below 120, so the
    curve-side read agrees. Build 26 extends 160 → 240 → 320.
  - **DESIGN — two fresh arms against the reused `v25b` anchor, `anneal_iters` 80 and `target_kl`
    0.06 throughout, warm-started from `offrl_gen9ou_wide_s0`, seeds 0/1/2.**

    | arm | `iters` | `anneal_iters` | role |
    |---|---|---|---|
    | `v25b` | 160 | 80 | anchor, reused; its schedule over 1–160 is identical to every new arm's |
    | `v26a` | 240 | **80** | update-count dose 1 |
    | `v26b` | 320 | **80** | update-count dose 2 |

    **The horizon question is CLOSED and gets no arm.** Build 25's #4 came back significant and
    **negative** (−0.093 pooled, **0/3** seeds, t = −4.54 terminal): scaling the anneal to the full
    budget *costs* ~9 points, and the frozen schedule is actively better. That also settled Build
    24's non-seed-robust +0.031 anneal half in the opposite direction. Pinning at 80 keeps `v25b` a
    legitimate anchor for the same reason it made `v23b` one in Build 25 — shared init, shared
    seeds, byte-identical schedule over the anchor's whole length.
  - **PRE-REGISTERED CONTRASTS — three, at α = 0.0167 (Bonferroni over 3).**

    | # | contrast | isolates |
    |---|---|---|
    | 1 | `v25b` → `v26a` | update count, 160 → 240 |
    | 2 | `v26a` → `v26b` | update count, 240 → 320 |
    | 3 | `v25b` → `v26b` | **total** update count, 160 → 320 |

    #1 + #2 = #3 by construction — a consistency check, not a fourth test. The gate prints all
    3·2/2 = 3 pairs, so here every printed pair is pre-registered.
  - **THE SELECTION BIAS IS RE-SIMULATED, AND THE SIMULATION IS NOW COMMITTED CODE.** Build 25
    committed to re-simulating rather than reusing, because the differential grows with the length
    ratio. Its own simulation was run ad hoc and never committed, so its table could not be
    re-derived — the correction a dose is declared against was unauditable. `scripts/
    selection_bias_sim.py` is that simulation, and `tests/test_selection_bias_sim.py` pins it,
    including a reproduction of Build 25's published +0.0045 / +0.0028 / +0.0073 to ~5e-4 (the
    residual is unattributable: Build 25 recorded neither its `mu` nor its exact draw count).
  - **σ_b IS RE-ESTIMATED, AND BUILD 25's VALUE WAS INFLATED.** Build 25 used **σ_b = 0.0428**,
    estimated from the Build 23/24 curves — arms whose lr/entropy schedule ran their *whole*
    length, so their "late window" still contained a live anneal. That books **schedule trend as
    checkpoint dispersion**. Estimated instead from the only arms with a genuine frozen plateau
    (`v25a`/`v25b`, window `iter > 80`, 6 seed-arms, dof 348):

    | estimate | σ_b | what it charges |
    |---|---|---|
    | detrended point | **0.0000** | at/below the binomial floor `p(1−p)/100` |
    | detrended 95% upper | **0.0000** | the bound is at the floor too |
    | raw (drift charged as dispersion) | **0.0257** | conservative |
    | raw 95% upper | **0.0328** | **pre-registered** |
    | Build 25's | 0.0428 | inflated by a live schedule |

    **Within the frozen window the checkpoints are not resolvably different in true strength**, and
    every one of the 6 seed-arms is still drifting *upward* (slope +0.0010 to +0.0029 per iter) —
    which is STILL CLIMBING showing up in the curve shape rather than in the gate. A point estimate
    of zero must not be pre-registered as "no correction", so the correction is taken at the **95%
    upper bound of the raw estimate, σ_b = 0.0328**, which charges the plateau's own residual climb
    as if it were dispersion.
  - **THE PRE-REGISTERED BIAS TABLE** (200k trials; draws = `iters − anneal_iters`, because every
    anneal-pinned Build 25 arm took its `argmax` inside the frozen window — `v25a` 118/105/114,
    `v25b` 132/125/159 — so the pre-anneal checkpoints never compete):

    | # | contrast | draws | differential bias @ σ_b 0.0328 | @ 0.0428 (Build 25's) |
    |---|---|---|---|---|
    | 1 | `v25b` → `v26a` | 80 → 160 | **+0.0045** | +0.0073 |
    | 2 | `v26a` → `v26b` | 160 → 240 | **+0.0025** | +0.0041 |
    | 3 | `v25b` → `v26b` | 80 → 240 | **+0.0070** | +0.0113 |

    As in Build 25: a contrast must clear **both** `p < α` **and** `diff − bias > 0`, and the
    correction is reported as applied whether or not it binds.
  - **THE LENGTH-RATIO WORRY WAS DIRECTIONALLY RIGHT BUT NEARLY CANCELS.** Because the horizon
    stays at 80 while the arm triples, the *draw* ratio for #3 is **3×** (80 → 240), not the 2× the
    iteration counts suggest — so at a fixed σ_b the correction does grow (+0.0113 at 0.0428
    against Build 25's booked +0.0073). But σ_b in the frozen window is materially smaller than
    Build 25's, and the two effects very nearly cancel: **+0.0070 for 160→320 against Build 25's
    +0.0073 for 80→160**. Re-simulating was still the right call — this is now measured rather
    than assumed.
  - **THE TERMINAL READ IS CO-PRIMARY, NOT DESCRIPTIVE.** Build 25's #2 was pooled-significant but
    **not** seed-robust on the seed-best read (per-seed +0.003 / −0.001 / +0.115, t = +1.03, 2/3),
    while the terminal read was the *stronger* one (t = +2.34, 3/3) — the reverse of the usual
    direction, and evidence the per-step dose is near the resolution of the seed-best read. So the
    same pinned-terminal gate (`scripts/pin_gate_checkpoint.py`, pinned to `iter_160` / `iter_240`
    / `iter_320`) is promoted to **co-primary**: it carries zero selection bias by construction,
    and a contrast is called only where **both** reads agree in sign. A sign disagreement is
    **itself the finding** and is to be booked as one, never resolved in favour of whichever read
    agrees with the hypothesis.
  - **PRE-REGISTERED GATE.**

    | #1 (160→240) | #2 (240→320) | verdict |
    |---|---|---|
    | sig, > 0 | sig, > 0 | **STILL CLIMBING AT 320** — three consecutive doses; the axis still has not saturated and the cost per dose is now the binding question, not the effect |
    | sig, > 0 | not sig | **KNEE BETWEEN 240 AND 320** — book the saturation point and stop extending; this is the anticipated modal outcome |
    | not sig | not sig | **SATURATED AT 160** — the axis closes, and `v25b` stands as the project's best configuration |
    | not sig | sig, > 0 | **NON-MONOTONE** — book outside the anticipated rows; a dose that reappears after a flat step is a red flag for the seed-best read and must be checked against the terminal read before any claim |
    | sig, < 0 | — | **REVERSAL** — more updates cost strength, as Build 25's #4 did for the horizon |
    | entropy → 0, or `vs_iter0` < 0.5 late | any | **COLLAPSE** — report as collapse, never as a NULL |
  - **ATTRIBUTABILITY BAR, unchanged.** The trust region must not bind: `v25b` ran **0/480** and
    `approx_kl_max` stayed under the 0.09 bar. If a Build 26 arm binds at Build 22's rate (59/150)
    the verdict is not attributable to update count and must be reported as such.
  - **COST AND THE TWO OPERATIONAL PREREQUISITES.** Budget off Build 25's *measured* rate, not
    plan.md's earlier 4.8–5.3 assumption: `sacct` on 7188055 shows `v25b`'s 160 iters took
    11:03:35 / 11:23:20 / 11:30:31, i.e. **~4.15–4.3 min/iter** ⇒ `v26a` ~17 h, `v26b` ~23 h.
    6 tasks ÷ 4 concurrent ⇒ ~**40 h wall-clock**.
    (a) **Walltime.** `ppo_seed.slurm` carried `--time=20:00:00`, which `v26b` blows outright;
    raised to **36 h** (medium QoS allows 48 h). Build 25 only survived because it was submitted
    with a 24 h command-line override — `sacct` reports `Timelimit=1-00:00:00` against the file's
    then-20 h — so the script and the real submission had already drifted apart. There is **no
    resume** in `ppo_continue_gate.py`, so a walltime kill loses a whole ~23 h arm.
    (b) **Port base.** `_job_common.sh` computes `8100 + TASK_ID`, so two concurrent `--array=0-2`
    arrays both claim 8100–8102, and colliding tasks **silently share one Showdown server** rather
    than failing. `v26b` must be submitted with `LATEGAME_SHOWDOWN_PORT_BASE=8200`.
    **Disk:** ~29 GB needed (measured: `ppo_v25b_s0` is 2.8 GB / 160 iters ≈ 17.5 MB/iter) against
    **78 GB** free — not the 146 GB quoted above, which is stale.
  - **ATTEMPT 1 FAILED 4/6 ARMS (2026-08-06 → 08-08). Booked here rather than quietly re-run,
    because one of the two causes is a standing limit on how long ANY arm can be.**

    | task | outcome | iters | MaxRSS | cause |
    |---|---|---|---|---|
    | `v26a` s1 | COMPLETED 15:56 | 240/240 | 59.3 GiB | — |
    | `v26a` s2 | COMPLETED 15:48 | 240/240 | 61.8 GiB | — |
    | `v26a` s0 | TIMEOUT 36 h | 140/240 | 37.3 GiB | disk quota |
    | `v26b` s0 | TIMEOUT 36 h | 119/320 | 32.2 GiB | disk quota |
    | `v26b` s2 | FAILED 13:12 | 158/320 | 42.2 GiB | `Disk quota exceeded` on `torch.save` |
    | `v26b` s1 | **OUT_OF_MEMORY** 17:46 | 304/320 | **67.1 GiB** | the 64 GB per-job cap |

    **(a) Disk — a shared 200 GB quota, not a per-project one.** A neighbouring project grew
    86 → 125 GB while Build 26 was adding 20 GB, and the quota hit 195/200 GB. The two arms that
    finished did so within ~16 h, *before* the crunch; the ones still running when it hit slowed
    to 15–18 min/iter (against a measured 3.95–4.0) and one died outright inside `_save_checkpoint`.
    **The pre-run disk check was wrong in kind:** it compared Build 26's own footprint against free
    space at submit time and concluded "disk is not a concern", which is not a claim that survives
    a shared quota over a 40 h run. Freed 49.7 GB by dropping the dead partials and pruning settled
    builds (`v22`–`v24`, `v25a`, `v25c`) to their referenced `best_checkpoint` + terminal only;
    `v25b` and the two completed `v26a` arms were left whole.
  - **(b) THE REAL FINDING: ARM LENGTH IS BOUNDED BY MEMORY, NOT WALLTIME.** RSS grows ~linearly
    with iterations — **40.7 GiB at 160** (Build 25), **59.3 GiB at 240**, i.e. **~0.23 GiB/iter**,
    projecting **~78 GiB at 320**. The `medium` QoS sets `MaxTRES=cpu=8,mem=64G` **per job**, so a
    larger `--mem` is *rejected at submit* rather than queued; `high` allows 128 GB but caps
    walltime at 24 h against a ~21.2 h run, and `default` allows only 4 CPU / 32 GB. So **no QoS
    envelope fits a 320-iteration arm in one process**, and raising walltime — Build 26's original
    fix — could never have worked. `v26b` s1 was killed at exactly the 64 GB cap, 304/320 in.
  - **CONSEQUENCE: `run_ppo` GAINS AN OPT-IN RESUME, AND `v26b` RUNS IN TWO CHUNKS.** A fresh
    process resets RSS, so resuming is what makes an arm past ~260 iterations runnable at all. The
    state file (`resume_state.pt`, written atomically beside the checkpoints) carries Adam's moment
    estimates and both RNG streams — dropping either would restart the optimizer cold mid-arm and
    re-draw the league from the top of the stream. It is kept **separate** from `iter_NN.pt`, which
    every agent and gate reads. `resume` defaults **off**, so Builds ≤25 reproduce unchanged.
    Resuming **requires `--anneal_iters` pinned**: `anneal_horizon` falls back to `iters`, which a
    chunked run raises between chunks, so an unpinned schedule would anneal over a different span
    per chunk — the exact confound Build 24 introduced the flag to remove. `v26b` runs 160, then
    resumes to 320, both at `anneal_iters` 80.
  - **ONE COMPARABILITY CAVEAT, TO BE REPORTED WITH THE RESULT.** `v26a` ran uninterrupted while
    `v26b` runs in two chunks. The schedule is identical (horizon pinned at 80) and optimizer and
    RNG state carry across, so the intended difference is still update count alone — but the two
    arms did not traverse byte-identical code paths, and **#2 (`v26a` → `v26b`) is the contrast
    that carries this**. If #2 comes back near the significance boundary it must be read with that
    in mind rather than as a clean dose.
  - **THE CHUNKED ARM MADE THE ANALYSIS PREFLIGHT UNSOUND, AND IT WAS CAUGHT LIVE (2026-08-11).**
    `build26_analysis.sh` refused on *missing* inputs but only checked that each per-seed JSON
    **existed**. Chunking broke that assumption: both chunks write the same
    `results/ppo_ou_gate_v26b_s<N>.json`, so between them the file is present, well-formed, and
    **160 iterations long**. Observed at 09:32 with chunk 2 at iteration ~290/320: all six arm
    files on disk, all seven inputs reported `ok`, and the script ready to run.
    **What it would have produced is not a weaker result but a fabricated one:** #2 would compare
    240 → **160** instead of 240 → 320 — backwards, i.e. the pre-registered **REVERSAL** row — and
    #3 would compare 160 → 160, a near-null. A verdict of "more updates COST strength", printed
    with every appearance of having passed its checks, from arms that were simply not finished.
    The preflight now also checks **arm length** (each curve must reach its pre-registered
    iteration count), because the existing guard was written when seed *count* was the only way an
    input could be incomplete — arm length only became a variable when memory forced the split.
  - **AND THE SAME SCRIPT WOULD HAVE RUN THE GATE AT 1/10 THE PRE-REGISTERED N.** Found while
    estimating the analysis runtime. `seed_strength_gate.py` defaults to `--n 300` and
    `build26_analysis.sh` left it unset — the identical shape to the `alpha` default its own header
    already guards against. At α = 0.0125 the measured MDE at 80% power is **0.079 / 0.045 / 0.032
    for N = 300 / 900 / 1800**, and Build 25 pre-registered **N = 3000** (MDE ~0.025) precisely
    because "the per-step doses here are plausibly ~0.02–0.04". Build 25's own doses landed at
    +0.0649 and +0.0362, and **Build 26's should be smaller still if the curve is flattening —
    which is the hypothesis under test.** So at N=300 both contrasts would sit under the detection
    threshold, come back NULL, and book the pre-registered **SATURATED AT 160** row: a real row,
    reached by an underpowered test rather than by evidence, and indistinguishable in the output
    from a genuine saturation. Build 26's pre-registration inherits Build 25's design without
    restating an N, so Build 25's split is what carries over: **3000 primary / 1800 co-primary**,
    now pinned in the script and overridable only for a plumbing smoke.
  - **RESULT — KNEE BETWEEN 240 AND 320 (2026-08-11).** Both gates ran clean in one job
    (`7237875`, 3:17:34): primary N=3000 (27,000 battles) and the selection-free terminal read at
    N=1800 (16,200). Pooled vs the heuristic, n=9000/arm: `v25b` **0.6691**, `v26a` **0.7250**,
    `v26b` **0.7420** [0.7329, 0.7509]. On the terminal read: 0.6883 / 0.7354 / **0.7513**.
    **0.7420 (and 0.7513 terminal) is the highest pooled rate in the project's history**, against
    Build 25's 0.6534 / 0.6807. The anchor re-calibrates at 0.6691 vs Build 25's 0.6534 — 0.0157
    apart, inside the ±0.027 cross-run band.
  - **THE THREE CONTRASTS, bias subtracted before any dose is declared, α = 0.0167.**

    | # | contrast | primary (adj) | seeds | terminal (adj) | seeds | call |
    |---|---|---|---|---|---|---|
    | 1 | `v25b` → `v26a` | **+0.0514** (z +8.16) | 3/3, t +2.44 | **+0.0425** (z +5.40) | 3/3, t +3.02 | **WIN, both reads** |
    | 2 | `v26a` → `v26b` | +0.0145 (z +2.58, p 0.0099) | **2/3**, t +0.88 | +0.0134 (z +1.90, p 0.0581) | 3/3, t +1.95 | **not called** |
    | 3 | `v25b` → `v26b` | **+0.0659** (z +10.73) | 3/3, t +5.69 | **+0.0560** (z +7.29) | 3/3, t +3.24 | **WIN, both reads** |

    #1 + #2 = #3 exactly on both reads (0.0559 + 0.0170 = 0.0729; 0.0470 + 0.0159 = 0.0629 ≈ 0.0630).
    *The bias correction was applied and did not bind*, as in Build 25 — though it is no longer
    immaterial: on #2 it eats **15%** of the raw effect (+0.0170 → +0.0145) against 8% on #1, which
    is what the pre-registration anticipated when it insisted the correction be re-simulated rather
    than reused.
  - **#1 SIGNIFICANT AND POSITIVE, #2 NOT CALLED ⇒ KNEE BETWEEN 240 AND 320** — the row the
    pre-registration named as the anticipated modal outcome, reached without having to leave it.
  - **THE TWO READS AGREE IN SIGN AND IN SIZE; THEY DISAGREE ONLY ON POWER, AND THAT IS NOT A
    FINDING.** #2's effect is +0.0145 (primary) and +0.0134 (terminal) — the same number twice. What
    differs is `n`: the primary read carries 9000/arm and the terminal 5400/arm, so an identical
    effect gives z +2.58 there and z +1.90 here. There is **no sign disagreement to book** — the
    case the pre-registration promoted the terminal read to co-primary in order to catch did not
    arise. What did arise is that **#2's true effect sits at the resolution limit of this design**:
    ~+0.014 against a per-contrast MDE of ~0.025 at N=3000. Calling it either way would be reading
    the sample size, not the model, so it is booked as not called.
  - **THE SEED-ROBUSTNESS PATTERN INVERTS, EXACTLY AS BUILD 25 WARNED.** On #2 the primary read is
    pooled-significant but only **2/3** seeds (+0.031 / +0.013 / +0.003, t +0.88), while the
    terminal read is pooled-null but **3/3** (t +1.95) — the same reversal Build 25 saw, and the
    reason the terminal read was promoted. Neither read is "the strong one" here; they are
    measuring a dose that is simply small.
  - **THE DOSE-RESPONSE CURVE IS NOW FOUR POINTS, AND THE MARGINAL RETURN DECAYS MONOTONICALLY.**
    Per *update*, ×10⁻³: **1.62** (80→120) → **0.91** (120→160) → **0.64** (160→240) → **0.18**
    (240→320). A ~9× decay first-to-last, monotone at every step. That is the real content of this
    build: not "320 is better than 240" but **the axis is saturating, and the cost per point is now
    the binding question rather than the effect.** `v26b`'s `best_iter` was 300 / 315 / 292 — still
    late in the arm, so the curve-side read does not contradict the gate, but the gain being bought
    at that end is ~4× cheaper per update than at 160.
  - **ATTRIBUTABLE, AND NO COLLAPSE.** The trust region **never bound**: `0/240` on all three
    `v26a` arms and `0/320` on all three `v26b` arms, with `approx_kl_max` 0.052–0.072 all under
    the 0.09 bar — against Build 22's verdict-costing 59/150. Final entropy 0.418–0.487 and
    `vs_iter0` **1.00 on all six**. So the NULL on #2 is attributable to update count and not to a
    binding optimizer constraint.
  - **THE COMPARABILITY CAVEAT LANDED ON EXACTLY THE CONTRAST THAT WAS PRE-REGISTERED TO CARRY
    IT.** `v26a` ran uninterrupted; `v26b` ran 160 → resume → 320. #2 is the `v26a` → `v26b`
    contrast, and #2 is the one that came back ambiguous. The schedule was pinned and optimizer/RNG
    state carried across, so the intended difference is still update count alone — but this is the
    build where "if #2 comes back near the significance boundary it must be read with that in mind"
    stops being hypothetical. It came back at p 0.0099 / 0.0581 across the two reads.
  - **THE RESUME MECHANISM IS VALIDATED, AND THE MEMORY FINDING HOLDS.** Chunk 2 peaked at
    **41.6–44.0 GiB** for a 320-iteration arm, against the **67.1 GiB** that OOM-killed the
    unchunked attempt at iteration 304 and against the 64 GB `medium` cap. A fresh process really
    does reset RSS: the arm that no QoS envelope could fit in one process finished with ~20 GiB of
    headroom. Elapsed 9:36–11:00 per chunk-2 task.
  - **The analysis is now one submission.** `scripts/cluster/build26_analysis.slurm` starts its own
    Showdown server (on port base **8400**, clear of the PPO bases 8100–8102 / 8200–8202 and of
    `eval_ladder.slurm`'s 8300) and runs the preflight *before* claiming a node. `build26_analysis.sh`
    never started a server of its own, and at the pre-registered N it is ~43,200 battles ≈ **3 h** at
    the 242 battles/min Build 25 measured — long enough that "start the server by hand first" is
    exactly the step that gets skipped.

- **M5 / G1 — LIVE PLAY, built and verified end-to-end (2026-08-10).** Every build through 26 plays
  a *fixed* baseline on a local server, where win rate is the sufficient statistic. G2 is stated in
  GXE/Glicko-1 instead, and those are ladder metrics. `lategame/live/` is the missing half: three
  modes (`challenge` / `accept` / `ladder`), a session supervisor, and Glicko-1/GXE telemetry,
  driven by `lategame live`. Verified against a real local server — two players, real websocket,
  real battle, the finalize sweep, GXE computed, results file written. **Three findings, each a
  case where the obvious implementation is silently wrong.**
  - **`ShowdownException` NEVER REACHES THE CALLER'S `await`.** poke-env dispatches each message in
    a detached `asyncio` task whose result nobody retrieves, so the exception raised on
    `|nametaken|` is never re-raised: a failed login **hangs** rather than failing. A `try/except`
    around the connect would be dead code. Detection is therefore a log handler (`LoginWatch`) plus
    a watchdog that samples `(n_finished_battles, Σ turn counters)` — a slow battle still advances
    turns and a dead socket advances neither, which is the distinction a flat wall-clock timeout
    cannot make. `listen()` likewise swallows a closed socket, and there is no reconnect anywhere
    in the library, so "recovery" means building a **new** player and merging records by battle tag.
  - **`battle.rating` IS PRE-BATTLE ELO, NOT POST-BATTLE GLICKO — and it is `None` when you would
    naturally read it.** poke-env fires `_battle_finished_callback` on the `|win|` line, while the
    `|raw| ...'s rating:` lines are parsed *afterwards* in the same loop, so a rating read at
    callback time is always `None` even on a rated game. Hence the mandatory `finalize` sweep that
    re-reads every battle before the final write; without it the rating columns would be uniformly
    empty and look exactly like an unrated session. And what the field holds is
    `int(rating_info[:4])` — the *first* number on the raw line, i.e. the rating **before** the
    game — which Showdown reports as **Elo**. It is recorded as `showdown_elo_before` and kept
    strictly out of the Glicko math: feeding an Elo into a Glicko update is a units error dressed
    up as precision.
  - **`rate_win_rate` CANNOT REPRESENT A TIE**, because it reconstructs wins as
    `round(win_rate * n)` and can only express scores in {0, 1}. So `summarize` builds its Glicko
    results list directly; a test pins that the two agree exactly on tie-free records. **The same
    bug exists one layer down in `poke_env.player.cross_evaluate`**, which reports
    `n_won_battles / n_finished_battles` — a tie drags win rate down without counting as a loss.
    That one is *booked, not fixed*: `eval/arena.py` was on the frozen path of the in-flight Build
    26 jobs, and its callers are win-rate gates on a format where singles ties are vanishingly
    rare. `eval/ladder.py` below avoids `cross_evaluate` for exactly this reason.
  - **The ladder gate holds at all three levels.** No ack fails; a wrong phrase is rejected by
    `argparse` `choices` rather than silently falling back to a non-ladder mode; and the right
    phrase without `LATEGAME_LIVE_ALLOW_LADDER=1` still refuses, with the policy note. Requiring
    both channels is the point: a CLI flag cannot be inherited from a stale export, and an
    environment variable cannot be picked up from shell history.
  - **G2 was still uncomputed after this build, and that is a measurement fact, not a gap.** A live
    *session* produces GXE, but `summarize`'s default pins every opponent at `REFERENCE`, under
    which GXE is a monotone reparameterisation of the score rate — the same degeneracy
    `eval/rating.py` warns about. Only a **varied field** makes it a measurement. Hence the next
    entry.

- **R-LADDER — the agent-only eval ladder: G2's headline metric, computed without touching the
  human ladder (2026-08-10).** §12's last line asks for evaluation "on a private/agent-only server
  or eval ladder wherever possible", and §16 Q5 is now answered: there is no unranked public
  ladder, so that route is the only one. `lategame/eval/ladder.py` (`lategame eval-ladder`) plays a
  round-robin over a heterogeneous field on the local server and fits every rating **jointly**.
  New file rather than an extension of `eval/rating.py` or `eval/arena.py`, both of which were on
  the frozen path of the running Build 26 jobs; it imports them and adds nothing to them.
  - **A SINGLE GLICKO PERIOD OVER A ROUND-ROBIN IS DEGENERATE, which is the whole difficulty.**
    Every agent starts at (1500, 350). Update each against opponents held at their priors and every
    opponent *is* the reference, so each rating collapses to a monotone function of that agent's own
    score rate — the same degeneracy reached by a longer road. The fit must iterate, using the
    current estimates as opponent ratings. At the fixed point every agent satisfies
    `Σ_j g(RD_j)(s_j − E_j) = 0`, the **Bradley-Terry score equation**, so what this computes is the
    BT/Elo maximum-likelihood fit reported on the Glicko scale. A two-agent field is checked against
    the closed form `−400/g(350) · log₁₀(1/s − 1)` to 1e-6.
  - **THE UNDAMPED ITERATION DOES NOT CONVERGE — IT OSCILLATES, and worst at two agents.** Because
    every agent is updated against the *previous* sweep (Jacobi, which keeps the result independent
    of walk order), a rating *difference* is corrected from both ends at once: with two agents the
    new difference overshoots to `2·target − current`, an error that flips sign and keeps its
    magnitude forever. The Glicko step is exactly a Newton step on the score equation, so the
    iteration matrix is `I − D⁻¹H` with `H` the BT Hessian; its rows sum to zero, putting `D⁻¹H`'s
    eigenvalues in `[0, 2]` with the 2 attained exactly at k=2. **Damping by 0.5 maps them to
    `[0, 1]`** — not a fudge factor but the value that exactly cancels the double correction. The
    residual eigenvalue 1 is the gauge direction, removed by the anchor. Measured: undamped fails to
    converge in 200 sweeps and walks to the clamp; damped converges in 8.
  - **THE FIT NEEDS A GAUGE FIX, and which agent is pinned is a real choice.** A round-robin matrix
    identifies rating *differences* only — add 100 to everyone and the likelihood is unchanged — so
    the fixed point is not unique. `heuristic` is pinned at 1500 because it is the fixed baseline
    every build from M1 onward is already reported against, which makes the ladder's Glicko
    commensurable with the entire win-rate history and makes GXE read as "expected score vs a
    heuristic-strength average player". Gauge invariance is pinned by test: a different anchor
    shifts every rating by one constant and leaves all differences unchanged.
  - **RD IS COMPUTED ONCE, AT CONVERGENCE.** Running a Glicko period per sweep would shrink RD on
    every sweep over the *same* games and manufacture certainty — an agent's deviation would depend
    on how many iterations the solver happened to take. The sweeps move means only; the deviation
    comes from a single period against the converged field. Pinned by a test that runs the solver to
    two different tolerances and requires the RDs to agree.
  - **SCORING A PPO CHECKPOINT THROUGH THE `ppo` AGENT MEASURES THE WRONG POLICY, and it is the
    natural thing to type.** `PPORecordingAgent` forces `sample=True` — its docstring says sampling
    is mandatory, because PPO needs the on-policy action distribution rather than greedy arg-max.
    Every published number in this project reads a PPO checkpoint through **`offrl`**
    (`seed_strength_gate.py:185`, `ppo_continue_gate.py`), because greedy is the deployed policy.
    Measured head-to-head on `v25b`'s terminal checkpoint vs the heuristic, n=120: **0.675 sampled
    against 0.767 greedy, a ~9-point gap.** This was caught by a smoke run whose standings came out
    inverted. `parse_field` now **refuses** `ppo@<checkpoint>` and names `offrl@` as the fix, rather
    than documenting the trap — a ~9-point systematic bias on every learned entry is not something a
    reader can be expected to notice in a table.
  - **NON-TRANSITIVITY IS A STATED LIMITATION, not a bug.** Bradley-Terry gives each agent one
    latent strength, so it cannot represent cycles: in a perfect rock-paper-scissors field the fit
    rates everyone equal, and says so. Pokémon carries real non-transitivity (team archetypes
    counter each other), so a cluster of near-equal ratings can mean "cyclic" rather than "equally
    strong". The full win matrix is written alongside the standings for exactly this reason.
  - **WHAT IT IS NOT, recorded in the results file itself so a published number carries its own
    basis.** (1) Not comparable to a Showdown GXE, which is measured against **humans** on the
    public ladder. (2) Not a replacement for `scripts/seed_strength_gate.py`, which remains the
    authoritative build-vs-build statistic. The default field also **excludes the in-flight `v26a`
    arms** on purpose: Build 26's verdict comes from the pre-registered gate, and a mid-flight arm
    in a rating table invites exactly the reading pre-registration exists to prevent.
  - **RESULT — G2's headline metric is now COMPUTED (2026-08-10).** `gen9ou`, 8 agents, 28 pairs
    × **300** battles = **8,400**, `loop_penalty` 4 (the authoritative gate's value, not 0), anchor
    `heuristic` = 1500, converged in 85 sweeps, every pair at full `n`, nothing clamped or
    unbounded. `results/eval_ladder_gen9ou.json`.

    | agent | W–L | score | Glicko | RD | GXE |
    |---|---|---|---|---|---|
    | `offrl@ppo_v25a_s0/iter_120` | 1597–503 | 0.760 | **1715.5** | 15.0 | **0.6962** |
    | `offrl@ppo_v25b_s0/iter_160` | 1570–530 | 0.748 | **1696.9** | 14.9 | **0.6809** |
    | `offrl@ppo_v23b_s0/iter_80` | 1532–568 | 0.730 | 1671.1 | 14.7 | 0.6590 |
    | `simpleheuristics` | 1394–706 | 0.664 | 1579.2 | 14.6 | 0.5756 |
    | `heuristic` (anchor) | 1276–824 | 0.608 | 1500.0 | 14.8 | 0.5000 |
    | `maxbasepower` | 636–1464 | 0.303 | 941.4 | 21.3 | 0.1044 |
    | `random` | 358–1742 | 0.171 | 575.1 | 26.2 | 0.0277 |
    | `offrl@offrl_gen9ou_wide_s0` | 37–2063 | 0.018 | −23.5 | 43.9 | 0.0029 |

    **The ordering reproduces the entire win-rate history**, which is the check that matters: the
    learned builds sit above `simpleheuristics` above `heuristic` above the naive bots, and the PPO
    warm-start init lands at the bottom — consistent with OU Builds 2–4, which found that
    checkpoint functionally dead on OU (it loses 37/2100 here, and `random` beats it).
    **An independent reproduction of a published number:** `simpleheuristics` vs `heuristic` comes
    back **0.623** against Lever 15's **0.633 [0.577, 0.686]**, well inside the CI.
  - **WHAT THIS DOES AND DOES NOT SETTLE.** It settles G2's *headline metric*: `v25b` is
    **Glicko 1696.9 ± 14.9, GXE 0.6809** on a varied field. It does **not** resolve the ordering
    among the top three. Two independent `n=150` runs preceded this one, and the `v25a`↔`v25b`
    pairing moved **0.460 → 0.587** between them — a 0.127 swing, ~3.1 binomial SE — while each
    agent's own GXE moved under 0.004. **The battles are not independent Bernoulli trials:** a
    12-team pool plus archetype counters clusters them by team matchup, so the effective sample is
    well under `n` and the reported **RD is a lower bound, not an interval.** The standings
    separate *bands*; they do not order neighbours. That is also consistent with the authoritative
    gate, where Build 25's 120→160 contrast was pooled-significant but **not** seed-robust — and
    the ladder is single-seed (`s0`), so it could not speak to build-vs-build regardless. This
    caveat is written into the results file's own basis string.
  - **ONE NUMERICAL COINCIDENCE, BOOKED AS ONE SO IT IS NOT MISREAD AS VALIDATION.** `v25b`'s GXE
    is **0.6809** and its published terminal win rate vs the heuristic is **0.6807**. These are
    *different quantities*: GXE discounts for the reference player's RD 350, whereas the published
    figure is a direct win rate against one specific opponent — and `v25b`'s direct head-to-head in
    this very run was **0.720**, not 0.681. The agreement to 2e-4 is a coincidence of magnitude,
    not an identity, and must not be cited as the ladder reproducing the gate.

- **THE LADDER, RE-RUN ON THE FINISHED BUILD 26 ARMS — G2's HEADLINE REFRESHED, AND THE FIRST
  MEASUREMENT OF WHAT THE STANDINGS CANNOT ORDER (2026-08-11).** The 2026-08-10 field deliberately
  excluded `v26a`/`v26b` as in-flight arms. They are no longer in flight, and `v26b` is the
  project's best configuration while G2's published Glicko still pointed at `v25b`. Re-run on
  `gen9ou` with the dose ladder as the field (9 agents, 36 pairs × **300** = **10,800** battles,
  `loop_penalty` 4, anchor `heuristic` = 1500), plus **300 cluster-bootstrap resamples** over the
  78 team matchups. `results/eval_ladder_gen9ou_v26.json`.

  | agent | score | Glicko | RD | GXE | cluster 95% CI |
  |---|---|---|---|---|---|
  | `offrl@ppo_v26a_s0/iter_240` | 0.696 | **1776.3** | 12.5 | **0.7434** | [1732, 1811] |
  | `offrl@ppo_v26b_s0/iter_320` | 0.696 | **1776.3** | 12.5 | **0.7434** | [1739, 1815] |
  | `offrl@ppo_v25a_s0/iter_120` | 0.679 | 1755.0 | 12.4 | 0.7275 | [1719, 1791] |
  | `offrl@ppo_v25b_s0/iter_160` | 0.642 | 1710.7 | 12.2 | 0.6924 | [1676, 1743] |
  | `offrl@ppo_v23b_s0/iter_80` | 0.593 | 1653.4 | 12.1 | 0.6435 | [1618, 1689] |
  | `simpleheuristics` | 0.525 | 1572.7 | 12.3 | 0.5695 | [1546, 1600] |
  | `heuristic` (anchor) | 0.465 | 1500.0 | 12.6 | 0.5000 | — |
  | `maxbasepower` | 0.166 | 1001.4 | 19.0 | 0.1280 | [938, 1053] |
  | `random` | 0.039 | 617.0 | 28.6 | 0.0325 | [530, 684] |

  - **G2's headline metric moves to `v26b`: Glicko 1776.3, GXE 0.7434**, from `v25b`'s 1696.9 /
    0.6809. Both caveats travel unchanged — agent-only field, not comparable to a Showdown GXE.
  - **RD WAS A LOWER BOUND, AND NOW THERE IS AN INTERVAL THAT IS NOT.** RD is derived from binomial
    noise alone. On a teambuilt format the battles are clustered by team matchup, so the effective
    sample is well under `n` — the evidence was already on the record (a pairing moved 0.460 →
    0.587 across two n=150 runs, ~3.1 binomial SE, while each agent's own GXE moved under 0.004)
    and unexplained by RD. `eval/ladder.py` now resamples **team matchups**, not battles, and
    refits the whole field per resample. Resampling battles would merely reproduce the binomial
    interval RD already gives; pinned by a test where a 1000-battle field whose outcomes are fixed
    by 10 matchups comes back >5× wider than RD implies, against an unclustered control that does
    not.
  - **THE RESULT IS THAT EVERY BAND BOUNDARY SEPARATES AND NOT ONE LEARNED NEIGHBOUR DOES.**
    learned > `simpleheuristics` > `heuristic` > `maxbasepower` > `random`: all four separated.
    `v26a`/`v26b`, `v26b`/`v25a`, `v25a`/`v25b`, `v25b`/`v23b`: none separated. The instruction
    "read the standings as separating BANDS, not as ordering neighbours" has been in the results
    file since it was written; this is the first run that **measures** it rather than asserting it.
  - **ONE COINCIDENCE, BOOKED AS ONE SO IT IS NOT MISREAD AS A BUG.** `v26a` and `v26b` post
    *identical* totals (1671–729), hence identical Glicko and GXE. They are demonstrably different
    agents: all seven per-opponent records differ and they went **165–135** head to head. `v26b`'s
    +30 wins across the rest of the field (chiefly **+38** vs `simpleheuristics`) exactly cancel
    its −30 head to head. A coincidence of arithmetic, not a shared checkpoint.

- **OPEN NEXT (BUILD 27) — DO NOT EXTEND THE DOSE AGAIN; the axis is closed and the remaining
  goals are elsewhere.** Every build from 16 onward named its successor; Build 26 was the first
  that did not, because its own result removed the obvious one.
  - **Why 480 updates is not Build 27.** At the observed 0.18×10⁻³/update, 320 → 480 buys **+0.029
    nominal**. But the decay ratio across segments runs ~1.8× → 1.4× → 3.6×, so the next segment
    projects to ~0.05×10⁻³/update ⇒ **~+0.008**, against a per-contrast MDE of ~**0.025** at
    N=3000. The arm is powered to detect approximately nothing. It would also cost ~33 h per seed
    across 2–3 resume chunks and ~25 GB of checkpoints — on the shared quota that already cost
    Build 26 four of six arms. Spending the entire remaining budget to measure an effect the curve
    predicts is 3× under the resolution limit is the one move this build's own data forbids.
  - **What is actually left, against the goals rather than against the lever list.** G1 met; G2 met
    on both halves and refreshed above; G3 booked above. That leaves **G4** (≥3 formats — VGC
    doubles is the missing third) and **G5** (the skill stack as *demonstrated* rather than merely
    implemented capabilities), plus **M7/§16 Q2**, where test-time search was retired at parity on
    gen9-RB — a format Lever 15 subsequently proved FORMAT_BOUND, i.e. the retirement was measured
    where nothing could beat the heuristic and says nothing about OU.

- **G4 GROUNDWORK, AND A MEASUREMENT THE OBVIOUS PROBE CANNOT MAKE (2026-08-11).** Following the
  Lever-15 idiom — measure a format's ceiling before spending on it — the VGC probe was built
  before the doubles pipeline. Three findings, in increasing order of consequence.
  - **`config.VGC_FORMAT` NAMED A FORMAT THAT DOES NOT EXIST.** It was `gen9vgc2024regh`, written
    as a Phase-3 placeholder and never exercised, so nothing ever failed. The pinned simulator
    (rev 393d5c8) offers gen 9 VGC 2023 Reg C/D, **2024 Reg G**, **2025 Reg I**, and the
    `[Gen 9 Champions]` 2026 Reg M-A/M-B mod — no Reg H. Corrected to **`gen9vgc2025regi`** and
    pinned by a test against the simulator's own format table. The `...bo3` ids are avoided
    deliberately: a "battle" there is a best-of-three **series**, which every win/loss counter in
    `eval/` would miscount without noticing. (`bestOfDefault: true` on Reg I turns out to be only
    a client-side UI bit — the enforced Bo3 comes from a `'Best of = 3'` ruleset entry that only
    the explicitly-named Bo3 formats carry.)
  - **A SINGLES-ONLY AGENT FAILS *SILENTLY* ON DOUBLES, so the refusal is at build time.** On a
    `DoubleBattle` poke-env passes `active_pokemon` as a list, so `HeuristicAgent` reaches
    `list.types` and raises `AttributeError`. That exception never reaches a caller: poke-env
    dispatches every protocol message through `asyncio.create_task` and only attaches
    `add_done_callback(discard)`, so the raise is logged and swallowed — the same detached-task
    pathology M5/G1 found for `ShowdownException`. The agent then simply never answers the request
    and the **server plays a default move for it on the timer**. A ceiling probe anchored on an
    agent in that state would have measured the timer, read it as enormous headroom, and been
    wrong in the most expensive possible direction. `arena.build_player` now refuses a singles-only
    agent on a doubles format, where the format string is in hand and the error is synchronous.
  - **THE PLANNED CHEAP PROBE CANNOT DECIDE G4, AND THE REASON IS IN THE PUBLISHED BANDS.** The
    plan was to anchor the VGC M1 sweep on `simpleheuristics` — poke-env's own bots are all
    doubles-native, so this needs no doubles code of ours. Comparing the RB and OU bands shows why
    that measures the wrong thing:

    | vs `heuristic` | `random` | `maxbasepower` | `simpleheuristics` |
    |---|---|---|---|
    | gen9-RB (FORMAT_BOUND) | 0.007 | 0.107 | **0.523** |
    | gen9ou (headroom) | 0.030 | 0.060 | **0.643** |

    The two formats are **indistinguishable at the bottom** of the skill gradient — a competent bot
    crushes naive bots in both. They differ *only at the top*: on RB the last increment of skill
    buys parity, on OU it buys 0.14. A probe anchored on `simpleheuristics` can only measure
    `random`/`maxbasepower` *below* it, i.e. precisely the region where FORMAT_BOUND and
    MODEL_BOUND look the same. **The discriminating measurement needs two agents near the top of
    the doubles gradient, and poke-env supplies exactly one.** So the VGC ceiling probe is blocked
    on a doubles-capable `HeuristicAgent` — which is not scope creep but M1's deliverable for a
    third format, and is the baseline any learned doubles agent would be scored against anyway.
    A validator-checked `gen9vgc2025regi` team pool (**10/10 legal**, built by the existing
    `scripts/build_ou_teampool.py` with no new code) supplies the teams.
  - **THE DOUBLES M1 BASELINE, BUILT.** `HeuristicAgent` now applies the same R-CALC rule per slot
    and joins the two into a `DoubleBattleOrder`. Doubles forces two things singles never did, both
    silent failures if got wrong: a move needs a legal **target** (chosen together with the move,
    since expected damage depends on which foe it lands on, and only used when
    `get_possible_showdown_targets` agrees — a rejected target costs the turn), and the two slots
    **may not switch to the same benched Pokemon** (an illegal order the server answers with a
    default move). Verified live: **0.900 vs `random` over 10 battles, 10/10 finished, 7.1 mean
    turns in 4 s** — fast enough to rule out the timer-default failure mode above, which is slow
    by construction.

- **VGC CEILING PROBE — RESULT: `INSUFFICIENT`, and the instrument is the reason (2026-08-11).**
  M1 on `gen9vgc2025regi`, n=300 per matchup, the 10-team pool, `results/format_ceiling_gate_vgc.json`:

  | vs `heuristic` | gen9-RB (FORMAT_BOUND) | gen9ou (headroom) | **gen9vgc2025regi** |
  |---|---|---|---|
  | mirror (sanity) | 0.513 | 0.493 | 0.523 |
  | `simpleheuristics` | **0.523** [0.467, 0.579] | **0.643** | **0.527** [0.470, 0.582] |
  | `maxbasepower` | 0.107 | 0.060 | 0.300 |
  | `random` | 0.007 | 0.030 | 0.017 |

  - **On the discriminating quantity VGC reproduces the gen9-RB signature almost exactly.** The
    strongest competent bot reaches **0.527 with a CI spanning 0.50** against RB's 0.523 [0.467,
    0.579] — and against OU's 0.643, whose CI clears the 0.58 headroom bar entirely.
    `simpleheuristics` is not even distinguishable from the heuristic's own **mirror** (0.523).
  - **But the verdict is INSUFFICIENT, not FORMAT_BOUND, and the distinction is the whole point of
    running this honestly.** Two readings fit the same numbers. (a) VGC's achievable ceiling over a
    competent heuristic really is ~parity, as on RB. (b) **Both bots are equally blind to
    doubles-specific skill** — target selection, Protect timing, speed control, spread-damage
    positioning, and the bring-6-pick-4 team-preview decision — so the instrument cannot resolve a
    gap that exists above both of them. `maxbasepower` at **0.300** on VGC against 0.107 on RB and
    0.060 on OU is the tell: a *naive* bot is far closer to competent play here than on either
    singles format, which is what reading (b) predicts and what a genuinely flat ceiling would also
    produce.
  - **Nothing breaks the tie, unlike on RB.** Lever 15's FORMAT_BOUND verdict rested on three legs
    — M1's skill band, **M2** (near-optimal depth-2 search with a white-box opponent model reaching
    0.500), and **M3** (team strength does not predict the winner, AUC 0.495). Only M1 exists for
    VGC: there is no doubles forward model, so no M2, and no scraped VGC replays, so no M3. On RB,
    M2 was what turned a suggestive band into a verdict.
  - **So G4 is neither greenlit nor descoped by this build, and that is the finding.** The
    Lever-15 idiom — measure the ceiling before spending on the pipeline — worked on RB because a
    cheap probe there was *decisive*. On doubles the same probe is **not decisive**, because every
    agent cheap enough to run before building the pipeline is a singles policy applied per slot.
    Deciding G4 by measurement therefore costs more than the idiom promised: it needs either a
    doubles M2 (extend `lategame/search/` to doubles — the forward model already serialises and
    steps arbitrary battles) or a doubles-competent reference to put at the top of the gradient.
    Recording the cost honestly is better than reading a suggestive M1 as a verdict it cannot bear.

- **BUILD 27 / GATE B — SEARCH DOES NOT COMPOUND ON gen9ou EITHER (NULL), AND THE HEAD-TO-HEAD
  SAYS IT IS MILDLY HARMFUL (2026-08-12).** L11–L14 retired test-time search at parity, but every
  one of those ran on gen9-RB — which Lever 15 then proved FORMAT_BOUND, where near-optimal search
  reaches 0.500 because *nothing* beats the heuristic. That retirement was therefore measured
  somewhere it could not have come out otherwise. This re-runs it where headroom is proven and the
  base is 0.77. Pre-registered before running; `results/rpredict_search_ou.json`.
  - **THE ARM WAS THE STRONGEST ONE AVAILABLE, DELIBERATELY.** Depth-2 expectimax with
    `opp_aggregation="model"` and the **white-box** opponent model — the eval opponent *is*
    `HeuristicAgent`, so it is modeled exactly (L14 measured that model at 0.958 agreement). Same
    checkpoint on both sides (`ppo_v26b_s0/iter_300`), so any gap is the search procedure and not
    the weights. n = **2500/arm** (MDE ≈ 0.025, Build 26's resolution), pooled from a 10-shard
    array because search is serial in its node driver at ~29 s/battle.
  - **THE RESULT.**

    | | rate | n |
    |---|---|---|
    | base (greedy `v26b`) vs `heuristic` | 0.7688 | 2500 |
    | depth-2 search vs `heuristic` | 0.7724 | 2500 |
    | **contrast** | **+0.0036** (z 0.303, p 0.76) | — |
    | search vs its own base, head-to-head | **0.3932** | 2500 |
    | search vs `random` (sanity) | 0.895 | 400 |

    The contrast is **NULL**, and not marginally: it is positive in **6 of 10** shards, which is a
    coin flip. **VERDICT: NULL** on the pre-registered rule.
  - **THE HEAD-TO-HEAD IS THE REAL CONTENT, AND IT IS NOT A NULL.** Search loses to the greedy
    policy it descends from at **0.3932** — **10.7 SE** below parity, p ≈ 1.3e-26, and below 0.500
    in **10 of 10** shards. So search is not neutral: it is *worse*, unanimously.
  - **WHY BOTH CAN BE TRUE, AND IT IS AN INTRANSITIVITY.** Search and its base are
    indistinguishable against the heuristic while the base beats search decisively head-to-head.
    Against an opponent that loses ~77% there is slack — a slightly worse policy still converts.
    Against an opponent strong enough to punish them, the same deviations cost games. The lesson is
    that **measuring search only against the fixed baseline would have reported "no effect" and
    missed that the effect is negative**; it took scoring search against its own base to see it.
    Note also that no single latent strength can represent this — exactly the limitation
    `eval/ladder.py`'s Bradley-Terry docstring warns about.
  - **THE ESCAPE HATCH L11/L12 NAMED IS NOW CLOSED.** Those builds retired search with "the
    opponent model was too weak" as the stated residual. Here the opponent model is *exact*, the
    forward model is validated at **0 mismatches on this very format** (Gate A′), the format has
    proven headroom, and the base is the strongest checkpoint the project has. Search still does
    not help. **§16 Q2 is answered — ship policy-only — and M7 closes on the record**: six
    independent mechanisms (gradient, depth-1, depth-2, curriculum, real-opponent-model search, and
    now depth-2-with-exact-model on a format with headroom).


- **BUILD 28 / B6 — THE VGC CAMPAIGN, AND THE SHARD THAT WAS 94% ONE BUG (2026-08-14 → 2026-08-15).**
  G4 was booked MET at `4303803` on the "playable end-to-end without rewriting the core" criterion.
  This build is the *strength* campaign that criterion deliberately did not require: BC → offline
  RL → PPO on `gen9vgc2025regi`, reusing `train/` and the `ppo_continue_gate` /
  `seed_strength_gate` / `merge_gate_seeds` toolchain. It is recorded here in one entry because
  B6a–B6e landed as commits without a notebook section, and because B6f's first act was to
  invalidate two of them.

  - **B6a — THE DOUBLES ACTION CODEC IS FACTORED, 2×107, NOT A JOINT 11,449** (`b27c16a`).
    `DoublesEnv` is 107 actions *per slot* and a turn commits both, so the joint space is
    107² = 11,449 outputs almost none of which are ever legal together. The head emits 214 logits
    read as two independent distributions — which is also the shape poke-env's own converters
    speak. One constraint COUPLES the slots and a factored mask cannot express it: both slots may
    not switch to the same benched Pokémon. Showdown answers such an order with a default move
    rather than an error, so it is a silently lost turn; `joint_switch_conflict` is the separate
    check every decoder and sampler has to apply. Build 5's canonical-move-order divergence is
    carried across unchanged.
  - **B6b — THE DOUBLES ENCODER, AND THE SINGLES LAYOUT FROZEN** (`9b196a1`). `OBS_DIM_DOUBLES`
    888 / `d1-09831e17c378`: the singles blocks reused verbatim, with move blocks per *active
    slot* (8 rather than 4) and a 12-scalar global block carrying `force_switch`, `maybe_trapped`,
    `first_turn` and the four targeting-context flags. `EntityTransformer._layout_for` selects the
    layout from `input_dim` alone, which is the seam that lets one architecture serve three
    formats.
  - **B6c — G4 MET: THREE FORMATS PLAY END TO END THROUGH ONE CORE** (`4303803`). The learned
    doubles agent, the arena's doubles-safety guard, and the exit criterion booked.
  - **B6d / B6e — BC AND FACTORED AWR ON VGC** (`550c524`, `1c28dec`). Six data-path defects found
    by measurement, each of which would have produced a trained model rather than an error: the
    ignored `force_switch` on a partial replacement; the "no mon here" guard rejecting the one slot
    being asked to act; the half-default read as poke-env's whole-order `-2` sentinel; the action
    mask read by VALUE instead of by index (a 2-legal-action mask on a turn with 4 moves and 2
    switches, driving BC's loss to 9.4e8); forced single-legal-action turns trained on as if they
    were decisions; and illegal-label rows dominating the AWR actor loss at 719,296.
  - **THESE TWO RESULTS ARE WITHDRAWN, AND B6f's FIRST ACT WAS FINDING OUT WHY.** See the next
    entry. The defect list above stands; the numbers those commits reported do not.

- **B6f STAGE A — 94.2% OF THE VGC SHARD CAME FROM 100 OF ITS 899 EPISODES, AND THE CAUSE WAS ONE
  LINE OF OURS (2026-08-15).** `results/vgc_loop_probe.json`. Before porting the policy gradient to
  a factored head, the shard it would warm-start from was measured directly. It should have been
  measured before B6d.

  | | `vgc_rl.npz` | `gen9ou_v7_rl.npz` |
  |---|---|---|
  | turns / episode, median | 7 | 19 |
  | turns / episode, max | **12,795** | 205 |
  | max ÷ median | **1,827.9** | 10.8 |
  | top-decile turn share | **0.922** | 0.239 |
  | episode-length Gini | **0.901** | 0.304 |
  | unique observations ÷ turns | **0.0033** | 1.000 |

  - **THE LONGEST EPISODE CARRIES 12,795 RECORDED TURNS OVER SEVEN UNIQUE OBSERVATION VECTORS.**
    51.9% of all turns are one signature — slot 0's legal set is `{pass}`, slot 1's is two switches
    — and 96.6% of all rewards are exactly zero.
  - **SPLIT BY EPISODE LENGTH THE PICTURE INVERTS.** The non-loop episodes (≤40 turns) are 5,553
    turns at decision density **0.906** and 14.71 legal actions per slot, against the whole shard's
    0.571 and 2.62. So VGC doubles is a normal, decision-DENSE format — denser per slot than OU's
    7.67 — and the "98.4% of recorded turns had exactly one legal action per slot" note written
    into `data/collect.py` measured the loop, not the format. Corrected in place.
  - **WHICH AGENT LOOPS, MEASURED RATHER THAN ASSUMED.** Four battles per pair, counting
    `choose_move` calls against the `battle.turn` actually reached:

        random           vs random               19 calls   turn 17
        simpleheuristics vs simpleheuristics      9 calls   turn  8
        maxbasepower     vs maxbasepower          8 calls   turn  6
        random           vs HEURISTIC           4001 calls   turns 1..7
        HEURISTIC        vs HEURISTIC           1153 calls   turns 1..4

    poke-env's own baselines never loop. Ours did — and `heuristic` is in every collection pool,
    which is the entire explanation.
  - **THE BUG.** `heuristic_agent._choose_doubles_move` gave a slot with no decision a
    `DefaultBattleOrder()`. That is poke-env's WHOLE-order sentinel: its message is
    `/choose default`, and `DoubleBattleOrder.message` joins by string surgery, so half a default
    serialised to `/choose default, move woodhammer 1` — not a legal Showdown command. The server
    rejected it, poke-env re-requested the identical state, and the agent answered identically,
    forever. The per-slot "do nothing" is `PassBattleOrder` (`/choose pass`).
  - **IT IS THE WRITE SIDE OF A BUG B6d FIXED ON THE READ SIDE.**
    `doubles_action_space.normalize_half_default` exists because poke-env *labels* a half default
    with `-2`, its whole-order sentinel, where the per-slot layout says "this slot does nothing" is
    action 0. B6d corrected how a half-default is read and never checked how one is written. The
    same sentence was the fix in both places, and the second half went unlooked-at for two builds.
  - **AFTER: `heuristic` vs `heuristic` 1153 → 9 calls; `random` vs `heuristic` 4001 → 14.** Every
    pair now sits within ~2 calls of `battle.turn`, matching poke-env's baselines. Collection got
    ~90× faster as a side effect — the loop was the cost.
  - **STAGE A GATE: LOOP_CLOSED**, on two pre-registered clauses whose bars are set from the two
    real shards rather than chosen: top-DECILE turn share ≤ 0.45 (top-*ten* is not comparable
    across shards, since with E episodes it cannot fall below 10/E) and unique-observations ÷ turns
    ≥ 0.90.
  - **READ HONESTLY: THE LOOP GUARD AND THE TURN CAP ARE NOT WHAT CLOSED THIS.** Both were built
    first, on the theory that the loop was a property of forced-replacement states; the capped and
    uncapped arms are the same shard to within noise (top-decile 0.173 vs 0.160). They are kept as
    backstops, at exact-identity defaults, and their docstrings say so — because this failure mode
    is silent and expensive and a second instance would otherwise be found the same way.
  - **RESOLVED:** 7.6 recorded turns per episode is what a short VGC battle between weak bots looks
    like, not evidence of dropped turns. The old shard's non-loop subset was 7.1.


- **B6f STAGE B — BC AND AWR RE-FIT ON THE CORRECTED SHARD, AND THE STOP RULE FIRES (2026-08-15).**
  `results/format_ceiling_gate_vgc_v2.json`. New shards: `data/vgc_rl_v2.npz` (70,773 turns /
  **9,578** episodes, against the old shard's 140,848 / 899) and `data/vgc_bc_v2.npz` (31,135
  turns). Same architecture, same hyperparameters, same seed — only the data changed.

  | | B6d/B6e (loop shard) | B6f (corrected) |
  |---|---|---|
  | BC val accuracy (strict, both slots) | **0.980** | **0.602** |
  | BC val loss | 0.061 | 1.291 |
  | AWR val accuracy | **0.975** | **0.476** |
  | AWR value MAE | **0.341** | **1.744** |
  | return distribution std the critic saw | **0.80** | **3.14** |
  | value MAE ÷ return std | 0.426 | 0.555 |

  - **THE ACCURACY WAS NOT MERELY INFLATED, IT WAS MEASURING A DIFFERENT PROBLEM.** 0.980 was
    scored on a set where 96.6% of rewards were zero and the modal turn was one repeated loop
    frame. 0.602 is a strict both-slots number over a 2×107 space with **14.0 legal actions per
    slot**, and it is finally comparable to the singles BC's 0.42.
  - **THE VALUE MAE WENT UP AND THAT IS THE HONEST DIRECTION.** 0.341 was measured against a
    target with standard deviation 0.80; the corrected target has std 3.14. Normalised, the critic
    is slightly *worse* (0.555 vs 0.426 of a standard deviation) — on a target four times wider
    and no longer near-constant. The earlier figure was not a calibration result.
  - **THE LADDER, n = 300 PER CELL** (B6d/B6e reported n = 100, SE ≈ 0.05; here SE ≈ 0.029):

    | arm | vs `heuristic` | ci95 | B6d/B6e reported |
    |---|---|---|---|
    | `mirror` (sanity) | 0.503 | [0.447, 0.560] | — |
    | `simpleheuristics` | 0.467 | [0.411, 0.523] | — |
    | `maxbasepower` | 0.310 | [0.260, 0.364] | — |
    | `random` | 0.023 | [0.011, 0.047] | — |
    | **BC** | **0.453** | [0.398, 0.510] | 0.390 |
    | **AWR** | **0.467** | [0.411, 0.523] | 0.350 |

    Both learned arms are *stronger* than reported, and AWR now sits exactly on
    `simpleheuristics`. The mirror at 0.503 says the harness is sound.
  - **THE PRE-REGISTERED STOP RULE FIRES, AND B6d SAID IT DID NOT.** The rule, written before the
    campaign: *"If the BC agent lands near `simpleheuristics` while `simpleheuristics` is still at
    parity with the heuristic, that is the FORMAT_BOUND signature arriving early."* Both clauses
    now hold — BC 0.453 against `simpleheuristics` 0.467 is 0.5 SE, indistinguishable; and
    `simpleheuristics`'s CI [0.411, 0.523] spans 0.50. B6d adjudicated it "DOES NOT FIRE" **by eye
    in a commit message**, on loop data, where BC read 0.390 and so looked safely *below* both
    competent bots. The arm the rule actually names (`format_ceiling_gate --bc-checkpoint`) was
    never run until now. Two process failures, not one: the rule was evaluated against corrupted
    numbers, and it was evaluated informally rather than by the instrument it names.
  - **WHAT THE RULE DOES AND DOES NOT KILL.** It was written to stop a *strength* campaign from
    plateauing against a flat ceiling, and on the `vs_heuristic` axis it does exactly that: a NULL
    there is now confounded with a genuinely flat VGC ceiling and cannot be read as "PPO fails on
    doubles". It says nothing about the **mechanism** question — whether a factored policy gradient
    improves a doubles policy at all — which is answered by the learner against its own warm start
    and is ceiling-independent. B6f therefore proceeds with `vs_iter0` **promoted to primary** and
    `vs_heuristic` demoted to secondary, and the fired stop rule is booked here rather than
    discovered afterwards.

- **B6f — PRE-REGISTERED (written 2026-08-15, BEFORE running).**

  **Lever:** does the factored policy gradient compound past the AWR ceiling on doubles? Every
  prior doubles result is off-policy AWR, which can only re-weight actions already present in the
  data. PPO is on-policy: it can push probability toward actions outside the demonstrator
  distribution. On OU this was the first method to clear the heuristic band (Build 16 onward). This
  is the same lever on the third format, through the same core.

  **DESIGN — one arm, three seeds. Nothing is being contrasted against a second hyperparameter
  setting; the contrast is against the warm start the arm descends from.**

  | field | value |
  |---|---|
  | init | `checkpoints/doubles_offrl_vgc_v2.pt` (corrected AWR) |
  | format / team pool | `gen9vgc2025regi` / `teams_gen9vgc.packed` |
  | seeds | 0, 1, 2 (sbatch array) |
  | iters / anneal_iters | 80 / **80 (pinned)** |
  | games_per_opp / pop_size / anchors | 48 / 2 / `simpleheuristics` |
  | ent_coef, lr | 0.01 → 0.0, 2.5e-4 → 5e-5 |
  | target_kl (kl_bar) | 0.03 (0.045) |
  | loop_penalty | **0** — refused on doubles by `run_ppo`; the doubles guard is a different penalty over a different layout and is not an OU-tuned value |
  | max_battle_turns | 300 (backstop; measured non-binding) |

  **PRE-REGISTERED CONTRASTS — two, at α = 0.025 (Bonferroni over 2), both scored by
  `scripts/seed_strength_gate.py` at n = 3000 per checkpoint, both arms in ONE invocation.**

  | # | contrast | isolates | role |
  |---|---|---|---|
  | C1 | PPO best-iter vs its own **warm start**, head-to-head | whether the policy gradient moved the policy at all | **PRIMARY** — ceiling-independent |
  | C2 | PPO best-iter vs `heuristic`, against AWR vs `heuristic` | strength on the fixed gradient | secondary — confounded by the fired stop rule |

  C1 is primary on Build 27's own lesson: measuring only against the fixed baseline reported "no
  effect" for search and *missed that the effect was negative*; it took scoring against its own
  base to see it. On a format whose ceiling is in doubt, that is the only contrast that can be read.

  **PRE-REGISTERED GATE.**

  | C1 | C2 | verdict |
  |---|---|---|
  | WIN | WIN | **COMPOUNDS** — the factored policy gradient works on doubles and converts to strength |
  | WIN | NULL | **MECHANISM CONFIRMED, STRENGTH CEILING-BOUND** — the anticipated modal outcome given the fired stop rule |
  | NULL | any | **NULL** — no evidence the factored gradient improves a doubles policy at this budget |
  | REGRESSION | any | **REGRESSION** |
  | final `vs_iter0` < 0.50 late, or entropy → 0 | any | **COLLAPSE** — reported as collapse, never as a NULL |

  **ATTRIBUTABILITY BAR.** A verdict is attributable only if the certificate
  (`results/ppo_vgc_telemetry_b6f_s{N}.json`) shows: trust region not bound in the great majority
  of iterations; `lp_drift_max` < 1e-2 (acting and training were the same distribution — measured
  1.4e-06 on the smoke); `invalid_frac_max` < 0.05 (recorded action was the executed one); and
  `dec_frac_min` > 0.5 (the reported KL is measured over rows that could move the policy). If any
  fails, the outcome is reported as unattributable rather than as a verdict.

  **COST, budgeted off a measured smoke rather than assumed.** 2 iterations at
  `games_per_opp=8, pop_size=1, eval_n=20` ran in 22.5 s wall, i.e. ~96 battles/iteration at ~7.5
  s. The arm is ~544 battles/iteration (144 rollout + 400 eval), projecting **~45 s/iteration** and
  **~60 min per 80-iteration seed** — far cheaper than OU's 4.15 min/iteration, because a VGC
  battle is ~7 recorded turns against OU's 19. Three seeds as one array: ~1 h wall. The n = 3000
  strength gate adds ~6000 battles, ~15 min. No `--resume` needed at this length.


- **B6f — MECHANISM CONFIRMED, STRENGTH CEILING-BOUND (2026-08-15).** The pre-registered gate's
  own anticipated modal outcome: **C1 WIN, C2 NULL**. `results/ppo_vgc_gate_b6f.json`,
  `results/seed_strength_gate_b6f_c1.json`, `results/seed_strength_gate_b6f_c2.json`,
  `results/ppo_vgc_telemetry_b6f_s{0,1,2}.json`. 3 seeds x 80 iterations, ~55 s/iteration, one
  sbatch array on `tron`.

  - **C1 (PRIMARY) — PPO's best checkpoint beats its own warm start. n = 3000 per seed.**

    | | rate | ci95 |
    |---|---|---|
    | seed 0 (`iter_62`) | 0.522 | [0.504, 0.540] |
    | seed 1 (`iter_63`) | 0.523 | [0.505, 0.541] |
    | seed 2 (`iter_57`) | 0.544 | [0.526, 0.562] |
    | **pooled, 4768/9000** | **0.530** | **[0.519, 0.540]** |

    The pooled CI excludes 0.50, and so does **every seed individually** — 3 of 3, no sign
    disagreement. **VERDICT: WIN.** The factored policy gradient does move a doubles policy off
    the AWR ceiling. It is a *modest* +3.0 points, and saying "modest and unambiguous" is the
    whole content: 9,000 battles is what makes +0.030 readable at all.

  - **C2 (SECONDARY) — against the fixed heuristic, NULL.** n = 3000 per checkpoint.

    | arm | rate | ci95 |
    |---|---|---|
    | AWR warm start | 0.454 | [0.437, 0.472] |
    | PPO (pooled over 3 seeds, 4197/9000) | 0.466 | [0.456, 0.477] |
    | **contrast** | **+0.012** (z +1.14, p 0.254, alpha 0.025) | CIs overlap |

    **VERDICT: NULL** — and, per the stop rule that fired in Stage B, an *uninterpretable* null:
    `simpleheuristics` sits at 0.467 with a CI spanning 0.50, so there is no established headroom
    above it for a strength gain to appear in. This is the row the pre-registration named as
    anticipated, and it is booked as ceiling-bound rather than as evidence against the method.

  - **THE TWO CONTRASTS DISAGREE, AND THAT IS THE FINDING, NOT A PROBLEM.** The same checkpoints
    beat their own warm start (0.530, 5.7 SE above parity) and are indistinguishable from it
    against the heuristic (+0.012, 1.1 SE). Both can be true because the heuristic is not a
    discriminating opponent on this format — it is at parity with `simpleheuristics`, which is at
    parity with its own mirror. A fixed baseline can only resolve a difference it is strong enough
    to punish. This is Build 27's Gate B lesson arriving with the opposite sign: there, measuring
    only against the fixed baseline hid a *negative* effect; here it hides a *positive* one.

  - **SELECTION BIAS, MEASURED RATHER THAN ASSUMED, AND IT IS LARGE.** The in-loop curve reports
    best-iteration `vs_heuristic` of **0.590 +- 0.014** (seeds 0.58 / 0.58 / 0.61 at iterations
    62 / 63 / 57). Re-scored at n = 3000, those same checkpoints read **0.466**. The gap is
    **0.124**, and it is pure argmax-over-80-noisy-estimates bias: `eval_n = 100` has SE ~ 0.05,
    and taking the maximum over 80 draws inflates by roughly that much. **Reading the curve's
    headline instead of the re-scored number would have overstated this build by 12 points.** It
    is what `scripts/pin_gate_checkpoint.py`'s note and `scripts/selection_bias_sim.py` exist to
    prevent, and it is the largest such gap the project has recorded.

  - **ATTRIBUTABILITY: CLEAN ON ALL FOUR CLAUSES.**

    | | seed 0 | seed 1 | seed 2 | bar |
    |---|---|---|---|---|
    | trust region bound | 4/80 | 4/80 | 5/80 | not the great majority |
    | `epochs_full_fraction` | 0.950 | 0.950 | 0.938 | — |
    | `approx_kl_max` | 0.0354 | 0.0546 | 0.0290 | bar 0.045 |
    | `approx_kl_mean_late` | 0.0160 | 0.0170 | 0.0163 | — |
    | `lp_drift_max` | 8.6e-06 | 6.7e-06 | 6.9e-06 | **< 1e-2** |
    | `invalid_frac_max` | 0.0262 | 0.0314 | 0.0245 | **< 0.05** |
    | `dec_frac_min` | 0.890 | 0.878 | 0.884 | **> 0.5** |

    The three doubles-specific clauses are what make this readable at all. `lp_drift` at ~7e-06
    says the acting distribution and the updated distribution were the *same function*, so the
    importance ratios the KL summarises are real. `dec_frac ~ 0.88` says the reported KL is
    measured over rows that could actually move the policy; without the decision-row denominator
    the same run would have reported a KL an order of magnitude smaller and certified a trust
    region that never bound. No seed collapsed: final `vs_iter0` 0.51 / 0.48 / 0.56, entropy
    stable at ~0.9.

  - **NOT CLAIMED.** That VGC's ceiling is flat — the stop rule's signature is *suggestive*, and
    Lever 15's RB verdict needed three legs (M1 band, M2 near-optimal search, M3 team-strength
    AUC) where doubles still has only M1. That PPO would not compound with more budget: 80
    iterations is one dose, and the OU axis needed 320 before it saturated. And nothing here
    speaks to team preview, which `Player.random_teampreview` still picks 4-of-6 at random on
    both sides — a large part of VGC skill the policy cannot express, adding variance to every
    number above.

- **RELEASE FOLLOW-UPS — M4 RUNS ON DOUBLES, THE CORRECTED LADDER IS PUBLISHED, AND THE B6f
  INTERMEDIATES ARE PRUNED (2026-08-15).** Four items closed after the release-readiness merge.

  - **CI WAS RED ON EVERY PUSH, INCLUDING THE PUSH THAT ADDED IT.** The workflow runs a bare
    `pytest -q`; four test modules do `from tests.conftest import ...`; `tests/` is not a package
    and `[tool.setuptools.packages.find] include = ["lategame*"]` makes the editable install
    finder-based, so nothing puts the repo root on `sys.path` for that invocation. The failure is
    at COLLECTION — `Interrupted: 4 errors during collection`, 0 tests run. The reported-green
    708/6 was measured through `python -m pytest`, the one form that hides it. `pythonpath = ["."]`
    makes both forms equivalent.

  - **M4 DOUBLES: THE DIAGNOSIS WAS WRONG AND THE REPAIR WAS BIGGER.** `"offrl"` in
    `train/selfplay.py` is an agent-registry name, not a format string; `battle_format` was already
    threaded correctly. The kill is `build_player` refusing a singles-only name on a doubles format
    (Build 26, deliberate). Beyond the five name sites: `_eval_point` passed `team=` to *none* of
    its four `build_player` calls, `collect_selfplay` had no `team_pool` parameter at all, and
    `run_selfplay` did none of `run_ppo`'s warm-start checks. `SelfPlayConfig` gains `team_pool`
    and `max_battle_turns`, both `None`-default; **not** `loop_penalty`, whose only honest doubles
    value is 0.0. `tests/test_arena.py`'s dispatch test already enumerated the modules that must
    ask rather than hardcode — `train.selfplay` was the one missing from that list.

  - **THE n = 300 BC/AWR RE-MEASURE WAS ALREADY DONE; THE README JUST NEVER SHOWED IT.** The table
    above (BC 0.453, AWR 0.467) has been in `results/format_ceiling_gate_vgc_v2.json` since
    `85a3a01`, and in this file, while the README said "withdrawn and re-run" and stopped there.
    Published now, with the withdrawn n = 100 reads printed as provenance rather than as a delta.
    The record itself carried no checkpoint paths and labels its learned rows `bc_v11` / `offrl_ou`
    (OU-era schema keys that describe nothing on a VGC run), so `_arm_record` now records the path
    going forward and a top-level `_provenance` block back-fills it, flagged three ways as
    hand-added. Only then could the ladder be promoted to a `check_artifacts.py` headline: against
    the pre-back-fill record that promotion fails `test_every_headline_maps_to_a_gate_file_that_
    cites_something` on `assert 0` — a headline that reports OK while checking nothing.

  - **G4's VGC COLUMN WAS A PRE-LOOP-FIX READ PRESENTED AS CURRENT.** Measured in `4647bb0`, three
    days before `18fe55c` found the loop, on the instrument whose output got BC/AWR withdrawn. All
    four of its cells were re-measured at the same n = 300 afterwards (mirror 0.523 → 0.503,
    `simpleheuristics` 0.527 → 0.467, `maxbasepower` 0.300 → 0.310, `random` 0.017 → 0.023). Now
    marked superseded. The INSUFFICIENT verdict is unaffected — the corrected competent bot sits
    *further* below the 0.58 headroom threshold, not nearer it.

  - **PRUNE APPLIED: 234 files, 705,479,112 B.** `checkpoints/` 2,183,910,041 → 1,478,430,929 B;
    361 → 127 files. Only the three `ppo_b6f_s{0,1,2}` arms had anything to delete; every other arm
    was already at its retention set. Each b6f arm kept exactly three files — `curve.json`, the
    gate-declared terminal `iter_80.pt`, and the cited best (`s0/iter_62`, `s1/iter_63`,
    `s2/iter_57`). `check_artifacts.py --json` is byte-identical across the prune apart from the
    headline entry added in the preceding commit: 121 cited, 63 present, 58 missing, 52 files
    citing a missing path, 4/4 headlines OK. A moved digit there would have meant the retention
    rules deleted something a result cites, since rule 1 keeps anything cited. The post-prune dry
    run reports `TOTAL 0 0.00G`.

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

1. ~~**First-format confirmation:** Gen 9 Random Battles as MVP — agreed? (Recommended.)~~
   **ANSWERED (Lever 15, 2026-07): NO — and by measurement, not preference.** gen9-RB is
   **FORMAT_BOUND**: the achievable ceiling vs a competent heuristic is ~parity (0.523, CI spans
   0.50), so G2 is unreachable there whatever the model. The MVP format is **`gen9ou`**, where the
   same diagnostic found genuine headroom (`simpleheuristics` 0.633, whole CI above the 0.58 bar).
   RB remains the format the pipeline was *built* on and every pre-OU build is reported against.
2. ~~**Search vs. pure policy:** ship Phase 1 as policy-only and add search in M7, or invest in search earlier for prediction quality?~~
   **ANSWERED (Build 27 Gate B, 2026-08-12): POLICY-ONLY — and this time the answer is not
   confounded by the format.** L11–L14 already found search at parity, but all of that was measured
   on gen9-RB, which Lever 15 subsequently proved FORMAT_BOUND: near-optimal search reaches 0.500
   there because nothing beats the heuristic, so the test could not have come out any other way.
   Re-run on `gen9ou` — proven headroom, a 0.77 base, an *exact* white-box opponent model, and a
   forward model validated at 0 mismatches on that format — depth-2 search comes back
   **+0.0036 (p 0.76, positive in 6/10 shards) at n=2500/arm**. And it is worse than neutral: it
   loses to the greedy policy it descends from **head-to-head at 0.3932, 10/10 shards, ~10.7 SE**.
   Search is not the missing ingredient at this level of play; **M7 closes**.
3. **Team generation:** how far to push beyond curated pools toward learned teambuilding for OU/VGC?
4. **Reuse vs. rebuild:** fork Metamon (dataset + baselines + reconstruction) as the foundation, or build the pipeline fresh for full control? (Forking is faster; rebuilding is more educational.)
5. ~~**Eval environment:** stand up a private Showdown server for clean evaluation, or use anonymized non-ranked live play?~~
   **ANSWERED (M5 / G1 build, 2026-08-10): the private/agent-only route — and the alternative turned
   out not to exist.** The question presupposes that "anonymized non-ranked live play" is available
   on the public server. It is not: `/search <format>` on the public sim **is** the rated ladder,
   and Showdown offers no unranked equivalent, so live public play cannot be made policy-safe —
   only policy-*explicit*. `lategame/live/policy.py` is that finding turned into a gate (two
   independent opt-in channels; see NG3 and §15), and `--server` is the supported private-eval
   path. The clean-evaluation half of the question is answered by `lategame/eval/ladder.py`: an
   **agent-only eval ladder** on the local server, which is what §12's "private/agent-only server
   or eval ladder" asks for and the only varied field reachable without touching the human ladder.
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
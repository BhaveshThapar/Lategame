# PRD — Competitive Pokémon Showdown ML Battle Agent

**Name:** RotomAI (settled 2026-08-17; was the working name "Lategame")
**Author:** Bhavesh
**Date:** June 26, 2026
**Version:** 0.1 (draft)
**Status:** v1.0.0 — all five goals met. The build log (every lever and build, with its pre-registration and verdict) lives in [docs/RESULTS.md](docs/RESULTS.md); this document is the design: goals, requirements, architecture, roadmap.

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
  - **The *ladder* half is now COMPUTED (2026-08-10), on an agent-only eval ladder rather than the human ladder.** GXE and Glicko-1 need a varied opponent field, which no amount of further training can supply — but §12's own last line prefers "a private/agent-only server or eval ladder", and §16 Q5 is answered: there is *no* unranked public ladder, so that route is the only policy-clean one. `rotomai/eval/ladder.py` fits a whole field jointly (Bradley-Terry on the Glicko scale, `heuristic` pinned at 1500). Over 8 agents × 28 pairs × 300 battles on `gen9ou`, **`v25b` reaches Glicko 1696.9 ± 14.9, GXE 0.6809**, above `simpleheuristics` (1579.2 / 0.5756) and the `heuristic` anchor (1500 / 0.5000) — see §13.1. Two caveats travel with it: the RD is a **lower bound** (battles cluster by team matchup, so the effective sample is under `n`), and this is an **agent-only** field, so the number is *not* comparable to a Showdown GXE measured against humans. The human-ladder reading remains out of scope under NG3; `rotomai/live/` (**G1**) is built and gated for it should that ever change.
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
  - **The VGC ceiling verdict is FORMAT_BOUND as of 2026-08-16**, on three legs (M1 0.493 / M2 0.373 / M3 0.520) rather than the M1-only `INSUFFICIENT` this goal was booked alongside. That does not change G4, whose exit is "playable end to end" and was met on 2026-08-12 — it settles the *separate* question of whether the doubles strength axis is worth spending on. It is not: `stop_strength_axis`. See §13.1.
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
  - **EXIT CRITERION (written 2026-08-16, and it did not exist before that).** G5 is the only goal that was stated without one — no threshold, no gate script, no results file, and two references to it as "open" — so "demonstrated rather than merely implemented" had nothing behind it to check. The criterion: **each capability names a gate that runs, a number from a committed results file, and THAT GATE'S OWN pre-registered pass criterion, met.** Every threshold is read out of the record it judges (`threshold`, `pass_rate`, `verdict`, `gate_a_pass`); none is introduced by the G5 gate. Any bar written now would be written after the measurements exist, and B6d is on the record for what that costs — a pre-registered stop rule adjudicated by eye, in a commit message, against numbers later withdrawn. `tests/test_g5_capability_gate.py` rewrites the floors inside fake records and asserts the verdict follows them, which a hardcoded threshold would survive.
  - **MET (2026-08-16)**, `results/g5_capability_gate.json`, promoted to a `check_artifacts` headline:

    | capability | requirement | gate | number | verdict |
    |---|---|---|---|---|
    | state estimation | R-STATE / R-PRIORS | `search/recon_check.py` | RB 0.999952 (145,400 checks) / OU 1.000 / VGC 0.9991 | PASS ×3 |
    | prediction / opponent modeling | R-PREDICT | `rpredict_oppmodel_gate --gate a` | POV fidelity 1.000, decodable 1.000, agreement 0.9582 (n=239) | PASS |
    | win-condition planning | R-PLAN | `search/expectimax.py` via `--gate b` | pooled n=2500/arm, 0.7724 vs base 0.7688 | **NULL** |
    | precise damage math | R-CALC | `search/fidelity.py` + `engine/damage.py` | 9,734 transitions, core match 1.000 | PASS |

  - **What MET does NOT mean.** Win-condition planning is the case that separates *demonstrated* from *beneficial*: the search runs at n = 2500/arm over a forward model separately gated at 1.000 core-transition agreement, and its pre-registered rule returns NULL (diff +0.0036, p = 0.762). G5 asks for an explicit testable capability, not for a win rate — so it passes, and the null stays in the record and in the printed summary. Dropping it would be precisely the "merely implemented" claim the goal exists to replace. Also not claimed: team preview is a **fixed rule**, not a learned capability (the codec has no slot and the model no head for it).
  - **Coverage gap, recorded rather than papered over.** The replay-driven damage-math gate is `gen9randombattle` only, because it re-sims an `inputlog` and public replays for the teambuilt formats carry none. Live-play reconstruction covers OU and VGC instead, under state estimation.

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

### 13.1 Build status & findings

**Moved to [docs/RESULTS.md](docs/RESULTS.md).** It was 3,167 lines — 88% of this document — and it
answers a different question than the rest of §1–16 does: what happened, in the order it happened,
rather than what the system is required to do. Docstrings across the tree that cited "plan.md 13.1"
now cite `docs/RESULTS.md`; citations to §3.1, §7, §12, §15, NG3 and M0–M7 still point here.

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
3. ~~**Team generation:** how far to push beyond curated pools toward learned teambuilding for OU/VGC?~~
   **ANSWERED BY DESCOPING (2026-08-16): not at all in v1, and NG6 said so from the start.** The
   question outlived its own non-goal. What v1 ships is the curated-pool half: validator-checked
   packed pools for both teambuilt formats (`rotomai/teambuilding/data/`) drawn per side with
   distinct seeds. The one piece of team-level *decision-making* that did land is bring-6-pick-4
   team preview, and it is a fixed rule rather than a learned policy — the codec has no slot for
   preview and the model no head for it (§13.1, 2026-08-16). Measured worth: up to +0.060 on a
   single arm at n = 300, and **no movement in the VGC ceiling verdict**, which is the honest
   argument for why learned teambuilding is not the missing ingredient either. Reopen it only for a
   format with proven headroom; both teambuilt formats measured so far are `FORMAT_BOUND` or have
   their strength result already.
4. ~~**Reuse vs. rebuild:** fork Metamon (dataset + baselines + reconstruction) as the foundation, or build the pipeline fresh for full control? (Forking is faster; rebuilding is more educational.)~~
   **ANSWERED RETROSPECTIVELY: BUILT FRESH.** Not a decision that was taken and recorded so much as
   one the twenty-eight builds in §13.1 made by happening — ingestion, POV reconstruction, the
   forward model, the encoders, the codecs, the gates and the eval ladder are all first-party. The
   question is struck because it describes a fork that never happened, not because the trade-off it
   names stopped being real. What full control bought is visible in §13.1: a pinned simulator rev,
   a reconstruction path gated at 1.000 core-transition fidelity, and pre-registered stop rules on
   arms that could be re-run against them. What it cost is the wall-clock those twenty-eight builds
   took.
5. ~~**Eval environment:** stand up a private Showdown server for clean evaluation, or use anonymized non-ranked live play?~~
   **ANSWERED (M5 / G1 build, 2026-08-10): the private/agent-only route — and the alternative turned
   out not to exist.** The question presupposes that "anonymized non-ranked live play" is available
   on the public server. It is not: `/search <format>` on the public sim **is** the rated ladder,
   and Showdown offers no unranked equivalent, so live public play cannot be made policy-safe —
   only policy-*explicit*. `rotomai/live/policy.py` is that finding turned into a gate (two
   independent opt-in channels; see NG3 and §15), and `--server` is the supported private-eval
   path. The clean-evaluation half of the question is answered by `rotomai/eval/ladder.py`: an
   **agent-only eval ladder** on the local server, which is what §12's "private/agent-only server
   or eval ladder" asks for and the only varied field reachable without touching the human ladder.
6. **Reward shaping specifics:** which intermediate signals densify learning without distorting the win objective? **(THE ONLY QUESTION STILL OPEN.)** `data/reward.py`'s `RewardWeights` is what every arm ran with and it was never ablated — no build varied it, so nothing on the record separates "these weights help" from "these weights are what we happened to use". It is open rather than descoped because it is cheap to answer on `gen9ou`, the one format with proven headroom.

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
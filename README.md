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

**Five follow-ups all find the GREEN policy near a local ceiling (AMBER).** Lever 10 —
on-policy PPO warm-started from the GREEN checkpoint — is stable (no collapse) but does not
compound. Lever 11 (**R-PREDICT**) builds the infrastructure the project never had: a
*faithful Showdown forward model* (serialize/fork/step, validated bit-for-bit, **0/9,734**
transition mismatches) + *determinization* of the hidden opponent (live POV → full battle,
**~100%** observable-faithful). But depth-1 test-time search on the frozen GREEN policy doesn't
beat it (head-to-head ~0.36 at n=120). Lever 12 deepens it to **depth-2** (same forward model,
recursion + policy-prior pruning): the over-switch is gone (vs random **1.000**) and search now
reaches **parity** with its base (h2h **0.500**) but still does not exceed it (vs heuristic
−0.025; the minimax arm regresses to h2h 0.275 — the determinized opponent model is too weak for
worst-case search). Lever 13 tests the last untested axis — the **tougher-opponent curriculum**:
the wall-clearing AWR mechanism (all turns, both sides) re-run from GREEN on self-play data vs
the tough `simpleheuristics` anchor + a fictitious-play league (heuristic held out), powered up
(the M4 loop was underpowered + never run from GREEN). A cheap pre-flight confirms the signal is
present (winner/loser start-return gap **6.30**), but 3 seeds × 12 iters stays **flat**: best-iter
vs heuristic **0.472±0.013** ≈ the 0.462 start, final head-to-head vs the start **0.443 < 0.50**,
and even win-rate vs the *trained* anchor doesn't rise. Lever 14 closes the exact axis the search
direction was retired on — the **opponent model**. L11/L12 modeled the foe as uniform/worst-case,
but the eval opponent is a fixed white-box heuristic, so it's modeled *exactly* (Gate A: the
white-box model agrees with the real `HeuristicAgent` **0.958**, opponent-POV fidelity **1.000**)
and fed to **probability-weighted expectimax**. Even this near-perfect opponent model reaches only
**parity** at n=120 (search-vs-heuristic **0.500** vs base **0.483**, delta **+0.017**; the n=40
+0.225 was base's unlucky low draw). Five independent mechanisms — gradient (L10), depth-1 (L11),
depth-2 (L12), curriculum (L13), real-opponent-model search (L14) — confirm the local ceiling; it
lives in neither the training loop nor the inference machinery (now exhausted on every axis: depth,
aggregation, *and* opponent-model quality).

**Lever 15 measures the ceiling *directly* and pivots the format (FORMAT_BOUND).** Levers 1–14 only
showed *our* methods fail; none measured what *any* agent can achieve vs the heuristic on gen9-RB. A
cheap, no-training gate does: (M1, n=300) the heuristic crushes naive bots (99.3% vs random, 89.3% vs
maxbp) but poke-env's **strongest** built-in `simpleheuristics` is at **statistical parity** (0.523,
CI [0.467, 0.579]) and GREEN — 9 levers of RL — **loses** to it (0.430); (M2) near-optimal depth-2
search with a near-perfect opponent model is **0.500**; (M3, n=500) team strength does **not** predict
the winner (effective-stat AUC **0.495**) — gen9-RB is balanced by design. The achievable ceiling vs a
competent heuristic is ~parity, so **G2 (decisively beat the heuristic) is unreachable no matter the
model** ⇒ scaling on gen9-RB is unjustified. **Next = the Gen 9 OU pivot** (PRD G4/M6): teambuilt,
higher skill ceiling, abundant human data; encoder/action head already singles-native, the one new
build is R-TEAM team provisioning. See `plan.md` §13.1.

**OU pivot Build 1 — R-TEAM + OU ceiling re-run confirms the higher ceiling (GREEN).** Before the full
OU pipeline, the same cheap probe re-runs on gen9ou. New R-TEAM (`lategame/teambuilding/pool.py`
`TeamPool` over a curated, Showdown-validator-checked packed pool — 12/12 legal in
`lategame/teambuilding/data/teams_gen9ou.packed`; `scripts/build_ou_teampool.py`), `team=` threaded through `eval.arena`, and
`format_ceiling_gate.py --format gen9ou`. **M1 (n=300):** the strongest competent bot `simpleheuristics`
beats the heuristic **0.633** [0.577, 0.686] — vs only **0.523 (parity)** on gen9-RB — with the CI
entirely above the 0.58 headroom bar (mirror 0.487 sanity ✓; band width 0.610 > RB 0.516). The exact
quantity that forced FORMAT_BOUND on RB shows **real headroom** on OU ⇒ the ceiling is genuinely higher;
**greenlight the OU pipeline.** M2 (OU near-optimal search) + M3 (OU replays) deferred to the next gate;
GREEN's OOD transfer to OU is 0.383. Suite 145 pass / 5 skip; `results/format_ceiling_gate_ou.json`.

**OU pivot Build 2 — human-replay ingestion → first OU checkpoint (AMBER: pipeline works, agent
non-functional, deeper POV gap diagnosed).** gen9ou public replays carry **no `inputlog`**, so `resim`
(seed-based) is impossible — but the logs open with `|poke|` team preview, so the seed-free `data.ingest`
is the right reconstructor. Built: `ingest._register_preview` (own team preview → `battle.team`),
`encoder._opponent_mons` (merge revealed + `teampreview_opponent_team` so the opponent roster matches at
train and eval — no-op for RB), `scripts/ou_ingest_gate.py` (fidelity KILL gate + strip-`|poke|` negative
control), `train-rl --seed`, `format_ceiling_gate.py --offrl-checkpoint`. Scraped **2,760** replays →
**120,012** all-turns turns; **Gate A PASS** (species coverage 1.000, control lift +0.315). Trained BC +
3 AWR seeds (value-MAE ~0.47–0.52, healthy). **Gate B:** offrl **0.007** vs heuristic, **0.495 vs random**
(no signal) despite 0.71 imitation accuracy. **Diagnosed:** the log reveals the player's *own*
item/ability/moves only progressively (own-active item 0.18→0.82, ability 0.45→1.00, moves 2.18→4.00
train→eval) while the live `|request|` gives the full team from turn 1 → OOD on identity channels →
random play. Team preview closes *species*, not *detail* — the log-vs-request gap that needed resim on RB,
which OU can't use. **Next lever: two-pass own-team completion.** Suite **154 pass**;
`results/ou_ingest_gate.json`, `results/format_ceiling_gate_ou_trained.json`.

**OU pivot Build 3 — two-pass own-team completion (AMBER/negative: log-only completion can't close the
POV gap).** Built `ingest._prescan_kits` (Pass 1: read each own mon's full-game-revealed moves/item/ability
off `battle.team`; item recovered from the raw `|-item|`/`|-enditem|` lines since poke-env resets a consumed
item to `None`) + `_complete_own_team` (backfill before every `embed_battle`: `_add_move` missing moves, fill
item only if the `unknown_item` sentinel so a consumed item stays `None` as live, ability only if unknown),
threaded via `_reconstruct_pov(kits=…)` with a default-on `complete_own_team` toggle (`--no-complete-own-team`).
Upgraded `ou_ingest_gate.py` to measure the **encoder ID channels** (item/ability/move) ON-vs-OFF with
teeth + ceiling + residual — the Build-2 lesson made concrete. **Gate A PASS** (n=200): two-pass lifts item
+0.114 and moves +0.604, but **ability is irreducible from public logs** (+0.016 — poke-env already
auto-assigns single-option abilities, so the unknowns are multi-ability mons that never triggered); residuals
vs the live POV stay large (item 0.70, ability 0.53, moves 1.38). Re-trained BC (val-acc 0.651) + 3 AWR seeds
(value-MAE 0.54–0.62). **Gate B still dead:** offrl **0.020** vs heuristic (Build-2 0.007), **0.16–0.45 vs
random** across seeds. A controlled OFF-vs-ON eval + a live-obs probe confirmed **no bug**: eval obs is *full*
(own-active item 0.89 / ability 1.00 / moves 4.00) while two-pass training reaches only 0.30/0.47/2.62 — a
real, large OOD gap. Log-only completion moves training toward eval but nowhere near it; **a partial POV fix
is functionally neutral** (OFF ≈ ON, both ~0.02 vs heuristic). **Next lever: usage-prior imputation** — fill
each own mon's *unrevealed* item/ability/moves from the species' standard competitive set so all six reach the
live full-kit detail. Suite **152 pass / 5 skip**; `results/ou_ingest_gate.json`,
`results/format_ceiling_gate_ou_v3.json`.

**OU pivot Build 4 — usage-prior imputation (RED: obs verified at eval-full density, agent still dead —
detail density was never the binding failure).** Built `data/usage_prior.py` + `scripts/build_usage_prior.py`:
Smogon chaos stats (gen9ou-1500, 2026-06) distilled into a committed per-species top-K artifact
(`features/data/usage_gen9ou.json`, 402 species, 0 out-of-vocab, `vocab_version` drift guard) and sampled
**usage-weighted + stably seeded** per (replay-POV, mon); `ingest._impute_kits` fills each kit's still-unrevealed
item/ability/moves once per POV between prescan and reconstruction — revealed truth always wins, the labelled
action can never be imputed, consumed items stay `None` (default-on, `--no-impute-usage`). **Gate A PASS** after
an honest metric fix: the arm first KILLed on an absolute item bar (0.760 < 0.85), but a per-decision
decomposition showed the residual is **0.233 consumed-`None`** (live-faithful Knock Off / Booster Energy states)
with only **0.0069 unfilled** — the gate now kills on the unfilled rate; ability 0.999, moves 3.990.
Re-ingested (120,001 turns — identical to v3, imputation changes values not counts), BC val-acc 0.636, 3 AWR
seeds healthy (value-MAE 0.51–0.61). **Gate B RED:** offrl **0.003** vs heuristic; **0.05–0.27 vs random**
across seeds (loses to random; harness clean, mirror 0.520). The monotone pattern vs random — v1 sparse
**0.495** → v3 two-pass **0.13–0.45** → v4 eval-full **0.05–0.27** — closes the POV-density hypothesis: more
own-kit train detail consistently makes the live agent *worse*. Strongest remaining candidate (fits the
monotonicity): **positional move-slot ORDER** — train slots are reveal-order + sorted backfill, live slots are
`|request|` declaration order, and the action space indexes moves positionally, so every completion step
scrambles more slot semantics. **Next lever: canonical move-slot ordering** at ingest *and* live encode
(BC-gateable before any retrain). Suite **162 pass / 5 skip**; `results/format_ceiling_gate_ou_v4.json`,
`results/gateb_v4_vs_random.json`.

**OU pivot Build 5 — canonical move-slot ordering (RED: slot order fixed and verified, agent still dead —
and pure BC craters too, so the failure is upstream of RL).** Gate A measured the suspected scramble
directly before any code change (`scripts/slot_order_gate.py`): **25.9%** of training move-labels change slot
under canonicalization, **91.7%** of eval-teampool mons declare moves non-canonically (~0.97 slot mean
displacement), zero out-of-4 truncation drops — PASS. The fix: `features/action_space.py` no longer delegates
to poke-env's `SinglesEnv` — every converter (label/decode/mask/synthesize) and the encoder's move blocks now
index `canonical_moves` (known moves sorted by id, first four), identical at ingest and live **by
construction**; **`OBS_VERSION` v2→v3** hard-rejects every v2-era shard/checkpoint (incl. the retired RB
GREEN) so orderings can never silently mix. 14 new tests incl. a request-backed round-trip on every masked
action. **Gate B1 PASS:** re-ingest needed zero ingest changes (119,996 turns, −0.005% vs v4 = the predicted
~zero new drops; `ou_ingest_gate` regression channel-identical), BC val-acc **0.647 ± 0.002** over 3 seeds —
above the 0.63 bar *and* the v4 baseline 0.636 (canonical labels are consistent across replays, hence more
learnable). **Gate B2 RED:** 3 AWR seeds healthy (value-MAE 0.45–0.55) but vs-random **0.02 / 0.11 / 0.06**
(≤ v4's 0.05–0.27) and vs-heuristic **0.010** at n=300 (harness clean: mirror 0.493, gradient
0.030 < 0.060 < 0.643). Post-hoc localization: the pure-BC v5 checkpoints also lose to random (**0.06 /
0.03**) — so the stall is *not* AWR-specific, *not* the harness, *not* move-slot order, *not* obs density:
four eliminated causes. Remaining candidates: **switch-slot/team-order semantics** (slots 0–5 + the six
own-mon obs blocks, the one ordering never probed), **live behavioral pathology** (what do these agents
actually *do* vs random — a 2-battle order-log probe is the cheapest next step), and **human-replay →
bot-eval distribution drift**. Suite **181 pass / 5 skip**; `results/slot_order_gate.json` (a/b1/b2 blocks),
`results/format_ceiling_gate_ou_v5.json`, `results/gateb_v5_vs_random.json`.

**OU pivot Build 6 — live behavioral probe (CONCLUSIVE: the agents over-switch — absorbing two-mon switch
loops; decode/mask and team-order causes eliminated).** `scripts/behavior_probe.py` instruments the live
decision path with observe-only hooks (the codec's silent random fallback, poke-env's level-25
`[Invalid/Unavailable choice]` rejections, per-decision action/mask/top-3-probs, packed-vs-live team order)
and classifies against a pre-registered tree (a decode/mask > b team-order > c pathological-legal > d drift).
n=20/arm on bc/offrl v5 s0 vs random (control mirror 0.45 sane, 3,753 decisions): fallback rate **0.000**,
**zero** rejections, **zero** all-False masks — (a) dead; packed upload order == live `team.values()`
**40/40** — (b) live half dead. The failure is **(c)**: sane openings (tera + attack), then absorbing
switch loops once a wall is active (Gholdengo↔Corviknight 70+ consecutive turns; voluntary switch fraction
**0.77/0.70** vs train base 0.184, max runs **117/129**, ping-pong 0.77/0.84, mean top-1 prob 0.62 —
confident, legal, losing). The loop is in the policy *mass* (~0.9 on switches in loop states — sampling
wouldn't fix it) and the memoryless obs makes a 2-cycle a fixed point; echoes L11's value-head over-switch,
now in the imitation policy. **Next lever: train-side switch-mass diagnostic** — human switch rate + policy
switch mass on the shard's wall-active states (imitated pivot prior composing into a loop vs OOD artifact);
offline, zero training. 40 per-turn transcripts committed as evidence. Suite **198 pass / 5 skip** (203 with
the server up); `results/behavior_probe.json`, `results/behavior_probe_transcripts/`.

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
python scripts/curriculum_gate.py        --out results/curriculum_gate.json   # Lever 13: tougher-opponent AWR self-play (AMBER)
python scripts/rpredict_oppmodel_gate.py --gate a   # Lever 14 Gate A: opponent-model fidelity (PASS)
python scripts/rpredict_oppmodel_gate.py --gate b --arms whitebox,learned --concurrency 6  # Lever 14 Gate B: real-opponent-model search (AMBER)

# M6 — human replays: fetch, then reconstruct each player's POV either from the public
# spectator log (v1) or by re-simulating the inputlog for the private |request| (v2)
python -m lategame.cli fetch-replays  --min-rating 1200 --limit 200
python -m lategame.cli ingest-replays --out data/ingest_gen9rb_rl.npz   # v1 (public-log POV)
python -m lategame.cli resim-replays  --out data/resim_gen9rb_rl.npz    # v2 (needs node + dist/)
```

## Develop

```bash
pytest            # 118 tests; server-gated smokes run when the local server is up
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

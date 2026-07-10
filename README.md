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

**OU pivot Build 7 — train-side switch-mass diagnostic (H2 UNANIMOUS: the loop is an OOD artifact — the
live ~0.9 switch mass never appears on training states; pivot-prior and amplification causes dead).**
`scripts/switch_mass_gate.py` (offline, zero training): decodes own/opp-active species from the shard obs
ID channels, conditions on the Build-6 empirical loop states (8 species / 27 pairs from the decisions
JSONL, frozen fallbacks committed), and compares the human switch rate **H** against each v5 checkpoint's
masked-softmax switch mass **P**, top-1-is-switch **T**, and the uniform-over-legal anchor **U** under a
pre-registered H1/H2/H3 tree with teeth (harness identity 1e-16, random-init band, attacker-specificity
≤ 0.30). All 6 ckpts × both full shards: on loop-species states H **0.205/0.210**, P **0.161–0.229**
(matched), T **< 6%** — vs live 0.77 fraction / ~0.9 mass; even on the exact live loop pairs P_pair
**0.165–0.234**. H1 dead (humans not pivot-heavy: replay-log true rate 0.211 ≈ shard 0.189); H3 dead (max
P−H +0.024; BC-vs-AWR delta −0.008). **Verdict bc=H2; offrl=H2** → the failure is distribution shift, not
learned behavior; next lever is drift-side — a live-vs-shard localization probe (NOT a history feature,
nothing to damp in-distribution). Suite **221 pass / 5 skip**; `results/switch_mass_gate.json`.
*(Build-7 also named "usage-prior imputation" as the next lever; **Build 8 found that stale** — Build 4
already did it → RED — and localized the real carrier, below.)*

**OU pivot Build 8 — live-vs-shard drift localization (CARRIER = a single encoder channel: the move
`pp-fraction`).** Phase A extends `scripts/behavior_probe.py` to dump the exact per-decision obs/mask the
model scored (gitignored npz) and adds a held-out-heuristic-opponent arm (the loop reproduces vs the real
eval opponent too). Phase B (`scripts/drift_probe.py`, offline, zero training) runs a **causal swap
bisection**: pair-match each live loop state to shard rows of the same (own,opp) species, then paste each
channel group across the manifold both ways and re-score the frozen policy (`deletion` live←shard: does
switch mass fall to ~0.2? `insertion` shard←live: does it rise to ~0.9?), holding action legality fixed.
**Carrier = `moves`, +0.875 of the gap** (bc +0.880 / offrl +0.870, both swap directions, all 6 ckpts);
every other group ≈ 0; controls pass (self-swap identity 0.0, ALL-swap positive control 0.98, harness
1.7e-7). A within-block sub-split pins it to **one channel — `pp_fraction` +0.875** (the R-CALC
expected-damage `score` is inert at +0.016). Mechanism (triangulated): live loop states keep **93.6%** of
moves at full pp (the agent never attacks) vs the shard's **55%**, so "all moves at full pp deep in a
game" is off-manifold and the policy reads it as a switch cue → the loop self-sustains through pp. This is
the concrete carrier of Build-6's memoryless absorbing loop and Build-7's H2. **Next lever (Build 9,
BC-gateable):** ablate/robustify the pp channel, or add an explicit last-action feature; then re-run the
behavior probe. Phase C (corpus-teampool A/B) deferred — it tests composition (G2), which Phase B
excluded. Suite **244 pass**; `results/drift_probe.json`.

**OU pivot Build 9 — the pp-channel fix, two gates (drop pp / add first_turn): both triangulate that the
fix must ROBUSTIFY pp, not remove or out-vote it.** *Gate A (drop pp)* — ablate `pp_fraction` to a constant
(re-ingest, byte-identical to v5 on every non-pp column), 3-seed BC. Val-acc collapses **0.647 → 0.390**:
pp is **load-bearing for imitation** (the encoder's implicit own-move-usage/recency trace), not droppable →
BC RED, kill-gate stops before a live probe. *Gate B (keep pp + add an explicit `first_turn` recency
channel)* — `active.first_turn` is drift-free (same protocol messages offline+live) and the shard signal is
ideal (`first_turn=1` → switch 9.8% vs 23.9%); BC val-acc **0.654** passes and *beats* the pp-only 0.647.
But live the loop **persists** (both arms `c_pathological`, max-switch-run 110/77). A causal counterfactual
on the v7 policy shows why: neutralizing pp to in-distribution drops live switch mass **0.604 → 0.244
(ΔP −0.36)**, while flipping `first_turn` moves it only **±0.03** — with pp present, pp out-weighs the
honest feature **~10×**. `first_turn` is **kept** (drift-free, +val-acc) but insufficient alone. **Next
lever (Build 10, BC-gateable):** robustify pp via train-time augmentation (noise/dropout, or synthesize
full-pp deep-game states). Suite **245 pass**; `results/bc_gate9a.json`, `results/first_turn_gate9b.json`,
`results/behavior_probe9b.json`.

**OU pivot Build 10 — synthesize full-pp deep-turn states: BC PASS but the loop PERSISTS; the fix is
region-local and cannot reach the loop.** New train-time augmentation (`lategame/train/augment.py`,
`--pp-aug-frac`): on a fraction of attack-labeled, deep-turn rows, force the active mon's pp channels to
full, making "full pp deep in a game → attack" in-distribution — a train-time-only transform, so the
encoder/shards are **unchanged** (no `OBS_VERSION` bump). 3-seed BC (frac 0.5, turn ≥ 0.15) val-acc
**0.653 ≥ 0.63** (imitation preserved), but the live probe stays `c_pathological` on both arms (max-switch-run
108/989, win 0.0). The pp-reliance diagnostic explains it: on the **identical** v7 loop states, v10's pp-ΔP is
**0.390 — unchanged from v7's 0.397** (neutralizing pp still collapses switch mass 0.597 → 0.206). The
synthesized examples are *real mid-game attack states with pp maxed* — a different region than the
repeated-switch loop context, so the policy keeps both "full pp + attack ctx → attack" and "full pp + loop
ctx → switch"; region-local label augmentation can't touch the loop corner (increasing frac won't help).
**Next lever (Build 11, BC-gateable):** the *other* pre-registered candidate — **noise/dropout on the pp
channel** — a global regularizer that blunts the exactly-1.0 → switch extrapolation in all contexts, including
the unseen loop region. Suite **253 pass**; `results/bc_gate10.json`, `results/behavior_probe10.json`,
`results/pp_reliance_diag10.json`.

**OU pivot Build 11 — global pp regularization (noise + resample), decide by gates: PARTIAL WIN — the first
mechanism to move the live loop, attenuating it ~4× without fully breaking it.** The *other* pre-registered
candidate: a **global** pp regularizer applied in every context (unlike Build 10's region-local synthesize), so
it can reach the loop corner. Two flavors beside `augment_pp_full` — `augment_pp_noise` (Gaussian jitter,
`--pp-noise-std`) and `augment_pp_resample` (resample a fraction of pp cells from the shard's ~50%-full pool,
`--pp-resample-frac`) — plus the pp-reliance diagnostic promoted to a committed `scripts/pp_reliance_diag.py`.
Train-time only, no `OBS_VERSION` bump. **Screen (seed 0):** Gaussian **ruled out** (collapses BC at every
strength — σ 0.05 → 0.575 ≪ 0.63, pp is too load-bearing for additive noise); resample shows a clean frontier
(p 0.10 → val **0.648**, ΔP 0.203; p 0.25 → 0.627, ΔP 0.113; p 0.50 → 0.551, ΔP 0.039). **Winner = resample
p 0.10** (only BC-pass). **Confirm (3-seed + live):** BC val-acc **0.644 ≥ 0.63**; on the identical v7 loop
states pp-reliance is **halved** (ΔP **0.187** vs v10 0.369 / v7 0.377, baseline switch mass 0.404 vs 0.597).
Live, the loop **attenuates ~4×** (max-switch-run **29 vs Build 10's 108**, voluntary-switch 0.33 now *below*
the 0.5 pathology bar) but stays `c_pathological` on the heuristic arm (short ping-pong loops, win 0.0); the
random arm flips to `b_team_order` (a separate decode issue surfacing once the loop shrinks). **Verdict: pp is
confirmed the causal carrier and a global regularizer does move the loop, but the BC-passing frontier caps
ΔP-reduction at ~0.19 — insufficient to fully break it.** Next (Build 12): encoder-level pp transform (would
need an `OBS_VERSION` bump) or a hybrid targeted+global resample. Suite **266 pass**; `results/bc_gate11.json`,
`results/bc_gate11_screen.json`, `results/behavior_probe_v11.json`.

**OU pivot Build 12 — resample p 0.25 relaxed-bar probe + close `b_team_order`: FRONTIER-CONFIRMED, the
loop is still not broken and the residual ping-pong is pp-INDEPENDENT → pivot off pp.** The cheap, decisive
follow-up to Build 11 (runs-only: no code, no `OBS_VERSION` bump, no re-ingest). Pushed resample from the
BC-passing p 0.10 to **p 0.25** (3 seeds, same arch) with a pre-registered relaxed bar, to test whether the
loop or BC is the binding constraint. **BC — marginal:** 3-seed val-acc **0.6236** (misses 0.63, ~within noise
below the 0.625 relaxed line; ~2pts under p 0.10's 0.644 — the imitation frontier is real). **pp-reliance keeps
dropping** (offline, identical v7 states): ΔP **0.122** (v11 0.187 / v7 0.377), baseline switch mass 0.343.
**Live (n=20):** the loop is **shorter but not broken** — `bc_vs_heuristic` stays `c_pathological` (max-switch-run
21 vs v11 29, but **ping-pong 0.55 ≈ v11 0.52**, win 0.0); `bc_vs_random` flips back to `c_pathological` with
max-switch-run 14 (v11 73) and **win 0.9** (v11 0.3). The pp-carried long-run component keeps shrinking, but the
short **A→B→A ping-pong is pp-INDEPENDENT** — its rate doesn't move as ΔP halves again. **`b_team_order`
resolved as noise** (v11's flag was `stable_rate` 0.95 from 1/20 battles with `match_rate` already 1.0; v12 shows
1.0/1.0 on both arms — didn't reproduce; candidate closed, no separate build). **Verdict: pushing pp-reliance
lower — more resample OR an encoder pp transform — won't break the ping-pong because it isn't pp-carried.
Deprioritize the encoder transform; pivot to the pp-independent ping-pong** (localize its carrier à la Build 8,
or a decision-time anti-repetition / RL loop-penalty). p 0.10 (v11) stays the best BC-passing resample point.
Suite **266 pass** (no code touched); `results/bc_gate12.json`, `results/pp_reliance_diag12.json`,
`results/behavior_probe_v12.json`.

**OU pivot Build 13 — localize the ping-pong carrier (`scripts/pingpong_probe.py`, clone of Build 8's causal
swap on the 2-cycle states): VERDICT `PP_CARRIED` (3-seed) — the pp-independent *rate* is a sticky-argmax
artifact; the 2-cycle *decision* is ~100% pp-carried → reverses Build 12's deprioritization of the encoder pp
transform.** Runs-only (no `OBS_VERSION` bump / re-ingest / retrain): new probe + `behavior_probe.two_cycle_rows`
(single-sources the ping-pong definition) + tests, reusing `drift_probe`'s swap/verdict/controls and
`pp_reliance_diag.neutralize_pp`; fresh aligned capture from the shipped v11 winner (`bc_gen9ou_v11_s0`, n=50).
The naive metric just re-finds pp (a 2-cycle active mon just switched in → full pp, Build 8's OOD cue), so the
gate is **two-stage** — neutralize pp (frac_full 0.99→0.51), then localize the *residual* — with metric
**P(return action)** conditioned on own active species. On **337** A→B→A rows (322 matched, 9 species; controls
pass): baseline P(return) **0.306 → pp-neutralized 0.073 → full-neut 0.076** (pp explains ~100% of the drop,
residual **0.237 < 0.40**), all 3 seeds `PP_CARRIED`; switch mass **0.611 → 0.152** (dP 0.46). The decisive
disambiguation: **`P(return|switch)` is pp-INVARIANT (0.501 → 0.479)** — pp drives *switch-vs-stay*, while the
bounce-back to A is a structural ~50% coin-flip (uniform floor ~0.157). Reconciles Build 12: resample shrank the
pp margin but couldn't flip the discrete argmax, so the rate stayed flat while the mechanism stayed pp.
**Verdict: the resample lever is exhausted (plateaued, and pp is load-bearing so can't be dropped). Build 14 =
(a) encoder pp-transform (change pp's representation to break the argmax lock; imitation, but re-ingest + BC
risk) or (b) decision-time anti-repetition / RL loop-penalty (attacks the pp-invariant structural bounce-back;
cheap, no retrain) — recommend (b) first.** Suite **271 pass**, ruff + mypy(lategame) clean;
`results/pingpong_probe.json` + `results/behavior_probe_v13.json`.

**OU pivot Build 14 — decision-time anti-repetition (loop guard): the absorbing switch loop is BROKEN
(`max_switch_run` 58/26 → 2/2), with a milder interleaved ping-pong residual persisting.** New torch-free
`lategame/agents/loop_guard.py` (`LoopGuard`) wired into `BCAgent`/`OfflineRLAgent.choose_move` between
`masked_logits` and the argmax; a `--loop-penalty` flag on `behavior_probe` threads it through for a clean A/B.
No `OBS_VERSION` bump / re-ingest / retrain; `LoopGuard(0)` is exact identity. **Two mechanism iterations:** (1)
*return-only* (penalize just the switch-back) — live it merely converted the tight A→B→A into a longer
roster-cycle via **fresh-mon escape** (ping_pong 0.44→0.13 but `max_run` 17→26, switch mass flat, win 0) → ruled
out; (2) *streak* — a **soft escalating penalty on every voluntary switch** once a consecutive run forms
(`penalty·max(0, run−free_switches)`, `free_switches=1` keeps a scout/double-switch free), pressuring the
pp-driven switch mass toward attacking. **Result (n=50, v11 winner, p=4):** vs random `max_run` **58→2**,
vol_switch 0.171→0.107, win **0.54→0.58**; vs heuristic `max_run` **26→2**, vol_switch 0.337→0.169, win
0.00→0.02; fallback/rejection clean. **The absorbing consecutive loop (acute since Build 8) is broken and the
agent is functional vs random.** Honest caveat: `ping_pong_rate` fell ~0.55→0.30-0.37 but stays **> 0.25** — the
run resets on each attack, so the guard kills *consecutive* loops but not a slower interleaved oscillation
(switch→attack→switch→attack); it is much milder (win vs random 0.58), and the ~0 heuristic win reflects the OU
policy's general weakness (FORMAT_BOUND, gated off), not the loop. Suite **276 pass**, ruff + mypy(lategame)
clean; `results/behavior_probe_v14_{off,on}.json`.

**OU pivot Build 15 — OU ceiling re-probe (loop-fixed): the OU FORMAT_BOUND label was inherited from RB and is
now measured WRONG — OU is `MODEL_BOUND`.** Build 14 gave the first loop-free OU agent; no win-rate harness had
ever scored it (`arena`/`format_ceiling_gate` didn't thread `loop_penalty` and targeted the `offrl` arm, while
the shipped winner is `bc`), and `assess_ou` had always *withheld* the FORMAT/MODEL verdict for teambuilt formats.
This build closes both gaps (runs + small eval wiring; no `OBS_VERSION` bump / re-ingest / retrain): `loop_penalty`
now threads through `arena.build_player` (`_LOOP_GUARD_AGENTS`), and `format_ceiling_gate` gains `--bc-checkpoint`
/`--loop-penalty`, a loop-fixed `bc_v11` M1 arm, and an OU FORMAT-vs-MODEL verdict in `assess_ou` (applies the
Lever-15 `HEADROOM=0.58` threshold to the competent-bot reference). The stale RB `offrl_green` arm (checkpoint
pinned to encoder v2/760, un-loadable since the v5/761 bump) is dropped on OU. **M1 (n=300, `bc_gen9ou_v11_s0` +
`LoopGuard(4)`):** harness clean (mirror **0.510**, gradient monotone random **0.027** < maxbasepower **0.060** <
simpleheuristics **0.620** [0.564, 0.673]), band width **0.593 > RB 0.516**. A *simple competent bot* clears the
heuristic by **0.62 ≥ HEADROOM** ⇒ OU rewards skill, the format is **not** the ceiling → **`MODEL_BOUND`,
FORMAT_BOUND rejected for OU**. Our loop-fixed winner sits at **bc_v11 0.053** [0.033, 0.085] — near the
random/maxbasepower floor, **model_gap 0.567** below the competent bot: the ~0 heuristic win is a *model* gap, not
a format cap. **This flips the project posture: OU has real headroom → OU strength (PPO self-play / better BC) is
the justified next build; the interleaved ping-pong residual deprioritizes (won't lift heuristic win).** Suite
**281 pass** (276 + 5), ruff + mypy(lategame) clean; `results/format_ceiling_gate_ou_v15.json`.

**OU pivot Build 16 — PPO self-play on OU: the first method to move OU vs-heuristic with CI-clean significance;
`AMBER` (mechanism validated, gap dented not closed).** Build 15 named PPO self-play as the strength build; the RB
PPO was `AMBER` but RB was *format-capped*, so Build 16 tests whether on-policy PPO compounds where OU headroom
provably exists. Preflight caught that no OU offrl checkpoint was on the current encoder (build-number "v5" ≠
encoder v5; all were v2/v3/760) and the v11 BC winner has no *fitted* critic, so a v5/761 fitted-critic warm-start
was retrained via `train-rl` on the v7 RL shard (BC-init v11) → `offrl_gen9ou_v7_s0.pt` (value-MAE 0.53). Wiring
(runs + team/format plumbing; no `OBS_VERSION` bump / re-ingest / BC-retrain): `team` + `loop_penalty` thread
through `collect_rollout`; `PPOConfig`/`run_ppo` (+ a format-consistency guard)/`_eval_point`; `ppo_continue_gate`
gains `--team-pool`/`--loop-penalty`/`--ckpt-prefix`; `format_ceiling_gate` gains a dedicated `offrl_ou` learned
arm + `model_gap`. A smoke settled the loop-guard fork (`lp=0` → the offrl agent loops to a 1000-turn auto-tie,
0.000 vs random; `lp=4` functional), and `PPORecordingAgent` records `old_log_prob` from *un-penalized* logits
(learner stays on-policy; the guard only stops opponents/eval-arms stalling). **Full gate (3 seeds × 10 iters, lp=4):
PPO works** — `vs_iter0` **0.849 ± 0.017** (decisive, no collapse), `vs_random` **0.40 → 0.78**. Authoritative M1
(n=300, best ckpt + `LoopGuard(4)`, harness clean): **`offrl_ou` 0.133 [0.099, 0.176]** vs **`bc_v11` 0.057 [0.036,
0.089]** — disjoint CIs, a significant **~2.3×** gain; **`model_gap` 0.567 → 0.510**. **Verdict `AMBER` (positive):
the RB AMBER did NOT transfer — PPO self-play genuinely improves the OU policy — but 0.133 « the competent bot 0.643,
so the gap is dented ~10%, not closed.** Suite **289 pass** (281 + 8), ruff + mypy(lategame) clean;
`results/ppo_ou_gate_v16.json`, `results/format_ceiling_gate_ou_v16.json`. Open next (AMBER follow-ups): expand the
12-team self-play pool (prime ceiling suspect), more PPO iters (curve still climbing), stronger warm-start / more BC data.

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

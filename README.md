# Lategame

A competitive Pokémon Showdown ML battle agent for **Gen 9 OU**, played on a local Showdown
server — with the whole pipeline built and reported on **Gen 9 Random Battles** first, until
Lever 15 measured that format's ceiling and forced the pivot (see Status). See [plan.md](plan.md)
for the full PRD and roadmap.

## Status

A complete experimental pipeline is built and verified — rule-based baseline → behavior
cloning → offline RL → self-play → entity transformer + on-policy PPO → human-replay
ingestion (public-log **and** full-fidelity re-simulation).

**Where the project stands against its own goals.** The target format for **G2** is **`gen9ou`**,
amended from Gen 9 Random Battles *by measurement*: Lever 15 found gen9-RB **FORMAT_BOUND** — the
strongest competent bot reaches 0.523 vs the heuristic (CI spanning 0.50) and near-optimal depth-2
search reaches 0.500, so G2 was unreachable there whatever the model. On `gen9ou` the same
diagnostic found real headroom (`simpleheuristics` 0.633, heuristic mirror 0.487).
**§12's "> 50% vs the heuristic" bar is cleared decisively**: `v26b`'s selection-free terminal read
is **0.7513** (pooled n=9000/arm, 3 seeds), far above the strongest scripted bot. **G2's headline
metric is computed too** — GXE/Glicko-1 need a varied opponent field, which no amount of further
training can supply, so it came from an **agent-only eval ladder** (`lategame eval-ladder`, the
"private/agent-only server or eval ladder" §12 asks for) rather than from the human ladder, which
NG3 puts out of scope. Over 9 agents × 36 pairs × 300 battles on `gen9ou`, **`v26b` reaches Glicko
1776.3, GXE 0.7434**, above `simpleheuristics` (1572.7 / 0.5695) and the `heuristic` anchor
(1500 / 0.5000). **G3 is booked** on the five-dose continual-improvement curve (80 → 320 updates,
monotone on both inference reads). **G1** is built and verified. **G4 is MET** — three formats play
end to end through one core, and Build 28 then ran the *strength* campaign on the third
(`gen9vgc2025regi`) that G4's exit criterion deliberately did not require: BC → offline RL →
factored PPO. That campaign's headline is a **correction**, not a win: the VGC shards it started
from were 94% frames from one looping bug of ours, so the previously reported doubles BC/AWR
numbers are withdrawn and re-run. On corrected data, factored PPO beats its own warm start
**0.530** [0.519, 0.540] over 9,000 battles (all three seeds individually clear 0.50) but is
**NULL** against the fixed heuristic — a format whose competent bots sit at parity with each
other, so the pre-registered stop rule fires and the strength axis is ceiling-bound. That leaves
**G5**, as *demonstrated* rather than merely implemented capability.

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

**OU pivot Build 17 — extend PPO self-play iterations (10 → 25): the run was cut short, not plateaued; extending it
~2.3×'d OU vs-heuristic. Stronger `AMBER`.** Build 16's curves settled which follow-up to run first: every metric was
still monotone-climbing at iter 10 with `best_iter` = the *final* iter for 2/3 seeds — you can't diagnose a
"12-team-pool ceiling" from a run that never plateaued. So Build 17 isolates the cheapest, only directly-evidenced
lever (more iters). Runs-only, no code change — a fresh 25-iter run from the **same** warm-start + 12-team pool + lp=4,
seeds 0/1/2, `--eval-n 100`, `--ckpt-prefix ppo_ou_long`. **Result:** every metric kept climbing with no collapse —
`best_vs_heuristic` **0.307 ± 0.065** (0.23/0.30/0.39 at iters 19/25/22, s1 peaks at the *final* iter), `vs_iter0`
**0.947**, `vs_random` → 0.91–0.98. Authoritative M1 (n=300, best ckpt `ppo_ou_long_s2/iter_22` + `LoopGuard(4)`,
harness clean — mirror 0.473, monotone gradient, band 0.607 > RB 0.516): **`offrl_ou` 0.303 [0.254, 0.358]** vs v16's
**0.133 [0.099, 0.176]** — **disjoint CIs**, a real **~2.3×** gain from iterations alone, ~5.3× over the unchanged
`bc_v11` 0.057; **`model_gap` 0.510 → 0.317** (44% below Build 15's 0.567). **Verdict stronger `AMBER`:** iterations,
not the pool, were the binding lever — the same 12 teams/warm-start rose 0.133 → 0.303 on more training, and the curve
is *still climbing* at iter 25. `MODEL_BOUND` reconfirmed; gap dented ~1/3 more, not closed. Suite **289 pass** (no
code change), ruff + mypy clean; `results/ppo_ou_gate_v17.json`, `results/format_ceiling_gate_ou_v17.json`. Open next:
plateau still not reached → **extend iters again** (cheapest, still-evidenced); team-pool expansion and a stronger
warm-start become diagnosable only once vs_heuristic flattens while vs_random stays high.

**OU pivot Build 18 — extend PPO self-play iterations (25 → 50): the curve PLATEAUED, at a *higher* level. Outcome 1
(plateau found) + stronger `AMBER`.** Build 17's pre-registered decision tree said extend once more to resolve the
asymptote. Single-lever, runs-only — only `iters` 25→50 (same warm-start, 12-team pool, lp=4, eval protocol),
`--ckpt-prefix ppo_ou_x50`. (Infra: session teardowns reap all processes *and* wipe scratchpad — even setsid-detached
daemons didn't survive — so the run was completed **one seed at a time**, persisting each iteration to the repo-disk
`checkpoints/`; seed 0 salvaged complete, seeds 1,2 re-run standalone, identical config.) **Result — UNANIMOUS PLATEAU:**
`best_iter` = **41 / 44 / 46** (all *interior*, vs v17's 19/25/22 with s1 at the final iter); each tail (35–50) is flat
within noise and *above* the mid-run mean (climb-then-flatten, not decline); `vs_iter0` never < **0.90** (final mean
**0.977**) → no collapse, the *destabilized* branch is ruled out. `best_vs_heuristic` per-seed **0.503 ± 0.048**
(0.57/0.46/0.48) — *higher* than v17's 0.307. Authoritative M1 (n=300, best ckpt `ppo_ou_x50_s0/iter_41` +
`LoopGuard(4)`, harness clean — mirror 0.520, monotone gradient 0.013 < 0.073 < 0.633, band 0.62 > RB 0.516):
**`offrl_ou` 0.453 [0.398, 0.510]** vs v17's **0.303 [0.254, 0.358]** — **disjoint CIs**, a real **~1.5×** gain from
iterations; `bc_v11` unchanged **0.043**; **`model_gap` 0.317 → 0.18** (68% below Build 15's 0.567), PPO now near parity
with the competent heuristic. `MODEL_BOUND` reconfirmed. **Verdict: plateau found — the iterations lever is exhausted at
50 and it paid off (gap halved).** Suite **289 pass** (no code change), ruff + mypy clean; `results/ppo_ou_gate_v18.json`,
`results/format_ceiling_gate_ou_v18.json`. Open next: the plateau makes the expensive levers diagnosable — **team-pool
expansion** (`build_ou_teampool.py`, 12 → ~24) and a **stronger warm-start** are the evidenced Build-19 candidates (each
isolated); lr-decay / PFSP hardening is *not* indicated (no instability observed); more iterations retired.

**OU pivot Build 19 — the PPO train/eval objective mismatch: MEASURED, FIXED, and ruled OUT as the plateau's cause.
`NULL` on strength; the schedule lever is RETIRED.** Re-reading the v18 curves overrode v18's own decision tree: the
agent **trains directly against `simpleheuristics`** (it is the PPO anchor, injected every iteration) and after 50 iters
still wins only **26–45%** against it — a *fixed, scripted, non-adapting* opponent already in the training mix is not
beaten by more opponent *variety*, so the plateau is in the **learner**, not the opponents. That **refutes team-pool
expansion as the binding lever** (and it is unmeasurable as specified: one pool feeds rollouts *and* eval, so expanding
it moves the metric out from under v18's CI). Capacity is weaker than assumed too — the net is already
over-parameterized for imitation (0.72M params on 61,723 BC rows). So Build 19 ran the cheapest learner-side lever: the
**`ent_coef`/`lr` schedule**, fixed for all 50 iters until now.

*Stage A — a probe qualified the spend before paying it (`scripts/policy_sharpness_diag.py`, new).* Reading the code
**falsified the naive story for free**: **eval is already greedy** (`_eval_point` → `sample=False` → argmax) while
**rollout samples**, so entropy cannot *directly* cost eval win-rate — only *indirectly*, by holding the distribution
soft so the argmax lags. The probe (v18 best ckpt, 1266 frozen live states + a greedy/sampled A/B at n=300) found the
mismatch is real and large: policy entropy **stalls exactly when the win-rate stalls** (`h_ratio` 0.571 → 0.493 → 0.472
→ **0.465**, `max_prob` flat at 0.682 from iter 10), and **greedy 0.487 vs sampled 0.347 → a +0.140 gap** — PPO was
maximizing the return of a distribution playing 14 points worse than the one we deploy. → **LIVE.**

*Stage B — schedule (`PPOConfig.ent_coef_final`/`lr_final`, a pure `anneal()`, four `ppo_continue_gate` flags;
`*_final=None` ⇒ constant ⇒ bit-identical to Builds 16–18, proven by smoke).* Run identical to v18 except `ent_coef
0.01 → 0.0`, `lr 2.5e-4 → 5e-5`. **The mechanism engaged and the metric did not move.** Sharpening worked (`h_ratio`
**0.465 → 0.336**, `max_prob` → **0.788**, near-deterministic decisions 27% → **44%**) and the objective mismatch
**closed to zero** (**greedy 0.433 vs sampled 0.437, gap −0.003**, was +0.140). But strength is flat: 3-seed
`best_vs_heuristic` **0.493 ± 0.063** vs v18's 0.503 ± 0.048, and authoritative M1 (n=300, harness clean — mirror 0.487,
gradient 0.030 < 0.067 < 0.643) gives **`offrl_ou` 0.490 [0.434, 0.546]** vs v18's **0.453 [0.398, 0.510]** — **CIs
overlap** where pre-registration demanded *disjoint above 0.510* ⇒ **NULL**. `model_gap` 0.18 → 0.153 (n.s.); sanity
guards clean (`vs_iter0` ≥ 0.96 — the lr decay did not destabilize).

**What the null teaches:** the +0.140 gap was **the cost of *sampling*, not headroom in the *argmax***. Annealing pulled
the **sampled** policy *up to* the argmax (0.347 → 0.437) without pushing the **argmax** higher — and the argmax is what
we score. The mismatch was real, is now fixed, and is **not** the plateau's cause. Keep the schedule as the OU default
(free, and it removes a confound) but claim no win from it. Suite **296 pass** (289 + 7), ruff + mypy clean;
`results/ppo_ou_gate_v19.json`, `results/format_ceiling_gate_ou_v19.json`, `results/policy_sharpness_diag_v19*.json`.
**Open next (Build 20)** — with entropy/lr *and* opponent-variety both eliminated, the suspects in cost order:
**(1) per-iteration sample budget** — `games_per_opp=16` × 3 opponents = **48 battles ≈ ~2K transitions ≈ ~32 gradient
steps per iteration** (~2,400 battles for the whole run): a very small RL budget and a plausible **noise-floor** plateau;
`--games-per-opp 48` is already a flag ⇒ **zero code change**. **(2) capacity** — a bigger warm-start (`factory.py`
already reads `d_model/n_layers/…` from `arch`; only `bc.py::_build_model` fails to populate them). *Methodology:
`best_vs_heuristic` is a max over 50 noisy n=100 evals — an optimistically biased statistic (winner's curse). Compare
builds on the authoritative n=300 CI.*

**OU pivot Build 20 — the per-iteration sample budget: the plateau is a STATIONARY POINT, not a sampling-noise floor.
`NULL` on strength; the whole optimization/sampling family of levers is RETIRED.** The lever was 3× the rollout
(`--games-per-opp 16 → 48`, zero code change). Reading the code first reprised its usual value: each iteration plays
**48 rollout battles but 400 eval battles** (`_eval_point` runs `eval_n=100` against 3 baselines *plus* iter0), so ~89%
of every PPO run's battle budget is **measurement, not learning**, and tripling the rollout costs only **~+21% battles**
— not the 3× wall-clock the Build-19 notes assumed. It also killed the naive mechanism: advantages are normalized
**per-buffer** and the epoch loop **KL-early-stops**, so per-iteration policy *displacement* is governed by the trust
region, not the step count. A bigger buffer buys a **lower-variance estimate of the gradient direction**, nothing else.

*Stage A — the probe (`scripts/grad_noise_diag.py`, new) reframed the build before it was paid for.* On shipped v19
checkpoints it takes the gradient at θ_old (ratio 1, clip inactive ⇒ the vanilla policy gradient) and compares it across
**independent rollouts**. Result — sharper than the pre-registered question: **the noise is constant and the signal
vanishes.** `tr(Σ)` is flat across training (**3284 → 2741 → 3295** at iters 10/47/50) while **`|G|²` collapses
(4.54 → 0.93 → −0.41**, a *negative* estimate ⇒ indistinguishable from zero); `B_simple` explodes 723 → 2945 → ∞ only
because it is the **ratio**, and the plateau lands exactly where the ~1.6K buffer crosses the noise scale. Verdict
`NOISE_LIMITED` (pre-registered) → run Stage B — **but flagged at the time as the Build-19 trap in a new costume:
`B_simple → ∞` is the signature of a growing noise floor *and* of a vanishing gradient (a stationary point), and a
bigger batch estimates a near-zero gradient more precisely — it cannot manufacture one.** Two method fixes, both
load-bearing: compare **independent rollouts**, never two halves of one buffer (halves share that rollout's
league/team/episode draw ⇒ biased *toward agreement* ⇒ could have manufactured a false `SIGNAL_LIMITED` and wrongly
cancelled Stage B); and run **two arms**, since `--games-per-opp` only buys battles against the mix an iteration
*already drew* (`same_mix` decides the verdict; `fresh_mix` isolates opponent-**selection** variance no budget can touch
— measured `opponent_draw_dominates=False` everywhere, refuting that alternative).

*Stage B — the run (zero code change).* **The lever landed and the metric did not move.** Telemetry is unambiguous:
gradient steps/iter **24 → 71** (2.96×), `epochs` held at **4 in 40/40** late iterations, `approx_kl` 0.008 → 0.014
against a 0.045 bar ⇒ **the trust region never bound**, and the critic even fit better (`vmae` 1.44 → 1.09). 3-seed
`best_vs_heuristic` rose **0.493 → 0.567 ± 0.029**… and it was **almost entirely winner's curse**.

*The measurement itself had to be fixed (`scripts/seed_strength_gate.py`, new).* The old authoritative protocol — score
the **single best checkpoint** at n=300 — is **not fit for these comparisons**: it is **underpowered** (a build-vs-build
difference has SE ≈ 0.041 ⇒ resolves only gaps **> 0.08**, while the candidate effect was 0.074) and
**selection-biased** (that checkpoint is the argmax over ~150 noisy curve evals, then re-scored ⇒ regression to the mean,
build-dependent: v20's best fell 0.600 → 0.480, v19's 0.493 → 0.490). The fix, applied **symmetrically to both builds**:
score **every seed's best** checkpoint and pool (SE 0.041 → **0.024**, resolving +0.07 at z ≈ 3), and read a **z-test**
alongside CI-disjointness (CI overlap is a *conservative* test — at +0.05/900 the intervals overlap while p = 0.034).
**Corrected result: v19 `0.448 [0.416,0.480]` → v20 `0.472 [0.440,0.505]`, diff +0.024, z = 1.04, p = 0.30 ⇒ NULL.**

**What the null teaches:** **3× the samples collapsed seed-to-seed variance ~7× (std 0.074 → 0.010) without moving the
mean.** That is precisely the signature of estimating a **near-zero** gradient more precisely — a far more *reproducible*
policy that converges to the same place. Stage A's reframing was right: **the plateau is a stationary point of the PPO
objective, not a noise floor.** Combined with Build 19 (entropy/lr, NULL) and the refutation of opponent variety, the
entire **optimization/sampling family is exhausted** — the binding constraint is the **model class**. Suite **328 pass**
(296 + 25 + 7), ruff + mypy clean; `results/grad_noise_diag_v20.json`, `results/ppo_ou_gate_v20.json`,
`results/format_ceiling_gate_ou_v20.json`, `results/seed_strength_gate_v20.json`.
**Open next (Build 21) — CAPACITY**, now the sole indicated lever: a bigger model changes the landscape and can carry a
nonzero gradient where the current **0.72M-param** net has none. `train-rl --model entity_transformer --d-model 256
--n-layers 4 --n-heads 8` trains a wide net **from scratch with zero code change** (PPO's `build_model(ckpt)` reads
`arch` and fine-tunes all of it), **bypassing BC's 0.63 val-acc gate** — which matters, since at 0.086 rows/param that
gate would likely reject a bigger teacher for *overfitting*, a false negative w.r.t. what PPO wants. Confound: from-scratch
offline-RL loses the BC warm-start, so the clean version still needs 4 edits to `bc.py`/`cli.py`. Footgun found:
**`--d-model` is silently ignored when `--bc-init` is passed** (`offline_rl.py:215-223` overwrites `model_meta` from the
checkpoint's `arch`). Secondary: the **critic** (EV ≈ 0.30 — a better critic shrinks `tr(Σ)` with no extra samples).
*Methodology, updated: compare builds with `seed_strength_gate.py` (pooled seed-bests + z-test). The single-checkpoint
n=300 gate is retired for build-vs-build — it cannot see effects below ~0.08.*

**OU pivot Build 21 — CAPACITY (0.72M → 4.56M params): NULL, and the model-side story is CLOSED.** Warm-start widened to
a 4.56M-param entity transformer, PPO otherwise identical to v20. `seed_strength_gate`: v20 `0.499 [0.466,0.531]` → v21
`0.452 [0.420,0.485]`, **diff −0.047, z −1.98, p 0.047**. 6.3× the parameters made the agent slightly *worse*. The NULL is
unconfounded — the trust region bound in only **4/150** iters (96–98% full-epoch). The explainer: **`|G|²` never recovers,
it was never there** — v21 enters PPO at `|G|²` 0.057, where the narrow net *ended up* after fully converging, failing the
pre-registered "a candidate must show `|G|²` recovering" filter outright. *Banked:* the wide net is a far better critic
(offline value MAE 0.531 → 0.370), and a from-scratch control matched the BC warm-start at 0.6485 accuracy **exactly**,
retiring a pipeline stage as decorative. *TRAP booked:* `B_simple` is a **ratio** `tr(Σ)/|G|²` — v21's exploded because the
denominator collapsed, not because noise grew (`tr(Σ)` *fell*, 3284 → 580), so its own `NOISE_LIMITED` "collect more
samples" verdict is exactly wrong. `results/{grad_noise_diag,ppo_ou_gate,seed_strength_gate}_v21.json`.

**OU pivot Build 22 — WEAKEN THE OFFLINE FIT (H22: the wide net is *born* flat, so PPO inherits a stationary point):
mechanism CONFIRMED, strength NULL — and the NULL is NOT ATTRIBUTABLE.** If capacity did not help because the wide net
arrives at PPO already converged, a *less*-converged init should carry gradient. Three stages.
*Stage A — the wide net is born flat (CONFIRMED).* `|G|²` measured at `iter_0`, before PPO touches anything: a **47×**
narrow/wide gap, far outside any noise this build later uncovered. Stage A stands.
*Stage B-0 — the dose-response (e3/e10/e30), and the finding that outlives the build.* The pre-registered gate PASSED on
the **e10** arm — then a replication at a second probe seed **removed half of what passed it**: `|G|²` swung **2.7× on a
fixed checkpoint from the probe seed alone**, larger than the **2.2×** effect the gate was built to detect. **`|G|²` is not
a usable instrument at this budget.** The *cosine* replicated tightly (0.312 → 0.317) and separates e10 (~0.31) from both
converged nets (~0.185); that half survived.
*Stage B — the strength test.* PPO warm-started from `offrl_gen9ou_wide_e10_s0` (acc 0.613 / vMAE 0.703) instead of v21's
converged wide net — offline convergence the only changed variable. v22 `0.4622` (416/900) vs v21 `0.4522`, **+0.010,
z 0.43, p 0.67 ⇒ NULL**. **But §13.1's own rule refuses to attribute it:** the trust region **bound in 59/150 iters** at
56–64% full-epoch, against v21's **4/150** at 96–98% (`approx_kl_mean_late` 0.031 vs 0.023). v22 was **throttled through
36–44% of its optimization**, and a NULL under a binding trust region cannot separate "the lever does nothing" from "the
lever worked and the optimizer would not follow it." The caveat was written into the merged gate's `_note` *before* the
comparison ran. Two independent signs the run was **cut off, not converged**: seed 1's best checkpoint is `iter_50` — the
**final** iteration — and it is also the strongest single checkpoint (0.5033).
*CALIBRATION FINDING — build-vs-build differences of ~0.03 are NOT resolvable in one run.* v20's **identical** checkpoints
scored **0.472** in one scoring and **0.499** in another — 0.027 apart, right at the 0.023 SE. Consistent with sampling
noise, not a bug, **but it makes Build 21's headline regression fragile**: against v20's *other* scoring the same v21 sits
only −0.020 away, nowhere near significance. Capacity's *direction* (it did not help) survives; "capacity is a
**regression**" does not. Score both arms **in one run** from here on (`scripts/cluster/strength_gate.slurm`).
`results/{grad_noise_diag_b22_*,ppo_ou_gate_v22,ppo_ou_telemetry_v22,seed_strength_gate_v22}.json`.
**Open next (Build 23) — RAISE THE KL BUDGET, indicated by telemetry rather than theory.** `kl_bar` is **0.045** and v22's
`approx_kl_max` reached **0.074**: for the first time in this project the binding constraint is the **trust region**, not a
vanished gradient — Builds 19–20 closed optimization/sampling, Build 21 closed capacity. Raising it is the one experiment
that converts this NULL into an attributable result; extending past 50 iterations is indicated by the same data. *Both
arms must be re-run at the new budget* — comparing a raised-budget arm against v21's old-budget checkpoints would differ in
init **and** budget, reproducing exactly the ambiguity that cost Build 22 its verdict.

**OU pivot Build 23 — OPEN THE TRUST REGION: H22 is REFUTED, and the refutation is ATTRIBUTABLE. Weakening the offline fit
does not merely fail to help — it HURTS.** `target_kl` 0.03 → 0.06 (`kl_bar` 0.045 → 0.090) and `iters` 50 → 80, raised
**identically in both arms** so offline convergence stays the only changed variable. Two fresh 3-seed sweeps: `v23a` from
the reduced-epoch `e10` init, `v23b` from the converged wide init.
*The throttle is gone.* `v23a` bound **6/240** iterations (96–99% full-epoch), `v23b` **0/240** (100%), against v22's
**59/150** at 56–64%. `approx_kl_max` 0.0785/0.0797 under the 0.090 bar; no collapse (final entropy 0.63–0.68, `vs_iter0`
0.99–1.00). The pre-registered precondition for attribution is met.
*The verdict.* One run, both arms, 1800 battles: `v23b` **0.5578** [0.525, 0.590] vs `v23a` **0.4733** [0.441, 0.506] —
**diff −0.084, z −3.58, p = 0.0003, CIs disjoint.** H22 predicted a *positive* diff. The measured effect is negative and
significant: the strongest signal this project has produced on the strength axis, pointing the opposite way from the
hypothesis. **Booked honestly: this fell outside the pre-registered rows**, which anticipated CONFIRMED or an attributable
NULL, not a significant *reversal*.
*Why the mechanism evidence was right and the inference wrong.* Everything Build 22 banked replicates: unthrottled, the
e10 init's `approx_kl_mean_late` is **0.056–0.058 against 0.032–0.038** — ~1.6× larger steps, exactly as the cosine
(0.31 vs 0.185) and the 20×-more-binding trust region predicted. **It takes bigger steps, and they go somewhere worse.**
`|G|²`-style "is there gradient here" reasoning cannot separate a *useful* gradient from a merely *large* one —
**real gradient ≠ useful gradient** — which demotes Build 21's pre-filter from sufficient to merely necessary.
*Tooling defect, recorded not buried:* `seed_strength_gate.py` computes `verdict = "WIN" if significant and diff > 0 else
"NULL"`, so it stamped this p = 0.0003 reversal **`NULL`**, identically to a p = 0.67 nothing. Read `diff`/`z`/`p`, not
`verdict`; a three-way WIN/NULL/REGRESSION is owed.
*The accidental finding, confounded and therefore NOT claimed:* **`v23b` = 0.5578 is the highest pooled rate in the
project's history**, against a previous best of 0.4989 and a v19–v22 band of 0.448–0.499. But it differs from v21 in *both*
`target_kl` and `iters`, and the comparison is cross-run (±0.027 per the calibration finding). +0.059 clears that noise
floor and is the first thing in five builds to move this far — so it is the **next thing to test**, not a result.
`results/{ppo_ou_gate_v23a,ppo_ou_gate_v23b,ppo_ou_telemetry_v23a,ppo_ou_telemetry_v23b,seed_strength_gate_v23}.json`.
**Open next (Build 24) — SEPARATE THE BUDGET LEVERS**, on the converged init: `target_kl` 0.06 @ 50 iters and 0.03 @ 80
iters, both scored against `v23b` in one gate, to resolve which half of the raise did the work. The init-quality axis is
**closed** (refuted, sign against it), as are capacity (21) and optimization/sampling (19–20).

**OU pivot Build 24 — PRE-REGISTERED (written 2026-08-01, before running): WHICH HALF OF THE BUDGET RAISE DID THE WORK?**
A **2×2 factorial on (`target_kl`, `iters`) plus a decomposition arm**, five arms scored in ONE gate run, all warm-started
from the converged `offrl_gen9ou_wide_s0` at seeds 0/1/2: `v23b` (0.06/80, reused), `v24a` (0.06/50), `v24b` (0.03/80),
`v24c` (0.03/50 — v21's config, retrained fresh), and `v24d` (0.06/80 with the **anneal horizon pinned to 50**).
*Why the anneal knob had to ship first.* `iters` silently doubled as the lr/ent anneal horizon, so "50 → 80" was never one
change — at iteration 40, v22 sat at lr 9.08e-05 / ent 0.0020 while v23b sat at 1.51e-04 / 0.0051. `config.iters` is used
in exactly two places (the loop range and `anneal_horizon`), so with the horizon pinned **`v24d`'s iterations 1–50 run the
identical code path to `v24a`** — and `v24d − v24a` differs in nothing but whether the loop kept going.
*Four pre-registered contrasts at α = 0.0125:* the **total** (`v24c`→`v23b`, does +0.059 reproduce within one run?), the
**KL half** (`v24b`→`v23b`), pure **update count** (`v24a`→`v24d`), and pure **anneal** (`v24d`→`v23b`). The last two sum
to the `iters` half by construction — reported as a consistency check, not a fifth test. The gate prints all 10 pairs;
**the other six are not pre-registered.** Pre-registered rows cover NOT REPLICATED / KL DID IT / ITERS DID IT / BOTH /
**UNDERPOWERED SPLIT** (a real total with both halves under the MDE — explicitly *not* a finding that neither matters) /
REVERSAL / COLLAPSE.
*A deliberate departure from Build 23's rule:* trust-region binding in the `kl` 0.03 arms is **the treatment, not a
defect**. Build 23 voided a comparison when the trust region bound because there it was a nuisance; here `target_kl` **is**
the lever, so the certificate characterises the dose rather than invalidating the arm.
*Scope, stated up front:* the pooled z-test treats 900+ battles as iid and **the seeds are not** — Build 23's between-seed
sd is **~0.0706 against a within-seed 0.0287**, so its p = 0.0003 is a paired **t = 2.04** (3/3 seeds positive, sign-test
p = 0.25). Direction replicates; procedure-level evidence is far weaker than the pooled p reads, and no feasible seed count
fixes it (~24 seeds/arm for 80% power). Every contrast now carries a `seed_level` block, and **every Build 24 verdict is
scoped to the checkpoints it scored**, not to the training procedure.
*`N=1800`, not 300:* battles are the cheapest power here (1800 ran in 9:14). At a ~0.030 half, N=300/900/1800 gives ~7% /
~24% / **~73%** power at α = 0.0125 — at N=900 the likely outcome is the UNDERPOWERED SPLIT row, which answers nothing.
The argmax-selection bias favouring longer arms was simulated and is **+0.0014**, negligible.
*Cost:* `tron` caps a user at `cpu=32,mem=256G` and each task asks 8 CPU / 64 GB, so **exactly 4 of the 12 tasks run at a
time** (measured on Build 23's `sacct`: queued tasks started the second a slot freed). Slurm backfills, so all 12 are
submitted at once — but the estimate is **~60 task-hours ÷ 4 ≈ 15–18 h**, not "the arms run concurrently." Plus 13.3 GB of
checkpoints and a ~2.3 h gate.

**OU pivot Build 24 — SEPARATE THE BUDGET LEVERS: ITERS DID THE WORK. The KL raise did essentially nothing.** Ran 17.4 h
(12 tasks, 4 concurrent, exactly as the QoS predicted) plus a 2:09 gate. **The outcome landed inside the pre-registered
rows** — unlike Build 23.
*The 2×2*, one run, N=1800/checkpoint (5400/arm, 27,000 battles):

| | `iters` 50 | `iters` 80 | **iters effect** |
|---|---|---|---|
| `target_kl` 0.03 | `v24c` **0.449** | `v24b` **0.533** | **+0.084** |
| `target_kl` 0.06 | `v24a` **0.444** | `v23b` **0.550** | **+0.106** |
| **KL effect** | **−0.005** | **+0.017** | |

*The four pre-registered contrasts (α = 0.0125):* **total** `v24c`→`v23b` **+0.102** (p < 0.0001) ✔ · **KL half** +0.017
(p = 0.0725) ✘ · **update count** `v24a`→`v24d` **+0.075** (p < 0.0001) ✔ · **anneal** `v24d`→`v23b` **+0.031**
(p = 0.0011) ✔. The last two sum to +0.107 against the `iters` half's +0.106, as they must by construction.
*The accidental finding replicates and is now attributable.* Build 23's +0.059 was cross-run and confounded across two
levers; measured within one run against a freshly trained v21-configuration baseline it is **+0.102**. `v23b` itself
re-scored **0.550** against 0.5578 in the Build 23 gate — 0.008 apart, well inside the ±0.027 calibration band. **The
highest pooled rate in the project's history is confirmed, and its cause is identified.**
*Update count is the driver — and the FIRST LEVER IN THIS PROJECT ROBUST AT BOTH INFERENCE LEVELS.* It carries +0.075 of
the +0.106, per-seed diffs +0.068 / +0.067 / +0.091 — **3/3 seeds, seed-level t = +9.76**, against Build 23's headline
t = 2.04. `v24a` and `v24d` share init, seed, and (because `config.iters` is used only at the loop range and in
`anneal_horizon`) the **identical code path over iterations 1–50**, so the between-seed variance that swamps every other
contrast largely cancels here. Pinning the horizon did not just disambiguate the lever — it bought a ~7× better
seed-level statistic. **The anneal contribution is real but NOT seed-robust** (+0.031 pooled, seed-level t = +0.72, only
2/3 seeds agreeing) and is not reported as though it were.
*The trust region was never the constraint.* `v24b` spent **56/240** iterations throttled and still scored 0.533 against
0.550 — a −0.017 cost that does not clear α. Build 23 spent a build's plumbing making `target_kl` reachable on the theory
that the trust region was binding the lever. It was binding; **it just did not matter.** The KL axis is now **closed**.
`results/{ppo_ou_gate_v24a..d,ppo_ou_telemetry_v24a..d,seed_strength_gate_v24}.json`.
**Open next (Build 25) — PUSH THE ITERATION BUDGET UNTIL THE RETURN FLATTENS.** First lever since Build 16 that is
significant, replicated, seed-robust *and* mechanistically identified: **more updates**. Every other axis is closed —
init quality (23), capacity (21), optimization/sampling (19–20), trust region (24). Not saturated either: `v24c` s0 peaked
at its `iter_50` cap and `v23b` s1 at its `iter_80` cap. Run `iters` 80 → 120 → 160 at `target_kl` 0.06, **anneal horizon
pinned across arms** so the contrast stays on update count.

**OU pivot Build 25 — PRE-REGISTERED (written 2026-08-02, before running): WHERE DOES THE UPDATE-COUNT RETURN FLATTEN?**
Four arms scored in one gate, all warm-started from `offrl_gen9ou_wide_s0` at seeds 0/1/2, `target_kl` 0.06 throughout:
`v23b` (80 iters, reused anchor), `v25a` (120, **anneal pinned to 80**), `v25b` (160, **anneal pinned to 80**), and `v25c`
(160, anneal 160).
*Why the horizon pins at 80 and not 50.* With `anneal_iters` 80, the new arms run `v23b`'s exact schedule over iterations
1–80 and then continue at the finals — so the anchor is a legitimate cell rather than a fourth configuration, and
`v23b`→`v25a` inherits the pairing that bought Build 24 a 7× better seed-level statistic.
*Why `v25c` exists.* Past iteration 80 the pinned arms run at lr 5e-5 / ent 0, so a flat `v25a`→`v25b` would be ambiguous
between "updates saturate" and "the schedule froze" — the same shape of confound the KL story had. `v25c` separates them.
*Four contrasts at α = 0.0125:* 80→120, 120→160, the **total** 80→160, and the anneal horizon at 160. The first two sum to
the third by construction, reported as a consistency check.
*The one methodological departure from Build 24 — the selection bias no longer cancels.* Each seed's checkpoint is an
`argmax` over ~`iters` noisy evals, and these arms differ in length by **2×**. Simulated at σ_b = 0.0428 (estimated from
the Build 23/24 curves themselves), the differential bias is **+0.0045 / +0.0028 / +0.0073** — against Build 24's
negligible +0.0014, and ~30% of the smallest dose worth calling real. So the bias is **subtracted before a dose is
declared**, and a second **selection-free** gate scores each arm's *terminal* checkpoint via a new
`scripts/pin_gate_checkpoint.py` (it rewrites a merged gate JSON's `best_checkpoint`; the authoritative gate is untouched).
**A sign disagreement between the two reads is itself the finding.**
*Pre-registered rows:* STILL CLIMBING / KNEE BETWEEN 120 AND 160 / ALREADY SATURATED AT 80 / **FROZEN-SCHEDULE ARTIFACT**
(#2 null but the anneal contrast positive — no saturation claim may be made from such a build) / REVERSAL / COLLAPSE. A
second saturation read comes from `best_iter` itself, and is pre-registered *because* it can contradict the gate.
*`N=3000`, not 1800:* the per-step doses here are plausibly ~0.02–0.04, where N=1800 has ~48% power; N=3000 gives MDE
**0.025** at ~80%, and 36,000 battles is ~2:52 against the gate's 8 h limit (Build 24 measured ~209 battles/min).
*Cost:* 9 tasks ≈ **107 task-hours**, 4 concurrent ⇒ ~32 h wall-clock. **Disk, not time, is the binding constraint** —
~23 GB against 64 GB free.

**OU pivot Build 25 — STILL CLIMBING: 160 updates is not the ceiling, and scaling the anneal to match COSTS ~9 points.**
Both gates ran 2026-08-06 and completed clean: primary N=3000 (36,000 battles, 2:28:51) and the selection-free terminal
read at N=1800 (21,600 battles, 1:54:09). Pooled vs the heuristic, n=9000/arm: `v23b` **0.5450**, `v25a` **0.6144**,
`v25b` **0.6534** [0.6435, 0.6632], `v25c` **0.5604**. **0.6534 is the highest pooled rate in the project's history**,
against the 0.550 Build 24 set. The anchor re-calibrates at 0.5450 vs Build 24's 0.5504 — 0.0054 apart, well inside the
±0.027 band.
*The four contrasts at α = 0.0125, bias subtracted before any dose is declared:* 80→120 **+0.0694 − 0.0045 = +0.0649**
(3/3 seeds, t = +2.12); 120→160 **+0.0390 − 0.0028 = +0.0362** (2/3, t = +1.03); the **total** 80→160
**+0.1084 − 0.0073 = +0.1011** (**3/3**, **t = +5.36**); anneal horizon at 160 **−0.0930** (**0/3**, t = −2.17). All four
p < 0.0001. #1 + #2 = #3 exactly, as it must. **#1 and #2 both significant and positive ⇒ STILL CLIMBING** — inside the
anticipated rows, and Build 26 extends again.
*The bias correction was applied and did not bind* — every dose clears its bias by an order of magnitude. Booked as
applied-and-immaterial rather than quietly dropped, because the commitment predated the sign.
*The selection-free read agrees in sign on all four* (+0.0411 / +0.0694 / +0.1106 / −0.0907), so **there is no sign
disagreement to book**; #3 is near-identical across reads (+0.1084 vs +0.1106), the strongest evidence the total effect is
not a selection artifact. **But #2 is pooled-significant and NOT seed-robust on the seed-best read** — per-seed diffs
+0.003 / −0.001 / +0.115, carried almost entirely by seed 2 — where the *terminal* read is the stronger one (t = +2.34,
3/3). That reversal of the usual direction carries into Build 26.
*#4 is a seed-robust regression, and NOT the row that was anticipated.* FROZEN-SCHEDULE ARTIFACT required the anneal
contrast to be significant and **positive**; it came back significant and **negative** (0/3 seeds, t = −4.54 on the
terminal read). The frozen schedule is not hiding the effect — **it is actively better**, and Build 24's non-seed-robust
+0.031 anneal half is settled in the opposite direction at the longer budget.
*The curve-side read, pre-registered as able to contradict the gate, does not:* `v25b`'s `best_iter` is **132/125/159**,
0/3 seeds at or below 120. *Attributable:* the trust region bound 1/360 (`v25a`), **0/480** (`v25b`), 19/480 (`v25c`),
`approx_kl_max` all under the 0.09 bar; no collapse (`vs_iter0` 0.99–1.00, final entropy 0.52–0.60).

### Build 26 — PRE-REGISTERED (written 2026-08-06, before running): EXTEND THE DOSE AGAIN, HORIZON PINNED AT 80

Two fresh arms — `v26a` (`iters` **240**) and `v26b` (**320**), both at `anneal_iters` **80**, `target_kl` 0.06, from
`offrl_gen9ou_wide_s0`, seeds 0/1/2 — scored against the reused `v25b` anchor in one gate. Three contrasts at
**α = 0.0167**: #1 160→240, #2 240→320, #3 **total** 160→320, with #1 + #2 = #3 as a consistency check. The horizon
question is **closed by Build 25's #4** (significant and *negative*, 0/3 seeds), so no arm is spent on it. Full
pre-registration, including the outcome table, in `plan.md` §13.

*The selection-bias correction is re-simulated, and the simulation is now committed* — `scripts/selection_bias_sim.py`
plus tests, where Build 25's was ad hoc and unauditable. **σ_b is re-estimated and Build 25's 0.0428 was inflated**: it
came from arms whose anneal ran their whole length, so their "late window" booked live schedule trend as checkpoint
dispersion. Measured on the only true frozen plateaus (`v25a`/`v25b`, `iter > 80`, dof 348), the detrended dispersion sits
**at the binomial floor** — post-anneal the checkpoints are not resolvably different in strength, and all 6 seed-arms are
still drifting upward (+0.0010 to +0.0029/iter), which is STILL CLIMBING showing up in the curve shape. A point estimate
of zero must not be pre-registered as "no correction", so the correction is taken at the raw 95% upper bound,
**σ_b = 0.0328**: **#1 +0.0045, #2 +0.0025, #3 +0.0070**. The length-ratio worry was directionally right — pinning the
horizon at 80 makes #3 a **3×** draw ratio, not 2× — but the smaller σ_b nearly cancels it, landing #3 at essentially
Build 25's booked +0.0073. *The terminal read is promoted to co-primary*, since Build 25's #2 was pooled-significant but
not seed-robust on the seed-best read while the terminal read was stronger (t = +2.34, 3/3); a contrast is called only
where both reads agree in sign, and a disagreement is booked as the finding.

*Two operational prerequisites, both now in place.* `ppo_seed.slurm`'s `--time` was **20 h**, which the 320-iter arm blows:
Build 25's `sacct` (7188055) measured 11:03–11:30 for 160 iters ⇒ ~4.15–4.3 min/iter ⇒ `v26a` ~17 h, `v26b` ~23 h. Raised
to **36 h** (medium QoS allows 48 h); this also closes a drift where Build 25 ran under a 24 h command-line override the
script never recorded. `ppo_continue_gate.py` has **no resume**, so a walltime kill loses a whole arm. And `v26b` must be
submitted with `LATEGAME_SHOWDOWN_PORT_BASE=8200`: `_job_common.sh` computes `8100 + TASK_ID`, so two concurrent
`--array=0-2` arrays claim the same ports and colliding tasks **silently share one server** instead of failing.
6 tasks ÷ 4 concurrent ⇒ ~**40 h wall-clock**; ~29 GB of checkpoints against **78 GB** free.

**Attempt 1 lost 4 of 6 arms, and the post-mortem changed a standing constraint.** Two causes.
*Disk:* the 200 GB scratch quota is **shared**, not per-project — a neighbouring project grew
86 → 125 GB mid-run and the quota hit 195/200, so `torch.save` failed outright on one arm and
slowed two others from 4.0 to 15–18 min/iter until they timed out. The pre-run check compared
Build 26's own footprint against free space *at submit time*, which is not a claim that survives a
shared quota over 40 h. *Memory — the real finding:* **arm length is bounded by memory, not
walltime.** RSS grows ~**0.23 GiB/iter** (40.7 GiB at 160, 59.3 at 240, projecting ~78 at 320),
while `medium` sets `MaxTRES=mem=64G` **per job** — a larger `--mem` is rejected at submit, not
queued. `high` has 128 GB but only 24 h against a ~21.2 h run. **No QoS envelope fits a 320-iter
arm in one process**, so raising walltime could never have fixed it; `v26b` died OOM at exactly
64 GB, 304/320 in.

So `run_ppo` gains an **opt-in `--resume`**: a fresh process resets RSS, which is what makes an arm
past ~260 iterations runnable at all. The state file carries Adam's moments and both RNG streams
(dropping either restarts the optimizer cold and re-draws the league from the top of the stream),
lives beside rather than inside `iter_NN.pt`, and defaults **off** so Builds ≤25 reproduce
unchanged. It refuses to run without `--anneal-iters` pinned, since `anneal_horizon` otherwise
falls back to `iters` and would anneal over a different span per chunk. `v26b` now runs 160 → resume
→ 320. **Caveat to report with the result:** `v26a` ran uninterrupted and `v26b` is chunked, so #2
(`v26a` → `v26b`) is the contrast carrying that difference.

### OU pivot Build 26 — KNEE BETWEEN 240 AND 320: the update-count axis is saturating

Both gates ran clean in one submission (3:17:34): primary N=3000 (27,000 battles) and the
selection-free terminal read at N=1800 (16,200). Pooled vs the heuristic, n=9000/arm: `v25b`
**0.6691**, `v26a` **0.7250**, `v26b` **0.7420** [0.7329, 0.7509]; terminal read 0.6883 / 0.7354 /
**0.7513**. **0.7420 (0.7513 terminal) is the highest pooled rate in the project's history**,
against Build 25's 0.6534 / 0.6807. The anchor re-calibrates 0.0157 from Build 25, inside the
±0.027 cross-run band.

*The three contrasts at α = 0.0167, bias subtracted before any dose is declared:*

| # | contrast | primary (adj) | terminal (adj) | call |
|---|---|---|---|---|
| 1 | `v25b` → `v26a` | **+0.0514** (z +8.16, 3/3) | **+0.0425** (z +5.40, 3/3) | **WIN, both reads** |
| 2 | `v26a` → `v26b` | +0.0145 (p 0.0099, 2/3) | +0.0134 (p 0.0581, 3/3) | **not called** |
| 3 | `v25b` → `v26b` | **+0.0659** (z +10.73, 3/3) | **+0.0560** (z +7.29, 3/3) | **WIN, both reads** |

#1 + #2 = #3 exactly on both reads. **#1 significant and positive with #2 not called ⇒ KNEE BETWEEN
240 AND 320**, the row the pre-registration named as the anticipated modal outcome.

*The two reads agree in sign and in size; they differ only in power, and that is not a finding.*
#2's effect is +0.0145 and +0.0134 — the same number twice. The primary read carries 9000/arm and
the terminal 5400/arm, so an identical effect gives z +2.58 there and +1.90 here. There is **no sign
disagreement to book**. What there is: #2's true effect (~+0.014) sits at the resolution limit of
this design against a ~0.025 MDE, so calling it either way would be reading the sample size rather
than the model. The seed-robustness pattern inverts exactly as Build 25 warned — primary
pooled-significant but 2/3 seeds, terminal pooled-null but 3/3.

**The real content is the curve, now four points.** Marginal return per *update*, ×10⁻³: **1.62**
(80→120) → **0.91** (120→160) → **0.64** (160→240) → **0.18** (240→320) — a ~9× decay, monotone at
every step. Not "320 beats 240" but *the axis is saturating, and cost per point is now the binding
question rather than the effect.*

*Attributable, no collapse:* the trust region **never bound** — 0/240 on every `v26a` arm and 0/320
on every `v26b` arm, `approx_kl_max` 0.052–0.072 under the 0.09 bar (against Build 22's
verdict-costing 59/150), final entropy 0.418–0.487, `vs_iter0` **1.00 on all six**.
*The caveat landed where it was pre-registered to:* `v26a` ran uninterrupted and `v26b` chunked, and
#2 — the contrast carrying that difference — is precisely the ambiguous one.
*The resume mechanism is validated:* chunk 2 peaked at **41.6–44.0 GiB** for a 320-iteration arm,
against the **67.1 GiB** that OOM-killed the unchunked attempt at iteration 304 and the 64 GB cap.

### M5 / G1 — live play: the half of the project G2 is actually defined in

Everything through Build 26 plays a *fixed* baseline on a local server, where win rate is the
sufficient statistic. G2 is stated in GXE/Glicko-1, which are ladder metrics. `lategame/live/`
(`lategame live`) is that client — three modes (`challenge` / `accept` / `ladder`), a session
supervisor, and Glicko-1/GXE telemetry — verified end-to-end against a real local server: two
players, real websocket, real battle, the finalize sweep, GXE computed, results file written.
**Three findings, each a case where the obvious implementation is silently wrong.**

*`ShowdownException` never reaches your `await`.* poke-env raises it inside a detached task nobody
retrieves, so a failed login **hangs** rather than raising and a `try/except` there would be dead
code. Detection is a log handler plus a stall watchdog sampling `(finished, Σ turns)` — a slow
battle still advances turns, a dead socket does not. `listen()` likewise swallows a closed socket
and the library has no reconnect, so recovery means building a **new** player and merging by tag.

*`battle.rating` is pre-battle **Elo**, not post-battle Glicko* — and it is `None` at callback time
regardless, because the rating line arrives after `|win|`. Hence the mandatory finalize sweep
(without it the rating columns are uniformly empty and look like an unrated session), and Elo kept
strictly out of the Glicko math.

*`rate_win_rate` cannot represent a tie* (it reconstructs wins as `round(win_rate·n)`), so the
summary builds its Glicko results list directly; a test pins that the two agree exactly on tie-free
records. **The same bug sits one layer down in poke-env's `cross_evaluate`**, which reports
`n_won / n_finished` — booked rather than fixed, since `eval/arena.py` was on the frozen path of the
in-flight Build 26 jobs.

*The ladder gate holds at all three levels:* no ack fails, a wrong phrase is rejected by argparse
`choices`, and the right phrase without `LATEGAME_LIVE_ALLOW_LADDER=1` still refuses with the policy
note. **G2 was still uncomputed after this build** — a session's GXE pins every opponent at the
reference by default, under which it is a reparameterisation of the score rate. Only a varied field
makes it a measurement.

### R-LADDER — the agent-only eval ladder: G2's headline metric, without touching the human ladder

`lategame/eval/ladder.py` (`lategame eval-ladder`) plays a round-robin over a heterogeneous field on
the local server and fits every rating **jointly**. New file rather than an extension of
`eval/rating.py` or `eval/arena.py`, both of which were on the frozen path of the running Build 26
jobs.

*A single Glicko period over a round-robin is degenerate*, which is the whole difficulty: everyone
starts at (1500, 350), so every opponent **is** the reference and each rating collapses back to a
monotone function of its own score rate. The fit must iterate, and at its fixed point every agent
satisfies the **Bradley-Terry score equation** — so this is the BT/Elo maximum-likelihood fit,
reported on the Glicko scale. Two agents are checked against the closed form to 1e-6.
*The undamped iteration oscillates rather than converging*, worst at two agents, because a
difference gets corrected from both ends at once; damping by exactly **0.5** maps the iteration
matrix's eigenvalues from `[0, 2]` into `[0, 1]`. *The fit needs a gauge fix* — a round-robin
identifies differences only — and `heuristic` is pinned at 1500 because it is the baseline every
build is already reported against. *RD is computed once at convergence*, never per sweep, or an
agent's deviation would depend on how many iterations the solver happened to take.

**Result** (`gen9ou`, 28 pairs × 300 battles = 8,400, `loop_penalty` 4, `results/eval_ladder_gen9ou.json`):

| agent | W–L | score | Glicko | RD | GXE |
|---|---|---|---|---|---|
| `offrl@ppo_v25a_s0/iter_120` | 1597–503 | 0.760 | 1715.5 | 15.0 | 0.6962 |
| `offrl@ppo_v25b_s0/iter_160` | 1570–530 | 0.748 | **1696.9** | 14.9 | **0.6809** |
| `offrl@ppo_v23b_s0/iter_80` | 1532–568 | 0.730 | 1671.1 | 14.7 | 0.6590 |
| `simpleheuristics` | 1394–706 | 0.664 | 1579.2 | 14.6 | 0.5756 |
| `heuristic` (anchor) | 1276–824 | 0.608 | 1500.0 | 14.8 | 0.5000 |
| `maxbasepower` | 636–1464 | 0.303 | 941.4 | 21.3 | 0.1044 |
| `random` | 358–1742 | 0.171 | 575.1 | 26.2 | 0.0277 |
| `offrl@offrl_gen9ou_wide_s0` | 37–2063 | 0.018 | −23.5 | 43.9 | 0.0029 |

The ordering reproduces the whole win-rate history, and `simpleheuristics` vs `heuristic` comes back
**0.623** against Lever 15's published **0.633** [0.577, 0.686] — an independent reproduction. The
PPO warm-start init lands at the bottom, consistent with OU Builds 2–4 finding it functionally dead
on OU (`random` beats it).

**What it does not settle.** The top three are *not* separated. Two earlier `n=150` runs moved the
`v25a`↔`v25b` pairing **0.460 → 0.587** (~3.1 binomial SE) while each agent's own GXE moved under
0.004: the battles are **not independent Bernoulli trials**, because a 12-team pool plus archetype
counters clusters them by team matchup, so the effective sample is under `n` and **RD is a lower
bound, not an interval**. Read the standings as separating bands, not ordering neighbours — which
also matches the authoritative gate, where Build 25's 120→160 contrast was pooled-significant but not
seed-robust. The ladder is single-seed and so cannot speak to build-vs-build at all.
*One coincidence, booked as one:* `v25b`'s GXE 0.6809 sits 2e-4 from its published 0.6807 win rate,
but these are different quantities (GXE discounts for the reference's RD 350; the direct head-to-head
here was 0.720). Not the ladder reproducing the gate.

### G2 refreshed on Build 26, and the first measurement of what the standings cannot order

The 2026-08-10 ladder excluded `v26a`/`v26b` as in-flight arms. Re-run now that they are not
(`gen9ou`, 9 agents, 36 pairs × 300 = **10,800** battles, `results/eval_ladder_gen9ou_v26.json`),
with **300 cluster-bootstrap resamples over the 78 team matchups**:

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

**G2's headline metric is now `v26b`: Glicko 1776.3, GXE 0.7434** (from `v25b`'s 1696.9 / 0.6809).

**RD was always a lower bound; now there is an interval that is not.** RD comes from binomial noise
alone, but on a teambuilt format battles cluster by team matchup, so the effective sample is well
under `n`. The bootstrap resamples **matchups**, not battles — resampling battles would just
reproduce the binomial interval RD already gives. The result: **every band boundary separates**
(learned > `simpleheuristics` > `heuristic` > `maxbasepower` > `random`) and **not one adjacent
learned pair does**. "Read the standings as separating bands, not as ordering neighbours" has been
in the results file since it was written — this is the first run that measures it.

*One coincidence, booked as one:* `v26a` and `v26b` post identical totals (1671–729) and therefore
identical Glicko/GXE, while being clearly different agents — all seven per-opponent records differ
and they went 165–135 head to head. `v26b`'s +30 across the rest of the field cancels its −30 there.

### G3 — BOOKED: the continual-improvement curve was already on the record

§12's continual-improvement row asks for "metric vs self-play volume → monotonic improvement".
Builds 24–26 are that curve; nobody had written it up as one. Five doses at a pinned schedule,
3 seeds, pooled n=9000/arm vs the heuristic, chained through the shared `v25b` anchor (which
re-calibrated +0.0157 between runs, inside the ±0.027 cross-run band):

| updates | 80 | 120 | 160 | 240 | 320 |
|---|---|---|---|---|---|
| seed-best read | 0.5607 | 0.6301 | 0.6691 | 0.7250 | 0.7420 |
| terminal (selection-free) | 0.5778 | 0.6189 | 0.6883 | 0.7354 | 0.7513 |

**Monotone at every step on both reads.** G3 is met — *and* the axis that met it is saturating:
the marginal return per update decays 1.62 → 0.91 → 0.64 → 0.18 (×10⁻³), ~9×, monotone.

The curve comes from `seed_strength_gate.py`, **not** from the ladder, and that is measured rather
than stylistic: the ladder's Glicko ordering is not even monotone in update count (120 sits above
160), and no adjacent learned pair separates — a single-seed field cannot speak to build-vs-build.
It is not evidence against the gate; it just cannot carry this curve.

### G4 / VGC ceiling probe — INSUFFICIENT, and the instrument is the reason

Following the Lever-15 idiom (measure a format's ceiling before building its pipeline), M1 was run
on `gen9vgc2025regi` — n=300 per matchup, a validator-checked 10-team pool,
`results/format_ceiling_gate_vgc.json`. Two prerequisites had to be built first: the
`VGC_FORMAT` constant named `gen9vgc2024regh`, **which the vendored simulator does not have**
(corrected to `gen9vgc2025regi` and pinned by test), and `HeuristicAgent` gained a doubles path,
because poke-env supplies only *one* competent doubles bot and the probe needs two near the top.

| vs `heuristic` | gen9-RB (FORMAT_BOUND) | gen9ou (headroom) | **gen9vgc2025regi** |
|---|---|---|---|
| mirror (sanity) | 0.513 | 0.493 | 0.523 |
| `simpleheuristics` | **0.523** [0.467, 0.579] | **0.643** | **0.527** [0.470, 0.582] |
| `maxbasepower` | 0.107 | 0.060 | 0.300 |
| `random` | 0.007 | 0.030 | 0.017 |

> **The VGC column is a pre-loop-fix read.** It was measured in `4647bb0`, three days before B6f
> Stage A found the doubles collection loop, on the same instrument whose output got the BC/AWR
> figures withdrawn. Every one of these four cells was re-measured afterwards at the same n = 300
> — mirror 0.503, `simpleheuristics` 0.467, `maxbasepower` 0.310, `random` 0.023 — in
> [the corrected ladder below](#build-28--b6f--the-vgc-shard-was-94-one-bug-and-it-was-ours). The
> verdict below is unchanged by that (if anything the corrected `simpleheuristics` sits *lower*,
> further from the 0.58 headroom threshold), but these specific numbers are superseded and the
> corrected ones are what to quote.

**On the discriminating quantity VGC reproduces the gen9-RB signature almost exactly** — the
strongest competent bot at 0.527 with a CI spanning 0.50, indistinguishable even from the
heuristic's own mirror. **But the verdict is INSUFFICIENT, not FORMAT_BOUND.** Two readings fit:
either VGC's ceiling really is ~parity, or both bots are equally blind to doubles-specific skill
(targeting, Protect timing, speed control, bring-6-pick-4) and the instrument cannot see a gap that
lives above them both. `maxbasepower` at **0.300** — far closer to competent play than its 0.107/
0.060 on the singles formats — is consistent with either.

Nothing breaks the tie the way it did on RB, where **M2** (near-optimal search reaching 0.500) turned
a suggestive band into a verdict: there is no doubles forward model and no scraped VGC replays, so
neither M2 nor M3 exists here. **G4 is therefore neither greenlit nor descoped** — the finding is
that on doubles the cheap probe is *not decisive*, because every agent cheap enough to run before
building the pipeline is a singles policy applied per slot.

### Build 27 / Gate B — search does not compound on OU either, and the h2h says it is harmful

L11–L14 retired test-time search at parity, but all of it was measured on gen9-RB — which Lever 15
then proved **FORMAT_BOUND**, where near-optimal search reaches 0.500 because nothing beats the
heuristic. That retirement was measured somewhere it could not have come out otherwise. Re-run on
`gen9ou`, where headroom is proven and the base is 0.77, with the strongest arm available: depth-2
expectimax, the **exact** white-box opponent model (the eval opponent *is* `HeuristicAgent`), same
checkpoint both sides, n=2500/arm pooled from a 10-shard array. `results/rpredict_search_ou.json`.

| | rate |
|---|---|
| base (greedy `v26b`) vs `heuristic` | 0.7688 |
| depth-2 search vs `heuristic` | 0.7724 |
| **contrast** | **+0.0036** (z 0.303, p 0.76) — positive in **6/10** shards |
| search vs its own base, head-to-head | **0.3932** — below 0.500 in **10/10** shards |

**NULL on the contrast — and the head-to-head is not a null.** Search loses to the policy it
descends from by ~10.7 SE (p ≈ 1e-26). Both can be true because against an opponent that loses 77%
there is slack for a slightly worse policy, while an opponent strong enough to punish it converts
the same deviations into losses. **Measuring search only against the fixed baseline would have
reported "no effect" and missed that the effect is negative.**

This also closes the escape hatch L11/L12 named — "the opponent model was too weak". Here it is
exact, the forward model is validated at **0 mismatches on this format** (Gate A′), and search
still does not help. §16 Q2 is answered (ship policy-only) and **M7 closes**: six independent
mechanisms now.

### G4 MET — three formats play end to end through one core

The doubles pipeline (G4/M6) is in: a per-slot action codec, a doubles encoder, and a learned
doubles agent. Verified against a live local server, 6/6 battles completed on each:

| format | agent | vs `random` | finished |
|---|---|---|---|
| `gen9randombattle` (singles random) | `heuristic` | 1.000 | 6/6 |
| `gen9ou` (singles teambuilt) | `offrl` @ `v26b` | 1.000 | 6/6 |
| `gen9vgc2025regi` (**doubles**) | `doubles` @ init | 0.667 | 6/6 |

**"Without rewriting the core" is the substantive half of that goal, and it held.** The model
factory needed *no change*: `build_model` already read `input_dim`/`n_actions` from checkpoint
metadata, so the doubles network is the same architecture at a different width. The only genuine
edit to shared code was making `EntityTransformer` take its token layout as a parameter rather than
importing the singles constant — and it resolves that layout from `input_dim`, so every existing
checkpoint still builds exactly the model it always did.

**The action space is factored: 2 × 107, not a joint 11,449.** What factoring cannot express is a
constraint coupling the slots, and exactly one matters — both slots switching to the same benched
Pokemon. Showdown answers that with a *default move rather than an error*, so it is a silently lost
turn; it is resolved explicitly after sampling rather than by the mask.

Singles is frozen and pinned by test (`OBS_DIM` 761, `OBS_VERSION` `v5-`, 26 actions). Doubles is
separately versioned (`d1-`, 888-d) on both fields, so a cross-format shard is rejected on either.

*Not claimed:* the doubles checkpoint is randomly initialised. G4's exit is "playable end to end";
VGC strength is a separate question and the 0.667 is 6 battles of an untrained policy.

### Build 28 / B6f — the VGC shard was 94% one bug, and it was ours

The VGC *strength* campaign (BC -> offline RL -> PPO) is the work G4's exit criterion deliberately
did not require. Its first act was to measure the shard everything downstream warm-starts from,
which is a thing that should have happened two builds earlier.

| | `data/vgc_rl.npz` | `data/gen9ou_v7_rl.npz` |
|---|---|---|
| turns / episode, median | 7 | 19 |
| turns / episode, max | **12,795** | 205 |
| top-decile turn share | **0.922** | 0.239 |
| episode-length Gini | **0.901** | 0.304 |
| unique observations / turns | **0.0033** | 1.000 |

**The longest episode carried 12,795 recorded turns over seven unique observation vectors** — the
same states re-requested thousands of times. 94.2% of the shard came from 100 of its 899 episodes,
and 96.6% of all rewards were exactly zero.

**Which agent loops was measured, not assumed.** Four battles per pair, counting `choose_move`
calls against the `battle.turn` actually reached:

| pair | calls / battle | turn reached |
|---|---|---|
| `random` vs `random` | 19 | 17 |
| `simpleheuristics` vs `simpleheuristics` | 9 | 8 |
| `maxbasepower` vs `maxbasepower` | 8 | 6 |
| `random` vs **`heuristic`** | **4,001** | spanned 1..7 |
| **`heuristic`** vs **`heuristic`** | **1,153** | spanned 1..4 |

poke-env's own baselines never loop. Ours did, and `heuristic` is in every collection pool.

**The bug was one line.** A doubles slot with no decision emitted `DefaultBattleOrder()` — poke-env's
*whole-order* sentinel, message `/choose default`. `DoubleBattleOrder.message` joins by string
surgery, so half a default serialised to `/choose default, move woodhammer 1`, which is not a legal
Showdown command. The server rejected it, poke-env re-requested the identical state, and the agent
answered identically forever. The per-slot "do nothing" is `PassBattleOrder` (`/choose pass`).

**It is the write side of a bug the previous build fixed on the read side.**
`normalize_half_default` exists because poke-env *labels* a half default with `-2`, its whole-order
sentinel, where the per-slot layout says "this slot does nothing" is action 0. The labelling was
corrected; the emission was never looked at.

After the fix, `heuristic` vs `heuristic` goes 1,153 -> 9 calls and every pair sits within ~2 calls
of `battle.turn`. Collection got ~90x faster as a side effect — the loop was the cost.

**What this corrects.** VGC doubles is a normal, decision-*dense* format: 0.923 decision density and
14.0 legal actions per slot on a decision turn, against OU's 0.9998 and 7.67. The "98.4% of recorded
turns carry no decision" note in the collection code measured the loop, not the format. And the
critic's target was compressed ~4x by zero-reward loop frames (return std 0.80 against the corrected
shard's 3.14), so the previously reported doubles value-MAE was measured against a near-constant
target. The BC/AWR figures from those builds are withdrawn, and the corrected ladder is below. They
are withdrawn rather than superseded on purpose: they were `n = 100` screening reads recorded in
commit bodies, and the corrected ladder scores at `n = 300`, so no before/after delta may be read
across the pair — the difference in `n` is larger than the effect either would have to resolve.

**The corrected ladder, n = 300 per cell** (`results/format_ceiling_gate_vgc_v2.json`; SE ≈ 0.029
against the withdrawn reads' ≈ 0.05). Every arm plays the fixed `heuristic`:

| arm | vs `heuristic` | ci95 | withdrawn `n = 100` read |
|---|---|---|---|
| mirror (sanity) | 0.503 | [0.447, 0.560] | — |
| `simpleheuristics` | 0.467 | [0.411, 0.523] | — |
| `maxbasepower` | 0.310 | [0.260, 0.364] | — |
| `random` | 0.023 | [0.011, 0.047] | — |
| **BC** — `checkpoints/doubles_bc_vgc_v2.pt` | **0.453** | [0.398, 0.510] | 0.390 |
| **AWR** — `checkpoints/doubles_offrl_vgc_v2.pt` | **0.467** | [0.411, 0.523] | 0.350 |

Both learned arms read *stronger* on corrected data than they did on loop data. The last column is
printed as provenance, not as a delta: it is the withdrawn number, and the paragraph above is why
subtracting it is not allowed. The mirror at 0.503 says the harness is sound.

**And the pre-registered stop rule fires, where B6d said it did not.** The rule, written before the
campaign: *"if the BC agent lands near `simpleheuristics` while `simpleheuristics` is still at parity
with the heuristic, that is the FORMAT_BOUND signature arriving early."* Both clauses now hold — BC
0.453 against `simpleheuristics` 0.467 is 0.5 SE, indistinguishable, and `simpleheuristics`'s CI
[0.411, 0.523] spans 0.50. B6d adjudicated the same rule "does not fire" **by eye in a commit
message**, on loop data, where BC read 0.390 and so looked safely below both competent bots. Two
process failures, not one: the rule was evaluated against corrupted numbers, and it was evaluated
informally rather than by the instrument it names (`format_ceiling_gate --bc-checkpoint`, which had
never been run). This is the ceiling that makes B6f's C2 null below uninterpretable rather than
merely negative.

*Read honestly:* a loop guard and a per-battle turn cap were built first, on the theory that this
was a property of forced-replacement states. They are not what closed it — capped and uncapped arms
are the same shard to within noise. Both are kept as backstops at exact-identity defaults, because
the failure mode is silent and a second instance would otherwise be found the same way: by noticing
94% of a shard inside 11% of its episodes, after training on it.

### Build 28 / B6f — factored PPO on VGC: mechanism confirmed, strength ceiling-bound

Three seeds, 80 iterations each, from the corrected AWR warm start. Two pre-registered contrasts
at alpha = 0.025, both scored at n = 3000 per checkpoint.

**C1 (primary) — PPO's best checkpoint vs its own warm start.** Ceiling-independent: both sides
are learned policies on one format.

| | rate | ci95 |
|---|---|---|
| seed 0 | 0.522 | [0.504, 0.540] |
| seed 1 | 0.523 | [0.505, 0.541] |
| seed 2 | 0.544 | [0.526, 0.562] |
| **pooled (4768/9000)** | **0.530** | **[0.519, 0.540]** |

Pooled CI excludes 0.50, and so does every seed individually. **WIN** — the factored policy
gradient moves a doubles policy off the AWR ceiling. Modest (+3.0 points) and unambiguous; 9,000
battles is what makes +0.030 readable.

**C2 (secondary) — vs the fixed `heuristic`.** AWR 0.454 [0.437, 0.472], PPO 0.466 [0.456, 0.477],
diff +0.012 (z 1.14, p 0.25). **NULL** — and an uninterpretable one: the pre-registered stop rule
fired before this ran, because `simpleheuristics` sits at 0.467 with a CI spanning 0.50. There is
no established headroom above it for a strength gain to appear in.

**The two contrasts disagree, and that is the result.** The same checkpoints beat their own warm
start by 5.7 SE and are indistinguishable from it against the heuristic. A fixed baseline can only
resolve a difference it is strong enough to punish. Build 27's Gate B taught this with the opposite
sign — there, measuring only against the baseline hid a *negative* effect.

**Selection bias, measured, and large.** The in-loop curve's best-iteration `vs_heuristic` reads
0.590 +- 0.014 at `eval_n = 100`. Re-scored at n = 3000 the same checkpoints read **0.466** — a
**0.124** gap, pure argmax-over-80-noisy-estimates inflation. Reading the curve's headline instead
of the re-scored number would have overstated the build by 12 points.

**Attributability clean on all four clauses**, including three that are new and doubles-specific:
`lp_drift_max` 6.7e-06 to 8.6e-06 (acting and training were the same function, so the importance
ratios are real), `invalid_frac_max` <= 0.031 against a 0.05 ceiling, `dec_frac_min` >= 0.878.
Without the decision-row denominator the same run would have reported a KL an order of magnitude
smaller and certified a trust region that never bound.

## Artifacts & reproducibility

**A clone of this repo contains no weights and no training shards.** `checkpoints/` and `data/` are
gitignored, so every number above was produced from files that exist only on the machine that ran
them. What *is* committed, and is the durable record, is the evidence rather than the artifacts: 221
`results/*.json` gate summaries, each arm's per-iteration `curve.json`, the validator-checked packed
team pools (`lategame/teambuilding/data/`), the encoder vocab and the gen9ou usage prior
(`lategame/features/data/`), and the pinned simulator rev.

**Which record backs which headline.** These name result files rather than checkpoint paths, because
a gate can be re-pinned and a result file cannot:

| headline | record |
|---|---|
| gen9ou **0.7513** selection-free terminal | `results/ppo_ou_gate_v26b_terminal.json`, `results/seed_strength_gate_v26_terminal.json` |
| Glicko **1776.3** / GXE **0.7434** | `results/eval_ladder_gen9ou_v26.json` |
| VGC B6f C1 **0.530** | `results/ppo_vgc_gate_b6f{,_s0,_s1,_s2}.json`, `results/seed_strength_gate_b6f_c1.json`, `results/awr_vgc_arm_b6f.json` |
| VGC corrected ladder **BC 0.453** / **AWR 0.467** | `results/format_ceiling_gate_vgc_v2.json` |

Every checkpoint those records name is present on the machine that produced them, and none of them
ship. `python scripts/check_artifacts.py` re-derives that statement rather than trusting this table.

**58 of the 121 checkpoint paths named across `results/**.json` no longer exist**, cited by 52 of the
221 result files. 27 are top-level warm starts (`bc_gen9ou_v*.pt`, `offrl_scale_*.pt`); the rest are
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

## Setup

```bash
# 1. Python env (Python 3.11, isolated) — environment.yml runs `pip install -e ".[dev,ml]"`,
#    torch included: train / grad_noise_diag / the bc agent all die at the first `import torch`
conda env create -f environment.yml
conda activate lategame

# 2. On a CPU-only box, install the CPU torch wheel FIRST to avoid pulling ~2.5 GB of CUDA:
#    pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

# 3. Local Showdown server + vendored simulator (fetches smogon/pokemon-showdown into
#    third_party/ and builds dist/ — dist/ is also used by replay re-simulation)
bash scripts/setup_server.sh
```

**The vendored simulator is pinned** (`SHOWDOWN_REV` in `scripts/setup_server.sh`), and bumping the
pin is a deliberate act. gen9randombattle sets change upstream, so the same PRNG seed rolls
different teams under a different rev and every recorded inputlog goes illegal partway through —
which is how the R-PREDICT fidelity and resim end-to-end tests silently broke against a simulator
that had moved on. Bump the pin and the inputlog fixture in `tests/conftest.py` must be regenerated
in the same commit:

```bash
node scripts/gen_inputlog_fixture.js third_party/pokemon-showdown
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

**Score a PPO checkpoint through `offrl`, never through `ppo`.** `PPORecordingAgent` is the
training rollout agent and forces `sample=True`; the same `v25b` terminal checkpoint measures
**0.675 sampled vs 0.767 greedy** against the heuristic. Every published number here reads these
checkpoints greedily through `offrl`, which is the deployed policy. `eval-ladder` refuses
`ppo@<checkpoint>` outright for this reason.

### The eval ladder (G2's metric) and live play

```bash
# R-EVAL: a VARIED opponent field, which is the only condition under which GXE/Glicko-1 carry
# information a win rate did not. Round-robin + one joint Bradley-Terry fit on the Glicko scale,
# with `heuristic` pinned at 1500 to fix the gauge. NOT a replacement for seed_strength_gate.py,
# and NOT comparable to a Showdown GXE (which is measured against humans).
python -m lategame.cli eval-ladder --format gen9ou \
    --team-pool lategame/teambuilding/data/teams_gen9ou.packed \
    --n 150 --out results/eval_ladder_gen9ou.json

# M5 deploy / G1 — live server. Default mode is `accept` (opt-in opponents only). Credentials come
# from $LATEGAME_PS_USERNAME / $LATEGAME_PS_PASSWORD and never reach --out.
python -m lategame.cli live --agent ppo --mode accept --n 5 --format gen9ou
python -m lategame.cli live --mode challenge --opponent <user> --n 3   # opt-in opponent
python -m lategame.cli live --server ws://localhost:8000/showdown/websocket --allow-guest --n 1

# `--mode ladder` is the PUBLIC RANKED ladder, which plan.md NG3 puts out of scope. There is no
# unranked ladder: `/search` on the public server IS the rated one. It needs BOTH opt-in channels,
# so neither a stale export nor a recalled command can start ranked play on its own:
#   export LATEGAME_LIVE_ALLOW_LADDER=1
#   python -m lategame.cli live --mode ladder --ladder-ack i-have-read-plan-md-section-15 \
#       --use-live-ratings --n 50
# --use-live-ratings rates each opponent at its OBSERVED rating instead of pinning the field;
# without it the session's GXE is a reparameterisation of its own score rate. Read plan.md 15 first.
```

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

# Gen 9 OU PPO builds (19-23) — the build-vs-build toolchain. Run in this order.
# On the cluster the seeds are an array job (scripts/cluster/ppo_seed.slurm); each writes its own _s{N}.json.
python scripts/ppo_telemetry.py --log 0 logs/ppo/v21/ppo_v21_s0.log --kl-bar 0.045 --out results/ppo_ou_telemetry_v21.json
#   ^ the TRUST-REGION CERTIFICATE. Run FIRST: the run log is gitignored, this JSON is the durable copy.
#     Job logs live under logs/ — logs/ppo/<build>/ (run stdout), logs/slurm/ (sbatch --output),
#     logs/showdown/<bucket>/ (per-job server). ppo_seed.slurm writes its own; only logs/slurm/.gitkeep
#     and logs/MANIFEST.tsv are tracked. sbatch does NOT create its --output dir, hence the .gitkeep.
#     A NULL is only attributable to the lever if the trust region did not bind (Build 22: it bound 59/150,
#     which cost that build its verdict). --kl-bar MUST be 1.5 * the run's --target-kl, or the certificate
#     names a bar the optimizer never enforced. ppo_seed.slurm derives both from TARGET_KL so they cannot drift.
python scripts/merge_gate_seeds.py --seed-json results/ppo_ou_gate_v21_s{0,1,2}.json \
    --ladder-source ... --note ... --out results/ppo_ou_gate_v21.json
#   ^ pools the per-seed runs. Drop a seed here and the z-test below silently loses its power.
python scripts/seed_strength_gate.py --build v20 ... --build v21 ... --out results/seed_strength_gate_v21.json
#   ^ THE AUTHORITATIVE verdict: every seed's best ckpt, n=300 each, pooled to 900/arm, two-proportion z.
#     Resolves ~+0.07 at z~3. The training curve does NOT decide WIN vs NULL. Absolute rates move
#     between runs (winner's curse); only the within-run DIFFERENCE is trustworthy.
#     ALWAYS pass BOTH arms to ONE invocation (cluster: scripts/cluster/strength_gate.slurm, which
#     preflights that each arm's best checkpoints are staged and NAMES the missing ones). Build 22
#     measured v20's IDENTICAL checkpoints at 0.472 and 0.499 in two runs -- 0.027 apart, at the 0.023
#     SE -- so a cross-run difference under ~0.03 is not a result.
python scripts/grad_noise_diag.py --policy <best> --init <warm-start> --league-dir <run> \
    --games-per-opp 48 --rollouts 6 --splits 5 --out results/grad_noise_diag_v21.json
#   ^ the EXPLAINER: reads |G|^2 (probes[*].arms.same_mix.policy.noise_scale.g_norm_sq).
#     --games-per-opp MUST match the run under test; --splits defaults to 20 (v20 used 5).
#     WARNING: its NOISE_LIMITED verdict answers Build 20's question, not yours. B_simple is a RATIO
#     (tr(Sigma)/|G|^2) — when |G|^2 -> 0 it explodes and "collect more samples" is exactly WRONG.
#     RETIRED for small effects (Build 22): a FIXED checkpoint swung 2.7x on the --seed alone, larger
#     than the 2.2x effect the gate was built to detect. Any gate reading a |G|^2 difference under ~3x
#     MUST run >=2 probe seeds (scripts/cluster/probe_replicate.slurm) and report both, or it is
#     reporting noise. The seed is NOT recorded in the output JSON -- provenance is the filename only.
#     The COSINE from the same runs did replicate (0.312 -> 0.317) and remains usable.

# Build 26's whole analysis as ONE submission (server + merge + both strength gates + bias table).
#   The preflight refuses a MISSING or SHORT arm before starting anything: v26b runs in two chunks
#   that write the same per-seed JSON, so between them the file exists and is 160 iters long.
#   N is pinned to Build 25's pre-registered 3000 / 1800 -- the gate's own default is 300, whose
#   MDE (0.079) is wider than any dose this build can produce, so it would book a NULL by design.
sbatch -p tron --qos=medium scripts/cluster/build26_analysis.slurm

# R-EVAL — the agent-only eval ladder. On the cluster, scripts/cluster/eval_ladder.slurm.
#   MUST set a port base clear of any in-flight PPO build: _job_common.sh computes 8100 + TASK_ID,
#   a non-array job takes TASK_ID=0 -> 8100, and colliding jobs SILENTLY SHARE one Showdown server.
#   The slurm wrapper defaults LATEGAME_SHOWDOWN_PORT_BASE=8300 for exactly this reason.
OUT=results/eval_ladder_gen9ou.json sbatch -p tron --qos=medium scripts/cluster/eval_ladder.slurm

# M6 — human replays: fetch, then reconstruct each player's POV either from the public
# spectator log (v1) or by re-simulating the inputlog for the private |request| (v2)
python -m lategame.cli fetch-replays  --min-rating 1200 --limit 200
python -m lategame.cli ingest-replays --out data/ingest_gen9rb_rl.npz   # v1 (public-log POV)
python -m lategame.cli resim-replays  --out data/resim_gen9rb_rl.npz    # v2 (needs node + dist/)
```

## Develop

```bash
pytest            # 714 tests, 0 skipped with the env active (node) + a local server up
                  #   LATEGAME_LIVE_TEST=1 also enables the opt-in live-client smoke
                  #   On a bare clone -- no server, no built dist/, no checkpoints/ -- 15 self-skip
                  #   and 699 pass. That is what CI runs; a 16th skip is a regression, not noise.
ruff check .
mypy lategame     # scoped to lategame/ on purpose: scripts/ carries 2 known pre-existing errors
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
| `lategame/search/` | R-PREDICT: forward model, determinization, expectimax, opponent model |
| `lategame/teambuilding/` | R-TEAM: validator-checked packed team pools for teambuilt formats |
| `lategame/eval/arena.py` | run N battles vs one fixed baseline, report win rates |
| `lategame/eval/rating.py` | R-EVAL: Glicko-1 + GXE (degenerate against a fixed baseline — read the docstring) |
| `lategame/eval/ladder.py` | agent-only eval ladder: round-robin + joint rating fit (the varied field G2 needs) |
| `lategame/live/` | M5 deploy / G1: live-server client, policy gate, session supervisor, telemetry |
| `lategame/cli.py` | all subcommands (eval / collect / train / data / live) |
| `scripts/` | local Showdown server + simulator setup/run, and every experiment gate |

## License & acknowledgements

This project is MIT-licensed — see [`LICENSE`](LICENSE). `CITATION.cff` carries the citation record.

- **[poke-env](https://github.com/hsahovic/poke-env)** (Haris Sahovic), MIT — the battle-client and
  baseline-player layer every agent here is built on. A pip dependency, declared in `pyproject.toml`.
- **[pokemon-showdown](https://github.com/smogon/pokemon-showdown)** (© 2011–2026 Guangcong Luo and
  other contributors), MIT — the simulator. It is **cloned, not vendored**: `scripts/setup_server.sh`
  fetches it into `third_party/` at the pinned rev `393d5c86`, and `.gitignore` keeps it out of this
  tree entirely. Nothing from it is redistributed here, and its license is its own — read it at
  `third_party/pokemon-showdown/LICENSE` after running the setup script.

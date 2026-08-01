"""OU Pivot Build 7 -- train-side switch-mass diagnostic: where did the loop come from?

Build 6 proved the v5 agents fail by *absorbing two-mon voluntary switch loops* (~0.9
policy mass on switches in loop states; the obs is memoryless so a 2-cycle is a fixed
point). This gate asks, offline and with zero training, whether that mass was LEARNED
from the data or INVENTED off-distribution. On the v5 shards it measures, per
conditioning, the human switch rate H (``action < 6``), each trained checkpoint's switch
mass P (masked-softmax prob mass on actions 0-5), the top-1-is-switch rate T, and the
uniform-over-legal baseline U (~0.45 on these shards -- the "no preference" anchor).

Hypotheses (pre-registered decision tree; per arm on its own training shard, primary
conditioning = ``loop_species``):

  H3 training amplification   P >= H + AMPLIFY_DELTA and P >= AMPLIFY_RATIO * H
                              -> objective/training-side. Localize: offrl P - bc P on
                              the SAME shard >= BC_VS_AWR_DELTA -> AWR side, else BC.
  H1 pivot-heavy human prior  H >= HUMAN_HEAVY and |P - H| <= MATCH_TOL
                              -> faithful imitation; memoryless composition loops.
                              Next: history/recency feature + retrain.
  H2 OOD artifact             |P - H| <= MATCH_TOL and H < HUMAN_HEAVY and
                              max(P, P_pair) < IN_DIST_HIGH
                              -> the live ~0.9 mass never appears in-distribution.
                              Next: drift-side work. Annotate ``argmax_amp`` if
                              T >= ARGMAX_AMP (deterministic play amplifies).
  MIXED otherwise             (margins reported; a finding, not a failure).

Controls (any failure -> INCONCLUSIVE, exit 1): C-harness (zero-logits masked softmax
must equal U exactly), C-random (random-init ET overall mass inside RANDOM_INIT_BAND),
C-invariant (every shard row's taken action is mask-legal), C-floors (primary-row
counts). Specificity control with teeth but no INCONCLUSIVE: C-attacker -- a trained
arm's mass on ``dex_attacker`` states must stay <= ATTACKER_LOW, else switch inflation
is not wall-conditioned and the verdict is forced to H3-GLOBAL (H1/H2 blocked).

Honesty note: the human-side conditionals were peeked at for feasibility while setting
floors -- H on loop-species states is ~0.20 (vs 0.184/0.189 overall), so H1 is unlikely
a priori and the live outcomes are expected to be H2 vs H3. The POLICY mass was not
measured before these thresholds were frozen. Shard masks are the permissive
``synthesize_action_mask`` (any live bench mon counts as switchable), so shard-mask P
is a floor on live-mask P; U per conditioning anchors the comparison.

Secondary (``--replay-dir``): the ingest pipeline drops first-reveal and forced
switches, so the shard H undercounts true human switching; a raw replay-log pass
(first qualifying |move|/|switch| per side per turn; post-faint, |drag|, and
``[from]``-pivot switches excluded) bounds that undercount.

    python scripts/switch_mass_gate.py                        # full: 6 ckpts x 2 shards
    python scripts/switch_mass_gate.py --limit 2000 --no-replay-rate \\
        --bc-checkpoints checkpoints/bc_gen9ou_v5_s0.pt \\
        --offrl-checkpoints checkpoints/offrl_gen9ou_v5_s0.pt   # smoke
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lategame.features.encoder import OBS_LAYOUT

_RESULTS = Path("results/switch_mass_gate.json")
_DECISIONS_DEFAULT = Path("results/behavior_probe_decisions.jsonl")
_REPLAY_DIR_DEFAULT = Path("replays/gen9ou")
_PRIORS_PATH = Path(__file__).resolve().parent.parent / (
    "lategame/features/data/id_priors_gen9.npz"
)

# Loop-state extraction from the Build-6 decision stream (pre-registered).
RUN_K = 10  # a "loop run" = >= 10 consecutive voluntary switch decisions
MIN_RUN_FRAC = 0.15  # loop species: our_active in >= 15% of loop runs
PAIR_MIN_COUNT = 25  # loop pairs: (own, opp) seen >= 25 times inside loop runs

# Conditioning definitions.
MIN_HP = 0.5  # "healthy": live loops ran at high HP, walls not in KO range
DEX_WALL_MIN = 1.0  # defensiveness z >= 1.0 ~ top decile (corvi/tinglu/zama/pex)
DEX_ATTACKER_MAX = -1.0  # clear attackers (dragapult -1.9, ironvaliant -2.9)

# Sample floors.
MIN_ROWS_PRIMARY = 1000  # per (shard, loop_species); measured ~9.7k/19k
MIN_ROWS_PAIR = 200  # below this loop_pair is advisory only (never gates)

# Decision-tree thresholds.
HUMAN_HEAVY = 0.40  # H1 precondition: >= ~2x the 0.184/0.189 overall base
MATCH_TOL = 0.10  # |P - H| <= 0.10 -> policy "matches" the human rate
AMPLIFY_DELTA = 0.20  # and ...
AMPLIFY_RATIO = 1.75  # ... P >= 1.75 * H -> amplification (needs P >= ~0.40 at H ~0.2)
IN_DIST_HIGH = 0.60  # live-like mass reproduced in-distribution (>> uniform ~0.45)
ARGMAX_AMP = 0.50  # T >= 0.5 while P matches H -> argmax-amplification annotation
ATTACKER_LOW = 0.30  # specificity-control ceiling on dex_attacker mass
BC_VS_AWR_DELTA = 0.15  # H3 localization: offrl P - bc P on the same shard

# Controls.
UNIFORM_HARNESS_TOL = 1e-5  # zero-logits masked softmax must equal U to this
RANDOM_INIT_BAND = (0.25, 0.65)  # random-init ET overall switch-mass sanity band
SEED_AGREEMENT = 2  # >= 2 of 3 seeds must share the verdict, else family = MIXED

# Frozen from the Build-6 JSONL (runs >= RUN_K) so the conditioning is reproducible
# even though results/behavior_probe_decisions.jsonl is gitignored.
LOOP_SPECIES_FALLBACK = (
    "gholdengo",
    "greattusk",
    "kingambit",
    "corviknight",
    "slowkinggalar",
    "tinglu",
    "zamazenta",
    "dragonite",
)
LOOP_PAIRS_FALLBACK = (
    ("corviknight", "gholdengo"),
    ("corviknight", "greattusk"),
    ("corviknight", "kingambit"),
    ("corviknight", "slowkinggalar"),
    ("gholdengo", "corviknight"),
    ("gholdengo", "gholdengo"),
    ("gholdengo", "greattusk"),
    ("gholdengo", "kingambit"),
    ("gholdengo", "slowkinggalar"),
    ("greattusk", "corviknight"),
    ("greattusk", "gholdengo"),
    ("greattusk", "greattusk"),
    ("greattusk", "kingambit"),
    ("kingambit", "corviknight"),
    ("kingambit", "gholdengo"),
    ("kingambit", "greattusk"),
    ("kingambit", "kingambit"),
    ("kingambit", "ragingbolt"),
    ("slowkinggalar", "corviknight"),
    ("slowkinggalar", "gholdengo"),
    ("slowkinggalar", "greattusk"),
    ("slowkinggalar", "kingambit"),
    ("slowkinggalar", "ragingbolt"),
    ("slowkinggalar", "slowkinggalar"),
    ("tinglu", "gholdengo"),
    ("tinglu", "kingambit"),
    ("zamazenta", "gholdengo"),
)

CONDITIONINGS = (
    "overall",
    "loop_species",  # primary
    "loop_species_any_hp",
    "dex_wall",
    "dex_attacker",  # specificity control
    "loop_pair",
)
PRIMARY = "loop_species"

# Obs offsets, derived from the encoder layout so they can never silently drift.
_PDIM = OBS_LAYOUT.pokemon_dim
_TEAM = OBS_LAYOUT.team_size
_ACTIVE_IDX = 1  # within a pokemon block: present, active, fainted, hp_fraction
_HP_IDX = 3
_SPECIES_IDX = OBS_LAYOUT.pokemon_numeric_dim  # first trailing ID channel
_FORCE_SWITCH_IDX = OBS_LAYOUT.global_start + OBS_LAYOUT.global_dim - 3


# --------------------------------------------------------------------------- #
# Pure-logic helpers (numpy / stdlib only; unit-tested without torch).
# --------------------------------------------------------------------------- #
def decode_actives(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row own/opp active species vocab id, own hp, and sanity counters.

    Rows with no flagged active get species 0 (== vocab UNK); ``n_own_active`` lets the
    caller assert the expected exactly-one invariant instead of silently trusting it.
    """
    n = obs.shape[0]
    blocks = obs[:, : OBS_LAYOUT.moves_start].reshape(n, 2 * _TEAM, _PDIM)
    own, opp = blocks[:, :_TEAM], blocks[:, _TEAM:]
    rows = np.arange(n)

    def _active(side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flags = side[:, :, _ACTIVE_IDX] > 0.5
        count = flags.sum(axis=1)
        idx = flags.argmax(axis=1)
        species = np.where(count > 0, side[rows, idx, _SPECIES_IDX], 0.0)
        hp = np.where(count > 0, side[rows, idx, _HP_IDX], 0.0)
        return species.astype(np.int64), hp.astype(np.float32), count.astype(np.int64)

    own_species, own_hp, n_own = _active(own)
    opp_species, _, n_opp = _active(opp)
    return {
        "own_species": own_species,
        "own_hp": own_hp,
        "opp_species": opp_species,
        "n_own_active": n_own,
        "n_opp_active": n_opp,
        "force_switch": obs[:, _FORCE_SWITCH_IDX] > 0.5,
    }


def defensiveness(species_matrix: np.ndarray) -> np.ndarray:
    """Per-vocab-row defensiveness: z(def) + z(spd) - z(atk) - z(spa).

    Columns 0-5 of the priors matrix are z-scored base stats (hp, atk, def, spa, spd,
    spe), so the score is ranking-valid only; DEX_WALL_MIN=1.0 sits at ~p90.
    """
    m = species_matrix
    return np.asarray(m[:, 2] + m[:, 4] - m[:, 1] - m[:, 3], dtype=np.float64)


def voluntary_runs(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Maximal runs of consecutive voluntary switch decisions per (arm, battle).

    Mirrors ``behavior_probe._max_switch_run``: forced rows neither extend nor break a
    run; a voluntary non-switch ends it; battle boundaries reset.
    """
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    key: tuple[str, str] | None = None
    for row in rows:
        row_key = (str(row.get("arm", "")), str(row.get("battle_tag", "")))
        if row_key != key:
            key = row_key
            if current:
                runs.append(current)
            current = []
        if row.get("forced"):
            continue
        if int(row["action"]) < 6:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def loop_species_from_runs(
    runs: Sequence[Sequence[dict[str, Any]]], run_k: int = RUN_K, min_run_frac: float = MIN_RUN_FRAC
) -> list[str]:
    """Species whose our_active appears in >= ``min_run_frac`` of runs of length >= k."""
    long_runs = [r for r in runs if len(r) >= run_k]
    if not long_runs:
        return []
    counts = Counter(
        species for run in long_runs for species in {str(d["our_active"]) for d in run}
    )
    return sorted(s for s, c in counts.items() if c / len(long_runs) >= min_run_frac)


def loop_pairs_from_runs(
    runs: Sequence[Sequence[dict[str, Any]]], run_k: int = RUN_K, min_count: int = PAIR_MIN_COUNT
) -> list[tuple[str, str]]:
    """(own, opp) active pairs seen >= ``min_count`` times inside runs of length >= k."""
    counts = Counter(
        (str(d["our_active"]), str(d["opp_active"]))
        for run in runs
        if len(run) >= run_k
        for d in run
    )
    return sorted(p for p, c in counts.items() if c >= min_count)


def species_to_ids(
    names: Iterable[str], table: dict[str, int]
) -> tuple[dict[str, int], list[str]]:
    """Map live species id-strs to vocab indices; exact first, forme-prefix fallback.

    Mirrors ``behavior_probe._species_match``: 'zacian' and 'zaciancrowned' are the
    same mon up to a forme suffix. Unmatched names are returned for the caller to warn.
    """
    mapped: dict[str, int] = {}
    missing: list[str] = []
    for name in names:
        if name in table:
            mapped[name] = table[name]
            continue
        candidates = sorted(
            (k for k in table if k.startswith(name) or name.startswith(k)),
            key=len,
        )
        if candidates:
            mapped[name] = table[candidates[0]]
        else:
            missing.append(name)
    return mapped, missing


def conditioning_masks(
    actives: dict[str, np.ndarray],
    loop_ids: set[int],
    pair_ids: set[tuple[int, int]],
    def_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Boolean row masks for every conditioning in ``CONDITIONINGS``."""
    own, opp, hp = actives["own_species"], actives["opp_species"], actives["own_hp"]
    known = own > 0
    healthy = hp >= MIN_HP
    in_loop = np.isin(own, sorted(loop_ids)) if loop_ids else np.zeros(own.shape, bool)
    score = def_score[own]
    pair_ok = np.zeros(own.shape, bool)
    if pair_ids:
        # Pack (own, opp) into one int so membership is a single vectorized isin;
        # rows with an undecodable opp (opp == 0) can never match a real pair.
        packed = own.astype(np.int64) * 1_000_000 + opp.astype(np.int64)
        wanted = sorted(a * 1_000_000 + b for a, b in pair_ids)
        pair_ok = np.isin(packed, wanted) & (opp > 0)
    return {
        "overall": np.ones(own.shape, bool),
        "loop_species": in_loop & healthy,
        "loop_species_any_hp": in_loop,
        "dex_wall": known & healthy & (score >= DEX_WALL_MIN),
        "dex_attacker": known & healthy & (score <= DEX_ATTACKER_MAX),
        "loop_pair": pair_ok,
    }


def uniform_switch_mass(mask: np.ndarray) -> np.ndarray:
    """Per-row switch mass of the uniform-over-legal policy: |legal switches|/|legal|."""
    legal = mask.sum(axis=1)
    return np.divide(
        mask[:, :6].sum(axis=1),
        legal,
        out=np.zeros(mask.shape[0], dtype=np.float64),
        where=legal > 0,
    )


def masked_softmax_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numpy reference for the model path: -1e9 fill (policy.NEG_INF), then softmax."""
    filled = np.where(mask, logits.astype(np.float64), -1e9)
    filled -= filled.max(axis=1, keepdims=True)
    exp = np.exp(filled)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True))


def summarize_conditioning(
    cond: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    switch_mass: np.ndarray | None = None,
    top1_switch: np.ndarray | None = None,
) -> dict[str, Any]:
    """{n, H, U} for a conditioning; plus {P, T} when policy rows are supplied."""
    n = int(cond.sum())
    out: dict[str, Any] = {"n": n}
    if n == 0:
        out.update({"human_rate": None, "uniform_mass": None})
        if switch_mass is not None:
            out.update({"policy_mass": None, "top1_switch_rate": None})
        return out
    out["human_rate"] = float((actions[cond] < 6).mean())
    out["uniform_mass"] = float(uniform_switch_mass(mask[cond]).mean())
    if switch_mass is not None:
        out["policy_mass"] = float(switch_mass[cond].mean())
    if top1_switch is not None:
        out["top1_switch_rate"] = float(top1_switch[cond].mean())
    return out


def classify_arm(
    primary: dict[str, Any], pair: dict[str, Any] | None, attacker: dict[str, Any]
) -> dict[str, Any]:
    """The pre-registered tree on plain floats (see module docstring)."""
    if primary.get("human_rate") is None or primary.get("policy_mass") is None:
        return {
            "verdict": "EMPTY",
            "argmax_amp": False,
            "attacker_ok": True,
            "flags": {},
            "margins": {},
        }
    h = float(primary["human_rate"])
    p = float(primary["policy_mass"])
    t = float(primary.get("top1_switch_rate") or 0.0)
    p_pair: float | None = None
    if pair is not None and int(pair["n"]) >= MIN_ROWS_PAIR and pair["policy_mass"] is not None:
        p_pair = float(pair["policy_mass"])
    attacker_mass = attacker.get("policy_mass")
    attacker_ok = attacker_mass is None or float(attacker_mass) <= ATTACKER_LOW

    amplified = p >= h + AMPLIFY_DELTA and p >= AMPLIFY_RATIO * h
    matched = abs(p - h) <= MATCH_TOL
    human_heavy = h >= HUMAN_HEAVY
    in_dist_live = max(p, p_pair if p_pair is not None else 0.0) >= IN_DIST_HIGH

    if not attacker_ok:
        verdict = "H3-GLOBAL"  # inflation is not wall-conditioned; H1/H2 blocked
    elif amplified:
        verdict = "H3"
    elif human_heavy and matched:
        verdict = "H1"
    elif matched and not human_heavy and not in_dist_live:
        verdict = "H2"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "argmax_amp": bool(matched and t >= ARGMAX_AMP),
        "attacker_ok": attacker_ok,
        "flags": {
            "amplified": amplified,
            "matched": matched,
            "human_heavy": human_heavy,
            "in_dist_live": in_dist_live,
        },
        "margins": {
            "H": h,
            "P": p,
            "T": t,
            "P_pair": p_pair,
            "attacker_mass": attacker_mass,
            "P_minus_H": p - h,
        },
    }


def family_verdict(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Majority verdict across seeds (>= SEED_AGREEMENT, capped at the seed count so a
    single-seed smoke run still classifies), else MIXED + disagreement."""
    verdicts = [str(c["verdict"]) for c in per_seed]
    counts = Counter(verdicts)
    top, top_n = counts.most_common(1)[0]
    if top_n >= min(SEED_AGREEMENT, len(verdicts)):
        return {"verdict": top, "seed_disagreement": len(counts) > 1, "seeds": verdicts}
    return {"verdict": "MIXED", "seed_disagreement": True, "seeds": verdicts}


def _log_id_str(name: str) -> str:
    """poke_env.to_id_str semantics (lowercase alnum) without importing poke-env here."""
    return "".join(c for c in name.lower() if c.isalnum())


def parse_replay_log(log_text: str, loop_species: set[str]) -> dict[str, int]:
    """Decision/switch counts from one raw Showdown log (both sides).

    A decision is each side's first qualifying |move|/|switch| per turn. Excluded from
    decisions (but still tracked for active/reveal state): pre-turn-1 leads, |drag|,
    ``[from]`` consequences (pivot moves, locked moves), and post-faint replacements
    (a |switch| after that side's |faint| in the same turn). This matches what the
    ingest sampler treats as a policy decision, while also counting the first-reveal
    voluntary switches ingest drops.
    """
    counts = {
        "decisions": 0,
        "switches": 0,
        "first_reveal_switches": 0,
        "loop_decisions": 0,
        "loop_switches": 0,
        "excluded_forced": 0,
        "excluded_from": 0,
        "excluded_drag": 0,
    }
    turn = 0
    active: dict[str, str] = {}
    revealed: dict[str, set[str]] = {"p1": set(), "p2": set()}
    acted: dict[str, bool] = {"p1": False, "p2": False}
    fainted: dict[str, bool] = {"p1": False, "p2": False}

    for line in log_text.split("\n"):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        if tag == "turn":
            turn = int(parts[2])
            acted = {"p1": False, "p2": False}
            fainted = {"p1": False, "p2": False}
            continue
        if tag not in ("move", "switch", "drag", "replace", "faint"):
            continue
        side = parts[2][:2]
        if side not in ("p1", "p2"):
            continue
        if tag == "faint":
            fainted[side] = True
            continue
        if tag == "move":
            if "[from]" in line or turn < 1 or acted[side]:
                continue
            counts["decisions"] += 1
            if active.get(side) in loop_species:
                counts["loop_decisions"] += 1
            acted[side] = True
            continue
        # switch / drag / replace: details field carries the species.
        species = _log_id_str(parts[3].split(",")[0]) if len(parts) > 3 else ""
        if tag == "switch" and turn >= 1:
            if "[from]" in line:
                counts["excluded_from"] += 1
            elif fainted[side]:
                counts["excluded_forced"] += 1
            elif not acted[side]:
                counts["decisions"] += 1
                counts["switches"] += 1
                if species not in revealed[side]:
                    counts["first_reveal_switches"] += 1
                if active.get(side) in loop_species:
                    counts["loop_decisions"] += 1
                    counts["loop_switches"] += 1
                acted[side] = True
        elif tag == "drag":
            counts["excluded_drag"] += 1
        revealed[side].add(species)
        active[side] = species
    return counts


# --------------------------------------------------------------------------- #
# Torch side (lazy imports so the test module never pulls torch).
# --------------------------------------------------------------------------- #
def load_policy(path: str, obs_version: str, obs_dim: int) -> Any:
    """Load a v5 checkpoint the way BCAgent does, guarded against the SHARD's stamps."""
    import torch

    from lategame.model.factory import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("obs_version") != obs_version or int(ckpt.get("input_dim", -1)) != obs_dim:
        raise SystemExit(
            f"checkpoint/shard encoder mismatch: {path} has "
            f"{ckpt.get('obs_version')}/{ckpt.get('input_dim')} vs shard "
            f"{obs_version}/{obs_dim}"
        )
    model = build_model(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def random_init_model(reference_ckpt: str, seed: int = 0) -> Any:
    """Same architecture as the reference checkpoint, freshly initialized (C-random)."""
    import torch

    from lategame.model.factory import build_model

    ckpt = torch.load(reference_ckpt, map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    model = build_model(ckpt)
    model.eval()
    return model


def score_shard(
    model: Any, obs: np.ndarray, mask: np.ndarray, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (switch mass, top-1-is-switch) under the masked softmax the agents use."""
    import torch

    from lategame.model.policy import masked_logits, policy_logits

    model.to(device)
    masses: list[np.ndarray] = []
    top1: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch_size):
            ob = torch.from_numpy(obs[i : i + batch_size]).float().to(device)
            mk = torch.from_numpy(mask[i : i + batch_size]).to(device)
            logits = masked_logits(policy_logits(model(ob)), mk)
            probs = torch.softmax(logits, dim=1)
            masses.append(probs[:, :6].sum(dim=1).float().cpu().numpy())
            top1.append((logits.argmax(dim=1) < 6).cpu().numpy())
    model.to("cpu")
    return np.concatenate(masses).astype(np.float64), np.concatenate(top1)


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _load_shard(path: str, limit: int) -> dict[str, Any]:
    npz = np.load(path)
    obs = npz["obs"][:limit] if limit else npz["obs"]
    actions = npz["action"][: obs.shape[0]]
    mask = npz["mask"][: obs.shape[0]]
    return {
        "path": path,
        "obs": np.asarray(obs, dtype=np.float32),
        "action": np.asarray(actions),
        "mask": np.asarray(mask, dtype=bool),
        "obs_version": str(npz["obs_version"]),
        "obs_dim": int(npz["obs_dim"]),
        "rows": int(obs.shape[0]),
    }


def _loop_states(
    decisions_path: Path,
) -> tuple[list[str], list[tuple[str, str]], str, int]:
    """Loop species/pairs from the Build-6 JSONL, or the frozen fallback constants."""
    if decisions_path.exists():
        rows = [
            json.loads(line)
            for line in decisions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runs = voluntary_runs(rows)
        species = loop_species_from_runs(runs)
        pairs = loop_pairs_from_runs(runs)
        if species:
            return species, pairs, "decisions_jsonl", len(rows)
    fallback_pairs: list[tuple[str, str]] = [(a, b) for a, b in LOOP_PAIRS_FALLBACK]
    return list(LOOP_SPECIES_FALLBACK), fallback_pairs, "frozen_fallback", 0


def _replay_log_rate(replay_dir: Path, loop_species: set[str]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    n_files = 0
    for path in sorted(replay_dir.glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8")).get("log", "")
        totals.update(parse_replay_log(log, loop_species))
        n_files += 1
    decisions = totals["decisions"]
    return {
        "n_replays": n_files,
        "n_decisions": decisions,
        "voluntary_switch_rate": totals["switches"] / decisions if decisions else None,
        "first_reveal_share": (
            totals["first_reveal_switches"] / totals["switches"] if totals["switches"] else None
        ),
        "loop_species_switch_rate": (
            totals["loop_switches"] / totals["loop_decisions"]
            if totals["loop_decisions"]
            else None
        ),
        "excluded": {
            k: int(totals[k]) for k in ("excluded_forced", "excluded_from", "excluded_drag")
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bc-checkpoints",
        default=",".join(f"checkpoints/bc_gen9ou_v5_s{s}.pt" for s in range(3)),
        help="Comma-separated BC checkpoints (train shard = --bc-shard)",
    )
    ap.add_argument(
        "--offrl-checkpoints",
        default=",".join(f"checkpoints/offrl_gen9ou_v5_s{s}.pt" for s in range(3)),
        help="Comma-separated offline-RL checkpoints (train shard = --rl-shard)",
    )
    ap.add_argument("--bc-shard", default="data/gen9ou_v5_bc.npz")
    ap.add_argument("--rl-shard", default="data/gen9ou_v5_rl.npz")
    ap.add_argument("--decisions", default=str(_DECISIONS_DEFAULT))
    ap.add_argument("--replay-dir", default=str(_REPLAY_DIR_DEFAULT))
    ap.add_argument(
        "--replay-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute the log-based true human switch rate (undercount bound)",
    )
    ap.add_argument(
        "--cross-shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score every checkpoint on both shards (needed for BC-vs-AWR localization)",
    )
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Row cap per shard (smoke)")
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"

    loop_species, loop_pairs, loop_source, n_decision_rows = _loop_states(Path(args.decisions))
    print(f"loop states from {loop_source}: species={loop_species} pairs={len(loop_pairs)}")

    from lategame.features.vocab import load_vocab

    table = load_vocab().tables["species"]
    species_ids, missing = species_to_ids(loop_species, table)
    if missing:
        print(f"WARNING: loop species not in vocab, dropped: {missing}")
    pair_ids = {
        (species_ids[a], species_ids[b])
        for a, b in loop_pairs
        if a in species_ids and b in species_ids
    }

    priors = np.load(_PRIORS_PATH)
    vocab_version = load_vocab().version
    if str(priors["vocab_version"]) != vocab_version:
        raise SystemExit(
            f"priors vocab_version {priors['vocab_version']} != vocab {vocab_version}"
        )
    def_score = defensiveness(priors["species"])

    shards = {
        "bc_shard": _load_shard(args.bc_shard, args.limit),
        "rl_shard": _load_shard(args.rl_shard, args.limit),
    }
    checks: dict[str, bool] = {}
    human: dict[str, Any] = {}
    controls: dict[str, Any] = {"random_init_mass": {}, "attacker": {"bc": [], "offrl": []}}
    harness_err = 0.0
    for name, shard in shards.items():
        actives = decode_actives(shard["obs"])
        conds = conditioning_masks(actives, set(species_ids.values()), pair_ids, def_score)
        shard["conds"] = conds
        human[name] = {
            c: summarize_conditioning(conds[c], shard["action"], shard["mask"])
            for c in CONDITIONINGS
        }
        human[name]["decode"] = {
            "one_own_active_frac": float((actives["n_own_active"] == 1).mean()),
            "opp_active_missing_frac": float((actives["n_opp_active"] == 0).mean()),
            "force_switch_frac": float(actives["force_switch"].mean()),
        }
        # C-harness: the zero-logits masked softmax must reproduce U exactly.
        zero_probs = masked_softmax_np(
            np.zeros((shard["rows"], shard["mask"].shape[1])), shard["mask"]
        )
        gap = zero_probs[:, :6].sum(axis=1) - uniform_switch_mass(shard["mask"])
        err = float(np.abs(gap).max())
        harness_err = max(harness_err, err)
        # C-invariant: every taken action is mask-legal.
        legal_taken = bool(
            shard["mask"][np.arange(shard["rows"]), shard["action"]].all()
        )
        checks[f"mask_invariant_{name}"] = legal_taken
        checks[f"sample_floor_{name}"] = int(conds[PRIMARY].sum()) >= (
            MIN_ROWS_PRIMARY if not args.limit else 1
        )
    checks["control_harness"] = harness_err <= UNIFORM_HARNESS_TOL
    controls["harness_identity_max_err"] = harness_err

    arm_specs = [
        (f"bc_s{i}", "bc", p, "bc_shard")
        for i, p in enumerate(x for x in args.bc_checkpoints.split(",") if x)
    ] + [
        (f"offrl_s{i}", "offrl", p, "rl_shard")
        for i, p in enumerate(x for x in args.offrl_checkpoints.split(",") if x)
    ]
    if not arm_specs:
        raise SystemExit("no checkpoints given")

    # C-random: a freshly initialized ET must sit near the uniform anchor.
    rand_model = random_init_model(arm_specs[0][2])
    rand_ok = True
    for name, shard in shards.items():
        mass, _ = score_shard(rand_model, shard["obs"], shard["mask"], args.batch_size, device)
        controls["random_init_mass"][name] = float(mass.mean())
        rand_ok &= RANDOM_INIT_BAND[0] <= float(mass.mean()) <= RANDOM_INIT_BAND[1]
    checks["control_random_init"] = rand_ok

    arms: dict[str, Any] = {}
    for arm, family, ckpt_path, train_shard in arm_specs:
        model = load_policy(
            ckpt_path, shards[train_shard]["obs_version"], shards[train_shard]["obs_dim"]
        )
        per_shard: dict[str, Any] = {}
        targets = shards if args.cross_shard else {train_shard: shards[train_shard]}
        for name, shard in targets.items():
            mass, top1 = score_shard(model, shard["obs"], shard["mask"], args.batch_size, device)
            per_shard[name] = {
                c: summarize_conditioning(
                    shard["conds"][c], shard["action"], shard["mask"], mass, top1
                )
                for c in CONDITIONINGS
            }
        own = per_shard[train_shard]
        classification = classify_arm(own[PRIMARY], own.get("loop_pair"), own["dex_attacker"])
        controls["attacker"][family].append(own["dex_attacker"]["policy_mass"])
        arms[arm] = {
            "checkpoint": ckpt_path,
            "train_shard": train_shard,
            "per_shard": per_shard,
            "classification": classification,
        }
        m = classification["margins"]
        print(
            f"  {arm:9} H {_fmt(m.get('H'))} P {_fmt(m.get('P'))} T {_fmt(m.get('T'))} "
            f"U {_fmt(own[PRIMARY]['uniform_mass'])} pairP {_fmt(m.get('P_pair'))} "
            f"atk {_fmt(m.get('attacker_mass'))} -> {classification['verdict']}"
        )

    families = {
        fam: family_verdict([arms[a]["classification"] for a, f, _, _ in arm_specs if f == fam])
        for fam in ("bc", "offrl")
        if any(f == fam for _, f, _, _ in arm_specs)
    }

    bc_vs_awr: dict[str, Any] | None = None
    if args.cross_shard and "bc" in families and "offrl" in families:
        bc_masses = [
            arms[a]["per_shard"]["rl_shard"][PRIMARY]["policy_mass"]
            for a, f, _, _ in arm_specs
            if f == "bc"
        ]
        rl_masses = [
            arms[a]["per_shard"]["rl_shard"][PRIMARY]["policy_mass"]
            for a, f, _, _ in arm_specs
            if f == "offrl"
        ]
        delta = float(np.mean(rl_masses) - np.mean(bc_masses))
        bc_vs_awr = {
            "shard": "rl_shard",
            "delta_primary_mass": delta,
            "awr_side": delta >= BC_VS_AWR_DELTA,
        }

    replay_rate: dict[str, Any] | None = None
    if args.replay_rate and Path(args.replay_dir).is_dir():
        replay_rate = _replay_log_rate(Path(args.replay_dir), set(loop_species))
        replay_rate["shard_rate_for_comparison"] = human["rl_shard"]["overall"]["human_rate"]

    inconclusive = not all(checks.values())
    verdict = (
        "INCONCLUSIVE"
        if inconclusive
        else "; ".join(f"{fam}={v['verdict']}" for fam, v in families.items())
    )

    result = {
        "thresholds": {
            "run_k": RUN_K,
            "min_run_frac": MIN_RUN_FRAC,
            "pair_min_count": PAIR_MIN_COUNT,
            "min_hp": MIN_HP,
            "dex_wall_min": DEX_WALL_MIN,
            "dex_attacker_max": DEX_ATTACKER_MAX,
            "min_rows_primary": MIN_ROWS_PRIMARY,
            "min_rows_pair": MIN_ROWS_PAIR,
            "human_heavy": HUMAN_HEAVY,
            "match_tol": MATCH_TOL,
            "amplify_delta": AMPLIFY_DELTA,
            "amplify_ratio": AMPLIFY_RATIO,
            "in_dist_high": IN_DIST_HIGH,
            "argmax_amp": ARGMAX_AMP,
            "attacker_low": ATTACKER_LOW,
            "bc_vs_awr_delta": BC_VS_AWR_DELTA,
            "uniform_harness_tol": UNIFORM_HARNESS_TOL,
            "random_init_band": RANDOM_INIT_BAND,
            "seed_agreement": SEED_AGREEMENT,
        },
        "provenance": {
            "bc_shard": {k: shards["bc_shard"][k] for k in ("path", "rows", "obs_version")},
            "rl_shard": {k: shards["rl_shard"][k] for k in ("path", "rows", "obs_version")},
            "loop_species": loop_species,
            "loop_pairs": [list(p) for p in loop_pairs],
            "loop_source": loop_source,
            "decisions_rows": n_decision_rows,
            "limit": args.limit,
            "device": device,
            "note_mask": "shard masks are permissive synthesize_action_mask; shard-mask P"
            " is a floor on live-mask P",
        },
        "human": human,
        "replay_log_rate": replay_rate,
        "arms": arms,
        "controls": controls,
        "families": families,
        "bc_vs_awr": bc_vs_awr,
        "checks": checks,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== Build 7 switch-mass gate ({loop_source}) ===")
    for name in shards:
        h = human[name]
        print(
            f"{name}: overall H {_fmt(h['overall']['human_rate'])} "
            f"U {_fmt(h['overall']['uniform_mass'])} | {PRIMARY} n {h[PRIMARY]['n']} "
            f"H {_fmt(h[PRIMARY]['human_rate'])} U {_fmt(h[PRIMARY]['uniform_mass'])}"
        )
    if replay_rate and replay_rate["voluntary_switch_rate"] is not None:
        print(
            f"replay-log true switch rate {_fmt(replay_rate['voluntary_switch_rate'])} "
            f"(shard records {_fmt(replay_rate['shard_rate_for_comparison'])}; first-reveal"
            f" share {_fmt(replay_rate['first_reveal_share'])})"
        )
    for fam, v in families.items():
        print(f"family {fam}: {v['verdict']} (seeds {v['seeds']})")
    if bc_vs_awr:
        print(
            f"bc-vs-awr on rl_shard: delta {bc_vs_awr['delta_primary_mass']:+.3f} "
            f"(awr_side={bc_vs_awr['awr_side']})"
        )
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    if inconclusive:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

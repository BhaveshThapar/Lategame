"""OU Pivot Build 8 -- live-vs-shard drift localization: WHICH channels cause the loop?

Build 7 proved the ~0.9 live switch mass is an OOD artifact (H2): in-distribution the v5
policy matches the human switch rate (P 0.16-0.23), so the pathology lives entirely in the
gap between live loop states and the human-shard manifold. This gate localizes that gap to a
channel group -- offline, zero training -- so Build 9 knows what to fix.

The instrument is a **causal swap bisection**. Phase A (``behavior_probe.py --obs-out``)
captured the exact obs/mask the model scored live; those loop states reproduce ~0.73 switch
mass offline (validated). We pair-match each live loop state to shard rows with the same
(own-active, opp-active) species, then for each channel group G paste G's values across the
manifold both ways and re-score the frozen policy:

  * deletion  (base = LIVE, donor = SHARD): replace G with an in-distribution value -> does
    switch mass FALL toward the shard's ~0.2? A group that cures the loop is the carrier.
  * insertion (base = SHARD, donor = LIVE): inject G's live value into a healthy state -> does
    switch mass RISE toward ~0.9? A group that induces the loop is the carrier.

A group's ``frac_explained`` = |ΔP| / |P_live - P_shard|. Carrier = the group(s) explaining
>= CARRIER_FRAC of the gap, in the SAME direction, consistent across BOTH checkpoint families
(bc + offrl). The swap holds action legality fixed (hybrid keeps the base row's mask), so it
isolates how obs *features* move switch mass, not how many switches are legal.

Channel groups (from ``features.encoder.OBS_LAYOUT``, mapped to the pre-registered drift
candidates G1-G5):

  own_active_ids      species/item/ability IDs of the own active mon   (G1 own-kit values)
  own_active_numeric  hp/status/boosts/stats of the own active mon
  own_bench           the five non-active own blocks                   (G2 own composition)
  opp_active          the whole opponent active block                  (G3 opponent side)
  opp_bench           the five non-active opponent blocks              (G3 opponent side)
  moves               the four active-move blocks                      (G1 own-kit values)
  global              weather/fields/hazards/turn flags                (G4 game state)

Controls with teeth (any failure -> INCONCLUSIVE, exit 1): C-self (donor == base row -> the
swap is an exact identity, |ΔP| <= SELF_TOL), C-full (swapping ALL channels must move base P
by >= FULL_MIN_FRAC of the gap -- proves the swap *can* move P), C-harness (the torch masked
softmax equals the numpy reference), C-floors (>= MIN_LIVE_LOOP live loop rows/family and
>= MIN_SHARD_MATCHED matched shard rows). Onset stratification (P vs turn on loop states)
tests G5 (compounding self-generated drift): high P on the FIRST loop-state decision each
battle => static off-manifold drift; P that ramps within a run => compounding.

    python scripts/drift_probe.py                                   # full: 6 ckpts
    python scripts/drift_probe.py --limit 4000 --cap-base 100 \\
        --bc-checkpoints checkpoints/bc_gen9ou_v5_s0.pt \\
        --offrl-checkpoints checkpoints/offrl_gen9ou_v5_s0.pt        # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lategame.features.encoder import OBS_LAYOUT

# Reuse the Build-7 gate's validated machinery (obs decode, policy load, scoring, the frozen
# loop-species constants, the numpy masked-softmax reference). It lives next to this script,
# not in the package, so make it importable before pulling it in.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import switch_mass_gate as smg  # noqa: E402

_RESULTS = Path("results/drift_probe.json")
_OBS_DEFAULT = Path("results/behavior_probe_obs.npz")

# Loop-state conditioning (identical to the Build-7 primary conditioning).
LOOP_SPECIES = smg.LOOP_SPECIES_FALLBACK
MIN_HP = smg.MIN_HP  # 0.5 "healthy"

# Swap sampling / thresholds.
MIN_PAIR = 15  # a (own,opp) pair must have >= this many rows on BOTH sides to swap
CAP_BASE = 300  # base rows per pair (subsampled, seeded)
CAP_DONOR = 60  # donor rows per pair
CARRIER_FRAC = 0.50  # a group must explain >= 50% of the P gap to be the carrier
SECONDARY_FRAC = 0.25  # a contributing (not sole) group
SELF_TOL = 1e-4  # C-self: donor==base identity swap must not move P
FULL_MIN_FRAC = 0.50  # C-full: the ALL-channel swap must move >= this frac of the gap
HARNESS_TOL = 1e-4  # C-harness: torch vs numpy masked softmax
MIN_LIVE_LOOP = 300  # C-floor: live loop rows per family
MIN_SHARD_MATCHED = 500  # C-floor: matched shard rows total
SWAP_SEED = 0

# Encoder-derived offsets (mirror switch_mass_gate; never hand-coded).
_PDIM = OBS_LAYOUT.pokemon_dim
_NUM = OBS_LAYOUT.pokemon_numeric_dim  # numeric channels before the trailing ID tail
_TEAM = OBS_LAYOUT.team_size
_MDIM = OBS_LAYOUT.move_dim
_MNUM = OBS_LAYOUT.move_numeric_dim  # numeric move channels before the trailing move-ID tail
_N_MOVES = OBS_LAYOUT.n_moves
_MOVES_START = OBS_LAYOUT.moves_start
_GLOBAL_START = OBS_LAYOUT.global_start

# Ordered so the own-kit (G1) split is adjacent; ALL is the positive control, not a group.
GROUPS = (
    "own_active_ids",
    "own_active_numeric",
    "own_bench",
    "opp_active",
    "opp_bench",
    "moves",
    "global",
)
# Diagnostic-only sub-splits of the ``moves`` block (reported separately so they never
# double-count against ``moves`` in the carrier verdict):
#   move_ids      the trailing vocab-id channel (move IDENTITY)
#   move_numeric  every derived numeric feature
#   move_context  the two STATE-dependent numeric channels only (pp fraction + expected-damage
#                 ``score``) -- these can differ train-vs-live even for the SAME move id, so if
#                 they carry the gap the fix is the R-CALC damage-proxy encoding, not the kit.
# ``_MOVE_PP_IDX`` / ``_MOVE_SCORE_IDX`` mirror encoder._encode_move's scalar order
# (present, base_power, accuracy, priority, pp_fraction, score).
_MOVE_PP_IDX = 4
_MOVE_SCORE_IDX = 5
MOVE_SUBGROUPS = ("move_ids", "move_numeric", "move_context", "move_score", "move_pp")


# --------------------------------------------------------------------------- #
# Pure-logic helpers (numpy only; unit-tested without torch).
# --------------------------------------------------------------------------- #
def active_block_indices(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row own/opp active *block* index (0-5). Argmax of the active flag per side.

    Rows with no flagged active resolve to block 0; callers restrict to decoded loop states
    where exactly one own active is present, so this is exact there.
    """
    n = obs.shape[0]
    blocks = obs[:, :_MOVES_START].reshape(n, 2 * _TEAM, _PDIM)
    own = blocks[:, :_TEAM, smg._ACTIVE_IDX] > 0.5
    opp = blocks[:, _TEAM:, smg._ACTIVE_IDX] > 0.5
    return own.argmax(axis=1).astype(np.int64), opp.argmax(axis=1).astype(np.int64)


def _blk(i: int) -> int:
    """Flat offset of Pokemon block ``i`` (own 0-5, opp 6-11)."""
    return i * _PDIM


def swap_group(
    base: np.ndarray,
    donor: np.ndarray,
    group: str,
    base_own: np.ndarray,
    base_opp: np.ndarray,
    donor_own: np.ndarray,
    donor_opp: np.ndarray,
) -> np.ndarray:
    """Return ``base`` with ``group``'s channels overwritten by ``donor``'s, role-aligned.

    ``base``/``donor`` are [M, D] and row-aligned (row i's donor is donor[i]). Active/bench
    groups follow each row's own active-block index, so the donor's active mon lands on the
    base's active slot even when the two rows keep their actives in different slots. ``ALL``
    copies the whole donor row (the C-full positive control).
    """
    hyb = base.copy()
    if group == "ALL":
        hyb[:] = donor
        return hyb
    for r in range(base.shape[0]):
        b_bench = [i for i in range(_TEAM) if i != int(base_own[r])]
        d_bench = [i for i in range(_TEAM) if i != int(donor_own[r])]
        bo_bench = [i for i in range(_TEAM) if i != int(base_opp[r])]
        do_bench = [i for i in range(_TEAM) if i != int(donor_opp[r])]
        if group == "own_active_ids":
            b, d = _blk(int(base_own[r])), _blk(int(donor_own[r]))
            hyb[r, b + _NUM : b + _PDIM] = donor[r, d + _NUM : d + _PDIM]
        elif group == "own_active_numeric":
            b, d = _blk(int(base_own[r])), _blk(int(donor_own[r]))
            hyb[r, b : b + _NUM] = donor[r, d : d + _NUM]
        elif group == "opp_active":
            b, d = _blk(_TEAM + int(base_opp[r])), _blk(_TEAM + int(donor_opp[r]))
            hyb[r, b : b + _PDIM] = donor[r, d : d + _PDIM]
        elif group == "own_bench":
            for bi, di in zip(b_bench, d_bench, strict=True):
                hyb[r, _blk(bi) : _blk(bi) + _PDIM] = donor[r, _blk(di) : _blk(di) + _PDIM]
        elif group == "opp_bench":
            for bi, di in zip(bo_bench, do_bench, strict=True):
                hyb[r, _blk(_TEAM + bi) : _blk(_TEAM + bi) + _PDIM] = donor[
                    r, _blk(_TEAM + di) : _blk(_TEAM + di) + _PDIM
                ]
        elif group == "moves":
            hyb[r, _MOVES_START:_GLOBAL_START] = donor[r, _MOVES_START:_GLOBAL_START]
        elif group == "move_ids":  # trailing ID channel of each of the 4 move blocks
            for j in range(_N_MOVES):
                s = _MOVES_START + j * _MDIM + _MNUM
                e = _MOVES_START + (j + 1) * _MDIM
                hyb[r, s:e] = donor[r, s:e]
        elif group == "move_numeric":  # base-power/acc/pp/score/category/type of each block
            for j in range(_N_MOVES):
                s = _MOVES_START + j * _MDIM
                hyb[r, s : s + _MNUM] = donor[r, s : s + _MNUM]
        elif group == "move_context":  # pp_fraction + expected-damage score only
            for j in range(_N_MOVES):
                s = _MOVES_START + j * _MDIM
                hyb[r, s + _MOVE_PP_IDX] = donor[r, s + _MOVE_PP_IDX]
                hyb[r, s + _MOVE_SCORE_IDX] = donor[r, s + _MOVE_SCORE_IDX]
        elif group == "move_score":  # expected-damage R-CALC proxy channel only
            for j in range(_N_MOVES):
                hyb[r, _MOVES_START + j * _MDIM + _MOVE_SCORE_IDX] = donor[
                    r, _MOVES_START + j * _MDIM + _MOVE_SCORE_IDX
                ]
        elif group == "move_pp":  # pp-fraction channel only
            for j in range(_N_MOVES):
                hyb[r, _MOVES_START + j * _MDIM + _MOVE_PP_IDX] = donor[
                    r, _MOVES_START + j * _MDIM + _MOVE_PP_IDX
                ]
        elif group == "global":
            hyb[r, _GLOBAL_START:] = donor[r, _GLOBAL_START:]
        else:
            raise ValueError(f"unknown group {group}")
    return hyb


def extract_group(
    obs: np.ndarray, own_idx: np.ndarray, opp_idx: np.ndarray, group: str
) -> np.ndarray:
    """Canonical fixed-width [N, w] view of a group's channels (active mon moved to front),
    for the distance screen. Bench blocks are concatenated in native order."""
    n = obs.shape[0]
    rows = np.arange(n)
    blocks = obs[:, :_MOVES_START].reshape(n, 2 * _TEAM, _PDIM)
    if group == "own_active_ids":
        return blocks[rows, own_idx, _NUM:_PDIM]
    if group == "own_active_numeric":
        return blocks[rows, own_idx, :_NUM]
    if group == "opp_active":
        return blocks[rows, _TEAM + opp_idx]
    if group in ("own_bench", "opp_bench"):
        out = np.zeros((n, 5 * _PDIM), dtype=obs.dtype)
        base = 0 if group == "own_bench" else _TEAM
        act = own_idx if group == "own_bench" else opp_idx
        for r in range(n):
            bench = [i for i in range(_TEAM) if i != int(act[r])]
            out[r] = np.concatenate([blocks[r, base + i] for i in bench])
        return out
    if group == "moves":
        return obs[:, _MOVES_START:_GLOBAL_START]
    if group == "global":
        return obs[:, _GLOBAL_START:]
    raise ValueError(f"unknown group {group}")


def group_distance(live: np.ndarray, shard: np.ndarray) -> float:
    """Standardized L2 distance between the live and shard means of a group's channels.

    Per channel: |mean_live - mean_shard| / pooled_std; report the RMS over channels so
    groups of different widths are comparable. A coarse ranking screen, not the verdict.
    """
    mu_l, mu_s = live.mean(axis=0), shard.mean(axis=0)
    pooled = np.sqrt(0.5 * (live.var(axis=0) + shard.var(axis=0))) + 1e-6
    z = np.abs(mu_l - mu_s) / pooled
    return float(np.sqrt(np.mean(z**2)))


def matched_pairs(
    live_key: np.ndarray, shard_key: np.ndarray, min_pair: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Pairs (packed (own,opp) id key) present in both sets with >= min_pair rows each.

    Returns ``{packed_key: (live_row_indices, shard_row_indices)}``.
    """
    live_by = _index_by(live_key)
    shard_by = _index_by(shard_key)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, li in live_by.items():
        si = shard_by.get(key)
        if si is not None and len(li) >= min_pair and len(si) >= min_pair:
            out[key] = (li, si)
    return out


def _index_by(keys: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    for i, k in enumerate(keys.tolist()):
        out.setdefault(int(k), []).append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def pack_pair(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Pack (own_id, opp_id) into one int key (mirrors switch_mass_gate.conditioning_masks)."""
    return own.astype(np.int64) * 1_000_000 + opp.astype(np.int64)


def assemble_base_donor(
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    base_obs: np.ndarray,
    base_mask: np.ndarray,
    base_own: np.ndarray,
    base_opp: np.ndarray,
    donor_obs: np.ndarray,
    donor_own: np.ndarray,
    donor_opp: np.ndarray,
    which_base: str,
    cap_base: int,
    cap_donor: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Build the row-aligned (base, donor) tables for one swap direction.

    ``which_base`` selects which side of each pair is the base ('live' or 'shard'); the other
    side donates. Per pair, up to ``cap_base`` base rows are kept and each is assigned a donor
    (cycled from up to ``cap_donor`` sampled donor rows), so donor detail is matched by the
    (own,opp) species pair. Returns the concatenated base obs/mask + donor obs + both sides'
    active-block indices, all row-aligned.
    """
    b_obs, b_mask, b_own, b_opp, d_obs, d_own, d_opp = [], [], [], [], [], [], []
    for _key, (live_i, shard_i) in sorted(pairs.items()):
        base_i, donor_i = (live_i, shard_i) if which_base == "live" else (shard_i, live_i)
        if len(base_i) > cap_base:
            base_i = rng.choice(base_i, cap_base, replace=False)
        if len(donor_i) > cap_donor:
            donor_i = rng.choice(donor_i, cap_donor, replace=False)
        assign = donor_i[np.arange(len(base_i)) % len(donor_i)]
        b_obs.append(base_obs[base_i])
        b_mask.append(base_mask[base_i])
        b_own.append(base_own[base_i])
        b_opp.append(base_opp[base_i])
        d_obs.append(donor_obs[assign])
        d_own.append(donor_own[assign])
        d_opp.append(donor_opp[assign])
    if not b_obs:
        w = base_obs.shape[1]
        return {
            "base_obs": np.zeros((0, w), np.float32),
            "base_mask": np.zeros((0, base_mask.shape[1]), bool),
            "base_own": np.zeros(0, np.int64),
            "base_opp": np.zeros(0, np.int64),
            "donor_obs": np.zeros((0, w), np.float32),
            "donor_own": np.zeros(0, np.int64),
            "donor_opp": np.zeros(0, np.int64),
        }
    return {
        "base_obs": np.concatenate(b_obs).astype(np.float32),
        "base_mask": np.concatenate(b_mask).astype(bool),
        "base_own": np.concatenate(b_own),
        "base_opp": np.concatenate(b_opp),
        "donor_obs": np.concatenate(d_obs).astype(np.float32),
        "donor_own": np.concatenate(d_own),
        "donor_opp": np.concatenate(d_opp),
    }


def frac_explained(delta_p: float, gap: float) -> float:
    """Signed fraction of the (P_live - P_shard) gap a swap moved, oriented so a carrier is
    positive in either direction (deletion lowers P, insertion raises it)."""
    if abs(gap) < 1e-9:
        return 0.0
    return delta_p / gap


def carrier_verdict(
    per_group: dict[str, dict[str, float]],
    carrier_frac: float,
    secondary_frac: float,
) -> dict[str, Any]:
    """Rank groups by mean frac_explained (over families x directions) and name the carrier.

    ``per_group[group] = {"bc": f_bc, "offrl": f_offrl, "mean": f}`` where each ``f`` is the
    average frac_explained across directions. Carrier requires >= carrier_frac in the mean AND
    both families individually >= secondary_frac (consistency). If none qualifies, reports the
    minimal cumulative set reaching carrier_frac, else NO_CARRIER (high-order/joint drift).
    """
    ranked = sorted(per_group.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    carriers = [
        g
        for g, f in ranked
        if f["mean"] >= carrier_frac
        and min(f["bc"], f["offrl"]) >= secondary_frac
    ]
    cumulative: list[str] = []
    total = 0.0
    for g, f in ranked:
        if total >= carrier_frac:
            break
        # Accumulate positive, family-consistent contributors in rank order (an
        # inconsistent group is not a carrier even if its mean alone clears the bar).
        if f["mean"] > 0 and min(f["bc"], f["offrl"]) >= secondary_frac:
            cumulative.append(g)
            total += f["mean"]
    if carriers:
        verdict = carriers[0]
    elif total >= carrier_frac and len(cumulative) <= 2:
        verdict = "+".join(cumulative)
    else:
        verdict = "NO_CARRIER"
    return {
        "verdict": verdict,
        "ranking": [
            {"group": g, "mean": f["mean"], "bc": f["bc"], "offrl": f["offrl"]}
            for g, f in ranked
        ],
        "sole_carriers": carriers,
        "cumulative_set": cumulative,
        "cumulative_frac": total,
    }


# --------------------------------------------------------------------------- #
# Assembly (torch side reuses switch_mass_gate.load_policy / score_shard).
# --------------------------------------------------------------------------- #
def _loop_mask(obs: np.ndarray, loop_ids: set[int]) -> np.ndarray:
    """Boolean row mask: own-active species in the loop set AND healthy (hp >= MIN_HP)."""
    act = smg.decode_actives(obs)
    in_loop = (
        np.isin(act["own_species"], sorted(loop_ids))
        if loop_ids
        else np.zeros(obs.shape[0], bool)
    )
    return in_loop & (act["own_hp"] >= MIN_HP)


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main() -> None:  # noqa: C901 - a linear gate assembly
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=str(_OBS_DEFAULT), help="Phase-A obs sidecar (npz)")
    ap.add_argument("--rl-shard", default="data/gen9ou_v5_rl.npz", help="in-distribution shard")
    ap.add_argument(
        "--bc-checkpoints",
        default=",".join(f"checkpoints/bc_gen9ou_v5_s{s}.pt" for s in range(3)),
    )
    ap.add_argument(
        "--offrl-checkpoints",
        default=",".join(f"checkpoints/offrl_gen9ou_v5_s{s}.pt" for s in range(3)),
    )
    ap.add_argument("--limit", type=int, default=0, help="shard row cap (smoke)")
    ap.add_argument("--cap-base", type=int, default=CAP_BASE)
    ap.add_argument("--cap-donor", type=int, default=CAP_DONOR)
    ap.add_argument("--min-pair", type=int, default=MIN_PAIR)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- Live loop states (Phase A) --------------------------------------- #
    live_npz = np.load(args.obs, allow_pickle=False)
    live_obs_all = live_npz["obs"].astype(np.float32)
    live_mask_all = live_npz["mask"].astype(bool)
    live_arm = live_npz["arm"]
    live_turn = live_npz["turn"]
    live_tag = live_npz["battle_tag"]

    from lategame.features.vocab import load_vocab

    table = load_vocab().tables["species"]
    species_ids, missing = smg.species_to_ids(list(LOOP_SPECIES), table)
    if missing:
        print(f"WARNING loop species missing from vocab: {missing}")
    loop_ids = set(species_ids.values())

    live_loop = _loop_mask(live_obs_all, loop_ids)
    live_obs = live_obs_all[live_loop]
    live_mask = live_mask_all[live_loop]
    live_turn_l = live_turn[live_loop]
    live_tag_l = live_tag[live_loop]
    live_arm_l = live_arm[live_loop]
    live_act = smg.decode_actives(live_obs)
    live_own_idx, live_opp_idx = active_block_indices(live_obs)
    live_key = pack_pair(live_act["own_species"], live_act["opp_species"])
    print(f"live loop states: {live_obs.shape[0]} rows (of {live_obs_all.shape[0]} captured)")

    # --- Shard in-distribution reference ---------------------------------- #
    shard = smg._load_shard(args.rl_shard, args.limit)
    shard_loop = _loop_mask(shard["obs"], loop_ids)
    shard_obs = shard["obs"][shard_loop]
    shard_mask = shard["mask"][shard_loop]
    shard_act = smg.decode_actives(shard_obs)
    shard_own_idx, shard_opp_idx = active_block_indices(shard_obs)
    shard_key = pack_pair(shard_act["own_species"], shard_act["opp_species"])
    print(f"shard loop states: {shard_obs.shape[0]} rows")

    pairs = matched_pairs(live_key, shard_key, args.min_pair)
    n_matched_shard = int(sum(len(si) for _li, si in pairs.values()))
    n_matched_live = int(sum(len(li) for li, _si in pairs.values()))
    print(f"matched pairs: {len(pairs)} | live {n_matched_live} shard {n_matched_shard}")

    # --- Checkpoints ------------------------------------------------------ #
    specs = [
        (f"bc_s{i}", "bc", p)
        for i, p in enumerate(x for x in args.bc_checkpoints.split(",") if x)
    ] + [
        (f"offrl_s{i}", "offrl", p)
        for i, p in enumerate(x for x in args.offrl_checkpoints.split(",") if x)
    ]
    ov, od = shard["obs_version"], shard["obs_dim"]
    models = {name: smg.load_policy(path, ov, od) for name, _fam, path in specs}

    def score(model: Any, obs: np.ndarray, mask: np.ndarray) -> float:
        if obs.shape[0] == 0:
            return float("nan")
        m, _ = smg.score_shard(model, obs, mask, args.batch_size, device)
        return float(m.mean())

    # --- Reference P_live / P_shard per checkpoint ------------------------ #
    p_live = {n: score(models[n], live_obs, live_mask) for n, _f, _p in specs}
    p_shard = {n: score(models[n], shard_obs, shard_mask) for n, _f, _p in specs}
    gap = {n: p_live[n] - p_shard[n] for n, _f, _p in specs}
    for n, _f, _p in specs:
        print(f"  {n:9} P_live {p_live[n]:.3f} P_shard {p_shard[n]:.3f} gap {gap[n]:+.3f}")

    # --- Build base/donor tables once per direction ----------------------- #
    rng = np.random.default_rng(SWAP_SEED)
    tables = {
        "deletion": assemble_base_donor(
            pairs, live_obs, live_mask, live_own_idx, live_opp_idx,
            shard_obs, shard_own_idx, shard_opp_idx,
            "live", args.cap_base, args.cap_donor, rng,
        ),
        "insertion": assemble_base_donor(
            pairs, shard_obs, shard_mask, shard_own_idx, shard_opp_idx,
            live_obs, live_own_idx, live_opp_idx,
            "shard", args.cap_base, args.cap_donor, rng,
        ),
    }

    # --- Controls: C-self (identity) + C-full (positive) ------------------ #
    controls: dict[str, Any] = {}
    self_err = 0.0
    full_frac: dict[str, float] = {}
    for name, _f, _p in specs:
        model = models[name]
        for _direction, tbl in tables.items():
            if tbl["base_obs"].shape[0] == 0:
                continue
            base_p = score(model, tbl["base_obs"], tbl["base_mask"])
            # C-self: donor is the base row itself -> exact identity for every group.
            for grp in ("own_active_ids", "own_bench", "moves", "global"):
                hyb = swap_group(
                    tbl["base_obs"], tbl["base_obs"], grp,
                    tbl["base_own"], tbl["base_opp"], tbl["base_own"], tbl["base_opp"],
                )
                self_err = max(
                    self_err, abs(score(model, hyb, tbl["base_mask"]) - base_p)
                )
        # C-full: swapping ALL channels moves base P toward donor P.
        tbl = tables["deletion"]
        if tbl["base_obs"].shape[0]:
            base_p = score(model, tbl["base_obs"], tbl["base_mask"])
            full_hyb = swap_group(
                tbl["base_obs"], tbl["donor_obs"], "ALL",
                tbl["base_own"], tbl["base_opp"], tbl["donor_own"], tbl["donor_opp"],
            )
            full_p = score(model, full_hyb, tbl["base_mask"])
            g = gap[name]
            full_frac[name] = frac_explained(base_p - full_p, g)  # deletion: base=live
    controls["self_swap_max_err"] = self_err
    controls["full_swap_frac"] = full_frac
    controls["full_swap_min"] = min(full_frac.values()) if full_frac else float("nan")

    # C-harness: torch masked softmax == numpy reference on a base sample.
    harness_err = _harness_check(models[specs[0][0]], tables["deletion"], device)
    controls["harness_err"] = harness_err

    # --- The swap sweep: ΔP per (checkpoint, group, direction) ------------ #
    # Base P per (checkpoint, direction) is constant across groups -> cache it, and build each
    # group's hybrid once per direction (checkpoint-independent) then score across checkpoints.
    base_p_cache = {
        (name, direction): score(models[name], tbl["base_obs"], tbl["base_mask"])
        for name, _fam, _p in specs
        for direction, tbl in tables.items()
    }
    hybrids = {
        (grp, direction): swap_group(
            tbl["base_obs"], tbl["donor_obs"], grp,
            tbl["base_own"], tbl["base_opp"], tbl["donor_own"], tbl["donor_opp"],
        )
        for grp in (*GROUPS, *MOVE_SUBGROUPS)
        for direction, tbl in tables.items()
        if tbl["base_obs"].shape[0]
    }
    per_ckpt: dict[str, Any] = {}
    for name, fam, _p in specs:
        model = models[name]
        g = gap[name]
        grp_out: dict[str, Any] = {}
        for grp in GROUPS:
            fr: dict[str, float] = {}
            for direction, tbl in tables.items():
                if (grp, direction) not in hybrids:
                    fr[direction] = float("nan")
                    continue
                base_p = base_p_cache[(name, direction)]
                hyb_p = score(model, hybrids[(grp, direction)], tbl["base_mask"])
                # deletion: base=live, carrier LOWERS P (base_p - hyb_p > 0 explains gap).
                # insertion: base=shard, carrier RAISES P (hyb_p - base_p > 0).
                delta = (base_p - hyb_p) if direction == "deletion" else (hyb_p - base_p)
                fr[direction] = frac_explained(delta, g)
            valid = [v for v in fr.values() if not np.isnan(v)]
            grp_out[grp] = {
                "deletion": fr.get("deletion"),
                "insertion": fr.get("insertion"),
                "mean_frac": float(np.mean(valid)) if valid else float("nan"),
            }
        per_ckpt[name] = {"family": fam, "gap": g, "groups": grp_out}

    # --- Aggregate to a carrier verdict ----------------------------------- #
    per_group: dict[str, dict[str, float]] = {}
    for grp in GROUPS:
        fam_means: dict[str, float] = {}
        for fam in ("bc", "offrl"):
            vals = [
                per_ckpt[n]["groups"][grp]["mean_frac"]
                for n, f, _p in specs
                if f == fam and not np.isnan(per_ckpt[n]["groups"][grp]["mean_frac"])
            ]
            fam_means[fam] = float(np.mean(vals)) if vals else 0.0
        per_group[grp] = {
            "bc": fam_means["bc"],
            "offrl": fam_means["offrl"],
            "mean": float(np.mean([fam_means["bc"], fam_means["offrl"]])),
        }
    verdict = carrier_verdict(per_group, CARRIER_FRAC, SECONDARY_FRAC)

    # Moves sub-split diagnostic: within the carrier move block, is the drift the move
    # IDENTITY (vocab id) or the DERIVED numeric features (base-power/type/expected damage)?
    moves_subsplit: dict[str, dict[str, float]] = {}
    for grp in MOVE_SUBGROUPS:
        fam_means = {}
        for fam in ("bc", "offrl"):
            vals = []
            for name, f, _p in specs:
                if f != fam:
                    continue
                for direction, tbl in tables.items():
                    if (grp, direction) not in hybrids:
                        continue
                    base_p = base_p_cache[(name, direction)]
                    hyb_p = score(models[name], hybrids[(grp, direction)], tbl["base_mask"])
                    delta = (base_p - hyb_p) if direction == "deletion" else (hyb_p - base_p)
                    vals.append(frac_explained(delta, gap[name]))
            fam_means[fam] = float(np.mean(vals)) if vals else 0.0
        moves_subsplit[grp] = {
            "bc": fam_means["bc"],
            "offrl": fam_means["offrl"],
            "mean": float(np.mean([fam_means["bc"], fam_means["offrl"]])),
        }

    # --- Distance screen + onset stratification --------------------------- #
    distances = {
        grp: group_distance(
            extract_group(live_obs, live_own_idx, live_opp_idx, grp),
            extract_group(shard_obs, shard_own_idx, shard_opp_idx, grp),
        )
        for grp in GROUPS
    }
    onset = _onset_stratify(
        models, specs, live_obs, live_mask, live_tag_l, live_turn_l, device, args.batch_size
    )
    pp_dist = {
        "live": _pp_fraction_stats(live_obs),
        "shard": _pp_fraction_stats(shard_obs),
    }

    # --- Controls / verdict gating ---------------------------------------- #
    checks = {
        "self_swap_identity": self_err <= SELF_TOL,
        "full_swap_moves_P": (controls["full_swap_min"] >= FULL_MIN_FRAC)
        if full_frac else False,
        "harness": harness_err <= HARNESS_TOL,
        "floor_live": n_matched_live >= MIN_LIVE_LOOP,
        "floor_shard": n_matched_shard >= MIN_SHARD_MATCHED,
    }
    inconclusive = not all(checks.values())

    result = {
        "thresholds": {
            "min_pair": args.min_pair, "cap_base": args.cap_base, "cap_donor": args.cap_donor,
            "carrier_frac": CARRIER_FRAC, "secondary_frac": SECONDARY_FRAC,
            "self_tol": SELF_TOL, "full_min_frac": FULL_MIN_FRAC, "harness_tol": HARNESS_TOL,
            "min_live_loop": MIN_LIVE_LOOP, "min_shard_matched": MIN_SHARD_MATCHED,
            "min_hp": MIN_HP, "loop_species": list(LOOP_SPECIES),
        },
        "provenance": {
            "obs": args.obs, "rl_shard": shard["path"], "obs_version": ov, "device": device,
            "live_loop_rows": int(live_obs.shape[0]), "shard_loop_rows": int(shard_obs.shape[0]),
            "n_pairs": len(pairs), "matched_live": n_matched_live, "matched_shard": n_matched_shard,
            "live_arm_counts": {a: int((live_arm_l == a).sum()) for a in set(live_arm_l.tolist())},
        },
        "reference": {"p_live": p_live, "p_shard": p_shard, "gap": gap},
        "distances": distances,
        "per_checkpoint": per_ckpt,
        "per_group": per_group,
        "moves_subsplit": moves_subsplit,
        "pp_distribution": pp_dist,
        "onset": onset,
        "controls": controls,
        "checks": checks,
        "carrier": verdict,
        "verdict": "INCONCLUSIVE" if inconclusive else verdict["verdict"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    print("\n=== Build 8 drift probe ===")
    ranked_dist = ", ".join(
        f"{g} {distances[g]:.2f}" for g in sorted(GROUPS, key=lambda g: -distances[g])
    )
    print(f"distance screen (standardized): {ranked_dist}")
    print("carrier frac_explained (mean over families x directions):")
    for row in verdict["ranking"]:
        print(
            f"  {row['group']:20} mean {row['mean']:+.3f} "
            f"(bc {row['bc']:+.3f} offrl {row['offrl']:+.3f})"
        )
    print("moves sub-split frac_explained: "
          + ", ".join(f"{g} {moves_subsplit[g]['mean']:+.3f}" for g in MOVE_SUBGROUPS))
    print(f"controls: self_err {self_err:.1e} full_min {controls['full_swap_min']:.2f} "
          f"harness {harness_err:.1e} | floors live {n_matched_live} shard {n_matched_shard}")
    print(f"onset: P first-loop-decision {onset['p_first']:.3f} vs later {onset['p_later']:.3f}")
    print(f"pp full-fraction: live {pp_dist['live']['frac_full']:.3f} "
          f"shard {pp_dist['shard']['frac_full']:.3f}")
    print(f"\nVERDICT: {result['verdict']}  -> {args.out}")
    if inconclusive:
        raise SystemExit(1)


def _harness_check(model: Any, tbl: dict[str, np.ndarray], device: str) -> float:
    """Torch masked-softmax switch mass vs the numpy reference on up to 512 base rows."""
    if tbl["base_obs"].shape[0] == 0:
        return 0.0
    import torch

    from lategame.model.policy import policy_logits

    obs = tbl["base_obs"][:512]
    mask = tbl["base_mask"][:512]
    model.to(device)
    with torch.no_grad():
        logits = policy_logits(model(torch.from_numpy(obs).float().to(device))).cpu().numpy()
    model.to("cpu")
    ref = smg.masked_softmax_np(logits, mask)[:, :6].sum(axis=1)
    m, _ = smg.score_shard(model, obs, mask, 512, device)
    return float(np.abs(m - ref).max())


def _onset_stratify(
    models: dict[str, Any],
    specs: list[tuple[str, str, str]],
    live_obs: np.ndarray,
    live_mask: np.ndarray,
    tags: np.ndarray,
    turns: np.ndarray,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Mean switch mass on the FIRST loop-state decision per battle vs later ones (G5 test).

    High P on the first entry => the loop state is off-manifold on arrival (static drift);
    P that only rises on later decisions => compounding self-generated drift.
    """
    order = np.lexsort((turns, tags))
    seen: set[Any] = set()
    is_first = np.zeros(live_obs.shape[0], bool)
    for i in order:
        t = tags[i]
        if t not in seen:
            is_first[i] = True
            seen.add(t)
    masses = []
    for name, _f, _p in specs:
        m, _ = smg.score_shard(models[name], live_obs, live_mask, batch_size, device)
        masses.append(m)
    mass = np.mean(masses, axis=0)
    return {
        "p_first": float(mass[is_first].mean()) if is_first.any() else float("nan"),
        "p_later": float(mass[~is_first].mean()) if (~is_first).any() else float("nan"),
        "n_first": int(is_first.sum()),
        "n_later": int((~is_first).sum()),
    }


def _pp_fraction_stats(obs: np.ndarray) -> dict[str, float]:
    """Distribution of the active mon's move pp-fraction over present move blocks.

    Corroborates the causal ``move_pp`` carrier: the loop keeps every move at full pp
    (the agent never attacks), which is off-manifold vs the human shard where players do
    attack. ``frac_full`` (share of present moves at pp == max) is the discriminating stat.
    """
    n = obs.shape[0]
    if n == 0:
        return {"n": 0, "mean_pp": float("nan"), "frac_full": float("nan")}
    vals = []
    for j in range(_N_MOVES):
        s = _MOVES_START + j * _MDIM
        present = obs[:, s] > 0.5
        vals.append(obs[present, s + _MOVE_PP_IDX])
    pp = np.concatenate(vals) if vals else np.zeros(0)
    return {
        "n": int(n),
        "mean_pp": float(pp.mean()) if pp.size else float("nan"),
        "frac_full": float((pp >= 0.999).mean()) if pp.size else float("nan"),
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")


if __name__ == "__main__":
    main()

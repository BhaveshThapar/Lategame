"""OU Pivot Build 13 -- localize the pp-INDEPENDENT A->B->A ping-pong carrier.

Build 8 (``drift_probe.py``) localized the general switch loop to the move ``pp_fraction`` channel:
in a loop the active mon just switched in and never attacked, so its moves sit at full pp -- an
off-manifold "just switched" cue the policy reads as "switch again". Builds 11/12 then split the
residual loop in two: the pp-carried long runs respond to the resample regularizer, but a short
A->B->A ping-pong (a 2-cycle) has a **pp-independent rate** -- flat as pp-reliance halves -- and it
costs every game vs the competent heuristic.

Naively localizing on the 2-cycle states just re-finds pp (the active mon in a 2-cycle also just
switched in -> full pp). So this gate is **two-stage**, controlling for the known carrier:

  Stage 1 (measure pp's grip): neutralize pp on the 2-cycle states (reuse
    ``pp_reliance_diag.neutralize_pp`` -- draw each present-move pp cell from the shard's ~50%-full
    pool) and measure how much of P(return action) it removes.
        pp_share = (baseline - ppneut) / (baseline - neut_all)
        residual = ppneut / baseline          # the pp-INDEPENDENT return preference
  Stage 2 (localize the residual): from the pp-neutralized state, paste each other channel group's
    own-species-matched, healthy, in-distribution shard value over it and re-score.
        drop_res(G) = ppneut - P_return(ppneut+swapG);  frac_res(G) = drop_res(G)/(ppneut-neut_all)

Metric = P(return action): the softmax probability on the recorded switch-back index (the move back
onto the just-left mon), gathered per row -- the ping-pong is about *which* mon, not just *whether*
to switch. Conditioning = own active species (the 2-cycle concentrates on a few own mons; matching
the donor on the full (own,opp) pair discards most rows). The verdict is read on the **generating**
checkpoint (the seed whose live loop was captured); other seeds are scored as robustness only.

Pre-registered fork (Build 14's lever), all on the primary/generating seed:
  * INCONCLUSIVE            a control/floor failed, or baseline P(return) < BASELINE_MIN.
  * PP_CARRIED             ``residual < RESIDUAL_MIN``: pp-neutralization removes most of the return
        preference -> the 2-cycle is a pp phenomenon after all; more resample plateaued (Build 12),
        so the lever is the encoder pp-transform (make pp robust so the argmax margin moves).
  * DYNAMICS (Outcome C)   pp-independent (residual high) AND ``retained_res = neut_all/ppneut >=
        RETAINED_MAX``: even a full matched healthy state keeps the residual -> not a feature
        artifact -> decision-time anti-repetition / RL loop-penalty.
  * FEATURE_CARRIER:<group> (Outcome A)  pp-independent, full neutralization removes the residual,
        and one group explains >= CARRIER_FRAC of it -> robustify that feature (imitation).
  * DISTRIBUTED (Outcome B)  pp-independent, removable, but no single group dominates.

Built-in sanity: ``move_pp``'s residual frac must be ~0 (pp is already neutralized in the base), and
``frac_full`` must drop from ~1 to ~0.5 -- else pp-neutralization is not wired right.

    python scripts/pingpong_probe.py \
      --obs results/behavior_probe_obs_v13.npz \
      --decisions results/behavior_probe_decisions_v13.jsonl \
      --shard data/gen9ou_v7_bc.npz \
      --primary checkpoints/bc_gen9ou_v11_s0.pt \
      --robustness checkpoints/bc_gen9ou_v11_s1.pt,checkpoints/bc_gen9ou_v11_s2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_probe as dp  # noqa: E402
import pp_reliance_diag as ppd  # noqa: E402
import switch_mass_gate as smg  # noqa: E402

_RESULTS = Path("results/pingpong_probe.json")

# Pre-registered thresholds (fixed before the committed run so the verdict is not post-hoc).
RESIDUAL_MIN = 0.40  # residual = ppneut/baseline; below this the return preference is pp-carried
RETAINED_MAX = 0.50  # of the residual, full neutralization must remove >= 50% to be feature-borne
BASELINE_MIN = 0.15  # analysis validity: the generator must actually prefer the return action
MIN_2CYCLE = 80  # C-floor: flagged A->B->A rows on the target arm
MIN_MATCHED_LIVE = 80  # C-floor: own-species-matched 2-cycle base rows scored
MIN_SHARD_MATCHED = 200  # C-floor: matched in-distribution shard donor rows
MIN_HP = smg.MIN_HP  # donors are healthy in-distribution states (no reason to flee)
RESID_DENOM_MIN = 0.02  # if (ppneut - neut_all) is below this there is no residual to localize
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Pure-logic helpers (numpy / stdlib only; unit-tested without torch).
# --------------------------------------------------------------------------- #
def gather_action_prob(logits: np.ndarray, mask: np.ndarray, action_idx: np.ndarray) -> np.ndarray:
    """Per-row masked-softmax probability at each row's ``action_idx`` (the numpy reference)."""
    probs = smg.masked_softmax_np(logits, mask)
    return probs[np.arange(action_idx.shape[0]), action_idx]


def classify_fork(
    residual: float,
    retained_res: float,
    baseline: float,
    carrier: str,
    *,
    residual_min: float,
    retained_max: float,
    baseline_min: float,
    controls_ok: bool,
) -> str:
    """Map the two-stage metrics + carrier to the pre-registered Build-14 lever."""
    if not controls_ok or not np.isfinite(baseline) or baseline < baseline_min:
        return "INCONCLUSIVE"
    if not np.isfinite(residual) or residual < residual_min:
        return "PP_CARRIED"
    if not np.isfinite(retained_res) or retained_res >= retained_max:
        return "DYNAMICS"
    if carrier in ("NO_CARRIER", "SEED_INCONSISTENT") or "+" in carrier:
        return "DISTRIBUTED"
    return f"FEATURE_CARRIER:{carrier}"


def select_2cycle(
    jsonl_rows: list[dict[str, Any]],
    npz_action: np.ndarray,
    npz_tag: np.ndarray,
    npz_arm: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (global row indices, return-action) of the A->B->A rows on ``arm``.

    ``jsonl_rows`` (the decisions log, one dict per row) is row-aligned with the obs sidecar; we
    guard that alignment on the sidecar's own ``action``/``battle_tag`` before selecting. Detection
    reuses ``behavior_probe.two_cycle_rows`` on the JSONL ``team_order`` (species-faithful to the
    committed ``_ping_pong_rate``), restricted to the target arm.
    """
    from types import SimpleNamespace

    import behavior_probe as bp  # lazy: pulls poke-env; keep module import torch/server-free

    n = len(jsonl_rows)
    if n != npz_action.shape[0]:
        raise SystemExit(f"decisions/obs misaligned: {n} JSONL rows vs {npz_action.shape[0]} obs")
    for i in range(n):
        if int(jsonl_rows[i]["action"]) != int(npz_action[i]) or str(
            jsonl_rows[i]["battle_tag"]
        ) != str(npz_tag[i]):
            raise SystemExit(f"decisions/obs row {i} mismatch (action/battle_tag)")

    global_idx = [i for i in range(n) if str(npz_arm[i]) == arm]
    decs = [
        SimpleNamespace(
            forced=bool(jsonl_rows[i]["forced"]),
            action=int(jsonl_rows[i]["action"]),
            battle_tag=str(jsonl_rows[i]["battle_tag"]),
            team_order=list(jsonl_rows[i]["team_order"]),
        )
        for i in global_idx
    ]
    tc = bp.two_cycle_rows(decs)
    sel = np.array([global_idx[t.index] for t in tc], dtype=np.int64)
    ra = np.array([t.return_action for t in tc], dtype=np.int64)
    return sel, ra


# --------------------------------------------------------------------------- #
# Torch side (lazy imports; mirrors switch_mass_gate.score_shard but GATHERS).
# --------------------------------------------------------------------------- #
def score_action_prob(
    model: Any, obs: np.ndarray, mask: np.ndarray, action_idx: np.ndarray,
    batch_size: int, device: str,
) -> np.ndarray:
    """Per-row masked-softmax probability at ``action_idx`` under the agents' masked softmax."""
    import torch

    from rotomai.model.policy import masked_logits, policy_logits

    model.to(device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch_size):
            ob = torch.from_numpy(obs[i : i + batch_size]).float().to(device)
            mk = torch.from_numpy(mask[i : i + batch_size]).to(device)
            ai = torch.from_numpy(action_idx[i : i + batch_size]).long().to(device)
            probs = torch.softmax(masked_logits(policy_logits(model(ob)), mk), dim=1)
            g = probs[torch.arange(ob.shape[0], device=device), ai]
            out.append(g.float().cpu().numpy())
    model.to("cpu")
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def _gather_harness(
    model: Any, obs: np.ndarray, mask: np.ndarray, action_idx: np.ndarray, device: str
) -> float:
    """C-harness for the gather: torch P(action) vs the numpy reference on up to 512 rows."""
    if obs.shape[0] == 0:
        return 0.0
    import torch

    from rotomai.model.policy import policy_logits

    o, m, a = obs[:512], mask[:512], action_idx[:512]
    model.to(device)
    with torch.no_grad():
        logits = policy_logits(model(torch.from_numpy(o).float().to(device))).cpu().numpy()
    model.to("cpu")
    ref = gather_action_prob(logits, m, a)
    got = score_action_prob(model, o, m, a, 512, device)
    return float(np.abs(got - ref).max())


# --------------------------------------------------------------------------- #
# Scoring one checkpoint over the two-stage swap set.
# --------------------------------------------------------------------------- #
def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def _score_checkpoint(
    model: Any,
    base_obs: np.ndarray,
    base_ppneut: np.ndarray,
    hyb_all: np.ndarray,
    hybrids_res: dict[str, np.ndarray],
    base_mask: np.ndarray,
    base_ra: np.ndarray,
    groups_all: list[str],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Two-stage P(return) metrics for one checkpoint."""
    def sc(o: np.ndarray) -> float:
        return _mean(score_action_prob(model, o, base_mask, base_ra, batch_size, device))

    baseline = sc(base_obs)
    ppneut = sc(base_ppneut)
    neut_all = sc(hyb_all)
    denom_res = ppneut - neut_all
    drop_res = {g: ppneut - sc(hybrids_res[g]) for g in groups_all}
    # When pp already explains the whole drop the residual pool is ~0; report 0 rather than
    # dividing noise by a near-zero denominator (Stage-2 localization is moot when PP_CARRIED).
    has_resid = denom_res > RESID_DENOM_MIN
    frac_res = {g: (drop_res[g] / denom_res) if has_resid else 0.0 for g in groups_all}
    return {
        "baseline_return": baseline,
        "ppneut_return": ppneut,
        "neutralized_all": neut_all,
        "pp_share": ((baseline - ppneut) / (baseline - neut_all))
        if abs(baseline - neut_all) > _EPS else float("nan"),
        "residual": (ppneut / baseline) if abs(baseline) > _EPS else float("nan"),
        "retained_res": (neut_all / ppneut) if abs(ppneut) > _EPS else float("nan"),
        "drop_res": drop_res,
        "frac_res": frac_res,
    }


def _carrier(frac_res: dict[str, float]) -> str:
    """Single-family collapse of carrier_verdict (bc==offrl==frac) over the top-level groups."""
    pg = {g: {"bc": frac_res[g], "offrl": frac_res[g], "mean": frac_res[g]} for g in dp.GROUPS}
    return str(dp.carrier_verdict(pg, dp.CARRIER_FRAC, dp.SECONDARY_FRAC)["verdict"])


def main() -> None:  # noqa: C901 - a linear gate assembly
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default="results/behavior_probe_obs_v13.npz", help="obs sidecar (npz)")
    ap.add_argument("--decisions", default="results/behavior_probe_decisions_v13.jsonl")
    ap.add_argument("--shard", default="data/gen9ou_v7_bc.npz", help="in-distribution donor shard")
    ap.add_argument("--primary", default="checkpoints/bc_gen9ou_v11_s0.pt",
                    help="the generating checkpoint; the verdict is read on it")
    ap.add_argument("--robustness", default="", help="comma seeds scored for robustness")
    ap.add_argument("--arm", default="bc_vs_heuristic", help="arm the ping-pong costs games on")
    ap.add_argument("--match", default="own", choices=("own", "own_opp"),
                    help="donor conditioning: own active species (default) or the (own,opp) pair")
    ap.add_argument("--min-pair", type=int, default=10)
    ap.add_argument("--cap-donor", type=int, default=dp.CAP_DONOR)
    ap.add_argument("--limit", type=int, default=0, help="shard row cap (smoke)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0, help="pp-neutralization rng seed")
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- 2-cycle live states -------------------------------------------------- #
    npz = np.load(args.obs, allow_pickle=False)
    obs_all = npz["obs"].astype(np.float32)
    mask_all = npz["mask"].astype(bool)
    jsonl = [json.loads(x) for x in Path(args.decisions).read_text().splitlines() if x.strip()]
    sel, return_action = select_2cycle(
        jsonl, npz["action"], npz["battle_tag"], npz["arm"], args.arm
    )
    arm_counts = {
        a: int(select_2cycle(jsonl, npz["action"], npz["battle_tag"], npz["arm"], a)[0].size)
        for a in sorted({str(v) for v in npz["arm"].tolist()})
    }
    live_obs = obs_all[sel]
    live_mask = mask_all[sel]
    n_2cycle = int(sel.size)
    print(f"2-cycle rows on {args.arm}: {n_2cycle} (all arms: {arm_counts})")

    live_own_idx, live_opp_idx = dp.active_block_indices(live_obs)
    live_act = smg.decode_actives(live_obs)

    # --- In-distribution healthy shard donors --------------------------------- #
    shard = smg._load_shard(args.shard, args.limit)
    shard_act = smg.decode_actives(shard["obs"])
    healthy = shard_act["own_hp"] >= MIN_HP
    shard_obs = shard["obs"][healthy]
    shard_own_idx, shard_opp_idx = dp.active_block_indices(shard_obs)

    if args.match == "own_opp":
        live_key = dp.pack_pair(live_act["own_species"], live_act["opp_species"])
        shard_key = dp.pack_pair(
            shard_act["own_species"][healthy], shard_act["opp_species"][healthy]
        )
    else:  # match on own active species only (the mon deciding whether to return)
        live_key = live_act["own_species"].astype(np.int64)
        shard_key = shard_act["own_species"][healthy].astype(np.int64)

    pairs = dp.matched_pairs(live_key, shard_key, args.min_pair)
    n_matched_live = int(sum(len(li) for li, _si in pairs.values()))
    n_matched_shard = int(sum(len(si) for _li, si in pairs.values()))
    print(f"matched ({args.match}): {len(pairs)} groups | "
          f"live {n_matched_live} shard {n_matched_shard}")

    # Base = matched 2-cycle-live, donor = matched healthy shard (deletion). cap_base huge so no
    # 2-cycle row is dropped -> base order is the sorted-key concatenation, so the per-row
    # return-action rebuilds deterministically in that same order.
    rng = np.random.default_rng(dp.SWAP_SEED)
    tbl = dp.assemble_base_donor(
        pairs, live_obs, live_mask, live_own_idx, live_opp_idx,
        shard_obs, shard_own_idx, shard_opp_idx,
        "live", 10**9, args.cap_donor, rng,
    )
    base_ra = (
        np.concatenate([return_action[li] for _k, (li, _si) in sorted(pairs.items())]).astype(
            np.int64
        )
        if pairs
        else np.zeros(0, np.int64)
    )
    base_obs, base_mask, donor_obs = tbl["base_obs"], tbl["base_mask"], tbl["donor_obs"]
    assert base_ra.shape[0] == base_obs.shape[0], "return-action/base row misalignment"

    # --- Stage-1 base: pp neutralized (reuse pp_reliance_diag) ---------------- #
    pp_pool = ppd.shard_pp_pool(args.shard)
    base_ppneut = (
        ppd.neutralize_pp(base_obs, pp_pool, np.random.default_rng(args.seed))
        if base_obs.shape[0]
        else base_obs
    )
    frac_full_before = ppd.frac_full(base_obs) if base_obs.shape[0] else float("nan")
    frac_full_after = ppd.frac_full(base_ppneut) if base_obs.shape[0] else float("nan")

    # --- Checkpoints ---------------------------------------------------------- #
    primary = args.primary
    robustness = [p for p in args.robustness.split(",") if p]
    ov, od = shard["obs_version"], shard["obs_dim"]
    if od != live_obs.shape[1]:
        raise SystemExit(f"obs dim mismatch: sidecar {live_obs.shape[1]} vs shard {od}")
    models = {p: smg.load_policy(p, ov, od) for p in [primary, *robustness]}

    # --- Hybrids (checkpoint-independent) built once -------------------------- #
    # Stage 2 swaps each group onto the pp-neutralized base; neut_all is the full replacement of the
    # ORIGINAL base (pp included) -> the in-distribution floor.
    groups_all = list(dp.GROUPS) + list(dp.MOVE_SUBGROUPS)

    def _swap(src: np.ndarray, grp: str) -> np.ndarray:
        return dp.swap_group(
            src, donor_obs, grp,
            tbl["base_own"], tbl["base_opp"], tbl["donor_own"], tbl["donor_opp"],
        )

    hybrids_res = {g: _swap(base_ppneut, g) for g in groups_all} if base_obs.shape[0] else {}
    hyb_all = _swap(base_obs, "ALL") if base_obs.shape[0] else base_obs

    # --- Controls (primary) --------------------------------------------------- #
    m0 = models[primary]
    ppneut0 = _mean(score_action_prob(m0, base_ppneut, base_mask, base_ra, args.batch_size, device))
    self_err = 0.0
    for grp in ("own_active_ids", "own_bench", "opp_active", "global"):
        if base_obs.shape[0]:
            ident = dp.swap_group(
                base_ppneut, base_ppneut, grp,
                tbl["base_own"], tbl["base_opp"], tbl["base_own"], tbl["base_opp"],
            )
            p = _mean(score_action_prob(m0, ident, base_mask, base_ra, args.batch_size, device))
            self_err = max(self_err, abs(p - ppneut0))
    harness_err = _gather_harness(m0, base_obs, base_mask, base_ra, device)

    # --- Per-checkpoint sweep ------------------------------------------------- #
    def sweep(p: str) -> dict[str, Any]:
        return _score_checkpoint(
            models[p], base_obs, base_ppneut, hyb_all, hybrids_res, base_mask, base_ra,
            groups_all, args.batch_size, device,
        )

    prim = sweep(primary)
    robust = {p: sweep(p) for p in robustness}
    carrier = _carrier(prim["frac_res"])

    # --- Secondary switch-mass cross-check (primary, deletion only) ----------- #
    # Also split P(return) = switch_mass * P(return|switch): does pp drive the switch-vs-stay
    # decision, or the return-target selection specifically? P(return|switch) that is pp-INVARIANT
    # means pp carries the switching and the bounce-back is structural (few viable targets).
    if base_obs.shape[0]:
        smb, _ = smg.score_shard(m0, base_obs, base_mask, args.batch_size, device)
        smp, _ = smg.score_shard(m0, base_ppneut, base_mask, args.batch_size, device)
        smb_m, smp_m = _mean(smb), _mean(smp)
        sm = {
            "baseline": smb_m, "ppneut": smp_m,
            "return_given_switch_baseline": (prim["baseline_return"] / smb_m)
            if abs(smb_m) > _EPS else float("nan"),
            "return_given_switch_ppneut": (prim["ppneut_return"] / smp_m)
            if abs(smp_m) > _EPS else float("nan"),
        }
    else:
        sm = {"baseline": float("nan"), "ppneut": float("nan")}

    # --- Controls / floors / verdict ------------------------------------------ #
    checks = {
        "self_swap_identity": self_err <= dp.SELF_TOL,
        "gather_harness": harness_err <= dp.HARNESS_TOL,
        "pp_neut_lowered_frac_full": (frac_full_after < frac_full_before - 0.1)
        if np.isfinite(frac_full_before) else False,
        "floor_2cycle": n_2cycle >= MIN_2CYCLE,
        "floor_matched_live": n_matched_live >= MIN_MATCHED_LIVE,
        "floor_shard": n_matched_shard >= MIN_SHARD_MATCHED,
    }
    controls_ok = all(checks.values())
    verdict = classify_fork(
        prim["residual"], prim["retained_res"], prim["baseline_return"], carrier,
        residual_min=RESIDUAL_MIN, retained_max=RETAINED_MAX, baseline_min=BASELINE_MIN,
        controls_ok=controls_ok,
    )
    robust_forks = {
        p: classify_fork(
            robust[p]["residual"], robust[p]["retained_res"], robust[p]["baseline_return"],
            _carrier(robust[p]["frac_res"]),
            residual_min=RESIDUAL_MIN, retained_max=RETAINED_MAX, baseline_min=BASELINE_MIN,
            controls_ok=controls_ok,
        )
        for p in robustness
    }

    ranked = sorted(dp.GROUPS, key=lambda g: -prim["frac_res"][g])
    result = {
        "thresholds": {
            "residual_min": RESIDUAL_MIN, "retained_max": RETAINED_MAX,
            "baseline_min": BASELINE_MIN, "min_2cycle": MIN_2CYCLE,
            "min_matched_live": MIN_MATCHED_LIVE, "min_shard_matched": MIN_SHARD_MATCHED,
            "min_hp": MIN_HP, "min_pair": args.min_pair, "cap_donor": args.cap_donor,
            "carrier_frac": dp.CARRIER_FRAC, "secondary_frac": dp.SECONDARY_FRAC,
            "self_tol": dp.SELF_TOL, "harness_tol": dp.HARNESS_TOL, "match": args.match,
        },
        "provenance": {
            "obs": args.obs, "decisions": args.decisions, "shard": shard["path"],
            "arm": args.arm, "arm_2cycle_counts": arm_counts, "obs_version": ov,
            "primary": primary, "robustness": robustness, "device": device,
            "n_2cycle": n_2cycle, "n_matched_live": n_matched_live,
            "n_matched_shard": n_matched_shard, "n_groups_matched": len(pairs),
            "pp_frac_full_before": frac_full_before, "pp_frac_full_after": frac_full_after,
        },
        "metric": "P(return action) = softmax prob on the recorded switch-back index",
        "primary": prim,
        "carrier_of_residual": carrier,
        "robustness": robust,
        "robustness_forks": robust_forks,
        "move_pp_residual_frac_sanity": prim["frac_res"].get("move_pp"),
        "switch_mass_crosscheck": sm,
        "controls": {"self_swap_max_err": self_err, "gather_harness_err": harness_err},
        "checks": checks,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, indent=2, default=dp._json_default), encoding="utf-8"
    )

    print("\n=== Build 13 ping-pong probe ===")
    print(f"primary {Path(primary).name}: baseline {prim['baseline_return']:.3f} | "
          f"pp-neut {prim['ppneut_return']:.3f} | neut-ALL {prim['neutralized_all']:.3f}")
    print(f"  pp_share {prim['pp_share']:.3f} | residual {prim['residual']:.3f} | "
          f"retained_res {prim['retained_res']:.3f}")
    print("residual frac by group (Stage 2, on pp-neutralized base):")
    for g in ranked:
        print(f"  {g:20} {prim['frac_res'][g]:+.3f}")
    print("moves sub-split residual frac: " + ", ".join(
        f"{g} {prim['frac_res'][g]:+.3f}" for g in dp.MOVE_SUBGROUPS))
    print(f"move_pp residual sanity (want ~0): {prim['frac_res']['move_pp']:+.3f} | "
          f"pp frac_full {frac_full_before:.3f} -> {frac_full_after:.3f}")
    print(f"carrier_of_residual {carrier} | robustness forks {robust_forks}")
    print(f"controls: self {self_err:.1e} harness {harness_err:.1e} | "
          f"floors 2cycle {n_2cycle} live {n_matched_live} shard {n_matched_shard}")
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    if not controls_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

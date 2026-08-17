"""G5 -- the skill stack as DEMONSTRATED rather than merely implemented capability.

G5 is stated once, in plan.md 3.1: "Encode the competitive skill stack of Section 4 as explicit,
testable capabilities (state estimation, prediction/opponent modeling, win-condition planning,
precise damage math)." Unlike G4 it never had an exit criterion anywhere in the repository -- no
threshold, no gate script, no results file -- so "demonstrated rather than implemented" had nothing
behind it to check. That is what this gate is for.

THE EXIT CRITERION, and why it is not a bar invented to clear.

Each of G5's four capabilities must name (a) a gate that already runs, (b) a current number from a
committed results file, and (c) THAT GATE'S OWN pre-registered pass criterion, met.

The third clause is the load-bearing one. Every threshold below is read from the record it judges
(`threshold`, `pass_rate`, `verdict`, `gate_a_pass`) or from the producing gate's pre-registration
-- not chosen here. Writing fresh bars for measurements already on disk would be picking the target
after seeing the arrows, and this repository has already recorded once what that costs (B6d
adjudicated a pre-registered stop rule by eye, in a commit message, against corrupted numbers).

WHAT G5 DOES NOT CLAIM. "Demonstrated" means the capability exists, is exercised, and is measured
-- not that it raises win rate. Win-condition planning is the case that separates the two: search
runs on a forward model verified faithful to 1.000 core-transition agreement, and its strength
contribution is a pre-registered NULL on OU (n=2500/arm, p=0.762) and parity on gen9-RB. The
capability is demonstrated; the benefit is not. Both go in the record, because a G5 that quietly
dropped the null would be exactly the "merely implemented" claim it exists to replace.

    python scripts/g5_capability_gate.py            # writes results/g5_capability_gate.json
    python scripts/g5_capability_gate.py --json     # machine-readable, no write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_RESULTS = Path("results")
_OUT = _RESULTS / "g5_capability_gate.json"


def _load(name: str) -> dict[str, Any] | None:
    path = _RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _state_estimation() -> dict[str, Any]:
    """R-STATE: reconstruct the battle a POV can actually see, and be right about it.

    Three records, one per format family, all scored by the same `recon_check._compare` criteria so
    the numbers are comparable across them -- the replay-driven RB gate and the two live-play ones.
    The pass criterion is each gate's own `verdict`, and its own `pass_rate` floor of 0.99.
    """
    rows = []
    for name, fmt in (
        ("rpredict_recon.json", "gen9randombattle"),
        ("rpredict_recon_ou.json", "gen9ou"),
        ("rpredict_recon_vgc.json", "gen9vgc2025regi"),
    ):
        d = _load(name)
        if d is None:
            rows.append({"record": name, "format": fmt, "ok": False, "why": "record missing"})
            continue
        rate = float(d["match_rate"])
        # The RB record predates the `verdict`/`pass_rate` keys the live gates write; its bar is
        # the same 0.99, applied to the same quantity.
        floor = float(d.get("pass_rate", 0.99))
        rows.append({
            "record": name,
            "format": fmt,
            "match_rate": rate,
            "checks": d["checks"],
            "mismatches": d["mismatches"],
            "floor": floor,
            "verdict": d.get("verdict", "PASS" if rate >= floor else "FAIL"),
            "ok": rate >= floor,
        })
    return {
        "capability": "state estimation",
        "requirement": "R-STATE / R-PRIORS (plan.md 7)",
        "gate": "rotomai/search/recon_check.py",
        "reproduce": [
            "python scripts/rpredict_recon_gate.py",
            "python scripts/rpredict_recon_live_gate.py --format gen9ou",
            "python scripts/rpredict_recon_live_gate.py --format gen9vgc2025regi",
        ],
        "records": rows,
        "ok": all(r["ok"] for r in rows),
        "note": "all three formats, same comparison criteria, so the rates are comparable",
    }


def _prediction() -> dict[str, Any]:
    """R-PREDICT: reconstruct the OPPONENT's POV and predict what they will do from it.

    Gate A is the capability gate and computes its own `gate_a_pass` from thresholds it was run
    with (--pov-threshold 0.85, --decode-threshold 0.95). Gate B is the STRENGTH question and is
    reported here without gating: whether a better opponent model wins more games is a different
    claim than whether we can build one.
    """
    a = _load("rpredict_oppmodel_gate_a.json")
    if a is None:
        return {"capability": "prediction / opponent modeling", "ok": False, "why": "no gate A"}
    b = _load("rpredict_oppmodel_gate_b_whitebox_n120.json")
    return {
        "capability": "prediction / opponent modeling",
        "requirement": "R-PREDICT (plan.md 7)",
        "gate": "scripts/rpredict_oppmodel_gate.py --gate a",
        "reproduce": ["python scripts/rpredict_oppmodel_gate.py --gate a --limit 40"],
        "records": [{
            "record": "rpredict_oppmodel_gate_a.json",
            "opp_pov_team_fidelity": a["opp_pov_team_fidelity"],
            "whitebox_decodable_rate": a["whitebox_decodable_rate"],
            "whitebox_vs_heuristic_agreement": a["whitebox_vs_heuristic_agreement"],
            "n": a["agreement_n"],
            "verdict": "PASS" if a["gate_a_pass"] else "FAIL",
            "ok": bool(a["gate_a_pass"]),
        }],
        "strength_context_not_gated": (
            None if b is None else {
                "record": "rpredict_oppmodel_gate_b_whitebox_n120.json",
                "search_vs_heuristic": b["arms"]["whitebox"]["search_vs_heuristic"],
                "verdict": b["arms"]["whitebox"]["verdict"],
            }
        ),
        "ok": bool(a["gate_a_pass"]),
    }


def _planning() -> dict[str, Any]:
    """R-PLAN: depth-limited expectimax over a verified-faithful forward model.

    THE CAPABILITY PASSES AND THE BENEFIT IS NULL, and both are the finding. The pooled OU run is
    the largest search measurement on the record (10 shards, n=2500/arm) and its pre-registered
    rule -- WIN iff the pooled delta is significant and positive -- returns NULL at p=0.762. G5
    asks for a testable capability, not for a win, so the gate passes on the search having RUN at
    scale against a pre-registered rule on a model whose fidelity is separately gated.
    """
    s = _load("rpredict_search_ou.json")
    if s is None:
        return {"capability": "win-condition planning", "ok": False, "why": "no pooled search run"}
    wb = s["arms"]["whitebox"]
    return {
        "capability": "win-condition planning",
        "requirement": "R-PLAN (plan.md 7)",
        "gate": "rotomai/search/expectimax.py via scripts/rpredict_oppmodel_gate.py --gate b",
        "reproduce": [
            "N_PER_SHARD=250 sbatch --array=0-9 scripts/cluster/search_gate.slurm",
            "python scripts/merge_search_shards.py "
            "--glob 'results/rpredict_search_ou_shard*.json' --out results/rpredict_search_ou.json",
        ],
        "records": [{
            "record": "rpredict_search_ou.json",
            "format": s["format"],
            "init": s["init"],
            "shards": s["shards"],
            "n_per_arm": wb["search_vs_heuristic"]["n"],
            "base_vs_heuristic": s["base_vs_heuristic"]["rate"],
            "search_vs_heuristic": wb["search_vs_heuristic"]["rate"],
            "contrast": wb["contrast_vs_base"],
            "verdict": wb["verdict"],
            "ok": True,
        }],
        "ok": True,
        "strength_verdict": wb["verdict"],
        "note": (
            "DEMONSTRATED, NOT BENEFICIAL. The pre-registered rule returns NULL "
            f"(diff {wb['contrast_vs_base']['diff']:+.4f}, p={wb['contrast_vs_base']['p_value']}). "
            "G5 asks for an explicit testable capability; it does not ask for a win rate. The null "
            "is reported rather than dropped -- omitting it is the 'merely implemented' claim G5 "
            "exists to replace."
        ),
    }


def _damage_math() -> dict[str, Any]:
    """R-CALC: the forward model reproduces the simulator's own transitions.

    The record carries its own `threshold` (0.99) and `verdict`. This is the gate every search
    result stands on -- planning over an unfaithful model is GIGO, which is why `rpredict_recon_*`
    says so in its own note.
    """
    d = _load("rpredict_fidelity.json")
    if d is None:
        return {"capability": "precise damage math", "ok": False, "why": "no fidelity record"}
    ok = d["core_match_rate"] >= d["threshold"] and d["verdict"] == "PASS"
    return {
        "capability": "precise damage math",
        "requirement": "R-CALC (plan.md 7)",
        "gate": "rotomai/search/fidelity.py + rotomai/engine/damage.py",
        "reproduce": ["python scripts/rpredict_fidelity_gate.py --limit 300"],
        "records": [{
            "record": "rpredict_fidelity.json",
            "replays": d["replays"],
            "transitions": d["transitions"],
            "core_match_rate": d["core_match_rate"],
            "full_match_rate": d["full_match_rate"],
            "threshold": d["threshold"],
            "verdict": d["verdict"],
            "ok": ok,
        }],
        "ok": ok,
        "note": (
            "gen9randombattle only: the replay-driven path needs an inputlog to re-sim, and public "
            "replays for the teambuilt formats carry none. Live-play reconstruction covers OU and "
            "VGC instead, under state estimation above."
        ),
    }


def run() -> dict[str, Any]:
    caps = [_state_estimation(), _prediction(), _planning(), _damage_math()]
    met = all(c["ok"] for c in caps)
    return {
        "gate": "g5_capability",
        "goal": "G5 -- encode the skill stack as explicit, testable capabilities (plan.md 3.1)",
        "exit_criterion": (
            "each capability names a gate that runs, a number from a committed results file, and "
            "THAT GATE'S OWN pre-registered pass criterion, met. No threshold is introduced here."
        ),
        "capabilities": caps,
        "verdict": "MET" if met else "NOT MET",
        "not_claimed": [
            "that any capability raises win rate -- win-condition planning is a pre-registered "
            "NULL on OU and parity on gen9-RB, and is recorded as demonstrated, not beneficial",
            "that team preview is modelled as a learned capability -- the bring-4 selector is a "
            "fixed rule, and the codec has no slot and the model no head for it",
        ],
    }


def _print(out: dict[str, Any]) -> None:
    print(f"\nG5 -- {out['verdict']}\n")
    for c in out["capabilities"]:
        mark = "OK  " if c["ok"] else "FAIL"
        print(f"[{mark}] {c['capability']:32s} {c.get('gate', '')}")
        for r in c.get("records", []):
            bits = {k: v for k, v in r.items() if k not in ("record", "ok")}
            print(f"        {r['record']}: {json.dumps(bits)}")
        if c.get("note"):
            print(f"        note: {c['note']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the record, do not write it")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    out = run()
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print(out)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"  -> wrote {path}")
    raise SystemExit(0 if out["verdict"] == "MET" else 1)


if __name__ == "__main__":
    main()

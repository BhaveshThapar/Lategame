"""Persist PPO per-iteration telemetry from a run log into results/ (committed).

Build 21 needed to know whether Build 20's trust region ever bound in its EARLY iterations, and
ppo_continue_gate's JSON could not say: it stores only the win-rate curve. Build 20's note certifies
"epochs held at 4 in 40/40 LATE iters" -- silent on the early ones, which is exactly where a bigger
model takes its largest step and where the KL early-stop is most likely to fire.

The raw stdout log DOES carry it, but the log is gitignored (*.log), so it survives only as long as
nobody cleans the disk -- it is not an artifact a later build can rely on. Hence this script: parse
the log while it exists and commit the result as JSON.

(A correction to this file's own history: its first version, and commit f77acd7's message, claimed
Build 20's logs were already GONE. They were not -- results/ppo_ou_budget_s{0,1,2}.log were on disk
the whole time, and Build 21 recomputed v20's certificate from them. The premise was wrong; the
script is still needed, for the durability reason above.)

So: extract the telemetry a build's verdict actually leans on -- did the trust region bind?
did the critic fit? -- and commit it alongside the gate JSON, so the NEXT build can audit this
one instead of hitting the same dead end.

    python scripts/ppo_telemetry.py --log 0 run_s0.log --out results/ppo_ou_telemetry_v21.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# "  ppo iter   9  turns 5983  pi_loss -0.0181 ... approx_kl 0.0259 ... vmae 0.680  epochs 4 ..."
STATS = re.compile(
    r"ppo iter\s+(?P<iter>\d+)\s+turns\s+(?P<turns>\d+)\s+"
    r"pi_loss\s+(?P<pi_loss>-?[\d.]+)\s+v_loss\s+(?P<v_loss>-?[\d.]+)\s+"
    r"entropy\s+(?P<entropy>-?[\d.]+)\s+approx_kl\s+(?P<approx_kl>[\d.]+)\s+"
    r"clip\s+(?P<clip_frac>[\d.]+)\s+R\[.*?\]\s+vmae\s+(?P<vmae>[\d.]+)\s+"
    r"epochs\s+(?P<epochs>\d+)\s+ent_coef\s+(?P<ent_coef>[\d.]+)\s+lr\s+(?P<lr>[\d.e+-]+)"
    # B6f: the factored (doubles) run appends four more fields. The whole group is OPTIONAL, so
    # every OU log ever written still parses to exactly the same rows -- and a VGC certificate
    # gains the two numbers that make it meaningful there. `dec_frac` says what fraction of the
    # rollout could move the policy at all, and the trust region is only interpretable once the
    # approx-KL is measured over those rows rather than diluted by forced ones; `lp_drift` says
    # the acting and training distributions were the same function.
    r"(?:\s+dec_frac\s+(?P<dec_frac>[\d.]+)\s+n_decision\s+(?P<n_decision>\d+)"
    r"\s+invalid_frac\s+(?P<invalid_frac>[\d.]+)\s+lp_drift\s+(?P<lp_drift>[\d.e+-]+))?"
)

MAX_EPOCHS = 4  # PPOConfig.epochs; fewer means the KL early-stop fired (ppo.py:212)
KL_BAR = 0.045  # 1.5 * PPOConfig.target_kl (0.03) -- the bar v17-v22 all ran at


_FLOATS = ("pi_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "vmae", "ent_coef", "lr")

#: B6f, factored runs only. Absent from every OU log line, so the group is None there and the row
#: simply does not carry these keys -- old certificates stay byte-identical.
_OPTIONAL_FLOATS = ("dec_frac", "invalid_frac", "lp_drift")


def parse_log(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = STATS.search(line)
        if m:
            d = m.groupdict()
            row: dict[str, Any] = {
                "iter": int(d["iter"]),
                "turns": int(d["turns"]),
                "epochs": int(d["epochs"]),
                **{k: float(d[k]) for k in _FLOATS},
            }
            if d.get("dec_frac") is not None:
                row.update({k: float(d[k]) for k in _OPTIONAL_FLOATS})
                row["n_decision"] = int(d["n_decision"])
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], kl_bar: float = KL_BAR) -> dict[str, Any]:
    """The trust-region certificate: a NULL can only be blamed on capacity if this is clean.

    ``kl_bar`` is REPORTED, not used to detect binding -- binding is read off the epoch count,
    which is bar-independent. But it must match the ``target_kl`` the run actually used, or the
    certificate claims a bar the optimizer never enforced. Build 23 is the first run to move it.
    """
    bound = [r for r in rows if r["epochs"] < MAX_EPOCHS]
    kls = [r["approx_kl"] for r in rows]
    late = [r for r in rows if r["iter"] > len(rows) // 5]  # past the opening transient
    kl_late = round(sum(r["approx_kl"] for r in late) / len(late), 4) if late else None
    out: dict[str, Any] = {
        "n_iters": len(rows),
        "kl_bar": kl_bar,
        "trust_region_bound_iters": [r["iter"] for r in bound],
        "trust_region_bound_count": len(bound),
        "epochs_full_fraction": round(1 - len(bound) / max(len(rows), 1), 4),
        "approx_kl_min": round(min(kls), 4) if kls else None,
        "approx_kl_max": round(max(kls), 4) if kls else None,
        "approx_kl_mean_late": kl_late,
        "vmae_first": rows[0]["vmae"] if rows else None,
        "vmae_last": rows[-1]["vmae"] if rows else None,
    }
    # B6f, factored runs only. APPENDED, so an OU certificate keeps exactly the ten keys every
    # published one has. Both numbers are prerequisites for reading the rest of this dict on a
    # doubles run: an approx-KL diluted by forced turns would certify a trust region that never
    # bound, and a large `lp_drift` means the ratios those KLs summarize were meaningless.
    if rows and "dec_frac" in rows[0]:
        out["dec_frac_min"] = round(min(r["dec_frac"] for r in rows), 4)
        out["dec_frac_mean"] = round(sum(r["dec_frac"] for r in rows) / len(rows), 4)
        out["n_decision_total"] = sum(r["n_decision"] for r in rows)
        out["invalid_frac_max"] = round(max(r["invalid_frac"] for r in rows), 4)
        out["lp_drift_max"] = max(r["lp_drift"] for r in rows)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ppo_telemetry")
    ap.add_argument("--log", action="append", nargs=2, metavar=("SEED", "PATH"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--kl-bar",
        dest="kl_bar",
        type=float,
        default=KL_BAR,
        help="1.5 * the run's target_kl; MUST match the run (omit = 0.045, the v17-v22 bar)",
    )
    args = ap.parse_args(argv)

    result: dict[str, Any] = {"seeds": {}}
    for seed, path in args.log:
        rows = parse_log(path)
        result["seeds"][seed] = {"summary": summarize(rows, args.kl_bar), "iters": rows}
        s = result["seeds"][seed]["summary"]
        print(
            f"seed {seed}: {s['n_iters']} iters | trust region bound at "
            f"{s['trust_region_bound_iters'] or 'NEVER'} "
            f"({s['epochs_full_fraction']:.0%} full-epoch) | kl "
            f"[{s['approx_kl_min']}, {s['approx_kl_max']}] bar {s['kl_bar']} | "
            f"vmae {s['vmae_first']} -> {s['vmae_last']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

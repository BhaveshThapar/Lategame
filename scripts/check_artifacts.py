"""Check that the checkpoints behind the published headlines are still on disk.

The README's `## Artifacts & reproducibility` section makes a counted claim: N checkpoint paths are
referenced across `results/**.json`, some fraction of them no longer exists, and *none of the
missing ones back a headline*. A counted claim written in prose rots the next time a scratch session
is torn down -- and then the honesty section becomes the dishonest part of the README. This makes it
one command.

**The retention question and the verification question share a scan.** `prune_checkpoints.py`
decides what is safe to delete by text-scanning `results/**.json` for `checkpoints/...pt`; that same
scan, inverted, asks whether a cited artifact survived. `referenced_checkpoints` is imported rather
than reimplemented, so the two can never disagree about what "cited" means.

**Headlines are pinned by RESULT FILE, not by checkpoint path.** Hand-listing the ~18 checkpoints
behind the three headline numbers would go stale the first time a gate is re-pinned to a different
iteration (which `pin_gate_checkpoint.py` exists to do). Hand-listing the result file cannot: the
gate rewrites which checkpoints it names, and this script re-derives them.

**Exit status is deliberately asymmetric.** A missing *headline* artifact is a failure -- a
published number can no longer be reproduced on this machine. A missing non-headline artifact is a
*documented state*: it backs a superseded intermediate build whose measured numbers are still in the
JSON, and the README says so. Reported, exit 0.

Hashes are deliberately out of scope. Verifying them needs a committed manifest, a manifest needs a
regeneration ritual, and none of it helps the only reader who matters -- someone with a clone, who
has no checkpoints at all. Existence is the whole question here.

    python scripts/check_artifacts.py             # report + headline verdict
    python scripts/check_artifacts.py --verbose   # also list every missing path, by arm
    python scripts/check_artifacts.py --json      # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_checkpoints import (
    _CKPT_RE,
    referenced_checkpoints,
)

#: Published claim -> the results files it is booked from. Every checkpoint named by these must
#: exist, or the number cannot be re-read on this machine. Keep in step with the README table.
HEADLINE: dict[str, tuple[str, ...]] = {
    "gen9ou 0.7513 selection-free terminal": (
        "ppo_ou_gate_v26b_terminal.json",
        "seed_strength_gate_v26_terminal.json",
    ),
    # The same arm, same protocol, against the baseline the FIELD shares rather than the one this
    # repo wrote -- the only strength number here an outside reader can situate. The controls are
    # part of the headline on purpose: they re-read `heuristic`, whose value this project has
    # already published, in the same protocol at the same n, and that is what licenses reading the
    # simpleheuristics rate as the ARM's rather than the RUN's. One control shares an allocation
    # with a simpleheuristics read, so the opponent gap is measured within a run as well as across.
    "gen9ou 0.7303 vs poke-env SimpleHeuristicsPlayer (n=9000, three runs)": (
        "ppo_ou_gate_v26b_terminal.json",
        "seed_strength_gate_v26b_simpleheuristics.json",
        "seed_strength_gate_v26b_simpleheuristics_paired.json",
        "seed_strength_gate_v26b_heuristic_control.json",
        "seed_strength_gate_v26b_heuristic_paired.json",
    ),
    "gen9ou Glicko 1776.3 / GXE 0.7434": ("eval_ladder_gen9ou_v26.json",),
    # G1b. The only number here measured against HUMANS, and the only one an outside reader can
    # situate absolutely. Cites the pre-registration alongside the record on purpose: the bands and
    # the n are what make a WEAK verdict a result rather than a disappointment, and a headline that
    # cited only the outcome could be re-read later against bands chosen to fit it.
    "gen9ou ranked ladder Elo 1004 / GXE 30% (n=100, pre-registered WEAK)": (
        "live_ladder_gen9ou.json",
        "live_ladder_gen9ou_prereg.json",
        "live_ladder_gen9ou_seg0.json",
        "live_ladder_gen9ou_seg1.json",
        "live_ladder_gen9ou_seg2.json",
        "live_ladder_gen9ou_seg3.json",
    ),
    "VGC B6f C1 0.530": (
        "ppo_vgc_gate_b6f.json",
        "ppo_vgc_gate_b6f_s0.json",
        "ppo_vgc_gate_b6f_s1.json",
        "ppo_vgc_gate_b6f_s2.json",
        "seed_strength_gate_b6f_c1.json",
        "awr_vgc_arm_b6f.json",
    ),
    # Verifiable only since the record was given a `_provenance` block: `run_m1` wrote no
    # checkpoint paths, so before that this entry would have registered as vacuously OK over an
    # empty checkpoint set -- a headline that checks nothing.
    "VGC corrected ladder BC 0.453 / AWR 0.467": ("format_ceiling_gate_vgc_v2.json",),
    # G5 is an ASSEMBLY of gates that already ran, so the checkpoint it cites is the one the
    # weakest leg stands on -- the pooled OU search init. If that weight is gone, the planning
    # capability cannot be re-measured and the G5 verdict is not re-derivable, which is exactly
    # what a headline is supposed to catch.
    "G5 MET -- four capabilities, each on its own gate's pass criterion": (
        "g5_capability_gate.json",
    ),
    # `paper/rotomai.md` reports the retired gradient-noise diagnostic as a RESULT -- the instrument
    # whose probe-seed swing (2.7x) exceeded the effect it was pre-registered to detect (2.2x). That
    # claim is only re-derivable from the two probes that differ in nothing but `--seed`, so they
    # are audited like any other headline. Their absence would leave a published methodology claim
    # standing on prose alone.
    "paper: the diagnostic retired for a 2.7x probe-seed swing on a FIXED checkpoint": (
        "grad_noise_diag_b22_stageB0_e10.json",
        "grad_noise_diag_b22_stageB0_e10_seed1.json",
    ),
}


def audit(results_dir: Path, checkpoints_dir: Path) -> dict:
    """Partition every cited checkpoint into present/missing, then re-check the headline set.

    `checkpoints_dir` is honoured rather than assumed: cited paths are written relative to the repo
    root as ``checkpoints/...``, so a caller pointing at a staged tree elsewhere gets its first
    component rewritten rather than a spurious wall of misses.
    """

    def resolve(rel: str) -> Path:
        tail = rel.split("/", 1)[1] if "/" in rel else rel
        return checkpoints_dir / tail

    refs = referenced_checkpoints(results_dir)
    present = sorted(r for r in refs if resolve(r).exists())
    missing = sorted(r for r in refs if not resolve(r).exists())

    headlines = {}
    for claim, files in HEADLINE.items():
        cited: set[str] = set()
        absent_records = []
        for name in files:
            path = results_dir / name
            if not path.is_file():
                absent_records.append(name)
                continue
            cited |= referenced_checkpoints_of(path)
        gone = sorted(c for c in cited if not resolve(c).exists())
        headlines[claim] = {
            "records": list(files),
            "missing_records": absent_records,
            "checkpoints": len(cited),
            "missing_checkpoints": gone,
            "ok": not gone and not absent_records,
        }

    # A result file cites a dead path -- the count the README quotes for "a reader following an
    # older record finds nothing".
    stale_records = sorted(
        path.name
        for path in results_dir.rglob("*.json")
        if referenced_checkpoints_of(path) & set(missing)
    )
    return {
        "referenced": len(refs),
        "present": len(present),
        "missing": missing,
        "records_total": len(list(results_dir.rglob("*.json"))),
        "records_citing_missing": stale_records,
        "headlines": headlines,
    }


def referenced_checkpoints_of(path: Path) -> set[str]:
    """`referenced_checkpoints` narrowed to one file -- same regex, so the two cannot diverge."""
    return set(_CKPT_RE.findall(path.read_text()))


def format_report(report: dict, *, verbose: bool) -> str:
    lines = [
        f"{report['referenced']} checkpoint paths cited by {report['records_total']} result files",
        f"  present {report['present']}    missing {len(report['missing'])}",
        f"  {len(report['records_citing_missing'])} result files cite at least one missing path",
        "",
    ]
    for claim, h in report["headlines"].items():
        mark = "OK " if h["ok"] else "GONE"
        lines.append(f"  [{mark}] {claim}  ({h['checkpoints']} checkpoints)")
        for name in h["missing_records"]:
            lines.append(f"           missing result file: {name}")
        for rel in h["missing_checkpoints"]:
            lines.append(f"           missing checkpoint:  {rel}")

    if verbose and report["missing"]:
        by_arm: dict[str, list[str]] = defaultdict(list)
        for rel in report["missing"]:
            tail = rel.split("/", 1)[1] if "/" in rel else rel
            by_arm["(top-level warm starts)" if "/" not in tail else tail.split("/")[0]].append(rel)
        lines.append("\nMISSING, by arm:")
        for arm in sorted(by_arm):
            lines.append(f"  {arm}  ({len(by_arm[arm])})")
            for rel in sorted(by_arm[arm]):
                lines.append(f"    {rel}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="checkpoints", help="checkpoint root")
    ap.add_argument("--results", default="results", help="results root to scan for references")
    ap.add_argument("--verbose", action="store_true", help="list every missing path, by arm")
    ap.add_argument("--json", action="store_true", dest="as_json", help="machine-readable output")
    args = ap.parse_args()

    results_dir, checkpoints_dir = Path(args.results), Path(args.checkpoints)
    if not results_dir.is_dir():
        raise SystemExit(f"no results dir at {results_dir}")

    report = audit(results_dir, checkpoints_dir)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report, verbose=args.verbose))

    failed = [claim for claim, h in report["headlines"].items() if not h["ok"]]
    if failed:
        # Only a headline can fail. Missing intermediates are a documented state, not an error.
        print(f"\nFAIL: {len(failed)} headline(s) can no longer be re-read on this machine")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

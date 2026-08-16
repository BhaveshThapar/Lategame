"""Delete the PPO iteration checkpoints no published result depends on.

Build 26 attempt 1 lost four of six arms to a full disk, and the post-mortem's finding was that
the 200 GB scratch quota is **shared, not per-project**: a neighbouring project grew 86 -> 125 GB
mid-run and `torch.save` started failing outright. This tree now holds **38 GB** of checkpoints
against 25 GB free, which is the same condition that cost those arms -- so the next build starts
by giving the space back rather than by hoping.

**What is safe to delete is derived, never assumed.** Every arm directory holds one checkpoint per
iteration (~17 MB x 320 for `v26b`), but only a handful are ever read again:

  1. any path named by a committed ``results/*.json`` -- each seed's ``best_checkpoint``, the
     ``init`` warm start, the eval-ladder field. Scanned out of the raw JSON text rather than
     out of known keys, so a checkpoint referenced by a *future* schema is still protected;
  2. each arm's TERMINAL iteration, whether or not a result names it. This is what
     ``pin_gate_checkpoint.py`` re-reads to produce the selection-free second read, and Build 26's
     0.7513 cannot be reproduced without it;
  3. ``curve.json`` -- tiny, and it is the per-iteration training curve that the four-point
     dose-response (G3) is booked from;
  4. every top-level ``checkpoints/*.pt`` -- the offline-RL warm starts every arm initialises from.

Everything else is an intermediate that was written so ``argmax`` had something to range over.

DRY RUN IS THE DEFAULT. ``--apply`` is required to unlink anything, because an arm costs ~40 h of
wall-clock and there is no way to re-derive one from a curve.

    python scripts/prune_checkpoints.py                    # report only
    python scripts/prune_checkpoints.py --apply            # actually delete
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Any ``checkpoints/...pt`` occurring anywhere in a results file, whatever key holds it.
_CKPT_RE = re.compile(r"checkpoints/[\w./-]+\.pt")

#: Non-``.pt`` files an arm directory keeps regardless. ``curve.json`` is the training curve.
_KEEP_NAMES = frozenset({"curve.json"})

#: ``ppo_continue_gate`` writes ``iter_{n:02d}.pt`` -- so ``iter_01`` but ``iter_320``.
_ITER_RE = re.compile(r"^iter_(\d+)\.pt$")

#: Adam moments + both RNG streams, written by ``--resume``. Only needed to EXTEND an arm.
_RESUME_NAME = "resume_state.pt"


class PruneError(RuntimeError):
    """Raised when the retention set cannot be established -- never a reason to delete anyway."""


def referenced_checkpoints(results_dir: Path) -> set[str]:
    """Every ``checkpoints/...pt`` path mentioned by any results JSON, as written.

    Text-scanned rather than key-walked on purpose: ``best_checkpoint``, ``init``, the ladder's
    field entries and ``confirmatory_ladder`` all spell checkpoints differently, and a retention
    rule that has to enumerate the schema is one schema change away from deleting a cited file.
    """
    refs: set[str] = set()
    for path in sorted(results_dir.rglob("*.json")):
        refs.update(_CKPT_RE.findall(path.read_text()))
    return refs


def terminal_iters(results_dir: Path) -> dict[str, int]:
    """``out_dir`` -> the arm's terminal iteration, read off the gate that produced it.

    The gate's ``iters`` is authoritative over what happens to be on disk: an arm killed at 304 of
    320 has a highest-numbered file that is *not* its terminal, and pinning to it would silently
    produce a different read than the published one.
    """
    terminal: dict[str, int] = {}
    for path in sorted(results_dir.rglob("*.json")):
        try:
            gate = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(gate, dict) or "iters" not in gate:
            continue
        iters = gate["iters"]
        if not isinstance(iters, int):
            continue
        for record in gate.get("records", []):
            out_dir = record.get("out_dir")
            if isinstance(out_dir, str):
                # Max, not overwrite: a chunked arm (v26b ran 160 -> resume -> 320) leaves an
                # intermediate gate file on disk whose `iters` is the CHUNK length, not the arm's.
                terminal[out_dir] = max(terminal.get(out_dir, 0), iters)
    return terminal


@dataclass
class ArmPlan:
    """One ``checkpoints/ppo_*_s*/`` directory's verdict."""

    out_dir: str
    keep: list[str] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    delete_bytes: int = 0


@dataclass
class PrunePlan:
    arms: list[ArmPlan] = field(default_factory=list)

    @property
    def delete_bytes(self) -> int:
        return sum(a.delete_bytes for a in self.arms)

    @property
    def delete_count(self) -> int:
        return sum(len(a.delete) for a in self.arms)

    def paths(self) -> list[str]:
        return [p for arm in self.arms for p in arm.delete]


def plan_prune(
    checkpoints_dir: Path,
    referenced: set[str],
    terminal: dict[str, int],
    *,
    keep_resume: bool = True,
) -> PrunePlan:
    """Decide, per arm directory, which files are intermediates.

    Top-level ``checkpoints/*.pt`` are never candidates: those are the offline-RL warm starts, and
    an arm's ``init`` is one of them.
    """
    plan = PrunePlan()
    for arm_dir in sorted(p for p in checkpoints_dir.iterdir() if p.is_dir()):
        out_dir = f"{checkpoints_dir.name}/{arm_dir.name}"
        arm = ArmPlan(out_dir=out_dir)

        on_disk = {p.name: p for p in arm_dir.iterdir() if p.is_file()}
        iters_present = {
            int(m.group(1)): name for name in on_disk if (m := _ITER_RE.match(name)) is not None
        }

        pinned = terminal.get(out_dir)
        if pinned is None and iters_present:
            # No gate claims this arm (pre-Build-22 runs). Its highest iteration IS its terminal.
            pinned = max(iters_present)
        if pinned is not None and pinned not in iters_present and iters_present:
            raise PruneError(
                f"{out_dir}: gate names terminal iteration {pinned}, which is not on disk "
                f"(present: {min(iters_present)}..{max(iters_present)}). Refusing to prune this "
                f"arm -- the terminal read could not be reproduced afterwards."
            )

        for name, path in sorted(on_disk.items()):
            rel = f"{out_dir}/{name}"
            if name == _RESUME_NAME:
                # Checked BEFORE the unknown-file fallback below, which would otherwise keep it
                # unconditionally and make --drop-resume silently do nothing.
                keep = keep_resume
            else:
                keep = (
                    rel in referenced
                    or name in _KEEP_NAMES
                    or (pinned is not None and iters_present.get(pinned) == name)
                    or _ITER_RE.match(name) is None  # unknown file: keep, don't guess
                )
            if keep:
                arm.keep.append(rel)
            else:
                arm.delete.append(rel)
                arm.delete_bytes += path.stat().st_size

        plan.arms.append(arm)
    return plan


def format_plan(plan: PrunePlan) -> str:
    lines = [f"{'arm':<24} {'keep':>6} {'delete':>7} {'frees':>10}"]
    for arm in plan.arms:
        if not arm.delete:
            continue
        lines.append(
            f"{arm.out_dir.split('/')[-1]:<24} {len(arm.keep):>6} {len(arm.delete):>7} "
            f"{arm.delete_bytes / 2**30:>9.2f}G"
        )
    lines.append(
        f"{'TOTAL':<24} {'':>6} {plan.delete_count:>7} {plan.delete_bytes / 2**30:>9.2f}G"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="checkpoints", help="checkpoint root")
    ap.add_argument("--results", default="results", help="results root to scan for references")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--drop-resume", action="store_true",
                    help="also delete resume_state.pt (~36 MB/arm; only needed to extend an arm)")
    ap.add_argument("--show-keep", action="store_true", help="list every retained checkpoint")
    args = ap.parse_args()

    checkpoints_dir, results_dir = Path(args.checkpoints), Path(args.results)
    if not checkpoints_dir.is_dir():
        raise SystemExit(f"no checkpoint dir at {checkpoints_dir}")

    referenced = referenced_checkpoints(results_dir)
    terminal = terminal_iters(results_dir)
    print(f"{len(referenced)} checkpoint paths referenced by {results_dir}/**.json")
    print(f"{len(terminal)} arms with a gate-declared terminal iteration\n")

    plan = plan_prune(
        checkpoints_dir, referenced, terminal, keep_resume=not args.drop_resume
    )
    print(format_plan(plan))

    if args.show_keep:
        print("\nRETAINED:")
        for arm in plan.arms:
            for rel in arm.keep:
                print(f"  {rel}")

    if not args.apply:
        print("\nDRY RUN -- nothing deleted. Re-run with --apply once the list looks right.")
        return

    for rel in plan.paths():
        Path(rel).unlink()
    print(f"\ndeleted {plan.delete_count} files, freed {plan.delete_bytes / 2**30:.2f} GiB")


if __name__ == "__main__":
    # `main` returns None -- every failure path already raises SystemExit itself, so wrapping the
    # call in `sys.exit` only ever passed None (exit 0) and hid that from the type checker.
    main()

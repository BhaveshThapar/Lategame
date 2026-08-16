"""Which weights a release should carry, and proof that a downloaded one is the right file.

A clone of this repository contains no weights: `checkpoints/` is gitignored, so every published
number was produced from files that exist only on the machine that ran them. The evidence ships and
the artifacts do not, which means a reader can re-derive a result in ~40 h of wall-clock but cannot
*run* the agent. Attaching a handful of checkpoints to the GitHub release closes that, and this
script is the half that can live in git.

TWO JOBS, and the second is the one that matters.

  `--manifest` derives the asset set from `check_artifacts.HEADLINE` rather than from a hand-kept
  list, so it cannot drift from the claims it backs: every headline's records are scanned for
  `checkpoints/*.pt` paths, and the ones a user needs to actually play a format are marked
  `runnable`. Writes `results/release_assets.json`.

  `--verify` re-hashes local files against that manifest. This is what makes a download checkable:
  a checkpoint is a pickle, `torch.load` runs code, and "trust the file on the release page" is not
  a security posture. sha256 in a committed, signed-tag-adjacent file is.

    python scripts/release_assets.py --manifest      # write results/release_assets.json
    python scripts/release_assets.py --verify        # re-hash and compare
    python scripts/release_assets.py --upload-hint   # print the gh/UI steps, no network

WHAT IS DELIBERATELY NOT HERE. No upload. There is no `gh` on the machine this was written on and
no API token, and a script that shells out to an uploader nobody can run is worse than a printed
instruction. The manifest is the contract; attaching the files is a two-minute manual step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_artifacts import HEADLINE  # noqa: E402

_RESULTS = Path("results")
_MANIFEST = _RESULTS / "release_assets.json"
_CKPT_RE = re.compile(r"checkpoints/[\w./-]+\.pt")

#: The minimum set that lets someone PLAY each format, as opposed to merely re-check a number.
#: Everything else a headline cites is provenance for a curve and is not needed to run anything.
RUNNABLE: dict[str, str] = {
    "checkpoints/ppo_v26b_s0/iter_320.pt": (
        "gen9ou -- the ladder top (Glicko 1776.3 / GXE 0.7434) and the strongest agent here"
    ),
    "checkpoints/doubles_offrl_vgc_v2.pt": (
        "gen9vgc2025regi -- the AWR doubles policy, and the only viable self-play warm start "
        "(the BC one carries no value support)"
    ),
    "checkpoints/doubles_bc_vgc_v2.pt": (
        "gen9vgc2025regi -- the BC doubles policy, the other half of the corrected VGC ladder"
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cited_by_headlines() -> dict[str, list[str]]:
    """checkpoint path -> the headline claims that cite it, derived not hand-listed."""
    out: dict[str, list[str]] = {}
    for claim, records in HEADLINE.items():
        for name in records:
            p = _RESULTS / name
            if not p.exists():
                continue
            for hit in sorted(set(_CKPT_RE.findall(p.read_text()))):
                out.setdefault(hit, [])
                if claim not in out[hit]:
                    out[hit].append(claim)
    return out


def build_manifest() -> dict[str, Any]:
    cited = _cited_by_headlines()
    assets, missing = [], []
    for path_str in sorted(cited):
        p = Path(path_str)
        if not p.exists():
            missing.append(path_str)
            continue
        assets.append({
            "path": path_str,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
            "backs": cited[path_str],
            "runnable": RUNNABLE.get(path_str),
        })
    runnable = [a for a in assets if a["runnable"]]
    return {
        "schema": "lategame.release_assets/1",
        "note": (
            "Derived from check_artifacts.HEADLINE, not hand-kept, so it cannot drift from the "
            "claims it backs. `runnable` marks the minimum set needed to PLAY a format; the rest "
            "is provenance for a curve."
        ),
        "verify": "python scripts/release_assets.py --verify",
        "n_assets": len(assets),
        "n_runnable": len(runnable),
        "runnable_bytes": sum(a["bytes"] for a in runnable),
        "total_bytes": sum(a["bytes"] for a in assets),
        "missing_locally": missing,
        "assets": assets,
    }


def verify(manifest: dict[str, Any]) -> int:
    bad = 0
    for a in manifest["assets"]:
        p = Path(a["path"])
        if not p.exists():
            print(f"  MISSING  {a['path']}")
            bad += 1
            continue
        actual = _sha256(p)
        ok = actual == a["sha256"] and p.stat().st_size == a["bytes"]
        print(f"  {'OK      ' if ok else 'MISMATCH'} {a['path']}")
        if not ok:
            print(f"           expected {a['sha256']}\n           actual   {actual}")
            bad += 1
    print(f"\n{len(manifest['assets']) - bad}/{len(manifest['assets'])} verified")
    return bad


def _upload_hint(manifest: dict[str, Any]) -> None:
    runnable = [a for a in manifest["assets"] if a["runnable"]]
    mb = manifest["runnable_bytes"] / 2**20
    print(f"\nAttach these {len(runnable)} files to the v1.0.0 release ({mb:.1f} MiB total):\n")
    for a in runnable:
        print(f"  {a['path']}   ({a['bytes'] / 2**20:.1f} MiB)")
        print(f"      {a['runnable']}")
    print(
        "\n  gh release upload v1.0.0 " + " ".join(a["path"] for a in runnable)
        + "\n\nNo gh on this host: use Releases -> Draft a new release -> pick the v1.0.0 tag ->"
        "\ndrag the files in. `results/release_assets.json` carries the sha256 of each, so a"
        "\ndownloader can check what they got before torch.load runs code from it."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="store_true", help="write results/release_assets.json")
    ap.add_argument("--verify", action="store_true", help="re-hash local files against it")
    ap.add_argument("--upload-hint", action="store_true", help="print the attach steps")
    args = ap.parse_args()

    if args.manifest:
        m = build_manifest()
        _MANIFEST.write_text(json.dumps(m, indent=2) + "\n")
        print(f"{m['n_assets']} assets ({m['total_bytes'] / 2**20:.1f} MiB), "
              f"{m['n_runnable']} runnable ({m['runnable_bytes'] / 2**20:.1f} MiB)")
        if m["missing_locally"]:
            print(f"  !! {len(m['missing_locally'])} cited paths absent locally")
        print(f"  -> wrote {_MANIFEST}")
        return

    if not _MANIFEST.exists():
        raise SystemExit(f"no manifest at {_MANIFEST}; run --manifest first")
    m = json.loads(_MANIFEST.read_text())
    if args.upload_hint:
        _upload_hint(m)
        return
    raise SystemExit(verify(m) and 1 or 0)


if __name__ == "__main__":
    main()

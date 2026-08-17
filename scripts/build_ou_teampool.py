"""Gate A for R-TEAM: pack + legality-validate the curated OU team pool.

Reads Showdown-paste teams from a ``===``-delimited source file, packs each via poke-env,
and validates it against the bundled Showdown ``validate-team`` CLI. Only teams the server
accepts are written to the runtime pool. An illegal team silently breaks battles (the server
rejects the challenge), so this is a loud kill-gate: it exits non-zero if too few teams pass.

    python scripts/build_ou_teampool.py                    # gen9ou, default paths
    python scripts/build_ou_teampool.py --format gen9ou --min 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rotomai.teambuilding.pool import pack_showdown_team

_SHOWDOWN = Path("third_party/pokemon-showdown/pokemon-showdown")
_DELIM = "==="


def _split_teams(src: str) -> list[tuple[str, str]]:
    """Split a source file into (label, paste) blocks on lines starting with ``===``."""
    blocks: list[tuple[str, str]] = []
    label = "team"
    lines: list[str] = []
    for line in src.splitlines():
        if line.startswith(_DELIM):
            if lines and any(x.strip() for x in lines):
                blocks.append((label, "\n".join(lines).strip()))
            label = line.strip(f" {_DELIM}") or f"team{len(blocks) + 1}"
            lines = []
        else:
            lines.append(line)
    if lines and any(x.strip() for x in lines):
        blocks.append((label, "\n".join(lines).strip()))
    return blocks


def _validate(packed: str, battle_format: str) -> tuple[bool, str]:
    """Run the Showdown validator on a packed team; return (legal, message)."""
    proc = subprocess.run(
        [str(_SHOWDOWN), "validate-team", battle_format],
        input=packed,
        capture_output=True,
        text=True,
    )
    msg = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", dest="battle_format", default="gen9ou")
    ap.add_argument("--src", default="rotomai/teambuilding/data/teams_gen9ou_src.txt")
    ap.add_argument("--out", default="rotomai/teambuilding/data/teams_gen9ou.packed")
    ap.add_argument("--min", type=int, default=8, help="fail if fewer than this many pass")
    args = ap.parse_args()

    if not _SHOWDOWN.exists():
        sys.exit(f"Showdown CLI not found at {_SHOWDOWN}")
    src_path = Path(args.src)
    if not src_path.exists():
        sys.exit(f"source teams file not found: {src_path}")

    blocks = _split_teams(src_path.read_text())
    print(f"[build-pool] {len(blocks)} candidate team(s) from {src_path} ({args.battle_format})")

    passed: list[str] = []
    for i, (label, paste) in enumerate(blocks, 1):
        try:
            packed = pack_showdown_team(paste)
        except Exception as exc:  # packing errors are as fatal as validation errors here
            print(f"  [{i:>2}] FAIL {label}: pack error: {exc}")
            continue
        legal, msg = _validate(packed, args.battle_format)
        if legal:
            passed.append(packed)
            print(f"  [{i:>2}] PASS {label}")
        else:
            print(f"  [{i:>2}] FAIL {label}: {msg}")

    print(f"[build-pool] {len(passed)}/{len(blocks)} legal")
    if len(passed) < args.min:
        sys.exit(f"only {len(passed)} legal teams (< min {args.min}) -- fix the source and rerun")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(passed) + "\n")
    print(f"  -> wrote {out_path} ({len(passed)} teams)")


if __name__ == "__main__":
    main()

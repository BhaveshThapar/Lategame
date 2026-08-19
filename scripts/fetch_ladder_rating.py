"""Read an account's FINAL ladder rating from Showdown's public user endpoint.

poke-env cannot supply this. `abstract_battle` parses `int(rating_info[:4])` off the `|raw|` line,
which is the rating BEFORE that game, so the last battle in a session carries the rating going into
it and never the one coming out. A ladder claim therefore cannot be read from the session file at
all -- it has to come from the account.

    https://pokemonshowdown.com/users/<userid>.json
      -> {"ratings": {"gen9ou": {"elo":.., "gxe":.., "rpr":.., "rprd":.., "w":.., "l":..}}}

`elo` is Showdown's sequential ladder Elo, centred at 1000 for a new account. `rpr`/`rprd` are the
Glicko pair, centred at 1500. They describe the same games on DIFFERENT SCALES and must not be
compared to each other -- mixing them is what made a 7-18 session report GXE 0.0242 where the
account page said 30% (see rotomai/eval/rating.elo_mle).

Network failure is not fatal here on purpose: this is usually run right after a session that just
took hours, and dying because a CDN blipped would be the worst possible moment. It reports and
exits non-zero instead, so the caller can retry without losing anything.

    python scripts/fetch_ladder_rating.py --username RotomLover12 --format gen9ou \
      --out results/live_ladder_gen9ou_rating.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

USER_URL = "https://pokemonshowdown.com/users/{userid}.json"
UA = "rotomai (github.com/BhaveshThapar/RotomAI)"


def to_id(name: str) -> str:
    """Showdown's `toID`: lowercase, strip everything non-alphanumeric."""
    return "".join(c for c in name.lower() if c.isalnum())


def fetch(username: str, *, timeout: float = 20.0) -> dict[str, Any]:
    url = USER_URL.format(userid=to_id(username))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    # The users endpoint is plain JSON; `action.php?act=ladderget` prefixes a `]` guard, so strip
    # one defensively rather than assuming which endpoint a future caller points here.
    if body.startswith("]"):
        body = body[1:]
    return json.loads(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username", required=True)
    ap.add_argument("--format", default="gen9ou", dest="battle_format")
    ap.add_argument("--out", default=None, help="write the format's rating block here")
    ap.add_argument("--raw-out", default=None, help="also write the whole users.json payload")
    args = ap.parse_args()

    try:
        payload = fetch(args.username)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"could not read the ladder rating: {exc}", file=sys.stderr)
        print("nothing was written; rerun when the network is back.", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.raw_out:
        Path(args.raw_out).write_text(json.dumps(payload, indent=2) + "\n")

    block = payload.get("ratings", {}).get(args.battle_format)
    if block is None:
        print(f"{args.username!r} has no {args.battle_format} ladder history yet. The endpoint "
              f"can lag a few minutes behind the last game.", file=sys.stderr)
        raise SystemExit(1)

    block = dict(block)
    block["userid"] = payload.get("userid")
    block["battle_format"] = args.battle_format
    block["source"] = USER_URL.format(userid=to_id(args.username))
    block["scale_note"] = (
        "`elo` is Showdown's sequential ladder Elo (new accounts start at 1000). `rpr`/`rprd` are "
        "the Glicko pair (centred at 1500). Same games, different scales -- do not compare them."
    )

    if args.out:
        Path(args.out).write_text(json.dumps(block, indent=2) + "\n")
        print(f"wrote {args.out}")
    games = (block.get("w") or 0) + (block.get("l") or 0) + (block.get("t") or 0)
    print(f"{payload.get('userid')} {args.battle_format}: elo {block.get('elo')}  "
          f"gxe {block.get('gxe')}  rpr {block.get('rpr')} +/- {block.get('rprd')}  "
          f"W/L {block.get('w')}/{block.get('l')}  ({games} games)")


if __name__ == "__main__":
    main()

"""The operator contract for live play: credentials, and the ranked-ladder gate.

This is the one file to read before pointing the agent at a public server. It is deliberately
DEPENDENCY-FREE -- no poke-env, no torch, nothing from the rest of the package -- because
``lategame/cli.py`` imports it at module scope to build ``--help`` text, and paying a torch import
for ``lategame --help`` would be absurd.

WHY THE LADDER IS GATED. plan.md NG3 puts "automated farming of the public *ranked* ladder" out of
scope, and section 15 says to default to challenge/unranked play. The awkward part is that this
CANNOT be satisfied by choosing a gentler API: ``/search <format>`` on the public server IS the
rated ladder, and Showdown offers no unranked equivalent. So ladder mode cannot be made
policy-safe, only policy-EXPLICIT -- which is what the double gate below buys.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Bot-account credentials. The password is read from the environment ONLY -- never a CLI flag, so
#: it cannot leak into ``ps`` output or shell history.
USERNAME_ENV = "LATEGAME_PS_USERNAME"
PASSWORD_ENV = "LATEGAME_PS_PASSWORD"

#: The second of the two ladder channels (the first is ``--ladder-ack``).
ALLOW_LADDER_ENV = "LATEGAME_LIVE_ALLOW_LADDER"

#: The exact phrase ``--ladder-ack`` must carry. Passed to argparse as ``choices=[LADDER_ACK]`` so a
#: typo is a parse error rather than a silent fallback to a non-ladder mode.
LADDER_ACK = "i-have-read-plan-md-section-15"

#: Printed to stderr before the first ladder search, and recorded verbatim in the results JSON so a
#: published ladder number carries its own provenance. Quoted from plan.md sections 15 and 3.2.
POLICY_NOTE = """\
RANKED LADDER MODE -- plan.md NG3 / section 15
  NG3: automated farming of the public *ranked* ladder is OUT OF SCOPE for v1.
  Section 15: operate only under a DEDICATED BOT ACCOUNT, never a primary/identity account;
  Showdown's rules prohibit "gaming the system", with moderators exercising discretion; the
  research community discourages botting the human ranked ladder and runs separate agent-only
  servers to avoid disrupting humans. Default to challenge/unranked play and a private eval server.
  There is no unranked ladder: `/search` on the public server IS the rated ladder, so this mode
  cannot be made unranked -- only deliberate. Prefer --mode accept, or --server <private-eval>.\
"""


class PolicyError(RuntimeError):
    """Raised when ladder mode is requested without both opt-in channels."""


def check_ladder_optin(ack: str | None, env: Mapping[str, str] | None = None) -> str:
    """Authorise ranked-ladder play, or raise ``PolicyError``.

    TWO INDEPENDENT CHANNELS are required, and that redundancy is the point: a CLI flag cannot be
    inherited from an exported shell variable, and an environment variable cannot be picked up from
    shell history or a copy-pasted command. Requiring both means neither a stale export nor a
    recalled command can start ranked play on its own.

    Returns the acknowledgement string so the caller can record it in the run's results JSON.
    """
    if env is None:
        import os

        env = os.environ

    if ack != LADDER_ACK:
        raise PolicyError(
            "ladder mode requires --ladder-ack "
            f"{LADDER_ACK!r} (got {ack!r}).\n\n{POLICY_NOTE}"
        )
    if env.get(ALLOW_LADDER_ENV, "").strip() not in {"1", "true", "TRUE", "yes"}:
        raise PolicyError(
            f"ladder mode also requires {ALLOW_LADDER_ENV}=1 in the environment. Both channels are "
            "required so that neither a stale export nor a recalled command can start ranked play "
            f"on its own.\n\n{POLICY_NOTE}"
        )
    return LADDER_ACK

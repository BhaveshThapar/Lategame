"""Forward-model fidelity tests (Lever 11 / R-PREDICT, Gate A).

The forward model R-PREDICT search is built on -- serialize a battle, deserialize a fork,
step the fork -- must reproduce, bit-for-bit, what stepping the battle directly produces
(the serialized PRNG seed makes a faithful fork an *exact* match). This guards that
primitive: it runs the real node driver over a known gen9 inputlog and asserts a perfect
match rate. Environment-gated exactly like ``test_resim``'s end-to-end test, since the
vendored simulator (``third_party/``) is not committed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lategame.search.fidelity import _DEFAULT_SHOWDOWN, run_fidelity

# A real gen9randombattle inputlog (PRNG seed + every choice); ``>forcelose`` ends it,
# which the fidelity driver ignores -- it scores only the clean decision turns.
_INPUTLOG = "\n".join([
    '>start {"formatid":"gen9randombattle",'
    '"seed":"sodium,b8493732a42936a1fd687ddf7988dbbc","rated":"Rated battle"}',
    '>player p1 {"name":"gummiworm","seed":"sodium,a3572989c38af097ba39a7fa38627767"}',
    '>player p2 {"name":"Qwuartz","seed":"sodium,d9f6a625ddf93230a09fa7bc6f0239d0"}',
    ">p1 move psyblade", ">p2 switch 3", ">p1 move psyblade", ">p2 move fireblast",
    ">p2 switch 4", ">p1 switch 5", ">p2 move tripleaxel", ">p1 move raindance",
    ">p2 switch 3", ">p1 move liquidation", ">p2 move thunderbolt", ">p1 move closecombat",
    ">p2 move taunt", ">p1 switch 3", ">p2 switch 3", ">p1 move seedflare",
    ">p2 move tripleaxel", ">p1 move airslash", ">p2 move knockoff", ">p1 switch 6",
    ">p1 move calmmind", ">p2 switch 5", ">p1 move moonblast", ">p2 move dragondance",
    ">p1 switch 5", ">p2 move earthquake", ">p1 move psyblade", ">p2 move stoneedge",
    ">p1 switch 4", ">p1 move shoreup terastallize", ">p2 move dragondance", ">forcelose p1",
])

_HAS_NODE = shutil.which("node") is not None
_HAS_SIM = (Path(_DEFAULT_SHOWDOWN) / "dist").is_dir()


@pytest.mark.skipif(not (_HAS_NODE and _HAS_SIM), reason="needs node + a built vendored simulator")
def test_forward_model_fork_is_exact() -> None:
    stats = run_fidelity([{"id": "sample", "inputlog": _INPUTLOG}])
    assert stats.replays == 1 and stats.errored == 0
    assert stats.transitions > 10  # a full battle's worth of decision rounds
    assert stats.drive_errors == 0
    # A faithful serialize/deserialize/step fork is an EXACT match -- core and full.
    assert stats.core_mismatch == 0 and stats.full_mismatch == 0
    assert stats.core_match_rate == 1.0 and stats.full_match_rate == 1.0

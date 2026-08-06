"""Forward-model fidelity tests (Lever 11 / R-PREDICT, Gate A).

The forward model R-PREDICT search is built on -- serialize a battle, deserialize a fork,
step the fork -- must reproduce, bit-for-bit, what stepping the battle directly produces
(the serialized PRNG seed makes a faithful fork an *exact* match). This guards that
primitive: it runs the real node driver over a known gen9 inputlog and asserts a perfect
match rate. Environment-gated exactly like ``test_resim``'s end-to-end test, since the
vendored simulator (``third_party/``) is not committed.
"""

from __future__ import annotations

from lategame.search.fidelity import run_fidelity
from tests.conftest import GEN9_INPUTLOG, requires_showdown


@requires_showdown
def test_forward_model_fork_is_exact() -> None:
    stats = run_fidelity([{"id": "sample", "inputlog": GEN9_INPUTLOG}])
    assert stats.replays == 1 and stats.errored == 0
    # Checked BEFORE the transition count: a drive error means the simulator REJECTED a recorded
    # choice, which is what a stale fixture looks like (see tests/conftest.py). Asserting the
    # count first reports that as a bare `assert 4 > 10` and buries the cause.
    assert stats.drive_errors == 0
    assert stats.transitions > 10  # a full battle's worth of decision rounds
    # A faithful serialize/deserialize/step fork is an EXACT match -- core and full.
    assert stats.core_mismatch == 0 and stats.full_mismatch == 0
    assert stats.core_match_rate == 1.0 and stats.full_match_rate == 1.0

"""Unit tests for the Build-5 slot-order gate's pure logic (no replays, no server).

Covers the packed-teampool move parser (declaration order preserved), the canonical
ordering rule, the displacement metric, and the per-decision slot probe's classification
(scramble / agreement / out-of-4 truncation). The replay sweep is exercised by the gated run.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "slot_order_gate.py"
_spec = importlib.util.spec_from_file_location("slot_order_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

_PACKED = (
    "Great Tusk||leftovers|protosynthesis|headlongrush,closecombat,rapidspin,stealthrock"
    "|Jolly|,252,4,,,252|||||,,,,,Steel"
    "]Kingambit||leftovers|supremeoverlord|suckerpunch,ironhead,swordsdance,lowkick"
    "|Adamant|252,252,,,4,|||||,,,,,Dark"
)


def _battle(move_ids: list[str]) -> SimpleNamespace:
    moves = {mid: SimpleNamespace(id=mid) for mid in move_ids}
    return SimpleNamespace(active_pokemon=SimpleNamespace(moves=moves))


def _parts(move: str) -> list[str]:
    return ["", "move", "p1a: Mon", move]


def test_packed_move_lists_preserves_declaration_order():
    mons = gate.packed_move_lists(_PACKED + "\n\n" + _PACKED)
    assert len(mons) == 4  # two teams x two mons, blank line ignored
    assert mons[0] == ["headlongrush", "closecombat", "rapidspin", "stealthrock"]
    assert mons[1] == ["suckerpunch", "ironhead", "swordsdance", "lowkick"]
    assert mons[0] != sorted(mons[0])  # the fixture is genuinely non-canonical


def test_canonical_move_ids_sorts_and_truncates():
    assert gate.canonical_move_ids(["c", "a", "b"]) == ["a", "b", "c"]
    assert gate.canonical_move_ids(["e", "d", "c", "b", "a"]) == ["a", "b", "c", "d"]


def test_mean_displacement():
    assert gate.mean_displacement(["a", "b", "c", "d"]) == 0.0
    assert gate.mean_displacement(["d", "c", "b", "a"]) == 2.0
    assert gate.mean_displacement([]) == 0.0


def test_probe_counts_scramble_and_agreement():
    probe = gate._SlotProbe()
    # Insertion c,a,b -> canonical a,b,c: used "c" sits at slot 0 vs 2 -> scrambled.
    probe.classify(_battle(["c", "a", "b"]), _parts("c"), kept=True)
    # Already-canonical insertion -> agreement.
    probe.classify(_battle(["a", "b"]), _parts("a"), kept=True)
    # Unresolvable move (not in the mon's set) is not a decision.
    probe.classify(_battle(["a", "b"]), _parts("z"), kept=False)
    rates = probe.rates()
    assert rates["n_move_decisions"] == 2.0
    assert rates["move_label_scramble_rate"] == 0.5
    assert rates["mon_order_scramble_rate"] == 0.5
    assert rates["out_of_insertion4_rate"] == 0.0


def test_probe_out_of_four_truncation():
    probe = gate._SlotProbe()
    # Five known moves: insertion keeps e,d,c,b; canonical keeps a,b,c,d.
    ids = ["e", "d", "c", "b", "a"]
    probe.classify(_battle(ids), _parts("a"), kept=False)  # beyond insertion[:4] only
    probe.classify(_battle(ids), _parts("e"), kept=True)  # beyond canonical[:4] only
    probe.classify(_battle(ids), _parts("c"), kept=True)  # in both, slot 2 -> 2: no scramble
    rates = probe.rates()
    assert rates["n_move_decisions"] == 3.0
    assert rates["out_of_insertion4_rate"] == 1 / 3
    assert rates["out_of_canonical4_rate"] == 1 / 3
    assert rates["move_label_scramble_rate"] == 0.0
    assert rates["n_kept"] == 2.0

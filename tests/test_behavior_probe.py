"""Unit tests for the Build-6 behavioral probe's pure logic (no server, no torch).

Covers the action-histogram metrics, switch-run/ping-pong detectors (incl. the
forced-switch exclusion), order matching (exact / permuted / unstable / forme-suffix),
the pre-registered classifier with its a > b > c > d precedence, the level-25 rejection
handler, the fallback spy's delegation, and the transcript renderer. The live sweep is
exercised by the gated run.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

_PROBE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "behavior_probe.py"
_spec = importlib.util.spec_from_file_location("behavior_probe", _PROBE_PATH)
assert _spec is not None and _spec.loader is not None
probe_mod = importlib.util.module_from_spec(_spec)
sys.modules["behavior_probe"] = probe_mod  # dataclasses resolves annotations via sys.modules
_spec.loader.exec_module(probe_mod)

_TEAM = ["glimmora", "greattusk", "kingambit", "gholdengo", "dragapult", "zamazenta"]


def _dec(**kw):
    base = dict(
        arm="bc",
        battle_tag="b1",
        turn=1,
        forced=False,
        action=6,
        order_str="/choose move stealthrock",
        fallback=False,
        legal_count=9,
        mask_all_false=False,
        top_k=[(6, 0.62), (7, 0.21)],
        our_active="glimmora",
        our_hp=1.0,
        opp_active="greattusk",
        opp_hp=1.0,
        team_order=list(_TEAM),
    )
    base.update(kw)
    return probe_mod.Decision(**base)


def _metrics(**kw):
    base = dict(
        n_decisions=600,
        n_raw_calls=600,
        n_battles=20,
        n_finished=20,
        win_rate=0.05,
        mean_turns=30.0,
        fallback_rate=0.0,
        rejection_rate=0.0,
        n_invalid=0,
        n_unavailable=0,
        all_false_masks=0,
        switch_fraction=0.2,
        voluntary_switch_fraction=0.2,
        max_switch_run=2,
        ping_pong_rate=0.0,
        top_action_share=0.3,
        entropy_bits=2.5,
        mean_top1_prob=0.5,
        action_hist={},
    )
    base.update(kw)
    return base


def _order(**kw):
    base = dict(
        n_battles_checked=20,
        n_packed_missing=0,
        match_rate=1.0,
        stable_rate=1.0,
        mismatched_battles=[],
        mismatch_examples=[],
        unstable_battles=[],
    )
    base.update(kw)
    return base


def test_action_name():
    assert probe_mod._action_name(0) == "switch0"
    assert probe_mod._action_name(5) == "switch5"
    assert probe_mod._action_name(6) == "move0"
    assert probe_mod._action_name(9) == "move3"
    assert probe_mod._action_name(10) == "mega0"
    assert probe_mod._action_name(25) == "tera3"
    assert probe_mod._action_name(-2) == "default"


def test_entropy_bits():
    assert probe_mod._entropy_bits({6: 5, 7: 5, 8: 5, 9: 5}) == 2.0
    assert probe_mod._entropy_bits({6: 10}) == 0.0
    assert probe_mod._entropy_bits({}) == 0.0


def test_switch_run_counts_voluntary_only():
    decs = [
        _dec(action=0),
        _dec(action=1),
        _dec(action=2, forced=True),  # forced: neither extends nor breaks the run
        _dec(action=3),
        _dec(action=6),  # a move ends the run
        _dec(action=4),
    ]
    assert probe_mod._max_switch_run(decs) == 3


def test_switch_run_resets_across_battles():
    decs = [
        _dec(action=0, battle_tag="b1"),
        _dec(action=1, battle_tag="b1"),
        _dec(action=0, battle_tag="b2"),
    ]
    assert probe_mod._max_switch_run(decs) == 2


def test_ping_pong_rate():
    # Switch targets glimmora -> greattusk -> glimmora: the third bounces back.
    decs = [_dec(action=0), _dec(action=1), _dec(action=0), _dec(action=6)]
    assert probe_mod._ping_pong_rate(decs) == 1 / 3
    # Forced switches don't count toward ping-pong.
    assert probe_mod._ping_pong_rate([_dec(action=0, forced=True)] * 3) == 0.0


def test_decision_metrics_rates():
    decs = [
        _dec(action=6, repeat=3),  # collapsed rejection retries
        _dec(action=0, fallback=True, turn=2),
        _dec(action=6, mask_all_false=True, turn=3),
        _dec(action=1, forced=True, turn=4),
    ]
    rejections = [{"kind": "invalid"}, {"kind": "invalid"}, {"kind": "unavailable"}]
    battles = [
        {"battle": "b1", "won": True, "turns": 30},
        {"battle": "b2", "won": False, "turns": 20},
        {"battle": "b3", "won": None, "turns": 5},  # unfinished: excluded from win rate
    ]
    m = probe_mod._decision_metrics(decs, rejections, battles)
    assert m["n_decisions"] == 4
    assert m["n_raw_calls"] == 6
    assert m["fallback_rate"] == 0.25
    assert m["rejection_rate"] == 0.75
    assert m["n_invalid"] == 2
    assert m["n_unavailable"] == 1
    assert m["all_false_masks"] == 1
    assert m["switch_fraction"] == 0.5
    assert m["voluntary_switch_fraction"] == 1 / 3  # forced switch excluded
    assert m["win_rate"] == 0.5
    assert m["n_finished"] == 2
    assert m["mean_turns"] == 55 / 3
    assert m["top_action_share"] == 0.5  # action 6 twice out of 4
    assert m["action_hist"]["move0"] == 2


def test_decision_metrics_empty():
    m = probe_mod._decision_metrics([], [], [])
    assert m["n_decisions"] == 0
    assert m["win_rate"] is None
    assert m["fallback_rate"] == 0.0


def test_order_metrics_exact_permuted_and_missing():
    packed = {"b1": ["a", "b"], "b2": ["a", "b"]}
    live = {"b1": ["a", "b"], "b2": ["b", "a"], "b3": ["a", "b"]}
    o = probe_mod._order_metrics(packed, live, set())
    assert o["n_battles_checked"] == 2
    assert o["n_packed_missing"] == 1  # b3 has no packed record
    assert o["match_rate"] == 0.5
    assert o["mismatched_battles"] == ["b2"]
    assert o["mismatch_examples"][0]["battle"] == "b2"
    assert o["stable_rate"] == 1.0


def test_order_metrics_forme_suffix_and_instability():
    packed = {"b1": ["zacian", "ogerpon"]}
    live = {"b1": ["zaciancrowned", "ogerponwellspring"]}
    o = probe_mod._order_metrics(packed, live, {"b1"})
    assert o["match_rate"] == 1.0  # forme suffixes match the base species
    assert o["stable_rate"] == 0.0
    assert o["unstable_battles"] == ["b1"]


def test_classify_each_cause_and_precedence():
    clean = probe_mod._classify(_metrics(), _order(), None)
    assert clean["primary_cause"] == "d_drift"  # win 0.05 with a/b/c clean

    a = probe_mod._classify(_metrics(fallback_rate=0.5), _order(), None)
    assert a["primary_cause"] == "a_decode_mask"

    b = probe_mod._classify(_metrics(), _order(match_rate=0.9), None)
    assert b["primary_cause"] == "b_team_order"
    unstable = probe_mod._classify(_metrics(), _order(stable_rate=0.95), None)
    assert unstable["primary_cause"] == "b_team_order"

    c = probe_mod._classify(_metrics(max_switch_run=7), _order(), None)
    assert c["primary_cause"] == "c_pathological"
    assert c["c_flags"]["switch_run"]

    # Precedence: a-symptoms suppress c as the primary cause.
    both = probe_mod._classify(_metrics(fallback_rate=0.5, max_switch_run=7), _order(), None)
    assert both["primary_cause"] == "a_decode_mask"
    assert not both["causes"]["c_pathological"]

    ok = probe_mod._classify(_metrics(win_rate=0.5), _order(), None)
    assert ok["primary_cause"] == "NO-REPRO"


def test_classify_switch_bar_scales_with_train_base():
    m = _metrics(voluntary_switch_fraction=0.55)
    assert probe_mod._classify(m, _order(), None)["causes"]["c_pathological"]
    # Train base 0.3 lifts the bar to 0.6, so 0.55 is no longer pathological.
    assert not probe_mod._classify(m, _order(), 0.3)["causes"]["c_pathological"]
    assert probe_mod._classify(m, _order(), 0.3)["switch_fraction_bar"] == 0.6


def test_classify_fallback_minor_flag():
    out = probe_mod._classify(_metrics(fallback_rate=0.05), _order(), None)
    assert out["fallback_minor"]
    assert not out["causes"]["a_decode_mask"]


def test_rejection_handler_matches_level25_messages():
    probe = probe_mod._BehaviorProbe()
    probe.arm = "bc"
    probe.current_tag = "battle-gen9ou-1"
    handler = probe_mod._RejectionHandler(probe)

    def _record(msg):
        return logging.LogRecord("p", 25, __file__, 0, msg, (), None)

    handler.emit(_record("Error message received: |error|[Invalid choice] ..."))
    handler.emit(_record("Error message received: |error|[Unavailable choice] ..."))
    handler.emit(_record("Error message received: something else"))
    assert [r["kind"] for r in probe.rejections] == ["invalid", "unavailable"]
    assert probe.rejections[0]["battle"] == "battle-gen9ou-1"
    assert probe.rejections[0]["arm"] == "bc"


def test_fallback_spy_flags_and_delegates():
    probe = probe_mod._BehaviorProbe()
    order = SimpleNamespace(message="/choose move x")
    battle = SimpleNamespace(valid_orders=[order])
    probe_mod._FallbackSpy.probe = probe
    try:
        out = probe_mod._FallbackSpy.choose_random_singles_move(battle)
    finally:
        probe_mod._FallbackSpy.probe = None
    assert out is order  # delegated to the real chooser over valid_orders
    assert probe.fallback_hit


def test_record_decision_collapses_repeats_and_tracks_orders():
    probe = probe_mod._BehaviorProbe()
    probe.arm = "bc"
    mons = {
        n: SimpleNamespace(species=n, current_hp_fraction=1.0)
        for n in ("glimmora", "kingambit")
    }
    battle = SimpleNamespace(
        battle_tag="b1",
        turn=3,
        force_switch=False,
        team=mons,
        active_pokemon=mons["glimmora"],
        opponent_active_pokemon=SimpleNamespace(species="greattusk", current_hp_fraction=0.5),
        _teambuilder_team=[SimpleNamespace(species="Glimmora", nickname=None),
                           SimpleNamespace(species=None, nickname="Kingambit")],
    )
    order = SimpleNamespace(message="/choose move stealthrock")
    probe.record_decision(6, battle, order, fallback=False)
    probe.record_decision(6, battle, order, fallback=False)  # rejection retry
    probe.record_decision(0, battle, order, fallback=True)
    assert len(probe.decisions) == 2
    assert probe.decisions[0].repeat == 2
    assert probe.decisions[1].fallback
    assert probe.packed_orders["b1"] == ["glimmora", "kingambit"]  # species-or-nickname
    assert probe.live_orders["b1"] == ["glimmora", "kingambit"]
    assert probe.unstable_tags == set()


def test_transcript_format():
    decs = [
        _dec(),
        _dec(action=0, order_str="/choose switch Kingambit", fallback=True, repeat=37, turn=7),
    ]
    info = {
        "won": False,
        "turns": 34,
        "preview": "/team 415263",
        "packed": _TEAM,
        "live": _TEAM,
        "order_match": True,
    }
    text = probe_mod._format_transcript("bc", "battle-gen9ou-42", decs, info)
    assert "=== bc | battle-gen9ou-42 | won=False turns=34 | preview=/team 415263" in text
    assert "ORDER MATCH" in text
    assert "-> move0 '/choose move stealthrock' p=0.62" in text
    assert "alt: move1 0.21" in text
    assert "FALLBACK" in text
    assert "(x37)" in text
    info["order_match"] = False
    assert "ORDER MISMATCH" in probe_mod._format_transcript("bc", "t", decs, info)


def test_train_baseline_missing_file():
    assert probe_mod._train_baseline(Path("does/not/exist.npz")) is None

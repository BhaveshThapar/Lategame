"""G5 capability gate: it must judge by the judged gate's OWN criterion, and report the null.

The risk this gate carries is not a bug, it is a temptation -- G5 had no exit criterion anywhere in
the repository, so any bar written now is written after the measurements exist. These tests pin the
two properties that keep it honest: no threshold is introduced here, and the one capability whose
strength contribution is NULL still says so.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_GATE_PATH = _ROOT / "scripts" / "g5_capability_gate.py"
_spec = importlib.util.spec_from_file_location("g5_capability_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
g5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g5)

_HAS_RESULTS = (_ROOT / "results" / "rpredict_search_ou.json").exists()
requires_results = pytest.mark.skipif(not _HAS_RESULTS, reason="needs committed results/")


def test_every_threshold_comes_from_the_record_it_judges(monkeypatch, tmp_path):
    """No bar is introduced by this gate. Rewrite the floors INSIDE the records and the verdict
    must follow them -- if a threshold were hardcoded here, this would still pass."""
    fake = tmp_path / "results"
    fake.mkdir()

    def w(name, obj):
        (fake / name).write_text(json.dumps(obj))

    # Every rate is comfortably high, but each record declares a floor ABOVE its own rate.
    for name in ("rpredict_recon.json", "rpredict_recon_ou.json", "rpredict_recon_vgc.json"):
        w(name, {"match_rate": 0.995, "checks": 10, "mismatches": 0, "pass_rate": 0.999})
    w("rpredict_fidelity.json", {
        "replays": 1, "transitions": 1, "core_match_rate": 0.995, "full_match_rate": 0.995,
        "threshold": 0.999, "verdict": "PASS",
    })
    w("rpredict_oppmodel_gate_a.json", {
        "opp_pov_team_fidelity": 1.0, "whitebox_decodable_rate": 1.0,
        "whitebox_vs_heuristic_agreement": 0.9, "agreement_n": 10, "gate_a_pass": False,
    })
    monkeypatch.setattr(g5, "_RESULTS", fake)

    out = g5.run()
    assert out["verdict"] == "NOT MET"
    by_name = {c["capability"]: c for c in out["capabilities"]}
    assert by_name["state estimation"]["ok"] is False
    assert by_name["precise damage math"]["ok"] is False
    assert by_name["prediction / opponent modeling"]["ok"] is False


def test_a_missing_record_fails_rather_than_being_skipped(monkeypatch, tmp_path):
    """An absent capability is not a passed one. A gate that quietly ignored what it could not find
    would report MET on an empty results/."""
    empty = tmp_path / "results"
    empty.mkdir()
    monkeypatch.setattr(g5, "_RESULTS", empty)
    out = g5.run()
    assert out["verdict"] == "NOT MET"
    assert all(not c["ok"] for c in out["capabilities"])


@requires_results
def test_planning_passes_on_capability_and_still_reports_the_null():
    """The case that separates 'demonstrated' from 'beneficial'. Search runs at n=2500/arm against
    a pre-registered rule and the rule returns NULL. G5 asks for a testable capability, not a win
    rate -- but a G5 that dropped the null would be the 'merely implemented' claim it replaces."""
    plan = g5._planning()
    assert plan["ok"] is True
    assert plan["strength_verdict"] == "NULL"
    assert "NOT BENEFICIAL" in plan["note"]
    assert plan["records"][0]["contrast"]["p_value"] > 0.05


@requires_results
def test_the_record_cites_a_checkpoint_so_the_headline_is_not_vacuous():
    """`check_artifacts` verifies a headline by the checkpoints its records name. A G5 record that
    named none would register as OK over an empty set -- the exact trap `16aa893` had to repair for
    the VGC ladder."""
    from scripts import check_artifacts  # noqa: F401  -- import guard only

    out = g5.run()
    assert "checkpoints/" in json.dumps(out), "nothing to verify the headline against"


@requires_results
def test_state_estimation_covers_all_three_formats():
    """G4's claim is that three formats run through one core; G5's state-estimation leg is where
    that is actually measured per format rather than asserted."""
    rows = g5._state_estimation()["records"]
    assert {r["format"] for r in rows} == {"gen9randombattle", "gen9ou", "gen9vgc2025regi"}
    assert all(r["ok"] for r in rows)


@requires_results
def test_the_committed_record_matches_a_fresh_run():
    """`results/g5_capability_gate.json` is an assembly of other records, so it can silently go
    stale when one of them is re-run. Recomputing must reproduce it."""
    committed = json.loads((_ROOT / "results" / "g5_capability_gate.json").read_text())
    assert committed == g5.run()


@requires_results
def test_every_capability_names_a_runnable_reproduction_command():
    """The reason three producing scripts read as orphans until now.

    The record named the *library* a capability rests on (`lategame/search/recon_check.py`) and not
    the command that regenerates the JSON, so a mechanical reference scan flagged
    `rpredict_recon_gate.py`, `rpredict_recon_live_gate.py` and `rpredict_fidelity_gate.py` as
    referenced by nothing — while they are the reproduction path for a `check_artifacts` headline.
    Deleting them would have left the claim standing and its evidence unreproducible.
    """
    import re

    out = g5.run()
    for cap in out["capabilities"]:
        cmds = cap.get("reproduce")
        assert cmds, f"{cap['capability']} names no reproduction command"
        for cmd in cmds:
            for script in re.findall(r"scripts/[\w/]+\.(?:py|slurm)", cmd):
                assert (_ROOT / script).exists(), f"{cap['capability']} -> missing {script}"

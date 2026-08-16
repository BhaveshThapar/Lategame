"""Unit tests for the Lever 15 format-ceiling gate's pure logic (no server, no node).

Covers the Wilson interval, the tie-aware ROC-AUC used for the team-RNG variance
decomposition (M3), and the decision rule that turns (M1 skill band, M2 upper bound,
M3 AUC) into a FORMAT_BOUND / MODEL_BOUND / AMBIGUOUS verdict. The server sweep (M1)
and the node re-sim (M3 data path) are exercised only by the gated run.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "format_ceiling_gate.py"
_spec = importlib.util.spec_from_file_location("format_ceiling_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_wilson_ci_brackets_the_rate_and_clamps():
    lo, hi = gate.wilson_ci(150, 300)
    assert lo < 0.5 < hi
    assert 0.03 < (hi - lo) < 0.12  # ~+/-0.056 at n=300
    lo0, hi0 = gate.wilson_ci(0, 10)
    assert lo0 == 0.0 and hi0 < 1.0
    assert gate.wilson_ci(0, 0) == (0.0, 1.0)


def test_roc_auc_extremes_and_ties():
    # Perfect: higher score always the winner.
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    assert gate.roc_auc(scores, labels) == 1.0
    # Perfectly anti-correlated.
    assert gate.roc_auc(scores, np.array([1, 1, 0, 0])) == 0.0
    # All identical scores -> pure ties -> 0.5.
    assert gate.roc_auc(np.zeros(4), np.array([0, 1, 0, 1])) == 0.5
    # Single class -> undefined.
    assert math.isnan(gate.roc_auc(scores, np.array([1, 1, 1, 1])))


def test_roc_auc_matches_probability_definition():
    # AUC = P(score_pos > score_neg) + 0.5 P(tie), checked by brute force.
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=200).astype(float)  # many ties
    labels = (rng.random(200) < 0.4).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    expected = wins / (len(pos) * len(neg))
    assert abs(gate.roc_auc(scores, labels) - expected) < 1e-9


def _signals(s: float, w: float, g: float, a: float, mirror: float = 0.5) -> tuple:
    m1 = {
        "simpleheuristics": {"rate": s},
        "offrl_green": {"rate": g},
        "mirror": {"rate": mirror},
    }
    m2 = {"search_vs_heuristic": w}
    m3 = {"auc": a}
    return m1, m2, m3


def test_verdict_format_bound_when_nothing_beats_heuristic():
    # Best competent agent at/below BAND_TOP -> pivot, regardless of the (corroborating) AUC.
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.70))
    assert d["verdict"] == "FORMAT_BOUND"
    assert d["next_branch"] == "ou_pivot"
    assert d["mirror_sanity_ok"] is True
    assert d["m3_corroboration"]["rng_bound_corroborated"] is True


def test_verdict_format_bound_even_with_low_auc():
    # AUC below AUC_HI must NOT flip the branch: a caveated proxy only corroborates.
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.45))
    assert d["verdict"] == "FORMAT_BOUND"
    assert d["next_branch"] == "ou_pivot"
    assert d["m3_corroboration"]["rng_bound_corroborated"] is False


def test_verdict_model_bound_on_headroom():
    d = gate.compute_verdict(*_signals(s=0.52, w=0.50, g=0.60, a=0.45))
    assert d["verdict"] == "MODEL_BOUND"
    assert d["next_branch"] == "scale_up"


def test_verdict_ambiguous_between_band_and_headroom():
    # Best competent in (BAND_TOP, HEADROOM) -> tie-break to a cheap scale-up probe.
    d = gate.compute_verdict(*_signals(s=0.55, w=0.50, g=0.47, a=0.70))
    assert d["verdict"] == "AMBIGUOUS"
    assert d["next_branch"] == "scale_up_probe"


def test_mirror_sanity_flag_trips():
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.70, mirror=0.62))
    assert d["mirror_sanity_ok"] is False


def _ou_m1(mirror=0.51, simple=0.60, maxbp=0.20, rand=0.03, green=0.30) -> dict:
    return {
        "mirror": {"rate": mirror},
        "simpleheuristics": {"rate": simple},
        "maxbasepower": {"rate": maxbp},
        "random": {"rate": rand},
        "offrl_green": {"rate": green},
    }


def test_assess_ou_clean_harness_wider_band():
    a = gate.assess_ou(_ou_m1())
    assert a["harness_ok"] is True
    assert a["mirror_sanity_ok"] is True
    assert a["gradient_ok"] is True
    # width 0.60 - 0.03 = 0.57 vs RB 0.523 - 0.007 = 0.516 -> wider
    assert a["band_width"]["wider_than_rb"] is True
    # The RB-style top-level FORMAT/MODEL verdict is still never applied to the OU smoke.
    assert "verdict" not in a
    # ...but the OU-specific FORMAT-vs-MODEL verdict is: simpleheuristics 0.60 >= HEADROOM 0.58.
    assert a["ou_verdict"]["verdict"] == "MODEL_BOUND"
    assert a["ou_verdict"]["format_bound_rejected"] is True


def test_assess_ou_model_bound_reports_model_gap():
    m1 = _ou_m1(simple=0.64)
    m1["bc_v11"] = {"rate": 0.03, "ci95": [0.01, 0.06]}
    a = gate.assess_ou(m1)
    v = a["ou_verdict"]
    assert v["verdict"] == "MODEL_BOUND"
    assert v["learned_bc"]["rate"] == 0.03
    assert abs(v["model_gap"] - (0.64 - 0.03)) < 1e-9


def test_assess_ou_insufficient_when_competent_below_headroom():
    # A clean harness but no wide-band evidence (simpleheuristics < HEADROOM) -> no MODEL_BOUND.
    a = gate.assess_ou(_ou_m1(simple=0.52, maxbp=0.20))
    assert a["harness_ok"] is True
    assert a["ou_verdict"]["verdict"] == "INSUFFICIENT"
    assert a["ou_verdict"]["format_bound_rejected"] is False


def test_assess_ou_flags_broken_mirror_and_gradient():
    a = gate.assess_ou(_ou_m1(mirror=0.65, maxbp=0.70))  # mirror off; maxbp > simple
    assert a["mirror_sanity_ok"] is False
    assert a["gradient_ok"] is False
    assert a["harness_ok"] is False
    # A dirty harness must not emit a trustworthy verdict.
    assert a["ou_verdict"]["verdict"] == "INSUFFICIENT"
    assert a["ou_verdict"]["format_bound_rejected"] is False


def test_build_matchups_appends_loop_fixed_bc_arm():
    base = gate._build_matchups(None)
    assert base == gate._MATCHUPS
    assert base is not gate._MATCHUPS  # a copy, not the module constant
    with_bc = gate._build_matchups("checkpoints/bc_gen9ou_v11_s0.pt")
    assert with_bc[-1] == ("bc_v11", "bc", "heuristic", "checkpoints/bc_gen9ou_v11_s0.pt")
    assert len(with_bc) == len(base) + 1


def test_build_matchups_can_drop_stale_offrl_green_arm():
    # On OU the RB offrl_green checkpoint can't load (older encoder); the arm is droppable while
    # the loop-fixed bc arm is kept.
    labels = [m[0] for m in gate._build_matchups(
        "checkpoints/bc_gen9ou_v11_s0.pt", include_offrl_green=False
    )]
    assert "offrl_green" not in labels
    assert labels[-1] == "bc_v11"


def test_build_matchups_appends_offrl_ou_arm():
    # Build 16: an offrl checkpoint adds a dedicated offrl_ou (PPO-self-play) learned arm.
    ms = gate._build_matchups(None, include_offrl_green=False, offrl_ckpt="checkpoints/ppo.pt")
    assert ms[-1] == ("offrl_ou", "offrl", "heuristic", "checkpoints/ppo.pt")


def test_build_matchups_appends_both_learned_arms():
    ms = gate._build_matchups(
        "checkpoints/bc.pt", include_offrl_green=False, offrl_ckpt="checkpoints/ppo.pt"
    )
    labels = [m[0] for m in ms]
    assert "offrl_ou" in labels and "bc_v11" in labels
    assert labels[-1] == "bc_v11"  # bc appended after offrl_ou


def test_assess_ou_model_gap_prefers_offrl_arm():
    # With both learned arms present, model_gap is measured against the stronger PPO offrl_ou arm.
    m1 = _ou_m1(simple=0.64)
    m1["bc_v11"] = {"rate": 0.03, "ci95": [0.01, 0.06]}
    m1["offrl_ou"] = {"rate": 0.20, "ci95": [0.16, 0.25]}
    v = gate.assess_ou(m1)["ou_verdict"]
    assert v["verdict"] == "MODEL_BOUND"
    assert v["learned_offrl"]["rate"] == 0.20
    assert v["learned_bc"]["rate"] == 0.03
    assert abs(v["model_gap"] - (0.64 - 0.20)) < 1e-9


def test_a_learned_arm_record_names_the_checkpoint_it_scored():
    """The record is the only place the weights behind a published ceiling number can be named.

    `run_m1` wrote `{p1, p2, rate, wins, ci95}` and nothing else, and the two learned labels are
    OU-era schema keys (`bc_v11`, `offrl_ou`) that describe nothing on a doubles run -- both arms
    build as the `doubles` agent there. So a reader of `results/format_ceiling_gate_vgc_v2.json`
    could not get from a number to the checkpoint that produced it.
    """
    rec = gate._arm_record(
        "doubles", "heuristic", 0.4533333333333333, 300, "checkpoints/doubles_bc_vgc_v2.pt"
    )
    assert rec["checkpoint"] == "checkpoints/doubles_bc_vgc_v2.pt"
    # Byte-identical arithmetic to what run_m1 did inline, checked against the committed record.
    assert rec["wins"] == 136
    assert abs(rec["ci95"][0] - 0.3979441581242692) < 1e-12
    assert abs(rec["ci95"][1] - 0.5099025620088528) < 1e-12


def test_a_baseline_arm_record_carries_no_checkpoint_key():
    """Baselines have no checkpoint, and adding a `"checkpoint": null` would make every record
    already committed structurally different from every record written after this change."""
    rec = gate._arm_record("heuristic", "heuristic", 0.5033333333333333, 300)
    assert "checkpoint" not in rec
    assert set(rec) == {"p1", "p2", "rate", "wins", "ci95"}
    assert rec["wins"] == 151


def test_the_recorded_checkpoint_is_visible_to_the_artifact_scanner():
    """Provenance the scanner cannot see is worse than none: `check_artifacts.py` would report a
    published headline as backed by zero checkpoints, and `prune_checkpoints.py` -- which uses the
    same scan as its allowlist -- would treat the file as uncited and prunable."""
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from prune_checkpoints import _CKPT_RE  # type: ignore[import-not-found]

    rec = gate._arm_record("doubles", "heuristic", 0.45, 300, "checkpoints/doubles_bc_vgc_v2.pt")
    found = _CKPT_RE.findall(json.dumps(rec))
    assert found == ["checkpoints/doubles_bc_vgc_v2.pt"]


# --------------------------------------------------------------------------- #
# The learned arm is read by ALIAS, so a teambuilt record reaches the verdict.
# --------------------------------------------------------------------------- #
def test_verdict_reads_the_teambuilt_learned_arm_names():
    """`run_m1` labels the learned arm `offrl_green` on RB and `offrl_ou` on a teambuilt format,
    and drops the green arm off RB entirely. `compute_verdict` used to read `offrl_green` by
    literal key, so a VGC record could not reach it at all -- KeyError, not a verdict.

    The names are the on-disk schema of every record ever written, so they are aliased rather
    than renamed; renaming would make old and new gates incomparable.
    """
    m1 = {
        "simpleheuristics": {"rate": 0.49},
        "offrl_ou": {"rate": 0.47},
        "bc_v11": {"rate": 0.45},
        "mirror": {"rate": 0.50},
    }
    d = gate.compute_verdict(m1, {"search_vs_heuristic": 0.50}, {"auc": 0.50})
    assert d["verdict"] == "FORMAT_BOUND"
    assert d["signals"]["offrl_green_g"] == 0.47, "the best learned arm present, by any alias"


def test_verdict_survives_a_record_with_no_learned_arm_at_all():
    """A first-look format has baselines and no trained policy. That must contribute nothing to
    `competent` rather than raising -- M1's band and M2 still decide the branch."""
    m1 = {"simpleheuristics": {"rate": 0.62}, "mirror": {"rate": 0.50}}
    d = gate.compute_verdict(m1, {"search_vs_heuristic": 0.50}, {"auc": 0.50})
    assert d["signals"]["offrl_green_g"] == 0.0
    assert d["verdict"] == "MODEL_BOUND"  # driven by simpleheuristics 0.62 >= HEADROOM


# --------------------------------------------------------------------------- #
# M3 off team-preview lines: no inputlog, no node.
# --------------------------------------------------------------------------- #
def _replay(p1_team, p2_team, winner, p1="alice", p2="bob"):
    lines = [f"|player|p1|{p1}|1|", f"|player|p2|{p2}|2|"]
    lines += [f"|poke|p1|{s}, L50, F|" for s in p1_team]
    lines += [f"|poke|p2|{s}, L50, F|" for s in p2_team]
    lines += ["|teampreview", "|start", f"|win|{winner}"]
    return {"log": "\n".join(lines)}


def test_preview_teams_reads_both_full_rosters_as_species_ids():
    d = _replay(["Flutter Mane"] * 6, ["Iron Hands"] * 6, "alice")
    p1, p2 = gate._preview_teams(d["log"])
    assert p1 == ["fluttermane"] * 6
    assert p2 == ["ironhands"] * 6


def test_preview_teams_strips_level_and_gender():
    """`|poke|p1|Dragonite, L50, F|` -- the species is everything before the FIRST comma. Keeping
    the rest would make every mon an out-of-vocabulary lookup scoring 0."""
    p1, _ = gate._preview_teams("|poke|p1|Dragonite, L50, F|")
    assert p1 == ["dragonite"]


def test_log_player_names_pairs_the_sides():
    p1, p2 = gate._log_player_names(_replay(["Ditto"], ["Ditto"], "alice")["log"])
    assert (p1, p2) == ("alice", "bob")


def test_m3_preview_scores_only_complete_six_mon_previews(tmp_path):
    """A preview screen that did not show all six is not a comparable sample, and a replay with no
    resolvable winner (tie, disconnect) is not a label. Both are dropped, and `n_battles_used`
    says how many survived -- the number a reader needs to judge the AUC."""
    import json as _json

    strong = ["Miraidon"] * 6
    weak = ["Sunkern"] * 6
    files = []
    for i, d in enumerate(
        [
            _replay(strong, weak, "alice"),          # kept
            _replay(weak, strong, "bob"),            # kept
            _replay(strong[:4], weak, "alice"),      # dropped: only 4 shown
            _replay(strong, weak, "nobody"),         # dropped: winner matches neither player
        ]
    ):
        p = tmp_path / f"r{i}.json"
        p.write_text(_json.dumps(d))
        files.append(str(p))

    out = gate.run_m3_preview(sorted(files))
    assert out["n_replays_scanned"] == 4
    assert out["n_battles_used"] == 2
    assert out["mode"] == "preview"
    # Both kept battles are won by the stronger side, so the proxy separates them perfectly.
    assert out["auc"] == 1.0
    assert out["caveats"], "the unrated sample and the brought-six limit travel with the number"


def test_m3_dispatches_to_the_preview_path_when_no_inputlog_is_present(tmp_path):
    """Public replays carry no inputlog, so the re-sim path cannot run on them. Dispatch is on the
    DATA rather than the format string, and one mode is chosen for the whole batch -- mixing a
    level-adjusted proxy with a species-level one inside one AUC compares incomparable numbers."""
    import json as _json

    for i in range(3):
        (tmp_path / f"r{i}.json").write_text(
            _json.dumps(_replay(["Miraidon"] * 6, ["Sunkern"] * 6, "alice"))
        )
    out = gate.run_m3(10, str(tmp_path / "*.json"))
    assert out["mode"] == "preview"


def test_auc_bootstrap_ci_reports_undefined_instead_of_crashing_on_one_class():
    """Every resample of a single-class sample is itself single-class, so the NaN filter emptied
    the array and numpy raised IndexError from inside a percentile. An undefined interval is a
    result; a traceback out of a helper is not."""
    lo, hi = gate.auc_bootstrap_ci(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert math.isnan(lo) and math.isnan(hi)
    # NaN != NaN, so the empty case is checked the same way rather than by tuple equality.
    elo, ehi = gate.auc_bootstrap_ci(np.array([]), np.array([]))
    assert math.isnan(elo) and math.isnan(ehi)


# --------------------------------------------------------------------------- #
# M2 is echoed from two different record shapes.
# --------------------------------------------------------------------------- #
def test_load_m2_reads_both_the_single_run_and_the_pooled_shape(tmp_path):
    """`rpredict_oppmodel_gate` writes scalar rates and a top-level `n`; `merge_search_shards`
    writes {wins, n, rate} objects and no top-level `n`, because once shards are summed the n is
    per-arm. Reading only the first shape is what made "echo the pooled search run as the M2 leg"
    impossible without hand-editing JSON -- and the pooled run is the larger measurement."""
    import json as _json

    single = tmp_path / "single.json"
    single.write_text(_json.dumps({
        "n": 120, "depth": 2, "base_vs_heuristic": 0.4833,
        "arms": {"whitebox": {"search_vs_heuristic": 0.5, "search_vs_base": 0.3167}},
    }))
    pooled = tmp_path / "pooled.json"
    pooled.write_text(_json.dumps({
        "shards": 10, "format": "gen9ou", "depth": 2,
        "base_vs_heuristic": {"wins": 1922, "n": 2500, "rate": 0.7688},
        "arms": {"whitebox": {
            "search_vs_heuristic": {"wins": 1931, "n": 2500, "rate": 0.7724},
            "search_vs_base": {"wins": 983, "n": 2500, "rate": 0.3932},
            "contrast_vs_base": {"diff": 0.0036, "p_value": 0.762101},
            "verdict": "NULL",
        }},
    }))

    a = gate.load_m2(single)
    assert a["search_vs_heuristic"] == 0.5 and a["n"] == 120

    b = gate.load_m2(pooled)
    assert b["search_vs_heuristic"] == 0.7724, "the rate, not the {wins,n,rate} object"
    assert b["n"] == 2500, "per-arm n, since a pooled record has no top-level n"
    assert b["shards"] == 10 and b["verdict"] == "NULL"
    # Whatever the shape, compute_verdict only ever needs a float here.
    assert isinstance(b["search_vs_heuristic"], float)


def test_load_m2_says_how_to_produce_a_missing_record(tmp_path):
    """The old hardcoded path meant a missing M2 was a bare FileNotFoundError from a json read."""
    with pytest.raises(SystemExit, match="rpredict_oppmodel_gate"):
        gate.load_m2(tmp_path / "absent.json")


# --------------------------------------------------------------------------- #
# The verdict text and branch were written for gen9-RB and asserted RB-only things.
# --------------------------------------------------------------------------- #
def test_a_record_with_no_format_key_is_still_read_as_random_battles():
    """`run_m1` only began writing `format` when teambuilt support landed, so the record holding
    the ORIGINAL FORMAT_BOUND verdict has none. A generic fallback would re-derive that historical
    decision with a different branch the next time anyone re-ran the gate."""
    d = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.45))
    assert d["format"] == "gen9randombattle"
    assert d["next_branch"] == "ou_pivot"


def test_a_teambuilt_format_is_not_told_to_pivot_to_ou():
    """"ou_pivot" is advice for a format measured BEFORE the OU pivot. On a teambuilt format the
    actionable branch is to stop spending on the strength axis, not to pivot again."""
    m1 = {
        "format": "gen9vgc2025regi",
        "simpleheuristics": {"rate": 0.493},
        "offrl_ou": {"rate": 0.447},
        "mirror": {"rate": 0.497},
    }
    d = gate.compute_verdict(m1, {"search_vs_heuristic": 0.353}, {"auc": 0.520})
    assert d["verdict"] == "FORMAT_BOUND"
    assert d["next_branch"] == "stop_strength_axis"


def test_the_reason_only_says_at_parity_when_m2_is_actually_at_parity():
    """The FORMAT_BOUND template interpolated M2 as "at parity" unconditionally. True of RB's
    0.500; simply false of a w nowhere near 0.50, and a verdict line stating a wrong number is
    worse than one stating none."""
    at_parity = gate.compute_verdict(*_signals(s=0.51, w=0.50, g=0.47, a=0.45))
    assert "at parity" in at_parity["reason"]

    far = gate.compute_verdict(*_signals(s=0.51, w=0.353, g=0.47, a=0.45))
    assert "at parity" not in far["reason"]
    assert "reaches only 0.353" in far["reason"]


def test_the_verdict_carries_which_search_leaf_backed_its_m2():
    """`ShapedOnlyPolicy` is a weaker instrument than a trained value head, and its pre-registered
    consequence is that a NULL is suggestive only. The caveat has to travel with the branch."""
    m1 = {"format": "gen9vgc2025regi", "simpleheuristics": {"rate": 0.49},
          "mirror": {"rate": 0.50}}
    d = gate.compute_verdict(
        m1, {"search_vs_heuristic": 0.35, "search_leaf": "shaped_only"}, {"auc": 0.5}
    )
    assert d["m2_leaf"] == "shaped_only"


# --------------------------------------------------------------------------- #
# Pooling shards that were not one run.
# --------------------------------------------------------------------------- #
def _merge_mod():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "merge_search_shards.py"
    spec = importlib.util.spec_from_file_location("merge_search_shards", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pooling_refuses_shards_that_disagree_on_the_search_leaf():
    """A pooled record is one number describing one experiment. Shards run with different leaves
    are two experiments, and taking shard 0's value -- the habit for format/init/depth -- would
    launder a real split into a single confident row."""
    m = _merge_mod()
    with pytest.raises(SystemExit, match="disagree on search_leaf"):
        m._one("search_leaf", [{"search_leaf": "shaped_only"}, {"search_leaf": "policy_value"}])


def test_pooling_carries_the_leaf_up_and_tolerates_shards_that_predate_it():
    m = _merge_mod()
    assert m._one("search_leaf", [{"search_leaf": "shaped_only"}] * 3) == "shaped_only"
    assert m._one("search_leaf", [{"n": 30}, {"n": 30}]) is None

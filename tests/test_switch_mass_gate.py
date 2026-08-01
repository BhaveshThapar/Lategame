"""Unit tests for the Build-7 switch-mass gate's pure logic (no server, no torch).

Covers the obs decoder (active/species/hp/force-switch channels), the defensiveness
score, loop-run extraction from the Build-6 decision stream, species->vocab mapping
with the forme fallback, the conditioning masks, the uniform-mass baseline and its
masked-softmax harness identity, the pre-registered H1/H2/H3 classifier (incl. the
attacker specificity control and the argmax-amplification annotation), the family
majority rule, and the raw replay-log parser. The torch scoring path is exercised by
actually running the gate.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "switch_mass_gate.py"
_spec = importlib.util.spec_from_file_location("switch_mass_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["switch_mass_gate"] = gate
_spec.loader.exec_module(gate)

_LAYOUT = gate.OBS_LAYOUT
_OBS_DIM = (
    _LAYOUT.team_size * 2 * _LAYOUT.pokemon_dim
    + _LAYOUT.n_moves * _LAYOUT.move_dim
    + _LAYOUT.global_dim
)


def _obs(own_block=2, own_species=42, own_hp=0.8, opp_block=1, opp_species=99, force=False):
    """A hand-built flat obs with one own and one opp active mon."""
    obs = np.zeros(_OBS_DIM, dtype=np.float32)
    p = _LAYOUT.pokemon_dim
    if own_block is not None:
        base = own_block * p
        obs[base + 1] = 1.0  # active
        obs[base + 3] = own_hp
        obs[base + gate._SPECIES_IDX] = own_species
    if opp_block is not None:
        base = (_LAYOUT.team_size + opp_block) * p
        obs[base + 1] = 1.0
        obs[base + gate._SPECIES_IDX] = opp_species
    obs[gate._FORCE_SWITCH_IDX] = float(force)
    return obs


def _row(**kw):
    base = dict(
        arm="bc", battle_tag="b1", forced=False, action=0,
        our_active="gholdengo", opp_active="corviknight",
    )
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Obs decoding
# --------------------------------------------------------------------------- #
def test_decode_actives_from_handbuilt_obs():
    obs = np.stack([_obs(), _obs(own_block=0, own_species=7, own_hp=0.3, force=True)])
    out = gate.decode_actives(obs)
    assert out["own_species"].tolist() == [42, 7]
    assert np.allclose(out["own_hp"], [0.8, 0.3])
    assert out["opp_species"].tolist() == [99, 99]
    assert out["n_own_active"].tolist() == [1, 1]
    assert out["force_switch"].tolist() == [False, True]


def test_decode_actives_no_active():
    obs = np.stack([_obs(own_block=None, opp_block=None)])
    out = gate.decode_actives(obs)
    assert out["own_species"].tolist() == [0]
    assert out["opp_species"].tolist() == [0]
    assert out["n_own_active"].tolist() == [0]
    assert out["n_opp_active"].tolist() == [0]


def test_defensiveness_score():
    mat = np.zeros((3, 26), dtype=np.float32)
    mat[1, 2], mat[1, 4] = 2.0, 1.0  # def, spd
    mat[2, 1], mat[2, 3] = 2.0, 1.5  # atk, spa
    score = gate.defensiveness(mat)
    assert score[0] == 0.0
    assert score[1] == 3.0
    assert score[2] == -3.5


# --------------------------------------------------------------------------- #
# Loop extraction from the Build-6 decision stream
# --------------------------------------------------------------------------- #
def test_voluntary_runs_split_on_forced_and_battles():
    rows = [
        _row(action=0),
        _row(action=1),
        _row(action=0, forced=True),  # forced: neither extends nor breaks
        _row(action=2),
        _row(action=6),  # voluntary move ends the run
        _row(action=3),
        _row(action=4, battle_tag="b2"),  # battle boundary resets
    ]
    runs = gate.voluntary_runs(rows)
    assert [len(r) for r in runs] == [3, 1, 1]
    assert [d["action"] for d in runs[0]] == [0, 1, 2]


def test_loop_species_extraction_thresholds():
    # 10 long runs; 'tinglu' in 3 of them (0.3 >= 0.15), 'dragonite' in 1 (excluded).
    runs = []
    for i in range(10):
        species = "tinglu" if i < 3 else "gholdengo"
        runs.append([_row(our_active=species) for _ in range(12)])
    runs.append([_row(our_active="dragonite")])  # short run: ignored entirely
    out = gate.loop_species_from_runs(runs, run_k=10, min_run_frac=0.15)
    assert out == ["gholdengo", "tinglu"]


def test_loop_pairs_min_count():
    run = [_row(our_active="a", opp_active="b") for _ in range(30)]
    run += [_row(our_active="c", opp_active="d") for _ in range(5)]
    out = gate.loop_pairs_from_runs([run], run_k=10, min_count=25)
    assert out == [("a", "b")]


def test_species_to_ids_forme_fallback():
    table = {"zacian": 5, "zaciancrowned": 6, "gholdengo": 7}
    mapped, missing = gate.species_to_ids(["gholdengo", "zaciancrowned", "missingno"], table)
    assert mapped == {"gholdengo": 7, "zaciancrowned": 6}
    assert missing == ["missingno"]
    # prefix fallback picks the shortest matching key
    mapped, _ = gate.species_to_ids(["zacianc"], table)
    assert mapped == {"zacianc": 5}


# --------------------------------------------------------------------------- #
# Conditioning + baselines
# --------------------------------------------------------------------------- #
def test_conditioning_masks():
    actives = {
        "own_species": np.array([10, 10, 20, 30, 0, 10]),
        "own_hp": np.array([0.9, 0.3, 1.0, 1.0, 1.0, 0.6], dtype=np.float32),
        "opp_species": np.array([40, 40, 40, 40, 40, 0]),
    }
    def_score = np.zeros(50)
    def_score[20], def_score[30] = 2.0, -2.0
    conds = gate.conditioning_masks(actives, {10}, {(10, 40)}, def_score)
    assert conds["overall"].all()
    assert conds["loop_species"].tolist() == [True, False, False, False, False, True]
    assert conds["loop_species_any_hp"].tolist() == [True, True, False, False, False, True]
    assert conds["dex_wall"].tolist() == [False, False, True, False, False, False]
    assert conds["dex_attacker"].tolist() == [False, False, False, True, False, False]
    # pair matches (10, 40) but never an undecodable opp (opp == 0)
    assert conds["loop_pair"].tolist() == [True, True, False, False, False, False]


def test_uniform_switch_mass():
    mask = np.zeros((2, 26), dtype=bool)
    mask[0, :5] = True  # 5 switches
    mask[0, 6:10] = True  # 4 moves -> 5/9
    mask[1, :3] = True  # all-switch -> 1.0
    out = gate.uniform_switch_mass(mask)
    assert np.allclose(out, [5 / 9, 1.0])


def test_masked_softmax_np_identity():
    rng = np.random.default_rng(0)
    mask = rng.random((50, 26)) < 0.4
    mask[:, 6] = True  # keep every row non-empty
    probs = gate.masked_softmax_np(np.zeros((50, 26)), mask)
    assert np.abs(probs[:, :6].sum(axis=1) - gate.uniform_switch_mass(mask)).max() < 1e-9


def test_summarize_conditioning_counts():
    actions = np.array([0, 7, 3, 8])
    mask = np.zeros((4, 26), dtype=bool)
    mask[:, :6] = True
    mask[:, 6:8] = True
    cond = np.array([True, True, False, False])
    out = gate.summarize_conditioning(cond, actions, mask)
    assert out["n"] == 2
    assert out["human_rate"] == 0.5  # actions 0 (switch), 7 (move)
    assert np.isclose(out["uniform_mass"], 6 / 8)
    out = gate.summarize_conditioning(np.zeros(4, bool), actions, mask, actions * 0.0, actions < 6)
    assert out["n"] == 0 and out["human_rate"] is None and out["policy_mass"] is None


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
def _primary(h, p, t=0.0, n=5000):
    return {"n": n, "human_rate": h, "policy_mass": p, "top1_switch_rate": t}


_ATTACKER_OK = {"n": 5000, "policy_mass": 0.10}


def test_classify_h3_amplified():
    c = gate.classify_arm(_primary(0.20, 0.41), None, _ATTACKER_OK)
    assert c["verdict"] == "H3"
    assert c["flags"]["amplified"]


def test_classify_h1_pivot_heavy():
    c = gate.classify_arm(_primary(0.42, 0.45), None, _ATTACKER_OK)
    assert c["verdict"] == "H1"


def test_classify_h2_ood():
    pair = {"n": 500, "policy_mass": 0.25}
    c = gate.classify_arm(_primary(0.20, 0.22), pair, _ATTACKER_OK)
    assert c["verdict"] == "H2"
    assert not c["argmax_amp"]


def test_classify_mixed():
    # P - H = 0.15: neither matched (<= 0.10) nor amplified (>= 0.20)
    c = gate.classify_arm(_primary(0.20, 0.35), None, _ATTACKER_OK)
    assert c["verdict"] == "MIXED"


def test_classify_attacker_control_teeth():
    c = gate.classify_arm(_primary(0.20, 0.22), None, {"n": 5000, "policy_mass": 0.55})
    assert c["verdict"] == "H3-GLOBAL"
    assert not c["attacker_ok"]


def test_classify_argmax_amp_annotation():
    c = gate.classify_arm(_primary(0.20, 0.22, t=0.6), None, _ATTACKER_OK)
    assert c["verdict"] == "H2"
    assert c["argmax_amp"]


def test_classify_pair_floor_advisory():
    # pair mass 0.9 would block H2 via in_dist_live, but n < MIN_ROWS_PAIR -> ignored
    pair = {"n": gate.MIN_ROWS_PAIR - 1, "policy_mass": 0.90}
    c = gate.classify_arm(_primary(0.20, 0.22), pair, _ATTACKER_OK)
    assert c["verdict"] == "H2"
    pair["n"] = gate.MIN_ROWS_PAIR
    c = gate.classify_arm(_primary(0.20, 0.22), pair, _ATTACKER_OK)
    assert c["verdict"] == "MIXED"
    assert c["flags"]["in_dist_live"]


def test_classify_empty_primary():
    c = gate.classify_arm({"n": 0, "human_rate": None, "policy_mass": None}, None, _ATTACKER_OK)
    assert c["verdict"] == "EMPTY"


def test_family_verdict_majority_and_disagreement():
    v = gate.family_verdict([{"verdict": "H2"}, {"verdict": "H2"}, {"verdict": "H3"}])
    assert v == {"verdict": "H2", "seed_disagreement": True, "seeds": ["H2", "H2", "H3"]}
    v = gate.family_verdict([{"verdict": "H2"}, {"verdict": "H3"}, {"verdict": "H1"}])
    assert v["verdict"] == "MIXED" and v["seed_disagreement"]
    v = gate.family_verdict([{"verdict": "H3"}, {"verdict": "H3"}, {"verdict": "H3"}])
    assert v == {"verdict": "H3", "seed_disagreement": False, "seeds": ["H3"] * 3}
    # single-seed smoke run: the cap keeps it classifiable
    v = gate.family_verdict([{"verdict": "H2"}])
    assert v["verdict"] == "H2" and not v["seed_disagreement"]


def test_bc_vs_awr_delta_localization():
    assert 0.40 - 0.20 >= gate.BC_VS_AWR_DELTA
    assert not 0.30 - 0.20 >= gate.BC_VS_AWR_DELTA


# --------------------------------------------------------------------------- #
# Replay-log parser
# --------------------------------------------------------------------------- #
_LOG = "\n".join(
    [
        "|player|p1|alice|1|1200",
        "|switch|p1a: Nick|Gholdengo|100/100",  # pre-turn-1 lead: not a decision
        "|switch|p2a: Foo|Great Tusk, M|100/100",
        "|turn|1",
        "|move|p2a: Foo|Headlong Rush|p1a: Nick",
        "|move|p1a: Nick|Make It Rain|p2a: Foo",
        "|turn|2",
        "|switch|p1a: Wall|Corviknight, F|100/100",  # voluntary, first reveal
        "|move|p2a: Foo|Knock Off|p1a: Wall",
        "|turn|3",
        "|move|p1a: Wall|U-turn|p2a: Foo",
        "|switch|p1a: Nick|Gholdengo|80/100|[from] U-turn",  # pivot: excluded
        "|move|p2a: Foo|Headlong Rush|p1a: Nick",
        "|faint|p1a: Nick",
        "|switch|p1a: Wall|Corviknight, F|90/100",  # post-faint: forced
        "|turn|4",
        "|move|p2a: Foo|Roar|p1a: Wall",
        "|drag|p1a: Nick|Gholdengo|80/100",  # dragged: excluded
    ]
)


def test_parse_replay_log():
    counts = gate.parse_replay_log(_LOG, loop_species={"gholdengo"})
    # decisions: T1 p1 move + p2 move, T2 p1 switch + p2 move, T3 p1 move + p2 move,
    # T4 p2 move (p1's only T4 action is the drag) = 7
    assert counts["decisions"] == 7
    assert counts["switches"] == 1
    assert counts["first_reveal_switches"] == 1
    assert counts["excluded_from"] == 1
    assert counts["excluded_forced"] == 1
    assert counts["excluded_drag"] == 1
    # loop-conditioned: p1 decisions while gholdengo active = T1 move, T2 switch
    assert counts["loop_decisions"] == 2
    assert counts["loop_switches"] == 1


def test_parse_replay_log_faint_before_acting():
    log = "\n".join(
        [
            "|switch|p1a: A|Gholdengo|100/100",
            "|switch|p2a: B|Dragapult|100/100",
            "|turn|1",
            "|move|p2a: B|Dragon Darts|p1a: A",
            "|faint|p1a: A",
            "|switch|p1a: C|Corviknight|100/100",  # p1 never acted: still forced
            "|turn|2",
        ]
    )
    counts = gate.parse_replay_log(log, loop_species=set())
    assert counts["decisions"] == 1  # only p2's move
    assert counts["switches"] == 0
    assert counts["excluded_forced"] == 1

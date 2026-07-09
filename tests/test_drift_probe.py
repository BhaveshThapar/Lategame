"""Unit tests for the Build-8 drift probe's pure logic (no server, no torch).

Covers the active-block decoder, the role-aligned channel swap (each group, self-swap
identity, ALL positive-control), the fixed-width group extractor, the standardized distance,
the (own,opp) pair packing/matching, the base/donor assembler (caps + donor cycling), the
frac_explained orientation, and the carrier verdict (sole carrier / cumulative set / no
carrier). The torch swap-scoring path is exercised by actually running the gate.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "drift_probe.py"
_spec = importlib.util.spec_from_file_location("drift_probe", _PATH)
assert _spec is not None and _spec.loader is not None
dp = importlib.util.module_from_spec(_spec)
sys.modules["drift_probe"] = dp
_spec.loader.exec_module(dp)

_L = dp.OBS_LAYOUT
_PDIM = _L.pokemon_dim
_NUM = _L.pokemon_numeric_dim
_TEAM = _L.team_size
_OBS_DIM = _TEAM * 2 * _PDIM + _L.n_moves * _L.move_dim + _L.global_dim


def _blk(i: int) -> int:
    return i * _PDIM


def _obs(tag: float, own_block: int = 1, opp_block: int = 2) -> np.ndarray:
    """A flat obs fingerprinted by ``tag``: every channel encodes its (block, position) and
    ``tag`` so any mis-slotted swap is detectable. Own active at ``own_block``, opp at
    ``opp_block`` (opp blocks are the second six)."""
    obs = np.arange(_OBS_DIM, dtype=np.float32) + tag * 10_000.0
    # Zero every Pokemon block's active-flag channel (position 1) so the decoder finds exactly
    # one active per side, then set the two active blocks -- otherwise the arange fingerprint
    # would trip the ``> 0.5`` flag test on every block.
    for i in range(2 * _TEAM):
        obs[_blk(i) + 1] = 0.0
    obs[_blk(own_block) + 1] = 1.0
    obs[_blk(_TEAM + opp_block) + 1] = 1.0
    return obs


# --------------------------------------------------------------------------- #
# Active-block decoding
# --------------------------------------------------------------------------- #
def test_active_block_indices():
    obs = np.stack([_obs(0.0, own_block=1, opp_block=2), _obs(1.0, own_block=4, opp_block=0)])
    own, opp = dp.active_block_indices(obs)
    assert own.tolist() == [1, 4]
    assert opp.tolist() == [2, 0]


# --------------------------------------------------------------------------- #
# Channel swap
# --------------------------------------------------------------------------- #
def _one(row: np.ndarray) -> np.ndarray:
    return row.reshape(1, -1)


def test_swap_own_active_ids_moves_only_the_id_tail():
    base = _obs(0.0, own_block=1)
    donor = _obs(9.0, own_block=3)  # donor active in a *different* slot
    out = dp.swap_group(
        _one(base), _one(donor), "own_active_ids",
        np.array([1]), np.array([2]), np.array([3]), np.array([2]),
    )[0]
    b1 = _blk(1)
    d3 = _blk(3)
    # ID tail of base's active block now equals donor's active block ID tail...
    assert np.array_equal(out[b1 + _NUM : b1 + _PDIM], donor[d3 + _NUM : d3 + _PDIM])
    # ...numeric part of base's active block is untouched...
    assert np.array_equal(out[b1 : b1 + _NUM], base[b1 : b1 + _NUM])
    # ...and every other channel is untouched.
    keep = np.ones(_OBS_DIM, bool)
    keep[b1 + _NUM : b1 + _PDIM] = False
    assert np.array_equal(out[keep], base[keep])


def test_swap_own_active_numeric():
    base = _obs(0.0, own_block=1)
    donor = _obs(9.0, own_block=1)
    out = dp.swap_group(
        _one(base), _one(donor), "own_active_numeric",
        np.array([1]), np.array([2]), np.array([1]), np.array([2]),
    )[0]
    b1 = _blk(1)
    assert np.array_equal(out[b1 : b1 + _NUM], donor[b1 : b1 + _NUM])
    assert np.array_equal(out[b1 + _NUM : b1 + _PDIM], base[b1 + _NUM : b1 + _PDIM])


def test_swap_opp_active_full_block():
    base = _obs(0.0, opp_block=2)
    donor = _obs(9.0, opp_block=5)
    out = dp.swap_group(
        _one(base), _one(donor), "opp_active",
        np.array([1]), np.array([2]), np.array([1]), np.array([5]),
    )[0]
    b = _blk(_TEAM + 2)
    d = _blk(_TEAM + 5)
    assert np.array_equal(out[b : b + _PDIM], donor[d : d + _PDIM])


def test_swap_own_bench_maps_five_blocks():
    base = _obs(0.0, own_block=1)
    donor = _obs(9.0, own_block=1)
    out = dp.swap_group(
        _one(base), _one(donor), "own_bench",
        np.array([1]), np.array([2]), np.array([1]), np.array([2]),
    )[0]
    # Active block (1) untouched; all five bench blocks now hold donor values.
    assert np.array_equal(out[_blk(1) : _blk(1) + _PDIM], base[_blk(1) : _blk(1) + _PDIM])
    for i in range(_TEAM):
        if i == 1:
            continue
        assert np.array_equal(out[_blk(i) : _blk(i) + _PDIM], donor[_blk(i) : _blk(i) + _PDIM])


def test_swap_moves_and_global_regions():
    base = _obs(0.0)
    donor = _obs(9.0)
    a = np.array([1])
    for grp, lo, hi in (
        ("moves", dp._MOVES_START, dp._GLOBAL_START),
        ("global", dp._GLOBAL_START, _OBS_DIM),
    ):
        out = dp.swap_group(_one(base), _one(donor), grp, a, a, a, a)[0]
        assert np.array_equal(out[lo:hi], donor[lo:hi])
        assert np.array_equal(out[:lo], base[:lo])


def test_swap_move_subgroups():
    base = _obs(0.0)
    donor = _obs(9.0)
    a = np.array([1])
    mdim = dp._MDIM
    mnum = dp._MNUM
    # move_ids: only the trailing ID channel of each move block moves.
    ids = dp.swap_group(_one(base), _one(donor), "move_ids", a, a, a, a)[0]
    for j in range(dp._N_MOVES):
        s = dp._MOVES_START + j * mdim
        assert np.array_equal(ids[s + mnum : s + mdim], donor[s + mnum : s + mdim])
        assert np.array_equal(ids[s : s + mnum], base[s : s + mnum])  # numeric untouched
    # move_context: only pp (idx 4) + score (idx 5) of each block move.
    ctx = dp.swap_group(_one(base), _one(donor), "move_context", a, a, a, a)[0]
    for j in range(dp._N_MOVES):
        s = dp._MOVES_START + j * mdim
        assert ctx[s + 4] == donor[s + 4] and ctx[s + 5] == donor[s + 5]
        assert ctx[s + 1] == base[s + 1]  # base_power (static) untouched
    # global block never touched by any move swap.
    assert np.array_equal(ids[dp._GLOBAL_START:], base[dp._GLOBAL_START:])


def test_swap_self_is_identity_for_every_group():
    base = np.stack([_obs(0.0, own_block=2, opp_block=3), _obs(1.0, own_block=0, opp_block=5)])
    own, opp = dp.active_block_indices(base)
    for grp in dp.GROUPS:
        out = dp.swap_group(base, base.copy(), grp, own, opp, own, opp)
        assert np.array_equal(out, base), grp


def test_swap_all_copies_the_donor():
    base = _obs(0.0)
    donor = _obs(9.0)
    a = np.array([1])
    out = dp.swap_group(_one(base), _one(donor), "ALL", a, a, a, a)[0]
    assert np.array_equal(out, donor)


# --------------------------------------------------------------------------- #
# Extraction + distance
# --------------------------------------------------------------------------- #
def test_extract_group_widths_and_content():
    obs = _obs(0.0, own_block=1, opp_block=2)[None]
    own, opp = dp.active_block_indices(obs)
    assert dp.extract_group(obs, own, opp, "own_active_ids").shape == (1, _PDIM - _NUM)
    assert dp.extract_group(obs, own, opp, "own_active_numeric").shape == (1, _NUM)
    assert dp.extract_group(obs, own, opp, "opp_active").shape == (1, _PDIM)
    assert dp.extract_group(obs, own, opp, "own_bench").shape == (1, 5 * _PDIM)
    # opp_active extraction equals the raw opp active block.
    b = _blk(_TEAM + 2)
    assert np.array_equal(dp.extract_group(obs, own, opp, "opp_active")[0], obs[0, b : b + _PDIM])


def test_group_distance_zero_when_identical_positive_when_shifted():
    live = np.ones((10, 4), np.float32)
    assert dp.group_distance(live, live.copy()) == 0.0
    shard = np.zeros((10, 4), np.float32)
    assert dp.group_distance(live, shard) > 0.0


# --------------------------------------------------------------------------- #
# Pairing + assembly
# --------------------------------------------------------------------------- #
def test_pack_pair_and_matched_pairs():
    live_own = np.array([1, 1, 2, 3])
    live_opp = np.array([5, 5, 6, 7])
    shard_own = np.array([1, 1, 1, 2, 9])
    shard_opp = np.array([5, 5, 5, 6, 9])
    live_key = dp.pack_pair(live_own, live_opp)
    shard_key = dp.pack_pair(shard_own, shard_opp)
    pairs = dp.matched_pairs(live_key, shard_key, min_pair=2)
    # (1,5) has 2 live + 3 shard -> kept; (2,6) has 1 each -> dropped; (3,7) not in shard.
    assert set(pairs) == {dp.pack_pair(np.array(1), np.array(5)).item()}
    li, si = pairs[dp.pack_pair(np.array(1), np.array(5)).item()]
    assert li.tolist() == [0, 1] and si.tolist() == [0, 1, 2]


def test_assemble_base_donor_caps_and_cycles_donors():
    # 4 live rows, 2 shard rows of one pair; deletion base = live, donor = shard.
    pairs = {10: (np.array([0, 1, 2, 3]), np.array([0, 1]))}
    live_obs = np.stack([_obs(float(i)) for i in range(4)])
    shard_obs = np.stack([_obs(100.0 + i) for i in range(2)])
    z = np.zeros(4, np.int64)
    tbl = dp.assemble_base_donor(
        pairs, live_obs, np.ones((4, 26), bool), z, z,
        shard_obs, np.zeros(2, np.int64), np.zeros(2, np.int64),
        "live", cap_base=3, cap_donor=2, rng=np.random.default_rng(0),
    )
    assert tbl["base_obs"].shape[0] == 3  # capped to cap_base
    assert tbl["donor_obs"].shape[0] == 3  # one donor per base row (cycled from 2)


# --------------------------------------------------------------------------- #
# Verdict logic
# --------------------------------------------------------------------------- #
def test_frac_explained_orientation():
    assert dp.frac_explained(0.25, 0.5) == 0.5
    assert dp.frac_explained(0.0, 0.0) == 0.0  # zero gap guarded
    assert dp.frac_explained(-0.25, 0.5) == -0.5


def _pg(**kw):
    base = {g: {"bc": 0.0, "offrl": 0.0, "mean": 0.0} for g in dp.GROUPS}
    for g, v in kw.items():
        base[g] = {"bc": v, "offrl": v, "mean": v}
    return base


def test_carrier_verdict_sole_carrier():
    v = dp.carrier_verdict(_pg(own_active_ids=0.7, moves=0.2), 0.5, 0.25)
    assert v["verdict"] == "own_active_ids"
    assert v["sole_carriers"] == ["own_active_ids"]
    assert v["ranking"][0]["group"] == "own_active_ids"


def test_carrier_verdict_requires_family_consistency():
    pg = _pg()
    pg["own_active_ids"] = {"bc": 0.9, "offrl": 0.1, "mean": 0.5}  # one family below secondary
    v = dp.carrier_verdict(pg, 0.5, 0.25)
    assert v["verdict"] != "own_active_ids"  # inconsistent across families


def test_carrier_verdict_cumulative_set():
    v = dp.carrier_verdict(_pg(own_active_ids=0.3, moves=0.3), 0.5, 0.25)
    assert v["verdict"] == "own_active_ids+moves"
    assert v["cumulative_frac"] >= 0.5


def test_carrier_verdict_no_carrier():
    v = dp.carrier_verdict(_pg(own_active_ids=0.15, moves=0.1), 0.5, 0.25)
    assert v["verdict"] == "NO_CARRIER"

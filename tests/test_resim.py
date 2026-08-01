"""Offline tests for re-simulated full-fidelity replay ingestion (M6 v2).

The core path is exercised *without node*: a hand-built event stream (public-log msgs
plus real gen9 ``|request|`` JSON and ``choice`` markers) drives the same reconstruction
the node driver feeds in production. This proves the v2 win -- full own-team POV (size 3,
not v1's reveal-as-you-go) and strict request-based labels for moves, tera, and switches
(switches v1 had to drop). A second, environment-gated test runs the real node driver
end-to-end when the vendored simulator + node are available.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from lategame.data.resim import (
    _DEFAULT_SHOWDOWN,
    _parse_inputlog_meta,
    _reconstruct_pov_resim,
    resim_replays,
)
from lategame.data.reward import RewardWeights
from lategame.features.encoder import OBS_DIM, OBS_VERSION

_WEIGHTS = RewardWeights()


def _mon(ident: str, details: str, active: bool, moves: list[str], tera: str) -> dict:
    return {
        "ident": ident,
        "details": details,
        "condition": "100/100",
        "active": active,
        "stats": {"atk": 120, "def": 120, "spa": 150, "spd": 120, "spe": 150},
        "moves": moves,
        "baseAbility": "static",
        "item": "lightball",
        "pokeball": "pokeball",
        "ability": "static",
        "commanding": False,
        "reviving": False,
        "teraType": tera,
        "terastallized": "",
    }


def _move(mid: str) -> dict:
    return {
        "move": mid.title(), "id": mid, "pp": 24, "maxpp": 24, "target": "normal", "disabled": False
    }


# (ident, details, move-ids, teraType) -- the party in slot order (slot 1, 2, 3).
_MONS = {
    "pikachu": ("p1: Pikachu", "Pikachu, L84, M",
                ["thunderbolt", "voltswitch", "surf", "irontail"], "Electric"),
    "bulbasaur": ("p1: Bulbasaur", "Bulbasaur, L88, M",
                  ["tackle", "vinewhip", "synthesis", "growth"], "Grass"),
    "charizard": ("p1: Charizard", "Charizard, L80, M",
                  ["flamethrower", "airslash", "roost", "dragonpulse"], "Fire"),
}
_PARTY = ["pikachu", "bulbasaur", "charizard"]


def _request(active: str) -> str:
    """A full-team gen9 request with ``active`` (a species key) the active mon."""
    pokemon = [
        _mon(_MONS[s][0], _MONS[s][1], s == active, _MONS[s][2], _MONS[s][3]) for s in _PARTY
    ]
    active_block = {
        "moves": [_move(m) for m in _MONS[active][2]], "canTerastallize": _MONS[active][3]
    }
    payload = {"active": [active_block], "side": {"name": "Alice", "id": "p1", "pokemon": pokemon}}
    return "|request|" + json.dumps(payload)


# p1 = Alice (3 mons), full POV from turn 1. Decisions: Thunderbolt, switch to Charizard
# (party slot 3 -> a *switch* label, which v1 could not produce), Flamethrower as the new
# active, then a `default` that must be dropped. Alice wins.
_OPEN = "\n".join([
    "|player|p1|Alice|1|", "|player|p2|Bob|2|", "|teamsize|p1|3", "|teamsize|p2|3",
    "|gen|9", "|tier|[Gen 9] Random Battle", "|start",
    "|switch|p1a: Pikachu|Pikachu, L84, M|100/100",
    "|switch|p2a: Bulbasaur|Bulbasaur, L88, M|100/100", "|turn|1",
])

_EVENTS = [
    {"t": "msg", "d": _OPEN + "\n" + _request("pikachu")},
    {"t": "choice", "d": "move thunderbolt"},
    {"t": "msg", "d": "|turn|2\n" + _request("pikachu")},
    {"t": "choice", "d": "switch 3"},
    {"t": "msg", "d": "|switch|p1a: Charizard|Charizard, L80, M|100/100\n|turn|3\n"
                      + _request("charizard")},
    {"t": "choice", "d": "move flamethrower"},
    {"t": "msg", "d": "|turn|4\n" + _request("charizard")},
    {"t": "choice", "d": "default"},  # undecodable -> dropped, not labelled
    {"t": "msg", "d": "|win|Alice"},
]


def test_full_own_team_pov_and_strict_labels() -> None:
    battle, records, values, dropped = _reconstruct_pov_resim(_EVENTS, "Alice", "t-p1", 9, _WEIGHTS)
    assert battle.player_role == "p1"
    # The v1 fidelity bug is gone: all three own mons are known from the request at every
    # decision, not revealed one-at-a-time.
    assert len(battle.team) == 3
    assert {m.base_species for m in battle.team.values()} == {"pikachu", "bulbasaur", "charizard"}
    actions = [a for _, a, _ in records]
    # Canonical slots (Build 5): Thunderbolt sorts to slot 2 of Pikachu's declared four
    # (irontail, surf, thunderbolt, voltswitch) -> action 8; switch to Charizard (party
    # slot 3) -> switch index 2 (v1 dropped switches); Flamethrower sorts to slot 2 of
    # Charizard's (airslash, dragonpulse, flamethrower, roost) -> 8. `default` dropped.
    assert actions == [8, 2, 8]
    assert dropped == 1
    assert battle.finished and battle.won


def test_masks_keep_taken_action_legal() -> None:
    _, records, _, _ = _reconstruct_pov_resim(_EVENTS, "Alice", "t-p1", 9, _WEIGHTS)
    for _, action, mask in records:
        assert mask.shape == (26,) and mask.dtype == bool
        assert mask[action]  # strict, request-derived mask keeps the taken action legal


def test_resim_dataset_shapes_rewards_and_dones() -> None:
    replay = {"id": "t", "inputlog": ">player p1 {\"name\":\"Alice\"}"}  # unused obs source
    # Drive the aggregation directly off the hand-built POV (no node needed).
    from lategame.data.ingest import _ShardBuilder

    builder = _ShardBuilder()
    battle, records, values, _ = _reconstruct_pov_resim(_EVENTS, "Alice", "t-p1", 9, _WEIGHTS)
    builder.add_pov(battle, records, values, _WEIGHTS)
    rl, bc = builder.finalize("gen9randombattle", 0.99, _WEIGHTS)

    assert rl.obs.shape == (3, OBS_DIM) and rl.obs.dtype == np.float32
    assert np.isfinite(rl.obs).all()
    assert rl.action.shape == (3,) and rl.mask.shape == (3, 26)
    assert rl.done.tolist() == [False, False, True]
    assert rl.reward[-1] > 0.0  # winner: terminal carries the +victory jump
    assert bc is not None and bc.obs.shape == (3, OBS_DIM)  # Alice won -> all turns kept
    assert replay["id"] == "t"


def test_parse_inputlog_meta() -> None:
    inputlog = "\n".join([
        ">start {\"formatid\":\"gen9randombattle\",\"seed\":\"sodium,abc\"}",
        ">player p1 {\"name\":\"gummiworm\",\"rating\":2058}",
        ">player p2 {\"name\":\"Qwuartz\"}",
        ">p1 move psyblade",
    ])
    assert _parse_inputlog_meta(inputlog) == ("gummiworm", "Qwuartz", 9)
    # Falls back to defaults on a bare inputlog.
    assert _parse_inputlog_meta(">p1 move tackle") == ("p1", "p2", 9)


# --------------------------------------------------------------------------- #
# Environment-gated end-to-end test through the real node driver + simulator.
# --------------------------------------------------------------------------- #

_START = (
    '>start {"formatid":"gen9randombattle",'
    '"seed":"sodium,b8493732a42936a1fd687ddf7988dbbc","rated":"Rated battle"}'
)
_INPUTLOG = "\n".join([
    _START,
    ">player p1 {\"name\":\"gummiworm\",\"seed\":\"sodium,a3572989c38af097ba39a7fa38627767\"}",
    ">player p2 {\"name\":\"Qwuartz\",\"seed\":\"sodium,d9f6a625ddf93230a09fa7bc6f0239d0\"}",
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
def test_resim_end_to_end_through_driver() -> None:
    replay = {"id": "sample", "inputlog": _INPUTLOG}
    rl, bc, stats = resim_replays([replay], weights=_WEIGHTS)

    assert stats.parsed == 1 and stats.skipped_replays == 0
    assert stats.episodes == 2  # both POVs reconstructed
    assert stats.turns > 20 and stats.drop_rate < 0.2
    assert rl.obs.shape[1] == OBS_DIM
    assert rl.mask[np.arange(len(rl.action)), rl.action].all()  # every taken action legal
    assert ((rl.action >= 0) & (rl.action < 6)).sum() > 0  # switches captured (v1 dropped these)


def test_save_schema_parity(tmp_path) -> None:
    from lategame.data.collect import save_rl
    from lategame.data.ingest import _ShardBuilder
    from lategame.data.rl_dataset import RLDataset

    builder = _ShardBuilder()
    battle, records, values, _ = _reconstruct_pov_resim(_EVENTS, "Alice", "t-p1", 9, _WEIGHTS)
    builder.add_pov(battle, records, values, _WEIGHTS)
    rl, _ = builder.finalize("gen9randombattle", 0.99, _WEIGHTS)
    path = tmp_path / "shard.npz"
    save_rl(rl, path)

    loaded = np.load(path, allow_pickle=False)
    assert str(loaded["obs_version"].item()) == OBS_VERSION
    assert int(loaded["obs_dim"].item()) == OBS_DIM
    assert len(RLDataset(path)) == 3

"""Tests for the R-PREDICT search stack (Lever 11).

Two layers: (1) ``determinize`` spec/digest shape checks run without node; (2) the forward
model (``reconstruct`` -> ``step``) runs the real node driver, env-gated like ``test_resim``,
since the vendored simulator (``third_party/``) is not committed. The full live search
(``choose_order`` over a server) is covered by ``scripts/rpredict_gate.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lategame.data.resim import _reconstruct_pov_resim
from lategame.data.reward import RewardWeights
from lategame.search.determinize import battle_to_spec, pokeenv_digest
from lategame.search.forward import _DEFAULT_SHOWDOWN
from tests.test_resim import _EVENTS

_HAS_NODE = shutil.which("node") is not None
_HAS_SIM = (Path(_DEFAULT_SHOWDOWN) / "dist").is_dir()


def test_determinize_spec_and_digest_shapes() -> None:
    battle, _, _, _ = _reconstruct_pov_resim(_EVENTS, "Alice", "t-p1", 9, RewardWeights())
    spec = battle_to_spec(battle, seed=1)
    assert spec["p1"]["team"] and len(spec["p1"]["team"]) == len(spec["p1"]["state"])
    assert 0 <= spec["p1"]["active"] < len(spec["p1"]["team"])
    assert spec["p2"]["fill"] == max(0, 6 - len(battle.opponent_team))
    # our full team is known, so every set carries its observed moves
    assert all("moves" in m for m in spec["p1"]["team"])

    digest = pokeenv_digest(battle)
    assert set(digest) >= {"turn", "weather", "terrain", "p1", "p2", "hazards"}
    for mon in digest["p1"].values():
        assert set(mon) == {"hp", "status", "fainted", "active", "boosts"}
        assert 0.0 <= mon["hp"] <= 1.0


@pytest.mark.skipif(not (_HAS_NODE and _HAS_SIM), reason="needs node + a built vendored simulator")
def test_forward_reconstruct_and_step() -> None:
    from lategame.search.forward import ForwardModel

    spec = {
        "seed": 7,
        "p1": {
            "team": [{"species": "dragonite", "moves": ["dragondance", "earthquake"]},
                     {"species": "toxapex", "moves": ["liquidation"]}],
            "state": [{"hp_frac": 0.5, "status": "par"}, {"hp_frac": 1.0}],
            "active": 0,
            "fill": 0,
        },
        "p2": {
            "team": [{"species": "garchomp", "moves": ["earthquake"]}],
            "state": [{"hp_frac": 0.8}],
            "active": 0,
            "fill": 5,
        },
        "field": {"weather": "raindance", "terrain": "", "pseudo": []},
        "hazards": {"p1": {"stealthrock": 1}, "p2": {"spikes": 2}},
        "turn": 9,
    }
    with ForwardModel() as fm:
        assert fm.ping()
        recon = fm.reconstruct(spec)
        digest = recon["digest"]
        assert digest["turn"] == 9 and digest["weather"] == "raindance"
        assert abs(digest["p1"]["dragonite"]["hp"] - 0.5) <= 0.02
        assert digest["p1"]["dragonite"]["status"] == "par"
        assert digest["hazards"]["p2"]["spikes"] == 2
        assert len(digest["p2"]) == 6  # filled to a full opponent team
        assert recon["p1_choices"] and recon["p2_choices"]

        mc = recon["p1_choices"][0]
        oc = next(c for c in recon["p2_choices"] if c["type"] == "move")
        res = fm.step(recon["state"], mc["choice"], oc["choice"])
        assert not res.get("illegal")
        assert "state" in res and (res["p1_delta"] or res.get("ended"))

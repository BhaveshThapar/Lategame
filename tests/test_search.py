"""Tests for the R-PREDICT search stack (Lever 11).

Two layers: (1) ``determinize`` spec/digest shape checks run without node; (2) the forward
model (``reconstruct`` -> ``step``) runs the real node driver, env-gated like ``test_resim``,
since the vendored simulator (``third_party/``) is not committed. The full live search
(``choose_order`` over a server) is covered by ``scripts/rpredict_gate.py``.
"""

from __future__ import annotations

from rotomai.data.resim import _reconstruct_pov_resim
from rotomai.data.reward import RewardWeights
from rotomai.search.determinize import battle_to_spec, pokeenv_digest
from tests.conftest import requires_showdown
from tests.test_resim import _EVENTS


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


@requires_showdown
def test_forward_reconstruct_and_step() -> None:
    from rotomai.search.forward import ForwardModel

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
        # depth-2 search recurses from the resulting node, so step() must also report the
        # legal choices there (driver frame: p1 = us), like reconstruct does.
        assert isinstance(res["p1_choices"], list) and isinstance(res["p2_choices"], list)


# --- server-free depth-limited recursion (Lever 12) ----------------------------------------
# _node_value's backup / opponent-aggregation / pruning are pure control flow once the forward
# model is mocked: terminal step results short-circuit before any poke-env node is built, and an
# empty delta makes _node_battle a no-op deepcopy (``_feed_line`` ignores non-"|" lines). So we
# drive it with a scripted fake ForwardModel + PolicyValue -- no node, no torch, no server.


class _FakeFM:
    """Scripted forward model: ``(state, p1, p2) -> res`` and a record of every step call."""

    def __init__(self, table: dict) -> None:
        self.table = table
        self.calls: list[tuple] = []

    def step(self, state, p1, p2):  # noqa: ANN001, ANN201
        self.calls.append((state, p1, p2))
        return self.table[(state, p1, p2)]


class _FakePV:
    v_max = 1.0
    v_min = -1.0

    def deepcopy_battle(self, b):  # noqa: ANN001, ANN201 -- identity; empty delta feeds nothing
        return b

    def value(self, node):  # noqa: ANN001, ANN201
        return 0.5

    def action_log_probs(self, node):  # noqa: ANN001, ANN201
        return {}


def _res(**kw):  # noqa: ANN002, ANN201
    base = {"state": kw.get("state", "S0"), "p1_delta": "", "p1_request": None}
    base.update(kw)
    return base


def test_node_value_depth2_backup_and_opponent_aggregation() -> None:
    from rotomai.search.expectimax import SearchConfig, _node_value

    # Two of our actions, two opponent moves; every step ends the game (no node needed).
    #   A: vs x -> we win (+1), vs y -> we lose (-1)
    #   B: vs x -> we lose (-1), vs y -> we lose (-1)
    fm = _FakeFM(
        {
            ("S0", "move A", "move x"): {"ended": True, "winner": "p1"},
            ("S0", "move A", "move y"): {"ended": True, "winner": "p2"},
            ("S0", "move B", "move x"): {"ended": True, "winner": "p2"},
            ("S0", "move B", "move y"): {"ended": True, "winner": "p2"},
        }
    )
    root = _res(
        p1_choices=[{"choice": "move A", "type": "move", "id": "a"},
                    {"choice": "move B", "type": "move", "id": "b"}],
        p2_choices=[{"choice": "move x", "type": "move", "id": "x"},
                    {"choice": "move y", "type": "move", "id": "y"}],
    )
    # top_k_my high -> no pruning; depth=1 here == one recursion ply then leaf (the depth-2 search).
    common = {"depth": 2, "shaped_coef": 0.0, "top_k_my": 10, "opp_cap_deep": 10}
    mean = SearchConfig(opp_aggregation="mean", **common)
    # expectimax: A=mean(+1,-1)=0, B=mean(-1,-1)=-1 -> best 0
    assert _node_value(fm, _FakePV(), object(), root, "p1", mean, depth=1) == 0.0

    worst = SearchConfig(opp_aggregation="min", **common)
    # minimax: A=min(+1,-1)=-1, B=min(-1,-1)=-1 -> best -1
    assert _node_value(_FakeFM(fm.table), _FakePV(), object(), root, "p1", worst, depth=1) == -1.0


def test_node_value_prunes_our_actions_to_top_k_my() -> None:
    from rotomai.search.expectimax import SearchConfig, _node_value

    class _Node:  # _to_order finds no match -> every prior -20 -> stable keep-first-k
        available_moves: list = []
        available_switches: list = []

    class _PV(_FakePV):
        def deepcopy_battle(self, b):  # noqa: ANN001, ANN201
            return _Node()

    fm = _FakeFM({("S0", "move A", None): {"ended": True, "winner": "p1"}})
    root = _res(
        p1_choices=[{"choice": "move A", "type": "move", "id": "a"},
                    {"choice": "move B", "type": "move", "id": "b"}],
        p2_choices=[],  # opp has no choice -> single [None] branch
    )
    cfg = SearchConfig(depth=2, shaped_coef=0.0, top_k_my=1, opp_cap_deep=3)
    assert _node_value(fm, _PV(), object(), root, "p1", cfg, depth=1) == 1.0
    # pruned to one action: only "move A" was stepped, never "move B".
    assert fm.calls == [("S0", "move A", None)]


def test_node_value_terminal_and_no_move_leaf() -> None:
    from rotomai.search.expectimax import SearchConfig, _node_value

    cfg = SearchConfig(depth=2, shaped_coef=0.0)
    fm = _FakeFM({})
    pv = _FakePV()
    call = lambda res: _node_value(fm, pv, object(), res, "p1", cfg, 1)  # noqa: E731
    # terminal short-circuits to v_max / v_min / 0 before any node is built
    assert call({"ended": True, "winner": "p1"}) == 1.0
    assert call({"ended": True, "winner": "p2"}) == -1.0
    assert call({"ended": True, "winner": None}) == 0.0
    assert call({"illegal": True}) is None
    # we can't move this ply (force-switch/wait) -> evaluate the leaf (pv.value, shaped off)
    leaf = _res(p1_choices=[], p2_choices=[])
    assert _node_value(fm, _FakePV(), object(), leaf, "p1", cfg, 1) == 0.5
    assert fm.calls == []  # never stepped


# --- format parameterization (Build 27 A1) --------------------------------------------------
@requires_showdown
def test_an_omitted_format_still_reconstructs_as_random_battles() -> None:
    """THE REGRESSION GUARD FOR EVERY PUBLISHED R-PREDICT NUMBER.

    `FORMAT` used to be a module constant pinned to gen9randombattle. It now comes from the spec,
    so a spec that omits it -- which is every caller before 2026-08 -- must still reconstruct
    exactly as before, right down to the driver sampling the RB pool to fill the hidden opponent.
    """
    from rotomai.search.forward import ForwardModel

    bare = {
        "seed": 7,
        "p1": {"team": [{"species": "dragonite", "moves": ["earthquake"]}],
               "state": [{"hp_frac": 1.0}], "active": 0, "fill": 0},
        "p2": {"team": [{"species": "garchomp", "moves": ["earthquake"]}],
               "state": [{"hp_frac": 1.0}], "active": 0, "fill": 5},
        "field": {"weather": "", "terrain": "", "pseudo": []},
        "hazards": {"p1": {}, "p2": {}},
        "turn": 1,
    }
    explicit = {**bare, "format": "gen9randombattle"}

    with ForwardModel() as fm:
        a = fm.reconstruct(bare)
        b = fm.reconstruct(explicit)
    # Same seed, same format => the determinization must be identical, not merely similar.
    assert a["digest"] == b["digest"]
    assert len(a["digest"]["p2"]) == 6  # RB still fills the hidden roster from its own pool


@requires_showdown
def test_a_teambuilt_format_never_invents_species() -> None:
    """On gen9ou the opponent's six are known from team preview, so there is nothing to sample --
    and there is no random-set generator to sample legally with. `fill` must be ignored rather
    than quietly producing a team the format would reject."""
    from rotomai.search.forward import ForwardModel

    spec = {
        "seed": 3,
        "format": "gen9ou",
        "p1": {"team": [{"species": "greattusk", "moves": ["earthquake"],
                         "ability": "protosynthesis", "item": "leftovers", "level": 100}],
               "state": [{"hp_frac": 1.0}], "active": 0, "fill": 0},
        "p2": {"team": [{"species": "kingambit", "moves": ["suckerpunch"],
                         "ability": "supremeoverlord", "item": "leftovers", "level": 100}],
               "state": [{"hp_frac": 1.0}], "active": 0, "fill": 5},  # fill must be IGNORED here
        "field": {"weather": "", "terrain": "", "pseudo": []},
        "hazards": {"p1": {}, "p2": {}},
        "turn": 1,
    }
    with ForwardModel() as fm:
        recon = fm.reconstruct(spec)
    assert len(recon["digest"]["p2"]) == 1, "a teambuilt format must not invent opponent species"
    assert recon["p1_choices"], "the reconstructed gen9ou battle must offer legal choices"

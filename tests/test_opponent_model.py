"""Server-free tests for the Lever 14 opponent models + probability-weighted expectimax.

No node, no torch, no server: the white-box model runs on plain poke-env ``Move``/``Pokemon``
(dex lookups only), and the weighted-expectimax backup is pure control flow once the forward
model is mocked (same ``_FakeFM``/``_FakePV`` trick as ``test_search``). The live end-to-end run
(opponent model inside search over a server) is covered by ``scripts/rpredict_oppmodel_gate.py``.
"""

from __future__ import annotations

from poke_env.battle import Move, Pokemon

from lategame.agents.heuristic_agent import heuristic_pick, matchup
from lategame.search.expectimax import SearchConfig, _combine, _node_value, _opp_weights
from lategame.search.opponent_model import (
    LearnedOpponent,
    WhiteBoxHeuristicOpponent,
    _match_choice,
    build_opponent_model,
)

# --- shared fakes --------------------------------------------------------------------------- #


class _FakeBattle:
    """Minimal OUR-POV battle: what the white-box opponent model reads off a node."""

    def __init__(self, opp_active, our_active, opp_team=None, gen=9) -> None:  # noqa: ANN001
        self.opponent_active_pokemon = opp_active
        self.active_pokemon = our_active
        self.opponent_team = opp_team or {}
        self.gen = gen


def _move_choices(*ids: str) -> list[dict]:
    return [{"choice": f"move {i + 1}", "type": "move", "id": m} for i, m in enumerate(ids)]


# --- heuristic_pick: the DRY core the white-box arm shares with HeuristicAgent --------------- #


def test_heuristic_pick_takes_highest_damage_move() -> None:
    attacker = Pokemon(9, species="garchomp")
    defender = Pokemon(9, species="pikachu")  # ground 2x, so EQ dominates
    moves = [Move("dragonclaw", 9), Move("earthquake", 9), Move("swordsdance", 9)]
    pick = heuristic_pick(attacker, defender, moves, [])
    assert pick is not None and pick[0] == "move" and pick[1].id == "earthquake"


def test_heuristic_pick_switches_when_move_weak_and_matchup_better() -> None:
    # pikachu into garchomp: thundershock is 0x (score 0 <= weak) and pikachu's matchup is awful,
    # so a bench tangrowth is a clear improvement -> switch out.
    pick = heuristic_pick(
        Pokemon(9, species="pikachu"),
        Pokemon(9, species="garchomp"),
        [Move("thundershock", 9)],
        [Pokemon(9, species="tangrowth")],
    )
    assert pick is not None and pick[0] == "switch" and pick[1].species == "tangrowth"


def test_heuristic_pick_keeps_strong_move_over_switch() -> None:
    # A strong move (EQ = 300 > weak threshold) is taken regardless of an available bench mon.
    pick = heuristic_pick(
        Pokemon(9, species="garchomp"),
        Pokemon(9, species="pikachu"),
        [Move("earthquake", 9)],
        [Pokemon(9, species="tangrowth")],
    )
    assert pick is not None and pick[0] == "move" and pick[1].id == "earthquake"


def test_matchup_sign() -> None:
    # Fire mon vs grass mon: strong offense (2x), resisted defense (0.5x) -> positive matchup.
    assert matchup(Pokemon(9, species="arcanine"), Pokemon(9, species="tangrowth")) > 0
    assert matchup(None, Pokemon(9, species="garchomp")) == 0.0


# --- WhiteBoxHeuristicOpponent: one-hot on the exact eval-opponent choice -------------------- #


def test_whitebox_is_one_hot_on_the_heuristic_move() -> None:
    battle = _FakeBattle(Pokemon(9, species="garchomp"), Pokemon(9, species="pikachu"))
    choices = _move_choices("dragonclaw", "earthquake")
    dist = WhiteBoxHeuristicOpponent().distribution(battle, choices)
    assert dist == {"move 2": 1.0}  # earthquake is choice 2


def test_whitebox_empty_choices() -> None:
    battle = _FakeBattle(Pokemon(9, species="garchomp"), Pokemon(9, species="pikachu"))
    assert WhiteBoxHeuristicOpponent().distribution(battle, []) == {}


def test_match_choice_maps_move_and_switch_and_none() -> None:
    move_choices = _move_choices("earthquake", "dragonclaw")
    assert _match_choice(("move", Move("dragonclaw", 9)), move_choices) == "move 2"
    switch_choices = [{"choice": "switch 3", "type": "switch", "id": "toxapex"}]
    assert _match_choice(("switch", Pokemon(9, species="toxapex")), switch_choices) == "switch 3"
    assert _match_choice(None, move_choices) is None
    assert _match_choice(("move", Move("closecombat", 9)), move_choices) is None  # not offered


# --- weight construction + aggregation ------------------------------------------------------ #


class _ScriptedOpp:
    """Returns a fixed distribution; ``needs_opp_pov`` False so no POV battle is built."""

    needs_opp_pov = False

    def __init__(self, dist: dict[str, float]) -> None:
        self._dist = dist

    def distribution(self, pov, choices, opp_pov=None):  # noqa: ANN001, ANN201
        return dict(self._dist)


def test_opp_weights_model_one_hot_collapses_to_single_branch() -> None:
    choices = _move_choices("a", "b", "c")
    cfg = SearchConfig(opp_aggregation="model", opp_cap=6)
    weights = _opp_weights(object(), choices, cfg, _ScriptedOpp({"move 2": 1.0}), cfg.opp_cap)
    assert weights == [(choices[1], 1.0)]


def test_opp_weights_model_keeps_top_cap_by_probability() -> None:
    choices = _move_choices("a", "b", "c")
    cfg = SearchConfig(opp_aggregation="model")
    opp = _ScriptedOpp({"move 1": 0.2, "move 2": 0.5, "move 3": 0.3})
    kept = _opp_weights(object(), choices, cfg, opp, cap=2)
    assert [c["choice"] for c, _ in kept] == ["move 2", "move 3"]  # top-2 by prob, ranked


def test_opp_weights_model_uniform_fallback_when_distribution_empty() -> None:
    choices = _move_choices("a", "b")
    cfg = SearchConfig(opp_aggregation="model")
    kept = _opp_weights(object(), choices, cfg, _ScriptedOpp({}), cap=6)
    assert len(kept) == 2 and abs(sum(w for _, w in kept) - 1.0) < 1e-9


def test_opp_weights_mean_reproduces_moves_first_uniform() -> None:
    # mean/min ignore the model: moves-first pool, uniform weights (the L11/L12 behavior).
    choices = _move_choices("a", "b") + [{"choice": "switch 3", "type": "switch", "id": "s"}]
    cfg = SearchConfig(opp_aggregation="mean")
    kept = _opp_weights(object(), choices, cfg, None, cap=6)
    assert [c["choice"] for c, _ in kept] == ["move 1", "move 2"]  # switch dropped
    assert all(abs(w - 0.5) < 1e-9 for _, w in kept)


def test_combine_weighted_mean_and_min() -> None:
    pairs = [(0.75, 1.0), (0.25, -1.0)]
    assert abs(_combine(pairs, "mean") - 0.5) < 1e-9  # (.75 - .25)/1.0
    assert abs(_combine(pairs, "model") - 0.5) < 1e-9  # model uses the same weighted mean
    assert _combine(pairs, "min") == -1.0


# --- integration: model weighting changes the depth-2 backup --------------------------------- #


class _FakeFM:
    def __init__(self, table: dict) -> None:
        self.table = table
        self.calls: list[tuple] = []

    def step(self, state, p1, p2):  # noqa: ANN001, ANN201
        self.calls.append((state, p1, p2))
        return self.table[(state, p1, p2)]


class _FakePV:
    v_max = 1.0
    v_min = -1.0

    def deepcopy_battle(self, b):  # noqa: ANN001, ANN201
        return b

    def value(self, node):  # noqa: ANN001, ANN201
        return 0.5

    def action_log_probs(self, node):  # noqa: ANN001, ANN201
        return {}


def test_node_value_model_aggregation_weights_by_opponent_policy() -> None:
    # Same board as test_search's depth-2 case:
    #   A: vs x -> win(+1), vs y -> lose(-1);  B: vs x/y -> lose(-1)
    # A one-hot opponent that always plays x makes A worth +1 (only the x branch counts),
    # where uniform expectimax rated A at mean(+1,-1)=0. So model aggregation *changes* the pick.
    fm = _FakeFM(
        {
            ("S0", "move A", "move x"): {"ended": True, "winner": "p1"},
            ("S0", "move A", "move y"): {"ended": True, "winner": "p2"},
            ("S0", "move B", "move x"): {"ended": True, "winner": "p2"},
            ("S0", "move B", "move y"): {"ended": True, "winner": "p2"},
        }
    )
    root = {
        "state": "S0",
        "p1_delta": "",
        "p1_request": None,
        "p1_choices": _move_choices("a", "b"),  # -> "move 1"/"move 2"; relabel below
        "p2_choices": [
            {"choice": "move x", "type": "move", "id": "x"},
            {"choice": "move y", "type": "move", "id": "y"},
        ],
    }
    root["p1_choices"] = [
        {"choice": "move A", "type": "move", "id": "a"},
        {"choice": "move B", "type": "move", "id": "b"},
    ]
    cfg = SearchConfig(opp_aggregation="model", depth=2, top_k_my=10, opp_cap_deep=10)
    opp = _ScriptedOpp({"move x": 1.0})
    val = _node_value(fm, _FakePV(), object(), root, "p1", cfg, depth=1, opp_model=opp)
    assert val == 1.0
    # one-hot collapsed opponent branching: only the "move x" responses were ever stepped.
    assert all(p2 == "move x" for _, _, p2 in fm.calls)


# --- LearnedOpponent + factory -------------------------------------------------------------- #


def test_learned_uniform_when_no_opp_pov() -> None:
    dist = LearnedOpponent(pv=None).distribution(object(), _move_choices("a", "b"), opp_pov=None)
    assert set(dist) == {"move 1", "move 2"} and all(abs(p - 0.5) < 1e-9 for p in dist.values())


def test_learned_softmax_over_actions(monkeypatch) -> None:  # noqa: ANN001
    import lategame.search.opponent_model as om

    choices = _move_choices("a", "b")
    # map choice -> action int by slot, and score action 0 far above action 1.
    monkeypatch.setattr(om, "_choice_to_action", lambda c, _b: int(c["choice"].split()[1] == "2"))

    class _PV:
        def action_log_probs(self, b):  # noqa: ANN001, ANN201
            return {0: 0.0, 1: -5.0}

    dist = LearnedOpponent(_PV()).distribution(object(), choices, opp_pov=object())
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["move 1"] > dist["move 2"]  # action 0 (higher log-prob) gets more mass


def test_build_opponent_model_factory() -> None:
    assert build_opponent_model("none") is None
    assert build_opponent_model("") is None
    assert isinstance(build_opponent_model("whitebox"), WhiteBoxHeuristicOpponent)
    assert isinstance(build_opponent_model("learned", pv=object()), LearnedOpponent)
    for bad in ("learned", "bogus"):
        try:
            build_opponent_model(bad) if bad == "bogus" else build_opponent_model("learned")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

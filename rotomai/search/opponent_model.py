"""Opponent models for probability-weighted expectimax (Lever 14 / R-PREDICT).

L11/L12 retired the search direction with one named residual: the *opponent model* was too
weak. The depth-1/2 expectimax modeled the foe as **uniform over every legal move** (``mean``)
or **worst-case** (``min``), yet the eval opponent is a *fixed, deterministic* rule
(``HeuristicAgent``). Hedging against a far more random/adversarial foe than the real one is a
textbook reason lookahead won't beat a reactive policy. This module replaces that assumption
with a *model of the opponent's actual policy*, so the search can do proper expectimax:
``agg = sum_oc p(oc) * v(oc)``.

Two arms:

* ``WhiteBoxHeuristicOpponent`` -- the eval opponent *is* the heuristic, so model it exactly by
  calling the shared ``agents.heuristic_agent.heuristic_pick`` on the modeled opponent's active
  mon + its determinized legal set (the driver's ``p2_choices``) against OUR active mon (read
  from the same our-POV battle the leaf already uses). Deterministic -> a one-hot distribution.
  Needs no opponent POV battle and is byte-identical to the eval opponent: the decisive upper
  bound on what a perfect opponent model buys search.

* ``LearnedOpponent`` -- a frozen policy applied to the reconstructed *opponent* POV (built via
  ``build_opp_pov`` from the driver's p2 stream), softmaxed over its legal choices. The
  generalizable arm; the gap white-box -> learned measures the cost of not knowing the foe.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from poke_env.battle import AbstractBattle, Move, Pokemon

# Driver choice label: {"choice": "move 1"|"switch 3", "type": "move"|"switch", "id": <str>}.
Choice = dict[str, Any]

# poke-env warns on every message it can't fully model; expected when replaying a POV, so quiet.
_OPP_LOGGER = logging.getLogger("rotomai.search.opp")
_OPP_LOGGER.setLevel(logging.CRITICAL)


class OpponentModel(Protocol):
    """Maps the opponent's legal choices at a node to a probability distribution.

    ``pov_battle`` is OUR-POV poke-env battle at the node (so ``opponent_active_pokemon`` is the
    modeled foe's active mon and ``active_pokemon`` is ours); ``opp_choices`` is the driver's
    ``p2_choices`` for that node; ``opp_pov`` (when ``needs_opp_pov``) is the reconstructed
    OPPONENT-POV battle to embed. Returns ``{choice_str: prob}`` over a subset of ``opp_choices``
    (models may zero out choices -- weighted expectimax renormalizes).
    """

    needs_opp_pov: bool

    def distribution(
        self,
        pov_battle: AbstractBattle,
        opp_choices: list[Choice],
        opp_pov: AbstractBattle | None = None,
    ) -> dict[str, float]: ...


def build_opp_pov(
    recon: dict[str, Any], tag: str = "opp", gen: int = 9
) -> AbstractBattle | None:
    """Fresh opponent-POV ``Battle`` from a driver ``reconstruct`` result (Lever 14 learned arm).

    Feeds the p2-perspective init log (leads revealed, own team full) + the state-corrected p2
    request, exactly as ``resim`` feeds a replayed POV. Returns ``None`` if it can't be built.
    """
    from poke_env.battle import Battle

    from rotomai.data.resim import _feed_line

    log = recon.get("p2_log") or ""
    req = recon.get("p2_request")
    if not log and not req:
        return None
    try:
        battle = Battle(battle_tag=tag, username="p2", gen=gen, logger=_OPP_LOGGER)
        for line in log.split("\n"):
            if line:
                _feed_line(battle, line)
        if req:
            _feed_line(battle, req)
    except Exception:  # noqa: BLE001 -- a bad POV must not crash search
        return None
    return battle


def opp_pov_child(
    parent_opp: AbstractBattle | None, res: dict[str, Any]
) -> AbstractBattle | None:
    """Advance an opponent-POV battle one ply: deepcopy + feed the step's p2 delta/request.

    Mirror of ``expectimax._node_battle`` for the p2 side (both driver-p2-framed, so no swap).
    """
    import copy

    from rotomai.data.resim import _feed_line

    if parent_opp is None:
        return None
    try:
        child = copy.deepcopy(parent_opp)
        for line in (res.get("p2_delta") or "").split("\n"):
            if line:
                _feed_line(child, line)
        req = res.get("p2_request")
        if req:
            _feed_line(child, req)
    except Exception:  # noqa: BLE001
        return None
    return child


def _gen_of(battle: AbstractBattle) -> int:
    return int(getattr(battle, "gen", 9) or 9)


def _move_objs(opp_choices: list[Choice], gen: int) -> list[Move]:
    """Build poke-env ``Move`` objects for the opponent's determinized legal moves."""
    moves: list[Move] = []
    for c in opp_choices:
        if c.get("type") != "move":
            continue
        try:
            moves.append(Move(str(c["id"]), gen))
        except Exception:  # noqa: BLE001 -- an unknown id must not abort the model
            continue
    return moves


def _switch_objs(
    opp_choices: list[Choice], pov_battle: AbstractBattle, gen: int
) -> list[Pokemon]:
    """Build ``Pokemon`` for the opponent's determinized bench (reusing revealed mons)."""
    from poke_env import to_id_str

    revealed = {to_id_str(m.species): m for m in pov_battle.opponent_team.values()}
    switches: list[Pokemon] = []
    for c in opp_choices:
        if c.get("type") != "switch":
            continue
        sid = str(c["id"])
        mon = revealed.get(to_id_str(sid))
        if mon is None:
            try:
                mon = Pokemon(gen=gen, species=sid)
            except Exception:  # noqa: BLE001
                continue
        switches.append(mon)
    return switches


class WhiteBoxHeuristicOpponent:
    """One-hot on the exact move the eval ``HeuristicAgent`` would pick (Lever 14 upper bound)."""

    needs_opp_pov = False

    def distribution(
        self,
        pov_battle: AbstractBattle,
        opp_choices: list[Choice],
        opp_pov: AbstractBattle | None = None,
    ) -> dict[str, float]:
        from rotomai.agents.heuristic_agent import heuristic_pick

        if not opp_choices:
            return {}
        gen = _gen_of(pov_battle)
        opp_active = pov_battle.opponent_active_pokemon  # attacker (the modeled foe)
        our_active = pov_battle.active_pokemon  # defender (us)
        moves = _move_objs(opp_choices, gen)
        switches = _switch_objs(opp_choices, pov_battle, gen)

        pick = heuristic_pick(opp_active, our_active, moves, switches)
        chosen = _match_choice(pick, opp_choices)
        if chosen is None:
            # Undecodable pick -> fall back to uniform so search still runs (never crash).
            p = 1.0 / len(opp_choices)
            return {c["choice"]: p for c in opp_choices}
        return {chosen: 1.0}


def _match_choice(pick: Any, opp_choices: list[Choice]) -> str | None:
    """Map a ``heuristic_pick`` result back to the driver ``opp_choices`` label."""
    from poke_env import to_id_str

    if pick is None:
        return None
    kind, obj = pick
    if kind == "move":
        mid = to_id_str(obj.id)
        for c in opp_choices:
            if c.get("type") == "move" and to_id_str(str(c["id"])) == mid:
                return str(c["choice"])
    else:  # switch
        sid = to_id_str(obj.species)
        base = to_id_str(getattr(obj, "base_species", "") or "")
        for c in opp_choices:
            if c.get("type") == "switch" and to_id_str(str(c["id"])) in (sid, base):
                return str(c["choice"])
    return None


class LearnedOpponent:
    """Softmax of a frozen policy over the opponent's legal choices, on its reconstructed POV.

    ``pv`` is a ``search.expectimax.PolicyValue`` (reused as the opponent policy). ``opp_pov``
    resolves OUR-POV node battle -> the opponent's POV battle to embed (``build_opp_pov``); when
    it can't be built we fall back to uniform so search still runs.
    """

    needs_opp_pov = True

    def __init__(self, pv: Any, temperature: float = 1.0) -> None:
        self._pv = pv
        self._temp = max(1e-3, float(temperature))

    def distribution(
        self,
        pov_battle: AbstractBattle,
        opp_choices: list[Choice],
        opp_pov: AbstractBattle | None = None,
    ) -> dict[str, float]:
        import math

        if not opp_choices:
            return {}
        if opp_pov is None:
            p = 1.0 / len(opp_choices)
            return {c["choice"]: p for c in opp_choices}
        logp = self._pv.action_log_probs(opp_pov)  # {action_int: log-prob}
        raw: dict[str, float] = {}
        for c in opp_choices:
            a = _choice_to_action(c, opp_pov)
            raw[str(c["choice"])] = logp.get(a, -20.0) if a is not None else -20.0
        m = max(raw.values())
        exps = {k: math.exp((v - m) / self._temp) for k, v in raw.items()}
        z = sum(exps.values()) or 1.0
        return {k: v / z for k, v in exps.items()}


def _choice_to_action(choice: Choice, opp_battle: AbstractBattle) -> int | None:
    """Map a driver ``p2_choices`` label to the opponent battle's integer action."""
    from poke_env import to_id_str
    from poke_env.player import SingleBattleOrder

    from rotomai.features.action_space import order_to_action

    order = None
    if choice.get("type") == "move":
        mid = to_id_str(str(choice["id"]))
        for mv in opp_battle.available_moves:
            if to_id_str(mv.id) == mid:
                order = SingleBattleOrder(mv)
                break
    else:
        sid = to_id_str(str(choice["id"]))
        for mon in opp_battle.available_switches:
            if to_id_str(mon.species) == sid or to_id_str(mon.base_species) == sid:
                order = SingleBattleOrder(mon)
                break
    if order is None:
        return None
    try:
        return order_to_action(order, opp_battle)
    except Exception:  # noqa: BLE001
        return None


def build_opponent_model(kind: str, pv: Any = None) -> OpponentModel | None:
    """Factory from a ``ROTOMAI_SEARCH_OPP_MODEL`` tag (``none``/``whitebox``/``learned``)."""
    if kind in ("", "none"):
        return None
    if kind == "whitebox":
        return WhiteBoxHeuristicOpponent()
    if kind == "learned":
        if pv is None:
            raise ValueError("learned opponent model needs a PolicyValue")
        return LearnedOpponent(pv)
    raise ValueError(f"unknown opponent model {kind!r}")

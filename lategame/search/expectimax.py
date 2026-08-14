"""Depth-limited expectimax over the forward model (Lever 11 R-PREDICT / Lever 12 depth-2).

Layered on the FROZEN GREEN checkpoint at decision time (no retrain). For each of our legal
actions we determinize the hidden opponent (K samples), fork the reconstructed state, step a
hypothetical (our action, opponent action), and evaluate the resulting node -- either as a
shaped leaf (depth-1, Lever 11) or by recursing ``cfg.depth - 1`` plies further (Lever 12),
maximizing over our policy-pruned actions and aggregating the opponent (``mean`` = uniform
expectimax, ``min`` = minimax). We average over determinizations, then pick the best action --
optionally blended with the policy prior so search only overrides the base when decisive.

Why depth-2: depth-1 is provably near-equivalent to the policy when the leaf is the same value
the policy trained against (the Lever 11 AMBER). The shaped term (HP/faints) only becomes
discriminative one ply deeper, where search sees 2-ply sequences (switch -> they KO -> revenge)
the 1-ply reactive policy cannot represent. Root expands all our actions; deeper plies prune to
the top-k by policy prior and cap opponent branching to keep the node-RPC cost tractable.

Leaf evaluation reuses the validated path: deepcopy the live poke-env battle and feed the
fork's clean fog-of-war delta (resim's ``_feed_line``) -> ``embed_battle`` -> V. The forward
fork/step is bit-for-bit faithful (Gate A) and reconstruction is ~100% observable-faithful
(the recon mini-gate); the irreducible approximation is the determinized opponent.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, SingleBattleOrder

from lategame.data.resim import _feed_line
from lategame.data.reward import RewardWeights, digest_state_value, state_value
from lategame.features.action_space import action_mask
from lategame.features.encoder import embed_battle
from lategame.search.determinize import battle_to_spec
from lategame.search.forward import ForwardModel

if TYPE_CHECKING:
    from lategame.search.opponent_model import OpponentModel

_WEIGHTS = RewardWeights()


class LeafPolicy(Protocol):
    """What the search actually needs from a model: a leaf value, priors, and a battle deepcopy.

    Named as an interface rather than pinned to ``PolicyValue`` because the VGC ceiling probe has
    no trained doubles model to supply one -- ``ShapedOnlyPolicy`` satisfies this and scores leaves
    from the shaped state-value alone.
    """

    #: Terminal leaf values, on whatever scale ``value`` returns.
    v_min: float
    v_max: float
    #: Policy-blend lambda; 0 disables blending, which is the only sane value with no policy.
    cfg_blend: float

    def deepcopy_battle(self, battle: AbstractBattle) -> AbstractBattle: ...

    def value(self, battle: AbstractBattle) -> float: ...

    def action_log_probs(self, battle: AbstractBattle) -> dict[int, float]: ...

    def digest_value(self, res: dict[str, Any], shaped_coef: float) -> float | None:
        """Leaf value straight from a step result's digest, or ``None`` to use the battle path."""
        ...


@dataclass
class SearchConfig:
    determinizations: int = 1  # opponent-team samples averaged per decision
    opp_aggregation: str = "mean"  # "mean" uniform | "min" worst-case | "model" p-weighted (L14)
    opp_cap: int = 6  # cap opponent branching at the ROOT (their moves first)
    policy_blend: float = 0.0  # lambda in [0,1]: mix z-scored search value with log-pi prior
    shaped_coef: float = 0.0  # leaf = V + shaped_coef * shaped state-value (tactical term)
    depth: int = 1  # plies of lookahead (1 = the L11 depth-1 path, unchanged)
    top_k_my: int = 3  # prune OUR actions by policy prior at deep plies (root expands all)
    opp_cap_deep: int = 3  # cap opponent branching at deep plies (root keeps opp_cap)
    seed: int = 0
    #: Cap on OUR branching at the ROOT. 0 = unlimited, which is what singles has always done and
    #: what every published R-PREDICT number used. DOUBLES NEEDS IT: the driver enumerates JOINT
    #: actions, ~70 on a real VGC turn against ~10 in singles, and the root is the one ply that
    #: expands everything -- so depth-2 there costs ~50x a singles decision and the gate would not
    #: finish. Ranked by the same static fallback `_top_k_by_prior` uses when there is no policy.
    root_cap: int = 0


def _opp_weights(
    pov_battle: AbstractBattle,
    opp_choices: list[dict[str, Any]],
    cfg: SearchConfig,
    opp_model: OpponentModel | None,
    cap: int,
    opp_pov: AbstractBattle | None = None,
) -> list[tuple[dict[str, Any] | None, float]]:
    """The opponent branches to expand, with weights (Lever 14).

    ``model`` aggregation asks ``opp_model`` for p(oc) over the *full* legal set (so a modeled
    switch is representable), keeps the top-``cap`` by probability, and returns those weights;
    ``mean``/``min`` reproduce L11/L12 exactly -- uniform weights over the moves-first,
    ``cap``-capped pool. ``[(None, 1.0)]`` when the side can't move this ply.
    """
    result: list[tuple[dict[str, Any] | None, float]] = []
    if not opp_choices:
        return [(None, 1.0)]
    cap = max(1, cap)
    if cfg.opp_aggregation == "model" and opp_model is not None:
        probs = opp_model.distribution(pov_battle, opp_choices, opp_pov)
        ranked = sorted(opp_choices, key=lambda c: probs.get(c["choice"], 0.0), reverse=True)
        for c in ranked[:cap]:
            w = probs.get(c["choice"], 0.0)
            if w > 0.0:
                result.append((c, w))
        if result:
            return result
        w = 1.0 / min(len(opp_choices), cap)  # model gave nothing -> uniform fallback
        for c in opp_choices[:cap]:
            result.append((c, w))
        return result
    pool = ([c for c in opp_choices if c["type"] == "move"] or opp_choices)[:cap]
    w = 1.0 / len(pool)
    for c in pool:
        result.append((c, w))
    return result


def _combine(evaluated: list[tuple[float, float]], aggregation: str) -> float:
    """Aggregate ``(weight, value)`` pairs: ``min`` = worst-case, else weight-normalized mean."""
    if aggregation == "min":
        return min(v for _, v in evaluated)
    num = sum(w * v for w, v in evaluated)
    den = sum(w for w, _ in evaluated)
    return num / den if den > 0 else statistics.fmean([v for _, v in evaluated])


def _swap_roles(text: str) -> str:
    """Swap p1<->p2 protocol tokens (the sim labels us p1; the live battle may be p2)."""
    return text.replace("p1", "\x00").replace("p2", "p1").replace("\x00", "p2")


def _node_battle(
    pv: LeafPolicy, live: AbstractBattle, res: dict[str, Any], role: str
) -> AbstractBattle:
    """Poke-env battle at the post-step node: deepcopy ``live`` + feed the fog-of-war delta.

    Reuses the validated leaf path (resim's ``_feed_line``: ``|request|`` -> ``parse_request``,
    which also populates ``available_moves``/``available_switches`` for the action mask). The
    delta/request are role-swapped when the live battle labels us p2 (the driver always p1=us).
    """
    delta = res.get("p1_delta") or ""
    req = res.get("p1_request")
    if role == "p2":
        delta = _swap_roles(delta)
        req = _swap_roles(req) if req else req
    node = pv.deepcopy_battle(live)
    for line in delta.split("\n"):
        _feed_line(node, line)
    if req:
        _feed_line(node, req)
    return node


def _leaf_eval(pv: LeafPolicy, node: AbstractBattle, shaped_coef: float) -> float:
    """Leaf value: outcome ``V`` + ``shaped_coef`` * shaped state-value (tactical term).

    The outcome value head V is a coarse long-horizon estimator (nearly flat at one ply, so
    pure-V search over-switches); the shaped state-value adds the immediate tactical signal
    (chip the opponent, preserve our HP) that a shallow lookahead needs to discriminate.
    """
    v = pv.value(node)
    if shaped_coef:
        v += shaped_coef * state_value(node, _WEIGHTS)
    return v


def _top_k_by_prior(
    choices: list[dict[str, Any]], node: AbstractBattle, pv: LeafPolicy, k: int
) -> list[dict[str, Any]]:
    """Keep the ``k`` choices the GREEN policy rates highest at ``node`` (always returns >= 1)."""
    if k <= 0 or len(choices) <= k:
        return choices
    logp = pv.action_log_probs(node)
    if not logp:
        # No policy to prune with (ShapedOnlyPolicy). Ranking every choice equally would make the
        # cut arbitrary, and on doubles the joint list is ~70 long, so what survives matters.
        # Attacking is the better default at a shallow depth -- a shaped leaf reads HP, and
        # switching never changes it on the ply that pays for the switch.
        return sorted(choices, key=lambda c: c.get("type") != "move")[:k]

    def prior(label: dict[str, Any]) -> float:
        a = _choice_action(label, node)
        return logp.get(a, -20.0) if a is not None else -20.0

    return sorted(choices, key=prior, reverse=True)[:k]


def _node_value(
    fm: ForwardModel,
    pv: LeafPolicy,
    live: AbstractBattle,
    res: dict[str, Any],
    role: str,
    cfg: SearchConfig,
    depth: int,
    opp_model: OpponentModel | None = None,
    live_opp: AbstractBattle | None = None,
) -> float | None:
    """Backed-up value of the node reached by ``res``, looking ``depth`` plies further ahead.

    ``depth <= 0`` evaluates the shaped leaf. Otherwise recurse one ply: maximize over our
    policy-pruned actions, aggregating the opponent's responses with ``opp_aggregation``
    (``mean`` = uniform expectimax, ``min`` = minimax, ``model`` = ``opp_model``-weighted
    expectimax, Lever 14). ``live_opp`` is the parent opponent-POV battle (learned arm), advanced
    one ply here. ``None`` if this branch was illegal.
    """
    if res.get("illegal"):
        return None
    if res.get("ended"):
        winner = res.get("winner") or ""
        if winner == "p1":
            return pv.v_max
        if winner == "p2":
            return pv.v_min
        return 0.0
    if depth <= 0:
        # A leaf that can be scored from the digest skips building a poke-env node entirely --
        # deepcopy + delta replay is ~170x the RPC that produced the digest. Interior nodes still
        # need a real battle (the opponent model and the action priors both read one), but at
        # depth 2 the leaves are 4 of every 5 nodes, so this is where the cost lives.
        from_digest = pv.digest_value(res, cfg.shaped_coef)
        if from_digest is not None:
            return from_digest
    node = _node_battle(pv, live, res, role)
    if depth <= 0:
        return _leaf_eval(pv, node, cfg.shaped_coef)

    my2 = res.get("p1_choices") or []
    if not my2:  # we can't move this ply (force-switch / wait) -> evaluate as a leaf
        return _leaf_eval(pv, node, cfg.shaped_coef)
    node_opp = _advance_opp_pov(opp_model, live_opp, res)
    opp_weights = _opp_weights(
        node, res.get("p2_choices") or [], cfg, opp_model, cfg.opp_cap_deep, node_opp
    )
    my2 = _top_k_by_prior(my2, node, pv, cfg.top_k_my)

    best: float | None = None
    for mc in my2:
        evaluated: list[tuple[float, float]] = []
        for oc, w in opp_weights:
            res2 = fm.step(res["state"], mc["choice"], oc["choice"] if oc else None)
            v = _node_value(fm, pv, node, res2, role, cfg, depth - 1, opp_model, node_opp)
            if v is not None:
                evaluated.append((w, v))
        if not evaluated:
            continue
        agg = _combine(evaluated, cfg.opp_aggregation)
        best = agg if best is None else max(best, agg)
    return best if best is not None else _leaf_eval(pv, node, cfg.shaped_coef)


def _advance_opp_pov(
    opp_model: OpponentModel | None, live_opp: AbstractBattle | None, res: dict[str, Any]
) -> AbstractBattle | None:
    """Advance the opponent-POV battle one ply, only when the model needs it (learned arm)."""
    if opp_model is None or not getattr(opp_model, "needs_opp_pov", False) or live_opp is None:
        return None
    from lategame.search.opponent_model import opp_pov_child

    return opp_pov_child(live_opp, res)


def choose_order(
    fm: ForwardModel,
    pv: LeafPolicy,
    battle: AbstractBattle,
    cfg: SearchConfig,
    opp_model: OpponentModel | None = None,
) -> BattleOrder | None:
    """Pick an action by depth-limited expectimax; ``None`` if search can't run (caller falls
    back). ``opp_model`` (Lever 14) enables ``opp_aggregation="model"`` probability-weighting."""
    role = battle.player_role or "p1"
    values: dict[str, list[float]] = {}
    labels: dict[str, dict[str, Any]] = {}

    for d in range(max(1, cfg.determinizations)):
        spec = battle_to_spec(battle, seed=cfg.seed * 1000 + d)
        try:
            recon = fm.reconstruct(spec)
        except RuntimeError:
            continue
        state = recon["state"]
        my_choices = recon.get("p1_choices") or []
        opp_choices = recon.get("p2_choices") or []
        root_opp = _root_opp_pov(opp_model, recon, battle)
        opp_weights = _opp_weights(battle, opp_choices, cfg, opp_model, cfg.opp_cap, root_opp)
        if not my_choices:
            continue
        if cfg.root_cap:
            my_choices = _top_k_by_prior(my_choices, battle, pv, cfg.root_cap)

        for mc in my_choices:
            key = mc["choice"]
            labels[key] = mc
            evaluated: list[tuple[float, float]] = []
            for oc, w in opp_weights:
                res = fm.step(state, mc["choice"], oc["choice"] if oc else None)
                v = _node_value(fm, pv, battle, res, role, cfg, cfg.depth - 1, opp_model, root_opp)
                if v is not None:
                    evaluated.append((w, v))
            if not evaluated:
                continue
            values.setdefault(key, []).append(_combine(evaluated, cfg.opp_aggregation))

    if not values:
        return None
    scored = {k: statistics.fmean(v) for k, v in values.items()}

    if cfg.policy_blend > 0:
        scored = _blend_policy_prior(scored, labels, pv, battle)

    best = max(scored, key=lambda k: scored[k])
    return _to_order(labels[best], battle)


def _root_opp_pov(
    opp_model: OpponentModel | None, recon: dict[str, Any], battle: AbstractBattle
) -> AbstractBattle | None:
    """Reconstruct the opponent-POV battle at the search root, only for the learned arm."""
    if opp_model is None or not getattr(opp_model, "needs_opp_pov", False):
        return None
    from lategame.search.opponent_model import build_opp_pov

    gen = int(getattr(battle, "gen", 9) or 9)
    return build_opp_pov(recon, tag=f"opp-{battle.battle_tag}", gen=gen)


def _blend_policy_prior(
    scored: dict[str, float],
    labels: dict[str, dict[str, Any]],
    pv: LeafPolicy,
    battle: AbstractBattle,
) -> dict[str, float]:
    """Add lambda * log-pi(prior) to z-scored search values so scales are comparable."""
    logp = pv.action_log_probs(battle)  # dict: action-int -> log prob
    vals = list(scored.values())
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) or 1.0
    out: dict[str, float] = {}
    lam = pv.cfg_blend
    for k, v in scored.items():
        a = _choice_action(labels[k], battle)
        prior = logp.get(a, -20.0) if a is not None else -20.0
        out[k] = (1 - lam) * ((v - mean) / std) + lam * prior
    return out


def _choice_action(label: dict[str, Any], battle: AbstractBattle) -> int | None:
    """Map a driver choice label back to this battle's integer action (for the policy prior)."""
    from lategame.features.action_space import order_to_action

    order = _to_order(label, battle)
    if order is None:
        return None
    try:
        return order_to_action(order, battle)
    except Exception:  # noqa: BLE001
        return None


def _slot_order(
    part: dict[str, Any], moves: list[Any], switches: list[Any]
) -> SingleBattleOrder | None:
    """One slot's poke-env order from a driver choice part (matched by move id / species)."""
    from poke_env import to_id_str
    from poke_env.player import DefaultBattleOrder

    if part["type"] == "pass":
        return DefaultBattleOrder()
    if part["type"] == "move":
        for mv in moves:
            if mv.id == part["id"]:
                target = part.get("target")
                return (
                    SingleBattleOrder(mv)
                    if target is None
                    else SingleBattleOrder(mv, move_target=int(target))
                )
        return None
    for mon in switches:
        if to_id_str(mon.species) == part["id"] or to_id_str(mon.base_species) == part["id"]:
            return SingleBattleOrder(mon)
    return None


def _to_order(label: dict[str, Any], battle: AbstractBattle) -> BattleOrder | None:
    """Build a poke-env order from a driver choice label (matched by move id / species).

    On doubles the driver emits JOINT labels carrying a ``parts`` list -- one entry per slot, each
    with its own move target. Both halves must decode or the order is dropped: a half-formed
    doubles order is rejected by the server, which answers with a default move rather than an
    error, so search would silently lose the turn it just spent computing.
    """
    from poke_env.player import DoubleBattleOrder

    parts = label.get("parts")
    if not parts:
        return _slot_order(label, list(battle.available_moves), list(battle.available_switches))

    orders: list[SingleBattleOrder] = []
    for slot, part in enumerate(parts):
        moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
        switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
        order = _slot_order(part, list(moves), list(switches))
        if order is None:
            return None
        orders.append(order)
    return DoubleBattleOrder(orders[0], orders[1]) if len(orders) == 2 else None


class ShapedOnlyPolicy:
    """A ``PolicyValue`` stand-in for formats with no trained model: shaped leaves, no priors.

    The VGC ceiling probe needs an M2 -- "what does near-optimal search achieve here?" -- and on
    gen9-RB that leg is what turned a suggestive M1 band into a FORMAT_BOUND verdict. But RB's M2
    evaluated leaves with GREEN's trained value head, and **no doubles value head exists**. So the
    leaf here is the shaped state-value alone (HP/faints, ``data.reward.state_value``), which needs
    no encoder and therefore no singles assumption.

    THIS IS A WEAKER INSTRUMENT THAN RB's M2, and the asymmetry is pre-registered rather than
    discovered afterwards: a WIN is decisive (search beating the heuristic proves skill headroom
    above both scripted bots), while a NULL is suggestive only, because a shallower leaf could
    produce it on its own.
    """

    #: A win/loss leaf on the shaped scale. `state_value` is bounded well inside this, so a
    #: terminal node dominates any non-terminal one -- which is the ordering search needs.
    v_min = -100.0
    v_max = 100.0
    cfg_blend = 0.0

    def __init__(self) -> None:
        import copy

        self._copy = copy

    def deepcopy_battle(self, battle: AbstractBattle) -> AbstractBattle:
        return self._copy.deepcopy(battle)

    def value(self, battle: AbstractBattle) -> float:
        return 0.0  # the whole signal is `shaped_coef * state_value` -- see _leaf_eval

    def action_log_probs(self, battle: AbstractBattle) -> dict[int, float]:
        return {}  # no priors; _top_k_by_prior falls back to its static ranking

    def digest_value(self, res: dict[str, Any], shaped_coef: float) -> float | None:
        """The whole point of this policy: score the leaf without a poke-env battle.

        Returns ``None`` only if the driver did not supply a digest (an older driver), so the
        caller silently falls back to the battle path rather than losing the node.
        """
        digest = res.get("digest")
        if not digest:
            return None
        return shaped_coef * digest_state_value(digest, _WEIGHTS, winner=res.get("winner"))


class PolicyValue:
    """Wraps the frozen GREEN model: policy logits, V(s), and leaf deepcopy, torch hidden inside."""

    def __init__(self, checkpoint_path: str, policy_blend: float = 0.0) -> None:
        import copy

        import torch

        from lategame.features.encoder import OBS_DIM, OBS_VERSION
        from lategame.model.actor_critic import value_from_logits, value_support
        from lategame.model.factory import build_model
        from lategame.model.policy import masked_logits

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("obs_version") != OBS_VERSION or ckpt.get("input_dim") != OBS_DIM:
            raise ValueError("search checkpoint encoder mismatch -- retrain")
        model = build_model(ckpt)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._torch = torch
        self._copy = copy
        self._model = model
        self._masked = masked_logits
        self._value_from_logits = value_from_logits
        self.v_min = float(ckpt["v_min"])
        self.v_max = float(ckpt["v_max"])
        self._centers = value_support(self.v_min, self.v_max, int(ckpt["n_bins"]))
        self.cfg_blend = policy_blend

    def deepcopy_battle(self, battle: AbstractBattle) -> AbstractBattle:
        return self._copy.deepcopy(battle)

    def digest_value(self, res: dict[str, Any], shaped_coef: float) -> float | None:
        """Not available: this leaf is ``V(s)`` from ``embed_battle``, which needs a real battle."""
        return None

    def value(self, battle: AbstractBattle) -> float:
        torch = self._torch
        obs = torch.from_numpy(embed_battle(battle)).float().unsqueeze(0)
        with torch.no_grad():
            _, value_logits = self._model(obs)
            return float(self._value_from_logits(value_logits, self._centers)[0].item())

    def action_log_probs(self, battle: AbstractBattle) -> dict[int, float]:
        torch = self._torch
        obs = torch.from_numpy(embed_battle(battle)).float().unsqueeze(0)
        mask = torch.from_numpy(action_mask(battle)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._model(obs)
            logp = torch.log_softmax(self._masked(logits, mask), dim=1)[0]
        return {i: float(logp[i].item()) for i in range(logp.shape[0])}

    def greedy_order(self, battle: AbstractBattle) -> BattleOrder:
        """Greedy policy action (the GREEN base) -- the search fallback + comparison baseline."""
        from lategame.features.action_space import action_to_order

        torch = self._torch
        obs = torch.from_numpy(embed_battle(battle)).float().unsqueeze(0)
        mask = torch.from_numpy(action_mask(battle)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._model(obs)
            action = int(self._masked(logits, mask).argmax(dim=1).item())
        return action_to_order(action, battle)

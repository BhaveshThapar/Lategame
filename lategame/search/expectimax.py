"""Depth-1 expectimax over the forward model (Lever 11 / R-PREDICT).

Layered on the FROZEN GREEN checkpoint at decision time (no retrain). For each of our legal
actions we determinize the hidden opponent (K samples), fork the reconstructed state, step a
hypothetical (our action, opponent action), and evaluate the leaf with the GREEN value head;
we aggregate over the opponent's actions (``mean`` = uniform expectimax, ``min`` = worst-case)
and over determinizations, then pick the best action -- optionally blended with the policy
prior so search only overrides the base policy when the lookahead is decisive.

Leaf evaluation reuses the validated path: deepcopy the live poke-env battle and feed the
fork's clean fog-of-war delta (resim's ``_feed_line``) -> ``embed_battle`` -> V. The forward
fork/step is bit-for-bit faithful (Gate A) and reconstruction is ~100% observable-faithful
(the recon mini-gate); the irreducible approximation is the determinized opponent.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder

from lategame.data.resim import _feed_line
from lategame.data.reward import RewardWeights, state_value
from lategame.features.action_space import action_mask
from lategame.features.encoder import embed_battle
from lategame.search.determinize import battle_to_spec
from lategame.search.forward import ForwardModel

_WEIGHTS = RewardWeights()


@dataclass
class SearchConfig:
    determinizations: int = 1  # opponent-team samples averaged per decision
    opp_aggregation: str = "mean"  # "mean" (uniform expectimax) | "min" (worst-case)
    opp_cap: int = 6  # cap opponent branching (their moves first)
    policy_blend: float = 0.0  # lambda in [0,1]: mix z-scored search value with log-pi prior
    shaped_coef: float = 0.0  # leaf = V + shaped_coef * shaped state-value (tactical term)
    seed: int = 0


def _swap_roles(text: str) -> str:
    """Swap p1<->p2 protocol tokens (the sim labels us p1; the live battle may be p2)."""
    return text.replace("p1", "\x00").replace("p2", "p1").replace("\x00", "p2")


def _leaf_value(
    pv: PolicyValue, live: AbstractBattle, res: dict[str, Any], role: str, shaped_coef: float
) -> float | None:
    """Value of the post-step leaf: terminal -> v_max/v_min/0, else V + shaped_coef * shaped.

    The outcome value head V is a coarse long-horizon estimator (nearly flat at one ply, so
    pure-V search over-switches); the shaped state-value adds the immediate tactical signal
    (chip the opponent, preserve our HP) that makes a one-ply lookahead discriminative.
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
    delta = res.get("p1_delta") or ""
    req = res.get("p1_request")
    if role == "p2":
        delta = _swap_roles(delta)
        req = _swap_roles(req) if req else req
    leaf = pv.deepcopy_battle(live)
    for line in delta.split("\n"):
        _feed_line(leaf, line)
    if req:
        _feed_line(leaf, req)
    v = pv.value(leaf)
    if shaped_coef:
        v += shaped_coef * state_value(leaf, _WEIGHTS)
    return v


def choose_order(
    fm: ForwardModel, pv: PolicyValue, battle: AbstractBattle, cfg: SearchConfig
) -> BattleOrder | None:
    """Pick an action by depth-1 expectimax; ``None`` if search can't run (caller falls back)."""
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
        opp_moves = [c for c in opp_choices if c["type"] == "move"] or opp_choices
        opp_moves = opp_moves[: max(1, cfg.opp_cap)]
        if not my_choices:
            continue

        for mc in my_choices:
            key = mc["choice"]
            labels[key] = mc
            leaves: list[float] = []
            for oc in opp_moves or [None]:
                res = fm.step(state, mc["choice"], oc["choice"] if oc else None)
                v = _leaf_value(pv, battle, res, role, cfg.shaped_coef)
                if v is not None:
                    leaves.append(v)
            if not leaves:
                continue
            agg = min(leaves) if cfg.opp_aggregation == "min" else statistics.fmean(leaves)
            values.setdefault(key, []).append(agg)

    if not values:
        return None
    scored = {k: statistics.fmean(v) for k, v in values.items()}

    if cfg.policy_blend > 0:
        scored = _blend_policy_prior(scored, labels, pv, battle)

    best = max(scored, key=lambda k: scored[k])
    return _to_order(labels[best], battle)


def _blend_policy_prior(
    scored: dict[str, float],
    labels: dict[str, dict[str, Any]],
    pv: PolicyValue,
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


def _to_order(label: dict[str, Any], battle: AbstractBattle) -> BattleOrder | None:
    """Build a poke-env order from a driver choice label (matched by move id / species)."""
    from poke_env.player import SingleBattleOrder

    if label["type"] == "move":
        for mv in battle.available_moves:
            if mv.id == label["id"]:
                return SingleBattleOrder(mv)
        return None
    for mon in battle.available_switches:
        from poke_env import to_id_str

        if to_id_str(mon.species) == label["id"] or to_id_str(mon.base_species) == label["id"]:
            return SingleBattleOrder(mon)
    return None


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

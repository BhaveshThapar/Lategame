"""R-PREDICT search agent (Lever 11/12): depth-limited expectimax on the frozen GREEN checkpoint.

At each decision it runs ``search.expectimax.choose_order`` -- determinize the opponent, fork
the reconstructed state, step hypothetical (our action, opponent action) pairs, and evaluate
leaves with the GREEN value head -- and falls back to the greedy policy action whenever search
can't run (so a battle never crashes). Search is test-time only: the checkpoint is frozen.
Config is read from the environment so a gate can sweep arms without touching ``build_player``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, Player

if TYPE_CHECKING:
    from lategame.search.expectimax import SearchConfig

DEFAULT_CHECKPOINT = "checkpoints/offrl_scale_et_prior_s0.pt"
CHECKPOINT_ENV_VAR = "LATEGAME_SEARCH_CHECKPOINT"


def _config_from_env() -> SearchConfig:
    from lategame.search.expectimax import SearchConfig

    return SearchConfig(
        determinizations=int(os.environ.get("LATEGAME_SEARCH_DETERMINIZATIONS", "1")),
        opp_aggregation=os.environ.get("LATEGAME_SEARCH_OPP_AGG", "mean"),
        opp_cap=int(os.environ.get("LATEGAME_SEARCH_OPP_CAP", "6")),
        policy_blend=float(os.environ.get("LATEGAME_SEARCH_BLEND", "0.0")),
        shaped_coef=float(os.environ.get("LATEGAME_SEARCH_SHAPED", "0.0")),
        depth=int(os.environ.get("LATEGAME_SEARCH_DEPTH", "1")),
        top_k_my=int(os.environ.get("LATEGAME_SEARCH_TOPK_MY", "3")),
        opp_cap_deep=int(os.environ.get("LATEGAME_SEARCH_OPP_CAP_DEEP", "3")),
        seed=int(os.environ.get("LATEGAME_SEARCH_SEED", "0")),
    )


class SearchAgent(Player):
    def __init__(
        self,
        *args: object,
        checkpoint_path: str | Path | None = None,
        sample: bool = False,  # accepted for build_player parity; search is deterministic
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        from lategame.search.expectimax import PolicyValue, choose_order
        from lategame.search.forward import ForwardModel
        from lategame.search.opponent_model import build_opponent_model

        self._choose_order = choose_order

        path = Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))
        if not path.exists():
            raise FileNotFoundError(
                f"Search checkpoint not found at '{path}'. Train one (`lategame train-rl`) "
                f"or set {CHECKPOINT_ENV_VAR}."
            )
        self._cfg = _config_from_env()
        self._pv = PolicyValue(str(path), policy_blend=self._cfg.policy_blend)
        # Lever 14: an opponent policy for probability-weighted expectimax (opp_agg="model").
        # The learned arm reuses the same frozen GREEN policy as the modeled opponent.
        self._opp_model = build_opponent_model(
            os.environ.get("LATEGAME_SEARCH_OPP_MODEL", "none"), pv=self._pv
        )
        self._fm = ForwardModel()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not battle.available_moves and not battle.available_switches:
            return self.choose_default_move()
        order: BattleOrder | None = None
        try:
            order = self._choose_order(self._fm, self._pv, battle, self._cfg, self._opp_model)
        except Exception:  # noqa: BLE001 -- search must never crash a live battle
            order = None
        return order if order is not None else self._pv.greedy_order(battle)

    def __del__(self) -> None:
        fm = getattr(self, "_fm", None)
        if fm is not None:
            fm.close()

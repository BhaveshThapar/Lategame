"""On-policy rollout collection for PPO (M5 Phase 2).

Plays the current learner (a ``PPORecordingAgent``) against frozen opponents and
returns a flat ``RolloutBuffer`` of per-turn ``(obs, action, mask, reward, done,
old_log_prob, value)``. Rewards/dones reuse ``data.collect._battle_rewards`` verbatim
(shaped state-value diffs with the terminal victory jump folded into the last turn),
so the only new signals over offline collection are ``old_log_prob`` and ``value`` --
captured by the learner at action time.

Only the LEARNER's turns are kept: the opponent acts under a different policy, so its
turns are off-policy and must not enter an on-policy PPO update. Battles run with a
raised ``max_concurrent_battles`` so ``cross_evaluate`` keeps many in flight at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from poke_env.player import Player, cross_evaluate
from poke_env.teambuilder.teambuilder import Teambuilder

from lategame.agents.ppo_record import PPORecordMixin
from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import PlayerSpec, _battle_rewards
from lategame.data.reward import RewardWeights
from lategame.eval.arena import build_player, rollout_agent
from lategame.model.policy import factored_has_choice

# recs, rewards, log_probs, values, executed
_Episode = tuple[list, list[float], list[float], list[float], list[bool]]

#: Ceiling on the share of a rollout whose recorded action was not the executed one. Above this
#: the on-policy assumption is void for a large fraction of the buffer, so the run is stopped
#: rather than trained on a ratio whose denominator is wrong.
_MAX_INVALID_FRAC = 0.05


def decision_stats(mask: torch.Tensor, executed: Sequence[bool]) -> dict[str, float]:
    """How much of this rollout actually carries a policy gradient.

    A turn whose every slot has exactly one legal action has ``log pi == 0`` for all parameters --
    the masked distribution is a point mass -- so it contributes precisely zero to the surrogate
    and to the entropy bonus. Keeping such turns is right (the critic wants the state, and GAE
    needs the episode contiguous), but counting them in the DENOMINATOR of the reported means is
    not: on a loop-contaminated VGC shard they were 43% of all turns, which would divide the
    policy loss, the entropy, and the approx-KL alike and quietly disarm the trust region.
    """
    legal = mask.sum(dim=-1)
    has_choice = factored_has_choice(mask) if mask.dim() == 3 else legal > 1
    n_choice = int(has_choice.sum())
    mean_legal = float(legal[has_choice].float().mean()) if n_choice else 0.0
    n = max(1, int(mask.shape[0]))
    return {
        "decision_frac": float(has_choice.float().mean()),
        "n_decision": float(n_choice),
        "mean_legal": mean_legal,
        "invalid_frac": sum(1 for e in executed if not e) / n,
    }


@dataclass
class RolloutBuffer:
    """Flat, multi-episode on-policy rollout (every episode ends ``done=True``).

    ``action`` and ``mask`` carry the FORMAT'S shape, not a unified one: ``[N]`` + ``[N, 26]`` on
    singles, ``[N, 2]`` + ``[N, 2, 107]`` on doubles. Same convention as the offline shards
    (``data.collect.TrajectoryDataset``) -- flattening a doubles turn into one 11,449-way index
    would erase the factoring that makes the format trainable. ``old_log_prob`` is correspondingly
    the JOINT log-prob of the pair.
    """

    obs: torch.Tensor  # [N, OBS_DIM] float32
    action: torch.Tensor  # [N] or [N, 2] int64
    mask: torch.Tensor  # [N, n_actions] or [N, 2, slot_actions] bool
    reward: torch.Tensor  # [N] float32
    done: torch.Tensor  # [N] bool
    old_log_prob: torch.Tensor  # [N] float32, log pi_old(a|s) at action time
    value: torch.Tensor  # [N] float32, V(s) at action time
    # Whether the recorded action is the one the server was sent (doubles' non-strict decode can
    # silently fall back to a random legal order). Trails the other fields and defaults to None so
    # every existing positional construction still works; None means "all executed", which is
    # exactly true on singles.
    executed: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def executed_mask(self) -> torch.Tensor:
        """``executed``, or an all-True vector when the field was never populated."""
        if self.executed is None:
            return torch.ones(len(self), dtype=torch.bool)
        return self.executed


def _learner_episodes(player: PPORecordMixin, weights: RewardWeights) -> list[_Episode]:
    """``(recs, rewards, log_probs, values, executed)`` per finished battle, aligned by tag.

    ``_battle_rewards`` returns the same ``recs`` list object stored at
    ``player.records[tag]``, so we recover the tag (and its parallel log-prob / value
    lists) by object identity -- reusing the reward/done logic without duplicating it.
    """
    tag_by_recs = {id(recs): tag for tag, recs in player.records.items()}
    episodes: list[_Episode] = []
    for recs, rewards in _battle_rewards(cast(Player, player), weights):
        tag = tag_by_recs[id(recs)]
        episodes.append(
            (
                recs,
                rewards,
                player.log_probs[tag],
                player.value_estimates[tag],
                player.valid[tag],
            )
        )
    return episodes


async def collect_rollout(
    learner_checkpoint: str,
    opponents: list[PlayerSpec],
    games_per_opp: int,
    battle_format: str = DEFAULT_FORMAT,
    weights: RewardWeights | None = None,
    max_concurrent: int = 20,
    team: str | Teambuilder | None = None,
    loop_penalty: float = 0.0,
    max_battle_turns: int | None = None,
) -> RolloutBuffer:
    """Roll out ``learner_checkpoint`` vs each opponent; return the learner's turns.

    ``team`` (a shared ``TeamPool``) is required for teambuilt formats (gen9ou) and left
    ``None`` for Random Battles. ``loop_penalty`` arms the Build-14 LoopGuard on the learner
    and any learned (offrl/ppo) opponents so an absorbing switch loop can't stall a battle;
    ``build_player`` ignores it for fixed baselines. Both default to the RB no-op.
    """
    if not opponents:
        raise ValueError("Need at least one opponent to generate rollouts.")
    weights = weights or RewardWeights()

    obs_c: list[np.ndarray] = []
    act_c: list[int | np.ndarray] = []
    mask_c: list[np.ndarray] = []
    rew_c: list[float] = []
    done_c: list[bool] = []
    lp_c: list[float] = []
    val_c: list[float] = []
    exec_c: list[bool] = []
    dropped = 0

    learner_name = rollout_agent(battle_format)
    for opp in opponents:
        learner = cast(
            PPORecordMixin,
            build_player(
                learner_name,
                battle_format,
                checkpoint_path=learner_checkpoint,
                sample=True,
                max_concurrent_battles=max_concurrent,
                team=team,
                loop_penalty=loop_penalty,
                max_battle_turns=max_battle_turns,
            ),
        )
        learner._reward_weights = weights
        opponent = build_player(
            opp.name,
            battle_format,
            checkpoint_path=opp.checkpoint_path,
            sample=opp.sample,
            max_concurrent_battles=max_concurrent,
            team=team,
            loop_penalty=loop_penalty,
            max_battle_turns=max_battle_turns,
        )
        await cross_evaluate([cast(Player, learner), opponent], n_challenges=games_per_opp)
        for recs, rewards, log_probs, values, executed in _learner_episodes(learner, weights):
            for i, ((obs, action, mask), r, lp, v, ex) in enumerate(
                zip(recs, rewards, log_probs, values, executed, strict=True)
            ):
                obs_c.append(obs)
                act_c.append(action)
                mask_c.append(mask)
                rew_c.append(r)
                done_c.append(i == len(recs) - 1)
                lp_c.append(lp)
                val_c.append(v)
                exec_c.append(ex)
        dropped += learner.dropped

    if not obs_c:
        raise RuntimeError("No finished learner trajectories -- is the local server running?")

    mask_t = torch.from_numpy(np.stack(mask_c).astype(bool))
    stats = decision_stats(mask_t, exec_c)
    print(
        f"rollout: collected {len(obs_c)} learner turns from {int(np.sum(done_c))} episodes "
        f"({dropped} unlabelable turns dropped) | decision_frac {stats['decision_frac']:.4f} "
        f"({int(stats['n_decision'])} rows) legal/slot {stats['mean_legal']:.2f} "
        f"invalid_frac {stats['invalid_frac']:.4f}"
    )
    # An action that was sampled from the mask but could not be decoded means the executed order
    # was NOT the recorded one, so its importance ratio is meaningless. A few are tolerable (they
    # are zero-weighted); a systematic rate is evidence of a joint constraint the factored mask
    # does not model, which invalidates the on-policy assumption rather than merely costing rows.
    if stats["invalid_frac"] > _MAX_INVALID_FRAC:
        raise RuntimeError(
            f"{stats['invalid_frac']:.1%} of recorded turns did not execute as recorded "
            f"(ceiling {_MAX_INVALID_FRAC:.0%}). The decoder is falling back to a random legal "
            f"order at a rate that makes log pi_old wrong for a large fraction of the buffer -- "
            f"this is a codec bug or an unmodelled joint constraint, not a tolerable loss."
        )

    return RolloutBuffer(
        obs=torch.from_numpy(np.stack(obs_c).astype(np.float32)),
        # [N] on singles, [N, 2] on doubles -- same `np.asarray` idiom as data.collect.
        action=torch.from_numpy(np.asarray(act_c, dtype=np.int64)),
        mask=mask_t,
        reward=torch.tensor(rew_c, dtype=torch.float32),
        done=torch.tensor(done_c, dtype=torch.bool),
        old_log_prob=torch.tensor(lp_c, dtype=torch.float32),
        value=torch.tensor(val_c, dtype=torch.float32),
        executed=torch.tensor(exec_c, dtype=torch.bool),
    )

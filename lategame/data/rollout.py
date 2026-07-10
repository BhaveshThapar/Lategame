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

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from poke_env.player import cross_evaluate
from poke_env.teambuilder.teambuilder import Teambuilder

from lategame.agents.ppo_agent import PPORecordingAgent
from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import PlayerSpec, _battle_rewards
from lategame.data.reward import RewardWeights
from lategame.eval.arena import build_player

_Episode = tuple[list, list[float], list[float], list[float]]  # recs, rewards, log_probs, values


@dataclass
class RolloutBuffer:
    """Flat, multi-episode on-policy rollout (every episode ends ``done=True``)."""

    obs: torch.Tensor  # [N, OBS_DIM] float32
    action: torch.Tensor  # [N] int64
    mask: torch.Tensor  # [N, n_actions] bool
    reward: torch.Tensor  # [N] float32
    done: torch.Tensor  # [N] bool
    old_log_prob: torch.Tensor  # [N] float32, log pi_old(a|s) at action time
    value: torch.Tensor  # [N] float32, V(s) at action time

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def _learner_episodes(player: PPORecordingAgent, weights: RewardWeights) -> list[_Episode]:
    """``(recs, rewards, log_probs, values)`` per finished battle, aligned by tag.

    ``_battle_rewards`` returns the same ``recs`` list object stored at
    ``player.records[tag]``, so we recover the tag (and its parallel log-prob / value
    lists) by object identity -- reusing the reward/done logic without duplicating it.
    """
    tag_by_recs = {id(recs): tag for tag, recs in player.records.items()}
    episodes: list[_Episode] = []
    for recs, rewards in _battle_rewards(player, weights):
        tag = tag_by_recs[id(recs)]
        episodes.append((recs, rewards, player.log_probs[tag], player.value_estimates[tag]))
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
    act_c: list[int] = []
    mask_c: list[np.ndarray] = []
    rew_c: list[float] = []
    done_c: list[bool] = []
    lp_c: list[float] = []
    val_c: list[float] = []
    dropped = 0

    for opp in opponents:
        learner = cast(
            PPORecordingAgent,
            build_player(
                "ppo",
                battle_format,
                checkpoint_path=learner_checkpoint,
                sample=True,
                max_concurrent_battles=max_concurrent,
                team=team,
                loop_penalty=loop_penalty,
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
        )
        await cross_evaluate([learner, opponent], n_challenges=games_per_opp)
        for recs, rewards, log_probs, values in _learner_episodes(learner, weights):
            for i, ((obs, action, mask), r, lp, v) in enumerate(
                zip(recs, rewards, log_probs, values, strict=True)
            ):
                obs_c.append(obs)
                act_c.append(action)
                mask_c.append(mask)
                rew_c.append(r)
                done_c.append(i == len(recs) - 1)
                lp_c.append(lp)
                val_c.append(v)
        dropped += learner.dropped

    if not obs_c:
        raise RuntimeError("No finished learner trajectories -- is the local server running?")

    print(
        f"rollout: collected {len(obs_c)} learner turns from {int(np.sum(done_c))} episodes "
        f"({dropped} unlabelable turns dropped)"
    )

    return RolloutBuffer(
        obs=torch.from_numpy(np.stack(obs_c).astype(np.float32)),
        action=torch.tensor(act_c, dtype=torch.int64),
        mask=torch.from_numpy(np.stack(mask_c).astype(bool)),
        reward=torch.tensor(rew_c, dtype=torch.float32),
        done=torch.tensor(done_c, dtype=torch.bool),
        old_log_prob=torch.tensor(lp_c, dtype=torch.float32),
        value=torch.tensor(val_c, dtype=torch.float32),
    )

"""OU Pivot Build 6 -- live behavioral probe: what does the dead agent actually DO?

Build 5 fixed the move-slot codec (Gate B1 PASS, BC val-acc 0.647) yet v5 agents still
lose >= 94% of live games to RANDOM -- and pure BC craters identically, so the stall is
upstream of RL. The failure is currently invisible: ``action_to_order`` silently converts
any decode failure into a uniform-random legal move (``features/action_space.py`` except
branch), and poke-env logs server rejections at level 25, below the default WARNING
cutoff. This probe makes the live decision stream observable over serial games vs
``RandomPlayer`` and classifies the failure into pre-registered causes (precedence
a > b > c > d; all evidence reported regardless of which fires first):

  (a) decode/mask failure: silent random fallbacks, server rejections, all-False masks;
  (b) team-order mismatch (live half): packed upload order vs live ``team.values()`` --
      switch actions 0-5 and the six own-mon obs blocks both key off it, and switches
      are sent BY NAME so a mismatch produces zero rejections; only this direct check
      can see it. Train-side |poke| order is a separate build if this fires;
  (c) pathological-but-legal play: switch loops, one-move spam, collapsed entropy;
  (d) human-replay -> bot-eval distribution drift, by elimination: sane-looking play
      that loses.

Observe-only: every hook calls through and returns the original result unchanged. This
is a probe, not a kill-gate -- every cause outcome is informative; the exit code is
nonzero only on INCONCLUSIVE (too few decisions, or the random-mirror control outside
its sanity band).

    python scripts/behavior_probe.py \\
        --bc-checkpoint checkpoints/bc_gen9ou_v5_s0.pt \\
        --offrl-checkpoint checkpoints/offrl_gen9ou_v5_s0.pt --n 20
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import secrets
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import numpy as np
from poke_env import to_id_str
from poke_env.player import Player, RandomPlayer

from lategame.config import LOCAL_SERVER, OU_FORMAT, local_account
from lategame.teambuilding.pool import DEFAULT_OU_POOL, TeamPool

_RESULTS = Path("results/behavior_probe.json")
_DECISIONS_OUT = Path("results/behavior_probe_decisions.jsonl")
_TRANSCRIPT_DIR = Path("results/behavior_probe_transcripts")
_REPLAY_DIR = Path("results/behavior_probe_replays")
# Build-8: the captured live obs/mask sidecar (gitignored; large, regenerable). Aligned
# row-for-row with the decisions JSONL; scripts/drift_probe.py is the consumer.
_OBS_OUT = Path("results/behavior_probe_obs.npz")

# Classification thresholds (pre-registered; see module docstring for the tree).
FALLBACK_MAJOR = 0.20  # (a) primary: >= 1 in 5 decisions silently randomized
FALLBACK_MINOR = 0.02  # (a) contributing flag; a correct codec should sit at ~0
MAX_REJECTION_RATE = 0.02  # (a): [Invalid]+[Unavailable choice] per decision
MIN_ORDER_MATCH = 0.98  # (b): packed upload order == live team.values() order
MAX_SWITCH_FRACTION = 0.50  # (c): voluntary switches / voluntary decisions
MAX_SWITCH_RUN = 6  # (c): consecutive voluntary switch decisions in one battle
MAX_PING_PONG = 0.25  # (c): A->B->A patterns per voluntary switch
MAX_TOP_ACTION_SHARE = 0.60  # (c): one-move spam
MIN_ENTROPY_BITS = 1.5  # (c): action-histogram entropy (< ~3 effective actions)
BROKEN_WIN_RATE = 0.20  # (d): below this with a/b/c clean -> drift by elimination
MIN_DECISIONS = 200  # sanity: fewer recorded decisions -> INCONCLUSIVE
CONTROL_BAND = (0.30, 0.70)  # sanity: random-mirror win rate outside -> INCONCLUSIVE
TOP_K = 3


@dataclass
class Decision:
    """One live policy decision, as seen at the ``action_to_order`` boundary."""

    arm: str
    battle_tag: str
    turn: int
    forced: bool  # battle.force_switch -- excluded from voluntary-switch metrics
    action: int
    order_str: str
    fallback: bool  # the codec's except-branch random fallback fired
    legal_count: int
    mask_all_false: bool
    top_k: list[tuple[int, float]]  # (action, prob) over masked logits
    our_active: str
    our_hp: float
    opp_active: str
    opp_hp: float
    team_order: list[str]  # live team.values() species order at this decision
    repeat: int = 1  # identical consecutive records collapsed (rejection retries)


def _action_name(action: int) -> str:
    """Human label for an action index (switch0-5, move0-3, mega/z/dmax/tera 0-3)."""
    if action < 0:
        return {-1: "forfeit", -2: "default"}.get(action, str(action))
    if action < 6:
        return f"switch{action}"
    kind = ("move", "mega", "z", "dmax", "tera")[(action - 6) // 4]
    return f"{kind}{(action - 6) % 4}"


def _entropy_bits(hist: dict[int, int]) -> float:
    total = sum(hist.values())
    if total == 0:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in hist.values() if c)


def _max_switch_run(decisions: Sequence[Decision]) -> int:
    """Longest run of consecutive voluntary switch decisions within one battle.

    Forced decisions neither extend nor break a run: the pathology is the policy
    *choosing* to switch over and over, not faint replacements in between.
    """
    best = run = 0
    tag = None
    for d in decisions:
        if d.forced:
            continue
        if d.battle_tag != tag:
            tag, run = d.battle_tag, 0
        run = run + 1 if d.action < 6 else 0
        best = max(best, run)
    return best


@dataclass
class TwoCycleRow:
    """A flagged A->B->A voluntary switch: its absolute index in the decision stream
    (row-aligned with the obs sidecar) and the switch-back action completing the cycle."""

    index: int
    return_action: int  # the switch-back action index (== d.action), i.e. onto the just-left mon


class SwitchDecision(Protocol):
    """The four fields ``two_cycle_rows`` actually reads off a decision.

    ``Decision`` satisfies this structurally, but ``pingpong_probe`` deliberately rebuilds its
    rows from the JSONL sidecar rather than replaying the probe, so it passes a light stand-in
    carrying only these four. Typing against the protocol says what the function needs instead of
    demanding a full ``Decision`` it never looks at."""

    forced: bool
    action: int
    battle_tag: str
    team_order: list[str]


def two_cycle_rows(decisions: Sequence[SwitchDecision]) -> list[TwoCycleRow]:
    """Stream indices of A->B->A voluntary switches: the switch target equals the target two
    voluntary switches earlier in the same battle.

    Single-sources the ping-pong definition: ``_ping_pong_rate`` counts these rows and Build 13's
    ``pingpong_probe`` selects their obs (via the row-aligned sidecar) to localize what sustains the
    2-cycle. ``return_action`` is the recorded ``action`` -- for a flagged row that IS the move back
    onto the just-left mon, so it is the per-row index the localizer scores P(return) at.
    """
    targets: dict[str, list[str]] = {}
    rows: list[TwoCycleRow] = []
    for i, d in enumerate(decisions):
        if d.forced or d.action >= 6:
            continue
        target = d.team_order[d.action] if d.action < len(d.team_order) else str(d.action)
        seq = targets.setdefault(d.battle_tag, [])
        seq.append(target)
        if len(seq) >= 3 and seq[-1] == seq[-3]:
            rows.append(TwoCycleRow(index=i, return_action=d.action))
    return rows


def _ping_pong_rate(decisions: Sequence[Decision]) -> float:
    """A->B->A patterns per voluntary switch: the target equals the target two
    voluntary switches earlier in the same battle."""
    total = sum(1 for d in decisions if not d.forced and d.action < 6)
    return len(two_cycle_rows(decisions)) / total if total else 0.0


def _decision_metrics(
    decisions: Sequence[Decision],
    rejections: Sequence[dict[str, str]],
    battles: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    n = len(decisions)
    hist = Counter(d.action for d in decisions)
    voluntary = [d for d in decisions if not d.forced]
    finished = [b for b in battles if b.get("won") is not None]
    wins = sum(1 for b in finished if b["won"])
    top1 = [d.top_k[0][1] for d in decisions if d.top_k]
    return {
        "n_decisions": n,
        "n_raw_calls": sum(d.repeat for d in decisions),
        "n_battles": len(battles),
        "n_finished": len(finished),
        "win_rate": wins / len(finished) if finished else None,
        "mean_turns": float(np.mean([b["turns"] for b in battles])) if battles else 0.0,
        "fallback_rate": sum(d.fallback for d in decisions) / n if n else 0.0,
        "rejection_rate": len(rejections) / n if n else 0.0,
        "n_invalid": sum(1 for r in rejections if r["kind"] == "invalid"),
        "n_unavailable": sum(1 for r in rejections if r["kind"] == "unavailable"),
        "all_false_masks": sum(d.mask_all_false for d in decisions),
        "switch_fraction": sum(1 for d in decisions if d.action < 6) / n if n else 0.0,
        "voluntary_switch_fraction": (
            sum(1 for d in voluntary if d.action < 6) / len(voluntary) if voluntary else 0.0
        ),
        "max_switch_run": _max_switch_run(decisions),
        "ping_pong_rate": _ping_pong_rate(decisions),
        "top_action_share": max(hist.values()) / n if n else 0.0,
        "entropy_bits": _entropy_bits(dict(hist)),
        "mean_top1_prob": float(np.mean(top1)) if top1 else 0.0,
        "action_hist": {_action_name(a): c for a, c in sorted(hist.items())},
    }


def _species_match(a: str, b: str) -> bool:
    """Same species up to a forme suffix (packed 'zacian' vs live 'zaciancrowned')."""
    return a == b or a.startswith(b) or b.startswith(a)


def _order_metrics(
    packed: dict[str, list[str]],
    live: dict[str, list[str]],
    unstable: set[str],
) -> dict[str, Any]:
    """Candidate (b), live half: does the packed upload order match live team.values()?"""
    tags = [t for t in live if t in packed]
    mismatched: list[str] = []
    examples: list[dict[str, Any]] = []
    for t in tags:
        p, live_order = packed[t], live[t]
        if len(p) == len(live_order) and all(
            _species_match(a, b) for a, b in zip(p, live_order, strict=True)
        ):
            continue
        mismatched.append(t)
        if len(examples) < 3:
            examples.append({"battle": t, "packed": p, "live": live_order})
    bad = unstable & set(live)
    return {
        "n_battles_checked": len(tags),
        "n_packed_missing": len(live) - len(tags),
        "match_rate": (len(tags) - len(mismatched)) / len(tags) if tags else 1.0,
        "stable_rate": 1 - len(bad) / len(live) if live else 1.0,
        "mismatched_battles": mismatched,
        "mismatch_examples": examples,
        "unstable_battles": sorted(bad)[:3],
    }


def _train_baseline(shard: Path) -> dict[str, float] | None:
    """Action-distribution stats from the BC shard, for calibrating (c) thresholds.

    Shard actions include forced switches (no flag stored), which inflates the switch
    base a little -- comparing the live *voluntary* fraction against 2x this base is
    therefore conservative.
    """
    if not shard.exists():
        return None
    actions = np.load(shard)["action"]
    hist = Counter(int(a) for a in actions.tolist())
    return {
        "n": float(len(actions)),
        "switch_fraction": float((actions < 6).mean()),
        "entropy_bits": _entropy_bits(dict(hist)),
        "top_action_share": max(hist.values()) / len(actions),
    }


def _classify(
    m: dict[str, Any], order: dict[str, Any], train_switch_base: float | None
) -> dict[str, Any]:
    """The pre-registered decision tree. Precedence a > b > c > d for the primary
    cause; b is measured independently of a, c requires a and b clean, d is by
    elimination."""
    switch_bar = MAX_SWITCH_FRACTION
    if train_switch_base is not None:
        switch_bar = max(switch_bar, 2 * train_switch_base)
    a = (
        m["fallback_rate"] > FALLBACK_MAJOR
        or m["rejection_rate"] > MAX_REJECTION_RATE
        or m["all_false_masks"] > 0
    )
    b = order["match_rate"] < MIN_ORDER_MATCH or order["stable_rate"] < 1.0
    c_flags = {
        "switch_fraction": m["voluntary_switch_fraction"] > switch_bar,
        "switch_run": m["max_switch_run"] >= MAX_SWITCH_RUN,
        "ping_pong": m["ping_pong_rate"] > MAX_PING_PONG,
        "top_action_share": m["top_action_share"] > MAX_TOP_ACTION_SHARE,
        "entropy": m["entropy_bits"] < MIN_ENTROPY_BITS,
    }
    c = not a and not b and any(c_flags.values())
    d = not (a or b or c) and m["win_rate"] is not None and m["win_rate"] < BROKEN_WIN_RATE
    causes = {"a_decode_mask": a, "b_team_order": b, "c_pathological": c, "d_drift": d}
    return {
        "causes": causes,
        "c_flags": c_flags,
        "switch_fraction_bar": switch_bar,
        "fallback_minor": m["fallback_rate"] > FALLBACK_MINOR,
        "primary_cause": next((k for k, v in causes.items() if v), "NO-REPRO"),
    }


def _format_transcript(
    arm: str, tag: str, decisions: Sequence[Decision], info: dict[str, Any]
) -> str:
    order_note = "ORDER MATCH" if info.get("order_match") else "ORDER MISMATCH"
    lines = [
        f"=== {arm} | {tag} | won={info.get('won')} turns={info.get('turns')} "
        f"| preview={info.get('preview')}",
        f"team packed: {','.join(info.get('packed') or ['?'])} "
        f"| live: {','.join(info.get('live') or ['?'])} | {order_note}",
    ]
    for d in decisions:
        row = (
            f"T{d.turn:<3} {d.our_active} {d.our_hp:.0%} vs {d.opp_active} {d.opp_hp:.0%}"
            f" | legal={d.legal_count} | -> {_action_name(d.action)} '{d.order_str}'"
        )
        if d.top_k:
            row += f" p={d.top_k[0][1]:.2f}"
        alts = ", ".join(f"{_action_name(a)} {p:.2f}" for a, p in d.top_k[1:])
        if alts:
            row += f" | alt: {alts}"
        if d.forced:
            row += " | FORCED"
        if d.fallback:
            row += " | FALLBACK"
        if d.repeat > 1:
            row += f" (x{d.repeat})"
        lines.append(row)
    return "\n".join(lines) + "\n"


class _BehaviorProbe:
    """Accumulates the live decision stream; the hooks write, the metrics read."""

    def __init__(self) -> None:
        self.arm = ""
        self.decisions: list[Decision] = []
        self.rejections: list[dict[str, str]] = []
        self.previews: dict[str, str] = {}
        self.packed_orders: dict[str, list[str]] = {}
        self.live_orders: dict[str, list[str]] = {}
        self.unstable_tags: set[str] = set()
        self.current_tag: str | None = None
        self.fallback_hit = False
        self.pending: dict[str, Any] | None = None
        # Build-8 obs capture: the live obs/mask the model actually saw, one row per
        # kept decision (aligned index-for-index with ``self.decisions``). ``pending_*``
        # are staged by the embed_battle / masked_logits hooks and consumed on append.
        self.pending_obs: np.ndarray | None = None
        self.pending_mask: np.ndarray | None = None
        self.obs: list[np.ndarray | None] = []
        self.masks: list[np.ndarray | None] = []

    def record_decision(self, action: int, battle: Any, order: Any, fallback: bool) -> None:
        tag = str(battle.battle_tag)
        self.current_tag = tag
        pending = self.pending or {}
        self.pending = None
        team_order = [to_id_str(p.species) for p in battle.team.values()]
        if tag not in self.packed_orders:
            packed = getattr(battle, "_teambuilder_team", None) or []
            if packed:
                self.packed_orders[tag] = [
                    to_id_str(m.species or m.nickname or "?") for m in packed
                ]
        if tag not in self.live_orders:
            self.live_orders[tag] = team_order
        elif team_order != self.live_orders[tag]:
            self.unstable_tags.add(tag)
        our, opp = battle.active_pokemon, battle.opponent_active_pokemon
        d = Decision(
            arm=self.arm,
            battle_tag=tag,
            turn=int(battle.turn),
            forced=bool(battle.force_switch),
            action=action,
            order_str=str(order),
            fallback=fallback,
            legal_count=int(pending.get("legal_count", -1)),
            mask_all_false=bool(pending.get("all_false", False)),
            top_k=list(pending.get("top_k", [])),
            our_active=to_id_str(our.species) if our else "none",
            our_hp=float(our.current_hp_fraction) if our else 0.0,
            opp_active=to_id_str(opp.species) if opp else "none",
            opp_hp=float(opp.current_hp_fraction) if opp else 0.0,
            team_order=team_order,
        )
        last = self.decisions[-1] if self.decisions else None
        if (
            last
            and last.battle_tag == d.battle_tag
            and last.turn == d.turn
            and last.action == d.action
            and last.order_str == d.order_str
        ):
            last.repeat += 1  # rejection retry: same obs, don't duplicate the capture
        else:
            self.decisions.append(d)
            self.obs.append(self.pending_obs)
            self.masks.append(self.pending_mask)
        self.pending_obs = None
        self.pending_mask = None


class _FallbackSpy(Player):
    """Stand-in for ``action_space.Player`` that makes the silent fallback visible.

    Only ``choose_random_singles_move`` (the except-branch escape in
    ``action_to_order``) is overridden; ``create_order`` is inherited, so the codec's
    happy path is byte-identical. Only ``action_space``'s module global is rebound --
    the RandomPlayer opponent goes through the real class and never sees this.
    """

    probe: ClassVar[_BehaviorProbe | None] = None

    @staticmethod
    def choose_random_singles_move(battle: Any) -> Any:
        if _FallbackSpy.probe is not None:
            _FallbackSpy.probe.fallback_hit = True
        return Player.choose_random_singles_move(battle)


class _RejectionHandler(logging.Handler):
    """Captures poke-env's level-25 '[Invalid choice]'/'[Unavailable choice]' records.

    Attribution uses the probe's last-seen battle tag -- exact under serial play.
    """

    def __init__(self, probe: _BehaviorProbe) -> None:
        super().__init__(level=1)
        self._probe = probe

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "[Invalid choice]" in msg:
            kind = "invalid"
        elif "[Unavailable choice]" in msg:
            kind = "unavailable"
        else:
            return
        self._probe.rejections.append(
            {
                "arm": self._probe.arm,
                "kind": kind,
                "battle": self._probe.current_tag or "?",
                "message": msg,
            }
        )


@contextmanager
def _probe_agents(probe: _BehaviorProbe) -> Iterator[None]:
    """Install the observe-only hooks around the live decision path.

    MUST be entered before agent construction: the agents bind ``masked_logits`` in
    ``__init__``, so a later patch would miss them (guarded by the zero-decisions
    abort in ``_run_arms``).
    """
    import torch

    from lategame.agents import bc_agent, offline_rl_agent
    from lategame.features import action_space
    from lategame.model import policy

    orig_masked = policy.masked_logits
    orig_decode = bc_agent.action_to_order
    orig_player = action_space.Player
    orig_embed_bc = bc_agent.embed_battle
    orig_embed_offrl = offline_rl_agent.embed_battle

    def spy_masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = orig_masked(logits, mask)
        legal = int(mask.sum().item())
        probs = torch.softmax(out, dim=1)[0]
        top = torch.topk(probs, min(max(legal, 1), TOP_K))
        probe.pending = {
            "legal_count": legal,
            "all_false": legal == 0,
            "top_k": [
                (int(a), float(p))
                for p, a in zip(top.values.tolist(), top.indices.tolist(), strict=True)
            ],
        }
        # The strict live mask the model actually scored under (Build-8): Phase B
        # re-scores these obs and needs the *same* mask to reproduce live switch mass.
        probe.pending_mask = mask.detach().cpu().numpy().reshape(-1).astype(bool)
        return out

    def spy_embed(battle: Any) -> np.ndarray:
        # embed_battle runs first in choose_move (before masked_logits/action_to_order),
        # so stash the obs here; record_decision consumes it once the action is decoded.
        obs = orig_embed_bc(battle)
        probe.pending_obs = np.asarray(obs, dtype=np.float32)
        return obs

    def spy_decode(action: int, battle: Any) -> Any:
        probe.fallback_hit = False
        order = orig_decode(action, battle)
        probe.record_decision(action, battle, order, probe.fallback_hit)
        return order

    _FallbackSpy.probe = probe
    policy.masked_logits = spy_masked
    bc_agent.action_to_order = spy_decode
    offline_rl_agent.action_to_order = spy_decode
    bc_agent.embed_battle = spy_embed
    offline_rl_agent.embed_battle = spy_embed
    action_space.Player = _FallbackSpy  # type: ignore[misc]
    try:
        yield
    finally:
        _FallbackSpy.probe = None
        policy.masked_logits = orig_masked
        bc_agent.action_to_order = orig_decode
        offline_rl_agent.action_to_order = orig_decode
        bc_agent.embed_battle = orig_embed_bc
        offline_rl_agent.embed_battle = orig_embed_offrl
        action_space.Player = orig_player  # type: ignore[misc]


def _record_teampreview(player: Player, probe: _BehaviorProbe) -> None:
    """Log the (random) lead order each game without changing it."""
    orig = player.teampreview

    def wrapped(battle: Any) -> str:
        msg = orig(battle)
        probe.previews[str(battle.battle_tag)] = str(msg)
        return cast(str, msg)

    player.teampreview = wrapped  # type: ignore[method-assign]


def _build_probe_player(
    kind: str,
    battle_format: str,
    checkpoint: str | None,
    pool: TeamPool,
    replay_dir: Path | None,
    loop_penalty: float = 0.0,
) -> Player:
    # arena.build_player forwards neither log_level (needed to see the level-25
    # rejection messages) nor save_replays, so the probe builds players itself.
    from lategame.agents.bc_agent import BCAgent
    from lategame.agents.heuristic_agent import HeuristicAgent
    from lategame.agents.offline_rl_agent import OfflineRLAgent

    classes: dict[str, type[Player]] = {
        "bc": BCAgent,
        "offrl": OfflineRLAgent,
        "random": RandomPlayer,
        "heuristic": HeuristicAgent,  # Build-8: the real eval opponent, as an arm
    }
    extra: dict[str, object] = {"checkpoint_path": checkpoint} if checkpoint else {}
    if kind in ("bc", "offrl"):
        extra["loop_penalty"] = loop_penalty  # Build 14 decision-time anti-repetition
    return classes[kind](
        account_configuration=local_account(f"{kind[:10]}-{secrets.token_hex(3)}"),
        battle_format=battle_format,
        server_configuration=LOCAL_SERVER,
        team=pool,
        # Only the probed policies (bc/offrl) need the level-25 handler for rejections;
        # opponents (random/heuristic) are never hooked.
        log_level=25 if kind in ("bc", "offrl") else None,
        save_replays=str(replay_dir) if replay_dir else False,
        **extra,  # type: ignore[arg-type]
    )


async def _run_arms(
    arms: list[tuple[str, str, str | None, str]],
    args: argparse.Namespace,
    probe: _BehaviorProbe,
    teams: list[str],
) -> list[dict[str, Any]]:
    seeds = itertools.count()
    summaries = []
    for arm, kind, ckpt, opp_kind in arms:
        probe.arm = arm
        replay_dir = _REPLAY_DIR / arm if args.save_replays and kind != "random" else None
        player = _build_probe_player(
            kind, args.format, ckpt, TeamPool(teams, seed=next(seeds)), replay_dir,
            args.loop_penalty,
        )
        opponent = _build_probe_player(
            opp_kind, args.format, None, TeamPool(teams, seed=next(seeds)), None
        )
        if kind != "random":
            _record_teampreview(player, probe)
            player.logger.addHandler(_RejectionHandler(probe))
        timed_out = False
        try:
            # Not arena.evaluate_built: cross_evaluate calls reset_battles() on the way
            # out, wiping the per-battle won/turns the probe harvests below.
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=args.n),
                timeout=args.n * args.per_battle_timeout,
            )
        except TimeoutError:
            timed_out = True
        battles = [
            {
                "battle": tag,
                "won": b.won,
                "turns": int(b.turn),
                "preview": probe.previews.get(tag),
            }
            for tag, b in player.battles.items()
        ]
        if kind != "random" and not any(d.arm == arm for d in probe.decisions):
            raise SystemExit(
                f"arm '{arm}' recorded no decisions -- the hooks did not fire; was an "
                "agent constructed outside _probe_agents, or did every game stall "
                "before a request?"
            )
        summaries.append(
            {"arm": arm, "kind": kind, "opponent": opp_kind, "checkpoint": ckpt,
             "timed_out": timed_out, "battles": battles}
        )
        n_dec = sum(1 for d in probe.decisions if d.arm == arm)
        print(f"  arm {arm}: {len(battles)} battles, {n_dec} decisions"
              f"{' [TIMED OUT]' if timed_out else ''}")
    return summaries


def _write_transcripts(
    out_dir: Path, arm: str, probe: _BehaviorProbe, summary: dict[str, Any], order: dict[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mismatched = set(order["mismatched_battles"])
    for b in summary["battles"]:
        tag = b["battle"]
        decs = [d for d in probe.decisions if d.arm == arm and d.battle_tag == tag]
        if not decs:
            continue
        info = {
            "won": b["won"],
            "turns": b["turns"],
            "preview": b["preview"],
            "packed": probe.packed_orders.get(tag),
            "live": probe.live_orders.get(tag),
            "order_match": tag not in mismatched and tag not in probe.unstable_tags,
        }
        (out_dir / f"{arm}__{tag}.txt").write_text(
            _format_transcript(arm, tag, decs, info), encoding="utf-8"
        )


def _save_obs(path: Path, probe: _BehaviorProbe) -> None:
    """Persist the captured obs/mask sidecar, aligned with ``probe.decisions``.

    Only rows whose obs *and* mask were captured are written (every live decision
    captures both -- the None case is the direct-unit-test path, never a live game).
    Row order matches the decisions JSONL, and each row carries its arm/action/tag so
    Phase B can select bc-vs-offrl loop states without re-reading the JSONL.
    """
    rows = [
        (o, m, d)
        for o, m, d in zip(probe.obs, probe.masks, probe.decisions, strict=True)
        if o is not None and m is not None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(path, obs=np.zeros((0, 0), np.float16))
        print(f"obs sidecar EMPTY -> {path}")
        return
    obs = np.stack([o for o, _, _ in rows]).astype(np.float16)
    masks = np.stack([m for _, m, _ in rows]).astype(bool)
    np.savez_compressed(
        path,
        obs=obs,
        mask=masks,
        arm=np.array([d.arm for _, _, d in rows]),
        action=np.array([d.action for _, _, d in rows], dtype=np.int64),
        battle_tag=np.array([d.battle_tag for _, _, d in rows]),
        turn=np.array([d.turn for _, _, d in rows], dtype=np.int64),
        forced=np.array([d.forced for _, _, d in rows], dtype=bool),
    )
    print(f"obs sidecar {obs.shape} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="Battles per arm (serial)")
    ap.add_argument("--bc-checkpoint", default="", help="BC checkpoint to probe")
    ap.add_argument("--offrl-checkpoint", default="", help="Offline-RL checkpoint to probe")
    ap.add_argument(
        "--control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run a random-vs-random control arm (threshold sanity)",
    )
    ap.add_argument("--team-pool", default=str(DEFAULT_OU_POOL), help="Packed eval teampool")
    ap.add_argument("--format", default=OU_FORMAT, help="Battle format")
    ap.add_argument(
        "--train-shard",
        default="data/gen9ou_v5_bc.npz",
        help="BC shard for the train action-distribution baseline (optional)",
    )
    ap.add_argument(
        "--loop-penalty",
        type=float,
        default=0.0,
        help="Build 14 decision-time anti-repetition penalty for probed bc/offrl policies "
        "(0 = off, reproduces the Build 13 baseline)",
    )
    ap.add_argument("--per-battle-timeout", type=float, default=120.0)
    ap.add_argument("--save-replays", action="store_true", help="Save poke-env HTML replays")
    ap.add_argument(
        "--opponents",
        default="random",
        help="Comma-separated opponent kinds per probed policy (Build-8: 'random,heuristic'"
        " runs the pathology vs the real eval opponent too; single value keeps Build-6"
        " arm names)",
    )
    ap.add_argument(
        "--obs-out",
        default=str(_OBS_OUT),
        help="Build-8: gitignored npz of the per-decision obs/mask the model scored "
        "(Phase B drift localization consumes it)",
    )
    ap.add_argument("--out", default=str(_RESULTS))
    ap.add_argument("--decisions-out", default=str(_DECISIONS_OUT))
    ap.add_argument("--transcript-dir", default=str(_TRANSCRIPT_DIR))
    args = ap.parse_args()

    opponents = [o for o in (x.strip() for x in args.opponents.split(",")) if o]
    if not opponents:
        raise SystemExit("--opponents needs at least one kind")
    probed: list[tuple[str, str | None]] = []
    if args.bc_checkpoint:
        probed.append(("bc", args.bc_checkpoint))
    if args.offrl_checkpoint:
        probed.append(("offrl", args.offrl_checkpoint))
    if not probed:
        raise SystemExit("pass --bc-checkpoint and/or --offrl-checkpoint")
    # Single opponent -> keep the Build-6 bare arm name ("bc"); multiple -> suffix the
    # opponent so each (policy, opponent) cell is a distinct arm.
    arms: list[tuple[str, str, str | None, str]] = [
        (kind if len(opponents) == 1 else f"{kind}_vs_{opp}", kind, ckpt, opp)
        for kind, ckpt in probed
        for opp in opponents
    ]
    if args.control:
        arms.append(("control", "random", None, "random"))

    teams = TeamPool.from_packed_file(args.team_pool).teams
    probe = _BehaviorProbe()
    with _probe_agents(probe):
        summaries = asyncio.run(_run_arms(arms, args, probe, teams))

    train = _train_baseline(Path(args.train_shard))
    control: dict[str, Any] | None = None
    arm_results: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for summary in summaries:
        arm = str(summary["arm"])
        battles = cast(list[dict[str, Any]], summary["battles"])
        if summary["kind"] == "random":
            control = _decision_metrics([], [], battles)
            wr = control["win_rate"]
            checks["control_ok"] = wr is not None and CONTROL_BAND[0] <= wr <= CONTROL_BAND[1]
            continue
        decs = [d for d in probe.decisions if d.arm == arm]
        rejections = [r for r in probe.rejections if r["arm"] == arm]
        tags = {b["battle"] for b in battles}
        metrics = _decision_metrics(decs, rejections, battles)
        order = _order_metrics(
            {t: o for t, o in probe.packed_orders.items() if t in tags},
            {t: o for t, o in probe.live_orders.items() if t in tags},
            probe.unstable_tags,
        )
        classification = _classify(
            metrics, order, train["switch_fraction"] if train else None
        )
        checks[f"sample_ok_{arm}"] = metrics["n_decisions"] >= MIN_DECISIONS
        arm_results[arm] = {
            "checkpoint": summary["checkpoint"],
            "timed_out": summary["timed_out"],
            "metrics": metrics,
            "order": order,
            "classification": classification,
            "battles": battles,
        }
        _write_transcripts(Path(args.transcript_dir), arm, probe, summary, order)

    inconclusive = not all(checks.values())
    verdict = (
        "INCONCLUSIVE"
        if inconclusive
        else "; ".join(
            f"{arm}={r['classification']['primary_cause']}" for arm, r in arm_results.items()
        )
    )

    result = {
        "n": args.n,
        "format": args.format,
        "thresholds": {
            "fallback_major": FALLBACK_MAJOR,
            "fallback_minor": FALLBACK_MINOR,
            "max_rejection_rate": MAX_REJECTION_RATE,
            "min_order_match": MIN_ORDER_MATCH,
            "max_switch_fraction": MAX_SWITCH_FRACTION,
            "max_switch_run": MAX_SWITCH_RUN,
            "max_ping_pong": MAX_PING_PONG,
            "max_top_action_share": MAX_TOP_ACTION_SHARE,
            "min_entropy_bits": MIN_ENTROPY_BITS,
            "broken_win_rate": BROKEN_WIN_RATE,
            "min_decisions": MIN_DECISIONS,
            "control_band": CONTROL_BAND,
        },
        "train_baseline": train,
        "control": control,
        "arms": arm_results,
        "checks": checks,
        "verdict": verdict,
        "note_a": "a high fallback rate alone predicts ~random (~0.5 vs random), not the"
        " observed ~0.05 -- worse-than-random needs a co-reported cause",
        "note_b": "order match clears only the LIVE half of candidate (b); train-side"
        " |poke| order is a separate build",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.decisions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.decisions_out, "w", encoding="utf-8") as fh:
        for d in probe.decisions:
            fh.write(json.dumps(asdict(d)) + "\n")
    _save_obs(Path(args.obs_out), probe)

    print(f"\n=== Build 6 behavioral probe (n={args.n}/arm, {args.format}) ===")
    if control:
        print(f"control random-mirror win rate {control['win_rate']}  (band {CONTROL_BAND})")
    for arm, r in arm_results.items():
        m, o, c = r["metrics"], r["order"], r["classification"]
        print(
            f"{arm:8} win {m['win_rate']} | fallback {m['fallback_rate']:.3f} "
            f"rejections {m['n_invalid']}+{m['n_unavailable']} allFalse {m['all_false_masks']}"
            f" | order match {o['match_rate']:.2f} stable {o['stable_rate']:.2f}"
            f" | vol-switch {m['voluntary_switch_fraction']:.2f} run {m['max_switch_run']}"
            f" pingpong {m['ping_pong_rate']:.2f} | top-share {m['top_action_share']:.2f}"
            f" H {m['entropy_bits']:.2f} bits | -> {c['primary_cause']}"
        )
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    print(f"transcripts -> {args.transcript_dir}/")
    if inconclusive:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

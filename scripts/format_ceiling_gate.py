"""Lever 15 -- Format-ceiling diagnostic: is the gen9-RB wall the *format* or the *model*?

Fourteen levers plateaued vs the ``heuristic`` in gen9 Random Battles; five independent
follow-ups all failed to compound on the GREEN base. But every one of those only shows *our*
model + inference can't do better -- none directly measures the *achievable* ceiling. Gen9-RB is
a random-team format, where team-assignment RNG structurally compresses skill. This gate measures
the ceiling directly and its verdict selects the next lever: an OU format pivot, or a substrate
scale-up on gen9-RB.

Three cheap, no-training measurements (all win-rates vs ``heuristic``, Wilson 95% CIs, n=300):

  M1 -- bot-skill-gradient sweep (server). How far can *play* move the needle?
        heuristic-vs-heuristic (mirror, ~0.50 sanity) | simpleheuristics | maxbasepower | random |
        offrl (GREEN). A thin competent band => format-bound.

        bash scripts/run_server.sh
        python scripts/format_ceiling_gate.py --stage m1 --n 300 --concurrency 20

  M2 -- strongest-inference upper bound (reuse). L14 white-box depth-2 expectimax with a
        near-perfect opponent model already reached 0.500 vs heuristic (n=120). Echoed from
        results/rpredict_oppmodel_gate_b_whitebox_n120.json.

  M3 -- team-RNG variance decomposition (no battles; node re-sim only). Recover both full teams +
        the winner for a sample of replays, score each team by summed base-stat z-scores, and
        compute the AUC of (team-strength difference -> winner). High AUC => RNG predicts outcomes.

        python scripts/format_ceiling_gate.py --stage m3 --limit 300

  decide -- merge M1+M2+M3 and write the verdict.

        python scripts/format_ceiling_gate.py --stage decide

Each stage merges into results/format_ceiling_gate.json; ``--stage all`` runs m3, m1, then decide.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_RESULTS = Path("results/format_ceiling_gate.json")
_M2_WHITEBOX = Path("results/rpredict_oppmodel_gate_b_whitebox_n120.json")
_GREEN_CKPT = "checkpoints/offrl_scale_et_prior_s0.pt"
_REPLAY_GLOB = "replays/gen9randombattle/*.json"

# Decision thresholds (plan.md Lever 15).
BAND_TOP = 0.53  # a competent bot at/below this vs heuristic => thin skill band
HEADROOM = 0.58  # any competent agent at/above this => real, uncaptured headroom
AUC_HI = 0.65  # M3 corroboration: team strength predicts the winner this well => RNG-bound
MIRROR_TOL = 0.05  # heuristic-vs-heuristic must land within this of 0.50

ArmRecord = dict[str, Any]

# M1 sweep: (label, p1, p2, p1_checkpoint).
_MATCHUPS: list[tuple[str, str, str, str | None]] = [
    ("mirror", "heuristic", "heuristic", None),
    ("simpleheuristics", "simpleheuristics", "heuristic", None),
    ("maxbasepower", "maxbasepower", "heuristic", None),
    ("random", "random", "heuristic", None),
    ("offrl_green", "offrl", "heuristic", _GREEN_CKPT),
]

# Learned p1 arms take the Build-14 ``loop_penalty`` (LoopGuard); baselines do not. `doubles`
# is here because it carries the per-slot `DoublesLoopGuard` (B6f), not the singles one.
_LEARNED_ARMS = {"bc", "offrl", "ppo", "doubles"}


def _build_matchups(
    bc_ckpt: str | None,
    include_offrl_green: bool = True,
    offrl_ckpt: str | None = None,
    learned_agent: str = "offrl",
    bc_agent: str = "bc",
) -> list[tuple[str, str, str, str | None]]:
    """The fixed skill gradient, plus loop-fixed learned arms when checkpoints are given.

    Pure (no server) so the matchup wiring is unit-testable. ``offrl_ckpt`` appends an
    ``offrl_ou`` arm (Build 16: the PPO-self-play OU actor-critic vs ``heuristic``) and
    ``bc_ckpt`` appends the ``bc_v11`` arm (the shipped BC winner vs ``heuristic``) -- both
    the OU-current-encoder learned arms the RB-oriented ``offrl_green`` arm can't load.
    ``include_offrl_green=False`` drops the RB green arm: its default checkpoint is an RB agent
    frozen at an older encoder version, so on OU it can't build -- the ``offrl_ou``/``bc_v11``
    arms are the meaningful learned arms there."""
    matchups = [m for m in _MATCHUPS if include_offrl_green or m[0] != "offrl_green"]
    if offrl_ckpt:
        matchups.append(("offrl_ou", learned_agent, "heuristic", offrl_ckpt))
    if bc_ckpt:
        matchups.append(("bc_v11", bc_agent, "heuristic", bc_ckpt))
    return matchups


def _arm_record(p1: str, p2: str, rate: float, n: int, checkpoint: str | None = None) -> ArmRecord:
    """One matchup row of the M1 block.

    ``checkpoint`` is written ONLY when the arm has one, so a baseline row stays byte-identical to
    every record already committed and only the learned rows grow a key.

    The LABEL is a schema key with OU-era spelling, not a description: ``assess_ou`` reads
    ``bc_v11`` and ``offrl_ou`` by literal name, so they cannot be renamed per format without
    making old and new records incomparable. On the VGC run those two labels describe nothing --
    both arms were built as the ``doubles`` agent -- and the path is what actually says which
    weights were scored. Recording it is also what lets `check_artifacts.py` verify a published
    ceiling number, since that scan looks for `checkpoints/...pt` in the record text.
    """
    k = round(rate * n)
    lo, hi = wilson_ci(k, n)
    record: ArmRecord = {"p1": p1, "p2": p2, "rate": float(rate), "wins": k, "ci95": [lo, hi]}
    if checkpoint:
        record["checkpoint"] = checkpoint
    return record

# On-record gen9-RB M1 band (results/format_ceiling_gate.json, Lever 15) for a side-by-side
# read on the teambuilt (OU) smoke. GREEN there is the RB-trained offrl checkpoint.
_RB_BAND = {
    "mirror": 0.513,
    "simpleheuristics": 0.523,
    "maxbasepower": 0.107,
    "random": 0.007,
    "offrl_green": 0.430,
}


# --------------------------------------------------------------------------- #
# Pure statistics (unit-tested; no server / node).
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for ``k`` successes in ``n`` Bernoulli trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of a real-valued score vs a binary label, with ties counted as 0.5.

    Mann-Whitney U form: (sum of average-ranks over positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[labels].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile 95% CI for ``roc_auc`` by resampling battles with replacement.

    ``roc_auc`` returns NaN for a single-class sample, and on a small or lopsided set EVERY
    resample can come back single-class -- which left nothing to take a percentile of and raised
    `IndexError` out of numpy rather than reporting an undefined interval. A sample with one class
    has no AUC and no interval; say so.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n == 0 or math.isnan(roc_auc(scores, labels)):
        return (float("nan"), float("nan"))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = roc_auc(scores[idx], labels[idx])
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def compute_verdict(m1: dict[str, Any], m2: dict[str, Any], m3: dict[str, Any]) -> dict[str, Any]:
    """Decide FORMAT vs MODEL bound. M1 (skill band) + M2 (inference ceiling) are primary;
    M3 (team-strength AUC) corroborates the *mechanism* but does not gate the branch -- a
    hand-crafted strength proxy can't cleanly prove play-dominance, so it must not force one.

    THE LEARNED ARM IS LOOKED UP BY ALIAS, not by one literal key. `run_m1` labels it `offrl_green`
    on Random Battles and `offrl_ou` on a teambuilt format, and drops the green arm entirely off RB
    (`include_green`). Those names are the ON-DISK schema of every record ever written, so renaming
    them would make old and new gates incomparable; an alias list reads all of them and leaves the
    files alone. A format with no learned arm at all contributes nothing to `competent` rather than
    raising KeyError -- M1's baseline band and M2 still decide the branch.

    THE BRANCH LABEL AND THE REASON STRING WERE WRITTEN FOR gen9-RB and asserted things that are
    only true there. `next_branch` said "ou_pivot" -- advice that is meaningless on a format
    reached *after* the OU pivot -- and the FORMAT_BOUND reason interpolated M2 as "at parity",
    which was true of RB's 0.500 and is simply false of a w that is nowhere near 0.50. Both are now
    derived from the record instead of assumed, because a verdict line that states a wrong number
    is worse than one that states none.
    """
    s = m1["simpleheuristics"]["rate"]
    learned = [m1[k]["rate"] for k in ("offrl_green", "offrl_ou", "bc_v11") if k in m1]
    g = max(learned, default=0.0)
    w = m2["search_vs_heuristic"]
    a = m3["auc"]
    mirror = m1["mirror"]["rate"]
    competent = max(s, w, g)  # the best agent/inference we can field vs the heuristic

    sanity_ok = abs(mirror - 0.5) <= MIRROR_TOL
    # An ABSENT format key means gen9-RB, and the default matters. `run_m1` only started writing
    # `format` when teambuilt support landed, so `results/format_ceiling_gate.json` -- the record
    # carrying the original FORMAT_BOUND verdict -- has none. A generic fallback would re-derive
    # that historical decision with a different `next_branch` the next time anyone re-runs it.
    fmt = m1.get("format") or m2.get("format") or "gen9randombattle"
    # "at parity" is only honest within a hair of 0.50; RB's M2 was 0.500 and the template said so
    # unconditionally. Anything else gets described, not labelled.
    w_desc = (
        f"white-box near-optimal inference {w:.3f} at parity"
        if abs(w - 0.5) <= 0.02
        else f"white-box near-optimal inference reaches only {w:.3f}"
    )
    if competent >= HEADROOM:
        verdict, branch = "MODEL_BOUND", "scale_up"
        reason = (
            f"best competent agent reaches {competent:.3f} >= {HEADROOM} vs heuristic on {fmt}: "
            "real, uncaptured headroom -> scale the model before pivoting"
        )
    elif competent <= BAND_TOP:
        # The RB run's branch was "pivot to OU". A teambuilt format has already done that, so the
        # actionable branch there is "stop spending on strength here", not "pivot" again.
        verdict = "FORMAT_BOUND"
        branch = "ou_pivot" if "randombattle" in fmt else "stop_strength_axis"
        reason = (
            f"nothing beats the heuristic: best competent agent {competent:.3f} <= {BAND_TOP} "
            f"({w_desc}) -> the ceiling is the format"
        )
    else:
        verdict, branch = "AMBIGUOUS", "scale_up_probe"
        reason = (
            f"best competent agent {competent:.3f} in ({BAND_TOP}, {HEADROOM}): neither clearly "
            "capped nor clear headroom -> cheap scale-up capacity probe before the OU build"
        )

    rng_corroborates = a >= AUC_HI
    m3_note = (
        f"team-strength AUC {a:.3f} >= {AUC_HI}: team assignment predicts the winner "
        "-> RNG-boundedness corroborated"
        if rng_corroborates
        else f"team-strength AUC {a:.3f} < {AUC_HI}: gross team strength does NOT predict the "
        "winner (RB balances via level) -> the format is balanced; wins come from matchup/"
        "variance + fine play a competent heuristic already captures, not from a stronger team"
    )

    return {
        "verdict": verdict,
        "next_branch": branch,
        "reason": reason,
        "mirror_sanity_ok": sanity_ok,
        "m3_corroboration": {"rng_bound_corroborated": rng_corroborates, "note": m3_note},
        "format": fmt,
        "signals": {"simpleheuristics_s": s, "offrl_green_g": g, "whitebox_w": w, "auc_a": a},
        # An M2 from `ShapedOnlyPolicy` is a weaker instrument than one from a trained value head,
        # and `expectimax.ShapedOnlyPolicy` pre-registers the consequence: a WIN is decisive, a
        # NULL suggestive only. Carried onto the verdict so the caveat travels with the branch.
        "m2_leaf": m2.get("search_leaf"),
        "thresholds": {"band_top": BAND_TOP, "headroom": HEADROOM, "auc_hi": AUC_HI},
    }


# --------------------------------------------------------------------------- #
# M3 -- team-RNG variance decomposition (node re-sim; no live server).
# --------------------------------------------------------------------------- #
def _species_strength_table() -> np.ndarray:
    """Per-species scalar strength = sum of the 6 z-scored base-stat columns (row 0 = UNK = 0)."""
    from lategame.features.embed_prior import _PRIORS_PATH

    species_feat = np.load(_PRIORS_PATH)["species"]  # (V+1, 6 stats + |types|)
    return species_feat[:, :6].sum(axis=1)


def _winner_side(log: str, p1_name: str, p2_name: str) -> str | None:
    """Return 'p1'/'p2' from the replay log's ``|win|`` line, or None (tie/unknown)."""
    from poke_env import to_id_str

    win_name = None
    for line in log.split("\n"):
        if line.startswith("|win|"):
            win_name = line.split("|")[2]
            break
    if win_name is None:
        return None
    wid = to_id_str(win_name)
    if wid == to_id_str(p1_name):
        return "p1"
    if wid == to_id_str(p2_name):
        return "p2"
    return None


def _preview_teams(log: str) -> tuple[list[str], list[str]]:
    """Both full rosters from the log's ``|poke|`` lines, as poke-env species ids.

    ``|poke|p1|Dragonite, L50, F|`` -- the species is everything before the first comma. On a
    teambuilt format with a preview screen this is the WHOLE team, declared before turn 1, which is
    why the doubles path needs no inputlog and no re-sim.
    """
    from poke_env import to_id_str

    sides: dict[str, list[str]] = {"p1": [], "p2": []}
    for line in log.split("\n"):
        if not line.startswith("|poke|"):
            continue
        parts = line.split("|")
        if len(parts) < 4 or parts[2] not in sides:
            continue
        sides[parts[2]].append(to_id_str(parts[3].split(",")[0]))
    return sides["p1"], sides["p2"]


def _log_player_names(log: str) -> tuple[str | None, str | None]:
    """p1/p2 usernames from the log's ``|player|`` lines (the inputlog's role, without one)."""
    names: dict[str, str] = {}
    for line in log.split("\n"):
        if not line.startswith("|player|"):
            continue
        parts = line.split("|")
        if len(parts) >= 4 and parts[2] in ("p1", "p2") and parts[3]:
            names.setdefault(parts[2], parts[3])
    return names.get("p1"), names.get("p2")


def run_m3_preview(files: list[str], team_size: int = 6) -> dict[str, Any]:
    """M3 off the TEAM PREVIEW lines alone -- no ``inputlog``, no node, no re-sim.

    Public Showdown replays do not carry an inputlog (measured: the VGC 2025 Reg I index returns
    none with one), so the RB path cannot run on them at all. It does not need to. RB has no preview
    screen and balances via level, which is why recovering real level-adjusted stats there was worth
    a full re-sim. A teambuilt doubles format declares both complete rosters in ``|poke|`` lines
    before turn 1 and every mon is L50, so the level-blind base-stat z-sum -- the RB path's
    *robustness* proxy -- is the primary and only defensible measure here.

    Two limits travel with this number and belong in the record, not in a reader's head:

      * The brought SIX is scored, not the played FOUR. The bring-4 decision is exactly the skill
        this measurement cannot see, so a real preview effect biases the AUC toward 0.5.
      * The available VGC replays are UNRATED -- the search index reports no rating for this format
        at all -- where RB's M3 used a rated >= 1200 sample. Skill level is unknown, not filtered.
    """
    bst_z = _species_strength_table()
    vocab_index = _species_index_fn()

    diffs: list[float] = []
    p1_wins: list[int] = []
    oov = 0
    scanned = 0
    for f in files:
        d = json.load(open(f))
        log = d.get("log")
        if not log:
            continue
        scanned += 1
        p1_name, p2_name = _log_player_names(log)
        if not p1_name or not p2_name:
            continue
        winner = _winner_side(log, p1_name, p2_name)
        if winner is None:  # tie, forfeit-without-win, or a name we cannot match
            continue
        p1_team, p2_team = _preview_teams(log)
        if len(p1_team) != team_size or len(p2_team) != team_size:
            continue  # a preview screen that did not show all six is not a comparable sample
        totals = []
        for team in (p1_team, p2_team):
            total = 0.0
            for species in team:
                idx = vocab_index(species)
                if idx == 0:
                    oov += 1
                total += float(bst_z[idx])
            totals.append(total)
        diffs.append(totals[0] - totals[1])
        p1_wins.append(1 if winner == "p1" else 0)

    labels = np.array(p1_wins, dtype=np.int64)
    arr = np.array(diffs, dtype=np.float64)
    used = len(p1_wins)
    auc = roc_auc(arr, labels) if used else float("nan")
    lo, hi = auc_bootstrap_ci(arr, labels) if used else (float("nan"), float("nan"))
    return {
        "n_replays_scanned": scanned,
        "n_battles_used": used,
        "oov_species": oov,
        "p1_win_frac": float(labels.mean()) if used else float("nan"),
        "auc": float(auc),
        "auc_ci95": [lo, hi],
        "auc_bst_z": float(auc),  # same quantity; kept so the schema matches the RB record
        "strength_proxy": "raw base-stat z-sum over the BROUGHT SIX, from |poke| preview lines",
        "mode": "preview",
        "caveats": [
            "scores the brought six, not the played four -- blind to the bring-4 decision",
            "UNRATED sample: the replay index reports no rating for this format",
        ],
    }


def run_m3(limit: int, replay_glob: str | None = None) -> dict[str, Any]:
    """Recover both teams + the winner per replay; AUC of (team-strength diff -> p1 wins).

    Dispatches on the DATA rather than on the format string: replays carrying an ``inputlog`` get
    the re-sim path (real level-adjusted stats), and replays carrying only a log get the preview
    path. Mixing the two proxies inside one AUC would compare incomparable numbers, so the whole
    batch takes one mode and the record says which.
    """
    files = sorted(glob.glob(replay_glob or _REPLAY_GLOB))[:limit]
    if not files:
        raise SystemExit(
            f"no replays under '{replay_glob or _REPLAY_GLOB}'. Fetch them first:\n"
            f"  python -c \"from lategame.data.replays import fetch_replays; \"\n"
            f"  \"fetch_replays(battle_format=FMT, cache_dir=..., require_rating=False)\""
        )
    with_inputlog = sum(1 for f in files if json.load(open(f)).get("inputlog"))
    if with_inputlog * 2 < len(files):
        print(
            f"[M3] {with_inputlog}/{len(files)} replays carry an inputlog -- using the TEAM "
            f"PREVIEW path (species-level strength, no re-sim)."
        )
        return run_m3_preview(files)
    return _run_m3_resim(limit, replay_glob)


def _run_m3_resim(limit: int, replay_glob: str | None = None) -> dict[str, Any]:
    """The original RB path: node re-sim of the inputlog to recover level-adjusted real stats."""
    from lategame.data.resim import _parse_inputlog_meta, _reconstruct_pov_resim, run_driver
    from lategame.data.reward import RewardWeights

    bst_z = _species_strength_table()  # raw base-stat z-sum (ignores RB level-balancing)
    vocab_index = _species_index_fn()

    files = sorted(glob.glob(replay_glob or _REPLAY_GLOB))[:limit]
    raw = []
    for f in files:
        d = json.load(open(f))
        if d.get("inputlog") and d.get("log"):
            raw.append(d)

    driver_out = {
        str(r.get("id")): r
        for r in run_driver([{"id": d["id"], "inputlog": d["inputlog"]} for d in raw])
    }
    weights = RewardWeights()

    # Two strength proxies per team: effective (level-adjusted real stats -- the defensible RB
    # measure, since the format balances via level) and raw base-stat z-sum (level-blind).
    eff_diffs: list[float] = []
    bstz_diffs: list[float] = []
    p1_wins: list[int] = []
    oov = 0
    used = 0
    for d in raw:
        out = driver_out.get(d["id"])
        if out is None or "error" in out:
            continue
        p1_name, p2_name, gen = _parse_inputlog_meta(d["inputlog"])
        winner = _winner_side(d["log"], p1_name, p2_name)
        if winner is None:
            continue
        eff: dict[str, float] = {}
        bstz: dict[str, float] = {}
        ok = True
        for side, uname in (("p1", p1_name), ("p2", p2_name)):
            raw_events = out.get(side)
            events = raw_events if isinstance(raw_events, list) else []
            battle, _, _, _ = _reconstruct_pov_resim(
                events, uname, f"{d['id']}-{side}", gen, weights
            )
            team = list(battle.team.values())
            if len(team) != 6 or any(m.stats is None for m in team):
                ok = False
                break
            eff[side] = float(sum(v for m in team for v in m.stats.values() if v is not None))
            total = 0.0
            for mon in team:
                idx = vocab_index(mon.species)
                if idx == 0:
                    oov += 1
                total += float(bst_z[idx])
            bstz[side] = total
        if not ok:
            continue
        eff_diffs.append(eff["p1"] - eff["p2"])
        bstz_diffs.append(bstz["p1"] - bstz["p2"])
        p1_wins.append(1 if winner == "p1" else 0)
        used += 1

    labels = np.array(p1_wins, dtype=np.int64)
    eff_arr = np.array(eff_diffs, dtype=np.float64)
    bstz_arr = np.array(bstz_diffs, dtype=np.float64)
    auc = roc_auc(eff_arr, labels)
    lo, hi = auc_bootstrap_ci(eff_arr, labels)
    return {
        "n_replays_scanned": len(raw),
        "n_battles_used": used,
        "oov_species": oov,
        "p1_win_frac": float(labels.mean()) if used else float("nan"),
        "auc": float(auc),  # primary: effective (level-adjusted) team strength
        "auc_ci95": [lo, hi],
        "auc_bst_z": float(roc_auc(bstz_arr, labels)),  # robustness: level-blind base-stat sum
        "strength_proxy": "sum of level-adjusted real stats (mon.stats), per team",
        "mode": "resim",
    }


def _species_index_fn():  # noqa: ANN202 -- returns a closure over the loaded vocab
    from lategame.features.vocab import load_vocab

    table = load_vocab().tables["species"]
    return lambda species: table.get(species, 0)


# --------------------------------------------------------------------------- #
# M1 -- bot-skill-gradient sweep (server).
# --------------------------------------------------------------------------- #
async def run_m1(
    n: int,
    concurrency: int,
    battle_format: str,
    team_pool: str,
    offrl_ckpt: str | None = None,
    bc_ckpt: str | None = None,
    loop_penalty: float = 0.0,
) -> dict[str, Any]:
    """Head-to-heads vs ``heuristic`` for the fixed skill gradient + GREEN, at n each.

    Random Battles leave teams to the server. For teambuilt formats (e.g. gen9ou) each player
    draws from the R-TEAM pool via its own ``TeamPool`` seeded distinctly, so the two sides get
    independent (non-locked) team draws -- varied matchups, mirror-fair in expectation.

    ``offrl_ckpt`` overrides the ``offrl`` arm's checkpoint (default: the RB GREEN ckpt), so the
    OU probe can score an OU-trained agent while the RB band stays the on-record RB GREEN.
    ``bc_ckpt`` appends a loop-fixed ``bc_v11`` arm (the shipped OU winner); ``loop_penalty``
    (Build 14) is applied to every learned arm so the agents are scored *with* the LoopGuard the
    live probe uses."""
    from lategame.eval.arena import build_player, evaluate_built, policy_agent

    teams: list[str] | None = None
    if "randombattle" not in battle_format:
        from lategame.teambuilding.pool import TeamPool

        teams = TeamPool.from_packed_file(team_pool).teams

    seeds = itertools.count()

    def _pool() -> object | None:  # fresh pool per player so p1/p2 draw independently
        if teams is None:
            return None
        from lategame.teambuilding.pool import TeamPool

        return TeamPool(teams, seed=next(seeds))

    # The offrl_green arm's default checkpoint is an RB agent pinned to an older encoder version,
    # so it's RB-only; on OU an ``offrl_ckpt`` is scored as the explicit ``offrl_ou`` arm instead.
    include_green = "randombattle" in battle_format
    block: dict[str, Any] = {"n": n, "format": battle_format}
    # B6f: the learned arms must be built with the agent the FORMAT wants. `bc`/`offrl` are in
    # `arena._SINGLES_ONLY_AGENTS` and are refused outright on doubles, so hardcoding them made
    # the stop rule's own learned arm unrunnable on the format it was written for.
    learned = policy_agent(battle_format)
    for label, p1, p2, ckpt in _build_matchups(
        bc_ckpt,
        include_offrl_green=include_green,
        offrl_ckpt=offrl_ckpt,
        learned_agent=learned,
        bc_agent="bc" if learned == "offrl" else learned,
    ):
        player1 = build_player(
            p1, battle_format, checkpoint_path=ckpt,
            max_concurrent_battles=concurrency, team=_pool(),  # type: ignore[arg-type]
            loop_penalty=loop_penalty if p1 in _LEARNED_ARMS else 0.0,
        )
        player2 = build_player(
            p2, battle_format, max_concurrent_battles=concurrency, team=_pool(),  # type: ignore[arg-type]
        )
        rate = await evaluate_built(player1, player2, n)
        block[label] = _arm_record(p1, p2, rate, n, ckpt)
        lo, hi = block[label]["ci95"]
        print(f"  M1 {label:>16}: {p1} vs {p2}  {rate:.3f}  ci95 [{lo:.3f}, {hi:.3f}]  (n={n})")
    return block


# --------------------------------------------------------------------------- #
# Teambuilt (OU) smoke -- harness/sanity read, NOT a FORMAT/MODEL verdict.
# --------------------------------------------------------------------------- #
def assess_ou(m1: dict[str, Any]) -> dict[str, Any]:
    """Score the teambuilt M1 smoke: is the harness clean and the band at least as wide as RB?

    Deliberately does not apply the RB BAND_TOP/HEADROOM thresholds -- poke-env's bots sit far
    below OU's skill ceiling and there is no OU near-optimal reference.

    STILL M1-ONLY, and still not a verdict, even now that a teambuilt run can reach
    `compute_verdict` -- this stays the harness read it always was. The OTHER two legs are no
    longer unavailable-in-principle for every teambuilt format: gen9vgc2025regi has both. Whether a
    given run has them is a property of that run, so the note is built from the record rather than
    asserted."""
    rate = {k: m1[k]["rate"] for k in _RB_BAND if k in m1 and "rate" in m1[k]}
    mirror_ok = abs(rate.get("mirror", 0.5) - 0.5) <= MIRROR_TOL
    gradient_ok = (
        rate.get("random", 1.0) < rate.get("maxbasepower", 0.0) < rate.get("simpleheuristics", 0.0)
    )
    harness_ok = mirror_ok and gradient_ok
    ou_width = rate.get("simpleheuristics", 0.0) - rate.get("random", 0.0)
    rb_width = _RB_BAND["simpleheuristics"] - _RB_BAND["random"]

    # FORMAT vs MODEL: apply the Lever-15 HEADROOM threshold to the competent-bot reference
    # (simpleheuristics), consistent with the RB verdict. On OU a *simple* bot already clears
    # the heuristic by a wide margin, so if it lands >= HEADROOM the format is not the ceiling
    # and our learned agent's low win is a MODEL gap. Only trust the verdict on a clean harness.
    competent = rate.get("simpleheuristics")
    learned_bc = m1.get("bc_v11")  # loop-fixed OU BC winner, if the bc arm ran
    learned_offrl = m1.get("offrl_ou")  # Build-16 PPO-self-play OU actor-critic, if it ran
    format_bound_rejected = competent is not None and competent >= HEADROOM
    if not harness_ok:
        verdict = "INSUFFICIENT"
        reason = "harness not clean (mirror/gradient) -- fix before trusting the OU band"
    elif format_bound_rejected:
        verdict = "MODEL_BOUND"
        reason = (
            f"competent bot (simpleheuristics {competent:.3f}) >= HEADROOM {HEADROOM} over the "
            "heuristic => OU rewards skill, format is not the ceiling; the learned agent's low "
            "win is a model gap, not a format cap. FORMAT_BOUND rejected for OU."
        )
    else:
        verdict = "INSUFFICIENT"
        reason = (
            f"competent bot (simpleheuristics {competent}) < HEADROOM {HEADROOM} -- no wide-band "
            "evidence of skill headroom; investigate harness or reconsider a genuine OU ceiling."
        )
    ou_verdict: dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "competent_rate": competent,
        "headroom": HEADROOM,
        "format_bound_rejected": bool(format_bound_rejected and harness_ok),
    }
    if learned_bc is not None:
        ou_verdict["learned_bc"] = {"rate": learned_bc["rate"], "ci95": learned_bc.get("ci95")}
    if learned_offrl is not None:
        ou_verdict["learned_offrl"] = {
            "rate": learned_offrl["rate"], "ci95": learned_offrl.get("ci95")
        }
    # model_gap uses the strongest learned arm present -- the PPO (offrl_ou) arm takes precedence
    # over bc_v11 once Build 16 has run, else it falls back to the bc winner.
    primary = learned_offrl or learned_bc
    if primary is not None and competent is not None:
        ou_verdict["model_gap"] = competent - primary["rate"]

    return {
        "note": (
            "M1 teambuilt harness read, not a verdict. The three-leg FORMAT/MODEL rule is "
            "`decision` (compute_verdict), present only when M2 and M3 are also in this record."
        ),
        "mirror_sanity_ok": mirror_ok,
        "gradient_ok": gradient_ok,
        "harness_ok": harness_ok,
        "ou_band": rate,
        "rb_band_reference": _RB_BAND,
        "band_width": {"ou": ou_width, "rb": rb_width, "wider_than_rb": ou_width >= rb_width},
        "ou_verdict": ou_verdict,
    }


def _print_ou_summary(a: dict[str, Any]) -> None:
    print("\n[OU smoke] band vs on-record gen9-RB band:")
    for k in _RB_BAND:
        ou = a["ou_band"].get(k)
        ou_s = f"{ou:.3f}" if ou is not None else "  -  "
        print(f"  {k:>16}: OU {ou_s}   RB {_RB_BAND[k]:.3f}")
    bw = a["band_width"]
    print(
        f"  mirror sanity: {'OK' if a['mirror_sanity_ok'] else 'FAIL'}   "
        f"gradient: {'OK' if a['gradient_ok'] else 'FAIL'}   "
        f"band width OU {bw['ou']:.3f} vs RB {bw['rb']:.3f} "
        f"({'wider' if bw['wider_than_rb'] else 'not wider'})"
    )
    print(f"  {a['note']}")
    if not a["harness_ok"]:
        print("  !! harness NOT clean -- investigate before trusting the OU band")
    v = a.get("ou_verdict")
    if v:
        gap = v.get("model_gap")
        gap_s = f"   model_gap {gap:.3f}" if gap is not None else ""
        bc = v.get("learned_bc")
        if bc is not None:
            print(f"  loop-fixed bc_v11 (vs heuristic): {bc['rate']:.3f}")
        offrl = v.get("learned_offrl")
        if offrl is not None:
            print(f"  PPO offrl_ou (vs heuristic): {offrl['rate']:.3f}{gap_s}")
        elif bc is not None:
            print(f"  model_gap (bc_v11):{gap_s}")
        print(f"  OU VERDICT: {v['verdict']} -- {v['reason']}")


# --------------------------------------------------------------------------- #
# M2 -- reuse the L14 white-box upper bound.
# --------------------------------------------------------------------------- #
def load_m2(path: Path | None = None) -> dict[str, Any]:
    """Echo a `rpredict_oppmodel_gate --gate b --arms whitebox` record as this gate's M2 leg.

    The path used to be the hardcoded RB `_M2_WHITEBOX`, which is what kept M2 an RB-only concept:
    the producer has taken `--format` for a while, but nothing here could read a record it wrote for
    any other format.
    """
    src = path or _M2_WHITEBOX
    if not src.exists():
        raise SystemExit(
            f"no M2 record at '{src}'. Produce one first:\n"
            f"  LATEGAME_SEARCH_SHAPED_ONLY=1 python scripts/rpredict_oppmodel_gate.py "
            f"--gate b --arms whitebox --depth 2 --format FMT --out {src}"
        )
    data = json.loads(src.read_text())
    wb = data["arms"]["whitebox"]

    # TWO SHAPES, because a pooled record is not a single-run record. `rpredict_oppmodel_gate`
    # writes scalar rates and a top-level `n`; `merge_search_shards` writes {wins, n, rate} objects
    # and no top-level `n`, since the n is per-arm once shards are summed. Reading only the first
    # shape is what made "echo the pooled search run as M2" impossible without hand-editing JSON.
    def _rate(v: Any) -> float:
        return float(v["rate"] if isinstance(v, dict) else v)

    def _n(v: Any) -> Any:
        return v["n"] if isinstance(v, dict) else data.get("n")

    return {
        "source": str(src),
        "n": _n(wb["search_vs_heuristic"]),
        "depth": data["depth"],
        "format": data.get("format"),
        "shards": data.get("shards"),
        "base_vs_heuristic": _rate(data["base_vs_heuristic"]),
        "search_vs_heuristic": _rate(wb["search_vs_heuristic"]),
        "search_vs_base": _rate(wb["search_vs_base"]) if "search_vs_base" in wb else None,
        "contrast_vs_base": wb.get("contrast_vs_base"),
        "verdict": wb.get("verdict"),
        "note": "depth-2 expectimax, near-perfect white-box opponent (ceiling from above)",
    }


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"gate": "format_ceiling"}


def _save(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"  -> wrote {path}")


def _maybe_decide(data: dict[str, Any]) -> None:
    if all(k in data for k in ("m1", "m2", "m3")):
        data["decision"] = compute_verdict(data["m1"], data["m2"], data["m3"])
        d = data["decision"]
        print(f"\nVERDICT: {d['verdict']} -> {d['next_branch']}\n  {d['reason']}")
        if not d["mirror_sanity_ok"]:
            print("  !! mirror sanity FAILED (heuristic-vs-heuristic not ~0.50) -- harness suspect")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["m1", "m3", "decide", "all"], default="all")
    ap.add_argument("--format", dest="battle_format", default="gen9randombattle")
    ap.add_argument("--team-pool", default="lategame/teambuilding/data/teams_gen9ou.packed",
                    help="packed team pool for teambuilt formats (R-TEAM)")
    ap.add_argument("--out", default=None, help="results JSON (default derived from --format)")
    ap.add_argument("--n", type=int, default=300, help="battles per M1 matchup")
    ap.add_argument("--concurrency", type=int, default=20, help="M1 max concurrent battles")
    ap.add_argument("--offrl-checkpoint", default=None,
                    help="OU offrl/PPO checkpoint; adds an offrl_ou learned arm to the M1 sweep")
    ap.add_argument("--bc-checkpoint", default=None,
                    help="OU BC checkpoint; adds a loop-fixed bc_v11 arm to the teambuilt M1 sweep")
    ap.add_argument("--loop-penalty", type=float, default=0.0,
                    help="Build-14 LoopGuard penalty applied to learned M1 arms (0 = off)")
    ap.add_argument("--limit", type=int, default=300, help="M3 replays to score")
    ap.add_argument("--replay-glob", default=None,
                    help="M3 replay cache (default replays/<format>/*.json)")
    ap.add_argument("--m2-record", default=None,
                    help="rpredict_oppmodel_gate --gate b record to echo as the M2 leg")
    args = ap.parse_args()

    fmt = args.battle_format
    teambuilt = "randombattle" not in fmt
    out = (
        Path(args.out) if args.out
        else (Path("results/format_ceiling_gate_ou.json") if teambuilt else _RESULTS)
    )
    replay_glob = args.replay_glob or f"replays/{fmt}/*.json"
    m2_record = Path(args.m2_record) if args.m2_record else None

    data = _load(out)
    data["format"] = fmt

    if teambuilt:
        # THE TEAMBUILT PATH USED TO RETURN HERE, before M2 or M3 could run, so no teambuilt format
        # could ever reach the three-leg verdict `compute_verdict` implements -- only `assess_ou`,
        # which reads M1 alone and cannot return FORMAT_BOUND at all. That was honest while the
        # other two legs did not exist for any teambuilt format. They exist now: the M2 producer
        # takes --format and runs on doubles under LATEGAME_SEARCH_SHAPED_ONLY=1, and M3 reads team
        # preview lines rather than an inputlog. So the M1-only assessment still runs and is still
        # recorded, and the run falls through to whichever other legs are actually available.
        print(f"[M1] bot-skill-gradient sweep on {fmt} (teambuilt)...")
        if args.stage in ("m1", "all"):
            data["m1"] = asyncio.run(
                run_m1(
                    args.n, args.concurrency, fmt, args.team_pool, args.offrl_checkpoint,
                    bc_ckpt=args.bc_checkpoint, loop_penalty=args.loop_penalty,
                )
            )
        if "m1" in data:
            data["ou_assessment"] = assess_ou(data["m1"])
            _print_ou_summary(data["ou_assessment"])

    if teambuilt:
        # Opt-in, because these two legs exist for doubles and not for OU: ask for them by pointing
        # at the record / the replay cache, rather than failing a run that only wanted M1.
        if m2_record is not None:
            data["m2"] = load_m2(m2_record)
            print(f"[M2] white-box search vs heuristic {data['m2']['search_vs_heuristic']:.3f} "
                  f"(n={data['m2']['n']}, from {data['m2']['source']})")
        if args.stage in ("m3", "all") and glob.glob(replay_glob):
            print(f"[M3] team-strength AUC from {replay_glob} ...")
            data["m3"] = run_m3(args.limit, replay_glob)
            print(f"  M3 AUC {data['m3']['auc']:.3f} ci95 {data['m3']['auc_ci95']} "
                  f"(n={data['m3']['n_battles_used']}, mode={data['m3']['mode']})")
        _maybe_decide(data)
        _save(data, out)
        return

    data.setdefault("m2", load_m2(m2_record))  # always cheap; keep it fresh

    if args.stage in ("m3", "all"):
        print("[M3] team-RNG variance decomposition (node re-sim, no server)...")
        data["m3"] = run_m3(args.limit, args.replay_glob)
        print(f"  M3 AUC {data['m3']['auc']:.3f} ci95 {data['m3']['auc_ci95']} "
              f"(n={data['m3']['n_battles_used']})")
        _save(data, out)
    if args.stage in ("m1", "all"):
        print("[M1] bot-skill-gradient sweep (server)...")
        data["m1"] = asyncio.run(
            run_m1(args.n, args.concurrency, fmt, args.team_pool, args.offrl_checkpoint)
        )
        _save(data, out)

    _maybe_decide(data)
    _save(data, out)


if __name__ == "__main__":
    main()

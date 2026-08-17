"""OU Pivot Build 2 -- Gate A: is log-based OU reconstruction faithful enough to train on?

Gen9-RB replays carry a PRNG seed, so ``data.resim`` recovers the full private POV by
re-simulating. Gen9-OU public replays carry **no ``inputlog``** (confirmed: keys are
format/id/log/players/rating/...), so there is no seed and no packed team -- resim is
impossible. But OU logs open with ``|poke|`` team-preview lines naming all six species
per side, so the *v1 log-based* ``data.ingest`` path is the right reconstructor once it
seeds those species from turn 0 (``ingest._register_preview``). This gate measures
whether that reconstruction is faithful before we spend anything on training.

This upgrade (Build 3) checks the channels the encoder actually feeds the model -- the
identity-embedding IDs (item/ability/move) -- not just species presence. Build 2's Gate A
passed on species but was blind to the own-team *detail* gap that made the trained agent OOD:
the log reveals the player's own item/ability/moves only progressively, while the live
``|request|`` hands over the full team from turn 1. ``ingest`` now closes this with two-pass
own-team completion (``_prescan_kits`` + ``_complete_own_team``); this gate measures how far
that goes on the exact channels that matter.

Build 4 adds the fourth arm: **usage-prior imputation** (``ingest._impute_kits``). Build 3
proved the log-only ceiling (item 0.30 / ability 0.47 / moves 2.78) leaves the channels OOD
vs the live-full POV and that a partial fix is worthless -- so the imputation arm must reach
the *live band* (measured by the Build-3 live-encoder probe: item 0.894 / ability 1.000 /
moves 4.000; item < 1.0 because a consumed or absent item is None live too). The gate also
reports (advisory, not a kill) how often the prior *agrees with revealed truth* on the
truth-revealed subset -- a reveal-biased but useful estimate of imputation noise.

The item check is an **unfilled-rate** check, not an absolute known-rate: the first Build-4
run measured item-known 0.760 and the per-decision decomposition proved the residual is
0.254 consumed-``None`` (Knock Off / Booster Energy / berries -- states the live POV also
shows as ``None``; human ladder games carry ~2.4x more item removal than eval-vs-heuristic
battles) with only 0.009 actually-unfilled sentinel. An absolute >= 0.85 bar conflated the
two; the failure mode the gate must catch is unfilled slots (a broken species lookup would
push the sentinel rate back to the ~0.70 Build-3 residual), so that is what it gates on,
while the known/consumed split and the live reference stay reported for the record.

Measurements over a sample of cached gen9ou replays (no server, no training):

  * parse rate      -- POV-episodes reconstructed without a fatal error.
  * species coverage -- mean over decisions of (ego + opp species present)/12, from the
                        per-Pokemon "present" flag. Preview should push this to ~1.0.
  * drop rate       -- decisions seen but undecodable (forced/first-reveal-nickname switches).
  * reward sign     -- winner-POV mean episode return must exceed loser-POV mean.
  * species control (teeth) -- re-run with ``|poke|`` stripped; coverage must drop materially.
  * own-active detail channels -- over decisions, the own active mon's item-known rate,
                        ability-known rate, and move-count (/4), read from the encoder's ID
                        channels via ``OBS_LAYOUT``. Reported WITH two-pass and WITH it OFF
                        (the v1 progressive-reveal control); the ON-vs-OFF *lift* is the teeth,
                        and the absolute ON value is the log *ceiling* (residual = ideal - ON,
                        where ideal = live 1.0/1.0/4.0).

KILL (do NOT train): parse rate < 0.80, species coverage < 0.95, drop rate > 0.40, reward gap
  <= 0, species control lift < 0.10, two-pass fails to lift the channels logs *can* fix
  (item-known lift < 0.05 or move-count lift < 0.30 vs the no-completion control), OR the
  imputation arm fails to reach the live band (item unfilled-sentinel rate > 0.02,
  ability < 0.99, moves < 3.95, usage-missing species rate > 0.02).

    python scripts/ou_ingest_gate.py --cache-dir replays/gen9ou --sample 200
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from rotomai.data import ingest
from rotomai.data.ingest import (
    _UNKNOWN_ITEM,
    _episode_rewards,
    _gen_from_log,
    _impute_kits,
    _prescan_kits,
    _reconstruct_pov,
)
from rotomai.data.replays import cached_replay_paths, fetch_replays
from rotomai.data.reward import RewardWeights
from rotomai.data.usage_prior import UsagePrior, load_usage_prior
from rotomai.features.encoder import OBS_LAYOUT

_RESULTS = Path("results/ou_ingest_gate.json")
_PDIM = OBS_LAYOUT.pokemon_dim  # each of the first 12 obs blocks is one Pokemon
# Trailing per-Pokemon ID channels (POKEMON_ID_FIELDS = species, item, ability).
_ITEM_CH = _PDIM - 2
_ABILITY_CH = _PDIM - 1

# KILL thresholds.
MIN_PARSE_RATE = 0.80
MIN_COVERAGE = 0.95
MAX_DROP_RATE = 0.40
MIN_COVERAGE_LIFT = 0.10  # preview must lift coverage vs the stripped-|poke| control
MIN_ITEM_LIFT = 0.05  # two-pass must lift own-active item-known vs the no-completion control
MIN_MOVE_LIFT = 0.30  # two-pass must lift own-active move-count vs the no-completion control
IDEAL_ITEM = 1.0  # live request reveals every own item...
IDEAL_ABILITY = 1.0  # ...ability...
IDEAL_MOVES = 4.0  # ...and full moveset from turn 1 (residual = ideal - ceiling)
# Live-encoder probe reference (Build 3 investigation, 2,918 live decisions): the band the
# imputation arm must reach. Item < 1.0 live because consumed/absent items are None there too.
LIVE_REFERENCE = {"item": 0.894, "ability": 1.0, "moves": 4.0}
MAX_ITEM_UNFILLED = 0.02  # decisions whose own-active item slot is still the unfilled sentinel
MIN_IMPUTE_ABILITY = 0.99
MIN_IMPUTE_MOVES = 3.95
MAX_USAGE_MISSING = 0.02  # own mons with no usage entry (left log-only); expected ~0


def _present_counts(obs: np.ndarray) -> tuple[int, int]:
    """(ego, opp) species present in one decision obs, from each block's present flag."""
    ego = sum(1 for i in range(6) if obs[i * _PDIM] > 0.5)
    opp = sum(1 for i in range(6, 12) if obs[i * _PDIM] > 0.5)
    return ego, opp


def _active_detail(obs: np.ndarray) -> tuple[float, float, float] | None:
    """Own active mon's (item-known, ability-known, move-count/4) from the encoder ID channels.

    Finds the own active block by its active flag (channel 1), then reads its trailing item/
    ability ID channels (nonzero = a real vocab id = known) and counts the active-move blocks
    whose move-ID channel is nonzero. Returns ``None`` if no own mon is active this decision.
    """
    for i in range(6):
        if obs[i * _PDIM + 1] > 0.5:  # active flag
            item = 1.0 if obs[i * _PDIM + _ITEM_CH] > 0.5 else 0.0
            ability = 1.0 if obs[i * _PDIM + _ABILITY_CH] > 0.5 else 0.0
            ms, md = OBS_LAYOUT.moves_start, OBS_LAYOUT.move_dim
            moves = sum(1.0 for j in range(4) if obs[ms + j * md + md - 1] > 0.5)
            return item, ability, moves
    return None


def _strip_preview(lines: list[str]) -> list[str]:
    """Drop ``|poke|`` team-preview lines -> RB-style progressive reveal (negative control)."""
    return [ln for ln in lines if not ln.startswith("|poke|")]


class _ItemStateProbe:
    """Per-decision own-active item-state counts (known / consumed-None / unfilled sentinel)."""

    def __init__(self) -> None:
        self.known = self.consumed = self.sentinel = 0

    def classify(self, battle: Any) -> None:
        mon = battle.active_pokemon
        if mon is None:
            return
        if mon.item == _UNKNOWN_ITEM:
            self.sentinel += 1
        elif mon.item is None:
            self.consumed += 1
        else:
            self.known += 1

    @property
    def total(self) -> int:
        return self.known + self.consumed + self.sentinel

    def rates(self) -> dict[str, float]:
        n = self.total
        return {
            "known_rate": self.known / n if n else 0.0,
            "consumed_none_rate": self.consumed / n if n else 0.0,
            "unfilled_rate": self.sentinel / n if n else 0.0,
            "n_decisions": float(n),
        }


@contextmanager
def _probe_item_states() -> Iterator[_ItemStateProbe]:
    """Wrap the ingest samplers to read the battle state at each labelled decision.

    The obs item channel cannot distinguish an unfilled sentinel (imputation failure) from a
    consumed item's ``None`` (live-faithful truth) -- both encode UNK -- so the decomposition
    must come from the battle object at snapshot time. The ``kits is not None`` guard excludes
    pass-1 prescan reconstructions (they run with ``kits=None`` through the same samplers).
    """
    probe = _ItemStateProbe()
    orig_move, orig_switch = ingest._move_sample, ingest._switch_sample

    def move(battle: Any, parts: list[str], tera: bool, weights: Any, kits: Any) -> Any:
        out = orig_move(battle, parts, tera, weights, kits)
        if out is not None and kits is not None:
            probe.classify(battle)
        return out

    def switch(battle: Any, ident: str, weights: Any, kits: Any) -> Any:
        out = orig_switch(battle, ident, weights, kits)
        if out is not None and kits is not None:
            probe.classify(battle)
        return out

    ingest._move_sample, ingest._switch_sample = move, switch
    try:
        yield probe
    finally:
        ingest._move_sample, ingest._switch_sample = orig_move, orig_switch


def _accuracy_update(kits: dict, prior: UsagePrior, acc: dict[str, float]) -> None:
    """Score the prior against the *revealed* truth in pre-imputation kits (advisory only).

    Reveal-biased (items reveal when they trigger, so Leftovers-likes are over-represented)
    and the mechanism under test is distributional match, not point recovery -- hence report,
    not kill. top1 = truth equals the prior's modal pick; topk = truth is anywhere in the
    stored top-K list (what sampling can actually draw).
    """
    for kit in kits.values():
        usage = prior.species.get(kit.species) if kit.species else None
        if usage is None:
            continue
        if kit.item is not None and usage.items:
            acc["n_item"] += 1
            acc["item_top1"] += float(kit.item == usage.items[0][0])
            acc["item_topk"] += float(kit.item in {i for i, _ in usage.items})
        if kit.ability is not None and usage.abilities:
            acc["n_ability"] += 1
            acc["ability_top1"] += float(kit.ability == usage.abilities[0][0])
            acc["ability_topk"] += float(kit.ability in {a for a, _ in usage.abilities})
        top_moves = {m for m, _ in usage.moves}
        for move in kit.moves:
            acc["n_moves"] += 1
            acc["moves_topk"] += float(move in top_moves)


def _score(
    replays: list[dict[str, Any]],
    weights: RewardWeights,
    strip: bool = False,
    complete: bool = True,
    prior: UsagePrior | None = None,
) -> dict[str, float]:
    """Reconstruct every POV; return coverage, detail channels + health counters.

    ``strip`` drops ``|poke|`` preview (species negative control); ``complete`` toggles two-pass
    own-team completion (``False`` = the v1 progressive-reveal detail control); ``prior`` adds
    the Build-4 usage-imputation arm (accuracy-vs-revealed-truth scored first, then imputed).
    """
    povs = fatal = 0
    turns = dropped = 0
    cov_sum = cov_n = 0.0
    item_sum = ability_sum = moves_sum = detail_n = 0.0
    imputed_mons = missing_mons = 0
    acc = dict.fromkeys(
        ("n_item", "item_top1", "item_topk", "n_ability", "ability_top1", "ability_topk",
         "n_moves", "moves_topk"),
        0.0,
    )
    winner_returns: list[float] = []
    loser_returns: list[float] = []

    for replay in replays:
        log = replay.get("log")
        players = replay.get("players")
        if not isinstance(log, str) or not isinstance(players, list) or len(players) != 2:
            continue
        lines = log.split("\n")
        if strip:
            lines = _strip_preview(lines)
        gen = _gen_from_log(log)
        tag = str(replay.get("id") or "r")
        for username in players:
            povs += 1
            pov_tag = f"{tag}-{username}"
            try:
                kits = (
                    _prescan_kits(lines, str(username), pov_tag, gen, weights)
                    if complete
                    else None
                )
                if kits is not None and prior is not None:
                    _accuracy_update(kits, prior, acc)  # against truth, so BEFORE imputing
                    imputed, missing = _impute_kits(kits, pov_tag, prior)
                    imputed_mons += imputed
                    missing_mons += missing
                battle, records, values, drop = _reconstruct_pov(
                    lines, str(username), pov_tag, gen, weights, kits=kits
                )
            except Exception:  # noqa: BLE001 -- one bad POV must not abort the gate
                fatal += 1
                continue
            dropped += drop
            turns += len(records)
            for obs, _, _ in records:
                ego, opp = _present_counts(obs)
                cov_sum += (ego + opp) / 12.0
                cov_n += 1.0
                detail = _active_detail(obs)
                if detail is not None:
                    item_sum += detail[0]
                    ability_sum += detail[1]
                    moves_sum += detail[2]
                    detail_n += 1.0
            rewards = _episode_rewards(battle, records, values, weights)
            if rewards:
                ret = float(sum(rewards))
                (winner_returns if battle.won else loser_returns).append(ret)

    seen = turns + dropped
    total_kits = imputed_mons + missing_mons
    return {
        "povs": float(povs),
        "parse_rate": (povs - fatal) / povs if povs else 0.0,
        "turns": float(turns),
        "drop_rate": dropped / seen if seen else 0.0,
        "species_coverage": cov_sum / cov_n if cov_n else 0.0,
        "item_known": item_sum / detail_n if detail_n else 0.0,
        "ability_known": ability_sum / detail_n if detail_n else 0.0,
        "move_count": moves_sum / detail_n if detail_n else 0.0,
        "winner_return": float(np.mean(winner_returns)) if winner_returns else 0.0,
        "loser_return": float(np.mean(loser_returns)) if loser_returns else 0.0,
        "imputed_mons": float(imputed_mons),
        "usage_missing_mons": float(missing_mons),
        "usage_missing_rate": missing_mons / total_kits if total_kits else 0.0,
        "item_top1": acc["item_top1"] / acc["n_item"] if acc["n_item"] else 0.0,
        "item_topk": acc["item_topk"] / acc["n_item"] if acc["n_item"] else 0.0,
        "ability_top1": acc["ability_top1"] / acc["n_ability"] if acc["n_ability"] else 0.0,
        "ability_topk": acc["ability_topk"] / acc["n_ability"] if acc["n_ability"] else 0.0,
        "moves_topk_containment": acc["moves_topk"] / acc["n_moves"] if acc["n_moves"] else 0.0,
        "n_item_truth": acc["n_item"],
        "n_ability_truth": acc["n_ability"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="replays/gen9ou", help="Cached gen9ou replay JSON dir")
    ap.add_argument("--sample", type=int, default=200, help="Max replays to score")
    ap.add_argument(
        "--fetch",
        type=int,
        default=0,
        help="If >0 and cache is thin, fetch this many gen9ou replays first",
    )
    ap.add_argument("--min-rating", type=int, default=1200)
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()

    if args.fetch > 0:
        fetch_replays(
            battle_format="gen9ou",
            min_rating=args.min_rating,
            limit=args.fetch,
            cache_dir=args.cache_dir,
            max_pages=200,
        )

    paths = cached_replay_paths(args.cache_dir)[: args.sample]
    replays: list[dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                replays.append(json.load(fh))
        except (OSError, ValueError):
            continue
    if not replays:
        raise SystemExit(f"no cached replays under {args.cache_dir} -- scrape first (or --fetch N)")

    prior = load_usage_prior("gen9ou")
    if prior is None:  # the gate must not silently score a no-op imputation arm
        raise SystemExit("no usage-prior artifact for gen9ou -- run scripts/build_usage_prior.py")

    weights = RewardWeights()
    with _probe_item_states() as probe:  # Build-4 training path + item-state decomposition
        impute_m = _score(replays, weights, complete=True, prior=prior)
    item_states = probe.rates()
    main_m = _score(replays, weights, strip=False, complete=True)  # log-only completion arm
    species_ctrl = _score(replays, weights, strip=True, complete=True)  # species lift control
    detail_ctrl = _score(replays, weights, strip=False, complete=False)  # v1 no-completion control

    coverage = main_m["species_coverage"]
    lift = coverage - species_ctrl["species_coverage"]
    reward_gap = main_m["winner_return"] - main_m["loser_return"]
    item_lift = main_m["item_known"] - detail_ctrl["item_known"]
    ability_lift = main_m["ability_known"] - detail_ctrl["ability_known"]
    move_lift = main_m["move_count"] - detail_ctrl["move_count"]

    checks = {
        "parse_rate_ok": main_m["parse_rate"] >= MIN_PARSE_RATE,
        "coverage_ok": coverage >= MIN_COVERAGE,
        "drop_rate_ok": main_m["drop_rate"] <= MAX_DROP_RATE,
        "reward_sign_ok": reward_gap > 0.0,
        "species_control_has_teeth": lift >= MIN_COVERAGE_LIFT,
        "item_completion_has_teeth": item_lift >= MIN_ITEM_LIFT,
        "move_completion_has_teeth": move_lift >= MIN_MOVE_LIFT,
        "impute_item_unfilled_ok": item_states["unfilled_rate"] <= MAX_ITEM_UNFILLED,
        "impute_ability_full": impute_m["ability_known"] >= MIN_IMPUTE_ABILITY,
        "impute_moves_full": impute_m["move_count"] >= MIN_IMPUTE_MOVES,
        "usage_species_coverage_ok": impute_m["usage_missing_rate"] <= MAX_USAGE_MISSING,
    }
    verdict = "PASS" if all(checks.values()) else "KILL"

    # Absolute ON-value = each arm's ceiling; residual = live-ideal - ceiling. Build 3 proved
    # the log-only residual leaves the agent OOD; the imputation arm must collapse it to ~0.
    residual = {
        "item": IDEAL_ITEM - main_m["item_known"],
        "ability": IDEAL_ABILITY - main_m["ability_known"],
        "moves": IDEAL_MOVES - main_m["move_count"],
    }
    residual_impute = {
        "item": IDEAL_ITEM - impute_m["item_known"],
        "ability": IDEAL_ABILITY - impute_m["ability_known"],
        "moves": IDEAL_MOVES - impute_m["move_count"],
    }
    accuracy = {
        k: impute_m[k]
        for k in (
            "item_top1", "item_topk", "ability_top1", "ability_topk",
            "moves_topk_containment", "n_item_truth", "n_ability_truth",
        )
    }

    result = {
        "n_replays": len(replays),
        "with_imputation": impute_m,
        "with_completion": main_m,
        "stripped_preview_control": species_ctrl,
        "no_completion_control": detail_ctrl,
        "coverage_lift": lift,
        "reward_gap": reward_gap,
        "detail_lift": {"item": item_lift, "ability": ability_lift, "moves": move_lift},
        "residual_vs_live": residual,
        "residual_vs_live_imputed": residual_impute,
        "live_reference": LIVE_REFERENCE,
        "item_state_decomposition": item_states,
        "imputation_accuracy": accuracy,
        "usage_prior_version": prior.version,
        "thresholds": {
            "min_parse_rate": MIN_PARSE_RATE,
            "min_coverage": MIN_COVERAGE,
            "max_drop_rate": MAX_DROP_RATE,
            "min_coverage_lift": MIN_COVERAGE_LIFT,
            "min_item_lift": MIN_ITEM_LIFT,
            "min_move_lift": MIN_MOVE_LIFT,
            "max_item_unfilled": MAX_ITEM_UNFILLED,
            "min_impute_ability": MIN_IMPUTE_ABILITY,
            "min_impute_moves": MIN_IMPUTE_MOVES,
            "max_usage_missing": MAX_USAGE_MISSING,
        },
        "checks": checks,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== OU ingest Gate A ({len(replays)} replays) ===")
    print(f"parse rate     {main_m['parse_rate']:.3f}   (>= {MIN_PARSE_RATE})")
    print(
        f"species cover  {coverage:.3f}   (>= {MIN_COVERAGE}); "
        f"stripped {species_ctrl['species_coverage']:.3f}; "
        f"lift {lift:+.3f} (>= {MIN_COVERAGE_LIFT})"
    )
    print(f"drop rate      {main_m['drop_rate']:.3f}   (<= {MAX_DROP_RATE})")
    print(
        f"reward sign    winner {main_m['winner_return']:+.3f} vs loser "
        f"{main_m['loser_return']:+.3f}  gap {reward_gap:+.3f} (> 0)"
    )
    print("own-active detail channels (IMPUTE | ON two-pass | OFF v1 | live ref):")
    print(
        f"  item-known   {impute_m['item_known']:.3f} | "
        f"{main_m['item_known']:.3f} | {detail_ctrl['item_known']:.3f} | "
        f"{LIVE_REFERENCE['item']:.3f}   [two-pass lift {item_lift:+.3f} >= {MIN_ITEM_LIFT}]"
    )
    print(
        f"  item states  known {item_states['known_rate']:.3f} / consumed-None "
        f"{item_states['consumed_none_rate']:.3f} (live-faithful) / UNFILLED "
        f"{item_states['unfilled_rate']:.4f} (<= {MAX_ITEM_UNFILLED})"
    )
    print(
        f"  ability-known {impute_m['ability_known']:.3f} (>= {MIN_IMPUTE_ABILITY}) | "
        f"{main_m['ability_known']:.3f} | {detail_ctrl['ability_known']:.3f} | "
        f"{LIVE_REFERENCE['ability']:.3f}   [two-pass lift {ability_lift:+.3f} reported]"
    )
    print(
        f"  move-count   {impute_m['move_count']:.3f} (>= {MIN_IMPUTE_MOVES}) | "
        f"{main_m['move_count']:.3f} | {detail_ctrl['move_count']:.3f} | "
        f"{LIVE_REFERENCE['moves']:.3f}   [two-pass lift {move_lift:+.3f} >= {MIN_MOVE_LIFT}]"
    )
    print(
        f"imputation     {int(impute_m['imputed_mons'])} mons imputed; missing-species rate "
        f"{impute_m['usage_missing_rate']:.4f} (<= {MAX_USAGE_MISSING})"
    )
    print(
        f"prior-vs-truth (advisory) item top1 {accuracy['item_top1']:.3f} / topK "
        f"{accuracy['item_topk']:.3f} (n={int(accuracy['n_item_truth'])}); ability top1 "
        f"{accuracy['ability_top1']:.3f} / topK {accuracy['ability_topk']:.3f} "
        f"(n={int(accuracy['n_ability_truth'])}); moves topK containment "
        f"{accuracy['moves_topk_containment']:.3f}"
    )
    if accuracy["item_topk"] < 0.6:
        print("  ADVISORY: item topK agreement < 0.6 -- prior may be noisy; investigate")
    print(f"turns kept     {int(main_m['turns'])} over {int(main_m['povs'])} POV-episodes")
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    if verdict == "KILL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

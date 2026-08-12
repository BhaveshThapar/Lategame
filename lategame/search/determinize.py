"""Determinize a poke-env POV into a Showdown reconstruction spec (Lever 11 / R-PREDICT).

Live search needs a *full* two-sided battle to fork+step, but our POV hides the opponent's
team/sets/exact stats. ``battle_to_spec`` transcribes everything we DO observe -- our full
team (known from the request), the opponent's revealed mons, and the shared field/hazards --
into the ``forward_driver.js`` spec. ``pokeenv_digest`` reads the same observable fields straight
from poke-env so the reconstruction mini-gate can check the driver honored them.

**What is hidden differs by format, and so does how it is filled.** On Random Battles the
opponent's *species* are unknown and the driver samples them from that format's own generator.
On a teambuilt format (gen9ou, VGC) team preview names all six up front and the unknown is the
*sets* -- so nothing is sampled species-wise (``fill=0``) and each mon's unrevealed
moves/item/ability come from the Smogon usage prior instead (``data.usage_prior``).

We always reconstruct ourselves as ``p1`` and the opponent as ``p2`` -- the encoder is
POV-relative (our team vs the opponent's), so the absolute slot label does not matter.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from poke_env.battle import AbstractBattle, Pokemon
from poke_env.battle.side_condition import STACKABLE_CONDITIONS

# Showdown weather ids differ from a couple of poke-env enum names; everything else is toID().
_WEATHER_ID = {
    "SNOWSCAPE": "snow",
    "HAIL": "hail",
    "RAINDANCE": "raindance",
    "SANDSTORM": "sandstorm",
    "SUNNYDAY": "sunnyday",
    "DESOLATELAND": "desolateland",
    "PRIMORDIALSEA": "primordialsea",
    "DELTASTREAM": "deltastream",
}


#: The format the driver assumes when a spec omits one -- i.e. every pre-2026-08 caller, so the
#: whole R-PREDICT history reconstructs byte-identically.
DEFAULT_FORMAT = "gen9randombattle"


def _to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _is_random_battle(battle_format: str) -> bool:
    """Whether the server generates the teams -- i.e. the opponent's SPECIES are the unknown.

    Keyed on ``random`` rather than ``randombattle``: ``gen9randomdoublesbattle`` contains no
    ``randombattle`` substring, and getting it wrong would send a random-team format down the
    teambuilt path, where it would look up a usage prior that does not exist for it and stop
    filling the hidden roster at all.
    """
    return "random" in _to_id(battle_format)


@lru_cache(maxsize=8)
def _usage_prior_for(battle_format: str) -> Any:
    """The Smogon usage prior for a teambuilt format, or ``None`` if no artifact exists.

    Cached because one agent determinizes thousands of times per session and the artifact is a
    multi-megabyte JSON. Returning ``None`` is not an error: the driver still reconstructs, the
    opponent is just limited to its revealed moves -- which the recon gate will show up as a
    weak-but-faithful foe rather than as a crash.
    """
    from lategame.data.usage_prior import load_usage_prior

    return load_usage_prior(battle_format)


def _mon_set_info(
    mon: Pokemon,
    *,
    full: bool,
    prior: Any = None,
    battle_tag: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    """Known set fields for a mon. ``full`` (our side) passes everything; opp passes revealed.

    ``prior`` (a ``UsagePrior``) fills the unrevealed remainder on a TEAMBUILT format. There the
    opponent's six species are known from team preview but their sets are not -- the mirror image
    of Random Battles, where the species are hidden and the driver's own generator invents legal
    sets. Without this the driver would reconstruct a gen9ou opponent holding one revealed move
    and nothing else, and search would evaluate a foe that cannot fight back.
    """
    info: dict[str, Any] = {"species": _to_id(mon.species)}
    moves = [m.id for m in mon.moves.values() if m and m.id]
    if mon.ability:
        info["ability"] = _to_id(mon.ability)
    # `None` means the item was CONSUMED (a fact); the sentinel means never revealed (a gap).
    # Only the gap may be filled -- see ingest._complete_own_team, which draws the same line.
    item_unknown = mon.item in ("unknown_item", "unknownitem")
    if mon.item and not item_unknown:
        info["item"] = _to_id(mon.item)
    if mon.tera_type is not None:
        info["teraType"] = mon.tera_type.name.title()

    if prior is not None:
        from lategame.data.usage_prior import kit_seed, sample_kit

        drawn = sample_kit(
            prior,
            _to_id(mon.species),
            known_moves={_to_id(m) for m in moves},
            need_item=item_unknown or not mon.item,
            need_ability=not mon.ability,
            seed=kit_seed(f"{battle_tag}|{seed}", _to_id(mon.species)),
        )
        if drawn is not None:
            extra_moves, item, ability = drawn
            moves = moves + [m for m in extra_moves if m not in moves]
            if item and "item" not in info:
                info["item"] = _to_id(item)
            if ability and "ability" not in info:
                info["ability"] = _to_id(ability)

    if moves:
        info["moves"] = moves
    if full:
        if mon.level:
            info["level"] = mon.level
        if mon.gender is not None:
            info["gender"] = {"MALE": "M", "FEMALE": "F"}.get(mon.gender.name, "")
    elif prior is not None and mon.level:
        # Level is format-defining on a teambuilt ladder (gen9ou is 100, VGC is 50) and the
        # opponent's is observable, so it must not be left to the driver's default.
        info["level"] = mon.level
    return info


def _mon_state(mon: Pokemon) -> dict[str, Any]:
    """Observed dynamic state (hp fraction, status, boosts, fainted)."""
    state: dict[str, Any] = {
        "hp_frac": 0.0 if mon.fainted else float(mon.current_hp_fraction or 0.0),
        "fainted": bool(mon.fainted),
    }
    if mon.status is not None and mon.status.name != "FNT":
        state["status"] = _to_id(mon.status.name)
    boosts = {k: v for k, v in mon.boosts.items() if v}
    if boosts:
        state["boosts"] = boosts
    return state


def _side_spec(
    mons: list[Pokemon],
    active: Pokemon | None,
    *,
    full: bool,
    fill: int,
    prior: Any = None,
    battle_tag: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    team = [
        _mon_set_info(m, full=full, prior=prior, battle_tag=battle_tag, seed=seed) for m in mons
    ]
    state = [_mon_state(m) for m in mons]
    active_idx = 0
    if active is not None:
        for i, m in enumerate(mons):
            if m is active:
                active_idx = i
                break
    return {"team": team, "state": state, "active": active_idx, "fill": max(0, fill)}


def _hazards(conditions: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for cond, value in conditions.items():
        if cond.name == "UNKNOWN":
            continue
        layers = int(value) if cond in STACKABLE_CONDITIONS else 1
        out[_to_id(cond.name)] = max(1, layers)
    return out


def _field(battle: AbstractBattle) -> dict[str, Any]:
    weather = ""
    for w in battle.weather:
        if w.name != "UNKNOWN":
            weather = _WEATHER_ID.get(w.name, _to_id(w.name))
            break
    terrain = ""
    pseudo: list[str] = []
    for f in battle.fields:
        if f.name == "UNKNOWN":
            continue
        if f.name.endswith("_TERRAIN"):
            terrain = _to_id(f.name)
        else:
            pseudo.append(_to_id(f.name))
    return {"weather": weather, "terrain": terrain, "pseudo": pseudo}


def battle_to_spec(battle: AbstractBattle, seed: int = 0) -> dict[str, Any]:
    """Transcribe a poke-env POV battle into a forward_driver reconstruction spec.

    ``format`` is carried so the driver starts the fork in the RIGHT format. It used to be a
    module constant pinned to ``gen9randombattle`` there; on any other format that silently
    reconstructs under the wrong legality and the search runs on a battle that could not exist.
    """
    our = list(battle.team.values())
    opp = list(battle.opponent_team.values())
    battle_format = battle.format or DEFAULT_FORMAT
    is_rb = _is_random_battle(battle_format)
    # The usage prior is the teambuilt counterpart of the driver's random-set generator, and it is
    # loaded only there: on RB the generator already invents a legal set from a bare species.
    prior = None if is_rb else _usage_prior_for(battle_format)
    return {
        "seed": int(seed),
        "format": battle_format,
        "p1": _side_spec(our, battle.active_pokemon, full=True, fill=0),
        "p2": _side_spec(
            opp,
            battle.opponent_active_pokemon,
            full=False,
            # SPECIES fill is a Random-Battle notion: there the opponent's roster is hidden and
            # the driver samples it. A teambuilt format reveals all six at team preview, so
            # nothing is missing -- what is unknown there is the SETS, filled from `prior`.
            fill=max(0, 6 - len(opp)) if is_rb else 0,
            prior=prior,
            battle_tag=battle.battle_tag or "",
            seed=int(seed),
        ),
        "field": _field(battle),
        "hazards": {
            "p1": _hazards(battle.side_conditions),
            "p2": _hazards(battle.opponent_side_conditions),
        },
        "turn": int(battle.turn),
    }


def _digest_side(mons: list[Pokemon], active: Pokemon | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in mons:
        out[_to_id(m.species)] = {
            "hp": 0.0 if m.fainted else round(float(m.current_hp_fraction or 0.0), 2),
            "status": _to_id(m.status.name) if (m.status and m.status.name != "FNT") else "",
            "fainted": bool(m.fainted),
            "active": m is active,
            "boosts": {k: v for k, v in m.boosts.items() if v},
        }
    return out


def pokeenv_digest(battle: AbstractBattle) -> dict[str, Any]:
    """Observable digest from poke-env, in the same shape ``forward_driver.digest`` emits."""
    return {
        "turn": int(battle.turn),
        "weather": _field(battle)["weather"],
        "terrain": _field(battle)["terrain"],
        "p1": _digest_side(list(battle.team.values()), battle.active_pokemon),
        "p2": _digest_side(list(battle.opponent_team.values()), battle.opponent_active_pokemon),
        "hazards": {
            "p1": _hazards(battle.side_conditions),
            "p2": _hazards(battle.opponent_side_conditions),
        },
    }

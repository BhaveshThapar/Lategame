"""Determinize a poke-env POV into a Showdown reconstruction spec (Lever 11 / R-PREDICT).

Live search needs a *full* two-sided battle to fork+step, but our POV hides the opponent's
team/sets/exact stats. ``battle_to_spec`` transcribes everything we DO observe -- our full
team (known from the request), the opponent's revealed mons, and the shared field/hazards --
into the ``forward_driver.js`` spec; the driver fills hidden opponent slots from the gen9
random-battle pool. ``pokeenv_digest`` reads the same observable fields straight from poke-env
so the reconstruction mini-gate can check the driver honored them.

We always reconstruct ourselves as ``p1`` and the opponent as ``p2`` -- the encoder is
POV-relative (our team vs the opponent's), so the absolute slot label does not matter.
"""

from __future__ import annotations

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


def _to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _mon_set_info(mon: Pokemon, *, full: bool) -> dict[str, Any]:
    """Known set fields for a mon. ``full`` (our side) passes everything; opp passes revealed."""
    info: dict[str, Any] = {"species": _to_id(mon.species)}
    moves = [m.id for m in mon.moves.values() if m and m.id]
    if moves:
        info["moves"] = moves
    if mon.ability:
        info["ability"] = _to_id(mon.ability)
    if mon.item and mon.item not in ("unknown_item", "unknownitem"):
        info["item"] = _to_id(mon.item)
    if mon.tera_type is not None:
        info["teraType"] = mon.tera_type.name.title()
    if full:
        if mon.level:
            info["level"] = mon.level
        if mon.gender is not None:
            info["gender"] = {"MALE": "M", "FEMALE": "F"}.get(mon.gender.name, "")
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
    mons: list[Pokemon], active: Pokemon | None, *, full: bool, fill: int
) -> dict[str, Any]:
    team = [_mon_set_info(m, full=full) for m in mons]
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
    """Transcribe a poke-env POV battle into a forward_driver reconstruction spec."""
    our = list(battle.team.values())
    opp = list(battle.opponent_team.values())
    return {
        "seed": int(seed),
        "p1": _side_spec(our, battle.active_pokemon, full=True, fill=0),
        "p2": _side_spec(
            opp, battle.opponent_active_pokemon, full=False, fill=max(0, 6 - len(opp))
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

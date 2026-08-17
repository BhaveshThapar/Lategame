"""M1 rule-based baseline agent.

A damage/STAB/type-aware policy: each turn it picks the highest expected-damage
move (via the R-CALC seed in ``engine.damage``), and switches out of a clearly
bad type matchup when a benched Pokemon is meaningfully better. It is the fixed
baseline that later learned agents (BC -> offline RL -> self-play) are measured
against -- see plan.md, milestone M1.

Switching is intentionally conservative: tempo loss from over-switching is a
classic failure mode, so we only switch out proactively when our best available
move is weak AND a bench Pokemon offers a clearly better matchup.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player import (
    BattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    Player,
    SingleBattleOrder,
)

from rotomai.engine.damage import score_move

# A move scoring at or below this expected-power proxy counts as "weak" (e.g. a
# resisted 80 BP STAB move scores 80 * 1.5 * 0.5 = 60), which makes us consider
# switching. Matchup is an (offense - defense) type-effectiveness differential;
# we only switch if a bench mon improves it by at least this margin.
WEAK_MOVE_SCORE = 60.0
SWITCH_MARGIN = 1.0

# The decision a heuristic turn resolves to: pick a move, switch to a bench mon, or
# no preference (caller falls back to the default move). ``str`` tag keeps it JSON-cheap
# and easy for the white-box opponent model (Lever 14) to map onto a driver choice.
HeuristicPick = tuple[str, Move] | tuple[str, Pokemon] | None


def matchup(mon: Pokemon | None, opponent: Pokemon | None) -> float:
    """Type-based (offense - defense) differential of ``mon`` vs ``opponent``.

    Offense: how hard ``mon``'s STAB types hit ``opponent``.
    Defense: how hard ``opponent``'s STAB types hit ``mon`` (penalty).
    """
    if mon is None or opponent is None:
        return 0.0
    offense = max(
        (opponent.damage_multiplier(t) for t in mon.types if t is not None),
        default=1.0,
    )
    defense = max(
        (mon.damage_multiplier(t) for t in opponent.types if t is not None),
        default=1.0,
    )
    return offense - defense


def heuristic_pick(
    active: Pokemon | None,
    opponent: Pokemon | None,
    available_moves: Sequence[Move],
    available_switches: Sequence[Pokemon],
) -> HeuristicPick:
    """The pure R-CALC decision rule, shared by ``HeuristicAgent`` and the Lever 14
    white-box opponent model.

    Picks the highest expected-damage move; switches out only when the best move is weak
    AND a bench mon meaningfully improves the type matchup (conservative tempo). Returns
    ``("move", Move)`` / ``("switch", Pokemon)`` / ``None`` (no decision -> default).
    Kept a free function of *raw* poke-env objects so the opponent model can evaluate it on
    the modeled opponent's active/legal set without a live ``Player`` or a second battle POV.
    """
    if available_moves:
        best_move = max(available_moves, key=lambda m: score_move(m, active, opponent))
        best_score = score_move(best_move, active, opponent)
        if available_switches and best_score <= WEAK_MOVE_SCORE:
            switch = max(available_switches, key=lambda m: matchup(m, opponent))
            if matchup(switch, opponent) > matchup(active, opponent) + SWITCH_MARGIN:
                return ("switch", switch)
        return ("move", best_move)

    if available_switches:
        return ("switch", max(available_switches, key=lambda m: matchup(m, opponent)))

    return None


def best_target(
    active: Pokemon | None,
    foes: Sequence[tuple[int, Pokemon]],
    available_moves: Sequence[Move],
) -> tuple[int, Pokemon] | None:
    """Which opposing slot this Pokemon most wants to hit, by the same R-CALC score.

    The natural doubles extension of "pick the highest expected-damage move": the move and the
    target are chosen together, because on doubles a move's value depends on which of the two
    foes it lands on. Returns ``(slot_index, pokemon)`` or ``None`` when nothing is targetable.
    """
    if not foes:
        return None
    if not available_moves:
        # No moves to compare with -- fall back to the worst matchup FOR US, which is the foe a
        # switch decision should be evaluated against.
        return min(foes, key=lambda pair: matchup(active, pair[1]))
    return max(
        foes,
        key=lambda pair: max(score_move(m, active, pair[1]) for m in available_moves),
    )


#: One slot's doubles decision: (Move, showdown target) or (Pokemon to switch to, unused), or
#: ``None`` for "no preference -- default". Ordered by slot.
DoublesPick = tuple[Move | Pokemon, int] | None


def doubles_pick(battle: DoubleBattle) -> list[DoublesPick]:
    """The doubles decision rule as a PURE function of the battle, one entry per slot.

    Free-standing for the same reason ``heuristic_pick`` is: the Lever-14 white-box opponent model
    has to evaluate *the eval opponent's* policy on a reconstructed POV, and it must call the
    identical function the live agent calls. Duplicating the rule is what would silently reopen the
    model-vs-reality gap that made L11/L12's opponent model too weak to be worth searching against.

    Two things doubles forces that singles never did.

    *A move needs a TARGET, and the target is part of the decision.* A move's expected damage
    depends on which foe it hits, so ``best_target`` picks the slot first and the move is then
    chosen against that foe. The chosen target is only used if the simulator agrees it is legal
    (spread and self-targeting moves have their own fixed target sets), otherwise the first legal
    target is taken -- guessing a target Showdown rejects would cost the turn.

    *The two slots are not independent.* Both may not switch to the same benched Pokemon, so a
    bench mon claimed by slot 0 is removed from slot 1's options before slot 1 decides. Without
    that the order is illegal and the server plays a default move instead -- a silent lost turn.
    """
    out: list[DoublesPick] = []
    claimed: set[str] = set()
    # WHICH SLOTS ARE ACTUALLY BEING ASKED. On a partial replacement -- one active fainted, the
    # other still standing -- `force_switch` is e.g. [False, True] and ONLY that slot may act; the
    # other must `pass`. poke-env still populates `available_switches` for the unasked slot, so a
    # rule that reads only that list produces an order the server REJECTS. It then re-requests,
    # which is why an unfixed version showed 28 choose_move calls for 10 turns and made 98% of
    # collected turns unlabelable. A forced slot may only switch, never move.
    force = battle.force_switch if isinstance(battle.force_switch, list) else [False, False]
    any_forced = any(force)
    foes = [
        (i, foe)
        for i, foe in enumerate(battle.opponent_active_pokemon)
        if foe is not None and not foe.fainted
    ]

    for slot in (0, 1):
        forced = force[slot] if slot < len(force) else False
        if any_forced and not forced:
            out.append(None)  # not this slot's decision -- it must pass
            continue
        mon = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        moves = (
            []
            if forced  # a replacement request offers switches only
            else list(battle.available_moves[slot])
            if slot < len(battle.available_moves)
            else []
        )
        switches = [
            m
            for m in (
                battle.available_switches[slot] if slot < len(battle.available_switches) else []
            )
            if m.species not in claimed
        ]
        # A FORCED slot has no active by definition -- that is what it is replacing -- so the
        # "no mon here" guard must not fire on it, or the one slot being asked to act returns
        # nothing and the server plays a default in its place.
        if (mon is None and not forced) or (not moves and not switches):
            out.append(None)
            continue

        target = best_target(mon, foes, moves)
        pick = heuristic_pick(mon, target[1] if target else None, moves, switches)
        if pick is None:
            out.append(None)
            continue
        # Narrowed on the object, not the ``str`` tag: the tag carries no type information.
        decision = pick[1]
        if isinstance(decision, Pokemon):
            claimed.add(decision.species)
            out.append((decision, DoubleBattle.EMPTY_TARGET_POSITION))
        elif mon is None:
            out.append(None)  # unreachable: a move implies an active, but do not guess
        else:
            legal = battle.get_possible_showdown_targets(decision, mon)
            wanted = target[0] + 1 if target else DoubleBattle.EMPTY_TARGET_POSITION
            chosen = (
                wanted if wanted in legal
                else (legal[0] if legal else DoubleBattle.EMPTY_TARGET_POSITION)
            )
            out.append((decision, chosen))
    return out


def doubles_order(
    battle: DoubleBattle,
    create_order: Callable[..., SingleBattleOrder],
    choose_default_move: Callable[[], BattleOrder],
) -> BattleOrder:
    """The M1 rule per slot, joined into a ``DoubleBattleOrder``. See ``_choose_doubles_move``.

    A FREE FUNCTION because ``SearchAgent`` needs it too and is not a ``HeuristicAgent``. In
    shaped-only mode the search has no trained policy to fall back on, so it falls back to the M1
    rule -- and it was falling back to ``heuristic_pick``, the SINGLES one, which on a
    ``DoubleBattle`` is handed ``available_moves`` as a list of per-slot lists and dies on
    ``'list' object has no attribute 'base_power'``.

    That failure is silent, which is why it survived being written down as fixed. poke-env
    dispatches protocol messages through a detached task, so the AttributeError is logged and
    swallowed, the agent never answers, and the SERVER plays a default move on the timer. An arm in
    that state reads as a very weak agent rather than a broken one -- the exact hazard
    ``arena._SINGLES_ONLY_AGENTS`` exists to refuse at build time, reappearing on the one path that
    is allowed through the refusal.

    Only ``create_order`` and ``choose_default_move`` are needed from the player, and both are
    ``Player`` methods, so nothing here assumes which agent is asking.
    """
    picks = doubles_pick(battle)
    orders: list[SingleBattleOrder] = [
        PassBattleOrder()
        if p is None
        else create_order(p[0])
        if isinstance(p[0], Pokemon)
        else create_order(p[0], move_target=p[1])
        for p in picks
    ]
    if all(isinstance(order, PassBattleOrder) for order in orders):
        return choose_default_move()
    return DoubleBattleOrder(orders[0], orders[1])


class HeuristicAgent(Player):
    """M1 baseline. Singles is the original rule; doubles applies the same rule per slot.

    The doubles path exists because the VGC ceiling probe needs TWO agents near the top of the
    doubles skill gradient and poke-env supplies exactly one (``SimpleHeuristicsPlayer``). The
    published RB and OU bands differ only at the top of the gradient -- both formats crush naive
    bots equally -- so a probe that can only measure below the strongest available bot cannot tell
    FORMAT_BOUND from MODEL_BOUND. See docs/RESULTS.md (G4 groundwork).
    """

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, DoubleBattle):
            return self._choose_doubles_move(battle)
        pick = heuristic_pick(
            battle.active_pokemon,
            battle.opponent_active_pokemon,
            battle.available_moves,
            battle.available_switches,
        )
        if pick is None:
            return self.choose_default_move()
        return self.create_order(pick[1])

    def _choose_doubles_move(self, battle: DoubleBattle) -> BattleOrder:
        """The same rule, once per slot, joined into a ``DoubleBattleOrder``.

        A SLOT WITH NO DECISION EMITS `pass`, NOT `default`, AND THE DIFFERENCE IS THE WHOLE
        DOUBLES LOOP (B6f). `DefaultBattleOrder` is poke-env's WHOLE-ORDER sentinel: its message
        is `/choose default`, and `DoubleBattleOrder.message` joins by string surgery
        (`f"{first.message}, {second.message[8:]}"`), so half a default produced
        `/choose default, move woodhammer 1` -- not a legal Showdown command. The server rejected
        it, poke-env re-requested the same state, and the agent answered identically forever.

        Measured before the fix, four battles per pair on gen9vgc2025regi:

            random vs random                    max  19 choose_move calls/battle (turn 17)
            simpleheuristics vs simpleheuristics max   9                          (turn  8)
            maxbasepower vs maxbasepower         max   8                          (turn  6)
            random vs HEURISTIC                 max 4001 calls, battle.turn 1..7
            HEURISTIC vs HEURISTIC              max 1153 calls, battle.turn 1..4

        poke-env's own baselines never loop; this agent did, and it is in every collection pool,
        which is why 94.2% of `data/vgc_rl.npz` came from 100 of 899 episodes.

        This is the WRITE side of the error B6d already fixed on the READ side:
        `doubles_action_space.normalize_half_default` exists because poke-env maps a half default
        to -2, its whole-order sentinel, when the per-slot layout says "this slot does nothing" is
        action 0, `pass`. The labelling was corrected then; the emission was not.

        Both slots idle IS the whole-order case, so that still answers `/choose default`.
        """
        return doubles_order(battle, self.create_order, self.choose_default_move)

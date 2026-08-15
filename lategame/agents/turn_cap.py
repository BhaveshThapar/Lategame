"""Per-battle decision ceiling: the hard half of the doubles loop fix.

``agents.loop_guard`` is a SOFT penalty and cannot break the state that actually dominated the
VGC shards. Measured on ``data/vgc_rl.npz``: 51.9% of all recorded turns had slot 0's legal set
= ``{pass}`` and slot 1's = two switches. Every legal option there is a switch, so there is
nothing for an escalating switch penalty to push the argmax toward -- the guard shortens
voluntary runs, it cannot terminate an absorbing forced-replacement cycle. The single worst
episode ran **12,795 recorded turns over 7 unique observation vectors**, and the top 100 of 899
episodes carried 94.2% of the shard.

So past a ceiling the battle is forfeited. Three things make that the honest choice rather than
a new bias:

* Those battles already LOSE -- episode 28's shaped reward sums to -3.18 -- they simply lose
  after thousands of wasted requests, on the server's inactivity timer.
* ``ForfeitBattleOrder`` is already a codec citizen: ``doubles_action_space.order_to_action``
  labels it ``[-1, -1]``, and ``train.offline_rl._run_epoch`` already zero-weights negative
  sentinels, so a forfeited tail cannot poison the actor loss.
* The ceiling is set far above any real battle (OU's longest recorded episode is 205 turns), so
  a run in which it fires at all is reporting a pathology, not being truncated.

``max_turns=None`` is exact identity -- nothing is counted and no dict is touched -- so every
path that does not opt in behaves exactly as it did before this module existed.
"""

from __future__ import annotations

from poke_env.battle import AbstractBattle

#: Generous by construction: ~6x the longest OU episode ever recorded (205 turns), and ~10x the
#: longest non-loop VGC one. A run that trips this is reporting a loop, not a long game.
DEFAULT_MAX_BATTLE_TURNS = 1200


class TurnCap:
    """Counts decisions per battle tag; reports when one has exceeded its ceiling.

    Deliberately counts ``choose_move`` CALLS rather than reading ``battle.turn``. The pathology
    is the server re-requesting the same turn -- ``battle.turn`` does not advance while it
    happens, so it is exactly the quantity that cannot see the loop.
    """

    def __init__(self, max_turns: int | None = DEFAULT_MAX_BATTLE_TURNS) -> None:
        self.max_turns = max_turns
        self._seen: dict[str, int] = {}
        self.capped: set[str] = set()

    def hit(self, battle: AbstractBattle) -> bool:
        """Count this decision; ``True`` once ``battle`` has passed its ceiling."""
        if self.max_turns is None:
            return False
        tag = str(battle.battle_tag)
        seen = self._seen.get(tag, 0) + 1
        self._seen[tag] = seen
        if seen > self.max_turns:
            self.capped.add(tag)
            return True
        return False

    def turns(self, battle_tag: str) -> int:
        """Decisions counted for ``battle_tag`` so far (0 when the cap is off)."""
        return self._seen.get(str(battle_tag), 0)

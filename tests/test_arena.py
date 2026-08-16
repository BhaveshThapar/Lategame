"""Arena smoke test. The end-to-end battle test needs a local Showdown server
on :8000; it is skipped automatically when the server is not running."""

import pytest

from lategame.eval.arena import build_player, evaluate
from tests.conftest import requires_server


def test_build_player_rejects_unknown_agent():
    with pytest.raises(ValueError):
        build_player("not-an-agent", "gen9randombattle")


def test_build_player_forwards_loop_penalty_only_to_learned_agents(monkeypatch):
    """loop_penalty must reach bc/offrl/ppo (they carry a LoopGuard) but never a baseline
    like random (RandomPlayer has no such kwarg)."""
    import lategame.eval.arena as arena

    captured: dict[str, dict] = {}

    def _fake(tag: str):
        class _Fake:
            def __init__(self, **kwargs):
                captured[tag] = kwargs

        return _Fake

    monkeypatch.setitem(arena.AGENTS, "bc", _fake("bc"))
    monkeypatch.setitem(arena.AGENTS, "random", _fake("random"))

    # Teams are passed because `build_player` now refuses a teamless teambuilt build; the
    # subject of this test is kwarg forwarding, and a real caller has to supply one anyway.
    build_player("bc", "gen9ou", checkpoint_path="x.pt", loop_penalty=4.0, team="T")
    build_player("random", "gen9ou", loop_penalty=4.0, team="T")

    assert captured["bc"]["loop_penalty"] == 4.0
    assert "loop_penalty" not in captured["random"]


class _FakePlayer:
    """A player whose counters advance by a scripted (won, lost, tied) on ``battle_against``.

    Counters are CUMULATIVE in poke-env (they run over ``player._battles``), so the fake is
    cumulative too -- ``evaluate_built`` differences around the games and a fake that reset would
    hide exactly the bug the differencing exists to avoid.
    """

    def __init__(self, won: int, lost: int, tied: int) -> None:
        self._script = (won, lost, tied)
        self.n_won_battles = 7  # nonzero starting offsets: a read of the ABSOLUTE
        self.n_lost_battles = 3  # counter instead of the delta must not pass
        self.n_tied_battles = 1
        self.n_finished_battles = 11

    async def battle_against(self, other: "_FakePlayer", n_battles: int) -> None:
        won, lost, tied = self._script
        self.n_won_battles += won
        self.n_lost_battles += lost
        self.n_tied_battles += tied
        self.n_finished_battles += won + lost + tied


async def test_a_tie_scores_a_half_not_a_loss():
    """``cross_evaluate`` reports ``n_won / n_finished``, so a tie sits in the denominator without
    contributing to the numerator -- scored as a loss. 6W/2L/2T is 0.70, not 0.60."""
    from lategame.eval.arena import evaluate_built

    p1 = _FakePlayer(won=6, lost=2, tied=2)
    assert await evaluate_built(p1, _FakePlayer(2, 6, 2), n_battles=10) == pytest.approx(0.70)


async def test_identical_to_the_old_win_rate_on_a_tie_free_record():
    """Every result this project has published is tie-free (singles ties are vanishingly rare on
    gen9ou), so the fix must not move a single one of them."""
    from lategame.eval.arena import evaluate_built

    p1 = _FakePlayer(won=7, lost=3, tied=0)
    assert await evaluate_built(p1, _FakePlayer(3, 7, 0), n_battles=10) == pytest.approx(7 / 10)


async def test_a_pair_where_nothing_finished_scores_zero_rather_than_dividing_by_zero():
    from lategame.eval.arena import evaluate_built

    p1 = _FakePlayer(won=0, lost=0, tied=0)
    assert await evaluate_built(p1, _FakePlayer(0, 0, 0), n_battles=10) == 0.0


@requires_server
async def test_random_vs_random_completes_a_battle():
    result = await evaluate("random", "random", n_battles=1)
    assert result.n_battles == 1
    assert 0.0 <= result.p1_win_rate <= 1.0


# --------------------------------------------------------------------------- #
# The doubles guard (G4/M6 groundwork)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "battle_format,doubles",
    [
        ("gen9randombattle", False),
        ("gen9ou", False),
        ("gen9vgc2025regi", True),
        ("gen9doublesou", True),
        ("gen9randomdoublesbattle", True),
        ("gen92v2doubles", True),
    ],
)
def test_doubles_formats_are_recognised(battle_format, doubles):
    from lategame.config import is_doubles_format

    assert is_doubles_format(battle_format) is doubles


def test_a_singles_only_agent_is_refused_on_a_doubles_format():
    """The refusal exists because the failure is SILENT otherwise.

    On a DoubleBattle poke-env passes `active_pokemon` as a list, so `HeuristicAgent` raises
    AttributeError -- but poke-env dispatches protocol messages through `asyncio.create_task` and
    never retrieves the result, so the raise is swallowed exactly as `ShowdownException` is on a
    failed login. The agent then never answers and the server plays default moves on the timer,
    which reads as a very weak agent rather than a broken one. A ceiling probe anchored on that
    would report enormous headroom for the wrong reason.
    """
    with pytest.raises(ValueError, match="SINGLES-ONLY"):
        build_player("bc", "gen9vgc2025regi", checkpoint_path="x.pt")
    with pytest.raises(ValueError, match="SINGLES-ONLY"):
        build_player("offrl", "gen9doublesou", checkpoint_path="x.pt")


def test_poke_envs_own_baselines_are_allowed_on_doubles(monkeypatch):
    """`RandomPlayer`, `MaxBasePowerPlayer` and `SimpleHeuristicsPlayer` all branch on DoubleBattle
    in poke-env itself, so the VGC ceiling probe can be run with them before any doubles code of
    ours exists -- which is what makes the G4 decision cheap."""
    import lategame.eval.arena as arena

    for name in ("random", "maxbasepower", "simpleheuristics", "heuristic"):
        monkeypatch.setitem(arena.AGENTS, name, lambda **kw: object())
        build_player(name, "gen9vgc2025regi", team="T")  # must not raise


def test_a_teambuilt_format_is_refused_without_a_team(monkeypatch):
    """Forgetting the team is not an error anywhere else, which is why it is one here.

    Showdown answers a teamless challenge on a teambuilt format with a popup -- "Your team was
    rejected ... This format requires you to use your own team" -- that poke-env logs at WARNING.
    Nothing raises; the player simply never battles, and the arm reads as a run that produced no
    wins rather than one that never started. THREE independent call sites in
    `scripts/curriculum_gate.py` had this defect at once (Gate A's collection, the self-play loop,
    and Gate C's ladder), each found only by watching a cluster job's log. `build_player` is the
    one choke point where it can be caught before a node is claimed.
    """
    import lategame.eval.arena as arena

    monkeypatch.setitem(arena.AGENTS, "heuristic", lambda **kw: object())
    for fmt in ("gen9ou", "gen9vgc2025regi"):
        with pytest.raises(ValueError, match="needs a `team`"):
            build_player("heuristic", fmt)
    # Random Battles must NOT require one -- the server supplies the teams there.
    build_player("heuristic", "gen9randombattle")


def test_the_singles_agents_are_still_fine_on_singles(monkeypatch):
    import lategame.eval.arena as arena

    monkeypatch.setitem(arena.AGENTS, "heuristic", lambda **kw: object())
    build_player("heuristic", "gen9ou", team="T")
    build_player("heuristic", "gen9randombattle")


def test_the_vgc_format_constant_exists_in_the_vendored_simulator():
    """It did not, until 2026-08-11: `gen9vgc2024regh` was a never-exercised placeholder and the
    pinned simulator has no such format. Pinned against the simulator's own format table so the
    constant cannot rot again silently."""
    import re
    from pathlib import Path

    from lategame.config import VGC_FORMAT

    formats = Path("third_party/pokemon-showdown/config/formats.ts")
    if not formats.exists():
        pytest.skip("vendored simulator not present")
    ids = {
        re.sub(r"[^a-z0-9]", "", name.lower())
        for name in re.findall(r'name:\s*"([^"]+)"', formats.read_text())
    }
    assert VGC_FORMAT in ids, f"{VGC_FORMAT!r} is not a format the pinned simulator has"


# --------------------------------------------------------------------------- #
# B6f: the doubles rollout agent, and the dispatch that keeps the singles guard intact.
# --------------------------------------------------------------------------- #
def test_the_doubles_rollout_agent_is_registered_and_doubles_safe():
    from lategame.eval.arena import (
        _CHECKPOINT_AGENTS,
        _DOUBLES_SAFE_AGENTS,
        _LOOP_GUARD_AGENTS,
        _SINGLES_ONLY_AGENTS,
        AGENTS,
    )

    assert "doubles_ppo" in AGENTS
    assert "doubles_ppo" in _CHECKPOINT_AGENTS, "it is loaded from a checkpoint"
    assert "doubles_ppo" in _DOUBLES_SAFE_AGENTS
    # NOT in the singles guard set: its members are handed `LoopGuard`, a 26-wide vector
    # penalizing indices 0-5, which on the doubles layout would penalize PASS and five of the six
    # switches. Two sets so the mistake is a build-time KeyError, not a plausible penalty vector.
    assert "doubles_ppo" not in _LOOP_GUARD_AGENTS
    assert "doubles" not in _LOOP_GUARD_AGENTS
    # Pinned literally: widening this set is how a singles agent silently reaches a doubles format
    # and gets scored as "very weak" while the server plays its moves on the timer.
    assert _SINGLES_ONLY_AGENTS == {"bc", "offrl", "ppo", "search"}


# Every module that builds a learned player from a caller-supplied format. Each must ASK
# `policy_agent`/`rollout_agent` rather than name the singles learner, which `build_player` refuses
# on a doubles format.
#
# This used to live in a docstring, and a docstring cannot fail. `train.selfplay` was omitted from
# it when the helper was introduced and kept five hardcoded names, which is why M4 died on VGC;
# `scripts/rpredict_oppmodel_gate` -- the M2 leg of the format-ceiling probe, explicitly documented
# as a doubles run in `arena._singles_only_agents` -- was still missing after that repair. So the
# list is data now, and `test_no_dispatch_module_hardcodes_the_singles_learner` reads it.
MUST_ASK_FOR_THE_AGENT = (
    "lategame/train/ppo.py",
    "lategame/train/selfplay.py",
    "scripts/ppo_continue_gate.py",
    "scripts/rpredict_oppmodel_gate.py",
    "scripts/curriculum_gate.py",
)


def test_no_dispatch_module_hardcodes_the_singles_learner():
    """Read as TEXT, not imported: two of the four are `scripts/` modules that pull torch and
    poke-env at import time, and the assertion is about source anyway.

    The quoted form deliberately does not match a checkpoint PATH like
    `checkpoints/offrl_gen9randombattle.pt`, which is not a registry name."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for rel in MUST_ASK_FOR_THE_AGENT:
        path = root / rel
        assert path.exists(), f"{rel} moved; update MUST_ASK_FOR_THE_AGENT"
        assert '"offrl"' not in path.read_text(), (
            f"{rel} names the singles learner literally; call `arena.policy_agent(fmt)` instead "
            f"-- `build_player` refuses that name on a doubles format"
        )


def test_agent_dispatch_returns_a_usable_name_for_every_format():
    """The dispatch helpers `MUST_ASK_FOR_THE_AGENT`'s members are required to call."""
    from lategame.config import is_doubles_format
    from lategame.eval.arena import (
        AGENTS,
        _singles_only_agents,
        policy_agent,
        rollout_agent,
    )

    for fmt in ("gen9randombattle", "gen9ou", "gen9vgc2025regi", "gen9doublesou"):
        for name in (policy_agent(fmt), rollout_agent(fmt)):
            assert name in AGENTS, f"{name} is not registered"
            if is_doubles_format(fmt):
                assert name not in _singles_only_agents(), f"{name} would be refused on {fmt}"

    assert policy_agent("gen9ou") == "offrl" and rollout_agent("gen9ou") == "ppo"
    assert policy_agent("gen9vgc2025regi") == "doubles"
    assert rollout_agent("gen9vgc2025regi") == "doubles_ppo"


def test_build_player_forwards_the_turn_cap_only_to_the_doubles_agents(monkeypatch):
    """The ceiling is meaningful only where the loop was measured, and `None` is exact identity,
    so a singles build must not even receive the kwarg."""
    import lategame.eval.arena as arena

    seen: dict[str, dict] = {}

    class Dummy:
        def __init__(self, **kwargs):
            seen.clear()
            seen.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "doubles_ppo", Dummy)
    monkeypatch.setitem(arena.AGENTS, "random", Dummy)

    arena.build_player(
        "doubles_ppo", "gen9vgc2025regi", checkpoint_path="x.pt", max_battle_turns=99, team="T"
    )
    assert seen["max_battle_turns"] == 99 and seen["loop_penalty"] == 0.0

    arena.build_player("random", "gen9vgc2025regi", max_battle_turns=99, team="T")
    assert "max_battle_turns" not in seen


def test_the_cli_defaults_a_team_pool_per_format():
    """`lategame evaluate --format gen9ou` was broken and reported a NUMBER, which is why nobody
    noticed: Showdown rejected every challenge with a team-required popup, no battle finished, and
    `evaluate_built` returned 0.0 for an empty denominator. `build_player`'s refusal turned that
    into an error; this turns it into a working command.

    Random Battles must stay None -- the server supplies the teams there and passing a pool is
    wrong, not merely unnecessary.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "lategame_cli", Path(__file__).resolve().parent.parent / "lategame" / "cli.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    assert cli._default_team_pool("gen9randombattle", None) is None
    assert cli._default_team_pool("gen9ou", None).endswith("teams_gen9ou.packed")
    assert cli._default_team_pool("gen9vgc2025regi", None).endswith("teams_gen9vgc.packed")
    # An explicit choice always wins, on any format.
    assert cli._default_team_pool("gen9ou", "custom.packed") == "custom.packed"
    # Every default the CLI can hand out must actually exist in the tree.
    for fmt in ("gen9ou", "gen9vgc2025regi"):
        assert Path(cli._default_team_pool(fmt, None)).exists()


async def test_evaluate_forwards_a_checkpoint_per_side(monkeypatch):
    """`--p1-checkpoint` / `--p2-checkpoint` are per SIDE because a mirror is a real matchup:
    `--p1 offrl --p2 offrl` with two checkpoints is how one build is scored against another.

    Before these flags the only way to point a learned agent at a downloaded weight was an
    agent-specific environment variable, which cannot express two different ones at once -- so a
    published release was unusable from the CLI for exactly the comparison it exists to enable.
    """
    import lategame.eval.arena as arena

    seen: list[tuple[str, object]] = []

    def fake_build(name, battle_format, **kw):
        seen.append((name, kw.get("checkpoint_path")))
        return object()

    async def fake_eval(p1, p2, n):
        return 0.5

    monkeypatch.setattr(arena, "build_player", fake_build)
    monkeypatch.setattr(arena, "evaluate_built", fake_eval)

    await arena.evaluate(
        "offrl", "offrl", 4, "gen9ou",
        team_pool="lategame/teambuilding/data/teams_gen9ou.packed",
        p1_checkpoint="a.pt", p2_checkpoint="b.pt",
    )
    assert seen == [("offrl", "a.pt"), ("offrl", "b.pt")]

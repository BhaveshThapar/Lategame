"""Offline tests for human-replay ingestion (M6). No server, no network.

A hand-written ~4-turn gen9 spectator log drives the full reconstruction path:
role assignment, move labelling, a dropped first-reveal switch, a kept pivot
switch, a terastallized move, reward signs, done flags, and on-disk schema parity
with the self-play collector.
"""

from __future__ import annotations

import numpy as np

from lategame.data.ingest import _reconstruct_pov, ingest_replays
from lategame.data.replays import _rating_of
from lategame.data.reward import RewardWeights
from lategame.features.encoder import OBS_DIM, OBS_VERSION

# p1 (Alice) wins. p1 line: Thunderbolt; switch to (unseen) Charizard [first reveal
# -> dropped]; switch back to Pikachu [pivot -> kept]; tera + Thunderbolt. p2 (Bob):
# Tackle, Tackle, Vine Whip, then is KO'd before acting on the last turn.
LOG = "\n".join(
    [
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|teamsize|p1|3",
        "|teamsize|p2|3",
        "|gen|9",
        "|tier|[Gen 9] Random Battle",
        "|start",
        "|switch|p1a: Pikachu|Pikachu, L84, M|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L88, M|100/100",
        "|turn|1",
        "|move|p1a: Pikachu|Thunderbolt|p2a: Bulbasaur",
        "|-damage|p2a: Bulbasaur|55/100",
        "|move|p2a: Bulbasaur|Tackle|p1a: Pikachu",
        "|-damage|p1a: Pikachu|80/100",
        "|turn|2",
        "|switch|p1a: Charizard|Charizard, L80, M|100/100",
        "|move|p2a: Bulbasaur|Tackle|p1a: Charizard",
        "|-damage|p1a: Charizard|85/100",
        "|turn|3",
        "|switch|p1a: Pikachu|Pikachu, L84, M|80/100",
        "|move|p2a: Bulbasaur|Vine Whip|p1a: Pikachu",
        "|-damage|p1a: Pikachu|55/100",
        "|turn|4",
        "|-terastallize|p1a: Pikachu|Electric",
        "|move|p1a: Pikachu|Thunderbolt|p2a: Bulbasaur",
        "|-damage|p2a: Bulbasaur|0 fnt",
        "|faint|p2a: Bulbasaur",
        "|win|Alice",
    ]
)

_WEIGHTS = RewardWeights()


def test_role_mapping_splits_teams_by_pov() -> None:
    battle, _, _, _ = _reconstruct_pov(LOG.split("\n"), "Alice", "t-Alice", 9, _WEIGHTS)
    assert battle.player_role == "p1"
    assert {m.base_species for m in battle.team.values()} == {"pikachu", "charizard"}
    assert "bulbasaur" in {m.base_species for m in battle.opponent_team.values()}
    assert battle.finished and battle.won


def test_p1_decisions_labelled_with_pivot_and_tera() -> None:
    _, records, _, dropped = _reconstruct_pov(LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    actions = [a for _, a, _ in records]
    # Thunderbolt (move slot 6), pivot switch back to Pikachu (switch slot 0),
    # tera Thunderbolt (move slot 6 + tera offset 16 = 22). The turn-2 first-reveal
    # switch to Charizard is undecodable -> dropped, not recorded.
    assert actions == [6, 0, 22]
    assert dropped == 1


def test_p2_move_indices_track_reveal_order() -> None:
    _, records, _, _ = _reconstruct_pov(LOG.split("\n"), "Bob", "t", 9, _WEIGHTS)
    # Tackle (first move -> slot 6), Tackle again (still slot 6), Vine Whip (second
    # revealed move -> slot 7). No 4th-turn decision: Bulbasaur faints first.
    assert [a for _, a, _ in records] == [6, 6, 7]


def test_masks_keep_taken_action_legal() -> None:
    _, records, _, _ = _reconstruct_pov(LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    for _, action, mask in records:
        assert mask.shape == (26,) and mask.dtype == bool
        assert mask[action]


def test_ingest_dataset_shapes_rewards_and_dones() -> None:
    replay = {"id": "t", "format": "gen9randombattle", "players": ["Alice", "Bob"], "log": LOG}
    rl, bc, stats = ingest_replays([replay], weights=_WEIGHTS)

    assert stats.episodes == 2 and stats.turns == 6 and stats.dropped_turns == 1
    assert rl.obs.shape == (6, OBS_DIM) and rl.obs.dtype == np.float32
    assert np.isfinite(rl.obs).all()
    assert rl.action.shape == (6,) and rl.mask.shape == (6, 26)
    # Episode boundaries: combined order is [p1 x3, p2 x3]; done on the last of each.
    assert rl.done.tolist() == [False, False, True, False, False, True]
    # Terminal reward carries the +/- victory jump: p1 won, p2 lost.
    assert rl.reward[2] > 0.0
    assert rl.reward[5] < 0.0
    # BC shard keeps only the winner's (p1) turns.
    assert bc is not None and bc.obs.shape == (3, OBS_DIM)


def test_schema_parity_loads_via_rl_dataset(tmp_path) -> None:
    from lategame.data.collect import save_rl
    from lategame.data.rl_dataset import RLDataset

    replay = {"id": "t", "format": "gen9randombattle", "players": ["Alice", "Bob"], "log": LOG}
    rl, _, _ = ingest_replays([replay], weights=_WEIGHTS)
    path = tmp_path / "shard.npz"
    save_rl(rl, path)

    loaded = np.load(path, allow_pickle=False)
    assert str(loaded["obs_version"].item()) == OBS_VERSION
    assert int(loaded["obs_dim"].item()) == OBS_DIM

    ds = RLDataset(path)
    assert len(ds) == 6  # MC returns computed without error


def test_rating_filter_helper() -> None:
    assert _rating_of({"rating": 1500}) == 1500
    assert _rating_of({"rating": None}) is None
    assert _rating_of({}) is None
    assert _rating_of("not a dict") is None


# --- OU (team-choice) reconstruction: team preview fills all 12 species from turn 0 ---

# p1 (Alice) wins. Order: lead Gholdengo (Make It Rain); turn 2 switch to the
# previewed-but-unseen Dragonite (a *first-reveal* switch -- dropped in RB, but kept
# here because preview seeded it); Earthquake x2. p2 (Bob) leads Garganacl, KO'd turn 4.
OU_LOG = "\n".join(
    [
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|teamsize|p1|6",
        "|teamsize|p2|6",
        "|gen|9",
        "|tier|[Gen 9] OU",
        "|clearpoke",
        "|poke|p1|Gholdengo|",
        "|poke|p1|Dragonite, F|",
        "|poke|p1|Kingambit, M|",
        "|poke|p1|Landorus-Therian|",
        "|poke|p1|Great Tusk|",
        "|poke|p1|Zamazenta|",
        "|poke|p2|Garganacl, M|",
        "|poke|p2|Dragapult, M|",
        "|poke|p2|Kingambit, F|",
        "|poke|p2|Slowking-Galar, M|",
        "|poke|p2|Corviknight, F|",
        "|poke|p2|Ogerpon-Wellspring, F|",
        "|teampreview",
        "|start",
        "|switch|p1a: Gholdengo|Gholdengo|100/100",
        "|switch|p2a: Garganacl|Garganacl, M|100/100",
        "|turn|1",
        "|move|p1a: Gholdengo|Make It Rain|p2a: Garganacl",
        "|-damage|p2a: Garganacl|70/100",
        "|move|p2a: Garganacl|Salt Cure|p1a: Gholdengo",
        "|-damage|p1a: Gholdengo|80/100",
        "|turn|2",
        "|switch|p1a: Dragonite|Dragonite, F|100/100",
        "|move|p2a: Garganacl|Recover|p2a: Garganacl",
        "|-heal|p2a: Garganacl|100/100",
        "|turn|3",
        "|move|p1a: Dragonite|Earthquake|p2a: Garganacl",
        "|-damage|p2a: Garganacl|40/100",
        "|move|p2a: Garganacl|Salt Cure|p1a: Dragonite",
        "|-damage|p1a: Dragonite|85/100",
        "|turn|4",
        "|move|p1a: Dragonite|Earthquake|p2a: Garganacl",
        "|-damage|p2a: Garganacl|0 fnt",
        "|faint|p2a: Garganacl",
        "|win|Alice",
    ]
)


def test_ou_preview_gives_full_rosters_in_encoded_obs() -> None:
    from lategame.features.encoder import OBS_LAYOUT, embed_battle

    battle, records, _, _ = _reconstruct_pov(OU_LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    # Own six are seeded into battle.team from preview (poke-env drops the ego preview).
    assert {m.base_species for m in battle.team.values()} == {
        "gholdengo", "dragonite", "kingambit", "landorus", "greattusk", "zamazenta"
    }
    # The opponent's six live in teampreview_opponent_team; the encoder merges them so the
    # encoded POV shows all twelve species -- identical to what a live player sees at eval.
    assert {m.base_species for m in battle.teampreview_opponent_team} == {
        "garganacl", "dragapult", "kingambit", "slowking", "corviknight", "ogerpon"
    }
    pdim = OBS_LAYOUT.pokemon_dim
    obs = records[0][0]  # first p1 decision obs (present flag is index 0 of each mon block)
    ego_present = sum(1 for i in range(6) if obs[i * pdim] > 0.5)
    opp_present = sum(1 for i in range(6, 12) if obs[i * pdim] > 0.5)
    assert ego_present == 6 and opp_present == 6

    obs = embed_battle(battle)  # end-of-battle POV also carries the full opponent roster
    assert sum(1 for i in range(6, 12) if obs[i * pdim] > 0.5) == 6


def test_ou_first_reveal_switch_is_labelled_not_dropped() -> None:
    _, records, _, dropped = _reconstruct_pov(OU_LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    # Make It Rain (slot 6), switch to previewed Dragonite (team slot 1), Earthquake x2.
    # The turn-2 switch would be a dropped first-reveal in RB; preview makes it labelable.
    assert [a for _, a, _ in records] == [6, 1, 6, 6]
    assert dropped == 0


def test_ou_masks_keep_taken_action_legal() -> None:
    _, records, _, _ = _reconstruct_pov(OU_LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    for _, action, mask in records:
        assert mask.shape == (26,) and mask.dtype == bool
        assert mask[action]


# A nicknamed mon: preview shows the species ("Dragonite"), the switch shows the
# nickname ("Draco"). The preview entry must re-key onto the switch, not duplicate.
NICK_LOG = "\n".join(
    [
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|teamsize|p1|2",
        "|teamsize|p2|1",
        "|gen|9",
        "|tier|[Gen 9] OU",
        "|clearpoke",
        "|poke|p1|Gholdengo|",
        "|poke|p1|Dragonite, F|",
        "|poke|p2|Garganacl, M|",
        "|teampreview",
        "|start",
        "|switch|p1a: Gholdengo|Gholdengo|100/100",
        "|switch|p2a: Garganacl|Garganacl, M|100/100",
        "|turn|1",
        "|switch|p1a: Draco|Dragonite, F|100/100",
        "|move|p2a: Garganacl|Salt Cure|p1a: Draco",
        "|-damage|p1a: Draco|85/100",
        "|turn|2",
        "|move|p1a: Draco|Earthquake|p2a: Garganacl",
        "|-damage|p2a: Garganacl|0 fnt",
        "|faint|p2a: Garganacl",
        "|win|Alice",
    ]
)


def test_ou_preview_switch_reconciles_nickname_without_duplicate() -> None:
    battle, _, _, _ = _reconstruct_pov(NICK_LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    # Preview seeded 2 mons; the nicknamed switch re-keys the Dragonite entry rather
    # than adding a 3rd. Species appears exactly once.
    assert len(battle.team) == 2
    assert sum(1 for m in battle.team.values() if m.base_species == "dragonite") == 1


# --- Two-pass own-team completion: fill own item/ability/moves to the live-request POV ---

# p1 (Alice) leads Kingambit (a *multi*-ability species, so poke-env does NOT auto-assign its
# ability). Turn 1 it uses Kowtow Cleave (its only revealed move so far). Turn 2 reveals the rest
# of its kit: ability Supreme Overlord (|-ability|), then its Air Balloon is *knocked off*
# (|-enditem| -> poke-env resets .item to None), then it uses Sucker Punch. So at the turn-1
# decision the v1 POV knows almost nothing about the mon, while the live |request| would have
# shown item+ability+full moveset from turn 0.
COMPLETE_LOG = "\n".join(
    [
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|teamsize|p1|2",
        "|teamsize|p2|1",
        "|gen|9",
        "|tier|[Gen 9] OU",
        "|clearpoke",
        "|poke|p1|Kingambit, M|",
        "|poke|p1|Gholdengo|",
        "|poke|p2|Garganacl, M|",
        "|teampreview",
        "|start",
        "|switch|p1a: Kingambit|Kingambit, M|100/100",
        "|switch|p2a: Garganacl|Garganacl, M|100/100",
        "|turn|1",
        "|move|p1a: Kingambit|Kowtow Cleave|p2a: Garganacl",
        "|-damage|p2a: Garganacl|60/100",
        "|move|p2a: Garganacl|Salt Cure|p1a: Kingambit",
        "|-damage|p1a: Kingambit|80/100",
        "|turn|2",
        "|-ability|p1a: Kingambit|Supreme Overlord",
        "|move|p2a: Garganacl|Knock Off|p1a: Kingambit",
        "|-damage|p1a: Kingambit|70/100",
        "|-enditem|p1a: Kingambit|Air Balloon|[from] move: Knock Off|[of] p2a: Garganacl",
        "|move|p1a: Kingambit|Sucker Punch|p2a: Garganacl",
        "|-damage|p2a: Garganacl|0 fnt",
        "|faint|p2a: Garganacl",
        "|win|Alice",
    ]
)


def _own_active_detail(obs: np.ndarray) -> tuple[bool, bool, int]:
    """(item-known, ability-known, active-move-count) read from the encoder ID channels."""
    from lategame.features.encoder import OBS_LAYOUT

    pdim = OBS_LAYOUT.pokemon_dim
    ms, md = OBS_LAYOUT.moves_start, OBS_LAYOUT.move_dim
    for i in range(6):
        if obs[i * pdim + 1] > 0.5:  # own active block (active flag = channel 1)
            item = bool(obs[i * pdim + pdim - 2] > 0.5)
            ability = bool(obs[i * pdim + pdim - 1] > 0.5)
            moves = sum(1 for j in range(4) if obs[ms + j * md + md - 1] > 0.5)
            return item, ability, moves
    raise AssertionError("no own active mon block found in obs")


def test_prescan_recovers_full_kit_including_consumed_item() -> None:
    from lategame.data.ingest import _prescan_kits, _team_key

    kits = _prescan_kits(COMPLETE_LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    kit = kits[_team_key("p1a: Kingambit")]
    # Air Balloon is recovered from the raw -enditem line even though end-of-game .item is None.
    assert kit.item == "airballoon"
    assert kit.ability == "supremeoverlord"
    assert {"kowtowcleave", "suckerpunch"} <= kit.moves


def test_two_pass_completes_active_kit_at_early_decision() -> None:
    from lategame.data.ingest import _prescan_kits
    from lategame.features import vocab
    from lategame.features.encoder import OBS_LAYOUT

    lines = COMPLETE_LOG.split("\n")
    _, off, _, _ = _reconstruct_pov(lines, "Alice", "t", 9, _WEIGHTS, kits=None)
    kits = _prescan_kits(lines, "Alice", "t", 9, _WEIGHTS)
    _, on, _, _ = _reconstruct_pov(lines, "Alice", "t", 9, _WEIGHTS, kits=kits)

    # Decision 0 is the turn-1 move, before the mon revealed its item/ability/2nd move in-line.
    assert _own_active_detail(off[0][0]) == (False, False, 1)  # v1 progressive-reveal POV
    assert _own_active_detail(on[0][0]) == (True, True, 2)  # completed to the live POV

    # The completed ID channels carry the real vocab ids the gate reads at those exact offsets.
    obs, pdim = on[0][0], OBS_LAYOUT.pokemon_dim
    active = next(i for i in range(6) if obs[i * pdim + 1] > 0.5)
    assert obs[active * pdim + pdim - 2] == vocab.load_vocab().index("items", "airballoon")
    assert obs[active * pdim + pdim - 1] == vocab.load_vocab().index("abilities", "supremeoverlord")


def test_two_pass_keeps_consumed_item_unknown_after_knockoff() -> None:
    from lategame.data.ingest import _prescan_kits

    lines = COMPLETE_LOG.split("\n")
    kits = _prescan_kits(lines, "Alice", "t", 9, _WEIGHTS)
    _, on, _, _ = _reconstruct_pov(lines, "Alice", "t", 9, _WEIGHTS, kits=kits)
    # Decision 1 is the turn-2 move, AFTER Air Balloon was knocked off: backfill must not
    # re-reveal a consumed item -- it reads None, exactly as the live POV shows post-knock.
    item_known, _, _ = _own_active_detail(on[1][0])
    assert item_known is False


# --- Usage-prior imputation (Build 4): fill the still-unrevealed kit to eval-full detail ---

def _fake_prior():
    """Single-item distributions per species -> deterministic draws for assertions.

    Kingambit's prior item is deliberately NOT its true Air Balloon, to prove revealed
    truth always beats the prior. Move lists carry exactly the mass needed to fill to 4.
    """
    from lategame.data.usage_prior import SpeciesUsage, UsagePrior

    return UsagePrior(
        version="test",
        vocab_version="test",
        source="",
        month="",
        cutoff=0,
        battle_format="gen9ou",
        species={
            "kingambit": SpeciesUsage(
                moves=(
                    ("kowtowcleave", 0.3),
                    ("suckerpunch", 0.3),
                    ("swordsdance", 0.2),
                    ("ironhead", 0.2),
                ),
                items=(("leftovers", 1.0),),
                abilities=(("defiant", 1.0),),
            ),
            "gholdengo": SpeciesUsage(
                moves=(
                    ("makeitrain", 0.3),
                    ("shadowball", 0.3),
                    ("nastyplot", 0.2),
                    ("recover", 0.2),
                ),
                items=(("airballoon", 1.0),),
                abilities=(("goodasgold", 1.0),),
            ),
        },
    )


def test_impute_fills_unrevealed_kit_to_full() -> None:
    from lategame.data.ingest import _impute_kits, _prescan_kits, _team_key
    from lategame.features import vocab
    from lategame.features.encoder import OBS_LAYOUT

    lines = COMPLETE_LOG.split("\n")
    kits = _prescan_kits(lines, "Alice", "t", 9, _WEIGHTS)
    imputed, missing = _impute_kits(kits, "t", _fake_prior())
    assert imputed == 2 and missing == 0
    battle, on, _, _ = _reconstruct_pov(lines, "Alice", "t", 9, _WEIGHTS, kits=kits)

    # Decision 0 (turn 1) now reads eval-full: item + ability + all 4 moves.
    assert _own_active_detail(on[0][0]) == (True, True, 4)
    # Revealed truth beats the prior: the item channel carries Air Balloon, not Leftovers.
    obs, pdim = on[0][0], OBS_LAYOUT.pokemon_dim
    active = next(i for i in range(6) if obs[i * pdim + 1] > 0.5)
    assert obs[active * pdim + pdim - 2] == vocab.load_vocab().index("items", "airballoon")
    assert obs[active * pdim + pdim - 1] == vocab.load_vocab().index(
        "abilities", "supremeoverlord"
    )

    # The never-switched-in bench Gholdengo (preview-only, zero reveals) reaches a full kit
    # too: 4 imputed moves + the prior item (its single-option ability was already assigned).
    bench = battle.team[_team_key("p1a: Gholdengo")]
    assert len(bench.moves) == 4 and bench.item == "airballoon" and bench.ability == "goodasgold"


def test_impute_deterministic_across_runs_and_timesteps() -> None:
    from lategame.data.ingest import _impute_kits, _prescan_kits
    from lategame.data.usage_prior import load_usage_prior

    prior = load_usage_prior("gen9ou")  # the committed artifact -- also exercises its ids
    assert prior is not None
    lines = COMPLETE_LOG.split("\n")

    runs = []
    for _ in range(2):
        kits = _prescan_kits(lines, "t-Alice", "t-Alice", 9, _WEIGHTS)
        _impute_kits(kits, "t-Alice", prior)
        _, records, _, _ = _reconstruct_pov(lines, "Alice", "t-Alice", 9, _WEIGHTS, kits=kits)
        runs.append(records)
    assert len(runs[0]) == len(runs[1]) > 0
    for (obs_a, act_a, mask_a), (obs_b, act_b, mask_b) in zip(runs[0], runs[1], strict=True):
        assert act_a == act_b and np.array_equal(obs_a, obs_b) and np.array_equal(mask_a, mask_b)


def test_impute_never_overwrites_consumed_item() -> None:
    from lategame.data.ingest import _impute_kits, _prescan_kits

    lines = COMPLETE_LOG.split("\n")
    kits = _prescan_kits(lines, "Alice", "t", 9, _WEIGHTS)
    _impute_kits(kits, "t", _fake_prior())
    _, on, _, _ = _reconstruct_pov(lines, "Alice", "t", 9, _WEIGHTS, kits=kits)
    # Post-Knock-Off decision: the consumed item stays None (live POV parity), never imputed.
    item_known, _, _ = _own_active_detail(on[1][0])
    assert item_known is False


def test_impute_missing_species_counts_and_falls_back() -> None:
    from lategame.data.ingest import _impute_kits, _prescan_kits, _team_key
    from lategame.data.usage_prior import UsagePrior

    prior = _fake_prior()
    gholdengo_only = UsagePrior(
        version="test",
        vocab_version="test",
        source="",
        month="",
        cutoff=0,
        battle_format="gen9ou",
        species={"gholdengo": prior.species["gholdengo"]},
    )
    kits = _prescan_kits(COMPLETE_LOG.split("\n"), "Alice", "t", 9, _WEIGHTS)
    imputed, missing = _impute_kits(kits, "t", gholdengo_only)
    assert imputed == 1 and missing == 1
    # The missing-species mon keeps its log-only two-pass kit, untouched.
    kit = kits[_team_key("p1a: Kingambit")]
    assert kit.moves == {"kowtowcleave", "suckerpunch"}
    assert kit.item == "airballoon" and kit.ability == "supremeoverlord"


def test_ingest_replays_impute_flag() -> None:
    replay = {"id": "t", "format": "gen9ou", "players": ["Alice", "Bob"], "log": COMPLETE_LOG}

    rl_on, _, stats_on = ingest_replays([replay], weights=_WEIGHTS, battle_format="gen9ou")
    assert stats_on.imputed_mons > 0 and stats_on.usage_missing_mons == 0
    # p1's decision 0 (turn-1 move) reaches eval-full detail via the committed artifact.
    assert _own_active_detail(rl_on.obs[0]) == (True, True, 4)

    rl_off, _, stats_off = ingest_replays(
        [replay], weights=_WEIGHTS, battle_format="gen9ou", impute_usage=False
    )
    assert stats_off.imputed_mons == 0
    # OFF reproduces the Build-3 log-only ceiling at the same decision.
    assert _own_active_detail(rl_off.obs[0]) == (True, True, 2)

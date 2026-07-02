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

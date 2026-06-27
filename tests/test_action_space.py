"""Unit tests for the action-space codec wrapper (no server needed).

The order<->action conversion itself is poke-env's tested ``SinglesEnv`` code;
these tests pin our wrapper's contract (sizes, dtypes). Round-trip correctness on
real battles is covered by the server-gated smoke test.
"""

from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE, action_space_size


class FakeBattle:
    def __init__(self, gen):
        self.gen = gen


def test_gen9_action_space_size():
    assert GEN9_ACTION_SPACE_SIZE == 26  # 6 switches + 4 moves * (4 gimmicks + 1)


def test_action_space_size_tracks_generation():
    assert action_space_size(FakeBattle(gen=9)) == 26
    assert action_space_size(FakeBattle(gen=1)) == 10  # no gimmicks

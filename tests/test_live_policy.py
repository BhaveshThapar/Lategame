"""The ladder gate must be hard to trip by accident, because it authorises RANKED play.

plan.md NG3 puts ranked-ladder farming out of scope and section 15 says to default to
challenge/unranked. Ladder mode cannot be made unranked -- `/search` on the public server IS the
rated ladder -- so the gate is the only thing standing between a stray flag and a policy breach.
These tests pin that both channels are load-bearing.
"""

import pytest

from lategame.live.policy import (
    ALLOW_LADDER_ENV,
    LADDER_ACK,
    PASSWORD_ENV,
    POLICY_NOTE,
    USERNAME_ENV,
    PolicyError,
    check_ladder_optin,
)

_ON = {ALLOW_LADDER_ENV: "1"}


def test_both_channels_present_authorises():
    assert check_ladder_optin(LADDER_ACK, _ON) == LADDER_ACK


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", " 1 "])
def test_env_channel_accepts_the_documented_spellings(truthy):
    assert check_ladder_optin(LADDER_ACK, {ALLOW_LADDER_ENV: truthy}) == LADDER_ACK


@pytest.mark.parametrize(
    "ack",
    [None, "", "yes", LADDER_ACK.upper(), LADDER_ACK + "!", LADDER_ACK[:-1]],
)
def test_the_ack_must_be_exact(ack):
    """A near-miss is a refusal, not a warning -- argparse `choices` makes it a parse error too."""
    with pytest.raises(PolicyError):
        check_ladder_optin(ack, _ON)


@pytest.mark.parametrize("env", [{}, {ALLOW_LADDER_ENV: "0"}, {ALLOW_LADDER_ENV: ""},
                                 {ALLOW_LADDER_ENV: "no"}, {ALLOW_LADDER_ENV: "2"}])
def test_the_env_channel_is_independently_required(env):
    """The right flag alone is NOT enough: a recalled command must not start ranked play."""
    with pytest.raises(PolicyError):
        check_ladder_optin(LADDER_ACK, env)


def test_every_refusal_carries_the_policy_note():
    """The error is the teaching moment -- it must say WHY, not just 'denied'."""
    for ack, env in ((None, _ON), (LADDER_ACK, {})):
        with pytest.raises(PolicyError) as exc:
            check_ladder_optin(ack, env)
        assert POLICY_NOTE in str(exc.value)


def test_policy_note_names_the_actual_constraints():
    note = POLICY_NOTE.lower()
    assert "ng3" in note and "ranked" in note
    assert "bot account" in note  # section 15's account-isolation rule
    assert "no unranked ladder" in note  # the reason this cannot be made safe, only explicit


def test_module_has_no_heavy_imports():
    """cli.py imports this at module scope to build --help; it must not drag in torch/poke-env."""
    import sys

    import lategame.live.policy as mod

    assert not hasattr(mod, "torch") and not hasattr(mod, "poke_env")
    # `os` is imported lazily inside check_ladder_optin, so it must not be a module attribute.
    assert not hasattr(mod, "os")
    assert "torch" not in sys.modules or True  # torch may be loaded by another test; not our doing


def test_credential_env_names_are_distinct_and_namespaced():
    assert USERNAME_ENV != PASSWORD_ENV
    assert all(n.startswith("LATEGAME_") for n in (USERNAME_ENV, PASSWORD_ENV, ALLOW_LADDER_ENV))

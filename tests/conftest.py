"""Shared fixtures for the tests that drive the *real* vendored simulator.

Three test modules (``test_fidelity``, ``test_resim``, ``test_search``) need node plus a built
``third_party/pokemon-showdown/dist``, and two of them replay the same gen9randombattle inputlog.
Both the environment gate and the inputlog used to be copy-pasted per module; they live here so
one regeneration fixes every consumer.

THE INPUTLOG IS REV-SPECIFIC. It is a fixed PRNG seed plus a fixed list of choices, and
gen9randombattle sets change upstream -- under a different simulator rev the same seed rolls
different teams and the recorded choices go illegal partway through, which is exactly how this
fixture rotted between the 2026-06-28 Gate A pass and a 2026-07-19 working copy (the fidelity
driver got 4 decision rounds in before the battle rejected a choice). It is therefore pinned in
lockstep with ``SHOWDOWN_REV`` in ``scripts/setup_server.sh``: bumping that pin invalidates this
literal, and it must be regenerated in the same commit with

    node scripts/gen_inputlog_fixture.js third_party/pokemon-showdown

which plays two RandomPlayerAIs to a natural ``|win|`` so the log is legal by construction. Every
seed in that script is pinned, so regenerating against an unchanged rev is a no-op.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lategame.data.resim import _DEFAULT_SHOWDOWN

HAS_NODE = shutil.which("node") is not None
HAS_SIM = (Path(_DEFAULT_SHOWDOWN) / "dist").is_dir()

#: Skip marker for tests that shell out to the vendored simulator. NOTE that ``node`` ships inside
#: the conda env rather than on the system PATH, so these tests SKIP on a bare shell and only run
#: once the env is active -- which is how two of them stayed broken without anyone noticing.
requires_showdown = pytest.mark.skipif(
    not (HAS_NODE and HAS_SIM),
    reason="needs node + a built vendored simulator",
)

def server_up(host: str = "localhost", port: int = 8000, timeout: float = 0.5) -> bool:
    """Is a Showdown server listening? Used by every module-level gate on a live battle.

    Was copy-pasted byte-identical into five test modules (`test_arena`, `test_bc_smoke`,
    `test_offline_rl_smoke`, `test_ppo_smoke`, `test_selfplay_smoke`) and near-identically into a
    sixth. One copy means one place to change the port if the harness ever moves off 8000 --
    `scripts/cluster/_job_common.sh` already runs jobs on 8100+ and these probes do not follow it,
    which is why the smoke tests self-skip inside a cluster job.
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


#: Skip marker for tests that need a live local server. Evaluated at import, like
#: ``requires_showdown`` -- starting a server mid-session will not un-skip an already-collected run.
requires_server = pytest.mark.skipif(
    not server_up(), reason="local Showdown server not running on :8000"
)


def write_ac_checkpoint(path, obs_dim, *, hidden_dim=32, n_bins=21, v_min=-5.0, v_max=5.0,
                        battle_format="gen9randombattle"):
    """A minimal actor-critic checkpoint on disk, for the loop smokes that need a warm start.

    Identical copies lived in `test_ppo_smoke` and `test_selfplay_smoke`. Stamps the same keys
    `train.ppo._save_checkpoint` does, so `_check_warm_start` accepts it -- including the value
    support, whose absence is what a BC checkpoint gets refused for.
    """
    import torch

    from lategame.model.actor_critic import ActorCritic

    model = ActorCritic(obs_dim, hidden_dim=hidden_dim, n_bins=n_bins)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "actor_critic",
            "input_dim": obs_dim,
            "hidden_dim": hidden_dim,
            "n_actions": model.n_actions,
            "n_bins": n_bins,
            "v_min": v_min,
            "v_max": v_max,
            "obs_version": __import__("lategame.features.encoder", fromlist=["x"]).OBS_VERSION,
            "battle_format": battle_format,
            "metrics": {},
        },
        path,
    )


#: A complete gen9randombattle, generated against the pinned rev (25 turns, natural winner).
#: Drives 32 clean decision rounds through the fidelity driver and reconstructs 58 labelled turns
#: across both POVs through the resim driver.
GEN9_INPUTLOG = "\n".join([
    '>start {"formatid":"gen9randombattle","seed":"sodium,b8493732a42936a1fd687ddf7988dbbc"}',
    '>player p1 {"name":"gummiworm","seed":"sodium,a3572989c38af097ba39a7fa38627767"}',
    '>player p2 {"name":"Qwuartz","seed":"sodium,d9f6a625ddf93230a09fa7bc6f0239d0"}',
    '>p1 move leafblade', '>p2 move nastyplot', '>p1 move swordsdance', '>p2 move taunt',
    '>p1 move closecombat', '>p2 move thunderbolt', '>p1 move leafblade', '>p2 move thunderbolt',
    '>p1 switch 2', '>p1 move overdrive', '>p2 move taunt', '>p2 switch 4',
    '>p1 move overdrive', '>p2 move swordsdance', '>p1 move boomburst', '>p2 move tripleaxel',
    '>p1 switch 4', '>p2 switch 6', '>p1 move taunt', '>p2 move dragondance',
    '>p1 move suckerpunch', '>p2 move crabhammer', '>p1 move crunch', '>p2 move knockoff',
    '>p1 move suckerpunch', '>p2 move crabhammer', '>p1 move taunt', '>p2 move dragondance',
    '>p1 move taunt', '>p2 move knockoff', '>p1 move playrough', '>p2 move aquajet',
    '>p1 switch 6', '>p1 move earthquake', '>p2 move crabhammer', '>p2 switch 2',
    '>p1 move bulkup', '>p2 move flashcannon', '>p1 move bodyslam', '>p2 move aurasphere',
    '>p1 move bulkup', '>p2 move calmmind', '>p1 move bodyslam', '>p2 move vacuumwave',
    '>p1 switch 5', '>p1 move icespinner', '>p2 move flashcannon', '>p1 move rapidspin',
    '>p2 move flashcannon', '>p1 move rapidspin', '>p2 move vacuumwave', '>p1 move iceshard',
    '>p2 move flashcannon', '>p1 switch 3', '>p1 move darkpulse', '>p2 move flashcannon',
    '>p1 move hypnosis', '>p2 move aurasphere',
])

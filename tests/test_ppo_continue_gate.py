"""Unit tests for the Lever-10 / Build-16 PPO continuation gate's pure config wiring.

The full seed sweep + confirmatory ladder are server-gated; here we only prove the OU
flags (``--team-pool`` / ``--loop-penalty`` / ``--ckpt-prefix``) thread into the ``PPOConfig``
the loop actually runs, so an OU run rolls out with teams + the LoopGuard and writes to a
non-clobbering checkpoint dir.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")  # the gate imports lategame.train.ppo, which imports torch

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ppo_continue_gate.py"
_spec = importlib.util.spec_from_file_location("ppo_continue_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_ppo_config_threads_ou_flags():
    cfg = gate._ppo_config(
        "checkpoints/offrl_gen9ou_v7_s0.pt", seed=1, iters=3, games_per_opp=4, eval_n=10,
        device="cpu", fmt="gen9ou", team_pool="lategame/teambuilding/data/teams_gen9ou.packed",
        loop_penalty=4.0, ckpt_prefix="ppo_ou_et_prior",
    )
    assert cfg.battle_format == "gen9ou"
    assert cfg.team_pool == "lategame/teambuilding/data/teams_gen9ou.packed"
    assert cfg.loop_penalty == 4.0
    assert cfg.out_dir == "checkpoints/ppo_ou_et_prior_s1"  # prefix + seed, no clobber of RB runs


def test_ppo_config_defaults_stay_rb_safe():
    # The RB-compat path: no team pool, LoopGuard off, the on-record RB checkpoint prefix.
    cfg = gate._ppo_config(
        "init.pt", seed=0, iters=1, games_per_opp=1, eval_n=1, device="cpu",
        fmt="gen9randombattle", team_pool=None, loop_penalty=0.0, ckpt_prefix="ppo_scale_et_prior",
    )
    assert cfg.team_pool is None
    assert cfg.loop_penalty == 0.0
    assert cfg.out_dir == "checkpoints/ppo_scale_et_prior_s0"


def test_ppo_config_threads_build19_schedule():
    cfg = gate._ppo_config(
        "init.pt", seed=0, iters=50, games_per_opp=16, eval_n=100, device="cpu",
        fmt="gen9ou", team_pool=None, loop_penalty=4.0, ckpt_prefix="ppo_ou_sched",
        ent_coef=0.01, ent_coef_final=0.0, lr=2.5e-4, lr_final=5e-5,
    )
    assert cfg.ent_coef == 0.01
    assert cfg.ent_coef_final == 0.0
    assert cfg.lr == 2.5e-4
    assert cfg.lr_final == 5e-5


def test_ppo_config_without_schedule_flags_is_build18_constant():
    # Omitting the *-final flags must reproduce Build 16-18 exactly: no schedule at all.
    cfg = gate._ppo_config(
        "init.pt", seed=0, iters=50, games_per_opp=16, eval_n=100, device="cpu",
        fmt="gen9ou", team_pool=None, loop_penalty=4.0, ckpt_prefix="ppo_ou_sched",
    )
    assert cfg.ent_coef_final is None
    assert cfg.lr_final is None
    assert cfg.ent_coef == 0.01  # the Build 16-18 constants
    assert cfg.lr == 2.5e-4

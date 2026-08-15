"""The factored PPO surrogate (B6f) -- no torch server, no Showdown.

Everything that determines whether a VGC PPO run is CORRECT rather than merely running is pinned
here, because each failure below trains happily and produces a checkpoint:

* an act/train ratio mismatch biases every gradient toward actions that were never played;
* a plain `.mean()` over forced turns divides the policy loss, the entropy bonus and the
  approx-KL alike, so the actor barely moves and the trust region never binds while
  `scripts/ppo_telemetry` certifies the run healthy;
* a row whose recorded action was not executed, or is illegal under its own restricted mask,
  contributes a ~1e9 cross-entropy that swamps the batch -- the same defect that read 719,296 on
  the AWR actor loss.

The singles path is asserted UNCHANGED in the same file: the decision-row denominator must not
reach it, or every published OU number is silently rescaled.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lategame.data.rollout import RolloutBuffer, decision_stats  # noqa: E402
from lategame.features.doubles_action_space import (  # noqa: E402
    GEN9_DOUBLES_ACTION_SPACE_SIZE,
    GEN9_DOUBLES_SLOT_ACTIONS,
)
from lategame.features.doubles_encoder import OBS_DIM_DOUBLES, OBS_VERSION_DOUBLES  # noqa: E402
from lategame.model.actor_critic import ActorCritic, value_support  # noqa: E402
from lategame.model.policy import factored_logits, sample_factored_action  # noqa: E402
from lategame.train.ppo import (  # noqa: E402
    PPOConfig,
    _save_checkpoint,
    _surrogate_weights,
    ppo_update,
)

A = GEN9_DOUBLES_SLOT_ACTIONS
N_BINS = 21


def _model(seed: int = 0):
    torch.manual_seed(seed)
    return ActorCritic(
        OBS_DIM_DOUBLES, hidden_dim=16, n_actions=GEN9_DOUBLES_ACTION_SPACE_SIZE, n_bins=N_BINS
    )


def _doubles_buffer(model, n: int = 24, *, decision_rows: int | None = None, seed: int = 0):
    """An ON-POLICY buffer: `old_log_prob` comes from the model these weights belong to.

    That is the whole point -- a buffer whose log-probs were invented would make the ratio-is-one
    test vacuous, which is exactly how the act/train split stayed invisible until now.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    decision_rows = n if decision_rows is None else decision_rows
    mask = torch.zeros(n, 2, A, dtype=torch.bool)
    for i in range(n):
        if i < decision_rows:
            for s in (0, 1):
                idx = rng.choice(A, size=int(rng.integers(2, 9)), replace=False)
                mask[i, s, idx] = True
        else:  # a forced turn: exactly one legal action per slot
            mask[i, 0, int(rng.integers(0, A))] = True
            mask[i, 1, int(rng.integers(0, A))] = True

    obs = torch.randn(n, OBS_DIM_DOUBLES)
    model.eval()  # as the acting agent does -- dropout on one side only IS an act/train mismatch
    with torch.no_grad():
        flat, _ = model(obs)
        action, log_prob, _ = sample_factored_action(
            factored_logits(flat, 2), mask, sample=True
        )
    done = torch.zeros(n, dtype=torch.bool)
    done[n // 2 - 1] = done[n - 1] = True
    return RolloutBuffer(
        obs=obs,
        action=action,
        mask=mask,
        reward=torch.randn(n) * 0.1,
        done=done,
        old_log_prob=log_prob,
        value=torch.randn(n) * 0.1,
        executed=torch.ones(n, dtype=torch.bool),
    )


def _run(model, buffer, **overrides):
    cfg = PPOConfig(**{"epochs": 2, "minibatch": 8, "target_kl": 1e9, **overrides})
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    centers = value_support(-1.0, 1.0, N_BINS)
    n = len(buffer)
    return ppo_update(
        model,
        opt,
        buffer,
        advantages=torch.randn(n) * 0.1,
        returns=torch.randn(n) * 0.1,
        centers=centers,
        sigma=0.1,
        config=cfg,
        device=torch.device("cpu"),
    )


def test_the_ratio_is_one_before_the_first_gradient_step():
    """THE test that would have caught both act/train hazards. The acting agent and the update
    read the same weights through the same masking function, so recomputing log pi over an
    on-policy buffer must reproduce it to floating-point noise: ratio == 1, KL == 0, clip == 0.

    A `-np.inf`-vs-`NEG_INF` split, or a post-hoc `resolve_switch_conflict` rewrite, both leave
    the loss finite and the run apparently healthy -- and both move this number."""
    model = _model()
    buf = _doubles_buffer(model)
    stats = _run(model, buf, epochs=1)
    assert stats["lp_drift"] < 1e-5, f"acting and training disagree by {stats['lp_drift']:.3g}"
    assert abs(stats["approx_kl"]) < 1e-3
    assert stats["clip_frac"] == 0.0


def test_forced_rows_do_not_dilute_the_reported_means():
    """99 forced turns beside 1 real decision must report what the 1-row buffer reports. Under a
    plain `.mean()` they divide policy loss, entropy and approx-KL by ~100 -- the actor stops
    moving relative to the critic and `target_kl` silently stops binding."""
    model = _model()
    buf = _doubles_buffer(model, n=100, decision_rows=4, seed=3)
    w = _surrogate_weights(buf)
    assert w is not None and float(w.sum()) == 4.0, "only the 4 decision rows carry weight"

    stats = _run(_model(), buf, epochs=1, minibatch=100)
    # The same batch scored with an unweighted mean would be ~25x smaller on every actor stat.
    assert stats["dec_frac"] == pytest.approx(0.04)
    assert stats["n_decision"] == 4.0
    assert abs(stats["entropy"]) > 1e-6, "forced rows contribute 0 entropy and must not be counted"


def test_unexecuted_and_illegal_rows_are_zero_weighted_not_dropped():
    """A row whose decode fell back to a random order, and a row whose label is illegal under its
    own restricted mask, both carry a meaningless ratio. They are zeroed rather than removed --
    `compute_gae` scans a contiguous buffer and resets on `done`, so deleting a row would splice
    two unrelated turns together."""
    model = _model()
    buf = _doubles_buffer(model, n=16, seed=5)
    buf.executed[2] = False
    buf.action[3] = torch.tensor([A - 1, A - 1])  # almost certainly illegal under row 3's mask

    w = _surrogate_weights(buf)
    assert w is not None
    assert w[2] == 0.0, "an unexecuted row must not enter the surrogate"
    assert w[3] == 0.0, "an illegal label must not enter the surrogate"
    assert len(buf) == 16, "rows are zeroed, never dropped"

    stats = _run(model, buf, epochs=1)
    assert np.isfinite(stats["policy_loss"]) and abs(stats["policy_loss"]) < 1e3
    assert abs(stats["approx_kl"]) < 1.0, "an illegal row would push this to ~1e9"
    assert stats["invalid_frac"] == pytest.approx(1 / 16)


def test_the_update_runs_and_every_reported_stat_is_finite():
    model = _model()
    stats = _run(model, _doubles_buffer(model))
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "vmae"):
        assert np.isfinite(stats[key]), key
    assert stats["epochs_run"] == 2.0


def test_the_singles_path_keeps_its_unweighted_means():
    """The decision-row denominator is factored-ONLY. Applying it on singles would rescale the
    policy loss, entropy and KL of every OU build ever published."""
    from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE
    from lategame.features.encoder import OBS_DIM
    from lategame.model.policy import masked_logits

    torch.manual_seed(0)
    n = 16
    mask = torch.zeros(n, GEN9_ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :4] = True
    mask[0] = False
    mask[0, 2] = True  # a forced singles row: exactly one legal action
    obs = torch.randn(n, OBS_DIM)
    action = torch.randint(0, 4, (n,))
    action[0] = 2
    model = ActorCritic(OBS_DIM, hidden_dim=16, n_actions=GEN9_ACTION_SPACE_SIZE, n_bins=N_BINS)
    model.eval()
    with torch.no_grad():
        logits, _ = model(obs)
        lp = torch.log_softmax(masked_logits(logits, mask), dim=1)
    buf = RolloutBuffer(
        obs=obs,
        action=action,
        mask=mask,
        reward=torch.randn(n) * 0.1,
        done=torch.zeros(n, dtype=torch.bool),
        old_log_prob=lp.gather(1, action.unsqueeze(1)).squeeze(1),
        value=torch.zeros(n),
    )
    assert _surrogate_weights(buf) is None, "singles must get no weight vector at all"

    stats = _run(model, buf, epochs=1)
    assert "dec_frac" not in stats, "the factored extras must not appear on a singles log line"
    assert "lp_drift" not in stats


def test_a_doubles_checkpoint_is_stamped_with_the_doubles_fingerprint(tmp_path):
    """`_save_checkpoint` used to stamp the singles encoder constants unconditionally, so every
    PPO iteration checkpoint claimed to be singles -- rejected by `DoublesAgent` at the first
    eval, an iteration of rollout too late."""
    from lategame.model.factory import build_model

    path = tmp_path / "iter_01.pt"
    _save_checkpoint(_model(), str(path), "gen9vgc2025regi", -1.0, 1.0, N_BINS)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["obs_version"] == OBS_VERSION_DOUBLES
    assert ck["input_dim"] == OBS_DIM_DOUBLES
    assert ck["n_actions"] == GEN9_DOUBLES_ACTION_SPACE_SIZE
    assert ck["battle_format"] == "gen9vgc2025regi"
    build_model(ck).load_state_dict(ck["state_dict"])  # round-trips


def test_run_ppo_refuses_a_singles_loop_penalty_on_a_doubles_format(tmp_path):
    """`--loop-penalty 4.0` is on every OU invocation and copy-pastes cleanly into a VGC one. The
    doubles guard is a different penalty over a different layout, so an OU-tuned value would be
    recorded in the gate JSON as if it meant the same thing."""
    import asyncio

    from lategame.train.ppo import run_ppo

    path = tmp_path / "init.pt"
    _save_checkpoint(_model(), str(path), "gen9vgc2025regi", -1.0, 1.0, N_BINS)
    cfg = PPOConfig(init=str(path), battle_format="gen9vgc2025regi", loop_penalty=4.0)
    with pytest.raises(ValueError, match="loop-penalty"):
        asyncio.run(run_ppo(cfg))


def test_run_ppo_refuses_a_singles_checkpoint_on_a_doubles_format(tmp_path):
    """The encoder check is now codec-driven, so the doubles format demands a `d1-`/888 warm start
    instead of silently accepting the singles constants."""
    import asyncio

    from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE
    from lategame.features.encoder import OBS_DIM
    from lategame.train.ppo import run_ppo

    path = tmp_path / "singles.pt"
    torch.save(
        {
            "state_dict": {},
            "model_type": "actor_critic",
            "arch": {},
            "input_dim": OBS_DIM,
            "hidden_dim": 16,
            "n_actions": GEN9_ACTION_SPACE_SIZE,
            "n_bins": N_BINS,
            "v_min": -1.0,
            "v_max": 1.0,
            "obs_version": "v5-09831e17c378",
            "battle_format": "gen9vgc2025regi",
            "metrics": {},
        },
        path,
    )
    cfg = PPOConfig(init=str(path), battle_format="gen9vgc2025regi")
    with pytest.raises(ValueError, match="encoder mismatch"):
        asyncio.run(run_ppo(cfg))


def test_decision_stats_reports_composition_for_both_formats():
    doubles = torch.zeros(4, 2, A, dtype=torch.bool)
    doubles[0, 0, :5] = True
    doubles[0, 1, :3] = True  # a decision
    doubles[1:, :, 0] = True  # three forced turns
    stats = decision_stats(doubles, [True, True, False, True])
    assert stats["decision_frac"] == pytest.approx(0.25)
    assert stats["n_decision"] == 1.0
    assert stats["mean_legal"] == pytest.approx(4.0)  # (5 + 3) / 2
    assert stats["invalid_frac"] == pytest.approx(0.25)

    singles = torch.zeros(2, 26, dtype=torch.bool)
    singles[0, :6] = True
    singles[1, 3] = True
    assert decision_stats(singles, [True, True])["decision_frac"] == pytest.approx(0.5)

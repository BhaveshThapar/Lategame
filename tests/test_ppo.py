"""Unit tests for the M5 Phase-2 PPO building blocks (no server needed).

Covers GAE (terminal bootstrap, per-episode reset, discounting), the masked
log-prob/entropy used by the surrogate, the categorical value target's out-of-support
saturation, a synthetic ``ppo_update`` step, the checkpoint roundtrip, and the arena
registration / ``max_concurrent_battles`` forwarding. The full loop is exercised by the
server-gated ``test_ppo_smoke``.
"""

import numpy as np
import pytest

from lategame.eval import arena
from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE
from lategame.features.encoder import OBS_DIM


def test_compute_gae_single_episode_terminal_bootstrap():
    torch = pytest.importorskip("torch")
    from lategame.train.ppo import compute_gae

    reward = torch.tensor([1.0, 1.0, 1.0])
    value = torch.zeros(3)
    done = torch.tensor([False, False, True])
    adv, returns = compute_gae(reward, value, done, gamma=1.0, gae_lambda=1.0)
    # zero baseline + gamma=lambda=1 -> advantage == undiscounted reward-to-go
    np.testing.assert_allclose(adv.numpy(), [3.0, 2.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(returns.numpy(), [3.0, 2.0, 1.0], atol=1e-5)


def test_compute_gae_resets_at_episode_boundary():
    torch = pytest.importorskip("torch")
    from lategame.train.ppo import compute_gae

    reward = torch.tensor([1.0, 1.0, 1.0, 1.0])
    value = torch.zeros(4)
    done = torch.tensor([False, True, False, True])
    adv, _ = compute_gae(reward, value, done, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(adv.numpy(), [2.0, 1.0, 2.0, 1.0], atol=1e-5)


def test_compute_gae_discounts_future_reward():
    torch = pytest.importorskip("torch")
    from lategame.train.ppo import compute_gae

    reward = torch.tensor([0.0, 0.0, 1.0])
    value = torch.zeros(3)
    done = torch.tensor([False, False, True])
    adv, _ = compute_gae(reward, value, done, gamma=0.9, gae_lambda=1.0)
    np.testing.assert_allclose(adv.numpy(), [0.81, 0.9, 1.0], atol=1e-5)


def test_masked_distribution_ignores_illegal_actions():
    torch = pytest.importorskip("torch")
    from lategame.model.policy import masked_logits

    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, False, False]])
    log_probs = torch.log_softmax(masked_logits(logits, mask), dim=1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=1)
    assert torch.isfinite(entropy).all()  # no NaN from 0 * -inf on illegal actions
    assert abs(float(entropy.item()) - float(np.log(2))) < 1e-4  # uniform over 2 legal
    assert float(log_probs.exp()[0, 2].item()) < 1e-6  # illegal carries ~zero mass


def test_hl_gauss_value_target_roundtrips_within_support():
    # ppo_update clamps returns to [v_min, v_max] before building the soft label, so the
    # property it relies on is a faithful round-trip in-support and saturation (within ~one
    # bin) at the clamped boundary -- not the degenerate far-out-of-support case.
    torch = pytest.importorskip("torch")
    from lategame.model.actor_critic import hl_gauss_target, value_from_logits, value_support

    centers = value_support(-5.0, 5.0, 21)
    sigma = 0.75 * (10.0 / 20)
    bin_width = 10.0 / 20

    target = hl_gauss_target(torch.tensor([2.0]), centers, sigma)
    value = value_from_logits(torch.log(target + 1e-12), centers)
    assert abs(float(value.item()) - 2.0) < 1e-3  # interior return round-trips exactly

    cap = hl_gauss_target(torch.tensor([5.0]), centers, sigma)
    cap_value = value_from_logits(torch.log(cap + 1e-12), centers)
    assert float(cap_value.item()) <= 5.0 + 1e-4  # never exceeds the support cap
    assert float(cap_value.item()) > 5.0 - bin_width  # saturates within one bin of the cap


def _synthetic_buffer(torch, n=24, n_bins=21):
    from lategame.model.actor_critic import ActorCritic, value_from_logits, value_support
    from lategame.model.policy import masked_logits
    from lategame.train.ppo import RolloutBuffer

    torch.manual_seed(0)
    model = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=n_bins)
    centers = value_support(-5.0, 5.0, n_bins)
    obs = torch.randn(n, OBS_DIM)
    mask = torch.ones(n, GEN9_ACTION_SPACE_SIZE, dtype=torch.bool)
    action = torch.randint(0, GEN9_ACTION_SPACE_SIZE, (n,))
    model.eval()
    with torch.no_grad():
        logits, value_logits = model(obs)
        lp = torch.log_softmax(masked_logits(logits, mask), dim=1)
        old_log_prob = lp.gather(1, action.unsqueeze(1)).squeeze(1)
        value = value_from_logits(value_logits, centers)
    done = torch.zeros(n, dtype=torch.bool)
    done[n // 2 - 1] = True
    done[n - 1] = True
    reward = torch.randn(n) * 0.1
    buffer = RolloutBuffer(obs, action, mask, reward, done, old_log_prob, value)
    return model, buffer, centers


def test_ppo_update_runs_and_reports_finite_stats():
    torch = pytest.importorskip("torch")
    from lategame.train.ppo import PPOConfig, compute_gae, ppo_update

    model, buffer, centers = _synthetic_buffer(torch)
    config = PPOConfig(epochs=2, minibatch=8, target_kl=10.0)  # no early stop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adv, returns = compute_gae(
        buffer.reward, buffer.value, buffer.done, config.gamma, config.gae_lambda
    )
    stats = ppo_update(
        model,
        optimizer,
        buffer,
        adv,
        returns,
        centers,
        sigma=0.375,
        config=config,
        device=torch.device("cpu"),
    )
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "vmae"):
        assert np.isfinite(stats[key]), f"{key} not finite: {stats[key]}"
    assert stats["entropy"] > 0.0  # a non-degenerate policy keeps some entropy
    assert stats["epochs_run"] == 2.0


def test_save_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from lategame.model.actor_critic import ActorCritic
    from lategame.model.factory import build_model
    from lategame.train.ppo import _save_checkpoint

    model = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    path = tmp_path / "ck.pt"
    _save_checkpoint(model, str(path), "gen9randombattle", -3.0, 3.0, 11)
    ck = torch.load(path, weights_only=False)
    assert ck["v_min"] == -3.0 and ck["v_max"] == 3.0 and ck["n_bins"] == 11
    rebuilt = build_model(ck)
    rebuilt.load_state_dict(ck["state_dict"])  # exact arch + shapes round-trip


def _tiny_entity_transformer(torch, n_bins=21):
    """Small EntityTransformer with the GREEN-checkpoint config (id_embed + dex priors)."""
    from lategame.model.entity_transformer import EntityTransformer

    return EntityTransformer(
        OBS_DIM,
        n_bins=n_bins,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_dim=32,
        id_embed=True,
        id_embed_init="prior",
    )


def test_ppo_warm_starts_entity_transformer_checkpoint(tmp_path):
    # The Lever 10 gate continues PPO from the GREEN *EntityTransformer* offline checkpoint.
    # ``run_ppo`` builds straight from the checkpoint meta, so prove the ET warm-start path
    # (save -> build_model -> load_actor_critic_weights) round-trips before the multi-hour run.
    torch = pytest.importorskip("torch")
    from lategame.model.actor_critic import load_actor_critic_weights
    from lategame.model.factory import build_model
    from lategame.train.ppo import _save_checkpoint

    model = _tiny_entity_transformer(torch, n_bins=11)
    path = tmp_path / "et.pt"
    _save_checkpoint(model, str(path), "gen9randombattle", -3.0, 3.0, 11)
    ck = torch.load(path, weights_only=False)
    assert ck["model_type"] == "entity_transformer"
    assert ck["arch"]["id_embed"] is True and ck["arch"]["id_embed_init"] == "prior"

    rebuilt = build_model(ck)
    load_actor_critic_weights(rebuilt, ck["state_dict"])  # full state dict, exact arch
    obs = torch.zeros(2, OBS_DIM)  # zeros keep the trailing ID channels at valid index 0
    with torch.no_grad():
        logits, value_logits = rebuilt(obs)
    assert logits.shape == (2, GEN9_ACTION_SPACE_SIZE)
    assert value_logits.shape == (2, 11)


def test_ppo_update_runs_on_entity_transformer():
    # The PPO surrogate + categorical value loss must run on the two-tower transformer, not
    # just the MLP -- this is the model the Lever 10 gate actually updates.
    torch = pytest.importorskip("torch")
    from lategame.model.actor_critic import value_from_logits, value_support
    from lategame.model.policy import masked_logits
    from lategame.train.ppo import PPOConfig, RolloutBuffer, compute_gae, ppo_update

    torch.manual_seed(0)
    n, n_bins = 24, 21
    model = _tiny_entity_transformer(torch, n_bins=n_bins)
    centers = value_support(-5.0, 5.0, n_bins)
    obs = torch.zeros(n, OBS_DIM)  # zeros keep the trailing ID channels at valid index 0
    mask = torch.ones(n, GEN9_ACTION_SPACE_SIZE, dtype=torch.bool)
    action = torch.randint(0, GEN9_ACTION_SPACE_SIZE, (n,))
    model.eval()
    with torch.no_grad():
        logits, value_logits = model(obs)
        lp = torch.log_softmax(masked_logits(logits, mask), dim=1)
        old_log_prob = lp.gather(1, action.unsqueeze(1)).squeeze(1)
        value = value_from_logits(value_logits, centers)
    done = torch.zeros(n, dtype=torch.bool)
    done[n // 2 - 1] = True
    done[n - 1] = True
    reward = torch.randn(n) * 0.1
    buffer = RolloutBuffer(obs, action, mask, reward, done, old_log_prob, value)

    config = PPOConfig(epochs=2, minibatch=8, target_kl=10.0)  # no early stop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adv, returns = compute_gae(
        buffer.reward, buffer.value, buffer.done, config.gamma, config.gae_lambda
    )
    stats = ppo_update(
        model, optimizer, buffer, adv, returns, centers, 0.375, config, torch.device("cpu")
    )
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "vmae"):
        assert np.isfinite(stats[key]), f"{key} not finite: {stats[key]}"
    assert stats["epochs_run"] == 2.0


def test_arena_registers_ppo_agent():
    from lategame.agents.ppo_agent import PPORecordingAgent

    assert arena.AGENTS["ppo"] is PPORecordingAgent
    assert "ppo" in arena._CHECKPOINT_AGENTS


def test_build_player_forwards_max_concurrent(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "random", Dummy)
    arena.build_player("random", "gen9randombattle", max_concurrent_battles=7)
    assert captured["max_concurrent_battles"] == 7


def test_build_player_omits_max_concurrent_when_unset(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "random", Dummy)
    arena.build_player("random", "gen9randombattle")
    assert "max_concurrent_battles" not in captured


def test_ppo_config_has_ou_team_and_loop_fields():
    # Build 16: OU self-play needs a team pool + LoopGuard; both default to the RB no-op.
    pytest.importorskip("torch")
    from lategame.train.ppo import PPOConfig

    cfg = PPOConfig()
    assert cfg.team_pool is None
    assert cfg.loop_penalty == 0.0


def test_run_ppo_rejects_format_mismatch(tmp_path):
    # If the warm-start checkpoint's format != config.battle_format, rollout (checkpoint format)
    # and eval (config format) would diverge -- run_ppo must fail loudly before any server call.
    import asyncio

    pytest.importorskip("torch")
    from lategame.model.actor_critic import ActorCritic
    from lategame.train.ppo import PPOConfig, _save_checkpoint, run_ppo

    path = tmp_path / "rb_init.pt"
    _save_checkpoint(ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11), str(path),
                     "gen9randombattle", -3.0, 3.0, 11)
    cfg = PPOConfig(init=str(path), battle_format="gen9ou")
    with pytest.raises(ValueError, match="format mismatch"):
        asyncio.run(run_ppo(cfg))


def test_collect_rollout_forwards_team_and_loop_penalty(monkeypatch):
    # Teams + LoopGuard must reach BOTH the ppo learner and the opponent build (build_player
    # filters loop_penalty to learned agents). Stub the server so this stays a unit test.
    import asyncio

    pytest.importorskip("torch")
    from lategame.data import rollout as rollout_mod
    from lategame.data.collect import PlayerSpec
    from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE

    calls: list[tuple[str, dict]] = []

    class Dummy:
        def __init__(self) -> None:
            self.dropped = 0

    def fake_build_player(name, battle_format, **kwargs):
        calls.append((name, kwargs))
        return Dummy()

    async def fake_cross_evaluate(players, n_challenges):
        return None

    def fake_episodes(player, weights):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        mask = np.ones(GEN9_ACTION_SPACE_SIZE, dtype=bool)
        # (recs, rewards, log_probs, values, executed) -- the 5th list is B6f's executed flag.
        return [([(obs, 0, mask)], [0.0], [0.0], [0.0], [True])]

    monkeypatch.setattr(rollout_mod, "build_player", fake_build_player)
    monkeypatch.setattr(rollout_mod, "cross_evaluate", fake_cross_evaluate)
    monkeypatch.setattr(rollout_mod, "_learner_episodes", fake_episodes)

    buf = asyncio.run(
        rollout_mod.collect_rollout(
            "ck.pt", [PlayerSpec("simpleheuristics")], games_per_opp=1,
            battle_format="gen9ou", team="POOL", loop_penalty=4.0,
        )
    )
    ppo_call = next(k for (n, k) in calls if n == "ppo")
    opp_call = next(k for (n, k) in calls if n == "simpleheuristics")
    assert ppo_call["team"] == "POOL" and ppo_call["loop_penalty"] == 4.0
    assert opp_call["team"] == "POOL" and opp_call["loop_penalty"] == 4.0
    assert len(buf) == 1


# --------------------------------------------------------------------------- #
# Build 19 (docs/RESULTS.md): linear ent_coef / lr schedules.
#
# Motivation is measured, not assumed: on the Build-18 plateau the SAMPLED policy PPO
# optimizes scored 0.347 vs the ARGMAX we deploy at 0.487 (scripts/policy_sharpness_diag.py),
# and policy entropy stalled at 0.465 of uniform. Annealing ent_coef -> 0 lets the
# distribution sharpen onto its own argmax and closes that train/eval gap.
# --------------------------------------------------------------------------- #
def test_anneal_none_final_holds_the_value_constant():
    from lategame.train.ppo import anneal

    # The Build 16-18 path: no schedule => every iteration runs at the start value.
    for k in (1, 7, 50):
        assert anneal(0.01, None, k, 50) == 0.01


def test_anneal_hits_both_endpoints_and_the_midpoint():
    from lategame.train.ppo import anneal

    assert anneal(0.01, 0.0, 1, 51) == pytest.approx(0.01)  # first iter = start
    assert anneal(0.01, 0.0, 51, 51) == pytest.approx(0.0)  # final iter = final
    assert anneal(0.01, 0.0, 26, 51) == pytest.approx(0.005)  # halfway
    assert anneal(2.5e-4, 5e-5, 51, 51) == pytest.approx(5e-5)


def test_anneal_is_monotone_and_clamps_out_of_range_iters():
    from lategame.train.ppo import anneal

    values = [anneal(0.01, 0.0, k, 50) for k in range(1, 51)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert anneal(0.01, 0.0, 0, 50) == pytest.approx(0.01)  # k < 1 clamps to the start
    assert anneal(0.01, 0.0, 99, 50) == pytest.approx(0.0)  # k > iters clamps to the final
    assert anneal(0.01, 0.0, 1, 1) == 0.01  # single-iter run: no interpolation to do


def test_ppo_update_reports_the_values_it_actually_ran_at():
    torch = pytest.importorskip("torch")
    from lategame.train.ppo import PPOConfig, compute_gae, ppo_update

    model, buffer, centers = _synthetic_buffer(torch)
    config = PPOConfig(epochs=1, minibatch=8, target_kl=10.0, ent_coef=0.004)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    adv, returns = compute_gae(
        buffer.reward, buffer.value, buffer.done, config.gamma, config.gae_lambda
    )
    stats = ppo_update(
        model, optimizer, buffer, adv, returns, centers,
        sigma=0.375, config=config, device=torch.device("cpu"),
    )
    # run_ppo anneals by handing ppo_update a per-iteration config + setting the optimizer lr;
    # the stats must echo both back so a scheduled run is auditable per iteration.
    assert stats["ent_coef"] == pytest.approx(0.004)
    assert stats["lr"] == pytest.approx(7e-5)


def test_ppo_config_schedule_defaults_to_no_schedule():
    from lategame.train.ppo import PPOConfig

    config = PPOConfig()
    assert config.ent_coef_final is None
    assert config.lr_final is None


# --------------------------------------------------------------------------- #
# Build 24: the anneal HORIZON, decoupled from `iters`.
#
# `--iters` silently doubled as the schedule horizon, so "more iterations" was really three
# changes at once. Measured in the shipped logs at the SAME iteration 40: v22 (iters=50) ran
# lr 9.08e-05 / ent 0.0020 while v23b (iters=80) ran lr 1.51e-04 / ent 0.0051; by iteration 50
# v22 was frozen at its finals and v23b was still at lr 1.26e-04. A strength difference between
# those budgets therefore could not be attributed to update count.
# --------------------------------------------------------------------------- #
def test_anneal_horizon_defaults_to_iters_so_v17_v23_are_unchanged():
    from lategame.train.ppo import PPOConfig

    assert PPOConfig().anneal_iters is None
    assert PPOConfig(iters=80).anneal_horizon == 80
    assert PPOConfig(iters=50).anneal_horizon == 50


def test_pinned_horizon_overrides_iters():
    from lategame.train.ppo import PPOConfig

    assert PPOConfig(iters=80, anneal_iters=50).anneal_horizon == 50


def test_pinned_horizon_holds_both_schedules_at_their_finals_past_it():
    """The v24d arm: 80 iterations of updates on a schedule that finished at 50."""
    from lategame.train.ppo import PPOConfig, anneal

    cfg = PPOConfig(iters=80, anneal_iters=50, lr=2.5e-4, lr_final=5e-5,
                    ent_coef=0.01, ent_coef_final=0.0)
    h = cfg.anneal_horizon
    assert anneal(cfg.lr, cfg.lr_final, 50, h) == pytest.approx(5e-5)
    for k in (51, 65, 80):  # past the horizon: frozen, not extrapolated past the final
        assert anneal(cfg.lr, cfg.lr_final, k, h) == pytest.approx(5e-5)
        assert anneal(cfg.ent_coef, cfg.ent_coef_final, k, h) == pytest.approx(0.0)


def test_the_horizon_is_what_made_v22_and_v23b_differ_mid_run():
    """Reproduces the confound: same iteration, same schedule endpoints, different horizon."""
    from lategame.train.ppo import anneal

    lr_at_40_short = anneal(2.5e-4, 5e-5, 40, 50)  # v22's budget
    lr_at_40_long = anneal(2.5e-4, 5e-5, 40, 80)  # v23b's budget
    assert lr_at_40_short == pytest.approx(9.08e-05, rel=0.01)
    assert lr_at_40_long == pytest.approx(1.51e-04, rel=0.01)
    assert lr_at_40_long > 1.6 * lr_at_40_short  # the same iteration, a 1.7x different optimizer

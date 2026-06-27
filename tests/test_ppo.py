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

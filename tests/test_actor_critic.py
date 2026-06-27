"""Unit tests for the actor-critic model + value-classification utils.

Skipped if torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from lategame.features.encoder import OBS_DIM  # noqa: E402
from lategame.model.actor_critic import (  # noqa: E402
    ActorCritic,
    hl_gauss_target,
    load_bc_weights,
    value_from_logits,
    value_support,
)
from lategame.model.policy import BCPolicy  # noqa: E402


def test_forward_shapes():
    model = ActorCritic(OBS_DIM, hidden_dim=32, n_bins=21)
    logits, value_logits = model(torch.zeros(4, OBS_DIM))
    assert logits.shape == (4, 26)
    assert value_logits.shape == (4, 21)


def test_value_from_logits_within_support():
    centers = value_support(-2.0, 2.0, 31)
    value = value_from_logits(torch.randn(8, 31), centers)
    assert value.shape == (8,)
    assert torch.all(value >= -2.0 - 1e-5) and torch.all(value <= 2.0 + 1e-5)


def test_hl_gauss_target_normalised_and_peaks_at_return():
    centers = value_support(-1.0, 1.0, 11)  # centres -1.0, -0.8, ..., 1.0
    returns = torch.tensor([0.0, 1.0])
    target = hl_gauss_target(returns, centers, sigma=0.1)
    assert torch.allclose(target.sum(dim=1), torch.ones(2), atol=1e-5)
    assert int(target[0].argmax()) == 5  # bin centred at 0.0
    assert int(target[1].argmax()) == 10  # bin centred at 1.0


def test_load_bc_weights_reproduces_policy_logits():
    bc = BCPolicy(OBS_DIM, hidden_dim=32).eval()
    ac = ActorCritic(OBS_DIM, hidden_dim=32, n_bins=21).eval()
    load_bc_weights(ac, bc.state_dict())

    x = torch.randn(5, OBS_DIM)
    with torch.no_grad():
        ac_logits, _ = ac(x)
        assert torch.allclose(ac_logits, bc(x), atol=1e-6)


def test_load_bc_weights_rejects_mismatched_checkpoint():
    ac = ActorCritic(OBS_DIM, hidden_dim=32)
    with pytest.raises(ValueError):
        load_bc_weights(ac, {"not.a.real.key": torch.zeros(1)})

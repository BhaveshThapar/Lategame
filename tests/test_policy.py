"""Unit tests for the BC policy network. Skipped if torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from rotomai.features.encoder import OBS_DIM  # noqa: E402
from rotomai.model.policy import NEG_INF, BCPolicy, masked_logits  # noqa: E402


def test_forward_shape():
    model = BCPolicy(OBS_DIM, hidden_dim=32)
    logits = model(torch.zeros(4, OBS_DIM))
    assert logits.shape == (4, 26)


def test_masking_blocks_illegal_actions():
    logits = torch.zeros(1, 26)
    mask = torch.zeros(1, 26, dtype=torch.bool)
    mask[0, 6] = True  # only action 6 is legal
    masked = masked_logits(logits, mask)
    assert masked[0, 6] == 0.0
    assert masked[0, 0] == NEG_INF
    assert int(masked.argmax(dim=1).item()) == 6

"""The factored (doubles) policy helpers.

A doubles turn commits both active slots, so the head is two independent 107-way distributions
rather than one 11,449-way joint one. Three properties matter and each is a way to be quietly
wrong: an empty field slot must still have something legal to pick, the loss must not be diluted
by the extra slot, and "correct" must mean the demonstrator's whole TURN rather than half of it.
"""

from __future__ import annotations

import torch

from lategame.model.policy import (
    NEG_INF,
    factored_accuracy,
    factored_cross_entropy,
    factored_logits,
    factored_masked_logits,
)


def test_a_flat_head_reshapes_into_per_slot_rows():
    flat = torch.arange(2 * 2 * 107, dtype=torch.float32).reshape(2, 2 * 107)
    out = factored_logits(flat, n_slots=2)
    assert out.shape == (2, 2, 107)
    assert torch.equal(out[0, 0], flat[0, :107])
    assert torch.equal(out[0, 1], flat[0, 107:])


def test_masking_is_per_slot():
    logits = torch.zeros(1, 2, 5)
    mask = torch.tensor([[[True, False, True, False, False],
                          [False, True, False, False, False]]])
    out = factored_masked_logits(logits, mask)
    assert out[0, 0, 1] == NEG_INF and out[0, 0, 0] == 0.0
    assert out[0, 1, 0] == NEG_INF and out[0, 1, 1] == 0.0


def test_a_slot_with_no_legal_action_can_still_pass():
    """An empty field slot masks to nothing. Left alone that is a uniform distribution over
    ILLEGAL actions -- not an error, just silently meaningless -- so `pass` is re-enabled, which
    is the order the server actually accepts there."""
    logits = torch.zeros(1, 2, 5)
    mask = torch.zeros(1, 2, 5, dtype=torch.bool)
    mask[0, 0, 3] = True  # slot 0 fine; slot 1 has nothing
    out = factored_masked_logits(logits, mask)
    assert out[0, 1, 0] > NEG_INF, "the dead slot must be able to pass"
    assert out[0, 1, 1] == NEG_INF
    assert out[0, 0, 0] == NEG_INF, "a slot WITH legal actions must not get a free pass"


def test_the_loss_is_summed_over_slots_not_averaged():
    """Each slot is a separate decision the demonstrator made. Averaging would halve the gradient
    on a turn where both slots matter, which is most of them."""
    logits = torch.zeros(4, 2, 7, requires_grad=True)
    action = torch.zeros(4, 2, dtype=torch.long)
    summed = factored_cross_entropy(logits, action)
    per_slot = torch.nn.functional.cross_entropy(
        logits.reshape(8, 7), action.reshape(8)
    )
    assert torch.isclose(summed, per_slot * 2)


def test_accuracy_reports_the_strict_turn_and_the_per_slot_count():
    """Both slots right is much harder than one, and only the strict number means 'imitated the
    turn'. Reporting per-slot alone would flatter the model."""
    logits = torch.zeros(3, 2, 4)
    logits[0, 0, 1] = logits[0, 1, 2] = 5.0  # both right
    logits[1, 0, 1] = logits[1, 1, 0] = 5.0  # half right
    logits[2, 0, 3] = logits[2, 1, 3] = 5.0  # both wrong
    action = torch.tensor([[1, 2], [1, 2], [1, 2]])
    both, per_slot = factored_accuracy(logits, action)
    assert both == 1
    assert per_slot == 3  # 2 + 1 + 0

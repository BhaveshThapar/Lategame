"""The factored (doubles) policy helpers.

A doubles turn commits both active slots, so the head is two independent 107-way distributions
rather than one 11,449-way joint one. Three properties matter and each is a way to be quietly
wrong: an empty field slot must still have something legal to pick, the loss must not be diluted
by the extra slot, and "correct" must mean the demonstrator's whole TURN rather than half of it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lategame.agents.doubles_agent import resolve_switch_conflict
from lategame.features.doubles_action_space import N_SWITCHES, SWITCH_BASE, joint_switch_conflict
from lategame.model.policy import (
    NEG_INF,
    factored_accuracy,
    factored_cross_entropy,
    factored_cross_entropy_none,
    factored_entropy,
    factored_has_choice,
    factored_log_prob,
    factored_logits,
    factored_masked_logits,
    restrict_slot1_mask,
    sample_factored_action,
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


# --------------------------------------------------------------------------- #
# B6f: the factored BEHAVIOUR policy.
#
# BC and AWR only SCORE a demonstrator's action; PPO also has to sample from the policy and then
# reproduce that sample's density at update time. Everything below pins a property that, if it
# broke, would still train -- it would just train on an importance ratio whose denominator is the
# density of an action that was never played.
# --------------------------------------------------------------------------- #
def _rand_mask(rng, n_slots=2, a=107, switch_heavy=True):
    mask = torch.zeros(1, n_slots, a, dtype=torch.bool)
    for s in range(n_slots):
        idx = rng.choice(a, size=int(rng.integers(1, 8)), replace=False)
        mask[0, s, idx] = True
        if switch_heavy and rng.random() < 0.5:
            picks = rng.random(N_SWITCHES) < 0.6
            mask[0, s, SWITCH_BASE : SWITCH_BASE + N_SWITCHES] = torch.from_numpy(picks)
        if not mask[0, s].any():
            mask[0, s, 0] = True
    return mask


def test_the_joint_log_prob_is_exactly_minus_the_awr_cross_entropy():
    """`factored_log_prob` and `factored_cross_entropy_none` must be the same number with opposite
    signs. That identity is why the PPO surrogate needs no new loss code -- and if it ever drifts,
    the on-policy path and the offline path would silently disagree about what pi(a|s) is."""
    torch.manual_seed(0)
    mask = torch.rand(64, 2, 40) < 0.3
    mask[:, :, 0] = True
    masked = factored_masked_logits(torch.randn(64, 2, 40), mask)
    action = torch.stack([m.float().multinomial(1).squeeze(1) for m in mask.unbind(1)], 1).long()
    lp = torch.log_softmax(masked, dim=-1)
    ce = factored_cross_entropy_none(masked, action)
    assert torch.allclose(factored_log_prob(lp, action), -ce)


def test_entropy_sums_over_slots_and_a_forced_slot_contributes_exactly_zero():
    """A slot with one legal action is a point mass: zero entropy, and zero gradient. That is why
    a forced turn is safe to keep -- and why counting it in the DENOMINATOR of the reported mean
    would shrink `entropy`, `approx_kl` and the policy loss alike, disarming the trust region
    while the telemetry still reads healthy."""
    mask = torch.zeros(1, 2, 20, dtype=torch.bool)
    mask[0, 0, :4] = True  # uniform over 4 -> log 4
    mask[0, 1, 7] = True  # exactly one legal action -> 0
    lp = torch.log_softmax(factored_masked_logits(torch.zeros(1, 2, 20), mask), dim=-1)
    assert math.isclose(float(factored_entropy(lp)), math.log(4), rel_tol=1e-6)
    assert factored_has_choice(mask).tolist() == [True]

    both_forced = torch.zeros(1, 2, 20, dtype=torch.bool)
    both_forced[0, 0, 3] = both_forced[0, 1, 5] = True
    lp2 = torch.log_softmax(factored_masked_logits(torch.randn(1, 2, 20), both_forced), dim=-1)
    assert abs(float(factored_entropy(lp2))) < 1e-6
    assert factored_has_choice(both_forced).tolist() == [False]


def test_restricting_slot_one_only_fires_for_a_slot_zero_SWITCH():
    """Two slots using the same MOVE index is ordinary double-attack play. Only switches collide,
    so the restriction must key on the codec's switch range, not on 'the actions are equal'."""
    move = SWITCH_BASE + N_SWITCHES + 3  # index 10: comfortably past the switch range
    mask = torch.ones(3, 2, 20, dtype=torch.bool)
    action0 = torch.tensor([SWITCH_BASE + 2, 0, move])  # switch, pass, move
    out = restrict_slot1_mask(mask, action0)
    assert not bool(out[0, 1, SWITCH_BASE + 2]), "slot 1 must lose slot 0's switch"
    assert bool(out[0, 0, SWITCH_BASE + 2]), "slot 0 keeps its own choice"
    assert bool(out[1, 1, 0]) and bool(out[2, 1, move]), "pass/move restrict nothing"
    # No switch anywhere -> the same tensor back, not a copy (the common case allocates nothing).
    assert restrict_slot1_mask(mask, torch.tensor([0, 0, move])) is mask


def test_greedy_sampling_is_the_same_function_as_the_shipped_conflict_repair():
    """THE eval-comparability pin. `DoublesAgent` resolves the joint-switch conflict AFTER argmax;
    the sampler resolves it BEFORE, by masking. For sample=False the two are provably identical --
    both take slot 1's best legal action other than slot 0's -- so the greedy eval path can stay
    byte-identical and serve as the reference implementation for the on-policy one."""
    rng = np.random.default_rng(0)
    conflicts = 0
    for _ in range(2000):
        mask = _rand_mask(rng)
        logits = torch.randn(1, 2, 107)
        ml = np.where(mask[0].numpy(), logits[0].numpy(), -np.inf)
        ref = np.array([int(np.argmax(ml[s])) for s in (0, 1)], dtype=np.int64)
        if joint_switch_conflict(ref, None):
            conflicts += 1
            ref = resolve_switch_conflict(ml, mask[0].numpy(), ref)
        got, _, _ = sample_factored_action(logits, mask, sample=False)
        assert np.array_equal(got[0].numpy(), ref)
    assert conflicts > 0, "the fixture never produced a conflict -- the test proved nothing"


def test_sampling_can_never_produce_a_joint_switch_conflict():
    """The constraint is enforced by construction rather than repaired afterwards, so there is no
    post-hoc rewrite and the recorded action is always the drawn one."""
    torch.manual_seed(0)
    mask = torch.zeros(1, 2, 107, dtype=torch.bool)
    mask[0, :, SWITCH_BASE : SWITCH_BASE + 3] = True  # only switches legal: maximal collision odds
    for _ in range(2000):
        action, _, _ = sample_factored_action(torch.randn(1, 2, 107), mask, sample=True)
        assert not joint_switch_conflict(action[0].numpy(), None)


def test_a_restriction_that_empties_slot_one_still_yields_a_finite_log_prob():
    """The one state where removing slot 0's switch could leave slot 1 with nothing. poke-env
    already makes `pass` legal there (doubles_env.py: `all(force_switch)` with one available
    switch -> `switch_space + [0]`), and `factored_masked_logits` re-enables it regardless -- so
    the density stays defined rather than becoming a NaN on the exact turns doubles is full of."""
    mask = torch.zeros(1, 2, 20, dtype=torch.bool)
    mask[0, 0, SWITCH_BASE] = True  # slot 0 must take that switch
    mask[0, 1, SWITCH_BASE] = True  # ...and it is slot 1's only option too
    action, log_prob, restricted = sample_factored_action(torch.randn(1, 2, 20), mask, sample=True)
    assert not bool(restricted[0, 1].any()), "slot 1's row is empty after the restriction"
    assert action[0, 1].item() == 0, "which leaves pass"
    assert torch.isfinite(log_prob).all()

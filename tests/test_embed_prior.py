"""Unit tests for dex-feature identity priors.

Exercises the train path only (``load_id_priors`` reads the committed npz + projects);
the build path needs GenData and is a maintenance tool, so it is not on the test path.
"""

import numpy as np

from rotomai.features.embed_prior import PRIOR_FIELDS, load_id_priors
from rotomai.features.vocab import vocab_sizes


def test_shapes_dtype_and_padding_row():
    sizes = vocab_sizes()
    priors = load_id_priors(32)
    assert set(priors) == set(PRIOR_FIELDS)
    for cat in PRIOR_FIELDS:
        mat = priors[cat]
        assert mat.shape == (sizes[cat], 32)
        assert mat.dtype == np.float32
        assert np.allclose(mat[0], 0.0)  # UNK / padding_idx row stays zero


def test_scale_matches_default_embedding_std():
    # The warm-start must start at the same scale as the random arm (nn.Embedding ~ N(0,1)),
    # so the gate isolates *structure*, not magnitude.
    for cat, mat in load_id_priors(32).items():
        assert abs(float(mat[1:].std()) - 1.0) < 1e-4, cat


def test_width_follows_id_embed_dim():
    assert load_id_priors(16)["species"].shape[1] == 16
    assert load_id_priors(64)["species"].shape[1] == 64


def test_deterministic_across_rebuilds():
    load_id_priors.cache_clear()
    first = load_id_priors(16)
    load_id_priors.cache_clear()
    second = load_id_priors(16)
    for cat in PRIOR_FIELDS:
        assert np.array_equal(first[cat], second[cat])


def test_rows_carry_structure():
    # Distinct entities get distinct vectors (the whole point of a prior).
    species = load_id_priors(32)["species"]
    assert not np.allclose(species[1], species[2])

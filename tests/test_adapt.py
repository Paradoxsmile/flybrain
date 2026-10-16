"""Prior matching of class sizes."""

from __future__ import annotations

import numpy as np

from flybrain.adapt import adapt_predictions, prior_weights


def test_matching_counts_keep_weights():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    v = prior_weights(probs, probs.sum(axis=0))
    np.testing.assert_allclose(v, [1.0, 1.0], atol=1e-6)


def test_biased_classifier_is_corrected():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    # class 1 is recognized weakly: half of its neurons look ambiguous and lean to class 0
    p1 = np.where(y == 1, rng.uniform(0.35, 0.9, len(y)), rng.uniform(0.0, 0.3, len(y)))
    probs = np.stack([1 - p1, p1], axis=1)
    before = adapt_predictions(probs, np.ones(2), 0.0)
    v = prior_weights(probs, np.array([500.0, 500.0]))
    after = adapt_predictions(probs, v, 1.0)
    q = probs * v
    np.testing.assert_allclose((q / q.sum(1, keepdims=True)).sum(0), [500, 500], rtol=1e-4)
    assert (after == y).mean() > (before == y).mean()
    assert np.array_equal(adapt_predictions(probs, v, 0.0), before)

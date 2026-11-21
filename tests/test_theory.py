"""Theorems 1-2 of the paper: CMC as the integral of the ROC against the Beta(r, n - r) kernel.

The exact cases are a perfect ROC (CMC is 1 at every rank) and a random ROC (CMC at rank r is r / n).
"""

from __future__ import annotations

import numpy as np
import pytest

from roc_cmc.theory import fractional_ranks, predict_cmc_batch, predict_cmc_from_roc


def test_fractional_ranks_are_r_over_n():
    assert fractional_ranks(4).tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_a_perfect_roc_gives_cmc_one_at_every_rank():
    cmc = predict_cmc_from_roc(np.array([0.0, 1.0]), np.array([1.0, 1.0]), n=5)
    assert cmc.shape == (5,)
    assert np.allclose(cmc, 1.0, atol=1e-3)


def test_a_random_roc_gives_cmc_equal_to_the_fractional_rank():
    alpha = np.linspace(0.0, 1.0, 2001)
    cmc = predict_cmc_from_roc(alpha, alpha.copy(), n=5)  # the diagonal ROC, rho = alpha
    assert np.allclose(cmc[:4], [0.2, 0.4, 0.6, 0.8], atol=2e-3)
    assert cmc[-1] == pytest.approx(1.0)


def test_cmc_is_nondecreasing_in_rank_for_any_nondecreasing_roc():
    alpha = np.linspace(0.0, 1.0, 501)
    cmc = predict_cmc_from_roc(alpha, alpha**0.3, n=12)  # a concave, valid ROC
    assert np.all(np.diff(cmc) >= -1e-9)
    assert np.all((cmc >= -1e-9) & (cmc <= 1 + 1e-9))


def test_batch_prediction_matches_one_roc_at_a_time():
    alpha = np.linspace(0.0, 1.0, 101)
    rocs = [(alpha, alpha), (alpha, alpha**2), (np.array([0.0, 1.0]), np.array([1.0, 1.0]))]
    batch = predict_cmc_batch(rocs, n=6)
    assert batch.shape == (3, 6)
    for row, (a, r) in zip(batch, rocs):
        assert np.allclose(row, predict_cmc_from_roc(a, r, n=6), atol=1e-9)

"""Empirical ROC, closed-set CMC, the ROC average and the prediction comparison."""

from __future__ import annotations

import numpy as np
import pytest

from roc_cmc.data.synthetic import score_matrix
from roc_cmc.metrics import empirical_cmc_closed_set, empirical_roc, predict_and_compare
from roc_cmc.metrics.roc_cmc import average_rocs
from roc_cmc.eval import fractional_rank_correlation


def test_roc_of_perfectly_separated_scores_reaches_full_tmr_at_zero_fmr():
    alpha, rho = empirical_roc(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
    assert alpha.tolist() == pytest.approx([0.0, 1.0])
    assert rho.tolist() == pytest.approx([1.0, 1.0])


def test_roc_of_identical_distributions_lies_near_the_diagonal():
    rng = np.random.default_rng(0)
    alpha, rho = empirical_roc(rng.normal(size=3000), rng.normal(size=3000))
    assert np.interp(0.5, alpha, rho) == pytest.approx(0.5, abs=0.05)


def test_roc_without_mated_scores_is_the_unit_square_diagonal():
    alpha, rho = empirical_roc(np.array([]), np.array([0.1, 0.2]))
    assert alpha.tolist() == [0.0, 1.0] and rho.tolist() == [0.0, 1.0]


def test_average_of_identical_rocs_is_that_roc():
    alpha, rho = empirical_roc(np.array([0.9, 0.8, 0.5]), np.array([0.1, 0.6, 0.2]))
    grid, mean_rho = average_rocs([(alpha, rho), (alpha, rho)])
    assert grid.shape == (201,)
    assert np.allclose(mean_rho, np.interp(grid, alpha, rho))


def test_closed_set_cmc_is_one_when_identities_are_separable():
    n, m = 4, 3
    X = np.zeros((n, m, n))
    for omega in range(n):
        X[omega, :, omega] = 1.0  # identical samples per identity, orthogonal across identities
    cmc, per = empirical_cmc_closed_set(X, "cosine", mg=2, rng=np.random.default_rng(0))
    assert cmc.shape == (n,)
    assert np.allclose(cmc, 1.0)
    assert set(per) == set(range(n))
    assert all(np.allclose(per[omega], 1.0) for omega in range(n))


def test_closed_set_cmc_is_a_nondecreasing_curve_in_unit_range():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 4, 4))
    cmc, _ = empirical_cmc_closed_set(X, "cosine", mg=3, rng=np.random.default_rng(2))
    assert np.all(np.diff(cmc) >= 0)
    assert np.all((cmc >= 0) & (cmc <= 1))


def test_prediction_comparison_reports_the_mean_absolute_gap():
    rng = np.random.default_rng(3)
    n, m = 3, 5
    X = rng.normal(size=(n, m, 4))
    flat = X.reshape(n * m, -1)
    S = score_matrix(flat, flat, "cosine")
    mated, nonmated = {}, {}
    for omega in range(n):
        block = S[omega * m : (omega + 1) * m, omega * m : (omega + 1) * m]
        mated[omega] = block[np.triu_indices(m, k=1)]
        cols = np.ones(n * m, dtype=bool)
        cols[omega * m : (omega + 1) * m] = False
        nonmated[omega] = S[omega * m : (omega + 1) * m][:, cols].ravel()
    emp = np.array([0.4, 0.7, 1.0])
    out = predict_and_compare(mated, nonmated, emp, n)
    assert out["n"] == n
    assert len(out["pred_cmc"]) == n
    expected_gap = float(np.mean(np.abs(np.array(out["pred_cmc"]) - emp)))
    assert out["mae_pred_vs_emp_cmc"] == pytest.approx(expected_gap)
    assert len(out["global_roc_alpha"]) == len(out["global_roc_rho"]) >= 2


def test_fractional_rank_correlation_is_one_when_the_cmc_is_linear_in_the_fractional_rank():
    assert fractional_rank_correlation(4, [0.1, 0.4, 0.7, 1.0]) == pytest.approx(1.0)


def test_fractional_rank_correlation_is_none_for_a_constant_cmc_instead_of_nan():
    assert fractional_rank_correlation(4, [1.0, 1.0, 1.0, 1.0]) is None

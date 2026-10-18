"""Theorem 1–2: CMC as integral of ROC against Beta(r, n−r) kernel."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


def _beta_kernels(n: int, n_grid: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Precompute β(α; r, n−r) for r=1..n−1 on a shared grid."""
    xs = np.linspace(1e-6, 1.0 - 1e-6, n_grid)
    kernels = np.zeros((n - 1, n_grid), dtype=float)
    for r in range(1, n):
        kernels[r - 1] = beta_dist.pdf(xs, r, n - r)
    return xs, kernels


def predict_cmc_from_roc(alpha: np.ndarray, rho: np.ndarray, n: int, kernels=None) -> np.ndarray:
    """
    cmc(r|n) = ∫_0^1 ρ(α) β(α; r, n−r) dα for r=1..n−1, cmc(n)=1.
    """
    alpha = np.asarray(alpha, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if alpha[0] > 0:
        alpha = np.concatenate([[0.0], alpha])
        rho = np.concatenate([[rho[0]], rho])
    if alpha[-1] < 1.0:
        alpha = np.concatenate([alpha, [1.0]])
        rho = np.concatenate([rho, [rho[-1]]])
    if kernels is None:
        xs, ker = _beta_kernels(n)
    else:
        xs, ker = kernels
    rho_x = np.interp(xs, alpha, rho)
    integ = getattr(np, "trapezoid", np.trapz)
    cmc = np.ones(n, dtype=float)
    for r in range(n - 1):
        cmc[r] = float(integ(rho_x * ker[r], xs))
    cmc[n - 1] = 1.0
    return cmc


def predict_cmc_batch(rocs: list[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    """Predict CMC for many ROCs sharing gallery size n. Returns (n_rocs, n)."""
    xs, ker = _beta_kernels(n)
    out = np.stack([predict_cmc_from_roc(a, r, n, kernels=(xs, ker)) for a, r in rocs], axis=0)
    return out


def fractional_ranks(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=float) / float(n)

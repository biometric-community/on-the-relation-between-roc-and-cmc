"""Empirical ROC / CMC from mated & nonmated scores."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from roc_cmc.theory import predict_cmc_batch, predict_cmc_from_roc


def empirical_roc(mated: np.ndarray, nonmated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise ROC via thresholds on mated scores (paper Eqs. 28–29)."""
    mated = np.asarray(mated, dtype=float)
    nonmated = np.asarray(nonmated, dtype=float)
    if len(mated) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    # Sort-based ROC: for each unique mated threshold, FMR/TMR via searchsorted
    thr = np.unique(mated)  # ascending
    nm_sorted = np.sort(nonmated) if len(nonmated) else np.array([])
    m_sorted = np.sort(mated)
    n_nm = max(len(nm_sorted), 1)
    n_m = len(m_sorted)
    # scores >= t  →  count from the right
    if len(nm_sorted):
        # index of first element >= t
        idx = np.searchsorted(nm_sorted, thr, side="left")
        alpha = (len(nm_sorted) - idx) / float(len(nm_sorted))
    else:
        alpha = np.zeros_like(thr, dtype=float)
    idx_m = np.searchsorted(m_sorted, thr, side="left")
    rho = (n_m - idx_m) / float(n_m)
    # keep max rho per alpha (descending thr walk)
    pts = {}
    for a, r in zip(alpha[::-1], rho[::-1]):
        a = float(a)
        r = float(r)
        if a not in pts or r > pts[a]:
            pts[a] = r
    alpha = np.array(sorted(pts.keys()), dtype=float)
    rho = np.array([pts[a] for a in alpha], dtype=float)
    if alpha[0] > 0:
        alpha = np.concatenate([[0.0], alpha])
        rho = np.concatenate([[rho[0]], rho])
    if alpha[-1] < 1.0 - 1e-12:
        alpha = np.concatenate([alpha, [1.0]])
        rho = np.concatenate([rho, [max(rho[-1], 1.0)]])
    return alpha, rho


def empirical_cmc_closed_set(
    X: np.ndarray,
    score_fn,
    mg: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Closed-set CMC: for each probe identity, pick one probe sample and one
    gallery sample per identity (mated entry + n-1 impostors), rank by score.
    Average over mg random galleries / mated-entry draws.
    Returns global CMC (n,) and per-identity CMC dict.
    """
    from roc_cmc.data.synthetic import score_matrix

    n, m, _ = X.shape
    global_hits = np.zeros(n, dtype=float)
    per = {ω: np.zeros(n, dtype=float) for ω in range(n)}
    counts = {ω: 0 for ω in range(n)}
    for _ in range(mg):
        for omega in range(n):
            jp, jg = rng.choice(m, size=2, replace=False)
            probe = X[omega, jp : jp + 1]  # (1,D)
            gal_idx = rng.integers(0, m, size=n)
            gal_idx[omega] = jg
            gallery = np.stack([X[nu, gal_idx[nu]] for nu in range(n)], axis=0)
            scores = score_matrix(probe, gallery, score_fn)[0]
            order = np.argsort(-scores)
            rank = int(np.where(order == omega)[0][0]) + 1
            hits = np.zeros(n, dtype=float)
            hits[rank - 1 :] = 1.0
            global_hits += hits
            per[omega] += hits
            counts[omega] += 1
    total = float(mg * n)
    global_cmc = global_hits / total
    per_cmc = {ω: per[ω] / max(counts[ω], 1) for ω in range(n)}
    return global_cmc, per_cmc


def average_rocs(rocs: List[Tuple[np.ndarray, np.ndarray]], grid: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    rhos = []
    for alpha, rho in rocs:
        rhos.append(np.interp(grid, alpha, rho))
    return grid, np.mean(rhos, axis=0)


def predict_and_compare(
    mated: Dict[int, np.ndarray],
    nonmated: Dict[int, np.ndarray],
    empirical_cmc_global: np.ndarray,
    n: int,
) -> dict:
    ind_rocs = [empirical_roc(mated[ω], nonmated[ω]) for ω in range(n)]
    ind_pred = predict_cmc_batch(ind_rocs, n)
    avg_alpha, avg_rho = average_rocs(ind_rocs)
    pred_global = np.mean(ind_pred, axis=0)
    pred_from_avg = predict_cmc_from_roc(avg_alpha, avg_rho, n)
    # global ROC from pooled scores
    all_m = np.concatenate([mated[ω] for ω in range(n)])
    all_nm = np.concatenate([nonmated[ω] for ω in range(n)])
    g_alpha, g_rho = empirical_roc(all_m, all_nm)
    mae = float(np.mean(np.abs(pred_global - empirical_cmc_global)))
    return {
        "mae_pred_vs_emp_cmc": mae,
        "pred_cmc": pred_global.tolist(),
        "emp_cmc": empirical_cmc_global.tolist(),
        "pred_cmc_from_avg_roc": pred_from_avg.tolist(),
        "avg_roc_alpha": avg_alpha.tolist(),
        "avg_roc_rho": avg_rho.tolist(),
        "global_roc_alpha": g_alpha.tolist(),
        "global_roc_rho": g_rho.tolist(),
        "n": n,
    }

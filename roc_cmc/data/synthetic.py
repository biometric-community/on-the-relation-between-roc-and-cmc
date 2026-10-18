"""Gaussian between-/within-identity synthetic features (paper §VI-A)."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

# Eq. (30)
SIGMA2 = np.array([0.5, 0.8, 0.85, 0.9], dtype=float)
B_DIAG = SIGMA2
W_DIAG = 1.0 - SIGMA2

ScoreName = Union[str, object]


def sample_features(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Return X with shape (n, m, 4)."""
    b = rng.normal(0.0, np.sqrt(B_DIAG), size=(n, 4))
    w = rng.normal(0.0, np.sqrt(W_DIAG), size=(n, m, 4))
    return b[:, None, :] + w


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def llr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. (31) log-likelihood-ratio similarity."""
    s2 = SIGMA2
    term0 = -0.5 * np.log(1.0 - s2**4).sum()
    diff = a - b
    summ = a + b
    term1 = -0.25 * np.sum((s2 / (1.0 - s2)) * diff**2)
    term2 = 0.25 * np.sum((s2 / (1.0 + s2)) * summ**2)
    return float(term0 + term1 + term2)


def _normalize_score_name(score_fn: ScoreName) -> str:
    if isinstance(score_fn, str):
        return score_fn
    name = getattr(score_fn, "__name__", "")
    if name == "llr_similarity" or score_fn is llr_similarity:
        return "llr"
    return "cosine"


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine between rows of A (p,D) and B (q,D) → (p,q)."""
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return A @ B.T


def llr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise LLR between rows of A and B → (p,q)."""
    s2 = SIGMA2
    term0 = -0.5 * np.log(1.0 - s2**4).sum()
    diff = A[:, None, :] - B[None, :, :]
    summ = A[:, None, :] + B[None, :, :]
    term1 = -0.25 * np.sum((s2 / (1.0 - s2)) * diff**2, axis=-1)
    term2 = 0.25 * np.sum((s2 / (1.0 + s2)) * summ**2, axis=-1)
    return term0 + term1 + term2


def score_matrix(A: np.ndarray, B: np.ndarray, score_fn: ScoreName) -> np.ndarray:
    kind = _normalize_score_name(score_fn)
    if kind == "llr":
        return llr_matrix(A, B)
    return cosine_matrix(A, B)


def pair_scores(X: np.ndarray, score_fn: ScoreName) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Per-identity mated / nonmated scores (paper §VI). X: (n, m, D)."""
    n, m, _ = X.shape
    flat = X.reshape(n * m, -1)
    S = score_matrix(flat, flat, score_fn)
    mated: Dict[int, np.ndarray] = {}
    nonmated: Dict[int, np.ndarray] = {}
    for omega in range(n):
        i0 = omega * m
        block = S[i0 : i0 + m, i0 : i0 + m]
        iu = np.triu_indices(m, k=1)
        mated[omega] = block[iu].astype(float)
        mask_cols = np.ones(n * m, dtype=bool)
        mask_cols[i0 : i0 + m] = False
        nonmated[omega] = S[i0 : i0 + m][:, mask_cols].ravel().astype(float)
    return mated, nonmated

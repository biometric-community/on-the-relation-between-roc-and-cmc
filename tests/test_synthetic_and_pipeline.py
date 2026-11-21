"""The Gaussian synthetic model, the score functions, the synthetic experiment and the configuration files.

The LFW branch of the training entry point downloads ImageNet weights, so these tests run the synthetic branch only.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from roc_cmc.data.synthetic import (
    SIGMA2,
    W_DIAG,
    _normalize_score_name,
    cosine_matrix,
    cosine_similarity,
    llr_matrix,
    llr_similarity,
    pair_scores,
    sample_features,
    score_matrix,
)
from roc_cmc.train import main, run_synthetic_case


def test_the_model_variances_add_up_to_one_per_dimension():
    assert np.allclose(SIGMA2 + W_DIAG, 1.0)
    assert np.allclose(SIGMA2, [0.5, 0.8, 0.85, 0.9])


def test_sampled_features_have_the_documented_shape_and_unit_variance():
    rng = np.random.default_rng(0)
    X = sample_features(200, 50, rng)
    assert X.shape == (200, 50, 4)
    assert np.allclose(X.var(), 1.0, atol=0.03)


def test_cosine_similarity_cases():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.zeros(3), a) == 0.0


def test_pairwise_matrices_agree_with_the_pairwise_functions():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(2, 4))
    C = cosine_matrix(A, B)
    L = llr_matrix(A, B)
    for i in range(3):
        for j in range(2):
            assert C[i, j] == pytest.approx(cosine_similarity(A[i], B[j]))
            assert L[i, j] == pytest.approx(llr_similarity(A[i], B[j]))


def test_the_llr_score_is_symmetric():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=4), rng.normal(size=4)
    assert llr_similarity(a, b) == pytest.approx(llr_similarity(b, a))


def test_score_names_are_recognised_from_strings_and_functions():
    assert _normalize_score_name("llr") == "llr"
    assert _normalize_score_name(llr_similarity) == "llr"
    assert _normalize_score_name(cosine_similarity) == "cosine"
    A = np.eye(3)
    assert np.allclose(score_matrix(A, A, "cosine"), np.eye(3))


def test_pair_scores_have_the_expected_counts_and_mated_values():
    n, m = 4, 3
    X = np.zeros((n, m, 4))
    for omega in range(n):
        X[omega, :, omega] = 1.0  # every sample of an identity is the same vector
    mated, nonmated = pair_scores(X, "cosine")
    for omega in range(n):
        assert len(mated[omega]) == m * (m - 1) // 2
        assert len(nonmated[omega]) == m * m * (n - 1)
        assert np.allclose(mated[omega], 1.0)
        assert np.allclose(nonmated[omega], 0.0)


def test_one_synthetic_case_returns_a_full_cmc_curve():
    result = run_synthetic_case(n=4, m=5, mg=2, classifier="cosine", seed=0)
    assert len(result["pred_cmc"]) == 4 == len(result["emp_cmc"])
    assert result["pred_cmc"][-1] == pytest.approx(1.0)
    assert result["emp_cmc"][-1] == pytest.approx(1.0)
    assert 0.0 <= result["mae_pred_vs_emp_cmc"] <= 1.0
    assert result["classifier"] == "cosine"


def test_the_llr_classifier_runs_too():
    result = run_synthetic_case(n=3, m=4, mg=2, classifier="llr", seed=1)
    assert result["classifier"] == "llr"
    assert len(result["pred_cmc"]) == 3


def test_the_entry_point_writes_the_synthetic_results_without_lfw(tmp_path):
    config = {
        "paper": {"slug": "test", "title": "test", "doi": "test"},
        "device": "cpu",
        "data": {"lfw_root": str(tmp_path / "absent"), "num_workers": 0},
        "synthetic": {"seed": 1, "classifiers": ["cosine", "llr"], "cases": [{"n": 4, "m": 5, "mg": 2}]},
        "lfw": {"enabled": False},
        "paths": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
            "pred_dir": str(tmp_path / "pred"),
            "figure_dir": str(tmp_path / "figs"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    main(["--config", str(config_path), "--device", "cpu", "--skip-lfw"])
    results = json.loads((tmp_path / "logs" / "synthetic_results.json").read_text(encoding="utf-8"))
    assert len(results["synthetic"]) == 2  # one case with two classifiers
    assert results["paper_table_i"] is True
    summary = json.loads((tmp_path / "logs" / "train_summary.json").read_text(encoding="utf-8"))
    assert "lfw_checkpoint" not in summary


def test_default_and_smoke_configs_cite_the_same_paper(default_cfg, smoke_cfg):
    assert default_cfg["paper"]["doi"] == smoke_cfg["paper"]["doi"] == "10.1109/TBIOM.2023.3298561"
    assert default_cfg["paper"]["slug"] == smoke_cfg["paper"]["slug"]


def test_default_run_covers_both_classifiers_and_two_gallery_sizes(default_cfg):
    assert set(default_cfg["synthetic"]["classifiers"]) == {"llr", "cosine"}
    sizes = sorted(case["n"] for case in default_cfg["synthetic"]["cases"])
    assert sizes == [10, 100]


def test_smoke_run_does_much_less_work_than_the_default(default_cfg, smoke_cfg):
    def work(cfg):
        return sum(c["n"] * c["m"] * c["mg"] for c in cfg["synthetic"]["cases"]) * len(cfg["synthetic"]["classifiers"])

    assert work(smoke_cfg) < work(default_cfg)
    assert smoke_cfg["lfw"]["epochs"] <= default_cfg["lfw"]["epochs"]

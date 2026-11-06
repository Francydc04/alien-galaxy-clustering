import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import models as md  # noqa: E402
from src import preprocessing as pp  # noqa: E402


def test_dominant_value_share_flags_constant_column():
    df = pd.DataFrame({"constant": [1, 1, 1, 1, 2], "varied": [1, 2, 3, 4, 5]})
    share = pp.dominant_value_share(df)
    assert share["constant"] == 0.8 and share["varied"] == 0.2


def test_masked_imputation_error_beats_mean_on_correlated_data():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    df = pd.DataFrame({f"x{i}": base + rng.normal(scale=0.1, size=500) for i in range(5)})
    df = df.mask(rng.random(df.shape) < 0.1)
    scores = pp.masked_imputation_error(pp.standardize(df)).set_index("method")
    assert scores.loc["KNN (k=5)", "rmse"] < scores.loc["Mean", "rmse"]


def test_methods_recover_well_separated_blobs():
    X, truth = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    for method in ("K-Means", "Hierarchical (Ward)", "Gaussian mixture"):
        labels = md.fit_labels(method, X, 3)
        assert pd.crosstab(labels, truth).max(axis=1).sum() == 300


def test_internal_scores_ignore_noise():
    X, _ = make_blobs(n_samples=100, centers=2, random_state=1)
    labels = np.r_[np.zeros(50, int), np.ones(49, int), -1]
    scores = md.internal_scores(X, labels)
    assert set(scores) == {"silhouette", "calinski_harabasz", "davies_bouldin"}


def test_bootstrap_stability_is_high_for_clear_structure():
    X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.5, random_state=2)
    assert md.bootstrap_stability(X, "K-Means", 2, n_boot=5).mean() > 0.95

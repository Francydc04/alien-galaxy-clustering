"""Model selection, stability and agreement for four clustering families."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

SEED = 42


def fit_labels(method: str, X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    if method == "K-Means":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
    if method == "Hierarchical (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if method == "Gaussian mixture":
        return GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed).fit_predict(X)
    raise ValueError(method)


def internal_scores(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Silhouette (higher is better), Calinski-Harabasz (higher), Davies-Bouldin (lower). Noise (-1) is excluded."""
    keep = labels != -1
    X, labels = X[keep], labels[keep]
    if len(set(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


def k_sweep(X: np.ndarray, methods: List[str], ks=range(2, 9)) -> pd.DataFrame:
    rows = []
    for m in methods:
        for k in ks:
            labels = fit_labels(m, X, k)
            rows.append({"method": m, "k": k, **internal_scores(X, labels)})
    return pd.DataFrame(rows)


def gmm_bic(X: np.ndarray, ks=range(1, 9)) -> pd.DataFrame:
    return pd.DataFrame([{"k": k, "bic": GaussianMixture(k, n_init=5, random_state=SEED).fit(X).bic(X)} for k in ks])


def k_distance(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance to each point's min_samples-th neighbour, used to pick DBSCAN's eps."""
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def dbscan_grid(X: np.ndarray, eps_values, min_samples_values) -> pd.DataFrame:
    rows = []
    for ms in min_samples_values:
        for eps in eps_values:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            n_clusters = len(set(labels) - {-1})
            rows.append({"eps": round(float(eps), 3), "min_samples": ms, "clusters": n_clusters,
                         "noise_share": float(np.mean(labels == -1)), **internal_scores(X, labels)})
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, method: str, k: int, n_boot: int = 30, seed: int = SEED) -> np.ndarray:
    """ARI between the full-data partition and partitions refit on bootstrap resamples (compared on the resampled points)."""
    rng = np.random.default_rng(seed)
    reference = fit_labels(method, X, k)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        labels = fit_labels(method, X[idx], k, seed=seed + b + 1)
        scores.append(adjusted_rand_score(reference[idx], labels))
    return np.array(scores)

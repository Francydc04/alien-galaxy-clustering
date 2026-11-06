"""Feature selection, encoding, imputation and scaling for the alien-galaxy dataset."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Step 1 - domain screening: identifiers, dates and abstract categorical traits are removed.
EXCLUDED_BY_DOMAIN = [
    "Planet_ID", "Discovery_Date", "Last_Contact_Days", "Colonization_Year",
    "Cultural_Exchange_Programs", "Species_Expansion_Response",
    "Dominant_Species_Social_Structure", "Young_Colonies",
    "Interstellar_Contact_Cost", "Interplanetary_Communications",
    "Inhabitants_Disputes", "Galactic_Trade_Revenue",
]

CIVILIZATION_ORDER: Dict[str, int] = {"Basic": 1, "Graduation": 2, "2n Cycle": 3, "Master": 4, "PhD": 5}

# Final feature set used for clustering (after variance and correlation screening, see notebook).
CLUSTERING_FEATURES: List[str] = [
    "Food_Production_Tons", "Mineral_Extraction_Tons", "Resource_Mining_Operations",
    "Resource_Allocation_Credits", "CO2_Concentration", "Liquid_Energy_Consumption_Terawatts",
    "Offspring_Colonies", "Biological_Research_Units",
    "Alien_Population_Count", "Exploration_Missions", "Civilization_Level",
]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded columns and ordinal-encode the civilization level."""
    out = df.drop(columns=[c for c in EXCLUDED_BY_DOMAIN if c in df.columns]).copy()
    out["Civilization_Level"] = out.pop("Alien_Civilization_Level").map(CIVILIZATION_ORDER)
    return out


def dominant_value_share(df: pd.DataFrame) -> pd.Series:
    """Share of non-missing rows taken by each column's most frequent value (1.0 = constant)."""
    return df.apply(lambda s: s.value_counts(normalize=True).iloc[0]).sort_values(ascending=False)


def correlated_pairs(df: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    corr = df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("r").reset_index()
    pairs.columns = ["feature_1", "feature_2", "r"]
    pairs = pairs[pairs.r.abs() >= threshold]
    return pairs.reindex(pairs.r.abs().sort_values(ascending=False).index).reset_index(drop=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column, keeping NaNs, so that KNN distances are not dominated by large units."""
    return pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)


def knn_impute(df_std: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return pd.DataFrame(KNNImputer(n_neighbors=k).fit_transform(df_std), index=df_std.index, columns=df_std.columns)


def masked_imputation_error(df_std: pd.DataFrame, share: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Hide a share of the *observed* values, impute them back, and score each method by RMSE (in SD units)."""
    rng = np.random.default_rng(seed)
    observed = np.argwhere(df_std.notna().to_numpy())
    hidden = observed[rng.choice(len(observed), int(share * len(observed)), replace=False)]
    masked = df_std.to_numpy().copy()
    truth = masked[hidden[:, 0], hidden[:, 1]].copy()
    masked[hidden[:, 0], hidden[:, 1]] = np.nan
    masked = pd.DataFrame(masked, columns=df_std.columns)
    methods = {
        "Mean": SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
        "KNN (k=10)": KNNImputer(n_neighbors=10),
    }
    rows = []
    for name, imp in methods.items():
        filled = imp.fit_transform(masked)
        err = filled[hidden[:, 0], hidden[:, 1]] - truth
        rows.append({"method": name, "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def scale_for_clustering(df: pd.DataFrame) -> np.ndarray:
    """Min-max scale to [0, 1] so every feature contributes equally to Euclidean distances."""
    return MinMaxScaler().fit_transform(df)

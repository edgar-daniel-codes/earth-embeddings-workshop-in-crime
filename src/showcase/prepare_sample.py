### Summer Internship - Earth Embeddings
### Showcase - Prepare Full-City Dataset
### By Edgar Daniel

"""
Builds the cached, self-contained dataset behind the interactive risk +
clustering showcase app (``src/showcase/app.py``).

Scores every street in the full 2025 CDMX street dataset with the
persisted supervised champion models, labels each one with its
regression risk tier (reusing ``visuals.geo_applications``' fixed
thresholds, so every street has both a predicted value and a label),
fits a k=4 spherical k-means and a 2 component PCA on the full city
(no subsampling — the whole database is cached), and attaches the real
street LineString geometry and name from the corner-split street
reference layer (``data/proc/pd/09e_corner_split.shp``) for the map.

Run:  python -m src.showcase.prepare_sample
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import pandas as pd

from src.supervised.predict import predict
from src.unsupervised.clustering_kmeans import (
    ModelConfig,
    load_normalized_features,
    spherical_kmeans,
)
from src.unsupervised.dim_reduction import compute_pca
from src.visuals.geo_applications import NEW_DATA_PATH, REGRESSION_THRESHOLDS, bucket_fixed

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "sample" / "showcase_streets_2025.parquet"
STREETS_SHP_PATH = REPO_ROOT / "data" / "proc" / "pd" / "09e_corner_split.shp"

FEATURE_COLUMNS = [f"A{i:02d}" for i in range(64)]
RISK_LABELS = {1: "Low", 2: "Medium", 3: "High"}

K = 4
RNG_SEED = 42

### -------------------------------------------------------------------------------
### Data preparation ----------------------------------------------------------------


def load_and_score(path: str | Path = NEW_DATA_PATH) -> pd.DataFrame:
    """Load the full 2025 street dataset and score every row with the
    persisted supervised champion models.

    Every street gets both a predicted value (``y_cnt_pred``) and a
    regression risk label (``risk_reg``), using the same fixed
    thresholds as the full-CDMX regression risk map. Returns a plain
    DataFrame (no geometry): the map uses the real street geometry
    attached later, from the corner-split reference layer, not the
    buffered polygon this training parquet carries.
    """
    gdf = gpd.read_parquet(path)
    preds = predict(gdf)[["y_ind_proba", "y_cnt_pred"]]
    preds["risk_reg"] = bucket_fixed(preds["y_cnt_pred"], REGRESSION_THRESHOLDS)
    out = pd.DataFrame(gdf[["CVEGEO", "CVE_MUN"] + FEATURE_COLUMNS]).join(preds)
    return out


def load_street_reference(path: str | Path = STREETS_SHP_PATH) -> gpd.GeoDataFrame:
    """Load the cached, full corner-split street network: real LineString
    geometry and street name, keyed by the same CVEGEO used throughout
    the training pipeline."""
    gdf = gpd.read_file(path)[["CVEGEO", "NOMVIAL", "geometry"]]
    return gdf.to_crs(4326)


def add_clusters_and_pca(
    sample: gpd.GeoDataFrame, k: int = K, seed: int = RNG_SEED,
) -> gpd.GeoDataFrame:
    """Fit a fresh spherical k-means (k=4) and a 2D PCA, both on the sample."""
    sample = sample.copy()

    X_norm = load_normalized_features(sample, FEATURE_COLUMNS)
    cfg = ModelConfig(k_range=[k], k_list=[k], rng_seed=seed)
    model = spherical_kmeans(X_norm, k, cfg)
    sample["cluster"] = model.predict(X_norm)

    embedding, _ = compute_pca(sample[FEATURE_COLUMNS], n_components=2)
    sample["PC1"] = embedding["PC1"].to_numpy()
    sample["PC2"] = embedding["PC2"].to_numpy()
    return sample


### -------------------------------------------------------------------------------
### Orchestration -------------------------------------------------------------------


def build_sample(
    path: str | Path = NEW_DATA_PATH,
    k: int = K,
    out_path: str | Path = OUT_PATH,
) -> Path:
    """Build and save the showcase dataset parquet — every street in
    the city, no subsampling.

    Returns
    -------
    Path
        ``out_path``, after the dataset has been written.
    """
    sample = load_and_score(path)
    sample = add_clusters_and_pca(sample, k)
    sample["risk_label"] = sample["risk_reg"].map(RISK_LABELS)

    streets = load_street_reference()
    missing = set(sample["CVEGEO"]) - set(streets["CVEGEO"])
    if missing:
        raise ValueError(
            f"{len(missing)} street(s) not found in {STREETS_SHP_PATH}: "
            f"{sorted(missing)[:5]}..."
        )
    sample = sample.merge(streets, on="CVEGEO", how="left")
    sample = sample.rename(columns={"NOMVIAL": "street_name"})

    sample = gpd.GeoDataFrame(sample, geometry="geometry", crs=streets.crs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        centroids = sample.geometry.centroid
    sample["lon"] = centroids.x
    sample["lat"] = centroids.y

    out_cols = [
        "CVEGEO", "CVE_MUN", "street_name", "lon", "lat",
        "y_ind_proba", "y_cnt_pred", "risk_reg", "risk_label", "cluster",
        "PC1", "PC2", "geometry",
    ]
    out_gdf = gpd.GeoDataFrame(sample[out_cols], geometry="geometry", crs=sample.crs)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_parquet(out_path, index=False)
    return out_path


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.prod import init_logger

    logger = init_logger()
    logger.info("Building the full-city showcase dataset...")
    try:
        path = build_sample()
        logger.info(f"Saved showcase dataset to {path}")
    except Exception as e:
        logger.error(f"Error building the showcase dataset. {e}")

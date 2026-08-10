### Summer Internship - Earth Embeddings
### Utils - Dimensionality reduction shared helpers
### By Edgar Daniel


"""

Shared data-loading, coloring and plotting utilities for the
dimensionality reduction and manifold learning modules.

Colors follow the shared house style (``src.utils.style``); the adverse
target ``y_cnt`` is rendered on the reserved red ramp by default.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.style import DEFAULT as PALETTE


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


# Data utilities


def load_years_data(
    parquet_path: str,
    years: list[int],
    feature_columns: list[str],
    target_col: str = "y_cnt",
    sort_col: str = "cnt",
) -> pd.DataFrame:
    """Load all requested years in a single pass.

    Reads only the needed columns (no geometry) via plain pandas, since a
    full geopandas read of a 10M+ row file pulls the geometry column
    across the wire for nothing - these functions never use it.
    """
    columns = list(dict.fromkeys(feature_columns + [target_col, sort_col, "year"]))

    df = (
        pd.read_parquet(parquet_path, columns=columns)
        .query("year in @years")
    )
    if df.empty:
        raise ValueError(
            f"No rows found for years={years} in {parquet_path}."
        )
    return df


def year_slice(
    df: pd.DataFrame,
    year: int,
    feature_columns: list[str],
    target_col: str = "y_cnt",
    sort_col: str = "cnt",
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice one year out of a frame already loaded by ``load_years_data``."""
    sub = df.query("year == @year").sort_values(sort_col)
    return sub[feature_columns], sub[target_col]


def global_color_norm(
    df: pd.DataFrame,
    target_col: str = "y_cnt",
) -> mpl.colors.Normalize:
    """Color normalization shared across all years."""
    values = df[target_col].dropna()
    if values.empty:
        raise ValueError(f"Column '{target_col}' has no valid values.")
    return mpl.colors.Normalize(vmin=values.min(), vmax=values.max())


def compute_colors(
    y: pd.Series,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap | None = None,
    alpha_min: float = 0.1,
    alpha_max: float = 1.0,
) -> np.ndarray:
    """RGBA colors with alpha proportional to the target.

    ``cmap=None`` resolves to the house red ramp reserved for adverse
    counts (``y_cnt``).
    """
    if cmap is None:
        cmap = PALETTE.bad_cmap()

    y_range = y.max() - y.min()
    if y_range == 0:
        # Constant target: full opacity everywhere (avoids 0-division).
        alpha = np.full(len(y), alpha_max)
    else:
        alpha = alpha_min + (alpha_max - alpha_min) * (y - y.min()) / y_range

    colors = cmap(norm(y))
    colors[:, 3] = alpha
    return colors


def standardize(X, center: bool = True) -> np.ndarray:
    """Scale to unit variance in float32 - halves memory vs float64 at 10M+ rows.

    ``center=False`` skips mean-subtraction, so plain SVD sees true singular
    directions from the origin instead of PCA's centered variance directions.
    """
    return StandardScaler(with_mean=center).fit_transform(X).astype(np.float32)


# Plotting


def add_colorbar(
    fig: mpl.figure.Figure,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap | None,
    label: str,
    rect: tuple[float, float, float, float] = (0.93, 0.15, 0.015, 0.70),
) -> mpl.colorbar.Colorbar:
    """Attach a shared vertical colorbar to a multi-panel figure."""
    if cmap is None:
        cmap = PALETTE.bad_cmap()

    cax = fig.add_axes(rect)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label)
    return cbar

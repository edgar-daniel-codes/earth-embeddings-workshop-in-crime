### Summer Internship - Earth Embeddings
### EDA - Data Quality
### By Edgar Daniel


"""

Relevant plots and processes for conducting a data quality assessment.
Everything here is *schema-agnostic*: given any ``DataFrame`` it infers which
columns are dates or coordinates (English **and** Spanish naming), and treats
whatever is left as numeric or categorical based on its dtype.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Allow both `python -m src.eda.dq` and direct `python src/eda/dq.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reuse the exact palette / theme / helpers (source: src.utils.style)
from src.eda.core import (
    BAD,
    BG,
    BLUE,
    GOOD,
    GRAY,
    INK,
    MUTED,
    NAVY,
    SEQ,
    SKY,
    apply_theme,
    fig_to_png_bytes,
)

# Projected CRS used when a geographic layer needs metric areas.
_AREA_CRS = "EPSG:6372"

### -------------------------------------------------------------------------------
### Functions and Helpers ---------------------------------------------------------

# Name hints
_DATE_HINT = re.compile(
    r"(fecha|date|datetime|timestamp|time|hora|hour)", re.IGNORECASE)
_LAT_HINT = re.compile(r"(^|[_\s])(lat|latitude|latitud)$", re.IGNORECASE)
_LON_HINT = re.compile(r"(^|[_\s])(lon|lng|long|longitude|longitud)$",
                       re.IGNORECASE)

# A datetime name-guess is only accepted if this share of values actually parses.
_PARSE_THRESHOLD = 0.80


def _parses_as_datetime(s: pd.Series, threshold: float = _PARSE_THRESHOLD) -> bool:
    """True when a name-hinted column parses to datetime for most non-null rows."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return parsed.notna().mean() >= threshold


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look temporal by name and actually parse as datetimes."""
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
        elif _DATE_HINT.search(str(c)) and _parses_as_datetime(s):
            out.append(c)
    return out


def _looks_like_coord(s: pd.Series, lo: float, hi: float) -> bool:
    """A numeric column whose values sit inside a plausible lat/lon range."""
    if not pd.api.types.is_numeric_dtype(s):
        return False
    v = s.dropna()
    return not v.empty and v.between(lo, hi).mean() >= 0.95


def detect_geo_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Find (lat, lon) columns by name first, then by numeric range as a fallback."""
    lat_col = lon_col = None
    for c in df.columns:
        if lat_col is None and _LAT_HINT.search(str(c)):
            lat_col = c
        if lon_col is None and _LON_HINT.search(str(c)):
            lon_col = c
    if lat_col is None or lon_col is None:
        for c in df.columns:
            if c in (lat_col, lon_col):
                continue
            if lat_col is None and _looks_like_coord(df[c], -90, 90):
                lat_col = c
            elif lon_col is None and _looks_like_coord(df[c], -180, 180):
                lon_col = c
    return lat_col, lon_col


def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every inferred datetime column cast to datetime64.

    Safe to call before profiling so date-aware plots and the profile table
    report proper temporal ranges. Non-date columns are left untouched.
    """
    df = df.copy()
    for c in detect_datetime_columns(df):
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    return df


def infer_schema(df: pd.DataFrame) -> dict[str, object]:
    """Group columns into datetime / numeric / categorical, plus geo columns.

    Anything not recognised as datetime falls back to numeric (numeric dtype)
    or categorical (everything else) — exactly "as is", no coercion of values.
    """
    datetime_cols = detect_datetime_columns(df)
    lat_col, lon_col = detect_geo_columns(df)
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c in datetime_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return {
        "datetime": datetime_cols,
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }


### -------------------------------------------------------------------------------
### Main functions and plots ------------------------------------------------------


# Tabular reports

def profile_df(df: pd.DataFrame, cat_sample: int = 6) -> pd.DataFrame:
    """Per-column profile: dtype, inferred type, missingness, range/samples."""
    schema = infer_schema(df)
    kind = {}
    for c in schema["datetime"]:
        kind[c] = "datetime"
    for c in schema["numeric"]:
        kind[c] = "numeric"
    for c in schema["categorical"]:
        kind[c] = "categorical"
    if schema["lat_col"]:
        kind[schema["lat_col"]] = "latitude"
    if schema["lon_col"]:
        kind[schema["lon_col"]] = "longitude"

    rows = []
    for col in df.columns:
        s = df[col]
        uniques = s.dropna().unique()
        info = {
            "column": col,
            "dtype": str(s.dtype),
            "type": kind.get(col, "categorical"),
            "nulls": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean()) * 100, 2),
            "distinct": int(len(uniques)),
        }
        if kind.get(col) in ("numeric", "datetime", "latitude",
                             "longitude") and len(uniques):
            info["range"] = f"{s.min()} → {s.max()}"
        else:
            info["range"] = ""
        info["sample_values"] = ", ".join(map(str, uniques[:cat_sample]))
        rows.append(info)
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False,
                                          ignore_index=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for every numeric column (count, spread, quantiles)."""
    rows = []
    for col in df.select_dtypes(include=np.number):
        s = df[col].dropna()
        if s.empty:
            continue
        mean = s.mean()
        rows.append({
            "column": col,
            "count": int(len(s)),
            "null_pct": round(float(df[col].isna().mean()) * 100, 2),
            "mean": mean,
            "median": s.median(),
            "sd": s.std(),
            "cv": s.std() / mean if mean else np.nan,
            "min": s.min(),
            "q10": s.quantile(.10),
            "q25": s.quantile(.25),
            "q75": s.quantile(.75),
            "q90": s.quantile(.90),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


# Missingness

def missingness_bar(df: pd.DataFrame) -> plt.Figure:
    """Null percentage per column — the at-a-glance quality summary."""
    miss = (df.isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(miss) + 1)))
    colors = [BAD if v >= 20 else (SKY if v > 0 else GOOD) for v in miss.values]
    ax.barh(miss.index.astype(str), miss.values, color=colors, height=0.6)
    ax.set_xlabel("% missing")
    ax.set_xlim(0, max(1, miss.max() * 1.15))
    ax.set_title("Missing Values by Column")
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(miss.values):
        ax.text(v + miss.max() * 0.01 + 0.1, i, f"{v:.1f}%",
                va="center", fontsize=8, color=GRAY)
    return fig


def null_heatmap(df: pd.DataFrame, max_rows: int = 2000) -> plt.Figure:
    """Row x column missingness pattern (sampled for large frames)."""
    sample = df.sample(max_rows, random_state=0) if len(df) > max_rows else df
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sample.isna(), ax=ax, cbar=False,
                cmap=[BG, BAD], vmin=0, vmax=1)
    ax.set_title(f"Missingness Pattern ({len(sample):,} rows sampled)")
    ax.set_ylabel("row")
    ax.set_yticks([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def nulls_over_time(df: pd.DataFrame, date_col: str, freq: str = "YE") -> plt.Figure:
    """Null share of each column aggregated over time buckets."""
    tmp = df.copy()
    tmp["_period"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.to_period(
        freq[0]).astype(str)
    miss = (tmp.drop(columns=[date_col])
            .groupby("_period")
            .apply(lambda x: x.drop(columns="_period", errors="ignore")
                   .isna().mean()))
    fig, ax = plt.subplots(figsize=(13, max(4, 0.35 * miss.shape[1] + 1)))
    sns.heatmap(miss.T, ax=ax, cmap=SEQ,
                cbar_kws={"label": "null share", "shrink": 0.6})
    ax.set_title("Null Share Through Time")
    ax.set_xlabel("")
    return fig


# Numeric column diagnostics

def numeric_distribution(df: pd.DataFrame, col: str) -> plt.Figure:
    """Histogram + KDE for one numeric column."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color=BLUE,
                 edgecolor="none", line_kws={"color": NAVY})
    ax.set_title(f"Distribution — {col}")
    ax.grid(axis="y")
    return fig


def numeric_cdf(df: pd.DataFrame, col: str) -> plt.Figure:
    """Empirical cumulative distribution for one numeric column."""
    s = df[col].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    if not s.empty:
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s.values, y, color=BLUE, lw=2)
        ax.fill_between(s.values, y, alpha=0.1, color=BLUE)
    ax.set_title(f"CDF — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("cumulative probability")
    ax.grid(axis="y")
    return fig


def numeric_boxplot(df: pd.DataFrame, col: str) -> plt.Figure:
    """Horizontal boxplot to surface spread and outliers for one column."""
    fig, ax = plt.subplots(figsize=(10, 2.4))
    sns.boxplot(x=df[col].dropna(), ax=ax, color=NAVY,
                flierprops={"marker": "o", "markerfacecolor": BAD,
                            "markeredgecolor": "none", "markersize": 3})
    ax.set_title(f"Boxplot — {col}")
    ax.grid(axis="x")
    return fig


# Categorical column diagnostics

def categorical_frequency(df: pd.DataFrame, col: str,
                          top: int = 25) -> plt.Figure:
    """Frequency bars for one categorical column (top values, high-cardinality safe)."""
    vc = df[col].value_counts().head(top).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(vc) + 1)))
    ax.barh(vc.index.astype(str), vc.values, color=BLUE, height=0.6)
    ax.set_xlabel("count")
    total = int(df[col].nunique())
    suffix = f" (top {top} of {total})" if total > top else ""
    ax.set_title(f"Frequencies — {col}{suffix}")
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(vc.values):
        ax.text(v * 1.01, i, f"{v:,}", va="center", fontsize=8, color=GRAY)
    return fig


# Geospatial sanity check

def geo_scatter(df: pd.DataFrame, lat_col: str, lon_col: str) -> plt.Figure:
    """Raw scatter of every geolocated row — reveals swaps, zeros and outliers."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df[lon_col], df[lat_col], s=4, alpha=0.4, color=SKY,
               edgecolors="none")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    ax.set_title("Geolocated Occurrences")
    ax.grid(True)
    return fig


# Embeddings Data Quality

def compute_embedding_coverage(
    pd_shp: str,
    embeddings_path: str,
    figures_path: str,
    label: str,
    pixel_area_m2: float = 100.0,
) -> pd.DataFrame:
    """Compute municipality embedding coverage and save a coverage table figure.

    Parameters
    ----------
    pd_shp : str
        Path to municipality shapefile.
    embeddings_path : str
        Path to embeddings parquet file/dataset.
    figures_path : str
        Directory where the figure will be saved (created if missing).
    label : str
        Label appended to the figure file name.
    pixel_area_m2 : float
        Ground area represented by one embedding point (default 10m x 10m).

    Returns
    -------
    pd.DataFrame
        Coverage dataframe.
    """
    apply_theme()

    # Read data. 
    gdf_pd = gpd.read_file(pd_shp)
    if gdf_pd.crs is not None and gdf_pd.crs.is_geographic:
        gdf_pd = gdf_pd.to_crs(_AREA_CRS)
    gdf_pd = gdf_pd.assign(mun_area=lambda x: x.geometry.area)

    df_emb = pd.read_parquet(embeddings_path)

    # Coverage: points-per-municipality x pixel area vs municipality area.
    coverage = (
        df_emb.groupby("CVE_MUN")
        .size()
        .reset_index(name="count")
        .assign(CVE_MUN=lambda x: x["CVE_MUN"].astype(str).str.zfill(3))
        .merge(
            gdf_pd[["CVE_MUN", "mun_area"]],
            on="CVE_MUN",
            how="left",
        )
        .assign(cov_area=lambda x: x["count"] * pixel_area_m2)
    )

    coverage["cov_pct_ap"] = (
        (coverage["cov_area"] / coverage["mun_area"])
        .clip(upper=1)
        .mul(100)
        .astype(int)
    )

    coverage_plot = (
        coverage[["CVE_MUN", "cov_pct_ap"]]
        .sort_values("cov_pct_ap", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(coverage_plot))))

    # Background reference bars
    ax.barh(coverage_plot["CVE_MUN"], 100, color="#DCE8F5", height=0.55,
            edgecolor="none")
    ax.barh(coverage_plot["CVE_MUN"], coverage_plot["cov_pct_ap"],
            color=BLUE, height=0.55, edgecolor="none")

    # Labels
    for i, (_, row) in enumerate(coverage_plot.iterrows()):
        ax.text(row["cov_pct_ap"] + 2, i, f'{row["cov_pct_ap"]}%',
                va="center", fontsize=10, color=INK, fontweight="medium")

    # Formatting
    ax.set_xlim(0, 110)
    ax.set_xlabel("Coverage (%)", fontsize=10, color=MUTED)
    ax.set_ylabel("")
    ax.set_title("Embedding Coverage by Municipality", fontsize=14,
                 fontweight="bold", loc="left", color=INK, pad=12)

    # Remove unnecessary ink
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", colors=MUTED)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.35)

    plt.tight_layout()

    out_dir = Path(figures_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"coverages_embeddings_{label}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    return coverage


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


if __name__ == "__main__":

    from src.utils.prod import init_logger

    logger = init_logger()

    EMBEDDINGS_PATH = "./data/proc/embeddings/alpha_earth/cdmx/year={}"
    PD_SHP = "./data/spatial/pd/09mun.shp"
    FIG_PATH = "./docs/resources/unsupervised/"
    YEARS_LIST = [2022, 2023, 2024, 2025]

    logger.info(
        f"Working on assessment for embeddings covering for political "
        f"division file {PD_SHP}"
    )
    for year in YEARS_LIST:
        logger.info(f"Working with year {year}")
        try:
            coverage = compute_embedding_coverage(
                pd_shp=PD_SHP,
                embeddings_path=EMBEDDINGS_PATH.format(year),
                figures_path=FIG_PATH,
                label=str(year),
            )
            logger.info(f"Coverage successfully calculated for year {year}")
        except Exception as e:
            logger.error(f"Error calculating coverage for year {year}. \n {e}")

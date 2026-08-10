### Summer Internship - Earth Embeddings
### EDA - Calidad de datos (versión en español)
### By Edgar Daniel


"""

Gráficas y procesos relevantes para una evaluación de calidad de datos,
con todos los textos visibles en español.

Copia en español de ``src.eda.dq``: la lógica es idéntica, únicamente
cambian las cadenas que se renderizan en las figuras.

Todo aquí es *agnóstico del esquema*: dado cualquier ``DataFrame`` infiere
qué columnas son fechas o coordenadas (nombres en inglés **y** español), y
trata el resto como numérico o categórico según su dtype.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Allow both `python -m src.eda.dq_esp` and direct `python src/eda/dq_esp.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reuse the exact palette / theme / helpers (source: src.utils.style)
from src.eda.core_esp import (
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
    """True cuando una columna sugerida por nombre se convierte a fecha en la
    mayoría de sus registros no nulos."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return parsed.notna().mean() >= threshold


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Columnas que parecen temporales por su nombre y sí convierten a fecha."""
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
        elif _DATE_HINT.search(str(c)) and _parses_as_datetime(s):
            out.append(c)
    return out


def _looks_like_coord(s: pd.Series, lo: float, hi: float) -> bool:
    """Columna numérica cuyos valores caen en un rango plausible de lat/lon."""
    if not pd.api.types.is_numeric_dtype(s):
        return False
    v = s.dropna()
    return not v.empty and v.between(lo, hi).mean() >= 0.95


def detect_geo_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Encuentra las columnas (lat, lon) por nombre y, si falla, por rango numérico."""
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
    """Regresa una copia con cada columna de fecha inferida convertida a datetime64.

    Es seguro llamarla antes del perfilado para que las gráficas y la tabla de
    perfil reporten rangos temporales correctos. Las demás columnas no se tocan.
    """
    df = df.copy()
    for c in detect_datetime_columns(df):
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    return df


def infer_schema(df: pd.DataFrame) -> dict[str, object]:
    """Agrupa las columnas en fecha / numérica / categórica, más las geográficas.

    Todo lo que no se reconoce como fecha cae a numérico (dtype numérico) o
    categórico (el resto) — exactamente "tal cual", sin forzar valores.
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
    """Perfil por columna: dtype, tipo inferido, valores faltantes, rango/ejemplos."""
    schema = infer_schema(df)
    kind = {}
    for c in schema["datetime"]:
        kind[c] = "fecha"
    for c in schema["numeric"]:
        kind[c] = "numérica"
    for c in schema["categorical"]:
        kind[c] = "categórica"
    if schema["lat_col"]:
        kind[schema["lat_col"]] = "latitud"
    if schema["lon_col"]:
        kind[schema["lon_col"]] = "longitud"

    rows = []
    for col in df.columns:
        s = df[col]
        uniques = s.dropna().unique()
        info = {
            "columna": col,
            "dtype": str(s.dtype),
            "tipo": kind.get(col, "categórica"),
            "nulos": int(s.isna().sum()),
            "pct_nulos": round(float(s.isna().mean()) * 100, 2),
            "distintos": int(len(uniques)),
        }
        if kind.get(col) in ("numérica", "fecha", "latitud",
                             "longitud") and len(uniques):
            info["rango"] = f"{s.min()} → {s.max()}"
        else:
            info["rango"] = ""
        info["valores_ejemplo"] = ", ".join(map(str, uniques[:cat_sample]))
        rows.append(info)
    return pd.DataFrame(rows).sort_values("pct_nulos", ascending=False,
                                          ignore_index=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos de resumen de cada columna numérica (conteo, dispersión, cuantiles)."""
    rows = []
    for col in df.select_dtypes(include=np.number):
        s = df[col].dropna()
        if s.empty:
            continue
        mean = s.mean()
        rows.append({
            "columna": col,
            "conteo": int(len(s)),
            "pct_nulos": round(float(df[col].isna().mean()) * 100, 2),
            "media": mean,
            "mediana": s.median(),
            "de": s.std(),
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
    """Porcentaje de nulos por columna — el resumen de calidad de un vistazo."""
    miss = (df.isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(miss) + 1)))
    colors = [BAD if v >= 20 else (SKY if v > 0 else GOOD) for v in miss.values]
    ax.barh(miss.index.astype(str), miss.values, color=colors, height=0.6)
    ax.set_xlabel("% de valores faltantes")
    ax.set_xlim(0, max(1, miss.max() * 1.15))
    ax.set_title("Valores faltantes por columna")
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(miss.values):
        ax.text(v + miss.max() * 0.01 + 0.1, i, f"{v:.1f}%",
                va="center", fontsize=8, color=GRAY)
    return fig


def null_heatmap(df: pd.DataFrame, max_rows: int = 2000) -> plt.Figure:
    """Patrón de faltantes fila x columna (muestreado en tablas grandes)."""
    sample = df.sample(max_rows, random_state=0) if len(df) > max_rows else df
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sample.isna(), ax=ax, cbar=False,
                cmap=[BG, BAD], vmin=0, vmax=1)
    ax.set_title(f"Patrón de valores faltantes ({len(sample):,} filas muestreadas)")
    ax.set_ylabel("fila")
    ax.set_yticks([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def nulls_over_time(df: pd.DataFrame, date_col: str, freq: str = "YE") -> plt.Figure:
    """Proporción de nulos de cada columna agregada por periodos de tiempo."""
    tmp = df.copy()
    tmp["_period"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.to_period(
        freq[0]).astype(str)
    miss = (tmp.drop(columns=[date_col])
            .groupby("_period")
            .apply(lambda x: x.drop(columns="_period", errors="ignore")
                   .isna().mean()))
    fig, ax = plt.subplots(figsize=(13, max(4, 0.35 * miss.shape[1] + 1)))
    sns.heatmap(miss.T, ax=ax, cmap=SEQ,
                cbar_kws={"label": "proporción de nulos", "shrink": 0.6})
    ax.set_title("Proporción de nulos a través del tiempo")
    ax.set_xlabel("")
    return fig


# Numeric column diagnostics

def numeric_distribution(df: pd.DataFrame, col: str) -> plt.Figure:
    """Histograma + KDE de una columna numérica."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color=BLUE,
                 edgecolor="none", line_kws={"color": NAVY})
    ax.set_title(f"Distribución — {col}")
    ax.grid(axis="y")
    return fig


def numeric_cdf(df: pd.DataFrame, col: str) -> plt.Figure:
    """Distribución acumulada empírica de una columna numérica."""
    s = df[col].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    if not s.empty:
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s.values, y, color=BLUE, lw=2)
        ax.fill_between(s.values, y, alpha=0.1, color=BLUE)
    ax.set_title(f"CDF — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("probabilidad acumulada")
    ax.grid(axis="y")
    return fig


def numeric_boxplot(df: pd.DataFrame, col: str) -> plt.Figure:
    """Diagrama de caja horizontal para mostrar dispersión y valores atípicos."""
    fig, ax = plt.subplots(figsize=(10, 2.4))
    sns.boxplot(x=df[col].dropna(), ax=ax, color=NAVY,
                flierprops={"marker": "o", "markerfacecolor": BAD,
                            "markeredgecolor": "none", "markersize": 3})
    ax.set_title(f"Diagrama de caja — {col}")
    ax.grid(axis="x")
    return fig


# Categorical column diagnostics

def categorical_frequency(df: pd.DataFrame, col: str,
                          top: int = 25) -> plt.Figure:
    """Barras de frecuencia de una columna categórica (valores principales,
    seguro ante alta cardinalidad)."""
    vc = df[col].value_counts().head(top).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(vc) + 1)))
    ax.barh(vc.index.astype(str), vc.values, color=BLUE, height=0.6)
    ax.set_xlabel("conteo")
    total = int(df[col].nunique())
    suffix = f" (top {top} de {total})" if total > top else ""
    ax.set_title(f"Frecuencias — {col}{suffix}")
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(vc.values):
        ax.text(v * 1.01, i, f"{v:,}", va="center", fontsize=8, color=GRAY)
    return fig


# Geospatial sanity check

def geo_scatter(df: pd.DataFrame, lat_col: str, lon_col: str) -> plt.Figure:
    """Dispersión cruda de cada registro geolocalizado — revela intercambios,
    ceros y valores atípicos."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df[lon_col], df[lat_col], s=4, alpha=0.4, color=SKY,
               edgecolors="none")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    ax.set_title("Ocurrencias geolocalizadas")
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
    """Calcula la cobertura de embeddings por alcaldía y guarda la figura.

    Parameters
    ----------
    pd_shp : str
        Ruta al shapefile de alcaldías.
    embeddings_path : str
        Ruta al archivo/dataset parquet de embeddings.
    figures_path : str
        Directorio donde se guarda la figura (se crea si no existe).
    label : str
        Etiqueta que se añade al nombre del archivo de la figura.
    pixel_area_m2 : float
        Área en el terreno representada por un punto del embedding
        (por defecto 10m x 10m).

    Returns
    -------
    pd.DataFrame
        Tabla de cobertura.
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
    ax.set_xlabel("Cobertura (%)", fontsize=10, color=MUTED)
    ax.set_ylabel("")
    ax.set_title("Cobertura de embeddings por alcaldía", fontsize=14,
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
    plt.savefig(out_dir / f"coverages_embeddings_{label}_esp.png",
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
        f"Evaluando la cobertura de los embeddings para el archivo de "
        f"división política {PD_SHP}"
    )
    for year in YEARS_LIST:
        logger.info(f"Trabajando con el año {year}")
        try:
            coverage = compute_embedding_coverage(
                pd_shp=PD_SHP,
                embeddings_path=EMBEDDINGS_PATH.format(year),
                figures_path=FIG_PATH,
                label=str(year),
            )
            logger.info(f"Cobertura calculada correctamente para el año {year}")
        except Exception as e:
            logger.error(f"Error al calcular la cobertura del año {year}. \n {e}")

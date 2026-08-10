### Summer Internship - Earth Embeddings
### ETL - Assign point events to geometries
### By Edgar Daniel


"""

Aggregate point events (crime incidents) onto reference geometries:
nearest street segment (LINESTRING) or containing polygon (POLYGON),
producing per-geometry event counts.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd

# Parameters

_VALID_GEOM_TYPES = ["LINESTRING", "POLYGON"]

OUT_PATH = "./data/proc/fgr_dbs/"

BBOX = {"lat_lo": 19.04, "lat_hi": 19.75, "lon_lo": -99.40, "lon_hi": -98.95}
FILTER_COL = "year"

POINTS_PATHS = {
    "./data/clean/carpetasFGJ_asaltos.csv": {
        "file_format": "csv",
        "lat_col": "lat",
        "lon_col": "lon",
        "label": "asaltos",
    },
    "./data/clean/carpetasFGJ_homicidios.csv": {
        "file_format": "csv",
        "lat_col": "lat",
        "lon_col": "lon",
        "label": "homicidios",
    },
    "./data/clean/carpetasFGJ_robo.csv": {
        "file_format": "csv",
        "lat_col": "lat",
        "lon_col": "lon",
        "label": "robo",
    },
}

GEOM_PATHS = {
    "./data/proc/pd/09e_corner_split.shp": {
        "geom_type": "LINESTRING",
        "label": "streets",
    },
    "./data/spatial/pd/09a.shp": {
        "geom_type": "POLYGON",
        "label": "ageb",
    },
}

FEAT_COLS = [
    "POBTOT", "POBFEM", "POBMAS", "P_0A2", "P_3YMAS", "P_15YMAS",
    "P_60YMAS", "REL_H_M", "POB0_14", "POB15_64", "POB65_MAS",
    "PEA", "PE_INAC", "POCUPADA", "PDESOCUP", "GRAPROES",
    "PSINDER", "PDER_SS", "TVIVHAB", "TVIVPARHAB", "PROM_OCUP",
    "OCUPVIVPAR", "VPH_S_ELEC", "VPH_AGUAFV", "VPH_NODREN",
    "VPH_INTER", "VPH_PC", "VPH_CEL",
]


### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def agg_points_to_geometry(
    points_gdf: gpd.GeoDataFrame,
    geom_gdf: gpd.GeoDataFrame,
    geom_type: Literal["LINESTRING", "POLYGON"],
    point_id_col: str | None = None,
    geometry_id_col: str | None = None,
    feature_list: list | None = None,
) -> gpd.GeoDataFrame:
    """Aggregate point counts to geometries.

    LINESTRING joins each point to its nearest line; POLYGON keeps points
    intersecting a polygon. Geometries with no points get ``cnt`` = 0.

    ``feature_list`` names attribute columns to carry through from
    ``geom_gdf`` — per-geometry values already assigned upstream (e.g.
    census features via nearest join), passed along as-is, never
    aggregated. Only ``cnt`` is computed here.
    """
    feature_list = list(feature_list or [])

    missing_feats = [c for c in feature_list if c not in geom_gdf.columns]
    if missing_feats:
        warnings.warn(
            f"feature_list columns not present on the geometry layer and "
            f"skipped: {missing_feats}"
        )
        feature_list = [c for c in feature_list if c in geom_gdf.columns]

    if points_gdf.crs is None:
        raise ValueError("points_gdf has no CRS.")
    if geom_gdf.crs is None:
        raise ValueError("geom_gdf has no CRS.")

    if point_id_col is None:
        point_id_col = "_point_id"
        points_gdf = points_gdf.reset_index(names=point_id_col)
    if geometry_id_col is None:
        geometry_id_col = "_geometry_id"
        geom_gdf = geom_gdf.reset_index(names=geometry_id_col)

    points = points_gdf.to_crs(geom_gdf.crs)
    geometries = geom_gdf[[geometry_id_col, "geometry"]]

    if geom_type == "LINESTRING":
        # each point -> nearest line
        joined = points.sjoin_nearest(geometries, how="left")
    elif geom_type == "POLYGON":
        joined = points.sjoin(geometries, how="inner", predicate="intersects")
    else:
        raise ValueError(f"Invalid geom_type: {geom_type!r}")

    counts = (
        joined
        .groupby(geometry_id_col)
        .size()
        .rename("cnt")
        .reset_index()
    )

    result = (
        geom_gdf
        .merge(counts, on=geometry_id_col, how="left")
        .fillna({"cnt": 0})
        .astype({"cnt": int})
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=geom_gdf.crs,
    )[feature_list + [geometry_id_col, "cnt", "geometry"]]


def read_point_data(
    path: str | Path,
    lat_col: str = "lat",
    lon_col: str = "lon",
    file_format: Literal["csv", "parquet"] = "csv",
) -> gpd.GeoDataFrame:
    """Read tabular point data and return a GeoDataFrame in WGS84."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if file_format == "csv":
        df = pd.read_csv(path)
    elif file_format == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format '{file_format}'. Expected 'csv' or 'parquet'."
        )

    missing = {lat_col, lon_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


if __name__ == "__main__":

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.utils.prod import init_logger

    logger = init_logger()

    for points_path, points_spec in POINTS_PATHS.items():
        for geom_path, geom_spec in GEOM_PATHS.items():
            logger.info(
                f"Processing aggregation for points in {points_path} "
                f"to geometries in {geom_path}"
            )

            try:
                point_gdf = read_point_data(
                    points_path,
                    points_spec["lat_col"],
                    points_spec["lon_col"],
                    points_spec["file_format"],
                )

                # Mask and valid coordinates
                mask = (
                    point_gdf[points_spec["lat_col"]]
                    .between(BBOX["lat_lo"], BBOX["lat_hi"])
                    & point_gdf[points_spec["lon_col"]]
                    .between(BBOX["lon_lo"], BBOX["lon_hi"])
                )

                point_gdf = point_gdf[point_gdf.geometry.notnull()]
                point_gdf = point_gdf[point_gdf.is_valid]
                point_gdf = point_gdf[mask]

                filters_list = point_gdf[FILTER_COL].drop_duplicates().to_list()

                geom_gdf = gpd.read_file(geom_path)

                # Features are attributes of the geometry layer; only the
                # ones it actually carries are passed through (see
                # agg_points_to_geometry). Log the gap so runs are auditable.
                present_feats = [c for c in FEAT_COLS if c in geom_gdf.columns]
                if len(present_feats) < len(FEAT_COLS):
                    logger.warning(
                        f"{geom_path} carries {len(present_feats)}/"
                        f"{len(FEAT_COLS)} feature columns; missing ones "
                        f"are skipped."
                    )

                for flt in filters_list:

                    out_file = (
                        f"{OUT_PATH}{points_spec['label']}_by_"
                        f"{geom_spec['label']}_{FILTER_COL}_{flt}.geojson"
                    )

                    try:
                        agg = (
                            agg_points_to_geometry(
                                point_gdf[point_gdf[FILTER_COL] == flt]
                                .to_crs(geom_gdf.crs),
                                geom_gdf,
                                geom_spec["geom_type"],
                                point_id_col=None,
                                geometry_id_col="CVEGEO",
                                feature_list=FEAT_COLS,
                            )
                            .assign(**{FILTER_COL: flt})
                        )

                        # Street buffer: replace segments with a 10-unit buffer
                        # around the source lines. NOTE: buffer units follow the
                        # layer CRS (metres only if the CRS is projected).
                        # ``.to_numpy()`` avoids index-alignment surprises.
                        if geom_spec["geom_type"] == "LINESTRING":
                            agg = agg.copy()
                            agg["geometry"] = (
                                geom_gdf.geometry.buffer(10).to_numpy()
                            )

                        agg.to_crs(4326).to_file(out_file, driver="GeoJSON")

                        logger.info(f"Aggregated file saved on {out_file}. ")

                    except Exception as e:
                        logger.error(f"Error processing aggregation. {e}")

            except Exception as e:
                logger.error(f"Error reading file pair. {e}")

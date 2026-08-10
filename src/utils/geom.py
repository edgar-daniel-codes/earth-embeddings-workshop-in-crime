### Summer Internship - Earth Embeddings
### Utils - Geometric operations and objects
### By Edgar Daniel


"""

Common geometric operations on geospatial objects such as streets
(LINESTRING) and buildings (POLYGON).

This module is the single implementation of the street-splitting logic;
the EDA layer imports from here (the former copy in ``eda_core`` was a
duplicate and has been removed).

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import math

import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from shapely.ops import split, substring


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


def split_streets_at_intersections(
    streets_gdf: gpd.GeoDataFrame, id_col: str = "STREET_ID"
) -> gpd.GeoDataFrame:
    """Split street geometries at all intersections, preserving the source ID.

    Returns a new GeoDataFrame whose IDs are suffixed per part:
    ``<STREET_ID>_0001``, ``<STREET_ID>_0002``, ...

    Notes
    -----
    O(n * k) where k is the local density of candidate neighbours from the
    spatial index; adequate for city-scale networks but slow enough that
    callers should cache the result.
    """
    streets = streets_gdf.copy()

    # Convert MultiLineStrings to individual LineStrings.
    streets = streets.explode(index_parts=False).reset_index(drop=True)

    split_rows = []
    sindex = streets.sindex

    for idx, row in streets.iterrows():
        line = row.geometry

        if line is None or line.is_empty:
            split_rows.append(row.copy())
            continue

        # Candidate neighbours from the spatial index (excluding self).
        candidate_idxs = [i for i in sindex.intersection(line.bounds) if i != idx]

        intersection_points = []
        for cand_idx in candidate_idxs:
            inter = line.intersection(streets.geometry.iloc[cand_idx])
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                intersection_points.append(inter)
            elif inter.geom_type == "MultiPoint":
                intersection_points.extend(inter.geoms)
            elif inter.geom_type == "GeometryCollection":
                intersection_points.extend(
                    g for g in inter.geoms if g.geom_type == "Point"
                )

        # De-duplicate points (8-decimal key ~ 1mm at CDMX latitudes).
        seen, unique_pts = set(), []
        for pt in intersection_points:
            key = (round(pt.x, 8), round(pt.y, 8))
            if key not in seen:
                seen.add(key)
                unique_pts.append(pt)

        # Do not split at the line's own endpoints.
        start, end = Point(line.coords[0]), Point(line.coords[-1])
        split_pts = [
            pt for pt in unique_pts
            if not pt.equals(start) and not pt.equals(end)
        ]

        if split_pts:
            try:
                parts = list(split(line, MultiPoint(split_pts)).geoms)
            except Exception:
                # shapely can fail on degenerate topologies; keep the line whole.
                parts = [line]
        else:
            parts = [line]

        for part_num, part in enumerate(parts, start=1):
            new_row = row.copy()
            new_row.geometry = part
            new_row[id_col] = f"{row[id_col]}_{part_num:04d}"
            split_rows.append(new_row)

    return gpd.GeoDataFrame(split_rows, crs=streets.crs)


def split_gdf_by_length(
    gdf: gpd.GeoDataFrame, max_length: float, id_col: str
) -> gpd.GeoDataFrame:
    """Split LineString geometries longer than ``max_length`` into multiple rows.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame (lengths measured in the CRS units of ``gdf``).
    max_length : float
        Maximum segment length.
    id_col : str
        Column containing the identifier to suffix per part.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        base_id = row[id_col]

        if geom is None or geom.is_empty or geom.length <= max_length:
            rows.append(row.copy())
            continue

        n_parts = math.ceil(geom.length / max_length)
        for i in range(n_parts):
            start_dist = i * max_length
            end_dist = min((i + 1) * max_length, geom.length)
            new_row = row.copy()
            new_row.geometry = substring(geom, start_dist, end_dist)
            new_row[id_col] = f"{base_id}_{i + 1}"
            rows.append(new_row)

    return gpd.GeoDataFrame(rows, crs=gdf.crs)


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.utils.prod import init_logger

    logger = init_logger()

    STREETS_SHP_PATH = "./data/spatial/pd/09e.shp"
    OUT_PATH = "./data/proc/pd/09e_corner_split.shp"

    FEAT_COLS = [
        "POBTOT", "POBFEM", "POBMAS", "P_0A2", "P_3YMAS", "P_15YMAS",
        "P_60YMAS", "REL_H_M", "POB0_14", "POB15_64", "POB65_MAS",
        "PEA", "PE_INAC", "POCUPADA", "PDESOCUP", "GRAPROES",
        "PSINDER", "PDER_SS", "TVIVHAB", "TVIVPARHAB", "PROM_OCUP",
        "OCUPVIVPAR", "VPH_S_ELEC", "VPH_AGUAFV", "VPH_NODREN",
        "VPH_INTER", "VPH_PC", "VPH_CEL",
    ]

    logger.info("Starting street splitting...")

    try:
        streets_gdf = (
            gpd.read_file(STREETS_SHP_PATH)
            .assign(
                STREET_ID=lambda df: (
                    df["CVEGEO"]
                    + df["CVE_ENT"]
                    + df["CVE_MUN"]
                    + df["CVE_LOC"]
                    + df["CVEVIAL"]
                    + df["CVESEG"]
                )
            )
            .drop(
                columns=[
                    "CVEGEO", "CVE_ENT", "CVE_MUN",
                    "CVE_LOC", "CVEVIAL", "CVESEG",
                ]
            )
        )

        # Split streets at intersections
        streets_split = (
            split_streets_at_intersections(streets_gdf, id_col="STREET_ID")
            .rename(columns={"STREET_ID": "CVEGEO"})
        )

        # Load block geometries
        mzn_gdf = gpd.read_parquet(
            "./data/raw/cdmx_manzana_09_2020.parquet"
        ).drop(columns=["CVEGEO"])

        # Spatial join
        merged = (
            streets_split.sjoin_nearest(
                mzn_gdf.to_crs(streets_split.crs),
                how="left",
            )
            .drop(columns=["index_right"])
        )

        # Fill selected feature columns with their medians
        merged[FEAT_COLS] = merged[FEAT_COLS].fillna(
            merged[FEAT_COLS].median(numeric_only=True)
        )

        # Fill remaining numeric columns
        numeric_cols = merged.select_dtypes(include="number").columns
        merged[numeric_cols] = merged[numeric_cols].fillna(
            merged[numeric_cols].median()
        )

        gpd.GeoDataFrame(
            merged,
            geometry="geometry",
            crs=streets_gdf.crs,
        ).to_file(OUT_PATH, driver="ESRI Shapefile")

        logger.info(f"Split streets saved to: {OUT_PATH}")

    except Exception as e:
        logger.exception(
            f"Error splitting streets into corner-to-corner segments: {e}"
        )

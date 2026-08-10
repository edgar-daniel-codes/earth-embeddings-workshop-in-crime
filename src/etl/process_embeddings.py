### Summer Internship - Earth Embeddings
### ETL - Process embedding files
### By Edgar Daniel


"""

Streaming conversion of massive multi-band EPSG:4326 GeoTIFFs (AlphaEarth
embeddings) into Hive-partitioned Parquet datasets, using sliding windows
so memory stays bounded regardless of raster size.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pqds
import rasterio
from rasterio.windows import Window


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


def tif_to_parquet_partition(
    tif_path: str | Path,
    out_dir: str | Path,
    out_name: str | None = None,
    partition_cols: Sequence[str] | None = None,
    col_names: Sequence[str] | str | None = None,
    col_values: Sequence | None = None,
    col_name: Sequence[str] | str | None = None,
    col_value: Sequence | None = None,
    index_col: str = "index",
    chunk_size: int = 2048,
) -> str | Path:
    """Memory-efficient streaming conversion of massive multi-band EPSG:4326
    GeoTIFFs to Hive-partitioned Parquet files using sliding windows.

    ``col_names``/``col_values`` append constant columns (year, municipal id,
    ...) that also become Hive partitions.
    """

    # Backward compatibility & setup
    # Manual col partition and constant value assignation

    if col_names is None and col_name is not None:
        col_names = col_name
    if col_values is None and col_value is not None:
        col_values = col_value
    if col_names is None:
        col_names = []
    elif isinstance(col_names, str):
        col_names = [col_names]
    if not isinstance(col_values, (list, tuple)):
        col_values = [col_values]
    if len(col_names) != len(col_values):
        raise ValueError("col_names and col_values must have the same length.")

    partition_cols = list(partition_cols or [])
    out_name = out_name or Path(tif_path).stem
    os.makedirs(out_dir, exist_ok=True)

    partitions = list(col_names) + partition_cols

    # Streaming windows engine

    with rasterio.open(tif_path) as src:
        # Strict sanity check to confirm the file is in fact EPSG:4326.
        if src.crs is None:
            raise ValueError(f"{tif_path} has no CRS; expected EPSG:4326.")
        if src.crs.to_string() != "EPSG:4326":
            raise ValueError(
                f"Expected EPSG:4326, but the input raster is "
                f"{src.crs.to_string()}."
            )

        nodata = src.nodata if src.nodata is not None else np.nan
        labels = [d if d else f"band_{i + 1}"
                  for i, d in enumerate(src.descriptions)]

        width = src.width
        height = src.height
        global_transform = src.transform

        # Generate structural window blocks over the entire raster grid
        for y_idx in range(0, height, chunk_size):
            for x_idx in range(0, width, chunk_size):

                w_width = min(chunk_size, width - x_idx)
                w_height = min(chunk_size, height - y_idx)
                window = Window(x_idx, y_idx, w_width, w_height)

                data_block = src.read(window=window)

                # Check background/nodata pixels using the primary array
                if np.isnan(nodata):
                    mask = ~np.isnan(data_block[0])
                else:
                    mask = data_block[0] != nodata

                if not np.any(mask):
                    continue

                rel_rr, rel_cc = np.where(mask)
                abs_rr = rel_rr + y_idx
                abs_cc = rel_cc + x_idx

                # Pixel-center coordinates directly in EPSG:4326 (lat/lon)
                lons, lats = rasterio.transform.xy(
                    global_transform, abs_rr, abs_cc, offset="center")

                cols = {
                    "lon": np.array(lons, dtype="float64"),
                    "lat": np.array(lats, dtype="float64"),
                }

                # Collect every embedding band for the valid masked elements
                for i, label in enumerate(labels):
                    cols[label] = data_block[i][rel_rr, rel_cc].astype("float32")

                # Append constant custom fields (year, municipal ID, etc.)
                n_points = len(lons)
                for c, v in zip(col_names, col_values):
                    cols[c] = np.full(n_points, v)

                chunk_table = pa.table(cols)

                # A unique file template per window index guarantees chunks
                # accumulate incrementally without wiping each other out.
                chunk_filename_template = f"chunk_{y_idx}_{x_idx}_{{i}}.parquet"

                # Append chunk cleanly down to the partitioned directory tree
                pqds.write_dataset(
                    chunk_table,
                    out_dir,
                    format="parquet",
                    partitioning=partitions,
                    partitioning_flavor="hive",
                    basename_template=chunk_filename_template,
                    existing_data_behavior="overwrite_or_ignore",
                )

                del data_block, mask, cols, chunk_table

    return out_dir


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.utils.prod import init_logger

    MUN_LIST = [i for i in range(2, 18)]
    YEAR_LIST = [i for i in range(2022, 2026)]
    TIF_DIR_PATH = "./data/raw/alpha_earth_cdmx_{}/mun_{}_{}.tif"
    EMBEDDING_PATH = "./data/proc/embeddings/alpha_earth/cdmx/"
    KEY_COL = "CVEGEO"
    PARTITION_COL = "year"

    logger = init_logger()

    logger.info("Starting Concatenation process for embeddings. ")
    logger.info("Start file reading loop.")

    # Merge all TIF files into a same dataset per year and region/city
    for year in YEAR_LIST:
        for mun in MUN_LIST:
            try:
                tif_to_parquet_partition(
                    TIF_DIR_PATH.format(year, str(mun).zfill(3), year),
                    EMBEDDING_PATH,
                    out_name=f"mun_{str(mun).zfill(3)}",
                    col_name=["year", "CVE_MUN"],
                    col_value=[year, f"{mun}".zfill(3)],
                )
                logger.info(
                    f"File saved on {EMBEDDING_PATH} for partition {[year, mun]}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to process embedding values for year {year}, "
                    f"mun {mun}. {e}"
                )

        logger.info("File processed for all districts")
    logger.info("File processed for all years")

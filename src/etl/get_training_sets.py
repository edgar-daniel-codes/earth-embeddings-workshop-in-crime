### Summer Internship - Earth Embeddings
### ETL - Batch process training datasets
### By Edgar Daniel


"""

Build the supervised training sets: join yearly event counts (aggregated to
street segments) with the mean, L2-normalised AlphaEarth embedding of each
segment, producing one parquet with ``y_ind`` / ``y_cnt`` targets per
CVEGEO-year.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `python -m src.etl.get_training_sets` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import normalize

from src.etl.assign_point_to_geometry import read_point_data
from src.utils.prod import init_logger


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":
    GEOMETRY_FILE_PATH = "./data/proc/fgr_dbs/asaltos_by_streets_year_{}.geojson"
    EMBEDDING_FILE_PATH = "./data/proc/embeddings/alpha_earth/cdmx/year={}/"
    OUTPUT_PATH = "./data/proc/training_sets/cdmx_asaltos.parquet"
    YEARS_LIST = [2022, 2023, 2024]

    feat_cols = [f"A{i:02d}" for i in range(64)]

    logger = init_logger()
    logger.info("Starting process for training set creation.")

    files_list = []

    for year in YEARS_LIST:
        try:
            geom_gdf = gpd.read_file(GEOMETRY_FILE_PATH.format(year))

            gdf = read_point_data(
                EMBEDDING_FILE_PATH.format(year),
                lat_col="lat",
                lon_col="lon",
                file_format="parquet",
            )

            joined = geom_gdf.sjoin_nearest(gdf.to_crs(geom_gdf.crs), how="left")

            gdf_train = (
                joined[feat_cols + ["CVEGEO"]]
                .groupby("CVEGEO", as_index=False)
                .mean()
                .pipe(
                    lambda df: df.assign(
                        **dict(
                            zip(
                                feat_cols,
                                normalize(df[feat_cols], norm="l2", axis=1).T,
                            )
                        )
                    )
                )
                .merge(
                    geom_gdf,
                    on="CVEGEO",
                    how="left",
                )
                .assign(
                    y_ind=lambda df: (df["cnt"] > 0).astype(int),
                    y_cnt=lambda df: df["cnt"],
                    CVE_MUN=lambda df: df["CVEGEO"].astype(str)
                                         .str[2:5].str.zfill(3),
                    year=year,
                )
                .pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry",
                                                  crs=geom_gdf.crs))
            )

            files_list.append(gdf_train)
            logger.info(f"Completed processing for year {year}.")

        except Exception:
            logger.exception(f"Processing failed for year {year}.")

    logger.info("Finished processing all years.")

    if not files_list:
        logger.error("No training data was generated. Nothing to save.")
    else:
        try:
            output_gdf = gpd.GeoDataFrame(
                pd.concat(files_list, ignore_index=True),
                geometry="geometry",
                crs=files_list[0].crs,
            )

            # Segments with no embedding coverage get a zero vector.
            output_gdf[feat_cols] = output_gdf[feat_cols].fillna(0)

            Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
            output_gdf.to_parquet(OUTPUT_PATH)
            logger.info(f"Training dataset saved to {OUTPUT_PATH}")

        except Exception:
            logger.exception(f"Error saving output file to {OUTPUT_PATH}")

### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Showcase driver: run the full supervised benchmark (classification,
regression and one tuning study) on the real CDMX training set, then
persist the champion (best holdout-scoring) classifier and regressor for
later inference via ``src.supervised.predict``.

    python -m src.supervised.model_single_run
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import sys
from pathlib import Path

# Allow both `python -m src.supervised.model_single_run` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import pandas as pd

from src.supervised.common import REPO_ROOT
from src.supervised.predict import fit_and_save_champion
from src.supervised.run_classification import run_classification
from src.supervised.run_regression import run_regression
from src.supervised.run_tuning import tune_model

TRAINING_SET = REPO_ROOT / "data" / "proc" / "training_sets" / "cdmx_asaltos.parquet"

# New/unseen points are only ever run through the embeddings extraction, not
# through the socioeconomic geometry join, so the deployment champion is
# fit and later scored on embeddings-only columns.
EMBEDDING_COLUMNS = [f"A{i:02d}" for i in range(64)]
OTHER_COLUMNS = [
    "POBTOT", "POBFEM", "POBMAS", "P_0A2", "P_3YMAS", "P_15YMAS",
    "P_60YMAS", "REL_H_M", "POB0_14", "POB15_64", "POB65_MAS",
    "PEA", "PE_INAC", "POCUPADA", "PDESOCUP", "GRAPROES",
    "PSINDER", "PDER_SS", "TVIVHAB", "TVIVPARHAB", "PROM_OCUP",
    "OCUPVIVPAR", "VPH_S_ELEC", "VPH_AGUAFV", "VPH_NODREN",
    "VPH_INTER", "VPH_PC", "VPH_CEL",
]
FEATURE_COLUMNS = EMBEDDING_COLUMNS + OTHER_COLUMNS

### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    gdf = gpd.read_parquet(TRAINING_SET)

    X = gdf[FEATURE_COLUMNS]
    y_ind = gdf["y_ind"]
    y_cnt = gdf["y_cnt"]

    # Benchmark every model across feature sets (embeddings / other / combined).
    res_clf = run_classification(X, y_ind, gdf)   # configs from config/*.yaml
    res_reg = run_regression(X, y_cnt, gdf)
    study = tune_model("clf_xgb", X, y_ind=y_ind, subclasses_df=gdf,
                       feature_set="combined")

    # Pick each task's embeddings-only winner and persist it as the
    # deployment champion (models/champion/{clf,reg}_*.joblib + champion.yaml).
    X_emb = X[EMBEDDING_COLUMNS]
    clf_champion = fit_and_save_champion(
        "classification", X_emb, y_ind, res_clf["metrics"], EMBEDDING_COLUMNS)
    reg_champion = fit_and_save_champion(
        "regression", X_emb, y_cnt, res_reg["metrics"], EMBEDDING_COLUMNS)

    print(f"[model_single_run] clasificador campeón: {clf_champion['label']} "
          f"({clf_champion['metric']}={clf_champion['value']:.4f})")
    print(f"[model_single_run] regresor campeón: {reg_champion['label']} "
          f"({reg_champion['metric']}={reg_champion['value']:.4f})")

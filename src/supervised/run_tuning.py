### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Optuna tuning driver, one model and one feature set at a time — each
benchmark scenario (embeddings / other / combined) gets its own optimum. The
search space lives in the model's YAML (tuning: section); CV is group-aware;
the holdout never enters.

    python -m src.supervised.run_tuning clf_xgb            # set: embeddings
    python -m src.supervised.run_tuning reg_nn combined    # also: other

Or call tune_model(model_name, X, ..., feature_set=...) with your own data.
Writes best params (mergeable back into the config) + the trials table.
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


import sys
from pathlib import Path

# Allow both `python -m src.supervised.run_tuning` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.supervised.common import OUTPUT_DIR, load_yaml
from src.supervised.cv import split_holdout
from src.supervised.features import feature_sets
from src.supervised.models import clf_logreg, clf_nn, clf_xgb, reg_nn, reg_xgb
from src.supervised.tuning import best_params, tune

REGISTRY = {  # name: (module, config file, target: "ind" | "cnt")
    "clf_xgb": (clf_xgb, "clf_xgb.yaml", "ind"),
    "clf_nn": (clf_nn, "clf_nn.yaml", "ind"),
    "clf_logreg": (clf_logreg, "clf_logreg.yaml", "ind"),
    "reg_xgb": (reg_xgb, "reg_xgb.yaml", "cnt"),
    "reg_nn": (reg_nn, "reg_nn.yaml", "cnt"),
}

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

def tune_model(model_name, X, y_cnt=None, y_ind=None, subclasses_df=None,
               feature_set="embeddings", data_cfg=None,
               out_dir=OUTPUT_DIR / "tuning"):
    module, cfg_file, target = REGISTRY[model_name]
    model_cfg = load_yaml(cfg_file)
    data_cfg = data_cfg or load_yaml("data.yaml")
    y = y_ind if target == "ind" else y_cnt
    if y is None:
        raise ValueError(
            f"{model_name} needs y_{target}= to be provided.")

    X_set = feature_sets(X, data_cfg["features"], sets=[feature_set])[feature_set]
    print(f"[tune] {model_name} en '{feature_set}' "
          f"({X_set.shape[1]} features)", flush=True)

    # Tune on the dev partition only; the holdout stays untouched.
    (X_dev, y_dev, sub_dev), _ = split_holdout(X_set, y, subclasses_df,
                                               data_cfg["holdout"])
    study = tune(module, X_dev, y_dev, sub_dev, model_cfg, data_cfg["cv"],
                 seed=data_cfg["cv"].get("seed", 69))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{feature_set}"
    with open(out_dir / f"{stem}_best.yaml", "w") as f:
        yaml.safe_dump({"model": model_name, "feature_set": feature_set,
                        "best_value": float(study.best_value),
                        "metric": model_cfg["tuning"]["metric"],
                        "params": best_params(model_cfg, study)}, f)
    study.trials_dataframe().to_csv(out_dir / f"{stem}_trials.csv", index=False)
    return study

### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


if __name__ == "__main__":
    from src.supervised.example_data import make_example_data

    name = sys.argv[1] if len(sys.argv) > 1 else "clf_xgb"
    fset = sys.argv[2] if len(sys.argv) > 2 else "embeddings"
    X, y_cnt, y_ind, subclasses_df = make_example_data()
    study = tune_model(name, X, y_cnt=y_cnt, y_ind=y_ind,
                       subclasses_df=subclasses_df, feature_set=fset)
    print(f"{name}/{fset}: mejor {study.best_value:.4f} con {study.best_params}")

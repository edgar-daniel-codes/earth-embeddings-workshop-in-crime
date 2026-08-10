### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Discrete-count workflow: XGBoost (Poisson / NB / hurdle) vs NN count head
(Poisson / NB / ZIP), benchmarked across feature sets — embeddings only,
other columns of X, and both combined. Group-aware CV on the dev partition,
final evaluation on the column-defined holdout. Per-set distribution-fit
figures land in out_dir/<feature_set>/, plus one cross-set summary.

Call:
    run_regression(X, y_cnt, subclasses_df)   # X = embeddings + extras
or execute as a script for the synthetic end-to-end example:
    python -m src.supervised.run_regression
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


import sys
from pathlib import Path

# Allow both `python -m src.supervised.run_regression` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.supervised import viz_esp as viz
from src.supervised.common import OUTPUT_DIR, load_yaml
from src.supervised.cv import cross_validate, split_holdout
from src.supervised.features import feature_sets
from src.supervised.metrics import count_metrics
from src.supervised.models import reg_nn, reg_xgb

MODULES = {"xgb_count": reg_xgb, "nn_count": reg_nn}
CONFIG_FILES = {"xgb_count": "reg_xgb.yaml", "nn_count": "reg_nn.yaml"}


### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def _run_one_set(set_name, X, y_cnt, subclasses_df, configs, data_cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (X_dev, y_dev, sub_dev), (X_out, y_out, _) = split_holdout(
        X, y_cnt, subclasses_df, data_cfg["holdout"])
    print(f"[run:{set_name}] {X.shape[1]} features - dev {len(X_dev)} filas / "
          f"holdout {len(X_out)} filas", flush=True)

    rows, out_mus = [], {}
    for name, cfg in configs.items():
        mod, params = MODULES[name], cfg.get("params", {})
        label = f"{name}[{params.get('family', '?')}]"
        print(f"\n=== {set_name}/{label}: CV por grupos de "
              f"{data_cfg['cv']['n_splits']} folds ===", flush=True)
        oof, fold_id = cross_validate(mod.fit, mod.predict_mean,
                                      X_dev, y_dev, sub_dev,
                                      data_cfg["cv"], params,
                                      label=f"{set_name}/{label}")
        seen = fold_id >= 0
        rows.append({"feature_set": set_name, "model": label, "eval": "cv_oof",
                     **count_metrics(y_dev[seen], oof[seen])})

        print(f"=== {set_name}/{label}: reajuste sobre todo dev + "
              f"puntuación del holdout ===", flush=True)
        model = mod.fit(X_dev, y_dev, params)
        mu = mod.predict_mean(model, X_out)
        p0 = mod.predict_zero_prob(model, X_out)
        rows.append({"feature_set": set_name, "model": label, "eval": "holdout",
                     **count_metrics(y_out, mu, pred_zero_prob=p0)})
        out_mus[label] = mu

    viz.plot_count_fit(y_out, out_mus, out_dir / "count_fit_holdout.png")
    viz.plot_count_calibration(y_out, out_mus,
                               out_dir / "count_calibration_holdout.png")
    return rows, out_mus, y_out


def run_regression(X, y_cnt, subclasses_df, configs=None, data_cfg=None,
                   out_dir=OUTPUT_DIR / "regression"):
    data_cfg = data_cfg or load_yaml("data.yaml")
    configs = configs or {k: load_yaml(v) for k, v in CONFIG_FILES.items()}
    out_dir = Path(out_dir)

    rows, mus, y_out = [], {}, None
    for set_name, X_s in feature_sets(X, data_cfg["features"]).items():
        print(f"\n##### conjunto de features: {set_name} #####", flush=True)
        r, m, y_out = _run_one_set(set_name, X_s, y_cnt, subclasses_df,
                                   configs, data_cfg, out_dir / set_name)
        rows += r
        mus[set_name] = m

    # Cross-set view: per model, distribution fit and calibration with one
    # series per feature set (holdout rows are identical across sets).
    for label in next(iter(mus.values())):
        by_set = {s: m[label] for s, m in mus.items() if label in m}
        viz.plot_count_fit(
            y_out, by_set, out_dir / f"count_fit_feature_sets_{label}.png",
            title=f"Distribución de conteos — conjuntos de features ({label})")
        viz.plot_count_calibration(
            y_out, by_set, out_dir / f"count_calibration_feature_sets_{label}.png",
            title=f"Calibración de conteos — conjuntos de features ({label})")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(viz.unique_path(out_dir / "metrics.csv"), index=False)
    viz.plot_metric_summary(metrics_df[metrics_df["eval"] == "holdout"],
                            ["poisson_deviance", "mae", "spearman"],
                            out_dir / "summary_feature_sets.png")
    viz.plot_metric_summary(metrics_df[metrics_df["eval"] == "cv_oof"],
                            ["poisson_deviance", "mae", "spearman"],
                            out_dir / "summary_feature_sets_cv.png",
                            title="Comparativa de conjuntos de features "
                                  "(CV fuera de fold)")
    return {"metrics": metrics_df, "holdout_means": mus}


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


if __name__ == "__main__":
    from src.supervised.example_data import make_example_data

    X, y_cnt, y_ind, subclasses_df = make_example_data()
    res = run_regression(X, y_cnt, subclasses_df)
    print(res["metrics"].round(4).to_string(index=False))

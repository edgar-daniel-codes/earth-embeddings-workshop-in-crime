### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel


"""Classification workflow: XGBoost vs NN head vs logistic regression,
benchmarked across feature sets — embedding columns only, all other columns
of X, and both combined. Group-aware CV on the dev partition, final scoring
on the column-defined holdout. Per-set comparison figures (ROC, PR, threshold
scenarios) land in out_dir/<feature_set>/, plus one cross-set summary.

Call with your own data:
    run_classification(X, y_ind, subclasses_df)   # X = embeddings + extras
or execute as a script for the synthetic end-to-end example:
    python -m src.supervised.run_classification
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


import sys
from pathlib import Path

# Allow both `python -m src.supervised.run_classification` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.supervised import viz as viz_en
from src.supervised import viz_esp
from src.supervised.common import OUTPUT_DIR, load_yaml
from src.supervised.cv import cross_validate, split_holdout
from src.supervised.features import feature_sets
from src.supervised.metrics import classification_metrics, threshold_scenarios
from src.supervised.models import clf_logreg, clf_nn, clf_xgb

MODULES = {"xgboost": clf_xgb, "nn_head": clf_nn, "logreg": clf_logreg}
CONFIG_FILES = {"xgboost": "clf_xgb.yaml", "nn_head": "clf_nn.yaml",
                "logreg": "clf_logreg.yaml"}

# Rendered text only: "es" routes through viz_esp, which also appends the
# ``_esp`` suffix to every output filename, so the two languages never
# overwrite each other's figures.
VIZ = {"en": viz_en, "es": viz_esp}
TITLES = {
    "en": {"roc_sets": "ROC — feature sets ({name})",
           "pr_sets": "Precision-Recall — feature sets ({name})",
           "summary_cv": "Feature-set benchmark (out-of-fold CV)"},
    "es": {"roc_sets": "ROC — conjuntos de features ({name})",
           "pr_sets": "Precisión-Recall — conjuntos de features ({name})",
           "summary_cv": "Comparativa de conjuntos de features "
                         "(CV fuera de fold)"},
}


### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def _run_one_set(set_name, X, y_ind, subclasses_df, configs, data_cfg, out_dir,
                 viz=viz_en):
    out_dir.mkdir(parents=True, exist_ok=True)
    (X_dev, y_dev, sub_dev), (X_out, y_out, _) = split_holdout(
        X, y_ind, subclasses_df, data_cfg["holdout"])
    print(f"[run:{set_name}] {X.shape[1]} features - dev {len(X_dev)} filas / "
          f"holdout {len(X_out)} filas", flush=True)

    rows, out_scores, scenarios = [], {}, {}
    for name, cfg in configs.items():
        mod, params = MODULES[name], cfg.get("params", {})
        print(f"\n=== {set_name}/{name}: CV por grupos de "
              f"{data_cfg['cv']['n_splits']} folds ===", flush=True)
        oof, fold_id = cross_validate(mod.fit, mod.predict_score,
                                      X_dev, y_dev, sub_dev,
                                      data_cfg["cv"], params,
                                      label=f"{set_name}/{name}")
        seen = fold_id >= 0
        rows.append({"feature_set": set_name, "model": name, "eval": "cv_oof",
                     **classification_metrics(y_dev[seen], oof[seen])})

        print(f"=== {set_name}/{name}: reajuste sobre todo dev + "
              f"puntuación del holdout ===", flush=True)
        model = mod.fit(X_dev, y_dev, params)          # refit on full dev
        s = mod.predict_score(model, X_out)
        rows.append({"feature_set": set_name, "model": name, "eval": "holdout",
                     **classification_metrics(y_out, s)})
        out_scores[name] = s
        scenarios[name] = threshold_scenarios(y_out, s)
        scenarios[name].to_csv(viz.unique_path(out_dir / f"thresholds_{name}.csv"),
                               index=False)

    viz.plot_roc(out_scores, y_out, out_dir / "roc_holdout.png")
    viz.plot_pr(out_scores, y_out, out_dir / "pr_holdout.png")
    viz.plot_calibration(out_scores, y_out, out_dir / "calibration_holdout.png")
    viz.plot_gain_lift(out_scores, y_out, out_dir / "gain_lift_holdout.png")
    viz.plot_score_dist(out_scores, y_out, out_dir / "score_dist_holdout.png")
    viz.plot_threshold_scenarios(scenarios,
                                 ["precision", "recall", "f1", "alert_rate"],
                                 out_dir / "threshold_scenarios.png")
    return rows, out_scores, scenarios, y_out


def run_classification(X, y_ind, subclasses_df, configs=None, data_cfg=None,
                       out_dir=OUTPUT_DIR / "classification", lang="en"):
    data_cfg = data_cfg or load_yaml("data.yaml")
    configs = configs or {k: load_yaml(v) for k, v in CONFIG_FILES.items()}
    out_dir = Path(out_dir)
    viz, titles = VIZ[lang], TITLES[lang]

    rows, scores, scenarios, y_out = [], {}, {}, None
    for set_name, X_s in feature_sets(X, data_cfg["features"]).items():
        print(f"\n##### conjunto de features: {set_name} #####", flush=True)
        r, sc, sn, y_out = _run_one_set(set_name, X_s, y_ind, subclasses_df,
                                        configs, data_cfg, out_dir / set_name,
                                        viz)
        rows += r
        scores[set_name], scenarios[set_name] = sc, sn

    # Cross-set view: for each model, one curve per feature set (the holdout
    # rows are the same across sets, so curves share y_out).
    for name in configs:
        by_set = {s: sc[name] for s, sc in scores.items() if name in sc}
        viz.plot_roc(by_set, y_out, out_dir / f"roc_feature_sets_{name}.png",
                     title=titles["roc_sets"].format(name=name))
        viz.plot_pr(by_set, y_out, out_dir / f"pr_feature_sets_{name}.png",
                    title=titles["pr_sets"].format(name=name))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(viz.unique_path(out_dir / "metrics.csv"), index=False)
    viz.plot_metric_summary(metrics_df[metrics_df["eval"] == "holdout"],
                            ["roc_auc", "pr_auc", "f1"],
                            out_dir / "summary_feature_sets.png")
    viz.plot_metric_summary(metrics_df[metrics_df["eval"] == "cv_oof"],
                            ["roc_auc", "pr_auc", "f1"],
                            out_dir / "summary_feature_sets_cv.png",
                            title=titles["summary_cv"])
    return {"metrics": metrics_df, "holdout_scores": scores,
            "scenarios": scenarios}


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    from src.supervised.example_data import make_example_data

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("en", "es"), default="en")
    args = ap.parse_args()

    X, y_cnt, y_ind, subclasses_df = make_example_data()
    res = run_classification(X, y_ind, subclasses_df, lang=args.lang)
    print(res["metrics"].round(4).to_string(index=False))

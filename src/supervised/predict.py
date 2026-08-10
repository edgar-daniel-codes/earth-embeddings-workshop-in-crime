### Summer Internship - Earth Embeddings
### Model - Supervised Learning
### By Edgar Daniel

"""Champion model persistence and inference.

``model_single_run.py`` benchmarks every classifier/regressor across
feature sets and calls :func:`fit_and_save_champion` here to persist the
single best-scoring model per task (holdout metric, embeddings-only
feature set — new/unseen points have not been through the socioeconomic
geometry join, so deployment must use exactly the columns available then).

This module owns the on-disk champion contract (``models/champion/``):
one fitted-model artifact per task plus a ``champion.yaml`` recording
which model won, its metric value, and the exact feature columns it was
fit on. :func:`predict` is the single function-call entry point for
scoring new data — it needs nothing but the new rows themselves.

    from src.supervised.predict import predict
    preds = predict(new_df)   # new_df: DataFrame with the A00..A63 columns

Or run as a script:
    python -m src.supervised.predict path/to/new_points.parquet
    python -m src.supervised.predict          # synthetic showcase, no args
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Allow both `python -m src.supervised.predict` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
import yaml

from src.supervised.common import OUTPUT_DIR, load_yaml
from src.supervised.run_classification import CONFIG_FILES as CLF_CONFIG_FILES
from src.supervised.run_classification import MODULES as CLF_MODULES
from src.supervised.run_regression import CONFIG_FILES as REG_CONFIG_FILES
from src.supervised.run_regression import MODULES as REG_MODULES

CHAMPION_DIR = OUTPUT_DIR / "champion"
CHAMPION_META_FILE = "champion.yaml"

# Selection metric per task: (column, direction). Matches each model's own
# tuning objective (average_precision / poisson_deviance), so "best" here
# means the same thing it means during hyperparameter search.
_SELECTION_METRIC = {
    "classification": ("pr_auc", "max"),
    "regression": ("poisson_deviance", "min"),
}
_TASK_MODULES = {"classification": CLF_MODULES, "regression": REG_MODULES}
_TASK_CONFIG_FILES = {"classification": CLF_CONFIG_FILES,
                      "regression": REG_CONFIG_FILES}
_NN_MODEL_NAMES = {"nn_head", "nn_count"}   # torch.nn.Module contracts

# run_regression labels its "model" column "<module_key>[<family>]" (e.g.
# "xgb_count[poisson]") since one module fits several distributional
# families; classification labels are already bare module keys.
_LABEL_RE = re.compile(r"^(?P<name>[^\[]+)(?:\[(?P<family>[^\]]+)\])?$")


### -------------------------------------------------------------------------------
### Selection ------------------------------------------------------------------

def _parse_model_label(label: str) -> tuple[str, dict]:
    """Split a metrics ``model`` label into ``(module_key, extra_params)``."""
    m = _LABEL_RE.match(label)
    if not m:
        return label, {}
    extra = {"family": m.group("family")} if m.group("family") else {}
    return m.group("name"), extra


def pick_best_model(
    metrics_df: pd.DataFrame,
    task: str,
    feature_set: str = "embeddings",
    eval_stage: str = "holdout",
) -> tuple[str, str, float]:
    """Return ``(model_label, metric_name, metric_value)`` for the best
    ``eval_stage`` score on ``task``.

    ``metrics_df`` is the frame written by ``run_classification`` /
    ``run_regression`` (columns: feature_set, model, eval, <metrics...>).
    """
    metric, direction = _SELECTION_METRIC[task]
    rows = metrics_df.query("eval == @eval_stage and feature_set == @feature_set")
    if rows.empty:
        raise ValueError(
            f"No {eval_stage!r} rows for feature_set={feature_set!r} in "
            f"metrics_df; available feature sets: "
            f"{sorted(metrics_df['feature_set'].unique())}"
        )
    idx = rows[metric].idxmax() if direction == "max" else rows[metric].idxmin()
    best = rows.loc[idx]
    print(f"[predict] mejor modelo de {task} en '{feature_set}': "
          f"{best['model']} ({metric}={best[metric]:.4f})", flush=True)
    return best["model"], metric, float(best[metric])


### -------------------------------------------------------------------------------
### Artifact I/O ------------------------------------------------------------------

def _artifact_path(out_dir: Path, task: str, module_key: str) -> Path:
    role = "clf" if task == "classification" else "reg"
    suffix = ".pt" if module_key in _NN_MODEL_NAMES else ".joblib"
    return out_dir / f"{role}_{module_key}{suffix}"


def _save_artifact(model, path: Path, module_key: str) -> None:
    if module_key in _NN_MODEL_NAMES:
        import torch
        torch.save(model, path)
    else:
        joblib.dump(model, path)


def _load_artifact(path: Path, module_key: str):
    if module_key in _NN_MODEL_NAMES:
        import torch
        return torch.load(path, weights_only=False)
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)


def _read_meta(out_dir: Path) -> dict:
    path = out_dir / CHAMPION_META_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_meta(out_dir: Path, meta: dict) -> None:
    with open(out_dir / CHAMPION_META_FILE, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


### -------------------------------------------------------------------------------
### Champion save / load ------------------------------------------------------

def fit_and_save_champion(
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_df: pd.DataFrame,
    feature_columns: list[str],
    feature_set: str = "embeddings",
    out_dir: str | Path = CHAMPION_DIR,
    params: Optional[dict] = None,
) -> dict:
    """Pick the winning ``task`` model, refit it on all of ``X`` and persist
    it as the deployment champion (artifact + a ``champion.yaml`` entry).

    ``X`` must already be restricted to ``feature_columns`` (embeddings-only
    by default) — training and future inference must share exactly the same
    columns. Refitting uses the model's default YAML params unless
    ``params`` overrides them; if you've folded a ``run_tuning.py`` best-run
    back into ``config/*.yaml``, the champion picks that up automatically.
    Returns the metadata dict written for this task.
    """
    if task not in _TASK_MODULES:
        raise ValueError(
            f"task must be 'classification' or 'regression', got {task!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label, metric, value = pick_best_model(metrics_df, task, feature_set)
    module_key, extra_params = _parse_model_label(label)
    module = _TASK_MODULES[task][module_key]
    model_params = params or {
        **load_yaml(_TASK_CONFIG_FILES[task][module_key]).get("params", {}),
        **extra_params,
    }

    print(f"[predict] reajustando el campeón {label} con {len(X):,} filas "
          f"({len(feature_columns)} features de '{feature_set}')...", flush=True)
    model = module.fit(X[feature_columns], y, model_params)

    path = _artifact_path(out_dir, task, module_key)
    _save_artifact(model, path, module_key)

    meta = _read_meta(out_dir)
    meta[task] = {
        "model": module_key,
        "label": label,
        "feature_set": feature_set,
        "metric": metric,
        "value": value,
        "feature_columns": list(feature_columns),
        "artifact": path.name,
    }
    _write_meta(out_dir, meta)
    print(f"[predict] campeón guardado {task}/{label} -> {path}", flush=True)
    return meta[task]


def load_champion(task: str, out_dir: str | Path = CHAMPION_DIR):
    """Load the persisted champion for ``task``.

    Returns ``(module_key, fitted_model, metadata)``.
    """
    out_dir = Path(out_dir)
    meta = _read_meta(out_dir)
    if task not in meta:
        raise FileNotFoundError(
            f"No champion saved for task={task!r} under {out_dir}. "
            f"Run model_single_run.py (or fit_and_save_champion) first."
        )
    info = meta[task]
    model = _load_artifact(out_dir / info["artifact"], info["model"])
    return info["model"], model, info


### -------------------------------------------------------------------------------
### Inference -----------------------------------------------------------------

def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix} (use .parquet/.csv)")


def predict(
    new_data: pd.DataFrame | str | Path,
    out_dir: str | Path = CHAMPION_DIR,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score new rows with the persisted champion classifier and regressor.

    ``new_data`` is a DataFrame (or a path to one, .parquet/.csv) that must
    contain every feature column the champions were trained on — recorded
    in ``champion.yaml`` at save time, so no column list needs to be passed
    in here. Returns a frame indexed like ``new_data`` with ``y_ind_proba``,
    ``y_ind_pred``, ``y_cnt_pred`` and ``y_cnt_zero_prob``.
    """
    new_df = new_data if isinstance(new_data, pd.DataFrame) else _read_table(new_data)

    clf_key, clf_model, clf_info = load_champion("classification", out_dir)
    reg_key, reg_model, reg_info = load_champion("regression", out_dir)

    needed = sorted(set(clf_info["feature_columns"]) | set(reg_info["feature_columns"]))
    missing = [c for c in needed if c not in new_df.columns]
    if missing:
        raise ValueError(f"new_data is missing feature columns: {missing}")

    clf_module, reg_module = CLF_MODULES[clf_key], REG_MODULES[reg_key]
    X_clf = new_df[clf_info["feature_columns"]]
    X_reg = new_df[reg_info["feature_columns"]]

    p_ind = clf_module.predict_score(clf_model, X_clf)
    mu_cnt = reg_module.predict_mean(reg_model, X_reg)
    p_zero = reg_module.predict_zero_prob(reg_model, X_reg)

    return pd.DataFrame(
        {
            "y_ind_proba": p_ind,
            "y_ind_pred": (p_ind >= threshold).astype(int),
            "y_cnt_pred": mu_cnt,
            "y_cnt_zero_prob": p_zero,
        },
        index=new_df.index,
    )


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Real usage: point at a parquet/csv of new rows (e.g. a fresh
        # AlphaEarth embeddings extract) carrying the champion's feature
        # columns (A00..A63 by default).
        preds = predict(sys.argv[1])
        print(preds.round(4).to_string())
    else:
        # No path given: synthetic showcase — train + save champions, then
        # predict on freshly sampled synthetic rows, all in one process.
        from src.supervised.example_data import make_example_data
        from src.supervised.run_classification import run_classification
        from src.supervised.run_regression import run_regression

        EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
        X, y_cnt, y_ind, subclasses_df = make_example_data()
        X_emb = X[EMBEDDING_COLS]

        clf_bench = run_classification(X, y_ind, subclasses_df)
        reg_bench = run_regression(X, y_cnt, subclasses_df)

        fit_and_save_champion("classification", X_emb, y_ind,
                              clf_bench["metrics"], EMBEDDING_COLS)
        fit_and_save_champion("regression", X_emb, y_cnt,
                              reg_bench["metrics"], EMBEDDING_COLS)

        new_X, _, _, _ = make_example_data(n=10, seed=7)
        preds = predict(new_X[EMBEDDING_COLS])
        print(preds.round(4).to_string())

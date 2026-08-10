### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Column-aware splitting: a final out-of-sample holdout defined on subclass
columns (e.g. year=[2024], CVE_MUN=[2,17,4]) plus group-aware k-fold CV where
no (year, CVE_MUN) cell straddles folds."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def holdout_mask(subclasses_df, holdout):
    """Boolean mask of rows reserved as out-of-sample controls.
    holdout: {"mode": "union"|"intersection", <col>: [values], ...}
    Values are compared as strings so 2024 matches "2024" (but not zero-padded
    codes: "002" needs "002" in the config)."""
    mode = holdout.get("mode", "union")
    missing = [c for c in holdout if c != "mode"
               and c not in subclasses_df.columns]
    if missing:
        raise ValueError(
            f"holdout spec references missing column(s) {missing}; "
            f"available: {list(subclasses_df.columns)}")
    masks = [subclasses_df[c].astype(str).isin([str(x) for x in v]).values
             for c, v in holdout.items() if c != "mode"]
    if not masks:
        raise ValueError("holdout spec has no column entries besides 'mode'.")
    combine = np.logical_or if mode == "union" else np.logical_and
    return combine.reduce(masks)


def split_holdout(X, y, subclasses_df, holdout):
    """-> (X_dev, y_dev, sub_dev), (X_out, y_out, sub_out)"""
    m = holdout_mask(subclasses_df, holdout)
    if not m.any() or m.all():
        found = {c: sorted(subclasses_df[c].astype(str).unique())[:20]
                 for c in holdout if c != "mode"}
        raise ValueError(
            f"holdout selects {int(m.sum())}/{len(m)} rows - the spec "
            f"{ {k: v for k, v in holdout.items() if k != 'mode'} } does not "
            f"partition the data. Values present: {found}. Adjust "
            f"config/data.yaml: holdout (or pass data_cfg) to match.")
    take = lambda df, keep: df.loc[keep].reset_index(drop=True)
    y = pd.Series(np.asarray(y))
    return (
        (take(X, ~m), take(y, ~m), take(subclasses_df, ~m)),
        (take(X, m), take(y, m), take(subclasses_df, m)),
    )


def group_kfold_indices(subclasses_df, group_cols, n_splits=5):
    """Yield (train_idx, val_idx) with folds grouped by the joint key of
    group_cols, so subclass cells never leak across folds."""
    groups = subclasses_df[group_cols].astype(str).agg("_".join, axis=1)
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(np.zeros(len(groups)), groups=groups)


def cross_validate(fit_fn, predict_fn, X, y, subclasses_df, cv_cfg, params,
                   label=""):
    """Group-aware CV. Returns out-of-fold predictions aligned with y and the
    fold assignment (-1 if a row was never validated). Prints per-fold
    progress; label tags the lines (e.g. the model name)."""
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    tag = f"[cv{':' + label if label else ''}]"
    n = cv_cfg["n_splits"]
    for k, (tr, va) in enumerate(group_kfold_indices(
            subclasses_df, cv_cfg["group_cols"], n)):
        t0 = time.time()
        print(f"{tag} fold {k + 1}/{n}: ajuste con {len(tr)}, "
              f"validación con {len(va)} ...", flush=True)
        model = fit_fn(X.iloc[tr], y[tr], params)
        oof[va] = predict_fn(model, X.iloc[va])
        fold_id[va] = k
        print(f"{tag} fold {k + 1}/{n} terminado en {time.time() - t0:.1f}s",
              flush=True)
    return oof, fold_id

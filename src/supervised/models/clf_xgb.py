### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""XGBoost binary classifier. Contract: fit(X, y, params) / predict_score."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import xgboost as xgb

DEFAULTS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight="auto", tree_method="hist",
                eval_metric="aucpr", n_jobs=-1)

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------

def fit(X, y, params=None):
    p = {**DEFAULTS, **(params or {})}
    verbose = p.pop("verbose", 50)   # rounds between train-metric logs; falsy = silent
    y = np.asarray(y)
    if p.get("scale_pos_weight") == "auto":
        p["scale_pos_weight"] = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.XGBClassifier(**p)
    if verbose:
        model.fit(X, y, eval_set=[(X, y)], verbose=int(verbose))
    else:
        model.fit(X, y)
    return model


def predict_score(model, X):
    return model.predict_proba(X)[:, 1]

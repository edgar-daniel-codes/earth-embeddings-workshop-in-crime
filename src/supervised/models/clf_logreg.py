### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Logistic regression baseline. Contract: fit(X, y, params) / predict_score."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULTS = dict(C=1.0, penalty="l2", class_weight="balanced", max_iter=2000)

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def fit(X, y, params=None):
    p = {**DEFAULTS, **(params or {})}
    solver = "liblinear" if p.get("penalty") == "l1" else "lbfgs"
    model = make_pipeline(StandardScaler(), LogisticRegression(solver=solver, **p))
    model.fit(X, y)
    return model


def predict_score(model, X):
    return model.predict_proba(X)[:, 1]

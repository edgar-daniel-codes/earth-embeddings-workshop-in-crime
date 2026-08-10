### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""XGBoost for discrete counts. Families:
  poisson  — native count:poisson objective
  negbin   — NB2 custom objective with fixed overdispersion alpha
  hurdle   — zero-inflated two-part: P(y>0) classifier x Poisson on positives
             (mean approximated as p * mu_pos; truncation not corrected)
Contract: fit(X, y, params) / predict_mean / predict_zero_prob."""


### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import xgboost as xgb

DEFAULTS = dict(family="poisson", alpha=1.0, num_boost_round=300, max_depth=4,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=1.0)
_TREE = ("max_depth", "learning_rate", "subsample", "colsample_bytree",
         "min_child_weight", "reg_lambda")

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def _nb2_objective(alpha):
    """NB2 NLL on raw score F = log(mu)."""
    def obj(pred, dtrain):
        y, mu = dtrain.get_label(), np.exp(pred)
        grad = (mu - y) / (1 + alpha * mu)
        hess = mu * (1 + alpha * y) / (1 + alpha * mu) ** 2
        return grad, hess
    return obj


def _train(X, y, p, objective, custom=None, base_score=None, part="count"):
    tree_p = {k: p[k] for k in _TREE}
    if not custom:
        tree_p["objective"] = objective
    if base_score is not None:
        tree_p["base_score"] = base_score
    verbose = p.get("verbose", 50)   # rounds between train-metric logs
    d = xgb.DMatrix(X, label=y)
    if verbose:
        print(f"[reg_xgb:{p['family']}] entrenando la parte {part} "
              f"({p['num_boost_round']} rondas, {d.num_row()} filas)", flush=True)
    return xgb.train(tree_p, d, num_boost_round=p["num_boost_round"],
                     obj=custom, evals=[(d, "train")] if verbose else [],
                     verbose_eval=int(verbose) if verbose else False)


def fit(X, y, params=None):
    p = {**DEFAULTS, **(params or {})}
    y = np.asarray(y, dtype=float)
    fam = p["family"]
    if fam == "poisson":
        return {"family": fam, "count": _train(X, y, p, "count:poisson")}
    if fam == "negbin":
        booster = _train(X, y, p, None, custom=_nb2_objective(p["alpha"]),
                         base_score=np.log(y.mean() + 1e-6))
        return {"family": fam, "count": booster}
    if fam == "hurdle":
        pos = y > 0
        if not pos.any():
            # All-zero targets leave the count part with no training rows.
            raise ValueError(
                "hurdle family needs at least one positive count in y.")
        zero_part = _train(X, pos.astype(float), p, "binary:logistic",
                           part="zero")
        count_part = _train(X.loc[pos], y[pos], p, "count:poisson")
        return {"family": fam, "zero": zero_part, "count": count_part}
    raise ValueError(f"unknown family: {fam}")


def predict_mean(model, X):
    d = xgb.DMatrix(X)
    if model["family"] == "poisson":
        return model["count"].predict(d)
    if model["family"] == "negbin":
        return np.exp(model["count"].predict(d, output_margin=True))
    return model["zero"].predict(d) * model["count"].predict(d)


def predict_zero_prob(model, X):
    if model["family"] == "hurdle":
        return 1 - model["zero"].predict(xgb.DMatrix(X))
    return np.exp(-predict_mean(model, X))   # Poisson proxy

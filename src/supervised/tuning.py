### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Optuna hyperparameter tuning, generic over any model module that follows
the fit/predict contract. Search spaces come from the model's YAML."""


### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score

from .cv import cross_validate
from .metrics import poisson_deviance

METRICS = {  # name: (fn(y, pred), direction)
    "average_precision": (average_precision_score, "maximize"),
    "roc_auc": (roc_auc_score, "maximize"),
    "poisson_deviance": (poisson_deviance, "minimize"),
}


### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


def suggest_params(trial, space):
    out = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                          log=spec.get("log", False))
        elif t == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                            log=spec.get("log", False))
        else:  # categorical (lists encoded as strings for optuna storage)
            choices = [tuple(c) if isinstance(c, list) else c for c in spec["choices"]]
            v = trial.suggest_categorical(name, choices)
            out[name] = list(v) if isinstance(v, tuple) else v
    return out


def tune(module, X, y, subclasses_df, model_cfg, cv_cfg, seed=69):
    """Group-aware CV objective on out-of-fold predictions. Returns the study;
    best params = {**model_cfg['params'], **study.best_params}-style merge."""
    tcfg = model_cfg["tuning"]
    metric_fn, direction = METRICS[tcfg["metric"]]
    predict = getattr(module, "predict_score", None) or module.predict_mean

    def objective(trial):
        params = {**model_cfg.get("params", {}), **suggest_params(trial, tcfg["space"])}
        params["verbose"] = False   # silence per-epoch/round logs during search
        oof, fold_id = cross_validate(module.fit, predict, X, y,
                                      subclasses_df, cv_cfg, params,
                                      label=f"trial {trial.number}")
        seen = fold_id >= 0
        return metric_fn(np.asarray(y)[seen], oof[seen])

    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=tcfg["n_trials"])
    return study


def best_params(model_cfg, study):
    merged = {**model_cfg.get("params", {}), **study.best_params}
    return {k: list(v) if isinstance(v, tuple) else v for k, v in merged.items()}

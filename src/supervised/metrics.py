### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Metric suites: full classification report (imbalance-aware), threshold
scenario table, and discrete-count regression metrics."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def classification_metrics(y_true, y_score, threshold=0.5):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_hat = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "log_loss": log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_true, y_score),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
        "mcc": matthews_corrcoef(y_true, y_hat),
        "prevalence": y_true.mean(),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def threshold_scenarios(y_true, y_score, thresholds=None):
    """Scenario table: operating metrics at each candidate threshold."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, y_score, threshold=t)
        m["alert_rate"] = (np.asarray(y_score) >= t).mean()
        m["fpr"] = m["fp"] / max(m["fp"] + m["tn"], 1)
        rows.append(m)
    return pd.DataFrame(rows)


def poisson_deviance(y_true, mu):
    y_true, mu = np.asarray(y_true, float), np.clip(np.asarray(mu), 1e-9, None)
    term = np.where(y_true > 0, y_true * np.log(y_true / mu), 0.0)
    return float(2 * np.mean(term - (y_true - mu)))


def count_metrics(y_true, mu, pred_zero_prob=None):
    """mu: predicted mean counts. pred_zero_prob: optional P(y=0) from the
    model, to check zero-mass calibration (else exp(-mu) Poisson proxy)."""
    y_true, mu = np.asarray(y_true, float), np.asarray(mu, float)
    p0 = np.exp(-np.clip(mu, 0, None)) if pred_zero_prob is None else np.asarray(pred_zero_prob)
    return {
        "mae": float(np.mean(np.abs(y_true - mu))),
        "rmse": float(np.sqrt(np.mean((y_true - mu) ** 2))),
        "poisson_deviance": poisson_deviance(y_true, mu),
        "spearman": float(spearmanr(y_true, mu).statistic),
        "mean_obs": float(y_true.mean()),
        "mean_pred": float(mu.mean()),
        "zero_rate_obs": float((y_true == 0).mean()),
        "zero_rate_pred": float(p0.mean()),
    }

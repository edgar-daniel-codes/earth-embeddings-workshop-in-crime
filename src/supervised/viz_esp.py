### Summer Internship - Earth Embeddings
### Model - Aprendizaje supervisado, visuales (versión en español)
### By Edgar Daniel

"""Visuales comparativos guardados a disco: ROC, precisión-recall, escenarios
de umbral y ajuste de la distribución de conteos, con todos los textos en
español. Los colores categóricos siguen la paleta compartida de la casa en
orden fijo — un color por modelo, nunca reciclado.

Copia en español de ``src.supervised.viz``: la lógica es idéntica; a cada
figura se le añade el sufijo ``_esp`` al nombre de archivo para no
sobrescribir las versiones en inglés."""


### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------


from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.utils.style import DEFAULT as PALETTE

# House palette (single source: src.utils.style).
INK, MUTED, GRID, BASE = (PALETTE.ink, PALETTE.muted, PALETTE.grid,
                          PALETTE.gray)
SURFACE = PALETTE.background


### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def _series_colors(named) -> list[str]:
    """Un color de orden fijo por serie."""
    return PALETTE.categorical(len(named))


def _ax(title, xlabel, ylabel, figsize=(6.5, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(alpha=0.9, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    return fig, ax


def unique_path(path):
    """Nunca sobrescribe: si la ruta existe, la versiona (name_v2.ext, name_v3...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem, i = path.stem, 1
    while path.exists():
        i += 1
        path = path.with_name(f"{stem}_v{i}{path.suffix}")
    return path


def _esp_path(path):
    """Añade el sufijo ``_esp`` al nombre del archivo si aún no lo tiene, para
    que las figuras en español no sobrescriban las versiones en inglés."""
    path = Path(path)
    if path.stem.endswith("_esp"):
        return path
    return path.with_name(f"{path.stem}_esp{path.suffix}")


def _save(fig, path):
    path = unique_path(_esp_path(path))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc(scores, y_true, path, title="ROC — comparación de modelos"):
    """scores: {nombre_serie: y_score} sobre un mismo y_true. Las series pueden
    ser modelos (vista por conjunto) o conjuntos de features (vista por modelo)."""
    fig, ax = _ax(title, "Tasa de falsos positivos", "Tasa de verdaderos positivos")
    for c, (name, s) in zip(_series_colors(scores), scores.items()):
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = roc_auc_score(y_true, s)
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=BASE, linewidth=1, linestyle="--")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def plot_pr(scores, y_true, path, title="Precisión-Recall — comparación de modelos"):
    fig, ax = _ax(title, "Recall", "Precisión")
    prev = float(np.mean(y_true))
    for c, (name, s) in zip(_series_colors(scores), scores.items()):
        prec, rec, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)
        ax.plot(rec, prec, color=c, linewidth=2, label=f"{name} (AP {ap:.3f})")
    ax.axhline(prev, color=BASE, linewidth=1, linestyle="--")
    ax.annotate(f"prevalencia {prev:.3f}", (0.02, prev), textcoords="offset points",
                xytext=(0, 4), color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def plot_calibration(scores, y_true, path, n_bins=10,
                     title="Calibración — comparación de modelos"):
    """Curvas de fiabilidad: bins por cuantiles de la probabilidad predicha
    contra la tasa observada de positivos, una línea por modelo."""
    from sklearn.calibration import calibration_curve
    fig, ax = _ax(title, "Probabilidad media predicha",
                  "Tasa observada de positivos")
    for c, (name, s) in zip(_series_colors(scores), scores.items()):
        frac, mean_pred = calibration_curve(y_true, s, n_bins=n_bins,
                                            strategy="quantile")
        ax.plot(mean_pred, frac, color=c, linewidth=2, marker="o",
                markersize=4, label=name)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color=BASE, linewidth=1, linestyle="--")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def plot_gain_lift(scores, y_true, path):
    """Ganancia acumulada (proporción de positivos capturados vs proporción de
    la población marcada, ordenada por puntaje) y lift sobre el azar, una línea
    por modelo."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (ttl, yl) in zip(axes, [("Ganancia acumulada",
                                     "proporción de positivos capturados"),
                                    ("Lift", "lift sobre el azar")]):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.9, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(ttl, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("proporción de la población marcada", color=MUTED, fontsize=9)
        ax.set_ylabel(yl, color=MUTED, fontsize=9)
    for c, (name, s) in zip(_series_colors(scores), scores.items()):
        order = np.argsort(-np.asarray(s))
        gain = np.cumsum(y_true[order]) / max(y_true.sum(), 1)
        frac = np.arange(1, len(y_true) + 1) / len(y_true)
        axes[0].plot(frac, gain, color=c, linewidth=2, label=name)
        axes[1].plot(frac, gain / frac, color=c, linewidth=2, label=name)
    axes[0].plot([0, 1], [0, 1], color=BASE, linewidth=1, linestyle="--")
    axes[1].axhline(1, color=BASE, linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Ganancia / lift — comparación de modelos", color=INK, x=0.01,
                 ha="left")
    return _save(fig, path)


def plot_score_dist(scores, y_true, path):
    """Distribuciones de puntaje por clase real, un panel por modelo — muestra
    la separación entre clases y dónde viven los umbrales útiles."""
    y_true = np.asarray(y_true)
    n = len(scores)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), facecolor=SURFACE,
                             sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 31)
    for ax, c, (name, s) in zip(axes, _series_colors(scores), scores.items()):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.9, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        s = np.asarray(s)
        ax.hist(s[y_true == 0], bins=bins, density=True, color=BASE,
                alpha=0.7, label="y=0")
        ax.hist(s[y_true == 1], bins=bins, density=True, color=c,
                alpha=0.7, label="y=1")
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("puntaje", color=MUTED, fontsize=9)
    axes[0].set_ylabel("densidad", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Distribución del puntaje por clase", color=INK, x=0.01, ha="left")
    return _save(fig, path)


def plot_threshold_scenarios(scenario_dfs, metric_cols, path):
    """scenario_dfs: {nombre_modelo: df de escenarios de umbral}. Un panel por
    métrica, con el umbral en el eje x — la vista de evaluación de escenarios."""
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metric_cols):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.9, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for c, (name, df) in zip(_series_colors(scenario_dfs), scenario_dfs.items()):
            ax.plot(df["threshold"], df[m], color=c, linewidth=2, label=name)
        ax.set_title(m, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("umbral", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Evaluación de escenarios de umbral", color=INK, x=0.01, ha="left")
    return _save(fig, path)


def plot_metric_summary(df, metric_cols, path, group_col="feature_set",
                        x_col="model",
                        title="Comparativa de conjuntos de features (holdout)"):
    """Barras agrupadas: un panel por métrica, modelos en el eje x, un color fijo
    por conjunto de features y etiquetas de valor en cada barra."""
    groups = list(dict.fromkeys(df[group_col]))
    models = list(dict.fromkeys(df[x_col]))
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(1.2 * len(models) * len(groups) + 2 * n, 4),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(models))
    width = 0.8 / len(groups)
    for ax, m in zip(axes, metric_cols):
        ax.set_facecolor(SURFACE)
        ax.grid(alpha=0.9, color=GRID, linewidth=0.6, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for j, g in enumerate(groups):
            sub = df[df[group_col] == g].set_index(x_col)[m].reindex(models)
            pos = xs + (j - (len(groups) - 1) / 2) * width
            ax.bar(pos, sub.values, width * 0.92, color=_series_colors(groups)[j], label=g)
            for x, v in zip(pos, sub.values):
                if np.isfinite(v):
                    ax.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom",
                                fontsize=7, color=INK)
        ax.set_title(m, color=INK, fontsize=10, loc="left")
        ax.set_xticks(xs, models)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, title=group_col,
                   title_fontsize=8)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    return _save(fig, path)


def plot_count_calibration(y_true, mus, path, n_bins=10,
                           title="Calibración de conteos — comparación de modelos"):
    """Conteo medio observado vs conteo medio predicho por decil de la media
    predicha, una línea por modelo; y=x es la calibración perfecta."""
    y_true = np.asarray(y_true, float)
    fig, ax = _ax(title, "Conteo medio predicho (deciles)",
                  "Conteo medio observado")
    top = 0.0
    for c, (name, mu) in zip(_series_colors(mus), mus.items()):
        mu = np.asarray(mu, float)
        qs = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(qs, mu, side="right") - 1, 0, n_bins - 1)
        mp = [mu[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
        mo = [y_true[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
        top = max(top, max(mp), max(mo))
        ax.plot(mp, mo, color=c, linewidth=2, marker="o", markersize=4,
                label=name)
    ax.plot([0, top], [0, top], color=BASE, linewidth=1, linestyle="--")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)


def plot_count_fit(y_true, mus, path, max_count=None,
                   title="Distribución de conteos — empírica vs predicha"):
    """Distribución empírica de conteos (barras) contra las distribuciones
    predichas por modelo (mezcla Poisson sobre las medias predichas, en líneas)."""
    y_true = np.asarray(y_true)
    kmax = int(max_count or min(y_true.max(), 20))
    ks = np.arange(kmax + 1)
    emp = np.array([(y_true == k).mean() for k in ks])
    fig, ax = _ax(title, "conteo", "probabilidad")
    ax.bar(ks, emp, color=GRID, edgecolor=BASE, width=0.8, label="empírica")
    from scipy.stats import poisson
    for c, (name, mu) in zip(_series_colors(mus), mus.items()):
        pred = np.array([poisson.pmf(k, np.clip(mu, 1e-9, None)).mean() for k in ks])
        ax.plot(ks, pred, color=c, linewidth=2, marker="o", markersize=4, label=name)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, path)

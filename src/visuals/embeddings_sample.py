### Summer Internship - Earth Embeddings
### Visuals - Embedding representations showcase
### By Edgar Daniel


"""

Functions that simulate and plot different visuals used to illustrate the
characteristics and use of embeddings (synthetic 2-D clusters and a 3-D
vector-space example).

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

# Allow both `python -m src.visuals.embeddings_sample` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.style import DEFAULT as PALETTE, apply_theme


### -------------------------------------------------------------------------------
### Functions ---------------------------------------------------------------------


def sample_gaussian_mixture_space(
    labels: List[str],
    point_arrays: Sequence[np.ndarray],
    noise_points: np.ndarray | None = None,
    out_dir: str | Path | None = None,
) -> plt.Figure:
    """Synthetic 2-D scatter with local clusters for the given labels."""
    apply_theme()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(PALETTE.background)
    ax.set_facecolor(PALETTE.background)

    # Background embeddings
    if noise_points is not None and len(noise_points):
        ax.scatter(
            noise_points[:, 0],
            noise_points[:, 1],
            s=12,
            color=PALETTE.muted,
            alpha=0.35,
            edgecolors="none",
            zorder=1,
        )

    # Blue-led ramp for the foreground semantic clusters (house sequential hue)
    colors = PALETTE.seq_cmap()([0.45, 0.65, 0.90])

    for data, color in zip(point_arrays, colors):
        ax.scatter(
            data[:, 0],
            data[:, 1],
            s=26,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.35,
            zorder=3,
        )

    # Labels at each cluster center
    for label, data in zip(labels, point_arrays):
        center = data.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            label.capitalize(),
            fontsize=13,
            weight="medium",
            color=PALETTE.ink,
            ha="center",
            va="center",
            zorder=5,
        )

    # Minimal style
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-5, 4.5)

    ax.set_title(
        "Example of scatter positions for embedding dimensions\n"
        "For text embedding example. (Just illustrative)",
        loc="left",
    )
    ax.set_ylabel("Embedding Dim 2")
    ax.set_xlabel("Embedding Dim 1")
    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "embeddings_ilustrative_cluster.png", dpi=200)

    plt.show()
    return fig


def plot_vector_embeddings(
    labels: List[str],
    embeddings: np.ndarray,
    out_dir: str | Path | None = None,
) -> plt.Figure:
    """Illustrative 3-D semantic embedding space: vectors from the origin
    plus dashed difference vectors between selected pairs."""
    apply_theme()

    # House colors
    accent = PALETTE.blue
    dark_gray = PALETTE.ink
    light_gray = PALETTE.grid

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Clean background
    fig.patch.set_facecolor(PALETTE.background)
    ax.set_facecolor(PALETTE.background)

    # Remove pane fills
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0
        axis.pane.fill = False
        axis.pane.set_edgecolor(light_gray)

    # Light grid
    ax.grid(True, color=light_gray, linewidth=0.6)

    # Plot vectors from origin
    for point, label in zip(embeddings, labels):
        ax.quiver(
            0, 0, 0,
            point[0], point[1], point[2],
            color=accent,
            linewidth=2.5,
            arrow_length_ratio=0.08,
        )
        ax.scatter(
            point[0], point[1], point[2],
            s=80,
            color=dark_gray,
            edgecolor="white",
            linewidth=1.2,
            zorder=10,
        )
        ax.text(
            point[0] + 0.05,
            point[1] + 0.05,
            point[2] + 0.03,
            label,
            fontsize=11,
            color=dark_gray,
            weight="bold",
        )

    # Illustrate semantic difference vectors
    pairs = [
        (0, 2),
        (0, 1),
    ]
    for i, j in pairs:
        start = embeddings[i]
        diff = embeddings[j] - start
        ax.quiver(
            start[0], start[1], start[2],
            diff[0], diff[1], diff[2],
            color=light_gray,
            linewidth=1.8,
            linestyle="--",
            arrow_length_ratio=0.08,
        )

    # Labels
    ax.set_xlabel("Embedding Dimension 1", color=dark_gray)
    ax.set_ylabel("Embedding Dimension 2", color=dark_gray)
    ax.set_zlabel("Embedding Dimension 3", color=dark_gray)
    ax.set_title(
        "Illustrative Semantic Embedding Space",
        fontsize=16,
        weight="bold",
        color=dark_gray,
        pad=18,
    )

    # View angle
    ax.view_init(elev=22, azim=-58)

    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "embeddings_ilustrative_vector.png", dpi=200)

    plt.show()
    return fig


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    # Parameters for showcase

    OUT_PATH = "./docs/figures/"
    N = 120                     # sample size per cluster
    rng = np.random.default_rng(42)

    apple = rng.multivariate_normal(
        [-3.2, 1.2], [[0.18, 0.02], [0.02, 0.15]], N)
    banana = rng.multivariate_normal(
        [-1.0, -2.0], [[0.22, -0.03], [-0.03, 0.18]], N)
    car = rng.multivariate_normal(
        [3.0, 0.4], [[0.28, 0.04], [0.04, 0.20]], N)
    noise = rng.multivariate_normal(
        [0, 0], [[9.5, 0], [0, 5.5]], N * 5)

    embeddings = np.array([
        [0.80, 0.65, 0.70],   # Apple
        [2.40, 1.90, 0.50],   # Car
        [0.35, 0.90, 0.85],   # Banana
    ])
    labels = ["apple", "banana", "car"]
    point_arrays = [apple, banana, car]

    # 1) Scatter mixture of gaussians
    sample_gaussian_mixture_space(labels, point_arrays, noise, OUT_PATH)

    # 2) Vector similarity generic example
    plot_vector_embeddings(labels, embeddings, OUT_PATH)

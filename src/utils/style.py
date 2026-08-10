### Summer Internship - Earth Embeddings
### Utils - Shared visual identity (palette + matplotlib theme)
### By Edgar Daniel


"""

Single source of truth for the project's visual identity, based on the
corporate light theme defined in the EDA module.

Every plotting module draws its colors from a :class:`Palette` instance so
the whole repository reads as one system.  All values are overridable:
build a custom ``Palette(...)`` (or ``dataclasses.replace(DEFAULT, ...)``)
and pass it to ``apply_theme`` / the plot builders that accept one.

Color roles
-----------
* ``qual``       fixed-order categorical set (never cycled) for series,
                 clusters and model identities.
* ``seq_cmap``   single-hue blue ramp for magnitudes (heatmaps, densities).
* ``bad_cmap``   red ramp reserved for adverse counts (``y_cnt`` events).
* ``good``/``bad``  semantic accents (e.g. missingness OK vs problematic).

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, to_hex


### -------------------------------------------------------------------------------
### Palette ------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Parametrizable corporate light palette (all fields overridable)."""

    # Surfaces and ink
    background: str = "#FFFFFF"   # figure / axes background
    panel: str = "#F7F9FC"        # subtle raised panel
    ink: str = "#1B1F23"          # primary text
    muted: str = "#6B7280"        # secondary text / axis labels
    grid: str = "#E5E7EB"         # gridlines / borders

    # Brand hues
    navy: str = "#0F2747"         # deep corporate navy
    blue: str = "#0057B8"         # primary accent
    sky: str = "#6FA8DC"          # secondary blue
    gray: str = "#9CA3AF"         # neutral gray

    # Semantic accents
    good: str = "#2E8B57"         # muted green - favourable highlights
    bad: str = "#B42318"          # dark red - reserved for adverse highlights

    # Fixed categorical order (never cycled)
    qual: tuple[str, ...] = (
        "#0057B8", "#0F2747", "#6FA8DC", "#9CA3AF", "#2B3E63", "#4A6FA5",
    )

    # Ramp anchors: light -> dark
    seq_anchors: tuple[str, ...] = ("#F8FAFC", "#DCE8F5", "#0057B8", "#0F2747")
    bad_anchors: tuple[str, ...] = (
        "#FBF1F0", "#F1C6BF", "#D45B4B", "#B42318", "#7A1710",
    )

    # Typography / output
    font_family: str = "sans-serif"
    dpi: int = 150
    fig_format: str = "png"

    def seq_cmap(self) -> LinearSegmentedColormap:
        """Single-hue blue ramp for magnitudes."""
        return LinearSegmentedColormap.from_list("seq", list(self.seq_anchors))

    def bad_cmap(self) -> LinearSegmentedColormap:
        """Red ramp reserved for adverse counts (``y_cnt`` events)."""
        return LinearSegmentedColormap.from_list("bad_seq", list(self.bad_anchors))

    def categorical(self, n: int) -> list[str]:
        """``n`` series colors in fixed order.

        Beyond ``len(self.qual)`` (rare), extra colors are sampled from a
        navy->sky ramp instead of cycling, so no two series ever share a hex.
        """
        if n <= 0:
            return []
        if n <= len(self.qual):
            return list(self.qual[:n])
        extra_n = n - len(self.qual)
        ramp = LinearSegmentedColormap.from_list(
            "qual_ext", [self.navy, self.blue, self.sky])
        # Sample strictly inside (0, 1) to avoid repeating the anchor hexes.
        extra = [to_hex(ramp((i + 1) / (extra_n + 1))) for i in range(extra_n)]
        return list(self.qual) + extra


DEFAULT = Palette()

# Axis tick formatter shared by count-valued charts.
THOUSANDS = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")


### -------------------------------------------------------------------------------
### Theme -------------------------------------------------------------------------


def apply_theme(palette: Palette = DEFAULT) -> None:
    """Apply the minimal corporate light theme to matplotlib.

    Idempotent - safe to call multiple times (e.g. per Streamlit rerun).
    """
    plt.rcParams.update({
        "figure.facecolor": palette.background,
        "savefig.facecolor": palette.background,
        "axes.facecolor": palette.background,
        "axes.edgecolor": palette.grid,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "grid.color": palette.grid,
        "grid.linewidth": 0.6,
        "text.color": palette.ink,
        "axes.labelcolor": palette.muted,
        "axes.titlecolor": palette.ink,
        "xtick.color": palette.muted,
        "ytick.color": palette.muted,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.family": palette.font_family,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": palette.ink,
    })

### Summer Internship - Earth Embeddings
### Utils - Similarity statistics over embedding vectors
### By Edgar Daniel


"""

Pairwise cosine-similarity statistics for large embedding sets, computed
without materialising the full O(n^2) similarity matrix.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters------------------------------------------------------

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


def group_cosine_stats(
    X: pd.DataFrame | gpd.GeoDataFrame | np.ndarray,
    max_exact_pairs: int = 5_000_000,
    sample_size: int = 2_000_000,
    chunk: int = 4096,
    rng: Optional[np.random.Generator] = None,
) -> Optional[dict]:
    """Mean / median / std of all pairwise cosine similarities in ``X``.

    Mean and std are exact via the Gram-trace identity, O(n*d + d^2).
    The median is exact (chunked upper-triangle) while the pair count fits
    in ``max_exact_pairs``; beyond that it is estimated on a random sample.

    Returns ``None`` when fewer than two rows are available.
    """
    rng = rng or np.random.default_rng()

    X = np.asarray(X, dtype=np.float32)
    n, _ = X.shape

    if n < 2:
        return None

    # Normalization: zero-norm rows stay at the origin instead of dividing by 0.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = (X / norms).astype(np.float32)

    # Exact mean/std via Gram-trace identity, O(n*d + d^2) complexity.
    S = U.sum(axis=0)
    G = U.T @ U
    n_pairs = n * (n - 1) // 2
    sum_s = (S @ S - n) / 2.0
    sum_s2 = (np.sum(G * G) - n) / 2.0
    mean = sum_s / n_pairs
    var = max(sum_s2 / n_pairs - mean ** 2, 0.0)
    std = np.sqrt(var)

    # Median: exact via chunked triu if feasible, else sampled.
    if n_pairs <= max_exact_pairs:
        vals = np.empty(n_pairs, dtype=np.float32)
        pos = 0
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = U[start:end] @ U.T
            for local_i, i in enumerate(range(start, end)):
                row = block[local_i, i + 1:]
                vals[pos:pos + len(row)] = row
                pos += len(row)
        median = float(np.median(vals))
    else:
        m = min(sample_size, n_pairs)
        i_idx = rng.integers(0, n, size=m)
        j_idx = rng.integers(0, n, size=m)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        if len(i_idx) == 0:
            # Degenerate draw (tiny n with all self-pairs): fall back to mean.
            median = float(mean)
        else:
            sample_sims = np.einsum("ij,ij->i", U[i_idx], U[j_idx])
            median = float(np.median(sample_sims))

    return {"mean": float(mean), "median": median, "std": float(std),
            "n_pairs": int(n_pairs)}

### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Synthetic example inputs matching the real schema: X = embedding columns
(A00.., in [-1,1]) plus a few non-embedding covariates ("other" set), y_cnt
(zero-inflated counts), y_ind (~1:20 positives), subclasses_df (year,
CVE_MUN). Swap for your real loader; every entry point takes the same four
objects."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import pandas as pd

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def make_example_data(n=6000, n_features=64, prefix="A", seed=69):
    rng = np.random.default_rng(seed)
    E = np.tanh(rng.normal(0, 0.6, size=(n, n_features)))
    cols = [f"{prefix}{i:02d}" for i in range(n_features)]

    subclasses_df = pd.DataFrame({
        "year": rng.choice([2022, 2023, 2024], n),
        "CVE_MUN": rng.choice(np.arange(2, 18), n),
    })

    # A sparse linear signal drives both the zero-inflation and the count rate.
    w = np.zeros(n_features)
    w[rng.choice(n_features, 8, replace=False)] = rng.normal(0, 1.2, 8)
    eta = E @ w

    # Non-embedding covariates: two carry part of the same signal (noisily),
    # two are pure noise — so the embeddings/other/combined benchmark is
    # meaningful on the example data too.
    X = pd.DataFrame(E, columns=cols).assign(
        pop_density=np.exp(1.0 + 0.5 * eta + rng.normal(0, 0.8, n)),
        poi_count=rng.poisson(np.exp(0.4 * eta + 0.5)),
        street_len=rng.gamma(2.0, 1.5, n),
        dist_center=rng.uniform(0, 20, n),
    )

    pi_zero = 1 / (1 + np.exp(-(2.2 - eta)))          # ~high structural zeros
    mu = np.exp(-0.5 + 0.8 * eta)
    y_cnt = np.where(rng.random(n) < pi_zero, 0, rng.poisson(mu))
    y_cnt = pd.Series(y_cnt, name="y_cnt")
    y_ind = (y_cnt > 0).astype(int).rename("y_ind")
    return X, y_cnt, y_ind, subclasses_df

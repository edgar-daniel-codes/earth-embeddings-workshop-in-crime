### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel


"""Feature-set split for benchmarking: embedding columns (matched by the
prefix in data.yaml) vs every other column of X vs both combined."""

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def embedding_columns(X, feats_cfg):
    names = [f"{feats_cfg['prefix']}{i:02d}"
             for i in range(feats_cfg["n_features"])]
    return [c for c in names if c in X.columns]


def feature_sets(X, feats_cfg, sets=None):
    """-> ordered {set_name: X_subset}. Sets with no columns (or 'combined'
    when it would duplicate another set) are skipped with a notice."""
    emb = embedding_columns(X, feats_cfg)
    other = [c for c in X.columns if c not in emb]
    catalog = {"embeddings": emb, "other": other, "combined": emb + other}
    wanted = sets or feats_cfg.get("sets", ["embeddings", "other", "combined"])
    out = {}
    for name in wanted:
        cols = catalog[name]
        if not cols or (name == "combined" and (not emb or not other)):
            print(f"[features] se omite el conjunto '{name}' "
                  f"({'sin columnas' if not cols else 'duplica otro conjunto'})",
                  flush=True)
            continue
        out[name] = X[cols]
    return out

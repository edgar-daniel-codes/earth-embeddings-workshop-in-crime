### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel


"""NN classification head over an n-dim embedding input (PyTorch MLP).
Imbalance handled via pos_weight in BCEWithLogitsLoss.
Contract: fit(X, y, params) / predict_score."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

DEFAULTS = dict(hidden_dims=[64, 32], dropout=0.2, lr=1e-3, weight_decay=1e-4,
                epochs=60, batch_size=256, pos_weight="auto", seed=69)

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def _mlp(n_in, hidden_dims, dropout, n_out=1):
    layers, d = [], n_in
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, n_out)]
    return nn.Sequential(*layers)


def _standardize(net, Xt, train=False):
    if train:  # store train stats so non-embedding covariate scales are tamed
        net.x_mean, net.x_std = Xt.mean(0), Xt.std(0).clamp(min=1e-6)
    return (Xt - net.x_mean) / net.x_std


def fit(X, y, params=None):
    p = {**DEFAULTS, **(params or {})}
    torch.manual_seed(p["seed"])
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    yt = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)

    pw = p["pos_weight"]
    if pw == "auto":
        pw = float((yt == 0).sum() / max(float((yt == 1).sum()), 1.0))
    net = _mlp(Xt.shape[1], list(p["hidden_dims"]), p["dropout"])
    Xt = _standardize(net, Xt, train=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=p["batch_size"], shuffle=True)
    verbose = p.get("verbose", True)
    every = max(1, p["epochs"] // 10)   # ~10 progress lines per fit
    net.train()
    for epoch in range(p["epochs"]):
        total, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        if verbose and ((epoch + 1) % every == 0 or epoch == 0):
            print(f"[clf_nn] época {epoch + 1}/{p['epochs']} - "
                  f"pérdida {total / seen:.4f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def predict_score(model, X):
    Xt = _standardize(model, torch.tensor(np.asarray(X, dtype=np.float32)))
    return torch.sigmoid(model(Xt)).squeeze(1).numpy()

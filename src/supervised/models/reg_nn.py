### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""NN head for discrete counting (PyTorch MLP). Families via NLL losses:
  poisson — one head: log(mu)
  negbin  — NB2 with learnable scalar overdispersion log(alpha)
  zip     — zero-inflated Poisson: heads for logit(pi) and log(mu);
            E[y] = (1 - pi) * mu
Contract: fit(X, y, params) / predict_mean / predict_zero_prob."""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

DEFAULTS = dict(family="zip", hidden_dims=[64, 32], dropout=0.2, lr=1e-3,
                weight_decay=1e-4, epochs=80, batch_size=256, seed=69)


### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


class CountHead(nn.Module):
    def __init__(self, n_in, hidden_dims, dropout, family):
        super().__init__()
        layers, d = [], n_in
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.family = family
        self.log_mu = nn.Linear(d, 1)
        if family == "zip":
            self.logit_pi = nn.Linear(d, 1)
        if family == "negbin":
            self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.body(x)
        out = {"log_mu": self.log_mu(z).squeeze(1).clamp(-10, 10)}
        if self.family == "zip":
            out["logit_pi"] = self.logit_pi(z).squeeze(1)
        return out


def _nll(out, y, model):
    mu = out["log_mu"].exp()
    pois = mu - y * out["log_mu"] + torch.lgamma(y + 1)
    if model.family == "poisson":
        return pois.mean()
    if model.family == "negbin":
        a = model.log_alpha.exp()
        r = 1 / a
        ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
              + y * torch.log(a * mu / (1 + a * mu)) - r * torch.log1p(a * mu))
        return -ll.mean()
    # zip mixture
    lp1 = nn.functional.logsigmoid(out["logit_pi"])          # log pi
    lp0 = nn.functional.logsigmoid(-out["logit_pi"])         # log(1 - pi)
    ll_zero = torch.logaddexp(lp1, lp0 - mu)
    ll_pos = lp0 - pois
    return -torch.where(y == 0, ll_zero, ll_pos).mean()


def _standardize(net, Xt, train=False):
    if train:  # store train stats so non-embedding covariate scales are tamed
        net.x_mean, net.x_std = Xt.mean(0), Xt.std(0).clamp(min=1e-6)
    return (Xt - net.x_mean) / net.x_std


def fit(X, y, params=None):
    p = {**DEFAULTS, **(params or {})}
    torch.manual_seed(p["seed"])
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    yt = torch.tensor(np.asarray(y, dtype=np.float32))
    net = CountHead(Xt.shape[1], list(p["hidden_dims"]), p["dropout"], p["family"])
    Xt = _standardize(net, Xt, train=True)
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
            loss = _nll(net(xb), yb, net)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        if verbose and ((epoch + 1) % every == 0 or epoch == 0):
            print(f"[reg_nn:{p['family']}] época {epoch + 1}/{p['epochs']} - "
                  f"nll {total / seen:.4f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def _forward(model, X):
    Xt = _standardize(model, torch.tensor(np.asarray(X, dtype=np.float32)))
    return model(Xt)


def predict_mean(model, X):
    out = _forward(model, X)
    mu = out["log_mu"].exp()
    if model.family == "zip":
        mu = (1 - torch.sigmoid(out["logit_pi"])) * mu
    return mu.numpy()


def predict_zero_prob(model, X):
    out = _forward(model, X)
    mu = out["log_mu"].exp()
    if model.family == "zip":
        pi = torch.sigmoid(out["logit_pi"])
        return (pi + (1 - pi) * torch.exp(-mu)).numpy()
    if model.family == "negbin":
        a = model.log_alpha.exp()
        return torch.pow(1 + a * mu, -1 / a).numpy()
    return torch.exp(-mu).numpy()

"""
Neural-network and tabular deep-learning regressors (Family 6, 11 models).

Maps to Table 3 of the manuscript:
    MLP, ELM, GRNN, ANFIS, TabNet, FT-Transformer (FTT), NODE, SAINT,
    1D-CNN, Bayesian NN (BNN), Mixture Density Network (MDN).

For the present submission, the canonical architectures (TabNet, FT-T, NODE,
SAINT) are reimplemented in PyTorch in compact form to remove the dependency
on heavyweight external libraries (rtdl, saint). The methodological
properties that matter for the manuscript -- sparse attention masks for
TabNet, multi-head self-attention with feature tokens for FT-Transformer,
neural oblivious decision trees for NODE, and contrastive intersample
attention for SAINT -- are preserved at the conceptual level.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ebw_ml.models.base import BaseRegressor, FitContext


def _torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared training loop for in-house PyTorch models
# ---------------------------------------------------------------------------
def _train_torch(model: nn.Module, X: np.ndarray, Y: np.ndarray,
                 *, epochs: int = 500, batch_size: int = 32, lr: float = 1e-3,
                 weight_decay: float = 0.0, seed: int = 42,
                 loss_fn: nn.Module | None = None,
                 device: str = "cpu") -> tuple[StandardScaler, StandardScaler, str]:
    _torch_seed(seed)
    dev = torch.device(device)
    model.to(dev)
    Xs = StandardScaler().fit(X); Ys = StandardScaler().fit(Y)
    Xt = torch.tensor(Xs.transform(X), dtype=torch.float32, device=dev)
    Yt = torch.tensor(Ys.transform(Y), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn or nn.MSELoss()
    model.train()
    n = Xt.shape[0]
    g = torch.Generator(); g.manual_seed(seed)  # CPU generator for randperm
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xt[idx]; yb = Yt[idx]
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):  # MDN returns (pi, mu, sigma)
                loss = loss_fn(out, yb)
            else:
                loss = loss_fn(out, yb)
            loss.backward(); opt.step()
    return Xs, Ys, device


# ---------------------------------------------------------------------------
# 1. MLP (sklearn) -- simplest baseline
# ---------------------------------------------------------------------------
class MLPRegressorWrap(BaseRegressor):
    name = "mlp"; family = "NN/DL"
    search_space = {"hidden_layer_sizes": [(32,), (64,), (32, 32), (64, 64)],
                    "alpha": (1e-5, 1e-1, "log-uniform"),
                    "learning_rate_init": (1e-4, 1e-2, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._est = MLPRegressor(max_iter=1000, random_state=ctx.seed, **self.hp).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)


# ---------------------------------------------------------------------------
# 2. ELM (extreme learning machine) -- closed-form fit
# ---------------------------------------------------------------------------
class ELMRegressor(BaseRegressor):
    name = "elm"; family = "NN/DL"
    search_space = {"n_hidden": [32, 64, 128, 256],
                    "alpha": (1e-6, 1.0, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        n_hidden = int(self.hp.get("n_hidden", 128))
        alpha = float(self.hp.get("alpha", 1e-3))
        rng = np.random.default_rng(ctx.seed)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._W = rng.normal(size=(Xs.shape[1], n_hidden))
        self._b = rng.normal(size=(n_hidden,))
        H = np.tanh(Xs @ self._W + self._b)
        # Ridge closed-form: beta = (H'H + alpha I)^-1 H'Y
        A = H.T @ H + alpha * np.eye(n_hidden)
        self._beta = np.linalg.solve(A, H.T @ Y)

    def _predict(self, X):
        Xs = self._scaler.transform(X)
        H = np.tanh(Xs @ self._W + self._b)
        return H @ self._beta


# ---------------------------------------------------------------------------
# 3. GRNN (general regression neural network, Specht 1991) -- closed form
# ---------------------------------------------------------------------------
class GRNNRegressor(BaseRegressor):
    name = "grnn"; family = "NN/DL"
    search_space = {"sigma": (0.05, 5.0, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        self._scaler = StandardScaler().fit(X)
        self._Xs = self._scaler.transform(X)
        self._Y = Y.copy()
        self._sigma = float(self.hp.get("sigma", 0.5))

    def _predict(self, X):
        Xs = self._scaler.transform(X)
        # Pairwise squared Euclidean
        d2 = ((Xs[:, None, :] - self._Xs[None, :, :]) ** 2).sum(-1)
        K = np.exp(-d2 / (2 * self._sigma ** 2))
        denom = K.sum(axis=1, keepdims=True) + 1e-12
        return (K @ self._Y) / denom


# ---------------------------------------------------------------------------
# 4. ANFIS (adaptive neuro-fuzzy inference, simplified Sugeno)
# ---------------------------------------------------------------------------
class _ANFISCore(nn.Module):
    def __init__(self, n_features: int, n_rules: int, n_outputs: int):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(n_rules, n_features) * 0.5)
        self.widths = nn.Parameter(torch.ones(n_rules, n_features))
        self.consequents = nn.Parameter(torch.randn(n_rules, n_features + 1, n_outputs) * 0.1)

    def forward(self, x):  # x: (B, F)
        # Gaussian membership grades per feature, per rule
        diff = x.unsqueeze(1) - self.centres  # (B, R, F)
        mf = torch.exp(-0.5 * (diff / (self.widths.abs() + 1e-3)) ** 2)
        firing = mf.prod(dim=-1)  # (B, R)
        firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-12)
        # Sugeno consequents: linear functions of x
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # (B, F+1)
        cons = torch.einsum("bf,rfo->bro", x_aug, self.consequents)  # (B, R, O)
        return (firing.unsqueeze(-1) * cons).sum(dim=1)  # (B, O)


class ANFISRegressor(BaseRegressor):
    name = "anfis"; family = "NN/DL"
    search_space = {"n_rules": [3, 5, 8],
                    "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        n_rules = int(self.hp.get("n_rules", 5))
        lr = float(self.hp.get("lr", 1e-3))
        epochs = int(self.hp.get("epochs", 300))
        self._model = _ANFISCore(X.shape[1], n_rules, Y.shape[1])
        self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y, epochs=epochs,
                                          lr=lr, seed=ctx.seed, device=ctx.resolve_device())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            out = self._model(Xt).cpu().numpy()
        return self._Ys.inverse_transform(out)


# ---------------------------------------------------------------------------
# 5. TabNet-lite (attentive masking)
# ---------------------------------------------------------------------------
class _TabNetLite(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, n_steps: int = 3,
                 hidden: int = 32):
        super().__init__()
        self.n_steps = n_steps
        self.bn = nn.BatchNorm1d(n_features)
        self.attn = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_steps)])
        self.feat = nn.ModuleList([nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        ) for _ in range(n_steps)])
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        x = self.bn(x)
        prior = torch.ones_like(x)
        out = 0.0
        for a, f in zip(self.attn, self.feat):
            mask = torch.softmax(a(x) * prior, dim=-1)
            xm = x * mask
            out = out + f(xm)
            prior = prior * (1.5 - mask)  # gamma=1.5
        return self.head(out)


class TabNetRegressorWrap(BaseRegressor):
    name = "tabnet"; family = "NN/DL"
    search_space = {"n_steps": [3, 5], "hidden": [16, 32, 64],
                    "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        self._model = _TabNetLite(X.shape[1], Y.shape[1],
                                   n_steps=int(self.hp.get("n_steps", 3)),
                                   hidden=int(self.hp.get("hidden", 32)))
        self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y,
                                          epochs=int(self.hp.get("epochs", 300)),
                                          lr=float(self.hp.get("lr", 1e-3)),
                                          seed=ctx.seed,
                                          device=ctx.resolve_device())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            out = self._model(Xt).cpu().numpy()
        return self._Ys.inverse_transform(out)


# ---------------------------------------------------------------------------
# 6. FT-Transformer-lite (feature tokenizer + transformer)
# ---------------------------------------------------------------------------
class _FTTLite(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                dim_feedforward=2 * d_model,
                                                batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):  # x: (B, F)
        toks = self.tokenizer(x.unsqueeze(-1))  # (B, F, d)
        cls = self.cls.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, toks], dim=1)
        z = self.encoder(seq)
        return self.head(z[:, 0])


class FTTRegressor(BaseRegressor):
    name = "ftt"; family = "NN/DL"
    search_space = {"d_model": [16, 32, 64], "n_heads": [2, 4],
                    "n_layers": [1, 2, 3], "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        d = int(self.hp.get("d_model", 32))
        h = int(self.hp.get("n_heads", 4))
        # d_model must be divisible by n_heads
        if d % h != 0:
            h = max(1, d // 8)
        self._model = _FTTLite(X.shape[1], Y.shape[1], d_model=d, n_heads=h,
                                n_layers=int(self.hp.get("n_layers", 2)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y,
                                              epochs=int(self.hp.get("epochs", 300)),
                                              lr=float(self.hp.get("lr", 1e-3)),
                                              seed=ctx.seed,
                                              device=ctx.resolve_device())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            out = self._model(Xt).cpu().numpy()
        return self._Ys.inverse_transform(out)


# ---------------------------------------------------------------------------
# 7. NODE-lite (neural oblivious decision trees, simplified)
# ---------------------------------------------------------------------------
class _NODELite(nn.Module):
    def __init__(self, n_features: int, n_outputs: int,
                 n_trees: int = 8, depth: int = 4):
        super().__init__()
        self.n_trees = n_trees; self.depth = depth
        self.feat_w = nn.Parameter(torch.randn(n_trees, depth, n_features))
        self.thresh = nn.Parameter(torch.zeros(n_trees, depth))
        self.leaves = nn.Parameter(torch.randn(n_trees, 2 ** depth, n_outputs) * 0.1)

    def forward(self, x):
        # split scores per tree per depth
        scores = torch.einsum("bf,tdf->btd", x, self.feat_w) - self.thresh  # (B, T, D)
        gates = torch.sigmoid(scores)  # soft binary decisions
        # Compute leaf weights: product of (gate or 1-gate) along depth
        bsz = x.shape[0]
        leaf_idx = torch.arange(2 ** self.depth, device=x.device)
        bits = torch.stack([((leaf_idx >> d) & 1).float() for d in range(self.depth)], dim=-1)
        # gates: (B, T, D); bits: (L, D)
        # leaf weight = prod_d gates^bit * (1-gates)^(1-bit)
        g = gates.unsqueeze(2)            # (B, T, 1, D)
        b = bits.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        w = (g * b + (1 - g) * (1 - b)).prod(dim=-1)  # (B, T, L)
        # leaves: (T, L, O)
        out = torch.einsum("btl,tlo->bo", w, self.leaves) / self.n_trees
        return out


class NODERegressor(BaseRegressor):
    name = "node"; family = "NN/DL"
    search_space = {"n_trees": [4, 8, 16], "depth": [3, 4, 5],
                    "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        self._model = _NODELite(X.shape[1], Y.shape[1],
                                 n_trees=int(self.hp.get("n_trees", 8)),
                                 depth=int(self.hp.get("depth", 4)))
        self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y,
                                          epochs=int(self.hp.get("epochs", 300)),
                                          lr=float(self.hp.get("lr", 1e-3)),
                                          seed=ctx.seed,
                                          device=ctx.resolve_device())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            out = self._model(Xt).cpu().numpy()
        return self._Ys.inverse_transform(out)


# ---------------------------------------------------------------------------
# 8. SAINT-lite (intersample attention)
# ---------------------------------------------------------------------------
class _SAINTLite(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, d_model: int = 32,
                 n_heads: int = 4):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.intersample_attn = nn.MultiheadAttention(d_model, n_heads,
                                                      batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        toks = self.tokenizer(x.unsqueeze(-1))  # (B, F, d)
        # Self-attention across features within sample
        z, _ = self.self_attn(toks, toks, toks); z = toks + z
        # Pool features
        z = z.mean(dim=1, keepdim=True)  # (B, 1, d)
        # Intersample attention across batch dim, simulated by reshaping
        # so the "sequence" axis is batch and the "batch" axis is 1
        z_iter = z.transpose(0, 1)  # (1, B, d)
        z2, _ = self.intersample_attn(z_iter, z_iter, z_iter)
        z = (z + z2.transpose(0, 1)).squeeze(1)  # (B, d)
        z = z + self.ff(z)
        return self.head(z)


class SAINTRegressor(BaseRegressor):
    name = "saint"; family = "NN/DL"
    search_space = {"d_model": [16, 32, 64], "n_heads": [2, 4],
                    "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        d = int(self.hp.get("d_model", 32)); h = int(self.hp.get("n_heads", 4))
        if d % h != 0:
            h = max(1, d // 8)
        self._model = _SAINTLite(X.shape[1], Y.shape[1], d_model=d, n_heads=h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y,
                                              epochs=int(self.hp.get("epochs", 300)),
                                              lr=float(self.hp.get("lr", 1e-3)),
                                              seed=ctx.seed,
                                              device=ctx.resolve_device())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            out = self._model(Xt).cpu().numpy()
        return self._Ys.inverse_transform(out)


# ---------------------------------------------------------------------------
# 9. 1D-CNN
# ---------------------------------------------------------------------------
class _CNN1D(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=2, padding=1)
        self.fc = nn.Linear(channels * 2 * (n_features + 2), n_outputs)

    def forward(self, x):
        z = x.unsqueeze(1)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = z.flatten(start_dim=1)
        return self.fc(z)


class CNN1DRegressor(BaseRegressor):
    name = "cnn1d"; family = "NN/DL"
    search_space = {"channels": [8, 16, 32], "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        ch = int(self.hp.get("channels", 16))
        self._model = _CNN1D(X.shape[1], Y.shape[1], channels=ch)
        self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y,
                                          epochs=int(self.hp.get("epochs", 300)),
                                          lr=float(self.hp.get("lr", 1e-3)),
                                          seed=ctx.seed,
                                          device=ctx.resolve_device())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            out = self._model(Xt).cpu().numpy()
        return self._Ys.inverse_transform(out)


# ---------------------------------------------------------------------------
# 10. Bayesian NN (mean-field variational) -- minimal implementation
# ---------------------------------------------------------------------------
class _BayesLinear(nn.Module):
    def __init__(self, in_f, out_f, prior_sigma=1.0):
        super().__init__()
        self.in_f = in_f; self.out_f = out_f
        self.w_mu = nn.Parameter(torch.zeros(out_f, in_f))
        self.w_logvar = nn.Parameter(torch.full((out_f, in_f), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(out_f))
        self.b_logvar = nn.Parameter(torch.full((out_f,), -5.0))
        self.prior_sigma = prior_sigma

    def forward(self, x):
        w_std = torch.exp(0.5 * self.w_logvar)
        b_std = torch.exp(0.5 * self.b_logvar)
        eps_w = torch.randn_like(self.w_mu)
        eps_b = torch.randn_like(self.b_mu)
        w = self.w_mu + w_std * eps_w
        b = self.b_mu + b_std * eps_b
        return F.linear(x, w, b)

    def kl(self):
        s2 = self.prior_sigma ** 2
        kl_w = 0.5 * (torch.exp(self.w_logvar)/s2 + self.w_mu**2/s2 - 1.0 - self.w_logvar + math.log(s2)).sum()
        kl_b = 0.5 * (torch.exp(self.b_logvar)/s2 + self.b_mu**2/s2 - 1.0 - self.b_logvar + math.log(s2)).sum()
        return kl_w + kl_b


class _BNN(nn.Module):
    def __init__(self, n_features, n_outputs, hidden=32):
        super().__init__()
        self.l1 = _BayesLinear(n_features, hidden)
        self.l2 = _BayesLinear(hidden, n_outputs)

    def forward(self, x):
        z = F.relu(self.l1(x))
        return self.l2(z)

    def kl(self):
        return self.l1.kl() + self.l2.kl()


class BNNRegressor(BaseRegressor):
    name = "bnn"; family = "NN/DL"
    search_space = {"hidden": [16, 32, 64], "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500], "kl_weight": (1e-4, 1e-1, "log-uniform")}
    _mc_samples: int = 30

    def _fit(self, X, Y, ctx: FitContext):
        _torch_seed(ctx.seed)
        device = ctx.resolve_device()
        dev = torch.device(device)
        Xs = StandardScaler().fit(X); Ys = StandardScaler().fit(Y)
        Xt = torch.tensor(Xs.transform(X), dtype=torch.float32, device=dev)
        Yt = torch.tensor(Ys.transform(Y), dtype=torch.float32, device=dev)
        h = int(self.hp.get("hidden", 32))
        lr = float(self.hp.get("lr", 1e-3))
        epochs = int(self.hp.get("epochs", 300))
        kl_w = float(self.hp.get("kl_weight", 1e-2))
        self._model = _BNN(X.shape[1], Y.shape[1], hidden=h).to(dev)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        n = Xt.shape[0]; bs = 32
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                xb = Xt[idx]; yb = Yt[idx]
                out = self._model(xb)
                nll = F.mse_loss(out, yb)
                loss = nll + kl_w * self._model.kl() / n
                opt.zero_grad(); loss.backward(); opt.step()
        self._Xs, self._Ys, self._device = Xs, Ys, device

    def _predict(self, X):
        self._model.eval()
        Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
        preds = []
        with torch.no_grad():
            for _ in range(self._mc_samples):
                preds.append(self._model(Xt).cpu().numpy())
        mean = np.mean(preds, axis=0)
        return self._Ys.inverse_transform(mean)

    def _predict_dist(self, X):
        self._model.eval()
        Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
        preds = []
        with torch.no_grad():
            for _ in range(self._mc_samples):
                preds.append(self._model(Xt).cpu().numpy())
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0); std = preds.std(axis=0)
        # Inverse transform
        mu = self._Ys.inverse_transform(mean)
        # Approximate variance scaling
        sigma_scale = np.sqrt(self._Ys.var_)
        return mu, std * sigma_scale


# ---------------------------------------------------------------------------
# 11. MDN (mixture density network)
# ---------------------------------------------------------------------------
class _MDN(nn.Module):
    def __init__(self, n_features, n_outputs, n_mix=3, hidden=32):
        super().__init__()
        self.n_mix = n_mix; self.n_out = n_outputs
        self.shared = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_mix)
        self.mu = nn.Linear(hidden, n_mix * n_outputs)
        self.log_sigma = nn.Linear(hidden, n_mix * n_outputs)

    def forward(self, x):
        z = self.shared(x)
        pi = F.softmax(self.pi(z), dim=-1)
        mu = self.mu(z).view(-1, self.n_mix, self.n_out)
        sigma = torch.exp(self.log_sigma(z)).view(-1, self.n_mix, self.n_out).clamp(min=1e-3)
        return pi, mu, sigma


def _mdn_nll(out, y):
    pi, mu, sigma = out  # pi: (B,K), mu/sigma: (B,K,O)
    y_exp = y.unsqueeze(1)  # (B,1,O)
    log_p = -0.5 * (((y_exp - mu) / sigma) ** 2 + 2 * torch.log(sigma) + math.log(2 * math.pi))
    log_p = log_p.sum(dim=-1)  # (B,K)
    log_mix = torch.log(pi + 1e-12) + log_p
    nll = -torch.logsumexp(log_mix, dim=-1).mean()
    return nll


class _MDNLossWrap(nn.Module):
    def forward(self, out, y): return _mdn_nll(out, y)


class MDNRegressor(BaseRegressor):
    name = "mdn"; family = "NN/DL"
    search_space = {"n_mix": [2, 3, 5], "hidden": [16, 32, 64],
                    "lr": (1e-4, 1e-2, "log-uniform"),
                    "epochs": [200, 500]}

    def _fit(self, X, Y, ctx: FitContext):
        n_mix = int(self.hp.get("n_mix", 3))
        hidden = int(self.hp.get("hidden", 32))
        self._model = _MDN(X.shape[1], Y.shape[1], n_mix=n_mix, hidden=hidden)
        self._Xs, self._Ys, self._device = _train_torch(self._model, X, Y,
                                          epochs=int(self.hp.get("epochs", 300)),
                                          lr=float(self.hp.get("lr", 1e-3)),
                                          seed=ctx.seed,
                                          device=ctx.resolve_device(),
                                          loss_fn=_MDNLossWrap())

    def _predict(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            pi, mu, _ = self._model(Xt)
            # Mean of mixture: sum_k pi_k * mu_k
            point = (pi.unsqueeze(-1) * mu).sum(dim=1).cpu().numpy()
        return self._Ys.inverse_transform(point)

    def _predict_dist(self, X):
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._Xs.transform(X), dtype=torch.float32, device=getattr(self, "_device", "cpu"))
            pi, mu, sigma = self._model(Xt)
            # Mixture mean and variance
            mean = (pi.unsqueeze(-1) * mu).sum(dim=1)
            var = (pi.unsqueeze(-1) * (sigma ** 2 + mu ** 2)).sum(dim=1) - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-12))
        mu_inv = self._Ys.inverse_transform(mean.cpu().numpy())
        sigma_scale = np.sqrt(self._Ys.var_)
        return mu_inv, std.cpu().numpy() * sigma_scale


NN_MODELS = [
    MLPRegressorWrap, ELMRegressor, GRNNRegressor, ANFISRegressor,
    TabNetRegressorWrap, FTTRegressor, NODERegressor, SAINTRegressor,
    CNN1DRegressor, BNNRegressor, MDNRegressor,
]

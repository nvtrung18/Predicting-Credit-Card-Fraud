"""Generic PyTorch training loop for the 5-fold DL models (notebooks 3.2 / 3.3).

Inputs are tuples of numpy arrays so the same loop serves a plain MLP (one float matrix)
and embedding models (numeric float matrix + integer category matrix).
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 12
    patience: int = 3
    grad_clip: float = 1.0
    scheduler: str = "none"          # "none" | "onecycle" | "cosine"
    pos_weight: float | None = None
    eval_batch: int = 16384


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_tensors(arrays):
    out = []
    for a in arrays:
        t = torch.from_numpy(np.ascontiguousarray(a))
        out.append(t.long() if np.issubdtype(a.dtype, np.integer) else t.float())
    return out


@torch.inference_mode()
def predict(model: nn.Module, arrays, batch: int = 16384) -> np.ndarray:
    model.eval()
    tensors = _to_tensors(arrays)
    n = len(tensors[0])
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch):
        xb = [t[s:s + batch].to(DEVICE, non_blocking=True) for t in tensors]
        out[s:s + batch] = torch.sigmoid(model(*xb).squeeze(1)).float().cpu().numpy()
    return out


def train_fold(model: nn.Module, train_arrays, y_train, valid_arrays, y_valid, cfg: TrainConfig, seed: int,
               log_prefix: str = ""):
    """Train with early stopping on validation ROC-AUC; returns (best_model, valid_pred, history)."""
    seed_everything(seed)
    model = model.to(DEVICE)
    tr_tensors = [t.to(DEVICE) for t in _to_tensors(train_arrays)]   # whole train fold fits on the GPU
    y_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).view(-1, 1).to(DEVICE)
    n = len(y_t)
    gen = torch.Generator().manual_seed(seed)
    steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size

    pw = None if cfg.pos_weight is None else torch.tensor([cfg.pos_weight], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.max_epochs,
                                                    steps_per_epoch=steps_per_epoch)
    elif cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs * steps_per_epoch)
    else:
        sched = None

    best_auc, best_state, best_epoch, stale, history = -np.inf, None, 0, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        loss_sum = 0.0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb = [t[idx] for t in tr_tensors]
            yb = y_t[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*xb), yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if sched is not None:
                sched.step()
            loss_sum += float(loss.item()) * len(idx)

        valid_pred = predict(model, valid_arrays, cfg.eval_batch)
        auc = roc_auc_score(y_valid, valid_pred)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_roc_auc": auc})
        print(f"{log_prefix}epoch {epoch:02d} | loss={loss_sum / n:.5f} | val_auc={auc:.6f} | {time.time() - t0:.1f}s",
              flush=True)
        if auc > best_auc + 1e-6:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, predict(model, valid_arrays, cfg.eval_batch), {"best_epoch": best_epoch, "best_auc": best_auc,
                                                                  "history": history}


class MLP(nn.Module):
    """Plain MLP on a single float matrix (architecture of DL task 2.1 by default)."""

    def __init__(self, in_dim: int, hidden=(128, 64), dropout=(0.20, 0.10)):
        super().__init__()
        layers, d = [], in_dim
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PeriodicEmbedding(nn.Module):
    """PLR numeric embedding (Gorishniy et al. 2022): per feature x -> [sin, cos](2*pi*c*x) -> Linear -> ReLU.

    Learns a smooth, spline-like basis per numeric column; expects roughly standardised inputs.
    """

    def __init__(self, n_features: int, n_frequencies: int = 16, sigma: float = 0.1, d_embedding: int = 16):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_frequencies, d_embedding))
        self.bias = nn.Parameter(torch.zeros(n_features, d_embedding))
        bound = 1 / np.sqrt(2 * n_frequencies)              # nn.Linear default init, per feature
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        self.out_dim = n_features * d_embedding

    def forward(self, x):                                   # x: (B, F)
        v = 2 * np.pi * self.freq[None] * x[..., None]      # (B, F, K)
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, F, 2K)
        v = torch.einsum("bfk,fkd->bfd", v, self.weight) + self.bias
        return torch.relu(v).flatten(1)


class ResNetHead(nn.Module):
    """Tabular ResNet (Gorishniy et al. 2021): Linear -> n x [BN -> Linear -> ReLU -> Dropout -> Linear -> Dropout
    + skip] -> BN -> ReLU -> Linear."""

    def __init__(self, in_dim: int, d: int = 256, n_blocks: int = 2, mult: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.BatchNorm1d(d), nn.Linear(d, d * mult), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d * mult, d), nn.Dropout(dropout)) for _ in range(n_blocks))
        self.out = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)


class EmbeddingMLP(nn.Module):
    """PLR embeddings for numeric columns + nn.Embedding per categorical column + raw binary columns -> MLP or
    ResNet head."""

    def __init__(self, n_numeric: int, n_binary: int, cat_cardinalities, hidden=(256, 128), dropout=(0.15, 0.10),
                 n_frequencies: int = 16, sigma: float = 0.1, d_num: int = 16, head: str = "mlp"):
        super().__init__()
        self.num = PeriodicEmbedding(n_numeric, n_frequencies, sigma, d_num)
        dims = [min(16, max(2, int(np.ceil(np.sqrt(k + 1))))) for k in cat_cardinalities]
        self.cats = nn.ModuleList(nn.Embedding(k + 1, d) for k, d in zip(cat_cardinalities, dims))  # index 0 = unknown
        in_dim = self.num.out_dim + n_binary + sum(dims)
        self.head = ResNetHead(in_dim, dropout=dropout[0]) if head == "resnet" else MLP(in_dim, hidden, dropout)

    def forward(self, x_num, x_bin, x_cat):
        parts = [self.num(x_num), x_bin] + [emb(x_cat[:, i]) for i, emb in enumerate(self.cats)]
        return self.head(torch.cat(parts, dim=1))


def encode_categories(train_values, *others):
    """Integer-encode columns with mapping learned on train only: 1..K for seen values, 0 for unseen.

    Each argument is a 2-D array-like (rows x columns). Returns (encoded arrays..., cardinalities).
    """
    import pandas as pd
    train_values = pd.DataFrame(train_values)
    cats = [pd.Index(sorted(train_values[c].dropna().unique())) for c in train_values.columns]

    def enc(values):
        values = pd.DataFrame(values)
        cols = [cats[i].get_indexer(values.iloc[:, i]) + 1 for i in range(len(cats))]  # -1 (unseen) -> 0
        return np.column_stack(cols).astype(np.int64)

    return (enc(train_values), *[enc(o) for o in others], [len(c) for c in cats])

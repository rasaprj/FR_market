import os
import json
import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Reproducibility
# =========================
def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Config
# =========================
@dataclass
class Config:
    xlsx_path: str = "I-V-SoC-60s.csv"
    out_dir: str = "soc_lstm_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dt: float = 1 / 60.0  # 1 minute if sampling is 60s

    n_splits: int = 5
    gap: int = 10

    # sequence settings
    seq_len: int = 50  # window length for LSTM

    batch_size: int = 256
    num_epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 40

    # LSTM hyperparams
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    soc_min: float = 0.0
    soc_max: float = 1.0

    seed: int = 42


# =========================
# Walk-forward split
# =========================
def walk_forward_splits(n: int, n_splits: int):
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    splits = []
    for k in range(1, n_splits):
        train_end = bounds[k]
        val_start = bounds[k]
        val_end = bounds[k + 1]
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return splits


# =========================
# Metrics
# =========================
def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = np.var(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse / (var + 1e-12))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "NMSE": float(nmse(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# =========================
# Feature construction
# =========================
def build_features(I: np.ndarray, V: np.ndarray, soc: np.ndarray, dt: float):
    I_amp = I / 1000.0  # mA -> A (change if your I already in A)
    Q = np.cumsum(I_amp) * dt

    dV = np.zeros_like(V)
    dV[1:] = V[1:] - V[:-1]

    X = np.column_stack([Q, V, I, dV]).astype(np.float32)
    y = soc.astype(np.float32).reshape(-1, 1)
    return X, y


# =========================
# Sequence dataset
# =========================
class SeqDataset(Dataset):
    """
    Uses the original timeline and builds sequences from indices.

    For a target at time t, input is X[t-seq_len+1 : t+1], target is y[t].
    Only indices >= seq_len-1 are valid.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

        # keep only indices that can form a full window
        self.indices = indices[indices >= (seq_len - 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = int(self.indices[i])
        x_seq = self.X[t - self.seq_len + 1 : t + 1]        # (seq_len, n_feat)
        y_t = self.y[t]                                     # (1,)
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)


# =========================
# LSTM Model
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float, soc_min: float, soc_max: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.soc_min = soc_min
        self.soc_max = soc_max

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, F)
        out, _ = self.lstm(x_seq)       # out: (B, T, H)
        last = out[:, -1, :]            # (B, H)
        yhat = self.fc(last)            # (B, 1)
        return torch.clamp(yhat, self.soc_min, self.soc_max)


# =========================
# Training one fold
# =========================
def train_one_fold(cfg: Config, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, fold_id: int):
    if cfg.gap > 0:
        val_start = int(val_idx.min())
        gap_band = np.arange(max(0, val_start - cfg.gap), val_start)
        train_idx = np.setdiff1d(train_idx, gap_band)

    train_ds = SeqDataset(X, y, train_idx, cfg.seq_len)
    val_ds = SeqDataset(X, y, val_idx, cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = LSTMRegressor(
        n_features=X.shape[1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        soc_min=cfg.soc_min,
        soc_max=cfg.soc_max,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_nmse = float("inf")
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)

            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        yhat_list = []
        ytrue_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yhat = model(xb).cpu().numpy()
                yhat_list.append(yhat)
                ytrue_list.append(yb.numpy())

        yhat_val = np.vstack(yhat_list)
        y_val = np.vstack(ytrue_list)

        metrics = compute_metrics(y_val, yhat_val)

        print(f"[Fold {fold_id}] Ep {epoch+1:03d} R2={metrics['R2']:.4f} NMSE={metrics['NMSE']:.4f}")

        if metrics["NMSE"] < best_nmse:
            best_nmse = metrics["NMSE"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_nmse


# =========================
# Predict on a block using sequences
# =========================
def predict_block(cfg: Config, model: nn.Module, X: np.ndarray, y: np.ndarray, idx: np.ndarray):
    ds = SeqDataset(X, y, idx, cfg.seq_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    model.eval()
    yhat_list = []
    ytrue_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            yhat = model(xb).cpu().numpy()
            yhat_list.append(yhat)
            ytrue_list.append(yb.numpy())

    yhat = np.vstack(yhat_list)
    ytrue = np.vstack(ytrue_list)
    return ytrue, yhat


# =========================
# Main
# =========================
def main():
    cfg = Config()
    seed_all(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.xlsx_path, header=None)
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows containing NaN values.")

    I = df.iloc[:, 0].values.astype(float)
    V = df.iloc[:, 1].values.astype(float)
    soc = df.iloc[:, 2].values.astype(float)

    print("Rows:", len(soc))
    print("Corr(SoC, V):", float(np.corrcoef(soc, V)[0, 1]))
    print("Corr(SoC, cumsum(I)):", float(np.corrcoef(soc, np.cumsum(I))[0, 1]))

    X, y = build_features(I, V, soc, cfg.dt)
    splits = walk_forward_splits(len(y), cfg.n_splits)

    all_nmse = []
    min_nmse = float("inf")

    best_state = None
    best_y = None
    best_yhat = None

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        model, fold_best_nmse = train_one_fold(cfg, X, y, train_idx, val_idx, fold_id)
        all_nmse.append(fold_best_nmse)

        y_val, yhat_val = predict_block(cfg, model, X, y, val_idx)

        if fold_best_nmse < min_nmse:
            min_nmse = fold_best_nmse
            best_state = copy.deepcopy(model.state_dict())
            best_y = y_val
            best_yhat = yhat_val

    print("\n===== CV SUMMARY (NMSE) =====")
    print("Mean NMSE:", float(np.mean(all_nmse)))
    print("Std  NMSE:", float(np.std(all_nmse)))

    if best_state is not None:
        best_model_path = os.path.join(cfg.out_dir, "best_model.pth")
        torch.save(best_state, best_model_path)
        print(f"Best model saved to {best_model_path}")

    if best_y is not None and best_yhat is not None:
        best_metrics = compute_metrics(best_y, best_yhat)

        print("\n===== BEST MODEL METRICS (REGRESSION) =====")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.6f}")

        metrics_path = os.path.join(cfg.out_dir, "best_model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"Best metrics saved to {metrics_path}")

        # Plot 1: Actual vs Predicted over samples (validation block)
        plt.figure(figsize=(10, 5))
        plt.plot(best_y, label="Actual")
        plt.plot(best_yhat, label="Predicted")
        plt.xlabel("Sample (validation block, aligned by seq_len)")
        plt.ylabel("SoC")
        plt.title("Best LSTM: Predicted vs Actual (Validation Block)")
        plt.legend()
        ts_path = os.path.join(cfg.out_dir, "best_model_vs_actual.png")
        plt.tight_layout()
        plt.savefig(ts_path)
        plt.close()
        print(f"Time-series plot saved to {ts_path}")

        # Plot 2: Scatter (25% random subset) actual (x) vs predicted (y) with diagonal
        y_true_flat = np.asarray(best_y).reshape(-1)
        y_pred_flat = np.asarray(best_yhat).reshape(-1)

        n_points = len(y_true_flat)
        subset_size = max(1, int(0.10 * n_points))
        rng = np.random.default_rng(cfg.seed)
        subset_idx = rng.choice(n_points, size=subset_size, replace=False)

        y_true_sub = y_true_flat[subset_idx]
        y_pred_sub = y_pred_flat[subset_idx]

        min_val = float(min(y_true_flat.min(), y_pred_flat.min()))
        max_val = float(max(y_true_flat.max(), y_pred_flat.max()))

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_sub, y_pred_sub, alpha=0.6)
        plt.plot([min_val, max_val], [min_val, max_val], "k--")
        plt.xlabel("Actual SoC")
        plt.ylabel("Predicted SoC")
        plt.title("Best LSTM: Actual vs Predicted (25% Sample)")
        plt.tight_layout()
        scatter_path = os.path.join(cfg.out_dir, "best_model_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"Scatter plot saved to {scatter_path}")


if __name__ == "__main__":
    main()

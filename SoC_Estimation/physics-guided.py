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
    out_dir: str = "soc_physics_guided_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dt: float = 1 / 60.0  # 1 minute, if your sampling is 60s

    n_splits: int = 5
    gap: int = 10

    batch_size: int = 256
    num_epochs: int = 300
    lr: float = 5e-3
    weight_decay: float = 0.0
    patience: int = 40

    soc_min: float = 0.0
    soc_max: float = 1.0

    seed: int = 42


# =========================
# Dataset
# =========================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =========================
# Walk-forward split
# =========================
def walk_forward_splits(n: int, n_splits: int):
    """
    Splits into sequential blocks. For k=1..n_splits-1:
      train = [0 .. bounds[k)-1]
      val   = [bounds[k] .. bounds[k+1)-1]
    """
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    splits = []
    for k in range(1, n_splits):
        train_end = bounds[k]
        val_start = bounds[k]
        val_end = bounds[k + 1]
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return splits


# =========================
# Model
# =========================
class PhysicsGuidedSoC(nn.Module):
    """
    SoC_hat = b + alpha * Q + beta * V + gamma * I + delta * dV
    then clamped to [soc_min, soc_max]
    """
    def __init__(self, soc_min: float = 0.0, soc_max: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        self.soc_min = soc_min
        self.soc_max = soc_max

    def forward(self, Q, V, I, dV):
        soc = self.bias + self.alpha * Q + self.beta * V + self.gamma * I + self.delta * dV
        return torch.clamp(soc, self.soc_min, self.soc_max)


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
    """
    Q computed by integrating current (assumes I is in mA -> convert to A).
    dV is a simple first difference.
    """
    I_amp = I / 1000.0  # mA -> A (adjust if your I is already in A)
    Q = np.cumsum(I_amp) * dt

    dV = np.zeros_like(V)
    dV[1:] = V[1:] - V[:-1]

    X = np.column_stack([Q, V, I, dV])
    y = soc.reshape(-1, 1)
    return X, y


# =========================
# Training
# =========================
def train_one_fold(cfg: Config, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, fold_id: int):
    # Optional gap band to reduce leakage near the boundary
    if cfg.gap > 0:
        val_start = int(val_idx.min())
        gap_band = np.arange(max(0, val_start - cfg.gap), val_start)
        train_idx = np.setdiff1d(train_idx, gap_band)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    model = PhysicsGuidedSoC(cfg.soc_min, cfg.soc_max).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_nmse = float("inf")
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            Q = xb[:, 0:1]
            V = xb[:, 1:2]
            Icur = xb[:, 2:3]
            dV = xb[:, 3:4]

            optimizer.zero_grad()
            yhat = model(Q, V, Icur, dV)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        yhat_list = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(cfg.device)
                Q = xb[:, 0:1]
                V = xb[:, 1:2]
                Icur = xb[:, 2:3]
                dV = xb[:, 3:4]
                yhat = model(Q, V, Icur, dV)
                yhat_list.append(yhat.cpu().numpy())

        yhat_val = np.vstack(yhat_list)
        metrics = compute_metrics(y_val, yhat_val)

        print(
            f"[Fold {fold_id}] Ep {epoch+1:03d} "
            f"R2={metrics['R2']:.4f} NMSE={metrics['NMSE']:.4f} "
            f"(alpha={model.alpha.item():.6f}, beta={model.beta.item():.6f})"
        )

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
# Main
# =========================
def main():
    cfg = Config()
    seed_all(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load data
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

    # Train across folds
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        model, fold_best_nmse = train_one_fold(cfg, X, y, train_idx, val_idx, fold_id)
        all_nmse.append(fold_best_nmse)

        # Evaluate fold model on its validation block
        X_val = X[val_idx]
        y_val = y[val_idx]

        with torch.no_grad():
            xb = torch.tensor(X_val, dtype=torch.float32).to(cfg.device)
            Q = xb[:, 0:1]
            Vb = xb[:, 1:2]
            Icur = xb[:, 2:3]
            dV = xb[:, 3:4]
            yhat = model(Q, Vb, Icur, dV).cpu().numpy()

        if fold_best_nmse < min_nmse:
            min_nmse = fold_best_nmse
            best_state = copy.deepcopy(model.state_dict())
            best_y = y_val
            best_yhat = yhat

    print("\n===== CV SUMMARY (NMSE) =====")
    print("Mean NMSE:", float(np.mean(all_nmse)))
    print("Std  NMSE:", float(np.std(all_nmse)))

    # Save best model state
    if best_state is not None:
        best_model_path = os.path.join(cfg.out_dir, "best_model.pth")
        torch.save(best_state, best_model_path)
        print(f"Best model saved to {best_model_path}")

    # Final best-model metrics + plots
    if best_y is not None and best_yhat is not None:
        best_metrics = compute_metrics(best_y, best_yhat)

        print("\n===== BEST MODEL METRICS (REGRESSION) =====")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.6f}")

        # Save metrics to JSON
        metrics_path = os.path.join(cfg.out_dir, "best_model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"Best metrics saved to {metrics_path}")

        # Plot 1: Actual vs Predicted over samples
        plt.figure(figsize=(10, 5))
        plt.plot(best_y, label="Actual")
        plt.plot(best_yhat, label="Predicted")
        plt.xlabel("Sample (validation block)")
        plt.ylabel("SoC")
        plt.title("Best Model: Predicted vs Actual (Validation Block)")
        plt.legend()
        ts_path = os.path.join(cfg.out_dir, "best_model_vs_actual.png")
        plt.tight_layout()
        plt.savefig(ts_path)
        plt.close()
        print(f"Time-series plot saved to {ts_path}")

        # Plot 2: Scatter actual (x) vs predicted (y) with diagonal
        y_true_flat = np.asarray(best_y).reshape(-1)
        y_pred_flat = np.asarray(best_yhat).reshape(-1)

        min_val = float(min(y_true_flat.min(), y_pred_flat.min()))
        max_val = float(max(y_true_flat.max(), y_pred_flat.max()))

        # Plot 2: Scatter actual (x) vs predicted (y) with diagonal (25% random subset)

        y_true_flat = np.asarray(best_y).reshape(-1)
        y_pred_flat = np.asarray(best_yhat).reshape(-1)

        # ---- Random 25% sampling ----
        n_points = len(y_true_flat)
        subset_size = int(0.10 * n_points)

        rng = np.random.default_rng(cfg.seed)  # reproducible
        subset_idx = rng.choice(n_points, size=subset_size, replace=False)

        y_true_sub = y_true_flat[subset_idx]
        y_pred_sub = y_pred_flat[subset_idx]

        # Diagonal bounds should still use full range
        min_val = float(min(y_true_flat.min(), y_pred_flat.min()))
        max_val = float(max(y_true_flat.max(), y_pred_flat.max()))

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_sub, y_pred_sub, alpha=0.6)
        plt.plot([min_val, max_val], [min_val, max_val], "k--")

        plt.xlabel("Actual SoC")
        plt.ylabel("Predicted SoC")
        plt.title("Best Model: Actual vs Predicted (25% Sample)")
        plt.tight_layout()

        scatter_path = os.path.join(cfg.out_dir, "best_model_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"Scatter plot saved to {scatter_path}")


if __name__ == "__main__":
    main()

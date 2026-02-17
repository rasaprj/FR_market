import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Your metrics ----
metrics_data = {
    "MLP": {
        "R2": 0.9280142784118652,
        "NMSE": 0.07198573648929596,
        "RMSE": 0.050602600950564014,
        "MAE": 0.03742043673992157,
    },
    "Physics Guided": {
        "R2": 0.6561609753109714,
        "NMSE": 0.34383902467974103,
        "RMSE": 0.11282473474257958,
        "MAE": 0.08816902732540971,
    },
    "GRU": {
        "R2": 0.9781760573387146,
        "NMSE": 0.021823933348059654,
        "RMSE": 0.027862245906925466,
        "MAE": 0.020102985203266144,
    },
    "LSTM": {
        "R2": 0.9850189685821533,
        "NMSE": 0.01498100720345974,
        "RMSE": 0.02308448156979593,
        "MAE": 0.01700819656252861,
    },
}

# ---- Table ----
df = pd.DataFrame(metrics_data).T  # rows=models, cols=metrics

# ---- Normalize to "higher is better" for coloring ----
df_norm = df.copy()

# R2: higher is better
df_norm["R2"] = (df_norm["R2"] - df_norm["R2"].min()) / (df_norm["R2"].max() - df_norm["R2"].min() + 1e-12)

# Errors: lower is better -> invert after min-max scaling
for col in ["NMSE", "RMSE", "MAE"]:
    scaled = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-12)
    df_norm[col] = 1.0 - scaled

# ---- Plot heatmap (blue shading) ----
plt.figure(figsize=(8, 4))
plt.imshow(df_norm.values, aspect="auto", cmap="Blues")
plt.colorbar(label="Normalized Score")
plt.xticks(range(df_norm.shape[1]), df_norm.columns)
plt.yticks(range(df_norm.shape[0]), df_norm.index)
plt.title("Model Performance Heatmap")

# Annotate with original metric values
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        # Set text color: white for MLP, GRU, LSTM; black for Physics Guided
        row_label = df.index[i]
        if row_label in ["MLP", "GRU", "LSTM"]:
            text_color = "white"
        else:
            text_color = "black"
        plt.text(j, i, f"{df.iloc[i, j]:.3f}", ha="center", va="center", color=text_color)

plt.tight_layout()
plt.savefig("model_performance_heatmap.png", dpi=200)
plt.show()

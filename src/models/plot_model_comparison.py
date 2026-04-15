"""
Plot model comparison for EDA: reads `artifacts/models_metrics.json`
and creates a histogram/bar chart comparing RMSE (and optionally MAE/R2).

Usage:
    python -m src.models.plot_model_comparison

Outputs:
    artifacts/model_comparison_rmse.png
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics(metrics: list[dict], out_path: Path):
    model_names = [m["model_name"] for m in metrics]
    rmses = np.array([m.get("rmse", float("nan")) for m in metrics])
    maes = np.array([m.get("mae", float("nan")) for m in metrics])
    r2s = np.array([m.get("r2", float("nan")) for m in metrics])

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_rmse = ax.bar(x - width/2, rmses, width, label="RMSE", color="#4C72B0")
    bars_mae = ax.bar(x + width/2, maes, width, label="MAE", color="#55A868")

    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)

    # Secondary axis for R2
    ax2 = ax.twinx()
    ax2.plot(x, r2s, color="#DD8452", marker="o", label="R2")
    ax2.set_ylabel("R2")
    ax2.set_ylim(0, 1)

    # Combine legends
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Annotate bar values
    for bar, val in zip(bars_rmse, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_mae, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    metrics_path = Path(__file__).parent.parent / "artifacts" / "models_metrics.json"
    out_path = Path(__file__).parent.parent / "artifacts" / "model_comparison_metrics.png"

    metrics = load_metrics(metrics_path)
    plot_metrics(metrics, out_path)
    print(f"Saved model comparison plot to {out_path}")


if __name__ == "__main__":
    main()

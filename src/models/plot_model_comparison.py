"""
Plot train vs validation RMSE comparison per model.

Reads `src/artifacts/models_metrics.json` produced by `train.py` and
creates a grouped bar chart comparing train and validation RMSE for each model.

Usage:
    python -m src.models.plot_model_comparison

Outputs:
    src/artifacts/model_comparison_rmse.png
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
    train_rmses = np.array([m.get("train_rmse", float("nan")) for m in metrics])
    val_rmses = np.array([m.get("val_rmse", float("nan")) for m in metrics])

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_train = ax.bar(x - width / 2, train_rmses, width, label="Train RMSE", color="#4C72B0")
    bars_val = ax.bar(x + width / 2, val_rmses, width, label="Validation RMSE", color="#DD8452")

    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)

    # Annotate bar values
    for bar, val in zip(bars_train, train_rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars_val, val_rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

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

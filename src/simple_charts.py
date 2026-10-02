"""Generate simple, presentation-friendly charts (no error bars, no jargon)
for showing this project to a non-technical audience, e.g. a class.

Run with:
    python src/simple_charts.py --dataset FD001
Saves PNGs to outputs/ (created if missing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loading import load_config
from src.predict import load_artifact, predict_latest_cycle


def make_accuracy_chart(preds, out_path: Path):
    """Predicted vs. true RUL scatter with a 'perfect prediction' line.
    Easiest chart type for a general audience: dots close to the
    diagonal line = good predictions."""
    fig, ax = plt.subplots(figsize=(7, 7))

    colors = {"Urgent": "#d62728", "Watch": "#ff9900", "Safe": "#2ca02c"}
    for tier, color in colors.items():
        subset = preds[preds["risk_tier"] == tier]
        ax.scatter(subset["true_rul"], subset["rul_pred"], c=color,
                   label=tier, s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

    max_val = max(preds["true_rul"].max(), preds["rul_pred"].max()) + 5
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual remaining life (cycles)", fontsize=13)
    ax.set_ylabel("Model's predicted remaining life (cycles)", fontsize=13)
    ax.set_title("How close were the model's predictions?", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def make_risk_summary_chart(preds, out_path: Path):
    """Simple bar chart: how many engines fall into each risk category."""
    counts = preds["risk_tier"].value_counts()
    order = ["Urgent", "Watch", "Safe"]
    counts = counts.reindex(order).fillna(0)
    colors = {"Urgent": "#d62728", "Watch": "#ff9900", "Safe": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(order, counts.values, color=[colors[t] for t in order])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, str(int(val)),
                ha="center", fontsize=14, fontweight="bold")

    ax.set_ylabel("Number of engines", fontsize=13)
    ax.set_title("Out of 100 engines, how many need attention?", fontsize=15, fontweight="bold")
    ax.set_ylim(0, counts.values.max() + 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifact = load_artifact(args.dataset)
    preds = predict_latest_cycle(args.dataset, cfg, artifact)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    make_accuracy_chart(preds, out_dir / "accuracy_chart.png")
    make_risk_summary_chart(preds, out_dir / "risk_summary_chart.png")


if __name__ == "__main__":
    main()

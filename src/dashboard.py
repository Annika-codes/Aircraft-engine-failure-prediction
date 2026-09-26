"""Streamlit dashboard: fleet-level view of predicted engine risk.

Run with:
    streamlit run src/dashboard.py -- --dataset FD001
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import streamlit as st

from src.data_loading import load_config
from src.predict import load_artifact, predict_latest_cycle


def parse_args():
    # Streamlit passes its own args before "--"; only look past that.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    st.set_page_config(page_title="Turbofan Fleet Risk", layout="wide")
    st.title(f"Fleet Risk Dashboard — {args.dataset}")

    artifact = load_artifact(args.dataset)
    preds = predict_latest_cycle(args.dataset, cfg, artifact)
    preds = preds.sort_values("rul_pred")

    tier_order = {"Urgent": 0, "Watch": 1, "Safe": 2}
    preds["_sort"] = preds["risk_tier"].map(tier_order)
    preds = preds.sort_values(["_sort", "rul_pred"]).drop(columns="_sort")

    n_urgent = (preds["risk_tier"] == "Urgent").sum()
    n_watch = (preds["risk_tier"] == "Watch").sum()
    n_safe = (preds["risk_tier"] == "Safe").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Urgent", n_urgent)
    c2.metric("Watch", n_watch)
    c3.metric("Safe", n_safe)

    st.subheader("Engines needing attention")
    st.dataframe(
        preds[preds["risk_tier"] != "Safe"][
            ["engine_id", "rul_pred", "rul_lower", "rul_upper", "risk_tier"]
        ],
        use_container_width=True,
    )

    st.subheader("Predicted RUL with 80% interval, all engines")
    fig, ax = plt.subplots(figsize=(10, max(4, len(preds) * 0.25)))
    colors = {"Urgent": "#d62728", "Watch": "#ff7f0e", "Safe": "#2ca02c"}
    for _, row in preds.iterrows():
        ax.plot(
            [row["rul_lower"], row["rul_upper"]],
            [row["engine_id"]] * 2,
            color=colors[row["risk_tier"]],
            linewidth=2,
        )
        ax.scatter(row["rul_pred"], row["engine_id"], color="black", s=10, zorder=3)
    ax.set_xlabel("Predicted RUL (cycles)")
    ax.set_ylabel("Engine ID")
    ax.set_title("Point estimate (black dot) and 10-90% interval, by engine")
    st.pyplot(fig)

    with st.expander("Full prediction table"):
        st.dataframe(preds, use_container_width=True)


if __name__ == "__main__":
    main()

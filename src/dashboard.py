"""Streamlit dashboard: fleet-level risk view, a single-engine prediction
card, and model performance metrics (classification + regression).

Run with:
    streamlit run src/dashboard.py -- --dataset FD001
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data_loading import load_config
from src.predict import (evaluate_classifier_on_test, feature_importance_by_category,
                          load_artifact, predict_latest_cycle)
from src.scoring import interval_coverage, mae, phm08_score, rmse


def parse_args():
    # Streamlit's `run script.py -- --dataset ALL` strips the "--" itself
    # before launching the script, so sys.argv is already just the
    # script's own args (e.g. ['dashboard.py', '--dataset', 'ALL']) --
    # no "--" token to search for. Just parse everything after argv[0].
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args(sys.argv[1:])


def render_fleet_overview(preds):
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

    colors = {"Urgent": "#d62728", "Watch": "#ff9900", "Safe": "#2ca02c"}
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("How many engines need attention?")
        counts = preds["risk_tier"].value_counts()
        order = ["Urgent", "Watch", "Safe"]
        counts = counts.reindex(order).fillna(0)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        bars = ax1.bar(order, counts.values, color=[colors[t] for t in order])
        for bar, val in zip(bars, counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, str(int(val)),
                      ha="center", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Number of engines")
        ax1.set_ylim(0, counts.values.max() + 10)
        fig1.tight_layout()
        st.pyplot(fig1)

    with col_b:
        st.subheader("How close were the predictions?")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for tier, color in colors.items():
            subset = preds[preds["risk_tier"] == tier]
            ax2.scatter(subset["true_rul"], subset["rul_pred"], c=color,
                        label=tier, s=45, alpha=0.85, edgecolors="white", linewidth=0.4)
        max_val = max(preds["true_rul"].max(), preds["rul_pred"].max()) + 5
        ax2.plot([0, max_val], [0, max_val], "k--", linewidth=1.3, label="Perfect prediction")
        ax2.set_xlabel("Actual remaining life")
        ax2.set_ylabel("Predicted remaining life")
        ax2.legend(fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)

    with st.expander("Full prediction table"):
        st.dataframe(preds, use_container_width=True)


def render_prediction_card(preds, artifact):
    st.subheader("Prediction Result")

    engine_ids = preds["engine_id"].tolist()
    selected = st.selectbox("Choose an engine", engine_ids)
    row = preds[preds["engine_id"] == selected].iloc[0]

    is_failure = row["predicted_status"] == "Likely Failure"
    failure_pct = row["failure_probability"] * 100
    normal_pct = 100 - failure_pct

    if is_failure:
        st.error(f"**HIGH FAILURE RISK** — Engine {selected} is likely to fail in the near future.")
    else:
        st.success(f"**LOW FAILURE RISK** — Engine {selected} looks healthy.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicted Status**")
        if is_failure:
            st.markdown(
                "<span style='background-color:#fecaca;color:#991b1b;"
                "padding:4px 10px;border-radius:6px;font-weight:600;'>Likely Failure</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='background-color:#bbf7d0;color:#166534;"
                "padding:4px 10px;border-radius:6px;font-weight:600;'>Likely Normal</span>",
                unsafe_allow_html=True,
            )

        st.markdown("**Failure Probability**")
        st.progress(int(normal_pct), text=f"Normal — {normal_pct:.0f}%")
        st.progress(int(failure_pct), text=f"Failure — {failure_pct:.0f}%")

    with col2:
        st.markdown("**Estimated Remaining Useful Life (RUL)**")
        st.markdown(f"## ~ {row['rul_pred']:.0f} cycles")
        st.caption(f"80% interval: {row['rul_lower']:.0f} – {row['rul_upper']:.0f} cycles")

    st.markdown("**Top Contributing Sensor Categories**")
    st.caption(
        "This dataset has no vibration sensor — categories below are grouped "
        "from the 21 real C-MAPSS temperature/pressure/speed/flow sensors."
    )
    importance_df = feature_importance_by_category(artifact)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(importance_df["category"], importance_df["importance"], color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("Relative importance")
    fig.tight_layout()
    st.pyplot(fig)


def render_model_performance(dataset, cfg, artifact, preds):
    st.subheader("Model Performance")
    tab1, tab2 = st.tabs(["Classification (XGBoost)", "Regression (RUL)"])

    with tab1:
        metrics = evaluate_classifier_on_test(dataset, cfg, artifact)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

        st.markdown("**Confusion Matrix**")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Failure"])
        ax.set_yticklabels(["Normal", "Failure"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            f"'Failure' = true RUL under {artifact['failure_threshold']} cycles. "
            "Derived from RUL labels, not separately annotated failure records."
        )

    with tab2:
        y_true = preds["true_rul"].values
        y_pred = preds["rul_pred"].values
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{rmse(y_true, y_pred):.2f}")
        c2.metric("MAE", f"{mae(y_true, y_pred):.2f}")
        c3.metric("PHM08 score", f"{phm08_score(y_true, y_pred):.1f}")
        cov = interval_coverage(y_true, preds["rul_lower"].values, preds["rul_upper"].values)
        c4.metric("80% interval coverage", f"{cov*100:.1f}%")
        st.caption("Lower RMSE/MAE/PHM08 is better. Interval coverage target is ~80%.")


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

    render_fleet_overview(preds)
    st.divider()
    render_prediction_card(preds, artifact)
    st.divider()
    render_model_performance(args.dataset, cfg, artifact, preds)


if __name__ == "__main__":
    main()

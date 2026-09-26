"""CLI entry point.

    python main.py train --dataset FD001
    python main.py evaluate --dataset FD001
"""
from __future__ import annotations

import argparse

from src.data_loading import load_config
from src.predict import load_artifact, predict_latest_cycle
from src.scoring import interval_coverage, mae, phm08_score, rmse
from src.train import run_training


def cmd_train(args):
    cfg = load_config(args.config)
    run_training(args.dataset, cfg)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    artifact = load_artifact(args.dataset)
    preds = predict_latest_cycle(args.dataset, cfg, artifact)

    y_true = preds["true_rul"].values
    y_pred = preds["rul_pred"].values

    print(f"Dataset: {args.dataset}  (n={len(preds)} test engines)")
    print(f"RMSE:            {rmse(y_true, y_pred):.2f}")
    print(f"MAE:             {mae(y_true, y_pred):.2f}")
    print(f"PHM08 score:     {phm08_score(y_true, y_pred):.2f}  (lower is better)")
    print(f"80% interval coverage: {interval_coverage(y_true, preds['rul_lower'].values, preds['rul_upper'].values):.1%}"
          f"  (target ~80%)")
    print()
    print(preds["risk_tier"].value_counts().rename("engine count"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", default="FD001")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--dataset", default="FD001")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

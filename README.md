# Turbofan Predictive Maintenance (RUL Prediction)

Predict Remaining Useful Life (RUL) of aircraft turbofan engines from
multivariate sensor streams, using NASA's C-MAPSS degradation dataset.

This isn't just "train XGBoost, report RMSE." It adds three things that
a plain tutorial version skips:

1. **Prediction intervals, not just a point estimate** — quantile
   regression gives a low/median/high RUL band, so a maintenance team
   knows how much slack they actually have.
2. **Asymmetric, cost-aware scoring (PHM08 score)** — plain RMSE treats
   "predicted failure 10 cycles late" the same as "10 cycles early."
   In reality, being late is far more costly (unplanned failure) than
   being early (wasted maintenance). This repo implements the scoring
   function used in the original NASA PHM08 prognostics challenge,
   which penalizes late predictions much more heavily.
3. **A risk dashboard** — a Streamlit app that turns raw predictions
   into a fleet-level "which engines need attention this week" view.

## Dataset

Uses NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset
(4 sub-datasets, FD001–FD004, varying operating conditions and fault
modes). It is not bundled in this repo — download it yourself:

- NASA Prognostics Data Repository, or
- Kaggle: search "NASA Turbofan Jet Engine Data Set"

Place the raw `train_FD00X.txt`, `test_FD00X.txt`, and
`RUL_FD00X.txt` files in `data/raw/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train a model on FD001 (point + quantile models, saved to models/)
python main.py train --dataset FD001

# Evaluate on the held-out test split: RMSE, MAE, and PHM08 score
python main.py evaluate --dataset FD001

# Launch the fleet risk dashboard
streamlit run src/dashboard.py -- --dataset FD001
```

## Repo layout

```
data/raw/          raw C-MAPSS files (you supply these)
data/processed/    cached feature tables (generated)
models/            trained model artifacts (generated)
src/
  data_loading.py  parse raw C-MAPSS text files into DataFrames
  rul_labels.py    compute Remaining Useful Life labels (with clipping)
  features.py      rolling-window sensor features, degradation slopes
  scoring.py        RMSE/MAE + the PHM08 asymmetric scoring function
  train.py         trains point (median) + quantile (10/90%) models
  predict.py       inference pipeline: point estimate + interval + risk tier
  dashboard.py     Streamlit fleet risk dashboard
main.py            CLI entry point (train / evaluate)
tests/             sanity tests for the scoring function and RUL labels
config.yaml        dataset paths, feature + model hyperparameters
```

## Results (FD001, real data, held-out test set)

Verified end-to-end against the actual C-MAPSS FD001 train/test files
(trained only on `train_FD001.txt`, scored against the true `RUL_FD001.txt`
labels — no leakage):

| Metric | Value |
|---|---|
| RMSE | 15.77 |
| MAE | 11.64 |
| PHM08 score (lower is better) | 554.13 |
| 80% interval coverage | 77.0% (target ~80%) |

For reference, a plain XGBoost-on-rolling-features baseline without the
quantile/interval or PHM08 layers (e.g. the common tutorial-style version
of this project) typically reports RMSE in the mid-30s on this same test
split. The gap here comes mainly from the rolling-window + slope features
and standard RUL clipping, not from anything exotic.

## Method notes

- RUL labels are **clipped** at a max value (default 125 cycles) — early
  in an engine's life, RUL is not really predictable from sensors, and
  training on unclipped labels there just adds noise. This is standard
  practice on this dataset, not a differentiator, but worth knowing.
- Sensors with ~zero variance across an operating condition (there are
  several in the raw 21-sensor set) are dropped automatically per
  sub-dataset.
- Point model: XGBoost regression (median). Interval model: XGBoost
  with quantile (pinball) loss at the 10th and 90th percentiles.
- "Risk tier" in the dashboard = a simple mapping from predicted RUL
  and interval width to {Safe, Watch, Urgent}, not a separate model.

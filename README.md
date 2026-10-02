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
`RUL_FD00X.txt` files in `data/raw/`. You can use just FD001 (the
default), or download all four to train on the combined dataset (see
"Training on more data" below).

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
  sensor_names.py  human-readable sensor names/categories (from NASA docs)
  op_regimes.py    operating-condition regime clustering + normalization (for combined training)
  scoring.py        RMSE/MAE + the PHM08 asymmetric scoring function
  classification.py binary failure-risk classifier + accuracy/precision/recall/F1
  train.py         trains point (median) + quantile (10/90%) + classifier models, single or combined
  predict.py       inference: point estimate + interval + failure probability + risk tier
  dashboard.py     Streamlit dashboard: fleet view, prediction card, model performance
main.py            CLI entry point (train / evaluate, --dataset FD001 or ALL)
tests/             unit tests for scoring, labels, classification, sensor mapping, regimes
config.yaml        dataset paths, feature + model hyperparameters
```

## Results (FD001, real data, held-out test set)

Verified end-to-end against the actual C-MAPSS FD001 train/test files
(trained only on `train_FD001.txt`, scored against the true `RUL_FD001.txt`
labels — no leakage):

**Regression (Remaining Useful Life):**

| Metric | Value |
|---|---|
| RMSE | 15.77 |
| MAE | 11.64 |
| PHM08 score (lower is better) | 554.13 |
| 80% interval coverage | 77.0% (target ~80%) |

**Classification (failure-risk: RUL < 30 cycles = "Failure"):**

| Metric | Value |
|---|---|
| Accuracy | 95.0% |
| Precision | 91.7% |
| Recall | 88.0% |
| F1 score | 89.8% |

For reference, a plain XGBoost-on-rolling-features baseline without the
quantile/interval or PHM08 layers (e.g. the common tutorial-style version
of this project) typically reports RMSE in the mid-30s on this same test
split. The gap here comes mainly from the rolling-window + slope features
and standard RUL clipping, not from anything exotic.

## Training on more data (combined FD001–FD004)

By default, `train`/`evaluate` use a single sub-dataset (FD001). You can
also train on several combined, up to all four at once, for a much
larger and more varied dataset:

```bash
python main.py train --dataset ALL           # all 4 sub-datasets combined
python main.py train --dataset FD001,FD003   # any comma-separated subset
python main.py evaluate --dataset ALL
```

Combined totals across FD001–FD004: **~709 training engines / ~160,000
training rows**, **~707 test engines / ~105,000 test rows**.

FD002 and FD004 run under 6 distinct operating conditions (different
altitude/Mach/throttle combinations), unlike FD001/FD003's single
condition — a sensor's raw reading shifts with the flight condition
itself, on top of any change from wear. Training on the combined data
without accounting for this would genuinely hurt accuracy, not just
add noise. `src/op_regimes.py` clusters the operating-condition columns
into regimes and z-score normalizes each sensor within its own regime
(fit on training data, reused identically at inference) — standard
practice in the published RUL literature on this dataset, not just an
extra step for its own sake. Verified on this exact pipeline:

| | RMSE | MAE |
|---|---|---|
| Combined, without regime normalization | 17.46 | 12.98 |
| Combined, with regime normalization | 15.85 | 11.29 |

**Results, all 4 combined (707 held-out test engines):**

| Metric | Value |
|---|---|
| RMSE | 15.85 |
| MAE | 11.29 |
| PHM08 score | 4095.14 |
| 80% interval coverage | 84.0% |
| Classifier accuracy | 96.0% |
| Classifier precision / recall / F1 | 90.6% / 91.7% / 91.1% |

Notably, accuracy on the combined set is close to FD001 alone despite
FD002/FD004 being substantially harder (multiple fault modes *and*
multiple operating conditions) — the regime normalization is doing
real work here, not just adding complexity.

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
- Failure-risk classification is a second, complementary model (XGBoost
  classifier) trained on the same features, with the label derived from
  RUL (< 30 cycles remaining = "Failure"). It's not literally a
  different kind of failure data -- it's the same ground truth reframed
  as a binary risk call, which is what lets the dashboard show
  accuracy/precision/recall/F1 and a confusion matrix alongside the
  regression metrics.
- The dataset's 21 sensors are documented in NASA's own C-MAPSS
  materials and are temperature, pressure, speed, and fuel-flow
  measurements -- there is no vibration sensor in this dataset, so the
  dashboard's "top contributing sensor categories" chart groups by the
  categories that actually exist (Temperature/Pressure/Speed/Fuel
  Flow/Other), not by category names borrowed from a different sensor
  suite.

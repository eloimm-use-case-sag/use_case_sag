# Fragrance Shelf-Life Stability

Some batches of fragrance were losing shelf life and nobody knew why.
This project finds out why, and builds a model that predicts shelf life
at production time instead of waiting months for the lab. The data is
259,200 batches from five plants, ten reactor models and 90 days of
production.

## Start here

```
pip install -r requirements.txt
streamlit run app/app.py
```

That opens the report. Six pages, in reading order: the data, the
finding, how the system works, one batch — real or built by hand — how
good it is, and the anomaly console.

**If you only read one thing**, read page 2. The finding is on it.

## How the code is laid out

```
01_exploration_and_cleaning.ipynb   the data, the repairs, the finding
run_pipeline.py                     runs everything else
requirements.txt                    pinned, CPython 3.13

src/          12 modules            the model and the detectors
app/          the Streamlit report
data/         the cleaned dataset, plus the model input
outputs/      models, figures, tables
```

Everything starts from the notebook: it reads the raw CSV, repairs it,
and writes `data/fragrance_clean.parquet`. The report also reads the raw
CSV when it needs to show an original value before cleaning.

### The modules

| file | what it does |
|---|---|
| `paths.py` | every file location, in one place |
| `model.py` | the column schema and the loader |
| `features.py` | optional torque z/CUSUM features from a frozen reference |
| `probabilistic_catboost.py` | five models that predict a value **and** a spread |
| `uncertainty_diagnostics.py` | the two uncertainty plots used by Streamlit |
| `conformal_calibration.py` | turns that spread into a 90 % range |
| `conformal_grouped.py` | one range rule per missing-sensor group |
| `explain.py` | SHAP: what each sensor contributed, in months |
| `simple_monitor.py` | anomalies and regime changes, per reactor |
| `batch_report.py` | one batch through the whole stack |
| `evaluate_simple_monitor.py` | validates the monitor on the known 8 May event |
| `evaluate_heldout_regime.py` | retrains after hiding the affected regime |

### Running it

```
streamlit run app/app.py          # the report, at localhost:8501

python run_pipeline.py            # reuses the saved models
python run_pipeline.py --only eval-heldout  # slow regime-holdout test
python run_pipeline.py --retrain  # trains from scratch, ~15 min
python run_pipeline.py --analysis # validates the current monitor
python run_pipeline.py --list     # what each stage does
```

The saved model records its feature profile. A normal rerun preserves
that profile. To change it, retraining is required:

```
python run_pipeline.py --retrain --no-temporal-features
python run_pipeline.py --retrain --temporal-features
```

Without saved models, the default is the 9 core features. The optional
profile adds `torque_z_ref`, `torque_cusum_up` and
`torque_cusum_down`; `features.py` remains available for reproducing
that experiment.

The report reads saved artifacts and the raw and cleaned data. It never
trains and never writes. Stop it with Ctrl-C.

## How the system works

```
clean dataset
   │
   ├──► five CatBoost models  ──►  a value, and how sure it is
   │           │
   │           ├──► conformal calibration  ──►  a 90 % range
   │           └──► SHAP                   ──►  why that number
   │
   └──► frozen reference per reactor
               ├──► anomaly, per batch
               └──► CUSUM, per reactor  ──►  has it changed for good
```

Two paths leave the data. One predicts. The other decides whether the
prediction can be trusted. They are deliberately separate: the model
only knows what it was trained on, so it cannot recognise a fault it has
never seen. The detector can, because it compares each reactor against
itself.

| what | number |
|---|---|
| R² on the time split | 0.796 |
| average error | 0.74 months (3.4 %) |
| batches predicted within 1.5 months | 90 % |
| range asked for / delivered | 90 % / 90.3 % |
| median range width | 2.72 months |
| hidden-regime stress-test MAE | 1.48 months |
| hidden-regime 90 % coverage | 18.4 % |
| anomaly detection / false alarms | 90.9 % / 0.000 % |
| regime change found | all 3 reactors, same day, no false alarms |
# use_case_sag

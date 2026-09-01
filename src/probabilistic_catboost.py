"""Five-member CatBoost ensemble for shelf life and uncertainty."""
from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import FROZEN
from model import CONTEXT, SENSORS, TARGET, load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import MODELS, ROOT, TABLES

RAIZ = ROOT
OUT = TABLES

# --------------------------------------------------------------- schema

# The real process unit is the plant/reactor pair: reactor IDs repeat
# across sites, so R03 is not one machine. Both `plant_location` and
# `reactor` are deterministic given `unit`; keeping them would duplicate
# the same categorical information.
UNIT = "unit"
CAT_FEATURES = [UNIT]

CORE_NUMERIC_FEATURES = SENSORS + ["n_missing"]
TEMPORAL_ENV = "FRAGRANCE_USE_TEMPORAL_FEATURES"


def temporal_features_enabled(default: bool = False) -> bool:
    """Read the optional temporal-feature profile from the environment."""
    value = os.getenv(TEMPORAL_ENV)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{TEMPORAL_ENV} must be 0/1, true/false or on/off")


def numeric_features_for(use_temporal_features: bool) -> list[str]:
    """Return the numeric columns for one training profile."""
    return CORE_NUMERIC_FEATURES + (FROZEN if use_temporal_features else [])


# Kept as the direct-script defaults. Saved ensembles restore their own
# feature list from config.json when loaded.
NUMERIC_FEATURES = numeric_features_for(temporal_features_enabled())
FEATURES = NUMERIC_FEATURES + CAT_FEATURES

FORBIDDEN = ["purity_grade", "oxidation_risk_flag"]

# Temporal boundaries: train, then validation, then test, never
# overlapping and never out of order. A random split would let a model
# learn from batches produced after the ones it is scored on, which no
# plant can do.
#
#   train       01 Apr - 07 Jun   195,568   75.5 %
#   validation  08 Jun - 15 Jun    23,155    8.9 %
#   test        16 Jun - 29 Jun    40,477   15.6 %
#
# The test window is the last fortnight. It was a full month until the
# split was revisited: moving the last week of May and the first of June
# into training buys 43,000 more training rows, at the cost of a smaller
# and therefore noisier evaluation set.
VAL_START = pd.Timestamp("2023-06-08")
TEST_START = pd.Timestamp("2023-06-16")

SEEDS = [0, 1, 2, 3, 4]

# Chosen for this dataset rather than copied from the LightGBM model,
# whose parameters were tuned for a different loss and library.
# depth 6 is ample for this feature set; the learning rate is low enough that
# early stopping, not the iteration count, decides the model size; and
# l2_leaf_reg is raised because the exploration phase showed four
# sensors carry little marginal signal and a loose model will happily
# fit noise into sigma.
PARAMS = dict(
    loss_function="RMSEWithUncertainty",
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=6.0,
    early_stopping_rounds=100,
    thread_count=-1,
    verbose=False,
    allow_writing_files=False,
)


@dataclass
class SplitSizes:
    train: int
    validation: int
    test: int


def prepare(df: pd.DataFrame,
            numeric_features: list[str] | None = None,
            cat_features: list[str] | None = None) -> pd.DataFrame:
    """Build the feature frame. Categoricals as strings for CatBoost."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    cat_features = cat_features or CAT_FEATURES
    missing = [c for c in numeric_features + cat_features
               if c not in df.columns]
    if missing:
        raise KeyError(f"model input is missing columns: {missing}")
    X = df[numeric_features].copy()
    for col in cat_features:
        # A null category becomes an explicit level, never an imputed
        # one: filling it from the target would be leakage.
        X[col] = df[col].astype("string").fillna("__MISSING__")
    return X


def splits(df: pd.DataFrame):
    """Temporal train / validation / test, in that order."""
    train = df[df.ts < VAL_START]
    val = df[(df.ts >= VAL_START) & (df.ts < TEST_START)]
    test = df[df.ts >= TEST_START]
    return train, val, test


# ------------------------------------------------- semantics of the loss

def verify_semantics(seed: int = 0) -> dict:
    """Verify that CatBoost returns variance, not log-sigma."""
    rng = np.random.default_rng(seed)
    n = 8000
    grp = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    sigma = np.where(grp == 0, 0.5, 2.0)
    y = 3.0 * x + rng.normal(0, sigma)
    X = pd.DataFrame({"x": x, "g": np.where(grp == 0, "A", "B")})

    m = CatBoostRegressor(loss_function="RMSEWithUncertainty",
                          iterations=300, learning_rate=0.1, depth=5,
                          random_seed=seed, verbose=False,
                          allow_writing_files=False)
    m.fit(Pool(X, y, cat_features=["g"]))
    raw = m.predict(X, prediction_type="RawFormulaVal")
    unc = m.predict(X, prediction_type="RMSEWithUncertainty")

    out = {
        "shape": list(unc.shape),
        "second_column_negatives": int((unc[:, 1] < 0).sum()),
        "raw_second_column_negatives": int((raw[:, 1] < 0).sum()),
        "max_abs_diff_variance_vs_exp2logsigma":
            float(np.abs(unc[:, 1] - np.exp(2 * raw[:, 1])).max()),
        "max_abs_diff_mean_columns":
            float(np.abs(unc[:, 0] - raw[:, 0]).max()),
    }
    ok = (out["second_column_negatives"] == 0
          and out["max_abs_diff_variance_vs_exp2logsigma"] < 1e-8)
    out["conclusion"] = (
        "column 1 of RMSEWithUncertainty is the VARIANCE; "
        "column 1 of RawFormulaVal is log(sigma)" if ok else
        "UNEXPECTED: the documented relationship does not hold")
    if not ok:
        raise AssertionError(out["conclusion"])
    return out


# --------------------------------------------------------------- model

class ProbabilisticEnsemble:
    """CatBoost members with an explicit variance decomposition."""

    def __init__(self, seeds=SEEDS, params: dict | None = None,
                 use_temporal_features: bool | None = None):
        self.seeds = list(seeds)
        self.params = {**PARAMS, **(params or {})}
        self.use_temporal_features = (
            temporal_features_enabled() if use_temporal_features is None
            else bool(use_temporal_features))
        self.numeric_features = numeric_features_for(
            self.use_temporal_features)
        self.cat_features = list(CAT_FEATURES)
        self.features = self.numeric_features + self.cat_features

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build input using the profile stored with this ensemble."""
        return prepare(df, self.numeric_features, self.cat_features)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        leaked = set(self.features) & set(FORBIDDEN)
        if leaked:
            raise AssertionError(f"target-derived feature: {leaked}")

        tr_pool = Pool(self.prepare(train), train[TARGET],
                       cat_features=self.cat_features)
        va_pool = Pool(self.prepare(val), val[TARGET],
                       cat_features=self.cat_features)

        self.members_, self.best_iterations_ = [], []
        for seed in self.seeds:
            m = CatBoostRegressor(random_seed=seed, **self.params)
            m.fit(tr_pool, eval_set=va_pool)
            self.members_.append(m)
            self.best_iterations_.append(int(m.get_best_iteration()))
            print(f"  member seed={seed}  best_iteration="
                  f"{m.get_best_iteration()}")
        return self

    def predict_members(self, df: pd.DataFrame):
        """(mu, variance) per member, shape (n, M) each."""
        X = self.prepare(df)
        mus, variances = [], []
        for m in self.members_:
            p = m.predict(X, prediction_type="RMSEWithUncertainty")
            mus.append(p[:, 0])
            variances.append(p[:, 1])
        return np.column_stack(mus), np.column_stack(variances)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the ensemble mean and total-variance decomposition."""
        mus, variances = self.predict_members(df)
        mu_bar = mus.mean(axis=1)
        aleatoric_var = variances.mean(axis=1)
        epistemic_var = ((mus - mu_bar[:, None]) ** 2).mean(axis=1)
        total_var = aleatoric_var + epistemic_var

        out = pd.DataFrame({
            "batch_id": df["batch_id"].to_numpy(),
            "unit": df[UNIT].to_numpy(),
            "n_missing": df["n_missing"].to_numpy(),
            "predicted_months": mu_bar,
            "aleatoric_var": aleatoric_var,
            "aleatoric_std": np.sqrt(aleatoric_var),
            "epistemic_var": epistemic_var,
            "epistemic_std": np.sqrt(epistemic_var),
            "total_var": total_var,
            "total_std": np.sqrt(total_var),
        })
        for j, seed in enumerate(self.seeds):
            out[f"mu_model_{seed}"] = mus[:, j]
            out[f"var_model_{seed}"] = variances[:, j]
        if TARGET in df:
            out["actual_months"] = df[TARGET].to_numpy()
        return out

    # ------------------------------------------------------ persistence

    def save(self, folder: Path = MODELS) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for seed, m in zip(self.seeds, self.members_):
            m.save_model(str(folder / f"catboost_seed{seed}.cbm"))
        (folder / "config.json").write_text(json.dumps({
            "seeds": self.seeds,
            "params": {k: v for k, v in self.params.items()
                       if k != "thread_count"},
            "best_iterations": self.best_iterations_,
            "features": self.features,
            "numeric_features": self.numeric_features,
            "cat_features": self.cat_features,
            "use_temporal_features": self.use_temporal_features,
            "val_start": str(VAL_START.date()),
            "test_start": str(TEST_START.date()),
        }, indent=2), encoding="utf-8")

    def load_saved(self, folder: Path = MODELS):
        cfg = json.loads((folder / "config.json").read_text("utf-8"))
        self.seeds = cfg["seeds"]
        self.features = list(cfg["features"])
        self.cat_features = list(cfg.get("cat_features", CAT_FEATURES))
        self.numeric_features = list(cfg.get(
            "numeric_features",
            [c for c in self.features if c not in self.cat_features]))
        self.use_temporal_features = bool(cfg.get(
            "use_temporal_features",
            any(c in self.numeric_features for c in FROZEN)))
        self.members_ = []
        for seed in self.seeds:
            m = CatBoostRegressor()
            m.load_model(str(folder / f"catboost_seed{seed}.cbm"))
            self.members_.append(m)
        self.best_iterations_ = cfg["best_iterations"]
        return self


# ---------------------------------------------------------------- main

def main() -> None:
    print("=" * 66)
    print("  PROBABILISTIC CATBOOST ENSEMBLE - training")
    print("=" * 66)

    print("\nVerifying what RMSEWithUncertainty returns...")
    checks = verify_semantics()
    print(f"  {checks['conclusion']}")
    print(f"  |variance - exp(2*log_sigma)| max = "
          f"{checks['max_abs_diff_variance_vs_exp2logsigma']:.2e}")

    use_temporal = temporal_features_enabled()
    df = load(include_temporal=use_temporal)
    train, val, test = splits(df)
    sizes = SplitSizes(len(train), len(val), len(test))
    print(f"\nsplit  train {sizes.train:,}  |  validation "
          f"{sizes.validation:,}  |  test {sizes.test:,}")
    print(f"       train  < {VAL_START.date()}")
    print(f"       val    {VAL_START.date()} .. {TEST_START.date()}")
    print(f"       test  >= {TEST_START.date()}")
    ens = ProbabilisticEnsemble(use_temporal_features=use_temporal)
    profile = "core + temporal" if use_temporal else "core only"
    print(f"\nfeature profile: {profile}")
    print(f"features ({len(ens.features)}): "
          f"{len(ens.numeric_features)} numeric (NaN kept as-is) + "
          f"{len(ens.cat_features)} categorical")
    print(f"  numeric    : {ens.numeric_features}")
    print(f"  categorical: {ens.cat_features}")

    print(f"\nTraining {len(SEEDS)} members...")
    ens.fit(train, val)
    ens.save()

    pred = ens.predict(test)
    pred.to_parquet(OUT / "test_predictions.parquet", index=False)
    (OUT / "split_sizes.json").write_text(
        json.dumps(asdict(sizes), indent=2), encoding="utf-8")
    (OUT / "semantics_check.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")

    print(f"\nsaved  {MODELS.name}/ (5 models + config)")
    print(f"       {OUT.name}/test_predictions.parquet "
          f"({len(pred):,} rows)")
    print("\nNext: python run_pipeline.py --only calibrate\n")


if __name__ == "__main__":
    main()

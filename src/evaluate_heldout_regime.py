"""Evaluate the ensemble after hiding the known faulty regime."""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from conformal_calibration import present, score_slice
from conformal_grouped import GroupedConformal
from model import CHANGE_DATE, FAULTY_UNITS, TARGET, load
from paths import TABLES
from probabilistic_catboost import (TEST_START, VAL_START,
                                    ProbabilisticEnsemble)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUMMARY = TABLES / "heldout_regime_evaluation.json"
PREDICTIONS = TABLES / "heldout_regime_predictions.parquet"


def main() -> None:
    df = load()
    hidden = df.unit.isin(FAULTY_UNITS) & (df.ts >= CHANGE_DATE)
    train = df[(df.ts < VAL_START) & ~hidden]
    val = df[(df.ts >= VAL_START) & (df.ts < TEST_START) & ~hidden]
    stress = df[hidden]

    if train["unit"].isin(FAULTY_UNITS).sum() == 0:
        raise AssertionError("affected units must remain known pre-change")
    if ((train.unit.isin(FAULTY_UNITS) &
         (train.ts >= CHANGE_DATE)).any()
            or (val.unit.isin(FAULTY_UNITS) &
                (val.ts >= CHANGE_DATE)).any()):
        raise AssertionError("faulty-regime rows leaked into fitting")

    print("=" * 72)
    print("  HELD-OUT REGIME STRESS TEST")
    print("=" * 72)
    print(f"  train       {len(train):8,d}  healthy regime only")
    print(f"  validation  {len(val):8,d}  healthy regime only")
    print(f"  stress test {len(stress):8,d}  affected units from 8 May")
    print("\nTraining the same five-member core ensemble...")

    ens = ProbabilisticEnsemble(use_temporal_features=False)
    ens.fit(train, val)

    cal = score_slice(val, ens)
    gc = GroupedConformal(coverage=0.90, min_group=100,
                          fallback="global").fit(cal)
    pred = ens.predict(stress)
    q = gc.q_for(stress)
    mu, lo, hi = present(pred.predicted_months.to_numpy(),
                         pred.total_std.to_numpy(), q)
    y = stress[TARGET].to_numpy()
    err = mu - y

    observed_train_torque = train.mixing_torque_nm.dropna()
    observed_stress_torque = stress.mixing_torque_nm.dropna()
    train_min = float(observed_train_torque.min())
    summary = {
        "definition": (
            "train and validation exclude Plant-04/R03,R07,R08 from "
            "2023-05-08; test contains exactly those excluded batches"),
        "feature_profile": "9 core features",
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_stress": int(len(stress)),
        "best_iterations": ens.best_iterations_,
        "metrics": {
            "actual_mean": float(y.mean()),
            "predicted_mean": float(mu.mean()),
            "bias_months": float(err.mean()),
            "mae_months": float(np.abs(err).mean()),
            "rmse_months": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2)
                        / np.sum((y - y.mean()) ** 2)),
            "coverage_pct": float(100 * ((y >= lo) & (y <= hi)).mean()),
            "median_interval_width": float(np.median(hi - lo)),
        },
        "torque_support": {
            "training_min": train_min,
            "stress_mean": float(observed_stress_torque.mean()),
            "stress_min": float(observed_stress_torque.min()),
            "stress_below_training_min_pct": float(
                100 * (observed_stress_torque < train_min).mean()),
        },
        "conformal_q_global": float(gc.q_global_),
    }

    out = pred.assign(
        ts=stress.ts.to_numpy(),
        presented_months=mu,
        lower=lo,
        upper=hi,
        error_months=err,
    )
    out.to_parquet(PREDICTIONS, index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    m, t = summary["metrics"], summary["torque_support"]
    print("\nRESULTS")
    print(f"  actual mean       {m['actual_mean']:.3f} months")
    print(f"  predicted mean    {m['predicted_mean']:.3f} months")
    print(f"  bias              {m['bias_months']:+.3f} months")
    print(f"  MAE               {m['mae_months']:.3f} months")
    print(f"  90% coverage      {m['coverage_pct']:.2f} %")
    print(f"  median width      {m['median_interval_width']:.3f} months")
    print(f"  torque train min  {t['training_min']:.3f} N.m")
    print(f"  stress below min  {t['stress_below_training_min_pct']:.2f} %")
    print(f"\n  saved {SUMMARY.name}")
    print(f"  saved {PREDICTIONS.name}\n")


if __name__ == "__main__":
    main()

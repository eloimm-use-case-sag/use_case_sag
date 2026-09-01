"""Exact SHAP contributions for the ensemble, globally and per batch."""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import Pool

from model import SENSORS, TARGET, load
from probabilistic_catboost import ProbabilisticEnsemble, TEST_START

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)

LABEL = {
    "temp_c": "Reaction temperature",
    "vessel_pressure_bar": "Vessel pressure",
    "refractive_index": "Refractive index",
    "density_g_cm3": "Density",
    "ph_level": "pH level",
    "mixing_torque_nm": "Mixing torque",
    "mass_flow_rate_kg_h": "Mass flow rate",
    "n_missing": "Sensors blank",
    "plant_location": "Plant",
    "reactor": "Reactor",
    "unit": "Plant x reactor",
}


def shap_contributions(ens: ProbabilisticEnsemble, df: pd.DataFrame):
    """Return ensemble SHAP contributions in months and the base value."""
    X = ens.prepare(df)
    pool = Pool(X, cat_features=ens.cat_features)
    per_member, bases = [], []
    for m in ens.members_:
        sv = np.asarray(m.get_feature_importance(pool, type="ShapValues"))
        sv = sv[:, 0, :] if sv.ndim == 3 else sv
        per_member.append(sv[:, :-1])
        bases.append(sv[:, -1])
    contrib = pd.DataFrame(np.mean(per_member, axis=0),
                           columns=ens.features, index=df.index)
    base = float(np.mean(bases))
    return contrib, base


def check_additivity(ens, df, contrib, base) -> float:
    """Contributions plus base must reproduce the prediction exactly."""
    pred = ens.predict(df)["predicted_months"].to_numpy()
    return float(np.abs(contrib.sum(axis=1).to_numpy() + base
                        - pred).max())


def global_importance(contrib: pd.DataFrame) -> pd.DataFrame:
    """What the model leans on across the fleet, in months."""
    t = pd.DataFrame({
        "mean_abs_months": contrib.abs().mean(),
        "sd_months": contrib.std(),
        "p5": contrib.quantile(0.05),
        "p95": contrib.quantile(0.95),
    })
    t.index = [LABEL.get(i, i) for i in t.index]
    return t.sort_values("mean_abs_months", ascending=False)


def explain_batch(contrib: pd.DataFrame, base: float, df: pd.DataFrame,
                  idx) -> pd.DataFrame:
    """The account for one batch: from the fleet average to its number."""
    c = contrib.loc[idx]
    row = df.loc[idx]
    out = pd.DataFrame({
        "variable": [LABEL.get(k, k) for k in c.index],
        "reading": [row[k] for k in c.index],
        "months": c.to_numpy(),
    })
    out["cumulative"] = base + out["months"].cumsum()
    return out.reindex(out.months.abs().sort_values(ascending=False)
                       .index).reset_index(drop=True)


def effect_shape(contrib: pd.DataFrame, df: pd.DataFrame,
                 col: str, bins: int = 10) -> pd.DataFrame:
    """Summarise a sensor's SHAP contribution by value quantile."""
    d = pd.DataFrame({"v": df[col], "c": contrib[col]}).dropna()
    q = pd.qcut(d.v, bins, duplicates="drop")
    g = d.groupby(q, observed=True).agg(
        batches=("v", "size"), reading=("v", "mean"),
        months=("c", "mean"))
    return g.round(4)


def main() -> None:
    print("=" * 76)
    print("  HOW EACH VARIABLE MOVES THE SHELF-LIFE PREDICTION")
    print("=" * 76)

    ens = ProbabilisticEnsemble().load_saved()
    df = load(include_temporal=ens.use_temporal_features)
    test = df[df.ts >= TEST_START]
    sample = test.sample(min(25000, len(test)), random_state=0)

    contrib, base = shap_contributions(ens, sample)
    err = check_additivity(ens, sample, contrib, base)
    print(f"  {len(sample):,} test batches, {len(ens.features)} variables")
    print(f"  base value (the fleet's average prediction): "
          f"{base:.3f} months")
    print(f"  contributions + base reproduce the prediction to "
          f"{err:.1e} months")

    print("\n" + "=" * 76)
    print("  GLOBAL - WHAT DRIVES SHELF LIFE ACROSS THE FLEET")
    print("=" * 76)
    print("  months added to or taken off the prediction\n")
    print(global_importance(contrib).round(4).to_string())

    print("\n" + "=" * 76)
    print("  THE SHAPE OF EACH EFFECT")
    print("=" * 76)
    for col in ("mixing_torque_nm", "temp_c", "ph_level"):
        s = effect_shape(contrib, sample, col)
        first, last = s.months.iloc[0], s.months.iloc[-1]
        peak = s.months.idxmax()
        shape = ("rises" if s.months.argmax() == len(s) - 1 else
                 "falls" if s.months.argmax() == 0 else "peaks")
        print(f"\n  {LABEL[col]} - {shape}"
              f"   (lowest decile {first:+.2f}, highest {last:+.2f})")
        print(s.to_string())

    print("\n" + "=" * 76)
    print("  PER BATCH - THE WORST-PREDICTED LOT IN THE SAMPLE")
    print("=" * 76)
    # predict() returns a fresh index, so pick positionally.
    pred = ens.predict(sample)["predicted_months"].to_numpy()
    idx = sample.index[int(pred.argmin())]
    row = sample.loc[idx]
    acc = explain_batch(contrib, base, sample, idx)
    print(f"  {row.batch_id} on {row.unit}, "
          f"{row.ts:%d %b %Y}\n")
    print(f"  starting from the fleet average of {base:.2f} months:\n")
    for _, r in acc.iterrows():
        v = r.reading
        shown = (f"{v:.3f}" if isinstance(v, (int, float, np.floating))
                 and not pd.isna(v) else
                 ("no reading" if pd.isna(v) else str(v)))
        print(f"    {r.variable:22s} {shown:>14s}   "
              f"{r.months:+6.2f} months")
    print(f"    {'':22s} {'':>14s}   {'-' * 13}")
    print(f"    {'prediction':22s} {'':>14s}   "
          f"{base + acc.months.sum():6.2f} months")
    if TARGET in row:
        print(f"\n  the lab later measured {row[TARGET]:.2f} months")
    print()


if __name__ == "__main__":
    main()

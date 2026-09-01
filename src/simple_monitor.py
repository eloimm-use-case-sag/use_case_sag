"""Frozen-z batch anomalies and daily CUSUM regime monitoring."""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from model import FAULTY_UNITS, SENSORS, load
from paths import TABLES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REFERENCE_DAYS = 21      # the window each unit is anchored on
MIN_REFERENCE = 200      # batches a unit needs before it can be scored
BATCH_CUT = 2.0          # outside the support a healthy reading can reach
CUSUM_K = 0.5            # slack, in baseline SDs
CUSUM_H = 8.0            # decision limit for the daily tally
CONFIRM_DAYS = 3         # recheck this many daily observations after breach
SEVERITY_CUT = 100.0     # peak tally below this is treated as noise

CHANGE_DATE = pd.Timestamp("2023-05-08")


# =================================================================
# The reference
# =================================================================

@dataclass
class Reference:
    """What one reactor looked like while it was still trusted."""
    unit: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_batches: int
    mean: pd.Series          # per sensor, batch level
    sd: pd.Series
    daily_mean: pd.Series    # per sensor, daily aggregate
    daily_sd: pd.Series


def fit_references(df: pd.DataFrame,
                   reference_days: int = REFERENCE_DAYS
                   ) -> dict[str, Reference]:
    """Anchor every unit on its own opening weeks. Strictly causal."""
    out: dict[str, Reference] = {}
    for unit, g in df.groupby("unit"):
        start = g.ts.min()
        end = start + pd.Timedelta(days=reference_days)
        ref = g[g.ts < end]
        if len(ref) < MIN_REFERENCE:
            continue
        daily = ref.groupby(ref.ts.dt.normalize())[SENSORS].mean()
        out[unit] = Reference(
            unit=unit, start=start, end=end, n_batches=len(ref),
            mean=ref[SENSORS].mean(), sd=ref[SENSORS].std(ddof=1),
            daily_mean=daily.mean(), daily_sd=daily.std(ddof=1))
    return out


def z_scores(df: pd.DataFrame, refs: dict[str, Reference]) -> pd.DataFrame:
    """One z per sensor per batch, against that reactor's reference."""
    out = pd.DataFrame(np.nan, index=df.index, columns=SENSORS)
    for unit, g in df.groupby("unit"):
        r = refs.get(unit)
        if r is None:
            continue
        sd = r.sd.replace(0.0, np.nan)
        out.loc[g.index, SENSORS] = (
            (g[SENSORS] - r.mean) / sd).to_numpy()
    return out


# =================================================================
# 1. Is this batch odd?
# =================================================================

def batch_scores(df: pd.DataFrame, refs: dict[str, Reference],
                 cut: float = BATCH_CUT) -> pd.DataFrame:
    """Score each batch by its largest available absolute z-score."""
    z = z_scores(df, refs)
    a = z.abs()
    n_obs = a.notna().sum(axis=1)
    worst = a.max(axis=1)
    idx = a.idxmax(axis=1)
    signed = pd.Series(np.nan, index=df.index)
    ok = idx.notna()
    signed[ok] = z.to_numpy()[
        np.arange(len(df))[ok], [SENSORS.index(c) for c in idx[ok]]]
    return pd.DataFrame({
        "max_abs_z": worst,
        "worst_sensor": idx,
        "worst_z": signed,
        "n_observed": n_obs,
        "is_anomaly": worst >= cut,
    }, index=df.index)


def healthy_support(df: pd.DataFrame, refs: dict[str, Reference],
                    exclude=FAULTY_UNITS) -> dict:
    """Summarise the observed z-score support on healthy units."""
    d = df[~df.unit.isin(exclude)]
    s = batch_scores(d, refs, cut=np.inf).max_abs_z.dropna()
    return {"n": int(len(s)), "p99": float(s.quantile(0.99)),
            "p999": float(s.quantile(0.999)), "max": float(s.max()),
            "theoretical_max": float(np.sqrt(3.0))}


# =================================================================
# 2. Has this reactor moved?
# =================================================================

def cusum(z: np.ndarray, k: float, h: float):
    """Two-sided tally. Returns both arms and the first breach index."""
    cp = cm = 0.0
    up, dn = np.empty(len(z)), np.empty(len(z))
    first = -1
    for t, v in enumerate(z):
        if np.isnan(v):
            v = 0.0                     # a blank day accumulates nothing
        cp = max(0.0, cp + v - k)
        cm = max(0.0, cm - v - k)
        up[t], dn[t] = cp, cm
        if first < 0 and max(cp, cm) > h:
            first = t
    return up, dn, first


def confirmation_holds(up: np.ndarray, dn: np.ndarray, first: int,
                       h: float, confirm_days: int) -> bool:
    """Recheck either CUSUM arm a fixed number of observations later."""
    confirm_at = first + confirm_days
    return bool(first >= 0 and confirm_at < len(up)
                and max(up[confirm_at], dn[confirm_at]) > h)


def detect_changes(df: pd.DataFrame, refs: dict[str, Reference],
                   h: float = CUSUM_H, k: float = CUSUM_K,
                   confirm_days: int = CONFIRM_DAYS) -> pd.DataFrame:
    """Detect confirmed per-unit, per-sensor changes on daily means."""
    rows = []
    for unit, g in df.groupby("unit"):
        r = refs.get(unit)
        if r is None:
            continue
        daily = g.groupby(g.ts.dt.normalize())[SENSORS].mean().sort_index()
        after = daily.index >= r.end            # never score the reference
        for s in SENSORS:
            sd = r.daily_sd[s]
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (daily[s].to_numpy() - r.daily_mean[s]) / sd
            z = np.where(after, z, np.nan)
            up, dn, first = cusum(z, k, h)
            if first < 0:
                continue
            held = confirmation_holds(up, dn, first, h, confirm_days)
            rows.append({
                "unit": unit, "sensor": s,
                "alarm_date": daily.index[first],
                "direction": "up" if up[first] > dn[first] else "down",
                "peak_cusum": float(max(up.max(), dn.max())),
                "confirmed": held,
            })
    cols = ["unit", "sensor", "alarm_date", "direction", "peak_cusum",
            "confirmed"]
    return pd.DataFrame(rows, columns=cols)


def cusum_trace(df: pd.DataFrame, refs: dict[str, Reference], unit: str,
                sensor: str, k: float = CUSUM_K) -> pd.DataFrame:
    """The daily z and both tally arms, for plotting one unit."""
    r = refs[unit]
    g = df[df.unit == unit]
    daily = g.groupby(g.ts.dt.normalize())[sensor].mean().sort_index()
    z = (daily.to_numpy() - r.daily_mean[sensor]) / r.daily_sd[sensor]
    z = np.where(daily.index >= r.end, z, np.nan)
    up, dn, _ = cusum(z, k, np.inf)
    return pd.DataFrame({"date": daily.index, "value": daily.to_numpy(),
                         "z": z, "cusum_up": up, "cusum_down": dn})


# =================================================================

def main() -> None:
    print("=" * 74)
    print("  SIMPLE MONITOR - frozen z, then a tally")
    print("=" * 74)

    df = load()
    refs = fit_references(df)
    print(f"  {len(df):,} batches   {len(refs)} reactors referenced")
    r0 = next(iter(refs.values()))
    print(f"  reference window: {REFERENCE_DAYS} days, "
          f"{r0.start:%d %b} to {r0.end:%d %b}")

    sup = healthy_support(df, refs)
    print(f"\n  how far a healthy reactor reaches, over {sup['n']:,} "
          f"batches")
    print(f"    p99 {sup['p99']:.2f}   p99.9 {sup['p999']:.2f}   "
          f"observed max {sup['max']:.2f}   "
          f"uniform bound sqrt(3) = {sup['theoretical_max']:.2f}")
    print(f"  cut set at {BATCH_CUT:.1f}, which is above anything the "
          f"healthy fleet can produce")

    sc = batch_scores(df, refs)
    shifted = df.unit.isin(FAULTY_UNITS) & (df.ts >= CHANGE_DATE)
    ok = sc.max_abs_z.notna()
    print(f"\n  scorable batches: {ok.mean() * 100:.1f} %")
    print(f"  flagged, post-change faulty : "
          f"{sc.is_anomaly[ok & shifted].mean() * 100:5.1f} %")
    print(f"  flagged, everything else    : "
          f"{sc.is_anomaly[ok & ~shifted].mean() * 100:5.3f} %")

    ch = detect_changes(df, refs)
    conf = ch[ch.confirmed]
    real = conf[conf.unit.isin(FAULTY_UNITS)]
    false = conf[~conf.unit.isin(FAULTY_UNITS)]
    print(f"\n  confirmed signals: {len(conf)} on {conf.unit.nunique()} "
          f"units")
    print(f"    on the three faulty units: {len(real)}, peaks "
          f"{real.peak_cusum.min():.0f} to {real.peak_cusum.max():.0f}")
    print(f"    on healthy units         : {len(false)}, worst peak "
          f"{false.peak_cusum.max():.0f}")
    keep = conf[conf.peak_cusum > SEVERITY_CUT]
    print(f"  above the severity cut of {SEVERITY_CUT:.0f}: {len(keep)}, "
          f"all on faulty units: "
          f"{set(keep.unit) <= set(FAULTY_UNITS)}")
    print()
    print(keep.sort_values("peak_cusum", ascending=False).to_string(
        index=False))

    keep.to_csv(TABLES / "simple_monitor_changes.csv", index=False)
    print(f"\n  saved: simple_monitor_changes.csv\n")



# =================================================================
# The approach this one replaced — kept, but not wired to anything
# =================================================================
#
# Before freezing the reference we tried the obvious thing: give each
# unit an expanding history and z-score against that.
#
#     z_t = (x_t - mu_<t) / sigma_<t
#
# It was built causally, with exclusive cumulative sums so that a batch
# never sees its own timestamp, and it was measured properly. Two
# results killed it.
#
#   1. The signal expires. On the three faulty units the torque z faded
#      from -2.63 to -0.74 over seven weeks while the torque itself
#      moved eight thousandths of a N.m. An expanding mean drifts
#      toward the new regime until the fault becomes "normal for this
#      unit" and the alarm switches itself off.
#
#   2. It made the model worse where it mattered. Held out on the three
#      units it had never seen break, MAE rose 7.4 %, because the
#      z-score extrapolates further outside its training range than the
#      raw reading does.
#
# The frozen reference above is the fix for (1): mu and sigma stop
# moving after three weeks, so the -4.15 stays at -4.15. Nothing calls
# the function below; it is here so the rejected path is on the record
# rather than only in a sentence.
#
# def expanding_history(df, value_col, prefix, min_history=20, ddof=1):
#     """Per-unit history from every strictly earlier timestamp."""
#     centre = float(df[value_col].mean())
#     v = df[value_col] - centre                  # centred for stability
#     per = (df.assign(_v=v, _n=v.notna().astype(int),
#                      _s=v.fillna(0.0), _q=(v ** 2).fillna(0.0))
#              .groupby(["unit", "ts"], as_index=False)
#              .agg(n=("_n", "sum"), s=("_s", "sum"), q=("_q", "sum")))
#     g = per.groupby("unit", sort=False)
#     # Exclusive: subtract this timestamp's own totals back out.
#     N = (g["n"].cumsum() - per["n"]).to_numpy()
#     S = (g["s"].cumsum() - per["s"]).to_numpy()
#     Q = (g["q"].cumsum() - per["q"]).to_numpy()
#     hist_mean = np.where(N > 0, centre + S / N, np.nan)
#     var = np.where(N > ddof, (Q - S ** 2 / N) / (N - ddof), np.nan)
#     hist_std = np.sqrt(np.maximum(var, 0.0))
#     per[f"{prefix}_history_mean"] = hist_mean
#     per[f"{prefix}_history_std"] = hist_std
#     per["_N"] = N
#     merged = df[["unit", "ts"]].merge(
#         per[["unit", "ts", "_N", f"{prefix}_history_mean",
#              f"{prefix}_history_std"]], on=["unit", "ts"], how="left")
#     merged.index = df.index
#     usable = ((merged["_N"] >= min_history)
#               & (merged[f"{prefix}_history_std"] > 1e-6))
#     return np.where(
#         usable,
#         (df[value_col] - merged[f"{prefix}_history_mean"])
#         / merged[f"{prefix}_history_std"], np.nan)


if __name__ == "__main__":
    main()

"""Normalized split-conformal calibration of ensemble intervals."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

from model import FAULTY_UNITS, TARGET, TARGET_RANGE, load  # noqa: E402
from probabilistic_catboost import (OUT, TEST_START, VAL_START,
                                    ProbabilisticEnsemble)  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import FIG_UNCERTAINTY as FIG
pd.set_option("display.width", 200)

BLUE, RED, BAND = "#2a78d6", "#d03b3b", "#cde2fb"
INK, INK2, MUTED, GRID, SURF = ("#0b0b0b", "#52514e", "#898781",
                                "#e1e0d9", "#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "figure.dpi": 130,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "savefig.bbox": "tight",
})

LEVELS = (0.80, 0.90, 0.95)
MAIN = 0.90


def rule(t: str) -> None:
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


# ------------------------------------------------------------ the method

def conformal_q(scores: np.ndarray, coverage: float) -> float:
    """Return the finite-sample conformal quantile."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * coverage))
    return float("inf") if k > n else float(s[k - 1])


def interval(mu, sigma, q, clip: bool = True):
    lo, hi = mu - q * sigma, mu + q * sigma
    if clip:
        # 18-30 is the statement's *normal* range, not a physical bound:
        # unlike the sensors, the target is never declared "cannot be
        # negative", and the data shows no censoring - the density
        # tapers 3,768 -> 1,942 -> 687 -> 129 -> 3 into the floor with
        # nothing piled at exactly 18.0, and the top never reaches 28.7.
        # What justifies the trim is measurement, not physics: across
        # 259,200 batches nothing was ever observed below 18.015, so on
        # the batches this bites the discarded mass is 66 % of the
        # interval width and has never once contained the truth.
        # Coverage is identical to three decimals with or without it.
        lo = np.clip(lo, *TARGET_RANGE)
        hi = np.clip(hi, *TARGET_RANGE)
    return lo, hi


def present(mu, sigma, q):
    """Return point and bounds clipped consistently to the target range."""
    lo, hi = interval(mu, sigma, q, clip=True)
    return np.clip(mu, *TARGET_RANGE), lo, hi


def score_slice(df: pd.DataFrame, ens) -> pd.DataFrame:
    out = ens.predict(df)
    out["ts"] = df["ts"].to_numpy()
    out["n_missing"] = df["n_missing"].to_numpy()
    out["faulty"] = df["unit"].isin(FAULTY_UNITS).to_numpy()
    out["score"] = ((out.actual_months - out.predicted_months).abs()
                    / out.total_std)
    return out


# ------------------------------------------------------------ reporting

def summarise(d: pd.DataFrame, lo, hi, label: str) -> dict:
    y = d.actual_months.to_numpy()
    inside = (y >= lo) & (y <= hi)
    w = hi - lo
    row = {"calibration": label, "n_test": len(d),
           "coverage_pct": 100 * inside.mean(),
           "median_width": float(np.median(w)),
           "mean_width": float(w.mean())}
    return row


def by_group(d: pd.DataFrame, lo, hi, key: str) -> pd.DataFrame:
    y = d.actual_months.to_numpy()
    t = d.assign(_in=(y >= lo) & (y <= hi), _w=hi - lo)
    g = t.groupby(key)
    out = pd.DataFrame({
        "batches": g.size(),
        "coverage_pct": g._in.mean() * 100,
        "median_width": g._w.median(),
        "mean_width": g._w.mean(),
        "mean_sigma": g.total_std.mean(),
    })
    return out


def figure(tab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))
    x = np.arange(len(tab))
    lab = tab.index.astype(str)

    ax = axes[0]
    ax.bar(x, tab.coverage_pct, color=BLUE, width=0.62)
    ax.axhline(100 * MAIN, color=RED, lw=1.3)
    ax.text(len(x) - 0.45, 100 * MAIN + 1.4, f"target {MAIN:.0%}",
            ha="right", fontsize=8.5, color=RED, fontweight="600")
    ax.set_ylim(0, 105)
    ax.set_xticks(x, lab)
    ax.set_xlabel("sensors blank on the batch")
    ax.set_ylabel("true value inside the interval (%)")
    ax.set_title("Coverage holds however thin the data")

    ax = axes[1]
    ax.bar(x, tab.median_width, color=BLUE, width=0.62)
    ax.set_xticks(x, lab)
    ax.set_xlabel("sensors blank on the batch")
    ax.set_ylabel("interval width (months)")
    ax.set_title("...and the width follows the information lost")
    fig.savefig(path)
    plt.close(fig)


# =================================================================

def main() -> None:
    print("=" * 78)
    print("  NORMALIZED SPLIT CONFORMAL CALIBRATION")
    print("=" * 78)

    ens = ProbabilisticEnsemble().load_saved()
    df = load(include_temporal=ens.use_temporal_features)
    print("  five saved members reloaded; nothing retrained")

    val = df[(df.ts >= VAL_START) & (df.ts < TEST_START)]
    test = df[df.ts >= TEST_START]
    cal_v = score_slice(val, ens)
    te = score_slice(test, ens)
    print(f"  calibration A: validation slice {VAL_START:%d %b} to "
          f"{TEST_START:%d %b}, {len(cal_v):,} batches")
    print(f"  test         : from {TEST_START:%d %b}, {len(te):,} batches")

    # A clean second calibration: half of June, chosen at random, so
    # exchangeability with the other half is not in question.
    rng = np.random.default_rng(0)
    mask = rng.random(len(te)) < 0.5
    cal_j, te_j = te[mask], te[~mask]
    print(f"  calibration B: random half of June, {len(cal_j):,} "
          f"batches, evaluated on the other {len(te_j):,}")

    rule("THE CALIBRATION CONSTANT")
    print("  q multiplies sigma(x). Under a normal it would be 1.645 at")
    print("  90 %; anything below that says the residuals have shorter")
    print("  tails than the normal assumes.\n")
    qs = {}
    for lv in LEVELS:
        qa = conformal_q(cal_v.score, lv)
        qb = conformal_q(cal_j.score, lv)
        qs[lv] = (qa, qb)
        z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960}[lv]
        print(f"    {lv:.0%}   q(validation) {qa:.4f}    "
              f"q(June half) {qb:.4f}    normal would use {z:.3f}")

    rule(f"HEADLINE - {MAIN:.0%} INTERVALS ON THE TEST SET")
    rows = []
    qa, qb = qs[MAIN]
    lo, hi = interval(te.predicted_months.to_numpy(),
                      te.total_std.to_numpy(), qa)
    rows.append(summarise(te, lo, hi, "A  validation slice"))
    lo_j, hi_j = interval(te_j.predicted_months.to_numpy(),
                          te_j.total_std.to_numpy(), qb)
    rows.append(summarise(te_j, lo_j, hi_j, "B  random half of June"))
    lo_u, hi_u = interval(te.predicted_months.to_numpy(),
                          te.total_std.to_numpy(), qa, clip=False)
    rows.append(summarise(te, lo_u, hi_u, "A, without range clipping"))
    print()
    print(pd.DataFrame(rows).set_index("calibration").round(4).to_string())
    print("\n  A and B agreeing means the early-stopping contamination")
    print("  in the validation slice does not move the result.")

    rule("EVERY LEVEL, FROM THE SAME MODEL")
    lv_rows = []
    for lv in LEVELS:
        q = qs[lv][0]
        l2, h2 = interval(te.predicted_months.to_numpy(),
                          te.total_std.to_numpy(), q)
        y = te.actual_months.to_numpy()
        lv_rows.append({
            "nominal": f"{lv:.0%}", "q": round(q, 4),
            "coverage_pct": round(100 * ((y >= l2) & (y <= h2)).mean(), 2),
            "median_width": round(float(np.median(h2 - l2)), 3),
            "mean_width": round(float((h2 - l2).mean()), 3)})
    print()
    print(pd.DataFrame(lv_rows).to_string(index=False))
    print("\n  No refitting between rows: one scalar per level.")

    rule("THE POINT OF THE EXERCISE - BY NUMBER OF BLANK SENSORS")
    tab = by_group(te, lo, hi, "n_missing")
    print()
    print(tab.round(3).to_string())
    dev = (tab.coverage_pct - 100 * MAIN).abs()
    big = tab[tab.batches >= 50]
    print(f"\n  groups with at least 50 batches: coverage ranges "
          f"{big.coverage_pct.min():.1f} % to {big.coverage_pct.max():.1f} %")
    print(f"  worst deviation from {MAIN:.0%}: "
          f"{dev[big.index].max():.1f} points")
    print(f"  median width grows "
          f"{tab.median_width.iloc[0]:.2f} -> "
          f"{tab.median_width.iloc[-1]:.2f} months "
          f"({tab.median_width.iloc[-1] / tab.median_width.iloc[0]:.2f}x)")

    rule("CONDITIONAL COVERAGE ELSEWHERE")
    print("\n  faulty units against the rest:")
    print(by_group(te, lo, hi, "faulty").round(3).to_string())
    print("\n  by predicted shelf life - the symmetric score cannot")
    print("  correct the asymmetry we measured near the 18-month floor,")
    print("  so this is where it should show:")
    q6 = pd.qcut(te.predicted_months, 6, duplicates="drop")
    print(by_group(te.assign(_b=q6), lo, hi, "_b").round(3).to_string())

    p = FIG / "conformal_coverage_by_missing.png"
    figure(tab, p)
    tab.round(4).to_csv(OUT / "conformal_by_missing.csv")
    (OUT / "conformal_q.json").write_text(json.dumps(
        {f"{k:.2f}": {"q_validation": v[0], "q_june_half": v[1]}
         for k, v in qs.items()}, indent=2), encoding="utf-8")

    rule("OUTPUT")
    print(f"  {p.parent.name}/{p.name}")
    print(f"  {OUT.name}/conformal_by_missing.csv")
    print(f"  {OUT.name}/conformal_q.json")
    print()


if __name__ == "__main__":
    main()

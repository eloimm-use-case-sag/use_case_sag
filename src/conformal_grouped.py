"""Group-conditional conformal calibration with explicit fallbacks."""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from conformal_calibration import (FIG, LEVELS, MAIN, BLUE, RED, INK2,
                                   MUTED, conformal_q, interval,
                                   score_slice)                # noqa: E402
from model import FAULTY_UNITS, load                           # noqa: E402
from probabilistic_catboost import (OUT, TEST_START, VAL_START,
                                    ProbabilisticEnsemble)     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)


# ------------------------------------------------------- grouping rules

def by_missing_count(df: pd.DataFrame) -> pd.Series:
    """The default taxonomy: how many sensors were blank."""
    return df["n_missing"]


def by_faulty_unit(df: pd.DataFrame) -> pd.Series:
    """An alternative, to show the grouping is a parameter."""
    return df["unit"].isin(FAULTY_UNITS).map(
        {True: "faulty unit", False: "rest of fleet"})


def by_sigma_decile(df: pd.DataFrame) -> pd.Series:
    """Grouping on the model's own stated uncertainty."""
    return pd.qcut(df["total_std"], 10, labels=False, duplicates="drop")


# --------------------------------------------------------------- model

@dataclass
class GroupedConformal:
    """One conformal quantile per group, with a configured fallback."""

    coverage: float = MAIN
    group_fn: Callable[[pd.DataFrame], pd.Series] = by_missing_count
    group_name: str = "n_missing"
    min_group: int = 100
    fallback: str = "global"
    clip: bool = True
    q_: dict = field(default_factory=dict, init=False)
    q_global_: float = field(default=np.nan, init=False)
    table_: pd.DataFrame = field(default=None, init=False)

    def fit(self, cal: pd.DataFrame) -> "GroupedConformal":
        if self.fallback not in ("global", "nearest"):
            raise ValueError("fallback must be 'global' or 'nearest'")
        scores = cal["score"].to_numpy()
        groups = self.group_fn(cal)
        self.q_global_ = conformal_q(scores, self.coverage)

        # First pass: whoever has enough calibration data earns its own.
        own, rows = {}, []
        for g, idx in groups.groupby(groups).groups.items():
            s = cal.loc[idx, "score"].to_numpy()
            q = conformal_q(s, self.coverage)
            eligible = len(s) >= self.min_group and np.isfinite(q)
            if eligible:
                own[g] = q
            rows.append({self.group_name: g, "n_calibration": len(s),
                         "own_q": q, "eligible": eligible})

        # Second pass: the thin groups take what the fallback gives.
        for r in rows:
            g = r[self.group_name]
            if r["eligible"]:
                r["q_used"], r["source"] = own[g], "own"
            elif self.fallback == "global" or not own:
                r["q_used"], r["source"] = self.q_global_, "global"
            else:
                donor = self._nearest(g, own)
                if donor is None:
                    r["q_used"], r["source"] = self.q_global_, "global"
                else:
                    r["q_used"] = own[donor]
                    r["source"] = f"from group {donor}"
            self.q_[g] = r["q_used"]

        self.table_ = pd.DataFrame(rows).set_index(self.group_name)
        return self

    @staticmethod
    def _nearest(g, own: dict):
        """Closest eligible group, when the labels are numeric."""
        try:
            gv = float(g)
            cand = [(abs(float(k) - gv), k) for k in own]
        except (TypeError, ValueError):
            return None
        return min(cand)[1] if cand else None

    def q_for(self, df: pd.DataFrame) -> np.ndarray:
        """The q each row is entitled to. Unseen groups get the pool."""
        g = self.group_fn(df)
        return g.map(self.q_).fillna(self.q_global_).to_numpy()

    def predict_interval(self, df: pd.DataFrame):
        return interval(df["predicted_months"].to_numpy(),
                        df["total_std"].to_numpy(),
                        self.q_for(df), clip=self.clip)


# ------------------------------------------------------------ scoring

def evaluate(df: pd.DataFrame, lo, hi, group_name: str,
             groups: pd.Series | None = None) -> pd.DataFrame:
    y = df["actual_months"].to_numpy()
    t = df.assign(_in=(y >= lo) & (y <= hi), _w=hi - lo,
                  _g=groups if groups is not None else df[group_name])
    g = t.groupby("_g")
    out = pd.DataFrame({
        "batches": g.size(),
        "coverage_pct": g._in.mean() * 100,
        "median_width": g._w.median(),
        "mean_width": g._w.mean(),
    })
    out.index.name = group_name
    return out


def headline(df, lo, hi) -> dict:
    y = df["actual_months"].to_numpy()
    inside = (y >= lo) & (y <= hi)
    w = hi - lo
    return {"coverage_pct": 100 * inside.mean(),
            "median_width": float(np.median(w)),
            "mean_width": float(w.mean())}


def figure(glob: pd.DataFrame, grp: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7))
    x = np.arange(len(glob))
    lab = glob.index.astype(str)
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, glob.coverage_pct, width=w, color=MUTED,
           label="one global q")
    ax.bar(x + w / 2, grp.coverage_pct, width=w, color=BLUE,
           label="one q per group")
    ax.axhline(100 * MAIN, color=RED, lw=1.3)
    ax.text(len(x) - 0.4, 100 * MAIN + 1.6, f"target {MAIN:.0%}",
            ha="right", fontsize=8.5, color=RED, fontweight="600")
    ax.set_ylim(70, 104)
    ax.set_xticks(x, lab)
    ax.set_xlabel("sensors blank on the batch")
    ax.set_ylabel("coverage (%)")
    ax.legend(loc="lower left")
    ax.set_title("Coverage inside each group")

    ax = axes[1]
    ax.bar(x - w / 2, glob.median_width, width=w, color=MUTED,
           label="one global q")
    ax.bar(x + w / 2, grp.median_width, width=w, color=BLUE,
           label="one q per group")
    ax.set_xticks(x, lab)
    ax.set_xlabel("sensors blank on the batch")
    ax.set_ylabel("interval width (months)")
    ax.legend(loc="upper left")
    ax.set_title("What that costs in width")
    fig.savefig(path)
    plt.close(fig)


# =================================================================

def main() -> None:
    print("=" * 78)
    print("  GROUP-CONDITIONAL (MONDRIAN) CONFORMAL CALIBRATION")
    print("=" * 78)

    ens = ProbabilisticEnsemble().load_saved()
    df = load(include_temporal=ens.use_temporal_features)
    cal = score_slice(df[(df.ts >= VAL_START) & (df.ts < TEST_START)],
                      ens)
    te = score_slice(df[df.ts >= TEST_START], ens)
    te["unit"] = df.loc[df.ts >= TEST_START, "unit"].to_numpy()
    cal["unit"] = df.loc[(df.ts >= VAL_START)
                         & (df.ts < TEST_START), "unit"].to_numpy()
    print(f"  calibration {len(cal):,}   test {len(te):,}   "
          f"target {MAIN:.0%}")

    gc = GroupedConformal(coverage=MAIN, min_group=100,
                          fallback="global").fit(cal)

    print(f"\n{'=' * 78}\n  THE LEARNED CONSTANTS\n{'=' * 78}")
    print(f"  pooled q (what the global method uses): "
          f"{gc.q_global_:.4f}\n")
    print(gc.table_.round(4).to_string())
    print("\n  Groups 4 and 5 fall back: 22 and 2 calibration batches")
    print("  are not enough to estimate a 90 % quantile, and their own")
    print("  q would be the maximum score rather than a quantile.")

    lo_g, hi_g = interval(te.predicted_months.to_numpy(),
                          te.total_std.to_numpy(), gc.q_global_)
    lo_m, hi_m = gc.predict_interval(te)

    print(f"\n{'=' * 78}\n  MARGINAL RESULT\n{'=' * 78}\n")
    print(pd.DataFrame({
        "one global q": headline(te, lo_g, hi_g),
        "one q per group": headline(te, lo_m, hi_m),
    }).T.round(4).to_string())

    print(f"\n{'=' * 78}\n  CONDITIONAL RESULT - THE POINT OF THIS\n"
          f"{'=' * 78}\n")
    a = evaluate(te, lo_g, hi_g, "n_missing")
    b = evaluate(te, lo_m, hi_m, "n_missing")
    cmp = pd.DataFrame({
        "batches": a.batches,
        "cov_global": a.coverage_pct, "cov_grouped": b.coverage_pct,
        "width_global": a.median_width, "width_grouped": b.median_width,
    })
    cmp["cov_gain"] = (b.coverage_pct - 100 * MAIN).abs() \
        - (a.coverage_pct - 100 * MAIN).abs()
    print(cmp.round(3).to_string())
    big = cmp[cmp.batches >= 50]
    print(f"\n  worst deviation from {MAIN:.0%}, groups over 50 batches:")
    print(f"    one global q    {(big.cov_global - 100*MAIN).abs().max():.2f} points")
    print(f"    one q per group {(big.cov_grouped - 100*MAIN).abs().max():.2f} points")
    print(f"  spread of coverage across those groups:")
    print(f"    one global q    {big.cov_global.max() - big.cov_global.min():.2f} points")
    print(f"    one q per group {big.cov_grouped.max() - big.cov_grouped.min():.2f} points")

    print(f"\n{'=' * 78}\n  THE PARAMETERS, EXERCISED\n{'=' * 78}\n")
    variants = [
        ("n_missing, min 100, fallback global",
         dict(group_fn=by_missing_count, group_name="n_missing",
              min_group=100, fallback="global")),
        ("n_missing, min 100, fallback nearest",
         dict(group_fn=by_missing_count, group_name="n_missing",
              min_group=100, fallback="nearest")),
        ("n_missing, min 500 (stricter)",
         dict(group_fn=by_missing_count, group_name="n_missing",
              min_group=500, fallback="global")),
        ("faulty unit vs rest of fleet",
         dict(group_fn=by_faulty_unit, group_name="unit_group",
              min_group=100, fallback="global")),
        ("decile of the model's own sigma",
         dict(group_fn=by_sigma_decile, group_name="sigma_decile",
              min_group=100, fallback="global")),
    ]
    rows = []
    for label, kw in variants:
        m = GroupedConformal(coverage=MAIN, **kw).fit(cal)
        lo, hi = m.predict_interval(te)
        r = headline(te, lo, hi)
        by = evaluate(te, lo, hi, "n_missing")
        bigby = by[by.batches >= 50]
        r["worst_group_dev"] = float(
            (bigby.coverage_pct - 100 * MAIN).abs().max())
        r["groups"] = len(m.table_)
        r["fell_back"] = int((m.table_.source != "own").sum())
        rows.append({"configuration": label, **r})
    print(pd.DataFrame(rows).set_index("configuration").round(3)
          .to_string())
    print("\n  worst_group_dev is measured by n_missing in every row, so")
    print("  the last two show what grouping on something else buys - or")
    print("  fails to buy - for that particular breakdown.")

    p = FIG / "conformal_grouped_vs_global.png"
    figure(a, b, p)
    cmp.round(4).to_csv(OUT / "conformal_grouped_comparison.csv")
    gc.table_.round(6).to_csv(OUT / "conformal_grouped_q.csv")
    (OUT / "conformal_grouped_q.json").write_text(json.dumps(
        {"q_global": gc.q_global_,
         "q_by_group": {str(k): v for k, v in gc.q_.items()},
         "min_group": gc.min_group, "fallback": gc.fallback,
         "coverage": gc.coverage}, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}\n  OUTPUT\n{'=' * 78}")
    print(f"  {p.parent.name}/{p.name}")
    print(f"  {OUT.name}/conformal_grouped_comparison.csv")
    print(f"  {OUT.name}/conformal_grouped_q.csv")
    print(f"  {OUT.name}/conformal_grouped_q.json")
    print()


if __name__ == "__main__":
    main()

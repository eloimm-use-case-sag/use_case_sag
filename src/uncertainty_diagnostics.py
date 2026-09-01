"""Generate the two uncertainty diagnostics shown in Streamlit."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import CHANGE_DATE, FAULTY_UNITS, load
from paths import FIG_UNCERTAINTY, TABLES
from probabilistic_catboost import ProbabilisticEnsemble

BLUE = "#2a78d6"
RED = "#d83a3a"
FILL = "#cde2fb"
GRID = "#deddd7"
INK = "#30302e"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b7b5ad")
    ax.tick_params(colors="#55534f")


def aleatoric_vs_missing(pred: pd.DataFrame) -> plt.Figure:
    """Plot aleatoric uncertainty as sensor readings disappear."""
    g = pred.groupby("n_missing")["aleatoric_std"]
    d = g.agg(n="size", median="median",
              q25=lambda x: x.quantile(0.25),
              q75=lambda x: x.quantile(0.75)).reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.fill_between(d.n_missing, d.q25, d.q75, color=FILL,
                    label="interquartile range")
    ax.plot(d.n_missing, d["median"], color=BLUE, marker="o",
            linewidth=2, label="median")
    ax.set(title="Aleatoric uncertainty as information is lost",
           xlabel="sensors blank on the batch",
           ylabel="aleatoric uncertainty (months)")
    ax.set_xticks(d.n_missing)
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def epistemic_regime_change(df: pd.DataFrame,
                            pred: pd.DataFrame) -> plt.Figure:
    """Plot daily ensemble disagreement around the known regime change."""
    d = df[["batch_id", "ts", "unit"]].merge(
        pred[["batch_id", "epistemic_std"]], on="batch_id",
        how="inner", validate="one_to_one")
    d["date"] = d.ts.dt.normalize()
    d["group"] = np.where(d.unit.isin(FAULTY_UNITS),
                          "Plant-04 / R03, R07, R08", "rest of the fleet")
    daily = (d.groupby(["date", "group"], as_index=False)
             .epistemic_std.mean())

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for name, colour in (("rest of the fleet", BLUE),
                         ("Plant-04 / R03, R07, R08", RED)):
        s = daily[daily.group == name]
        ax.plot(s.date, s.epistemic_std, color=colour, linewidth=1.7,
                label=name)
    ax.axvline(CHANGE_DATE, color=INK, linewidth=1.4)
    ax.annotate("8 May", (CHANGE_DATE, 1), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points",
                ha="left", va="top", color="#55534f")
    ax.set(title="Daily disagreement, with the regime change marked",
           ylabel="daily mean epistemic uncertainty (months)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def main() -> None:
    ens = ProbabilisticEnsemble().load_saved()
    df = load(include_temporal=ens.use_temporal_features)
    test = pd.read_parquet(TABLES / "test_predictions.parquet")
    full = ens.predict(df)

    outputs = (
        (aleatoric_vs_missing(test),
         FIG_UNCERTAINTY / "diag1_aleatoric_vs_missing.png"),
        (epistemic_regime_change(df, full),
         FIG_UNCERTAINTY / "diag2_epistemic_regime_change.png"),
    )
    for fig, path in outputs:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()

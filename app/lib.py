"""Shared plumbing for the Streamlit app: theme, caching, figures.

Everything expensive is cached. The prediction pipeline loads the five
CatBoost members and fits the conformal calibration once per session.
The frozen-z monitor has its own cached reference per reactor. After
that a widget move reuses both objects, which keeps the simulator usable
live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ------------------------------------------------------------- palette

BLUE, RED, BAND = "#2a78d6", "#d03b3b", "#cde2fb"
AQUA, AMBER = "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = ("#0b0b0b", "#52514e", "#898781",
                                "#e1e0d9", "#fcfcfb")
GRADE_COLOURS = {"Technical Grade": "#86b6ef",
                 "Industrial Grade": "#2a78d6",
                 "Perfumery Grade": "#104281"}

SENSOR_LABEL = {
    "temp_c": "Reaction temperature (°C)",
    "vessel_pressure_bar": "Vessel pressure (bar)",
    "refractive_index": "Refractive index",
    "density_g_cm3": "Density (g/cm³)",
    "ph_level": "pH level",
    "mixing_torque_nm": "Mixing torque (N·m)",
    "mass_flow_rate_kg_h": "Mass flow rate (kg/h)",
    # not sensors: the three drift features and the context columns
    "torque_z_ref": "Torque vs its own normal (σ)",
    "torque_cusum_up": "Evidence torque is rising",
    "torque_cusum_down": "Evidence torque is falling",
    "n_missing": "Sensors not reporting",
    "unit": "Which reactor",
    "reactor": "Reactor model",
}


def label_of(key: str) -> str:
    """A readable name for any feature, never a KeyError."""
    return SENSOR_LABEL.get(key, str(key)).split(" (")[0]
SPEC = {
    "temp_c": (75.0, 100.0), "vessel_pressure_bar": (2.0, 3.0),
    "refractive_index": (1.43, 1.49), "density_g_cm3": (0.80, 0.90),
    "ph_level": (5.0, 6.0), "mixing_torque_nm": (9.0, 12.5),
    "mass_flow_rate_kg_h": (380.0, 520.0),
}

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1280px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .kpi {background:#f4f4f1; border-radius:10px; padding:14px 16px;
        border:1px solid #e6e5df; height:100%;}
  .kpi .v {font-size:1.65rem; font-weight:700; color:#2a78d6;
           line-height:1.15;}
  .kpi .v.bad {color:#d03b3b;}
  .kpi .l {font-size:0.76rem; color:#52514e; margin-top:2px;
           line-height:1.3;}
  .lead {font-size:1.05rem; color:#333; line-height:1.55;}
  .pill {display:inline-block; padding:3px 11px; border-radius:999px;
         font-size:0.76rem; font-weight:600;}
  .pill.ok  {background:#e2eefb; color:#1c5cab;}
  .pill.warn{background:#fdf1dd; color:#8a5b06;}
  .pill.bad {background:#fbe4e4; color:#a02a2a;}
  .note {border-left:3px solid #cde2fb; padding:2px 0 2px 12px;
         color:#52514e; font-size:0.9rem;}
  .point {background:#eef4fd; border-left:4px solid #2a78d6;
          padding:11px 15px; margin:4px 0 16px 0; color:#12365e;
          font-size:1.0rem; line-height:1.5; border-radius:0 8px 8px 0;}
  .point b {color:#0b2748;}
  div[data-testid="stMetricValue"] {font-size:1.5rem;}
</style>
"""


def kpi(col, value: str, label: str, bad: bool = False) -> None:
    col.markdown(
        f'<div class="kpi"><div class="v{" bad" if bad else ""}">{value}'
        f'</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True)


def pill(text: str, kind: str = "ok") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def point(text: str) -> None:
    """One short line the presenter can read out loud.

    Every section gets exactly one. Short sentences, plain words: this
    is the spoken script, not the footnote.
    """
    st.markdown(f'<p class="point">{text}</p>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, legend: bool = True):
    # Keep whatever title the caller set. Passing title=dict(font=...)
    # without a `text` key builds a title object that serialises with no
    # text at all, and plotly.js renders that as the literal word
    # "undefined" in the top-left corner of the chart.
    text = fig.layout.title.text or ""
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=10, r=10, t=42 if text else 16, b=10),
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12,
                  color=INK2),
        title=dict(text=text, font=dict(size=14.5, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", y=1.03, x=0, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# --------------------------------------------------------- the pipeline

@st.cache_resource(show_spinner="Loading the model and detectors…")
def get_pipeline():
    from batch_report import Pipeline
    return Pipeline()


@st.cache_data(show_spinner=False)
def get_clean() -> pd.DataFrame:
    return get_pipeline().clean


@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    from paths import RAW_CSV
    return pd.read_csv(RAW_CSV, low_memory=False)


@st.cache_data(show_spinner=False)
def constants():
    from model import CHANGE_DATE, FAULTY_UNITS, SENSORS
    return list(SENSORS), list(FAULTY_UNITS), CHANGE_DATE


# ------------------------------------------------------ derived tables

@st.cache_data(show_spinner=False)
def daily_by_unit(col: str) -> pd.DataFrame:
    df = get_clean()
    d = (df.groupby([df.ts.dt.normalize(), "unit"])[col]
         .mean().reset_index())
    d.columns = ["date", "unit", "value"]
    return d


@st.cache_data(show_spinner=False)
def unit_summary() -> pd.DataFrame:
    df = get_clean()
    _, faulty, change = constants()
    g = df.groupby("unit")
    t = pd.DataFrame({
        "batches": g.size(),
        "shelf_life": g.stability_months.mean(),
        "torque": g.mixing_torque_nm.mean(),
        "plant": g.plant_location.first(),
        "reactor": g.reactor.first(),
    }).reset_index()
    t["state"] = np.where(t.unit.isin(faulty), "regime shift", "stable")
    return t.sort_values("shelf_life")


@st.cache_data(show_spinner=False)
def driver_table() -> pd.DataFrame:
    df = get_clean()
    sensors, _, _ = constants()
    rows = []
    for s in sensors:
        d = df[[s, "stability_months"]].dropna()
        dec = d.groupby(pd.qcut(d[s], 10, duplicates="drop"),
                        observed=True)["stability_months"].mean()
        rows.append({
            "sensor": SENSOR_LABEL[s], "key": s,
            "swing": float(dec.max() - dec.min()),
            "corr": float(d[s].corr(d.stability_months)),
            "shape": ("rises" if dec.values.argmax() == len(dec) - 1
                      else "falls" if dec.values.argmax() == 0
                      else "peaks in the middle"),
        })
    return pd.DataFrame(rows).sort_values("swing", ascending=False)


@st.cache_data(show_spinner=False)
def decile_profile(col: str) -> pd.DataFrame:
    df = get_clean()
    d = df[[col, "stability_months"]].dropna()
    q = pd.qcut(d[col], 10, duplicates="drop")
    g = d.groupby(q, observed=True).agg(
        reading=(col, "mean"), shelf=("stability_months", "mean"),
        n=(col, "size"))
    return g.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def shap_sample(n: int = 6000):
    """SHAP over a slice of the test period, for the global views."""
    from explain import shap_contributions
    from probabilistic_catboost import TEST_START
    p = get_pipeline()
    df = get_clean()
    s = df[df.ts >= TEST_START].sample(n, random_state=0)
    contrib, base = shap_contributions(p.ens, s)
    return s.reset_index(drop=True), contrib.reset_index(drop=True), base


@st.cache_data(show_spinner=False)
def defects_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"defect": "Two targets can cause data leakage: purity_grade and oxidation_risk_flag", "n": "259,200 rows"},
        {"defect": "Temperature recorded in two units", "n": "1,506 rows"},
        {"defect": "Failure codes stored as measurements",
         "n": "2,171 cells"},
        {"defect": "Three date formats, one ambiguous", "n": "25,107 rows"},
        {"defect": "Reactor IDs repeat across plants", "n": "37 real units"},
        {"defect": "Blank readings, incl. 5,344 blackouts",
         "n": "105,869 rows"},
    ])


# ----------------------------------------------------------- prediction

def predict_row(row: pd.DataFrame) -> dict:
    """One assembled batch through model, interval and SHAP."""
    from conformal_calibration import present
    from explain import shap_contributions
    from model import to_grade, to_oxidation_risk
    p = get_pipeline()

    pred = p.ens.predict(row)
    tot = float(pred.total_std.iloc[0])
    q = p.gc.q_.get(int(row.n_missing.iloc[0]), p.gc.q_global_)
    mu, lo, hi = present(float(pred.predicted_months.iloc[0]), tot, q)
    mu, lo, hi = float(mu), float(lo), float(hi)

    contrib, base = shap_contributions(p.ens, row)
    return {
        "mu": mu, "lo": lo, "hi": hi,
        "aleatoric": float(pred.aleatoric_std.iloc[0]),
        "epistemic": float(pred.epistemic_std.iloc[0]),
        "total": tot, "q": q,
        "grade": str(to_grade(mu)),
        "oxidation": bool(to_oxidation_risk(mu)),
        "contrib": contrib.iloc[0], "base": base,
    }


def grades_possible(lo: float, hi: float) -> list[str]:
    out = []
    for name, a, b in (("Technical Grade", -np.inf, 20.0),
                       ("Industrial Grade", 20.0, 25.0),
                       ("Perfumery Grade", 25.0, np.inf)):
        if hi > a and lo < b:
            out.append(name)
    return out


# --------------------------------------------------------------- plots

def fig_step_change(col: str = "mixing_torque_nm") -> go.Figure:
    """Daily mean per reactor: the healthy band against the three."""
    _, faulty, change = constants()
    d = daily_by_unit(col)
    healthy = d[~d.unit.isin(faulty)]
    band = healthy.groupby("date")["value"].agg(
        lo=lambda s: s.quantile(0.10), hi=lambda s: s.quantile(0.90),
        mid="median").reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=band.date, y=band.hi, line=dict(width=0), showlegend=False,
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=band.date, y=band.lo, fill="tonexty", line=dict(width=0),
        fillcolor="rgba(205,226,251,0.85)", name="fleet p10–p90",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=band.date, y=band.mid, line=dict(color=BLUE, width=2),
        name="rest of the fleet (median)"))
    for i, u in enumerate(faulty):
        g = d[d.unit == u]
        fig.add_trace(go.Scatter(
            x=g.date, y=g.value, line=dict(color=RED, width=1.4),
            name="Plant-04 / R03, R07, R08" if i == 0 else None,
            showlegend=i == 0, legendgroup="faulty",
            hovertemplate=f"{u}<br>%{{x|%d %b}}  %{{y:.2f}}"
                          "<extra></extra>"))
    fig.add_vline(x=change, line=dict(color=INK2, width=1.2))
    fig.add_annotation(x=change, y=1.0, yref="paper", text="  8 May",
                       showarrow=False, xanchor="left",
                       font=dict(size=11, color=INK2))
    fig.update_yaxes(title=SENSOR_LABEL.get(col, col))
    return style(fig, 400)


def fig_in_spec() -> go.Figure:
    """The argument: the fault lived inside the declared band."""
    df = get_clean()
    _, faulty, change = constants()
    post = df[(df.ts >= change) & df.mixing_torque_nm.notna()]
    lo, hi = SPEC["mixing_torque_nm"]

    fig = go.Figure()
    for mask, name, colour in (
            (~post.unit.isin(faulty), "rest of the fleet", BLUE),
            (post.unit.isin(faulty), "Plant-04 / R03, R07, R08", RED)):
        fig.add_trace(go.Histogram(
            x=post.loc[mask, "mixing_torque_nm"], name=name,
            marker_color=colour, opacity=0.9, nbinsx=110,
            histnorm="probability density"))
    fig.add_vrect(x0=lo, x1=hi, fillcolor=GRID, opacity=0.45,
                  line_width=0, layer="below")
    for x in (lo, hi):
        fig.add_vline(x=x, line=dict(color=INK2, width=1.2))
    fig.add_annotation(x=(lo + hi) / 2, y=1.0, yref="paper",
                       text="declared normal range 9 – 12.5 N·m",
                       showarrow=False, yanchor="top",
                       font=dict(size=11.5, color=INK2))
    fig.update_layout(barmode="overlay",
                      xaxis_title="Mixing torque (N·m)",
                      yaxis_title="density")
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(range=[8.6, 13.1])
    return style(fig, 360)


def fig_units() -> go.Figure:
    """Every plant x reactor, ranked by mean shelf life."""
    t = unit_summary()
    colours = [RED if s == "regime shift" else BLUE for s in t.state]
    fig = go.Figure(go.Bar(
        x=t.shelf_life, y=t.unit, orientation="h",
        marker_color=colours,
        hovertemplate="%{y}<br>%{x:.2f} months<extra></extra>"))
    fig.update_layout(xaxis_title="mean shelf life (months)",
                      xaxis_range=[18, t.shelf_life.max() + 0.3])
    return style(fig, 720, legend=False)


def fig_effect(col: str) -> go.Figure:
    """A sensor's observed effect across its own range."""
    d = decile_profile(col)
    fig = go.Figure(go.Scatter(
        x=d.reading, y=d.shelf, mode="lines+markers",
        line=dict(color=BLUE, width=2.4), marker=dict(size=8),
        hovertemplate="%{x:.3f} → %{y:.2f} months<extra></extra>"))
    fig.update_layout(xaxis_title=SENSOR_LABEL[col],
                      yaxis_title="mean shelf life (months)")
    return style(fig, 330, legend=False)


def fig_waterfall(contrib: pd.Series, base: float,
                  labels: dict) -> go.Figure:
    """From the fleet average to this batch, contribution by contribution."""
    c = contrib[contrib.abs() > 0.005].sort_values(key=abs,
                                                   ascending=False)
    names = [labels.get(k, k) for k in c.index]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(c) + ["total"],
        x=names + ["prediction"],
        y=list(c.values) + [0],
        base=base,
        decreasing=dict(marker_color=RED),
        increasing=dict(marker_color=BLUE),
        totals=dict(marker_color=INK2),
        connector=dict(line=dict(color=GRID)),
        hovertemplate="%{x}<br>%{delta:+.2f} months<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="months",
        title=f"Starting from the fleet average of {base:.2f} months")
    fig.update_xaxes(tickangle=-35)
    return style(fig, 400, legend=False)


def fig_interval(mu: float, lo: float, hi: float,
                 actual: float | None = None) -> go.Figure:
    """The prediction on the shelf-life scale, with the grade bands."""
    fig = go.Figure()
    for name, a, b in (("Technical", 18.0, 20.0),
                       ("Industrial", 20.0, 25.0),
                       ("Perfumery", 25.0, 29.0)):
        fig.add_vrect(x0=a, x1=b, fillcolor=GRADE_COLOURS[f"{name} Grade"],
                      opacity=0.13, line_width=0, layer="below",
                      annotation_text=name, annotation_position="top",
                      annotation_font_size=10.5)
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[0, 0], mode="lines",
        line=dict(color=BLUE, width=9), name="90 % interval",
        hovertemplate="%{x:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[mu], y=[0], mode="markers", name="prediction",
        marker=dict(color=INK, size=15, symbol="line-ns",
                    line=dict(width=3, color=INK))))
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=[actual], y=[0], mode="markers", name="lab result",
            marker=dict(color=RED, size=13, symbol="diamond")))
    fig.update_yaxes(visible=False, range=[-0.5, 0.5])
    fig.update_xaxes(range=[17.8, 29.0], title="shelf life (months)")
    return style(fig, 210)


# ------------------------------------------------- how good is it

@st.cache_data(show_spinner=False)
def quality_frame() -> pd.DataFrame:
    """Test-window predictions joined to the context they need.

    Read from the saved parquet rather than re-run, so the page opens
    on the same numbers the evaluation reported.
    """
    from paths import TABLES
    t = pd.read_parquet(TABLES / "test_predictions.parquet")
    d = get_clean().set_index("batch_id")
    t["ts"] = d.ts.reindex(t.batch_id).to_numpy()
    return t


@st.cache_data(show_spinner=False)
def calibration_curve() -> pd.DataFrame:
    """Asked coverage against delivered, over the whole range.

    One conformal q per level, fitted on the validation window and
    measured on the test window - the same split the headline uses.
    """
    from conformal_calibration import conformal_q
    from probabilistic_catboost import TEST_START, VAL_START

    p = get_pipeline()
    d = get_clean()
    cal = d[(d.ts >= VAL_START) & (d.ts < TEST_START)]
    pc = p.ens.predict(cal)
    s = ((cal.stability_months.to_numpy() - pc.predicted_months.to_numpy())
         / pc.total_std.to_numpy())
    scores = np.abs(s)

    t = quality_frame()
    mu = t.predicted_months.to_numpy()
    sd = t.total_std.to_numpy()
    y = t.actual_months.to_numpy()
    rows = []
    for asked in np.arange(0.50, 0.991, 0.05):
        q = conformal_q(scores, float(asked))
        lo, hi = np.clip(mu - q * sd, 18, 30), np.clip(mu + q * sd, 18, 30)
        rows.append({"asked": asked * 100,
                     "delivered": ((lo <= y) & (y <= hi)).mean() * 100,
                     "width": float(np.median(hi - lo))})
    return pd.DataFrame(rows)


def fig_pred_vs_actual(n: int = 12000) -> go.Figure:
    """The plot every data scientist looks for first."""
    t = quality_frame().sample(n, random_state=0)
    y, mu = t.actual_months.to_numpy(), t.predicted_months.to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=y, y=mu, mode="markers", name="test batches",
        marker=dict(color=BLUE, size=3, opacity=0.22),
        hovertemplate="lab %{x:.2f}<br>predicted %{y:.2f}<extra></extra>"))
    line = [17.9, 29.0]
    fig.add_trace(go.Scatter(
        x=line, y=line, mode="lines", name="perfect prediction",
        line=dict(color=INK, width=1.4, dash="dash")))
    fig.update_xaxes(title="what the lab measured (months)", range=line)
    fig.update_yaxes(title="what the model predicted (months)", range=line)
    return style(fig, 430)


def fig_calibration() -> go.Figure:
    """Asked against delivered coverage - the conformal promise, tested."""
    c = calibration_curve()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[50, 99], y=[50, 99], mode="lines", name="perfect calibration",
        line=dict(color=INK, width=1.4, dash="dash")))
    fig.add_trace(go.Scatter(
        x=c.asked, y=c.delivered, mode="lines+markers", name="delivered",
        line=dict(color=BLUE, width=2.4), marker=dict(size=7),
        hovertemplate="asked %{x:.0f} %<br>got %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title="coverage asked for (%)")
    fig.update_yaxes(title="coverage actually delivered (%)")
    return style(fig, 400)


def fig_coverage_by_missing() -> go.Figure:
    """One q for everyone against one q per missing-sensor group."""
    from paths import TABLES
    g = pd.read_csv(TABLES / "conformal_grouped_comparison.csv")
    g = g[g.batches >= 100]
    fig = go.Figure()
    fig.add_hline(y=90, line=dict(color=INK, width=1.3, dash="dash"))
    for col, name, colour in (("cov_global", "one q for everyone", RED),
                              ("cov_grouped", "one q per group", BLUE)):
        fig.add_trace(go.Bar(
            x=g.n_missing, y=g[col], name=name, marker_color=colour,
            hovertemplate="%{x} blank: %{y:.1f} %<extra>" + name
                          + "</extra>"))
    fig.update_xaxes(title="sensors with no reading", dtick=1)
    fig.update_yaxes(title="coverage delivered (%)", range=[80, 95])
    return style(fig, 380)


# --------------------------------------------- spec, before and after

# What the acquisition system writes when a sensor fails, per column.
# These are the values the cleaning turns into a declared absence.
FAILURE_CODES = {
    "temp_c": (210.0, 410.0), "vessel_pressure_bar": (-1.0,),
    "ph_level": (14.0,), "density_g_cm3": (860.0,),
}
RAW_COLUMN = {"temp_c": "temp_value"}


@st.cache_data(show_spinner=False)
def spec_comparison(sensor: str) -> dict:
    """What the file delivered for one sensor against what survived.

    The histogram on this page reads the repaired data, which is the
    only scale that works - a density code of 860 against readings near
    0.86 would flatten the axis to a single bar. So the numbers that
    would otherwise be invisible are reported here instead.
    """
    lo, hi = SPEC[sensor]
    raw = pd.to_numeric(get_raw()[RAW_COLUMN.get(sensor, sensor)],
                        errors="coerce")
    clean = get_clean()[sensor]
    codes = FAILURE_CODES.get(sensor, ())
    return {
        "codes": codes,
        "n_codes": int(raw.isin(codes).sum()) if codes else 0,
        "out_raw": float(((raw < lo) | (raw > hi)).sum()
                         / raw.notna().sum() * 100),
        "out_clean": float(((clean < lo) | (clean > hi)).sum()
                           / clean.notna().sum() * 100),
        "blank_raw": float(raw.isna().mean() * 100),
        "blank_clean": float(clean.isna().mean() * 100),
    }


def fig_spec_hist(sensor: str, delivered: bool = False) -> go.Figure:
    """Distribution against the declared band, before or after repair.

    "As delivered" has a scale problem: a density failure code of 860
    sits three orders of magnitude from readings near 0.86, so a single
    linear axis renders the real distribution as one bar. The answer is
    a broken axis - the ordinary range on one panel, the off-scale
    readings on a narrow panel beside it, y shared so the rarity stays
    honest. The panel goes on whichever side the outliers actually are,
    which for vessel pressure is the left: its code is -1.0.
    """
    lo, hi = SPEC[sensor]
    clean = get_clean()[sensor].dropna()
    x0, x1 = float(clean.min()), float(clean.max())
    pad = (x1 - x0) * 0.06
    main = (min(x0, lo) - pad, max(x1, hi) + pad)

    if delivered:
        v = pd.to_numeric(get_raw()[RAW_COLUMN.get(sensor, sensor)],
                          errors="coerce").dropna()
        colour = RED
    else:
        v, colour = clean, BLUE

    inside = v[(v >= main[0]) & (v <= main[1])]
    off = v[(v < main[0]) | (v > main[1])]

    def main_trace():
        return go.Histogram(
            x=inside, nbinsx=70, marker_color=colour, opacity=0.9,
            hovertemplate="%{x}<br>%{y:,} batches<extra></extra>")

    if off.empty:
        fig = go.Figure(main_trace())
        fig.add_vrect(x0=lo, x1=hi, fillcolor=GRID, opacity=0.4,
                      line_width=0, layer="below")
        for x in (lo, hi):
            fig.add_vline(x=x, line=dict(color=INK2, width=1.2))
        fig.update_layout(xaxis_title=SENSOR_LABEL[sensor],
                          yaxis_title="batches")
        fig.update_xaxes(range=list(main))
        return style(fig, 330, legend=False)

    left = bool((off < main[0]).mean() > 0.5)
    widths = [0.24, 0.76] if left else [0.76, 0.24]
    main_col, off_col = (2, 1) if left else (1, 2)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=widths, horizontal_spacing=0.035)
    fig.add_trace(main_trace(), row=1, col=main_col)

    # Few distinct codes read better as labelled bars than as a
    # histogram whose bins would be narrower than the line width.
    counts = off.value_counts().sort_index()
    if len(counts) <= 5:
        fig.add_trace(go.Bar(
            x=[f"{x:g}" for x in counts.index], y=counts.to_numpy(),
            marker_color=RED, text=[f"{n:,}" for n in counts],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}: %{y:,} batches<extra></extra>"),
            row=1, col=off_col)
    else:
        fig.add_trace(go.Histogram(
            x=off, nbinsx=24, marker_color=RED, opacity=0.9,
            hovertemplate="%{x}<br>%{y:,} batches<extra></extra>"),
            row=1, col=off_col)

    fig.add_vrect(x0=lo, x1=hi, fillcolor=GRID, opacity=0.4,
                  line_width=0, layer="below", row=1, col=main_col)
    for x in (lo, hi):
        fig.add_vline(x=x, line=dict(color=INK2, width=1.2),
                      row=1, col=main_col)
    fig.update_xaxes(range=list(main), title_text=SENSOR_LABEL[sensor],
                     row=1, col=main_col)
    fig.update_xaxes(title_text=f"off scale &nbsp;({len(off):,})",
                     title_font=dict(color=RED), row=1, col=off_col)
    fig.update_yaxes(title_text="batches", row=1, col=1)
    return style(fig, 330, legend=False)


# ----------------------------------------------------- anomaly control

@st.cache_resource(show_spinner="Fitting the reference windows…")
def get_monitor():
    """References, per-batch scores and confirmed changes, once."""
    import simple_monitor as SM
    df = get_clean()
    refs = SM.fit_references(df)
    scores = SM.batch_scores(df, refs)
    changes = SM.detect_changes(df, refs)
    changes = changes[changes.confirmed
                      & (changes.peak_cusum > SM.SEVERITY_CUT)]
    return refs, scores, changes


@st.cache_data(show_spinner=False)
def monitor_summary() -> pd.DataFrame:
    """One row per reactor: how far it strays and whether it moved."""
    _, faulty, change = constants()
    df = get_clean()
    _, scores, changes = get_monitor()
    s = scores.assign(unit=df.unit.to_numpy(), ts=df.ts.to_numpy())
    g = s.groupby("unit")
    t = pd.DataFrame({
        "batches": g.size(),
        "flagged": g.is_anomaly.sum(),
        "flagged_pct": g.is_anomaly.mean() * 100,
        "worst_z": g.max_abs_z.max(),
    }).reset_index()
    moved = changes.groupby("unit").agg(
        sensors=("sensor", lambda x: ", ".join(
            sorted(SENSOR_LABEL.get(v, v).split(" (")[0] for v in x))),
        since=("alarm_date", "min"))
    t = t.merge(moved, on="unit", how="left")
    t["state"] = np.where(t.unit.isin(changes.unit.unique()),
                          "regime change", "steady")
    return t.sort_values("worst_z", ascending=False)


def fig_z_separation() -> go.Figure:
    """Why the cut needs no p-value: the two populations do not touch."""
    import simple_monitor as SM
    _, faulty, change = constants()
    df = get_clean()
    _, scores, _ = get_monitor()
    shifted = (df.unit.isin(faulty) & (df.ts >= change)).to_numpy()
    v = scores.max_abs_z.to_numpy()
    ok = ~np.isnan(v)
    fig = go.Figure()
    for mask, name, colour in ((ok & ~shifted, "steady production", BLUE),
                               (ok & shifted, "Plant-04 after 8 May", RED)):
        fig.add_trace(go.Histogram(
            x=v[mask], name=name, marker_color=colour, opacity=0.75,
            nbinsx=90, histnorm="probability"))
    fig.add_vline(x=SM.BATCH_CUT,
                  line=dict(color=INK, width=1.6, dash="dash"),
                  annotation_text=f"cut {SM.BATCH_CUT:g}",
                  annotation_position="top")
    fig.add_vline(x=float(np.sqrt(3)),
                  line=dict(color=MUTED, width=1.2, dash="dot"),
                  annotation_text="√3", annotation_position="top left")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="worst sensor, in that reactor's own σ")
    fig.update_yaxes(title="share of batches")
    return style(fig, 380)


def fig_reference_centres(sensor: str) -> go.Figure:
    """Reference centre of every unit during its first 21 days."""
    refs, _, _ = get_monitor()
    d = pd.DataFrame([
        {"unit": unit, "mean": float(r.mean[sensor]),
         "sd": float(r.sd[sensor]), "n_batches": r.n_batches}
        for unit, r in refs.items()
    ]).sort_values("mean")
    fleet_centre = float(d["mean"].mean())

    # Lollipops expose small, stable offsets without implying that the
    # fleet centre is the reference used by the detector.
    stem_x, stem_y = [], []
    for row in d.itertuples():
        stem_x.extend([fleet_centre, row.mean, None])
        stem_y.extend([row.unit, row.unit, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines", showlegend=False,
        line=dict(color=GRID, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d["mean"], y=d["unit"], mode="markers", showlegend=False,
        customdata=np.column_stack([d["sd"], d["n_batches"]]),
        marker=dict(color=BLUE, size=8),
        hovertemplate=("<b>%{y}</b><br>μ<sub>ref</sub> "
                       "%{x:.5g}<br>σ<sub>ref</sub> "
                       "%{customdata[0]:.5g}<br>"
                       "%{customdata[1]:,.0f} reference batches"
                       "<extra></extra>")))
    fig.add_vline(
        x=fleet_centre, line=dict(color=INK2, width=1.3, dash="dot"),
        annotation_text="average of reactor centres",
        annotation_position="top")
    fig.update_xaxes(title=f"21-day reference centre — {SENSOR_LABEL[sensor]}")
    fig.update_yaxes(
        title=None, categoryorder="array", categoryarray=d["unit"].tolist())
    return style(fig, max(560, 18 * len(d)), legend=False)


def fig_cusum_trace(unit: str, sensor: str) -> go.Figure:
    """Daily deviation on top, the accumulated tally underneath."""
    import simple_monitor as SM
    from plotly.subplots import make_subplots
    refs, _, changes = get_monitor()
    tr = SM.cusum_trace(get_clean(), refs, unit, sensor)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.42, 0.58], vertical_spacing=0.07)
    fig.add_trace(go.Scatter(
        x=tr.date, y=tr.z, mode="lines", name="daily z",
        line=dict(color=INK2, width=1.3),
        hovertemplate="%{x|%d %b}  z %{y:.2f}<extra></extra>"),
        row=1, col=1)
    fig.add_hrect(y0=-SM.CUSUM_K, y1=SM.CUSUM_K, fillcolor=BAND,
                  opacity=0.6, line_width=0, layer="below", row=1, col=1)
    for col, name, colour in (("cusum_down", "evidence it fell", RED),
                              ("cusum_up", "evidence it rose", BLUE)):
        fig.add_trace(go.Scatter(
            x=tr.date, y=tr[col], mode="lines", name=name,
            line=dict(color=colour, width=1.8),
            hovertemplate="%{x|%d %b}  %{y:,.0f}<extra></extra>"),
            row=2, col=1)
    fig.add_hline(y=SM.CUSUM_H, line=dict(color=INK, width=1.2,
                                          dash="dash"), row=2, col=1)
    hit = changes[(changes.unit == unit) & (changes.sensor == sensor)]
    for _, r in hit.iterrows():
        fig.add_vline(x=r.alarm_date, line=dict(color=RED, width=1.4,
                                                dash="dot"))
    fig.update_yaxes(title_text="daily z", row=1, col=1)
    fig.update_yaxes(title_text="accumulated evidence", type="log",
                     row=2, col=1)
    return style(fig, 440)


def fig_batch_z(batch_id: str) -> go.Figure:
    """The seven z of one batch against the cut."""
    import simple_monitor as SM
    sensors, _, _ = constants()
    df = get_clean()
    refs, _, _ = get_monitor()
    row = df[df.batch_id == batch_id]
    z = SM.z_scores(row, refs).iloc[0]
    lab = [SENSOR_LABEL[s] for s in sensors]
    vals = [z[s] for s in sensors]
    fig = go.Figure(go.Bar(
        x=vals, y=lab, orientation="h",
        marker_color=[RED if abs(v) >= SM.BATCH_CUT else BLUE
                      for v in vals],
        hovertemplate="%{y}: %{x:+.2f} σ<extra></extra>"))
    for x in (-SM.BATCH_CUT, SM.BATCH_CUT):
        fig.add_vline(x=x, line=dict(color=INK, width=1.2, dash="dash"))
    fig.add_vrect(x0=-np.sqrt(3), x1=np.sqrt(3), fillcolor=BAND,
                  opacity=0.45, line_width=0, layer="below")
    fig.update_xaxes(title="deviation from this reactor's reference (σ)")
    return style(fig, 330, legend=False)


def anomaly_for(row: pd.DataFrame) -> dict:
    """The frozen-z verdict for one assembled batch."""
    import simple_monitor as SM
    refs, _, _ = get_monitor()
    if row.unit.iloc[0] not in refs:
        return {"z": np.nan, "sensor": None, "signed": np.nan,
                "n_observed": 0, "is_anomaly": False,
                "cut": SM.BATCH_CUT}
    s = SM.batch_scores(row, refs).iloc[0]
    return {"z": float(s.max_abs_z) if pd.notna(s.max_abs_z) else np.nan,
            "sensor": s.worst_sensor,
            "signed": float(s.worst_z) if pd.notna(s.worst_z) else np.nan,
            "n_observed": int(s.n_observed),
            "is_anomaly": bool(s.is_anomaly),
            "cut": SM.BATCH_CUT}


def regime_for(unit: str, ts) -> pd.DataFrame:
    """Confirmed changes on a reactor that had already fired by `ts`."""
    _, _, changes = get_monitor()
    return changes[(changes.unit == unit)
                   & (changes.alarm_date <= ts)].sort_values("alarm_date")


# -------------------------------------------------- the ensemble itself

@st.cache_data(show_spinner=False)
def ensemble_example(n_missing: int = 0) -> dict:
    """One batch with its five members laid bare.

    Read from the saved test predictions rather than re-run, so the
    numbers on screen are the ones the evaluation reported.
    """
    t = quality_frame()
    pool = t[t.n_missing == n_missing]
    if pool.empty:
        pool = t
    # Use the batch closest to its group's median sigma so rare or
    # multimodal groups do not produce a misleading illustration.
    target = pool.total_std.median()
    row = pool.loc[(pool.total_std - target).abs().idxmin()]
    mus = np.array([row[f"mu_model_{i}"] for i in range(5)])
    sds = np.sqrt([row[f"var_model_{i}"] for i in range(5)])
    return {
        "batch_id": row.batch_id, "unit": row.unit,
        "n_missing": int(row.n_missing),
        "mus": mus, "sds": sds,
        "mu": float(row.predicted_months),
        "aleatoric": float(row.aleatoric_std),
        "epistemic": float(row.epistemic_std),
        "total": float(row.total_std),
        "actual": float(row.actual_months),
    }


def fig_ensemble(n_missing: int = 0) -> go.Figure:
    """Five Gaussians and the mixture they average into.

    The picture is the decomposition: how wide each curve is becomes the
    aleatoric term, how far apart their peaks sit becomes the epistemic
    one.
    """
    e = ensemble_example(n_missing)
    lo = e["mu"] - 4.2 * e["total"]
    hi = e["mu"] + 4.2 * e["total"]
    x = np.linspace(lo, hi, 400)

    def pdf(m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    fig = go.Figure()
    for i, (m, s) in enumerate(zip(e["mus"], e["sds"])):
        fig.add_trace(go.Scatter(
            x=x, y=pdf(m, s), mode="lines",
            name=f"member {i + 1}",
            line=dict(color=MUTED, width=1.1),
            hovertemplate=(f"member {i + 1}<br>μ {m:.3f}"
                           f"<br>σ {s:.3f}<extra></extra>")))
    mix = np.mean([pdf(m, s) for m, s in zip(e["mus"], e["sds"])], axis=0)
    fig.add_trace(go.Scatter(
        x=x, y=mix, mode="lines", name="ensemble",
        line=dict(color=BLUE, width=3),
        hovertemplate="ensemble<extra></extra>"))
    fig.add_vline(x=e["mu"], line=dict(color=INK, width=1.4, dash="dash"),
                  annotation_text=f"μ = {e['mu']:.2f}",
                  annotation_position="top right")
    fig.add_vline(x=e["actual"], line=dict(color=RED, width=1.4,
                                           dash="dot"),
                  annotation_text=f"lab {e['actual']:.2f}",
                  annotation_position="top left")
    fig.update_xaxes(title="shelf life (months)")
    fig.update_yaxes(title="density", showticklabels=False)
    return style(fig, 360)


def ensemble_table(n_missing: int = 0) -> pd.DataFrame:
    """The same five members as numbers."""
    e = ensemble_example(n_missing)
    rows = [{"model": f"member {i + 1}", "μ": m, "σ": s}
            for i, (m, s) in enumerate(zip(e["mus"], e["sds"]))]
    rows.append({"model": "ensemble", "μ": e["mu"], "σ": e["total"]})
    return pd.DataFrame(rows)


def fig_sigma_vs_missing() -> go.Figure:
    """Does the stated uncertainty track the error it has to cover?

    The test of an honest σ is not that it is small but that it grows
    when it should. Each bar pair is one missing-sensor group: what the
    model claimed beforehand against what the error turned out to be.
    """
    t = quality_frame()
    g = t.groupby("n_missing")
    d = pd.DataFrame({
        "n_batches": g.size(),
        "mean_aleatoric_std": g.aleatoric_std.mean(),
        "observed_rmse": g.apply(
            lambda x: float(np.sqrt(((x.actual_months
                                      - x.predicted_months) ** 2).mean()))),
    }).reset_index()
    d = d[d.n_batches >= 50]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d.n_missing, y=d.mean_aleatoric_std, name="σ the model stated",
        marker_color=BLUE,
        hovertemplate="%{x} blank: σ %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(
        x=d.n_missing, y=d.observed_rmse, name="error actually observed",
        marker_color=RED, opacity=0.85,
        hovertemplate="%{x} blank: RMSE %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="sensors with no reading", dtick=1)
    fig.update_yaxes(title="months")
    return style(fig, 380)


def fig_residual_calibration(n: int = 40000) -> go.Figure:
    """Standardized residuals against the normal they claim to be."""
    t = quality_frame().sample(n, random_state=0)
    z = ((t.actual_months - t.predicted_months) / t.total_std).to_numpy()
    z = z[np.isfinite(z)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=z, nbinsx=90, marker_color=BLUE, opacity=0.85,
        histnorm="probability density", name="standardized residuals",
        hovertemplate="%{x:.2f}<extra></extra>"))
    x = np.linspace(-4, 4, 300)
    fig.add_trace(go.Scatter(
        x=x, y=np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), mode="lines",
        name="standard normal", line=dict(color=INK, width=1.8,
                                          dash="dash")))
    fig.update_xaxes(title="(lab result − prediction) / stated σ",
                     range=[-4, 4])
    fig.update_yaxes(title="density", showticklabels=False)
    return style(fig, 380)


@st.cache_data(show_spinner=False)
def residual_stats() -> dict:
    """The three numbers that say whether σ means what it claims."""
    t = quality_frame()
    z = ((t.actual_months - t.predicted_months) / t.total_std).to_numpy()
    z = z[np.isfinite(z)]
    return {"mean": float(z.mean()), "sd": float(z.std(ddof=1)),
            "within_1": float((np.abs(z) <= 1).mean() * 100),
            "within_1_645": float((np.abs(z) <= 1.645).mean() * 100),
            "within_1_96": float((np.abs(z) <= 1.96).mean() * 100),
            "excess_kurtosis": float(pd.Series(z).kurtosis())}


def fig_ensemble_flow(n_missing: int = 0) -> go.Figure:
    """The batch fanning out to five models and back to one answer.

    The point of the picture is the shape, not the arithmetic: one
    input, five models running at the same time, two numbers out of
    each, one pair at the end.
    """
    e = ensemble_example(n_missing)
    fig = go.Figure()

    def box(x0, x1, y0, y1, fill, line, width=1.6):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color=line, width=width),
                      layer="below")

    def text(x, y, s, colour=INK, size=12, bold=False):
        fig.add_annotation(x=x, y=y, text=(f"<b>{s}</b>" if bold else s),
                           showarrow=False, font=dict(size=size,
                                                      color=colour),
                           align="center")

    def arrow(x0, y0, x1, y1):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.1,
                           arrowwidth=1.3, arrowcolor="#b9b8b1", text="")

    mid, ys = 2.0, [4.0, 3.0, 2.0, 1.0, 0.0]

    # ---- the batch
    box(0.1, 2.2, mid - 0.55, mid + 0.55, BAND, BLUE)
    text(1.15, mid + 0.22, "one batch", INK, 12.5, bold=True)
    text(1.15, mid - 0.16, f"{7 - n_missing} of 7 sensors", INK2, 11)

    # ---- the five members
    for y, m, s in zip(ys, e["mus"], e["sds"]):
        box(3.9, 6.5, y - 0.36, y + 0.36, "#ffffff", BLUE, 1.4)
        text(4.55, y, "CatBoost", INK2, 11)
        text(5.85, y + 0.14, f"μ = {m:.2f}", INK, 11.5, bold=True)
        text(5.85, y - 0.15, f"σ = {s:.2f}", RED, 11.5, bold=True)
        arrow(2.25, mid, 3.85, y)
        arrow(6.55, y, 7.95, mid)

    # ---- the answer
    box(8.0, 10.1, mid - 0.75, mid + 0.75, BLUE, BLUE)
    text(9.05, mid + 0.42, "ensemble", "#ffffff", 12, bold=True)
    text(9.05, mid + 0.06, f"μ = {e['mu']:.2f}", "#ffffff", 13, bold=True)
    text(9.05, mid - 0.3, f"σ = {e['total']:.2f}", "#ffffff", 13,
         bold=True)

    text(5.2, 4.85, "five models, run at the same time", MUTED, 11.5)
    text(9.05, 4.85, "one answer", MUTED, 11.5)

    fig.update_xaxes(visible=False, range=[-0.1, 10.4])
    fig.update_yaxes(visible=False, range=[-0.75, 5.2])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return style(fig, 400, legend=False)


# ------------------------------------------------------ accuracy, plain

@st.cache_data(show_spinner=False)
def accuracy_stats() -> dict:
    """The headline numbers, including the ones a business reader wants."""
    t = quality_frame()
    y, mu = t.actual_months.to_numpy(), t.predicted_months.to_numpy()
    e = np.abs(y - mu)
    return {
        "n": int(len(t)),
        "mae": float(e.mean()),
        "median": float(np.median(e)),
        "rmse": float(np.sqrt(((y - mu) ** 2).mean())),
        "mape": float((e / y).mean() * 100),
        "r2": float(1 - ((y - mu) ** 2).sum()
                    / ((y - y.mean()) ** 2).sum()),
        "within_1": float((e <= 1.0).mean() * 100),
        "within_1_5": float((e <= 1.5).mean() * 100),
    }


def fig_error_curve() -> go.Figure:
    """How many batches land within a given number of months."""
    t = quality_frame()
    e = np.abs(t.actual_months - t.predicted_months).to_numpy()
    x = np.linspace(0, 3.0, 121)
    y = [(e <= v).mean() * 100 for v in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="share within",
        line=dict(color=BLUE, width=3),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="within %{x:.2f} months: %{y:.1f} %<extra></extra>"))
    for v in (0.5, 1.0, 1.5):
        share = (e <= v).mean() * 100
        fig.add_trace(go.Scatter(
            x=[v], y=[share], mode="markers+text",
            marker=dict(color=INK, size=9),
            text=[f" {share:.0f} % within {v:g}"], textposition="top left",
            textfont=dict(size=11.5, color=INK), showlegend=False,
            hoverinfo="skip"))
    fig.update_xaxes(title="how close the prediction was (months)",
                     range=[0, 3])
    fig.update_yaxes(title="share of batches (%)", range=[0, 100])
    return style(fig, 380, legend=False)


def fig_error_by_grade() -> go.Figure:
    """Where the error sits, in the language the business uses."""
    from model import to_grade
    t = quality_frame()
    y = t.actual_months.to_numpy()
    e = np.abs(y - t.predicted_months.to_numpy())
    g = pd.Series(y).map(to_grade)
    d = pd.DataFrame({"grade": g, "err": e, "pct": e / y * 100})
    order = ["Technical Grade", "Industrial Grade", "Perfumery Grade"]
    m = d.groupby("grade").agg(mae=("err", "mean"),
                               mape=("pct", "mean"),
                               n=("err", "size")).reindex(order)
    fig = go.Figure(go.Bar(
        x=[o.split()[0] for o in order], y=m.mae,
        marker_color=[GRADE_COLOURS[o] for o in order],
        text=[f"{v:.2f} months<br>{p:.1f} %" for v, p in zip(m.mae,
                                                             m.mape)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>MAE %{y:.3f} months<extra></extra>"))
    fig.update_yaxes(title="mean error (months)",
                     range=[0, m.mae.max() * 1.45])
    return style(fig, 380, legend=False)


@st.cache_data(show_spinner=False)
def heldout_regime_results() -> dict:
    """Metrics from the deliberately hidden-regime stress test."""
    from paths import TABLES
    path = TABLES / "heldout_regime_evaluation.json"
    return json.loads(path.read_text("utf-8"))


def fig_heldout_regime() -> go.Figure:
    """Actual and predicted shelf life after hiding the faulty regime."""
    from paths import TABLES
    t = pd.read_parquet(TABLES / "heldout_regime_predictions.parquet")
    fig = go.Figure()
    for col, name, colour in (
            ("actual_months", "what the lab measured", RED),
            ("presented_months", "what the model predicted", BLUE)):
        fig.add_trace(go.Histogram(
            x=t[col], name=name, marker_color=colour, opacity=0.68,
            nbinsx=70, histnorm="probability density",
            hovertemplate=f"{name}<br>%{{x:.2f}} months<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="shelf life (months)", range=[17.8, 23.5])
    fig.update_yaxes(title="share of batches", showticklabels=False)
    return style(fig, 390)


# ---------------------------------------------- one batch in, six out

def fig_output_flow(batch_id: str) -> go.Figure:
    """A batch on the left, everything it produces on the right.

    Each output box names the component that made it, so the diagram
    doubles as a legend for the architecture on the same page.
    """
    sensors, _, _ = constants()
    df = get_clean()
    row = df[df.batch_id == batch_id]
    out = predict_row(row)
    an = anomaly_for(row)
    ch = regime_for(row.unit.iloc[0], row.ts.iloc[0])
    top = out["contrib"].reindex(
        out["contrib"].abs().sort_values(ascending=False).index)

    rows = [
        ("Shelf life", f"{out['mu']:.2f} months", "CatBoost × 5", BLUE),
        ("Range, 90 %", f"{out['lo']:.2f} – {out['hi']:.2f}",
         "Conformal", BLUE),
        ("Grade", out["grade"].split()[0], "from the range", BLUE),
        ("Main driver",
         f"{label_of(top.index[0])} "
         f"{top.iloc[0]:+.2f} months", "SHAP", AQUA),
        ("Anomaly",
         (f"YES · {SENSOR_LABEL[an['sensor']].split(' (')[0]} "
          f"{an['signed']:+.2f} σ" if an["is_anomaly"] else "no"),
         "frozen z", RED if an["is_anomaly"] else AMBER),
        ("Regime change",
         (f"YES · since {ch.iloc[0].alarm_date:%d %b}"
          if len(ch) else "no"),
         "CUSUM", RED if len(ch) else AMBER),
    ]

    fig = go.Figure()

    def box(x0, x1, y0, y1, fill, line, width=1.5):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color=line, width=width),
                      layer="below")

    def text(x, y, s, colour=INK, size=12, bold=False, anchor="center"):
        fig.add_annotation(x=x, y=y, xanchor=anchor, showarrow=False,
                           text=(f"<b>{s}</b>" if bold else s),
                           font=dict(size=size, color=colour))

    n = len(rows)
    ys = [n - 1 - i for i in range(n)]
    mid = (n - 1) / 2

    # ---- the batch that goes in
    box(0.0, 2.5, mid - 1.05, mid + 1.05, BAND, BLUE, 1.8)
    text(1.25, mid + 0.62, "ONE BATCH", INK, 12.5, bold=True)
    text(1.25, mid + 0.18, batch_id, INK2, 10.5)
    text(1.25, mid - 0.2, row.unit.iloc[0], INK2, 10.5)
    text(1.25, mid - 0.6,
         f"{int(row.n_missing.iloc[0])} of 7 sensors blank", MUTED, 10)

    # ---- what comes out
    for y, (label, value, source, colour) in zip(ys, rows):
        box(4.3, 9.3, y - 0.38, y + 0.38, "#ffffff", colour, 1.5)
        text(4.55, y, label, INK2, 11, anchor="left")
        text(6.55, y, value, colour, 12, bold=True, anchor="left")
        text(9.55, y, source, MUTED, 10.5, anchor="left")
        fig.add_annotation(x=4.25, y=y, ax=2.55, ay=mid,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.1,
                           arrowwidth=1.2, arrowcolor="#c9c8c1", text="")

    text(6.8, n - 0.35, "what the engineer receives", MUTED, 11)
    text(9.55, n - 0.35, "made by", MUTED, 10.5, anchor="left")

    fig.update_xaxes(visible=False, range=[-0.1, 11.6])
    fig.update_yaxes(visible=False, range=[-0.75, n + 0.1])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return style(fig, 400, legend=False)


def fig_boosting() -> go.Figure:
    """Why it is called boosting: each tree fixes what is left over."""
    fig = go.Figure()

    def box(x0, x1, y0, y1, fill, line):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line=dict(color=line, width=1.5),
                      layer="below")

    def text(x, y, s, colour=INK, size=11, bold=False):
        fig.add_annotation(x=x, y=y, showarrow=False,
                           text=(f"<b>{s}</b>" if bold else s),
                           font=dict(size=size, color=colour))

    labels = ["tree 1", "tree 2", "tree 3", "…", "tree 3000"]
    for i, lab in enumerate(labels):
        x = i * 2.05
        box(x, x + 1.6, 0.6, 1.5, "#ffffff", BLUE)
        text(x + 0.8, 1.05, lab, INK2, 11)
        if i < len(labels) - 1:
            fig.add_annotation(
                x=x + 2.0, y=1.05, ax=x + 1.65, ay=1.05,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.1,
                arrowwidth=1.2, arrowcolor="#c9c8c1", text="")
            text(x + 1.83, 0.28, "what is", MUTED, 9)
            text(x + 1.83, 0.02, "still wrong", MUTED, 9)

    box(0.0, 9.85, -1.15, -0.35, BAND, BLUE)
    text(4.9, -0.75, "the sum of all the trees  =  one prediction",
         INK, 12, bold=True)
    fig.update_xaxes(visible=False, range=[-0.3, 10.1])
    fig.update_yaxes(visible=False, range=[-1.5, 1.8])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return style(fig, 230, legend=False)


@st.cache_data(show_spinner=False)
def gaps_vs_shelf() -> pd.DataFrame:
    """Shelf life by how many sensors went blank."""
    df = get_clean()
    g = df.groupby("n_missing").stability_months
    t = pd.DataFrame({"batches": g.size(), "mean": g.mean(),
                      "sd": g.std()})
    t["sem"] = t["sd"] / np.sqrt(t["batches"])
    return t.reset_index()


def fig_gaps_vs_shelf() -> go.Figure:
    """The check that says filling the gaps would not bias the mean."""
    t = gaps_vs_shelf()
    t = t[t.batches >= 100]
    base = float(t.loc[t.n_missing == 0, "mean"].iloc[0])
    fig = go.Figure()
    fig.add_hline(y=base, line=dict(color=INK2, width=1.2, dash="dash"))
    fig.add_trace(go.Scatter(
        x=t.n_missing, y=t["mean"], mode="markers",
        marker=dict(color=BLUE, size=9),
        error_y=dict(type="data", array=1.96 * t["sem"], color=BLUE,
                     thickness=1.4, width=6),
        name="mean shelf life",
        hovertemplate="%{x} blank: %{y:.3f} months<extra></extra>"))
    fig.update_xaxes(title="sensors with no reading, out of 7", dtick=1)
    fig.update_yaxes(title="mean shelf life (months)",
                     range=[base - 0.55, base + 0.35])
    return style(fig, 360, legend=False)

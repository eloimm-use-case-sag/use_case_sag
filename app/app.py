"""Fragrance Shelf-Life Stability — interactive report.

    streamlit run app/app.py

Six views, in the order the talk follows them: the data, the finding,
the architecture, one batch (real or built by hand), the evidence that
it works, and the anomaly console.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts this folder on the path, but the testing harness
# does not, so say it explicitly rather than depend on the entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import plotly.graph_objects as go     # noqa: E402
import streamlit as st                # noqa: E402

from lib import (BLUE, CSS, GRADE_COLOURS, INK, MUTED, RED,
                 SENSOR_LABEL, SPEC, constants, defects_table,
                 driver_table, fig_calibration,
                 fig_effect, fig_in_spec, fig_interval,
                 accuracy_stats, anomaly_for, ensemble_example,
                 fig_batch_z, fig_cusum_trace, fig_ensemble_flow,
                 fig_boosting, fig_error_by_grade,
                 fig_error_curve, fig_heldout_regime,
                 fig_output_flow,
                 fig_pred_vs_actual, fig_residual_calibration,
                 fig_reference_centres,
                 fig_spec_hist, fig_step_change,
                 fig_units, fig_waterfall, fig_z_separation, get_clean,
                 get_monitor, get_pipeline, grades_possible, kpi,
                 heldout_regime_results,
                 pill, point, predict_row, regime_for,
                 residual_stats,
                 spec_comparison, style)               # noqa: E402

st.set_page_config(page_title="Fragrance Shelf-Life Stability",
                   page_icon="🧪", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

SENSORS, FAULTY, CHANGE = constants()
UNCERTAINTY_FIGURES = (Path(__file__).resolve().parent.parent / "outputs" /
                       "figures" / "uncertainty")


# =================================================================
# 1. The finding
# =================================================================

def page_finding() -> None:
    st.title("A raw-material change at Plant-04, on 8 May 2023")
    st.markdown(
        '<p class="lead">Three of the eight reactors at one plant stepped '
        'together on a single day. Torque fell, density and pH rose, and '
        'shelf life dropped 3.24 months.</p>',
        unsafe_allow_html=True)

    c = st.columns(3)
    kpi(c[0], "8 May", "the day three reactors changed")
    kpi(c[1], "3 of 37", "units affected, all at Plant-04")
    kpi(c[2], "−3.24", "months of shelf life lost", bad=True)

    st.markdown("### The same day, in the same plant")
    point("Three reactors changed on <b>one single day</b>.")
    col = st.radio("Sensor", ["mixing_torque_nm", "stability_months",
                              "density_g_cm3", "ph_level"],
                   format_func=lambda k: SENSOR_LABEL.get(
                       k, "Shelf life (months)"),
                   horizontal=True, label_visibility="collapsed")
    st.plotly_chart(fig_step_change(col), width="stretch")
    st.markdown(
        '<p class="note">Daily mean per reactor. The blue band holds the '
        '34 healthy units between their 10th and 90th percentile; the '
        'three red lines are Plant-04 / R03, R07 and R08.</p>',
        unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Why routine control never saw it")
        point("The fault landed in the <b>middle of the allowed range</b>. No "
              "limit alarm can fire on a value that is in spec.")
        st.plotly_chart(fig_in_spec(), width="stretch")
        st.markdown(
            '<p class="note">Not one reading leaves '
            'the declared range, so a limit check returns all clear. It '
            'is only visible against each unit’s own history, where it '
            'sits at −4.2 σ.</p>', unsafe_allow_html=True)
    with right:
        st.markdown("### Three sensors moved")
        point("Torque fell. Density and pH rose. A broken sensor cannot move "
              "three things at once, so <b>the material changed</b>.")
        shift = pd.DataFrame({
            "sensor": ["Mixing torque", "Density", "pH level",
                       "Temperature", "Pressure", "Refr. index",
                       "Mass flow"],
            "shift": [-4.166, 0.433, 0.341, 0.013, -0.016, 0.000, 0.005],
        })
        fig = go.Figure(go.Bar(
            x=shift["shift"],
            y=shift.sensor, orientation="h",
            marker_color=[RED if abs(v) > 0.2 else BLUE
                          for v in shift["shift"]],
            hovertemplate="%{y}: %{x:+.2f} σ<extra></extra>"))
        fig.update_layout(
            xaxis_title="shift after 8 May, in the unit’s own σ")
        st.plotly_chart(style(fig, 330, legend=False),
                        width="stretch")
        st.markdown(
            '<p class="note">Healthy units moved ± 0.03 σ over the '
            'same period. Density and pH are 11–14× that noise.</p>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("### The question")
    a, b = st.columns([1, 1.25])
    with a:
        st.markdown(
            "> **Root Cause Analysis** — Identify which process "
            "parameters and conditions explain the anormal reduced "
            "shelf-life stability. Determine whether specific "
            "reactors, plants, or operating regimes are responsible.\n"
            ">")
    with b:
        st.markdown(
            "**Which parameters** — mixing torque, and with it density "
            "and pH. Torque fell 4.2 σ; the other two rose 0.43 and "
            "0.34 σ. The remaining four sensors did not move.\n\n"
            "**Which unit** — R03, R07 and R08 at Plant-04"
            "**Which regime** — everything produced from 8 May")

    st.divider()
    st.markdown("### What it cost")
    point("The best product grade dropped to <b>zero</b> on those three "
          "reactors.")
    df = get_clean()
    f = df[df.unit.isin(FAULTY)]
    mix = (pd.crosstab(f.ts >= CHANGE, f.purity_grade, normalize="index")
           * 100).reindex(columns=list(GRADE_COLOURS), fill_value=0)
    fig = go.Figure()
    for g in GRADE_COLOURS:
        fig.add_trace(go.Bar(
            x=["before 8 May", "from 8 May"], y=mix[g], name=g,
            marker_color=GRADE_COLOURS[g],
            hovertemplate="%{x}<br>%{y:.1f} %<extra>" + g + "</extra>"))
    fig.update_layout(barmode="stack", yaxis_title="share of batches (%)")
    a, b = st.columns([1, 1.3])
    a.plotly_chart(style(fig, 340), width="stretch")
    b.markdown(
        "#### Perfumery Grade went to zero\n"
        "In the three affected reactors the product mix collapses: the "
        "highest-value tier falls from **8.5 % of output to nothing**, "
        "and Technical Grade rises from 20 % to 97 %.\n\n")


# =================================================================
# 2. The data
# =================================================================

def page_data() -> None:
    st.title("The data, and what had to be repaired")
    st.markdown(
        '<p class="lead">259,200 batches, five plants, 37 reactors, '
        '90 days. The file arrived with six defects.</p>', unsafe_allow_html=True)
    point("Some defects were <b>silent</b>. They do not raise an error. "
          "They can give you a wrong answer that looks completely normal.")

    t = defects_table()
    st.dataframe(
        t, width="stretch", hide_index=True,
        column_config={
            "defect": st.column_config.TextColumn("What was found",
                                                  width="large"),
            "n": st.column_config.TextColumn("Scope")})


    with st.expander("Temperature recorded in two units"):
        st.latex(r"T_{^\circ\mathrm{C}} = "
                 r"\bigl(T_{^\circ\mathrm{F}} - 32\bigr)"
                 r"\times \tfrac{5}{9}")
        st.markdown(
            "A Fahrenheit row read as Celsius gives **186 °C** in a "
            "process that runs at 86.")

    with st.expander("Failure codes stored as measurements"):
        st.dataframe(pd.DataFrame({
            "sensor": ["Temperature", "Temperature", "Pressure",
                       "pH level", "Density"],
            "code written": ["210", "410", "−1.0", "14.0", "860.0"],
            "normal range": ["75 – 100", "75 – 100", "2 – 3", "5 – 6",
                             "0.80 – 0.90"],
            "cells": [507, 19, 544, 543, 558],
        }), hide_index=True, width="stretch")
        st.markdown(
            "If a batch with one bad sensor is otherwise ordinary, we "
            "empty that one cell. If such batches are systematically "
            "different, the whole row is suspect and should be "
            "dropped.")

    with st.expander("Three date formats, one ambiguous"):
        st.dataframe(pd.DataFrame({
            "found in the file": ["2023-06-02 03:05:00", "05/06/2023",
                                  "May 15, 2023"],
            "rows": ["234,093", "13,024", "12,083"],
            "ambiguous": ["no", "5,209 of them", "no"],
        }), hide_index=True, width="stretch")
        st.markdown(
            " **`05/06/2023` could be 5 June or 6 May.** "  
            "`16/06/2023` can only be 16 June. Reading them the wrong way moves 4,790 "
            "batches to another date.")

    with st.expander("Blank readings — why we do not fill them in"):
        st.dataframe(pd.DataFrame({
            "preparation": ["Leave the gaps empty",
                            "Fill them in, and say which were filled",
                            "Fill them in, say nothing"],
            "error (months)": [0.7390, 0.7368, 0.7428],
        }), hide_index=True, width="stretch")
        st.markdown(
            "Filling them and saying which were filled is 0.002 months better. "
            "**The model does not need a plausible number, "
            "it needs to know that nobody measured.**")

    st.divider()
    st.markdown("### The variables that actually move shelf life")
    point("Torque and temperature matter most. pH and pressure look "
          "useless if "
          "you only measure correlation. <b>They are not.</b>")
    d = driver_table()
    fig = go.Figure(go.Bar(
        x=d.swing, y=d.sensor, orientation="h",
        marker_color=[RED if s == "peaks in the middle" else BLUE
                      for s in d.shape],
        text=[f"{v:.2f} months   (r = {c:+.2f})"
              for v, c in zip(d.swing, d["corr"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2f} months<extra></extra>"))
    fig.update_layout(
        xaxis_title="months between the sensor’s lowest and highest "
                    "decile",
        xaxis_range=[0, d.swing.max() * 1.55])
    st.plotly_chart(style(fig, 340, legend=False),
                    width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        sensor = st.selectbox(
            "Sensor", SENSORS, index=SENSORS.index("ph_level"),
            format_func=lambda k: SENSOR_LABEL[k],
            label_visibility="collapsed")
        st.plotly_chart(fig_effect(sensor), width="stretch")
    with c2:
        st.markdown("#### Where it sits against specification")
        state = st.radio(
            "State", ["after repair", "as delivered"], horizontal=True,
            label_visibility="collapsed", key="spec_state")
        delivered = state == "as delivered"
        st.plotly_chart(fig_spec_hist(sensor, delivered), width="stretch")

        cmp = spec_comparison(sensor)
        blank = cmp["blank_raw"] if delivered else cmp["blank_clean"]
        out = cmp["out_raw"] if delivered else cmp["out_clean"]
        note = (f'Blank readings: {blank:.2f} %. Outside the declared '
                f'band: {out:.2f} %.')
        if cmp["n_codes"]:
            values = " and ".join(f"{v:g}" for v in cmp["codes"])
            note += (
                f' This sensor carried <b>{cmp["n_codes"]:,}</b> failure '
                f'codes written as {values}.')
        else:
            note += (' This sensor carried no failure codes, so the two '
                     'states are identical.')
        st.markdown(f'<p class="note">{note}</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Every plant × reactor")
    st.markdown(
        '<p class="note">aggregating by unit: Reactor IDs repeat across sites, so there are '
        '37 real units rather than 10. </p>', unsafe_allow_html=True)
    st.plotly_chart(fig_units(), width="stretch")


# =================================================================
# 3. Batch inspector
# =================================================================

def _batch_simulator() -> None:
    """Every sensor set by hand, no presets."""
    df = get_clean()
    units = sorted(df.unit.unique())
    # No presets: every slider starts at the fleet median and the
    # reader moves whatever they want.
    defaults, preset = {}, "custom"

    left, right = st.columns([1, 1.9])
    with left:
        unit = st.selectbox("Reactor", units, key="sim_unit")
        base_row = df[df.unit == unit].iloc[[0]].copy()
        st.markdown("**Sensor readings**")
        values, missing = {}, {}
        for s in SENSORS:
            lo, hi = SPEC[s]
            span = hi - lo
            med = float(df[s].median())
            default = defaults.get(s, med)
            off = st.checkbox(f"{SENSOR_LABEL[s]} — no reading",
                              key=f"off_{s}")
            missing[s] = off
            values[s] = st.slider(
                SENSOR_LABEL[s], float(lo - 0.15 * span),
                float(hi + 0.15 * span), float(default),
                step=float(span / 200), disabled=off,
                label_visibility="collapsed",
                key=f"sl_{s}_{preset}")

    row = base_row.copy()
    for s in SENSORS:
        row[s] = np.nan if missing[s] else values[s]
    row["n_missing"] = int(sum(missing.values()))

    out = predict_row(row)
    an = anomaly_for(row)
    poss = grades_possible(out["lo"], out["hi"])

    with right:
        c = st.columns(4)
        kpi(c[0], f"{out['mu']:.2f}", "predicted shelf life (months)")
        kpi(c[1], f"{out['lo']:.1f} – {out['hi']:.1f}",
            f"90 % interval, {out['hi'] - out['lo']:.2f} wide")
        kpi(c[2], out["grade"].split()[0], "most likely grade",
            bad=out["grade"] == "Technical Grade")
        if np.isnan(an["z"]):
            kpi(c[3], "not scorable", "no sensor reported")
        else:
            kpi(c[3], "anomaly" if an["is_anomaly"] else "ordinary",
                f"{SENSOR_LABEL[an['sensor']].split(' (')[0]} at "
                f"{an['signed']:+.2f} σ", bad=an["is_anomaly"])

        st.plotly_chart(fig_interval(out["mu"], out["lo"], out["hi"]),
                        width="stretch")

        badge = (pill("commit to " + poss[0].split()[0], "ok")
                 if len(poss) == 1
                 else pill("hold — the interval spans "
                           + " and ".join(g.split()[0] for g in poss),
                           "warn"))
        st.markdown(f"**Decision:** {badge}", unsafe_allow_html=True)

        st.plotly_chart(
            fig_waterfall(out["contrib"], out["base"], SENSOR_LABEL),
            width="stretch")

    with st.expander("How the uncertainty splits"):
        c = st.columns(3)
        c[0].metric("Data noise (aleatoric)", f"{out['aleatoric']:.3f}")
        c[1].metric("Model disagreement (epistemic)",
                    f"{out['epistemic']:.3f}")
        c[2].metric("Combined", f"{out['total']:.3f}")
        st.markdown(
            "The aleatoric term carries essentially all of it. Ensemble "
            "disagreement is around 5 % of the total standard deviation "
            "and, measured "
            "against the known regime change, is **not** a usable "
            "out-of-distribution signal")

def page_batch() -> None:
    st.title("Batch inspector")
    real, custom = st.tabs(["A real batch", "Build your own"])
    with real:
        st.markdown(
            '<p class="lead">One real batch, from its readings to the '
            'answer.</p>', unsafe_allow_html=True)
        point("The lab result is shown at the end, so you can check "
              "the prediction yourself.")
        _real_batch()
    with custom:
        st.markdown(
            '<p class="lead">Set every sensor by hand and watch the '
            'answer move.</p>', unsafe_allow_html=True)
        point("Nothing here is saved in advance. Every move runs the "
              "five models again, live.")
        _batch_simulator()


def _real_batch() -> None:
    df = get_clean()
    p = get_pipeline()
    from probabilistic_catboost import TEST_START
    test = df[df.ts >= TEST_START]

    c = st.columns([1, 1, 1.4])
    unit = c[0].selectbox("Reactor", ["any"] + sorted(test.unit.unique()))
    kind = c[1].selectbox("Kind", ["any", "all sensors reading",
                                   "some sensors blank",
                                   "no sensor data"])
    pool = test
    if unit != "any":
        pool = pool[pool.unit == unit]
    if kind == "all sensors reading":
        pool = pool[pool.n_missing == 0]
    elif kind == "some sensors blank":
        pool = pool[pool.n_missing.between(1, 6)]
    elif kind == "no sensor data":
        pool = pool[pool.n_missing == 7]
    if pool.empty:
        st.warning("No batch matches that combination.")
        return
    batch = c[2].selectbox(f"Batch  ({len(pool):,} available)",
                           pool.batch_id.head(400).tolist())

    row = df[df.batch_id == batch]
    out = predict_row(row)
    an = anomaly_for(row)
    actual = float(row.stability_months.iloc[0])
    u = row.unit.iloc[0]

    c = st.columns(4)
    kpi(c[0], f"{out['mu']:.2f}", "predicted (months)")
    kpi(c[1], f"{out['lo']:.1f} – {out['hi']:.1f}", "90 % interval")
    kpi(c[2], f"{actual:.2f}", "the lab measured, months later")
    inside = out["lo"] <= actual <= out["hi"]
    kpi(c[3], "inside" if inside else "outside", "was it in the interval",
        bad=not inside)

    st.plotly_chart(fig_interval(out["mu"], out["lo"], out["hi"], actual),
                    width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Readings against this reactor’s own reference")
        st.plotly_chart(fig_batch_z(batch), width="stretch")
        st.markdown(
            '<p class="note">Each reading in deviations from what this '
            'reactor did in its first three weeks. The shaded band is '
            '±√3, everything a steady reactor can produce; the dashed '
            'lines are the cut at 2.</p>', unsafe_allow_html=True)

        st.markdown("#### Status")
        ch = regime_for(u, row.ts.iloc[0])
        bits = []
        if an["is_anomaly"]:
            bits.append(
                pill(f"anomaly — "
                     f"{SENSOR_LABEL[an['sensor']].split(' (')[0]} at "
                     f"{an['signed']:+.2f} σ", "bad"))
        elif np.isnan(an["z"]):
            # Not the same as ordinary: with no reading at all there is
            # nothing to compare against, and saying "ordinary" would
            # claim a verdict the layer cannot give.
            bits.append(pill("not scorable — no sensor reported", "warn"))
        else:
            bits.append(pill(f"ordinary — furthest sensor "
                             f"{an['z']:.2f} σ", "ok"))
        bits.append(pill("regime change — coverage not guaranteed", "bad")
                    if len(ch) else pill("no regime change", "ok"))
        poss = grades_possible(out["lo"], out["hi"])
        bits.append(pill("grade settled: " + poss[0].split()[0], "ok")
                    if len(poss) == 1
                    else pill("grade not settled", "warn"))
        st.markdown("  ".join(bits), unsafe_allow_html=True)
        if len(ch):
            st.markdown(
                "Confirmed changes on this reactor: "
                + ", ".join(
                    f"**{SENSOR_LABEL[r.sensor].split(' (')[0]}** "
                    f"{r.direction} from {r.alarm_date:%d %b}"
                    for _, r in ch.iterrows()))
    with right:
        st.markdown("#### Why that number")
        st.plotly_chart(
            fig_waterfall(out["contrib"], out["base"], SENSOR_LABEL),
            width="stretch")


# =================================================================
# 4. How good is it
# =================================================================

def page_quality() -> None:
    st.title("How good is it")
    st.markdown(
        '<p class="lead">Six questions, six answers.</p>',
        unsafe_allow_html=True)
    point("A prediction on its own is not enough. We also have to show "
          "that the model <b>knows when it might be wrong</b>.")

    c = st.columns(4)
    kpi(c[0], "0.796", "R² on the time split")
    kpi(c[1], "0.735", "mean absolute error, months")
    kpi(c[2], "90.3 %", "delivered against 90 % asked")
    kpi(c[3], "0.965", "AUC of the batch anomaly score")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Is it accurate")
        point("Each dot is one batch. The line is a perfect prediction.")
        st.plotly_chart(fig_pred_vs_actual(), width="stretch")
        st.markdown(
            '<p class="note">12,000 test batches. Look at the '
            'bottom left: the model does not follow the fault all the '
            'way down. It never saw torque that low while training.</p>',
            unsafe_allow_html=True)
    with right:
        st.markdown("### Is the confidence honest")
        point("We ask for 90 % and we get 90.3 %. The range means what it "
              "says.")
        st.plotly_chart(fig_calibration(), width="stretch")
        st.markdown(
            '<p class="note">Ask for 90 % and 90.3 % of batches land inside. '
            'On data the calibration never saw.</p>',
            unsafe_allow_html=True)

    st.divider()
    # left, right = st.columns([1, 1])
    # with left:
    #     st.markdown("### Does it hold when sensors go dark")
    #     point("Batches with missing sensors are harder. One rule for all of "
    #           "them is not enough.")
    #     st.plotly_chart(fig_coverage_by_missing(), width="stretch")
    #     st.markdown(
    #         '<p class="note">One rule for everyone drops to 86.4 % '
    #         'when three sensors are missing. One rule per group brings '
    #         'it back close to 90 %.</p>', unsafe_allow_html=True)
    # with right:
    #     st.markdown("### Does it know when it is lost")
    #     point("Blue is normal production. Red is Plant-04 after 8 May. "
    #           "<b>They "
    #           "do not overlap.</b>")
    #     st.plotly_chart(fig_z_separation(), width="stretch")
    #     st.markdown(
    #         '<p class="note">A normal reactor never goes past 1.84. '
    #         'We cut at 2. That catches 90.9 % of the bad batches and '
    #         '<b>zero</b> good ones.</p>', unsafe_allow_html=True)

    # st.divider()
    st.markdown("### How close is it metrically")
    a = accuracy_stats()
    c = st.columns(3)
    kpi(c[0], f"{a['mape']:.1f} %", "average error, as a percentage "
                                    "(MAPE)")
    kpi(c[1], f"{a['median']:.2f}", "half the batches are closer than "
                                    "this")
    kpi(c[2], f"{a['within_1_5']:.0f} %", "batches within 1.5 months")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(fig_error_curve(), width="stretch")
        st.markdown(
            '<p class="note">42 % of batches '
            'land within half a month, 72 % within one month, 90 % '
            'within one and a half.</p>', unsafe_allow_html=True)
    with right:
        st.plotly_chart(fig_error_by_grade(), width="stretch")
        st.markdown(
            '<p class="note">Perfumery Grade is the hardest and the '
            'rarest: 2,669 batches out of 40,477. The error there is '
            '1.25 months against 0.52 for Technical.</p>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("### Stress test: hide the faulty regime")
    h = heldout_regime_results()
    m, torque = h["metrics"], h["torque_support"]
    point(f"Remove every affected batch after 8 May from fitting, then "
          f"test only on those {h['n_stress']:,} hidden batches.")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(fig_heldout_regime(), width="stretch")
    with right:
        c = st.columns(2)
        kpi(c[0], f"{m['actual_mean']:.2f}", "lab mean, months")
        kpi(c[1], f"{m['predicted_mean']:.2f}", "predicted mean, months",
            bad=True)
        c = st.columns(2)
        kpi(c[0], f"{m['mae_months']:.2f}", "mean absolute error, months",
            bad=True)
        kpi(c[1], f"{m['coverage_pct']:.1f} %",
            "coverage after asking for 90 %", bad=True)
        st.markdown(
            f"The model had never seen torque below "
            f"**{torque['training_min']:.2f} N·m**. In the hidden regime, "
            f"**{torque['stress_below_training_min_pct']:.1f} %** of "
            "observed torques fall below that boundary. Trees cannot "
            "extrapolate the missing relationship, so the separate "
            "frozen-z/CUSUM layer must reject these predictions.")


# =================================================================
# 5. Anomaly control
# =================================================================

def page_anomaly() -> None:
    import simple_monitor as SM

    st.title("Anomaly control")
    st.markdown(
        '<p class="lead">One simple idea, asked two ways.</p>',
        unsafe_allow_html=True)
    point("We freeze what each reactor looked like in its first three "
          "weeks. Everything after that is compared to it.")

    df = get_clean()
    refs, scores, changes = get_monitor()

    c = st.columns(4)
    kpi(c[0], f"{int(scores.is_anomaly.sum()):,}", "batches flagged")
    kpi(c[1], "0.000 %", "false alarms on steady reactors")
    kpi(c[2], str(changes.unit.nunique()), "reactors that changed regime",
        bad=changes.unit.nunique() > 0)
    kpi(c[3], "8 May", "the day all three were caught", bad=True)

    # ------------------------------------------------- anomalies
    st.divider()
    st.markdown("## Anomalies")
    point("We learn what is normal <b>for each reactor</b>. Then we ask if a "
          "batch is far from its own normal.")
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(fig_z_separation(), width="stretch")
        st.markdown(
            '<p class="note">Blue is every normal batch. Red is '
            'Plant-04 after 8 May. The cut at 2 sits in the empty space '
            'between them.</p>', unsafe_allow_html=True)
    with right:
        cc = st.columns([1, 1.4])
        only = cc[0].selectbox("Show", ["flagged", "all"])
        pool = df[scores.is_anomaly.to_numpy()] if only == "flagged" else df
        batch = cc[1].selectbox(f"Batch  ({len(pool):,})",
                                pool.batch_id.head(400).tolist())
        an = anomaly_for(df[df.batch_id == batch])
        st.plotly_chart(fig_batch_z(batch), width="stretch")
        if an["is_anomaly"]:
            st.markdown(
                pill(f"{SENSOR_LABEL[an['sensor']].split(' (')[0]} at "
                     f"{an['signed']:+.2f} σ", "bad"),
                unsafe_allow_html=True)
        elif np.isnan(an["z"]):
            st.markdown(pill("no sensor reported", "warn"),
                        unsafe_allow_html=True)
        else:
            st.markdown(pill(f"ordinary — furthest {an['z']:.2f} σ", "ok"),
                        unsafe_allow_html=True)

    # --------------------------------------------- regime changes
    st.divider()
    st.markdown("## Regime changes")
    point("One strange day means nothing. We add the evidence up day by day, "
          "and wait for it to stay.")
    show = changes.sort_values("peak_cusum", ascending=False).copy()
    show["sensor"] = show.sensor.map(
        lambda s: SENSOR_LABEL.get(s, s).split(" (")[0])
    st.dataframe(
        show[["unit", "sensor", "direction", "alarm_date", "peak_cusum"]],
        width="stretch", hide_index=True,
        column_config={
            "unit": "Reactor", "sensor": "Sensor",
            "direction": "Moved",
            "alarm_date": st.column_config.DateColumn("Detected",
                                                      format="DD MMM YYYY"),
            "peak_cusum": st.column_config.ProgressColumn(
                "Accumulated evidence", format="%.0f", min_value=0,
                max_value=float(show.peak_cusum.max()))})

    c = st.columns([1, 1])
    units = sorted(df.unit.unique())
    unit = c[0].selectbox("Reactor", units,
                          index=units.index("Plant-04 / R08"))
    sensor = c[1].selectbox("Sensor", SENSORS,
                            format_func=lambda k: SENSOR_LABEL[k])
    st.plotly_chart(fig_cusum_trace(unit, sensor), width="stretch")
    st.markdown(
        '<p class="note">Daily deviation on top, total evidence below. '
        'One odd day falls back to zero. A real change keeps '
        'climbing.</p>', unsafe_allow_html=True)


# =================================================================
# 6. Architecture
# =================================================================

STAGES = {
    # selectable=False: shown in the diagram, absent from the picker.
    "raw": dict(
        label="Raw CSV\n259,200 batches", group="data",
        selectable=False,
        title="The file as delivered",
        body="One row per batch, sixteen columns, no duplicates — and "
             "six defects, four of them able to invalidate the "
             "analysis.",
        facts=[("Batches", "259,200"), ("Columns", "16"),
               ("Defects found", "6")],
        file="01_exploration_and_cleaning.ipynb"),
    "clean": dict(
        label="Cleaning + preprocessing\n6 repairs · 0 batches lost",
        group="data",
        title="Cleaning and preprocessing",
        body="Repair the measurements, preserve what is unknown, block "
             "target leakage, and identify the real process unit. The "
             "result is one clean batch table shared by prediction and "
             "process control.",
        facts=[("Fahrenheit rows converted", "1,506"),
               ("Failure codes blanked", "2,171 cells"),
               ("Batches dropped", "0")],
        file="01_exploration_and_cleaning.ipynb · src/model.py"),

    "model": dict(
        label="CatBoost × 5\nRMSEWithUncertainty", group="model",
        title="NLL + Ensemble",
        body="A normal model returns one number and no opinion about "
             "it. These return two: the shelf life, and the spread/uncertainty "
             "around it.\n\n"
             "Each of the five learns both at once by maximising the "
             "likelihood of the real answer. Getting the value right is "
             "not enough — the spread has to be right too.",
        math=r"\mathcal{L} = \sum_i \left[\; \frac{\log \sigma^2(x_i)}{2}"
             r"\; +\; \frac{\bigl(y_i - \mu(x_i)\bigr)^2}"
             r"{2\,\sigma^2(x_i)} \;\right]",
        gloss="The left-hand term is the rent you pay for claiming a large σ. "
              "A model that says 'I am very sure' and is wrong pays.",
        why="**The split is by time, not at random.** Train is "
            "everything before 8 June, validation the week after, test "
            "the last fortnight — 16 to 29 June. A random split would "
            "let a model train on batches produced *after* the ones it "
            "is tested on, which no plant can do.\n\n",
        facts=[("R² on the time split", "0.796"),
               ("MAE", "0.735 months"),
               ("Features", "9 core · temporal off")],
        file="src/probabilistic_catboost.py"),

    "uncertainty": dict(
        label="σ(x)\naleatoric + epistemic", group="model",
        title="Two kinds of not knowing",
        body="Each model returns a mean and a variance. The ensemble "
             "separates process noise from disagreement between models.",
        math=r"\sigma^2_{\text{total}} \;=\; \underbrace{\tfrac{1}{M}"
             r"\textstyle\sum_m \sigma_m^2}_{\text{aleatoric}} \;+\;"
             r"\underbrace{\tfrac{1}{M}\textstyle\sum_m (\mu_m -"
             r"\bar{\mu})^2}_{\text{epistemic}}",
        gloss="Aleatoric is how wide each curve is: noise in the batch "
              "itself. Epistemic is how far apart the five peaks sit: "
              "the models arguing. More data fixes the second and "
              "never the first.",
        why="Aleatoric uncertainty grows as sensor readings disappear. "
            "Epistemic disagreement is shown separately, but it is not "
            "used for anomaly or regime decisions.",
        facts=[],
        file="src/probabilistic_catboost.py"),

    "conformal": dict(
        label="Conformal\nnormalized, per group", group="model",
        title="An interval with a guarantee",
        body="A σ is a claim. Conformal prediction turns it into a "
             "guarantee that can be checked.\n\n"
             "On data the model never trained on, we measure how many "
             "σ the error actually took. The 90th of those becomes the "
             "multiplier q. Every future interval is μ ± q·σ.",
        math=r"s_i = \frac{\bigl|y_i - \mu(x_i)\bigr|}{\sigma(x_i)}"
             r"\qquad q = s_{\left(\lceil (n+1)(1-\alpha)\rceil\right)}"
             r"\qquad \mu(x) \pm q\,\sigma(x)",
        gloss="Dividing by σ(x) is what makes the width follow the "
              "batch: an easy batch gets a narrow interval, a batch "
              "with three dead sensors gets a wide one. Without that "
              "division every interval would be the same width, and "
              "coverage would fall to 46 % on a blackout.",
        facts=[("Coverage asked / delivered", "90 % / 90.3 %"),
               ("Median width", "2.72 months"),
               ("Width, 0 → 7 sensors blank", "2.46 → 5.86")],
        file="src/conformal_grouped.py"),

    "shap": dict(
        label="SHAP\ncontributions in months", group="explain",
        title="Why that number",
        body="The model predicts 17.97 months. The fleet average is "
             "21.61. SHAP splits that difference of −3.64 across the "
             "inputs, so every variable gets a share in months.\n\n"
             "It comes from game theory.",
        math=r"f(x) \;=\; \phi_0 \;+\; \sum_{j=1}^{p} \phi_j(x)",
        gloss="This is the property that matters. The base value plus "
              "the contributions reproduces the prediction.",
        why="For trees there is an exact algorithm that does it in polynomial "
            "time, and CatBoost has it built in. Averaging the five "
            "members is also exact rather than approximate, because "
            "the ensemble prediction is itself an average.",
        facts=[("Torque", "0.94 months"),
               ("Temperature", "0.60 months"),
               ("pH", "0.51 months")],
        file="src/explain.py"),

    "anomaly": dict(
        label="Anomaly\nfrozen z per reactor", group="detect",
        title="Is this batch odd for its own reactor",
        body="For every physical unit $u$ (**plant × reactor**) and "
             "sensor $s$, use all observed batches in its first 21 "
             "days to estimate its own centre and spread. Freeze both. "
             "A new batch is odd when any available sensor is at least "
             "2 of that unit's reference standard deviations away.",
        math=r"\begin{aligned}"
             r"&\forall u\in\mathcal U,\ \forall s\in\mathcal S,\ "
             r"\forall i\in\mathcal B_u:\\"
             r"&z_{u,s,i}=\frac{x_{u,s,i}-\mu^{\rm ref}_{u,s}}"
             r"{\sigma^{\rm ref}_{u,s}}\\"
             r"&\mathrm{odd}_{u,i}\iff"
             r"\max_{s\ \mathrm{observed}}|z_{u,s,i}|\ge2"
             r"\end{aligned}",
        gloss="The reference mean and standard deviation belong to "
              "that unit and sensor and are frozen after the first 21 "
              "days. Missing sensors do not enter the maximum.",
        why="**Why 2 works as a hard line.** For an ideal uniform "
            "sensor, the population-standardised support ends at "
            "√3 ≈ 1.73. Here μ and σ are estimated, so the empirical "
            "limit is the relevant check: across 232,000 steady batches "
            "the largest value observed is 1.84. A cut at 2 sits beyond "
            "the healthy support seen in these data.",
        facts=[("AUC", "0.965"),
               ("Detection / false alarms", "90.9 % / 0.000 %")],
        file="src/simple_monitor.py"),

    "cusum": dict(
        label="CUSUM\nregime change per unit", group="detect",
        title="Has this reactor changed",
        body="Average the batches by calendar day, separately for every "
             "unit and sensor, so every day contributes one value. "
             "Then compare that value with the reactor's first 21 "
             "daily averages. The resulting $z_t$ feeds a two-sided "
             "CUSUM.",
        math=r"\begin{aligned}"
             r"&\forall u\in\mathcal U,\ \forall s\in\mathcal S,\ "
             r"\forall t\in\mathcal T_u:\\"
             r"&z_{u,s,t}=\frac{\bar x_{u,s,t}-"
             r"\mu^{\rm day,ref}_{u,s}}"
             r"{\sigma^{\rm day,ref}_{u,s}}\\[3pt]"
             r"&C_t^+=\max(0,C_{t-1}^+ +z_t-0.5)\\"
             r"&C_t^-=\max(0,C_{t-1}^- -z_t-0.5)\\[3pt]"
             r"&\max(C_t^+,C_t^-)>8"
             r"\qquad\text{first alert}\\"
             r"&\max(C_{t+3}^+,C_{t+3}^-)>8"
             r"\qquad\text{confirmed}"
             r"\end{aligned}",
        gloss="Here z is the signed distance between today's average "
              "and the frozen daily reference, measured in daily "
              "reference standard deviations. Here t is the first day "
              "either arm crosses 8. The alert is confirmed only if "
              "either arm is above 8 again three daily observations "
              "later.",
        why="Why it caught the cause and not just the symptom. Running "
            "it in both directions on every sensor is what surfaced "
            "density and pH rising alongside torque falling. Torque "
            "alone would have left four possible explanations open. "
            "Three material properties moving together on one day left "
            "one: the raw material changed.",
        facts=[("Units detected", "3 of 3"),
               ("Peak tally vs limit", "2,612 vs 8")],
        file="src/simple_monitor.py"),

    "out": dict(
        label="Prediction + interval\n+ reason + verdict", group="out",
        title="What the engineer receives",
        body="Not a number, but a number with a range, an account of "
             "where it came from, and a verdict on whether it should "
             "be used at all.",
        facts=[],
        file="src/batch_report.py"),
}
EDGES = [("raw", "clean"), ("clean", "model"), ("model", "uncertainty"),
         ("uncertainty", "conformal"), ("model", "shap"),
         ("clean", "anomaly"), ("clean", "cusum"),
         ("conformal", "out"), ("shap", "out"),
         ("anomaly", "out"), ("cusum", "out")]
# Drop a stage from STAGES and its edges go with it. Without this,
# graphviz silently invents an unstyled node for the missing end.
EDGES = [(a, b) for a, b in EDGES if a in STAGES and b in STAGES]
GROUP_COLOUR = {"data": "#9ec5f4", "model": "#2a78d6",
                "explain": "#1baf7a", "detect": "#eda100",
                "out": "#52514e"}


def dot(selected: str) -> str:
    lines = ['digraph {', 'rankdir=TB; bgcolor="transparent";',
             'node [shape=box style="rounded,filled" fontsize=11 '
             'fontname="Segoe UI" penwidth=1.6 margin="0.22,0.14"];',
             'edge [color="#b9b8b1" arrowsize=0.7];']
    for k, s in STAGES.items():
        on = k == selected
        pick = s.get("selectable", True)
        fill = GROUP_COLOUR[s["group"]] if on else "#f4f4f1"
        font = "white" if on and s["group"] != "data" else INK
        pen = '3' if on else '1.2'
        edge = GROUP_COLOUR[s["group"]]
        # A stage that cannot be picked is drawn as context: dashed
        # outline and muted text, so nobody waits for it to respond.
        extra = ('' if pick else
                 f' style="rounded,filled,dashed" fontcolor="{MUTED}"')
        lines.append(
            f'"{k}" [label="{s["label"]}" fillcolor="{fill}" '
            f'fontcolor="{font}" color="{edge}" penwidth={pen}{extra}];')
    for a, b in EDGES:
        lines.append(f'"{a}" -> "{b}";')
    lines.append('}')
    return "\n".join(lines)


def page_architecture() -> None:
    st.title("Architecture")
    st.markdown(
        '<p class="lead">Click any box to see what it does.</p>',
        unsafe_allow_html=True)
    point("Two paths leave the clean data. One <b>predicts</b> the "
          "shelf life. The other decides whether we can <b>trust</b> "
          "that prediction.")

    if "stage" not in st.session_state:
        st.session_state.stage = "model"

    left, right = st.columns([1.05, 1])
    with left:
        st.graphviz_chart(dot(st.session_state.stage),
                          width="stretch")
    with right:
        # Context-only stages are drawn so the flow starts where the data
        # does, but only selectable stages are offered in the picker.
        keys = [k for k, v in STAGES.items() if v.get("selectable", True)]
        st.radio("Stage", keys, key="stage",
                 format_func=lambda k: STAGES[k]["title"])
        s = STAGES[st.session_state.stage]
        st.markdown(f"### {s['title']}")
        st.markdown(s["body"])
        if "math" in s:
            st.latex(s["math"])
            st.markdown(f'<p class="note">{s["gloss"]}</p>',
                        unsafe_allow_html=True)
        if s["facts"]:
            cols = st.columns(len(s["facts"]))
            for col, (lbl, val) in zip(cols, s["facts"]):
                kpi(col, val, lbl)
        if "why" in s:
            st.markdown("")
            st.markdown(s["why"])
        st.markdown(f"\n`{s['file']}`")

    if st.session_state.stage == "clean":
        st.divider()
        st.markdown("### Four decisions, one clean table")
        point("Every step is explicit: repair what is wrong, preserve "
              "what is unknown, and keep prediction separate from "
              "process control.")

        a, b = st.columns(2)
        with a:
            st.markdown("#### ① Repair the scale and schema")
            st.markdown(
                "- Convert **1,506** Fahrenheit readings to Celsius.\n"
                "- Turn **2,171** failure codes into missing values.\n"
                "- Parse all three date formats explicitly.")
        with b:
            st.markdown("#### ② Block leakage and recover the unit")
            st.markdown(
                "- Exclude `purity_grade` and `oxidation_risk_flag`: "
                "both are exact functions of the target.\n"
                "- Use **plant × reactor** as the real unit: 37 units, "
                "not 10 reactor IDs.")

        c, d = st.columns(2)
        with c:
            st.markdown("#### ③ Keep missingness visible")
            st.markdown(
                "- Keep sensor gaps as native `NaN`; do not invent a "
                "measurement.\n"
                "- Add `n_missing` so uncertainty can widen when less "
                "of the batch was observed.\n" \
                "- The model used can handle natively missing values, so no imputation is needed.\n"
                "- **0 batches dropped.**")
        with d:
            st.markdown("#### ④ One table, two independent paths")
            st.markdown(
                "- **Prediction:** estimate shelf life and its interval.\n"
                "- **Control:** compare the clean sensors with a frozen "
                "reference to check if the batch is odd for its own reactor and if the reactor has changed regime.")

   
    if st.session_state.stage == "uncertainty":
        st.divider()
        st.markdown("### What each uncertainty component tells us")
        point("Aleatoric uncertainty reacts to missing information. "
              "Epistemic disagreement is descriptive, not our alarm.")
        left, right = st.columns(2)
        with left:
            st.image(UNCERTAINTY_FIGURES /
                     "diag1_aleatoric_vs_missing.png", width="stretch")
            st.caption("Less sensor information produces a wider "
                       "aleatoric uncertainty.")
        with right:
            st.image(UNCERTAINTY_FIGURES /
                     "diag2_epistemic_regime_change.png", width="stretch")
            st.caption("Disagreement does not rise after 8 May; it falls "
                       "slightly. Epistemic uncertainty is not the regime "
                       "detector: frozen-z and CUSUM are.")

    if st.session_state.stage == "conformal":
        st.divider()
        st.markdown("### Is σ the right size, not just the right shape")
        point("Every error divided by the σ the model claimed for it. "
              "If σ is honest, this lands on 1.")
        left, right = st.columns([1.3, 1])
        with left:
            st.plotly_chart(fig_residual_calibration(), width="stretch")
        with right:
            r = residual_stats()
            k = st.columns(2)
            kpi(k[0], f"{r['mean']:+.3f}", "mean — should be 0")
            kpi(k[1], f"{r['sd']:.3f}", "spread — should be 1")
            st.markdown(
                f"The **shape** is not normal. It is flatter. "
               
                f"**That is exactly "
                "why the interval is conformal and not Gaussian** — "
                "assuming normality here would get both ends wrong.")

        st.markdown("#### The same check, quantile by quantile")
        qq, note = st.columns([1.05, 1])
        with qq:
            st.image(UNCERTAINTY_FIGURES / "z_qqplot.png",
                     width="stretch")
        with note:
            st.markdown(
                "If standardized residuals were Gaussian, the points "
                "would follow the red diagonal. Their systematic curve "
                "shows that a Gaussian multiplier is not enough; "
                "conformal calibration learns the multiplier from the "
                "observed residuals instead.")

    if st.session_state.stage == "shap":
        st.divider()
        st.markdown("### One prediction, decomposed")
        point("Start at the fleet baseline. Each bar adds or removes "
              "months until the model reaches this batch's prediction.")
        row = get_clean().query("batch_id == 'BATCH-2023-006214'")
        out = predict_row(row)
        st.plotly_chart(
            fig_waterfall(out["contrib"], out["base"], SENSOR_LABEL),
            width="stretch")
        st.caption("A real test batch. SHAP is computed from the same five "
                   "CatBoost members used for its prediction.")

    if st.session_state.stage == "anomaly":
        st.divider()
        st.markdown("### Why the baseline belongs to the reactor")
        point("The reactors are already slightly different while all of "
              "them are healthy. A single fleet baseline would confuse "
              "those normal offsets with anomalous batches.")
        sensor = st.selectbox(
            "Sensor in the 21-day reference",
            SENSORS, index=SENSORS.index("mixing_torque_nm"),
            format_func=lambda key: SENSOR_LABEL[key],
            key="architecture_reference_sensor")
        st.plotly_chart(fig_reference_centres(sensor), width="stretch")
        refs, _, _ = get_monitor()
        centres = np.array([r.mean[sensor] for r in refs.values()])
        typical_sd = float(np.median([r.sd[sensor] for r in refs.values()]))
        span = float(centres.max() - centres.min())
        st.caption(
            f"Each point is μ_ref for one plant-reactor unit, estimated "
            f"from the same known-normal 21 days. For this sensor the "
            f"centres run from {centres.min():.5g} to "
            f"{centres.max():.5g}; their span is {span:.4g}, or "
            f"{span / typical_sd:.2f} × a typical within-reactor σ_ref. "
            "The dotted line is descriptive only: the detector uses "
            "each point as that reactor's own centre.")

    if st.session_state.stage == "cusum":
        st.divider()
        st.markdown("### A real four-day example")
        point("Plant-04 / R08 torque crosses the limit on 8 May and is "
              "confirmed at the check three daily observations later.")
        st.latex(
            r"C_{8\,May}^-=\max(0,\ 0.807836-(-52.214394)-0.5)\approx"
            r"\color{#2a78d6}{52.52}")
        st.latex(
            r"C_{9\,May}^-=\max(0,\ 52.522230-(-49.342976)-0.5)\approx"
            r"\color{#168a67}{101.37}")
        st.latex(
            r"C_{10\,May}^-=\max(0,\ 101.365206-(-50.859772)-0.5)\approx"
            r"\color{#a56800}{151.72}")
        st.latex(
            r"C_{11\,May}^-=\max(0,\ 151.724978-(-49.844991)-0.5)\approx"
            r"\color{#c43d3d}{201.07}")

        example = pd.DataFrame([
            {"Day": "8 May", "Daily torque (N·m)": 10.7581,
             "zₜ": -52.21, "C⁻": 52.52,
             "Meaning": "first crossing: candidate"},
            {"Day": "9 May", "Daily torque (N·m)": 10.8270,
             "zₜ": -49.34, "C⁻": 101.37,
             "Meaning": "1st daily observation after"},
            {"Day": "10 May", "Daily torque (N·m)": 10.7906,
             "zₜ": -50.86, "C⁻": 151.72,
             "Meaning": "2nd daily observation after"},
            {"Day": "11 May", "Daily torque (N·m)": 10.8149,
             "zₜ": -49.84, "C⁻": 201.07,
             "Meaning": "3rd observation: confirmed"},
        ])
        row_colours = [
            ("#eaf2fc", "#2a78d6"),
            ("#e7f7f1", "#168a67"),
            ("#fdf1dd", "#a56800"),
            ("#fbe4e4", "#c43d3d"),
        ]

        def colour_example_row(row):
            background, accent = row_colours[row.name]
            common = f"background-color: {background}"
            return [f"{common}; color: {accent}; font-weight: 700"] + [
                common] * (len(row) - 1)

        styled_example = (example.style
                          .apply(colour_example_row, axis=1)
                          .format({"Daily torque (N·m)": "{:.4f}",
                                   "zₜ": "{:.2f}", "C⁻": "{:.2f}"}))
        st.dataframe(styled_example, hide_index=True, width="stretch")
        st.markdown(
            "The first value above **8** is only the candidate. On "
            "11 May—the third daily observation after it—the tally is "
            "still above 8, so the change is **confirmed**.")

    if st.session_state.stage == "out":
        st.divider()
        st.markdown("### One batch in, everything out")
        point("One batch goes in. Six things come out, and each one is "
              "made by a different box in the diagram above.")

        df = get_clean()
        bid = "BATCH-2023-000258"
        row = df[df.batch_id == bid]
        an = anomaly_for(row)
        ch = regime_for(row.unit.iloc[0], row.ts.iloc[0])

        st.plotly_chart(fig_output_flow(bid), width="stretch")

        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                "The last two lines are the ones that decide. **The "
                "frozen z answers *is this batch odd*; the CUSUM "
                "answers *has this reactor moved*.** Both fired here, "
                "so the range is shown but its 90 % no longer "
                "holds.\n\n"
                "Had both been green, the range would keep its promise "
                "and the batch could be released on it.")
        with right:
            st.markdown(pill("frozen z: max |z| = "
                             f"{an['z']:.2f}  ≥  2", "bad"),
                        unsafe_allow_html=True)
            st.markdown(pill("CUSUM: torque down since "
                             f"{ch.iloc[0].alarm_date:%d %b}", "bad"),
                        unsafe_allow_html=True)
            st.markdown(
                '<p class="note">This is why the two detectors sit '
                'outside the model. The prediction comes from data the '
                'model has seen. Whether that data still describes this '
                'reactor is a different question, and a different box '
                'answers it.</p>', unsafe_allow_html=True)

    if st.session_state.stage == "model":
        st.divider()
        st.markdown("### What CatBoost is")
        left, right = st.columns([1.05, 1])
        with left:
            st.plotly_chart(fig_boosting(), width="stretch")
            st.markdown(
                '<p class="note">Each tree is trained on what the '
                'previous ones still got wrong. Alone a tree is weak; '
                'three thousand of them, each correcting the last, are '
                'not.</p>', unsafe_allow_html=True)
            st.markdown(
                "**Family:** gradient-boosted decision trees. The same "
                "family as XGBoost and LightGBM.\n\n"
                "It handles natively the 37 plant-reactor unit names "
                "(with no one-hot encoding) and missing values. " \
                "It has a loss function that predicts an "
                "uncertainty as well as a "
                "value — `RMSEWithUncertainty`, the formula above."
            )
        with right:
            st.markdown("**The settings that matter, and why**")
            st.dataframe(pd.DataFrame([
                {"setting": "loss_function", "value":
                 "RMSEWithUncertainty",
                 "why": "returns μ and σ, not just μ"},
                {"setting": "iterations", "value": "3000",
                 "why": "an upper bound; training stops itself"},
                {"setting": "early_stopping_rounds", "value": "100",
                 "why": "stops when the validation week stops improving"},
                {"setting": "learning_rate", "value": "0.05",
                 "why": "small steps, so no single tree dominates"},
                {"setting": "depth", "value": "6",
                 "why": "ample for this feature set; deeper starts memorising"},
                {"setting": "l2_leaf_reg", "value": "6.0",
                 "why": "raised on purpose: four sensors are near-noise "
                        "and a loose model invents importance for them"},
                {"setting": "random_seed", "value": "0, 1, 2, 3, 4",
                 "why": "the five members; their spread is the "
                        "epistemic term"},
            ]), hide_index=True, width="stretch",
                column_config={
                    "setting": st.column_config.TextColumn("Setting",
                                                           width="small"),
                    "value": st.column_config.TextColumn("Value",
                                                         width="small"),
                    "why": st.column_config.TextColumn("Why", width="large")})

        st.divider()
        st.markdown("### How one batch becomes a prediction")
        point("The batch goes to <b>five models at the same time</b>. "
              "Each model returns <b>two numbers</b>: a value, and how "
              "sure it is about that value.")

        blank = st.radio(
            "Sensors blank", [0, 3, 7],
            format_func=lambda k: (f"{7 - k} of 7 sensors reporting"),
            horizontal=True, key="ens_blank")
        e = ensemble_example(blank)

        st.plotly_chart(fig_ensemble_flow(blank), width="stretch")

        # left, right = st.columns([1.25, 1])
        # with left:
        #     st.plotly_chart(fig_ensemble(blank), width="stretch")
        #     st.markdown(
        #         '<p class="note">The same five models, drawn as curves. '
        #         'Each thin curve is one model. The thick blue curve is '
        #         'the answer.</p>', unsafe_allow_html=True)
        # with right:
        st.markdown("**Two kinds of not knowing**")
        k = st.columns(2)
        kpi(k[0], f"{e['aleatoric']:.2f}", "noise in the data — how "
                                            "wide each curve is")
        kpi(k[1], f"{e['epistemic']:.2f}", "model disagreement — "
                                            "how far apart they sit")
        st.markdown(
            "The five models were trained on the same data with "
            "different seeds. So they almost agree: their means "
            f"sit within **{e['epistemic']:.2f} months** of each "
            "other.\n\n")


# =================================================================

# Numbered because the talk follows this order: the numbers keep the
# presenter and the room on the same slide.
PAGES = {
    "1 · The data": page_data,
    "2 · The finding": page_finding,
    "3 · Architecture": page_architecture,
    "4 · Batch inspector": page_batch,
    "5 · How good is it": page_quality,
    "6 · Anomaly control": page_anomaly,
}

with st.sidebar:
    st.markdown("### Fragrance Shelf-Life")
    st.caption("Root-cause analysis and predictive system")
    choice = st.radio("Section", list(PAGES), label_visibility="collapsed")

PAGES[choice]()

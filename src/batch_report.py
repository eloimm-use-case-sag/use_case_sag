"""End-to-end console report for representative batches."""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import simple_monitor as SM
from paths import RAW_CSV
from explain import shap_contributions
from conformal_calibration import present, score_slice
from conformal_grouped import GroupedConformal
from model import (FAULTY_UNITS, SENSORS, TARGET, TARGET_RANGE,
                   load, to_grade, to_oxidation_risk)
from probabilistic_catboost import (ProbabilisticEnsemble,
                                    TEST_START, VAL_START)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)

TECHNICAL_MAX, INDUSTRIAL_MAX = 20.0, 25.0
LABEL = {
    "temp_c": "Reaction temperature (C)",
    "vessel_pressure_bar": "Vessel pressure (bar)",
    "refractive_index": "Refractive index",
    "density_g_cm3": "Density (g/cm3)",
    "ph_level": "pH level",
    "mixing_torque_nm": "Mixing torque (N.m)",
    "mass_flow_rate_kg_h": "Mass flow rate (kg/h)",
}


def grades_possible(lo: float, hi: float) -> list[str]:
    out = []
    for name, a, b in (("Technical Grade", -np.inf, TECHNICAL_MAX),
                       ("Industrial Grade", TECHNICAL_MAX,
                        INDUSTRIAL_MAX),
                       ("Perfumery Grade", INDUSTRIAL_MAX, np.inf)):
        if hi > a and lo < b:
            out.append(name)
    return out


class Pipeline:
    """Everything assembled once, then queried per batch."""

    def __init__(self):
        print("Assembling the pipeline...")
        self.raw = pd.read_csv(RAW_CSV,
                               low_memory=False).set_index("batch_id")
        self.ens = ProbabilisticEnsemble().load_saved()
        self.clean = load(
            include_temporal=self.ens.use_temporal_features)

        # Interval: calibrated on the validation slice, one q per
        # missing-count group.
        cal = score_slice(self.clean[(self.clean.ts >= VAL_START)
                                     & (self.clean.ts < TEST_START)],
                          self.ens)
        self.gc = GroupedConformal(coverage=0.90, min_group=100,
                                   fallback="global").fit(cal)

        # The Streamlit app needs only the model and calibration from
        # this object; its monitor has a separate cached instance. Keep
        # the report's monitor lazy so loading the app does not fit the
        # same references twice.
        self.references = None
        self.anomalies = None
        self.changes = None
        print(f"  ready: q by group {self.gc.q_.get(0):.3f}-"
              f"{max(self.gc.q_.values()):.3f}\n")

    def _ensure_monitor(self) -> None:
        """Build the current frozen-z monitor only for console reports."""
        if self.references is not None:
            return
        self.references = SM.fit_references(self.clean)
        self.anomalies = SM.batch_scores(self.clean, self.references)
        ch = SM.detect_changes(self.clean, self.references)
        self.changes = ch[
            ch.confirmed & (ch.peak_cusum > SM.SEVERITY_CUT)]

    # ---------------------------------------------------------- report

    def report(self, batch_id: str) -> None:
        self._ensure_monitor()
        row = self.clean[self.clean.batch_id == batch_id]
        i = row.index[0]
        raw = self.raw.loc[batch_id]

        print("=" * 76)
        print(f"  BATCH {batch_id}")
        print("=" * 76)
        print(f"  unit {row.unit.iloc[0]}      "
              f"produced {row.ts.iloc[0]:%d %b %Y %H:%M}")

        # ---- 1. cleaning
        print("\n  1. WHAT THE FILE SAID, AND WHAT WAS REPAIRED")
        fixes = []
        if raw.temp_unit == "F":
            fixes.append(f"temperature arrived in F ({raw.temp_value:.2f})"
                         f" and was converted")
        for s, code in (("vessel_pressure_bar", -1.0),
                        ("ph_level", 14.0), ("density_g_cm3", 860.0)):
            if raw[s] == code:
                fixes.append(f"{s} held the failure code {code:g}, "
                             "blanked")
        if raw.temp_value in (210.0, 410.0):
            fixes.append(f"temperature held the failure code "
                         f"{raw.temp_value:g}, blanked")
        print("     " + ("\n     ".join(fixes) if fixes
                         else "nothing to repair"))

        # ---- 2. the readings, against this unit's own normal
        base = self.references.get(row.unit.iloc[0])
        print("\n  2. THE READINGS, AGAINST THIS REACTOR'S OWN BASELINE")
        for j, s in enumerate(SENSORS):
            v = row[s].iloc[0]
            if pd.isna(v):
                print(f"     {LABEL[s]:28s}   no reading")
                continue
            sd = base.sd[s] if base else np.nan
            z = (v - base.mean[s]) / sd if base and sd > 0 else np.nan
            mark = "  <-- far from normal" if abs(z) > 3 else ""
            print(f"     {LABEL[s]:28s} {v:9.4f}   "
                  f"{z:+6.2f} sigma{mark}")

        # ---- 3. prediction and uncertainty
        pred = self.ens.predict(row)
        # The raw model output is kept, because the interval has to be
        # centred on what was calibrated. `present` below returns the
        # display value: the same number clipped to the declared 18-30
        # month range, which is what section 3b reports against.
        mu_raw = float(pred.predicted_months.iloc[0])
        mu = float(np.clip(mu_raw, *TARGET_RANGE))
        al = float(pred.aleatoric_std.iloc[0])
        ep = float(pred.epistemic_std.iloc[0])
        tot = float(pred.total_std.iloc[0])
        print("\n  3. PREDICTION")
        print(f"     expected shelf life            {mu:8.2f} months")
        print(f"     data noise      (aleatoric)    {al:8.3f}")
        print(f"     model disagreement (epistemic) {ep:8.3f}")
        print(f"     combined                       {tot:8.3f}")

        # ---- 3b. why that number
        # Exact SHAP: the contributions and the fleet-wide base add up
        # to the model's own output, so this is an account in months
        # rather than a ranking.
        contrib, fleet_base = shap_contributions(self.ens, row)
        c = contrib.iloc[0].sort_values(key=abs, ascending=False)
        print("\n  3b. WHY THAT NUMBER")
        print(f"      starting from the fleet average of "
              f"{fleet_base:.2f} months")
        for kf, v in c.items():
            if abs(v) < 0.01:
                continue
            reading = row[kf].iloc[0]
            shown = ("no reading" if pd.isna(reading)
                     else f"{reading:.3f}"
                     if isinstance(reading, (int, float, np.floating))
                     else str(reading))
            print(f"      {LABEL.get(kf, kf):26s} {shown:>16s}  "
                  f"{v:+6.2f}")
        model_out = fleet_base + c.sum()
        print(f"      {'':26s} {'':>16s}  {'-' * 6}")
        print(f"      {'model output':26s} {'':>16s}  "
              f"{model_out:6.2f}")
        if abs(model_out - mu) > 0.005:
            print(f"      held at the {TARGET_RANGE[0]:.0f}-month floor"
                  f" the process cannot go below -> {mu:.2f}")

        # ---- 4. interval
        q = self.gc.q_.get(int(row.n_missing.iloc[0]),
                           self.gc.q_global_)
        _, lo, hi = present(mu_raw, tot, q)
        lo, hi = float(lo), float(hi)
        print("\n  4. INTERVAL  (90 % conformal, normalized)")
        print(f"     q for the {int(row.n_missing.iloc[0])}-blank group  "
              f"{q:.4f}")
        print(f"     interval                       "
              f"{lo:8.2f} to {hi:.2f}   ({hi - lo:.2f} wide)")

        # ---- 5. business
        poss = grades_possible(lo, hi)
        print("\n  5. WHAT THIS MEANS COMMERCIALLY")
        print(f"     most likely grade              {to_grade(mu)}")
        print(f"     oxidation risk flag            "
              f"{bool(to_oxidation_risk(mu))}")
        print(f"     grades the interval allows     "
              f"{' or '.join(g.split()[0] for g in poss)}")
        print(f"     decision                       "
              f"{'commit' if len(poss) == 1 else 'hold for the lab'}")

        # ---- 6. anomaly
        anomaly = self.anomalies.loc[i]
        print("\n  6. IS ANY SENSOR FAR FROM THIS REACTOR'S NORMAL?")
        if pd.isna(anomaly.max_abs_z):
            print("     not scorable - no sensor readings at all")
        else:
            print(f"     furthest sensor                "
                  f"{LABEL.get(anomaly.worst_sensor, anomaly.worst_sensor)}")
            print(f"     deviation                      "
                  f"{anomaly.worst_z:+8.2f} sigma "
                  f"(on {int(anomaly.n_observed)} sensors)")
            print(f"     anomaly cut                    "
                  f"{SM.BATCH_CUT:8.2f} sigma")
            print(f"     verdict                        "
                  f"{'ANOMALOUS' if anomaly.is_anomaly else 'ordinary'}")

        # ---- 7. regime
        unit_ch = self.changes[
            (self.changes.unit == row.unit.iloc[0])
            & (self.changes.alarm_date <= row.ts.iloc[0])]
        print("\n  7. HAS THIS REACTOR CHANGED REGIME?")
        if len(unit_ch) == 0:
            print("     no confirmed change before this batch")
        else:
            first = unit_ch.alarm_date.min()
            print(f"     regime change confirmed on    "
                  f"{first:%d %b %Y}")
            for _, c in unit_ch.sort_values("alarm_date").iterrows():
                print(f"       {c.sensor:22s} {c.direction:5s} "
                      f"from {c.alarm_date:%d %b}  "
                      f"(CUSUM {c.peak_cusum:.0f})")

        # ---- 8. verdict
        shifted = len(unit_ch) > 0
        is_anomaly = bool(anomaly.is_anomaly)
        wide = (hi - lo) > 4.0
        if shifted:
            state = ("REGIME SHIFT - the 90 % coverage is no longer "
                     "guaranteed for this unit")
        elif is_anomaly:
            state = "ANOMALOUS BATCH - treat the prediction with care"
        elif wide:
            state = "IN DOMAIN, LOW CONFIDENCE - too little sensor data"
        else:
            state = "IN DOMAIN, CONFIDENT"
        print(f"\n  8. VERDICT\n     {state}")

        if TARGET in row:
            actual = float(row[TARGET].iloc[0])
            print(f"\n  ---- what the lab measured months later: "
                  f"{actual:.2f} months ({to_grade(actual)})")
            print(f"       inside the interval: "
                  f"{lo <= actual <= hi}")
        print()


def main() -> None:
    p = Pipeline()
    clean = p.clean
    post = clean[clean.ts >= TEST_START]

    picks = {
        "an ordinary batch, everything reading":
            post[(post.n_missing == 0)
                 & ~post.unit.isin(FAULTY_UNITS)].iloc[500],
        "a faulty unit after the change":
            post[post.unit.isin(FAULTY_UNITS)
                 & (post.n_missing == 0)].iloc[200],
        "a batch with no sensor data at all":
            post[post.n_missing == 7].iloc[10],
    }
    for label, row in picks.items():
        print(f"\n\n{'#' * 76}\n#  {label.upper()}\n{'#' * 76}")
        p.report(row.batch_id)


if __name__ == "__main__":
    main()

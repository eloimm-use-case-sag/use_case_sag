"""Evaluate batch anomalies and regime changes on the known 8 May event."""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

import simple_monitor as SM
from model import FAULTY_UNITS, load
from paths import TABLES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check_confirmation_rule() -> None:
    """A breach must survive to the actual confirmation day."""
    dn = np.zeros(5)
    transient = np.array([0.0, 9.0, 0.0, 0.0, 0.0])
    persistent = np.array([0.0, 9.0, 9.0, 9.0, 9.0])

    assert not SM.confirmation_holds(
        transient, dn, first=1, h=8.0, confirm_days=3)
    assert SM.confirmation_holds(
        persistent, dn, first=1, h=8.0, confirm_days=3)
    assert not SM.confirmation_holds(
        persistent, dn, first=3, h=8.0, confirm_days=3)


def main() -> None:
    print("=" * 74)
    print("  SIMPLE MONITOR - CURRENT IMPLEMENTATION EVALUATION")
    print("=" * 74)

    check_confirmation_rule()
    print("  confirmation rule: transient, persistent and short-tail checks OK")

    df = load()
    shifted = (df.unit.isin(FAULTY_UNITS)
               & (df.ts >= SM.CHANGE_DATE)).to_numpy()
    refs = SM.fit_references(df)
    scores = SM.batch_scores(df, refs)
    score = scores.max_abs_z.to_numpy()
    flag = scores.is_anomaly.to_numpy()
    ok = ~np.isnan(score)

    batch = {
        "n_batches": int(len(df)),
        "n_known_post_change": int(shifted.sum()),
        "scorable_pct": float(ok.mean() * 100),
        "auc": float(roc_auc_score(shifted[ok], score[ok])),
        "detection_pct": float(flag[ok & shifted].mean() * 100),
        "false_alarm_pct": float(flag[ok & ~shifted].mean() * 100),
        "unscorable_batches": int((~ok).sum()),
    }

    print("\n  BATCH ANOMALIES")
    print(f"    scorable       {batch['scorable_pct']:8.3f} %")
    print(f"    AUC            {batch['auc']:8.3f}")
    print(f"    detection      {batch['detection_pct']:8.3f} %")
    print(f"    false alarms   {batch['false_alarm_pct']:8.3f} %")
    print(f"    no sensors     {batch['unscorable_batches']:8,d}")

    support = SM.healthy_support(df, refs)
    print("\n  HEALTHY SUPPORT")
    print(f"    observed max |z| {support['max']:.3f}")
    print(f"    anomaly cut      {SM.BATCH_CUT:.3f}")

    changes = SM.detect_changes(df, refs)
    confirmed = changes[changes.confirmed]
    material = confirmed[confirmed.peak_cusum > SM.SEVERITY_CUT]
    found = set(material.unit)
    false_material = material[~material.unit.isin(FAULTY_UNITS)]
    torque = material[material.sensor == "mixing_torque_nm"]

    if found != set(FAULTY_UNITS):
        raise AssertionError(
            f"material regime signals found {sorted(found)}, expected "
            f"{sorted(FAULTY_UNITS)}")
    if len(false_material):
        raise AssertionError(
            f"{len(false_material)} material signals on healthy units")

    regime = {
        "confirmed_signals": int(len(confirmed)),
        "material_signals": int(len(material)),
        "units_found": int(len(found)),
        "false_material_signals": int(len(false_material)),
        "first_torque_alarm": (
            torque.alarm_date.min().strftime("%Y-%m-%d")
            if len(torque) else None),
        "sensors": sorted(material.sensor.unique().tolist()),
    }

    print("\n  REGIME CHANGES")
    print(f"    confirmed signals      {regime['confirmed_signals']:8,d}")
    print(f"    above severity cut     {regime['material_signals']:8,d}")
    print(f"    affected units found   {regime['units_found']:8,d}")
    print(f"    false material signals {regime['false_material_signals']:8,d}")
    print(f"    first torque alarm     {regime['first_torque_alarm']}")
    print(f"    sensors                {', '.join(regime['sensors'])}")

    out = TABLES / "simple_monitor_evaluation.json"
    out.write_text(json.dumps({
        "known_event_definition": (
            "faulty unit and timestamp on or after 2023-05-08"),
        "batch": batch,
        "healthy_support": support,
        "regime": regime,
    }, indent=2), encoding="utf-8")
    print(f"\n  saved: {out.name}\n")


if __name__ == "__main__":
    main()

"""One entry point for the whole thing.

    python run_pipeline.py              the pipeline, reusing the models
    python run_pipeline.py --retrain    from scratch, ~15 min
    python run_pipeline.py --analysis   the evaluations and ablations
    python run_pipeline.py --list       what the stages are

Use --no-temporal-features or --temporal-features with --retrain to
change the saved model's feature profile.

Training is skipped when the five saved models are already on disk,
because it is the only slow step and nothing downstream changes it.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from paths import MODELS  # noqa: E402

# name, module, one-line purpose
PIPELINE = [
    ("features", "features",
     "optional torque z/CUSUM features, written to data/"),
    ("train", "probabilistic_catboost",
     "five CatBoost members with RMSEWithUncertainty"),
    ("uncertainty-plots", "uncertainty_diagnostics",
     "the two uncertainty diagnostics shown in Streamlit"),
    ("calibrate", "conformal_calibration",
     "normalized conformal intervals"),
    ("calibrate-groups", "conformal_grouped",
     "one conformal q per missing-sensor group"),
    ("monitor", "simple_monitor",
     "frozen-z anomaly flags and regime changes"),
    ("explain", "explain",
     "SHAP contributions, global and per batch"),
    ("report", "batch_report",
     "worked examples through the whole stack"),
]

ANALYSIS = [
    ("eval-monitor", "evaluate_simple_monitor",
     "the frozen-z monitor on the known 8 May event"),
]

EXPERIMENTS = [
    ("eval-heldout", "evaluate_heldout_regime",
     "refit after hiding the affected regime, then test on it"),
]


def saved_temporal_profile() -> bool | None:
    """Read the profile of the saved models, including older configs."""
    path = MODELS / "config.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text("utf-8"))
    if "use_temporal_features" in cfg:
        return bool(cfg["use_temporal_features"])
    temporal = {"torque_z_ref", "torque_cusum_up", "torque_cusum_down"}
    return bool(temporal & set(cfg.get("features", [])))


def run(stages, retrain: bool, use_temporal_features: bool) -> None:
    for name, module, purpose in stages:
        if name == "features" and not use_temporal_features:
            print("\n>>> features: optional temporal profile disabled, skipping")
            continue
        if name == "train" and not retrain and \
                (MODELS / "config.json").exists():
            print(f"\n>>> {name}: models already on disk, skipping "
                  "(--retrain to force)")
            continue
        print(f"\n{'=' * 72}\n>>> {name}  -  {purpose}\n{'=' * 72}")
        t0 = time.time()
        runpy.run_module(module, run_name="__main__")
        print(f"--- {name} finished in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retrain", action="store_true",
                   help="refit the five CatBoost members")
    p.add_argument("--analysis", action="store_true",
                   help="run the evaluations instead of the pipeline")
    p.add_argument("--list", action="store_true",
                   help="print the stages and exit")
    p.add_argument("--only", metavar="STAGE",
                   help="run a single stage by name")
    profile = p.add_mutually_exclusive_group()
    profile.add_argument(
        "--temporal-features", dest="temporal_features",
        action="store_true", default=None,
        help="train with torque z/CUSUM features")
    profile.add_argument(
        "--no-temporal-features", dest="temporal_features",
        action="store_false",
        help="train with the core batch features only")
    a = p.parse_args()

    if a.temporal_features is not None and not a.retrain:
        p.error("changing the feature profile requires --retrain")
    saved_profile = saved_temporal_profile()
    use_temporal = (a.temporal_features
                    if a.temporal_features is not None
                    else saved_profile if saved_profile is not None
                    else False)
    os.environ["FRAGRANCE_USE_TEMPORAL_FEATURES"] = (
        "1" if use_temporal else "0")

    stages = ANALYSIS if a.analysis else PIPELINE
    if a.list:
        source = "saved model" if saved_profile is not None else "default"
        name = "core + temporal" if use_temporal else "core only"
        print(f"\nfeature profile: {name} ({source})")
        for label, group in (("pipeline", PIPELINE),
                             ("analysis", ANALYSIS),
                             ("experiments", EXPERIMENTS)):
            print(f"\n{label}:")
            for name, module, purpose in group:
                print(f"  {name:18s} {module:34s} {purpose}")
        return
    if a.only:
        stages = [s for s in PIPELINE + ANALYSIS + EXPERIMENTS
                  if s[0] == a.only]
        if not stages:
            p.error(f"unknown stage {a.only!r}; try --list")

    t0 = time.time()
    run(stages, a.retrain, use_temporal)
    print(f"\n{'=' * 72}\nall done in {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()

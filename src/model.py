"""Shared data schema, loader and business-label helpers."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import CLEAN_PARQUET, ROOT

RAIZ = ROOT
PARQUET = CLEAN_PARQUET

# ------------------------------------------------------------- schema

SENSORS = [
    "temp_c", "vessel_pressure_bar", "refractive_index", "density_g_cm3",
    "ph_level", "mixing_torque_nm", "mass_flow_rate_kg_h",
]
# Available the moment a batch closes, unlike the lab result.
CONTEXT = ["plant_location", "reactor"]
FEATURES = SENSORS + CONTEXT
TARGET = "stability_months"

# Derived from the target, so never inputs. See exploration phase.
FORBIDDEN = ["purity_grade", "oxidation_risk_flag"]

LABELS = {
    "temp_c": "Reaction temperature",
    "vessel_pressure_bar": "Vessel pressure",
    "refractive_index": "Refractive index",
    "density_g_cm3": "Density",
    "ph_level": "pH level",
    "mixing_torque_nm": "Mixing torque",
    "mass_flow_rate_kg_h": "Mass flow rate",
    "plant_location": "Plant",
    "reactor": "Reactor",
}

FAULTY_UNITS = ["Plant-04 / R03", "Plant-04 / R07", "Plant-04 / R08"]
CHANGE_DATE = pd.Timestamp("2023-05-08")

# Declared in the brief. Used to keep predictions physically possible:
# a gradient-boosted model does not know its target is bounded, and
# clipping to the true support can only help interval coverage.
TARGET_RANGE = (18.0, 30.0)


def load(path: Path | None = None,
         include_temporal: bool = False) -> pd.DataFrame:
    """Load clean data, optionally adding the three temporal features."""
    path = path or PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run 01_exploration_and_cleaning.ipynb "
            "first: it writes the cleaned dataset this model reads."
        )
    df = pd.read_parquet(path)
    leaked = set(FEATURES) & set(FORBIDDEN)
    if leaked:
        raise AssertionError(f"target-derived column in features: {leaked}")
    if not include_temporal:
        return df

    # Reuse the materialised optional features only when they are newer
    # than the clean dataset they derive from.
    from features import add_frozen_features
    from paths import MODEL_INPUT
    if (MODEL_INPUT.exists()
            and MODEL_INPUT.stat().st_mtime >= path.stat().st_mtime):
        return pd.read_parquet(MODEL_INPUT)
    return add_frozen_features(df)


# ------------------------------------------------------- business view

def to_grade(months: np.ndarray) -> np.ndarray:
    """Map shelf life to its commercial grade."""
    return np.select(
        [np.asarray(months) < 20.0, np.asarray(months) < 25.0],
        ["Technical Grade", "Industrial Grade"],
        default="Perfumery Grade",
    )


def to_oxidation_risk(months: np.ndarray) -> np.ndarray:
    return np.asarray(months) < 21.0

"""Central paths for data, models, figures and tables."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "fragrance_stability_dataset_corrupted_20260601_161558.csv"

DATA = ROOT / "data"
CLEAN_PARQUET = DATA / "fragrance_clean.parquet"
MODEL_INPUT = DATA / "fragrance_model_input.parquet"

OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

# Figures are grouped by the question they answer, not by the script
# that drew them.
FIG_EDA = FIGURES / "exploration"
FIG_MODEL = FIGURES / "model"
FIG_UNCERTAINTY = FIGURES / "uncertainty"

DOCS = ROOT / "docs"

for _d in (DATA, MODELS, FIGURES, TABLES, DOCS,
           FIG_EDA, FIG_MODEL, FIG_UNCERTAINTY):
    _d.mkdir(parents=True, exist_ok=True)

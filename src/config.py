"""
Configuration and constants for the Heart Disease Predictive Modeling project.
Centralizes paths, feature group categorizations, seeds, and plotting aesthetics.
"""
from pathlib import Path

# ==============================================================================
# 1. Project Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "heart_disease_prediction_2026.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure output directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. General Experimentation Settings
# ==============================================================================
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# 3. Feature Taxonomy & Groupings (Per UASLP Proposal & Dataset Specs)
# ==============================================================================
TARGET_COL = "heart_disease"

DEMOGRAPHIC_COLS = [
    "age",
    "sex",
    "bmi",
    "region",
    "income_level",
]

LIFESTYLE_COLS = [
    "smoking_status",
    "alcohol_intake",
    "physical_activity",
    "diet_quality",
    "sleep_hours",
    "stress_level",
]

MEDICAL_HISTORY_COLS = [
    "diabetes",
    "hypertension",
    "family_history",
    "previous_heart_event",
]

CLINICAL_MARKER_COLS = [
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "max_heart_rate",
    "resting_ecg",
    "exercise_angina",
    "st_depression",
    "st_slope",
]

WEARABLE_COLS = [
    "avg_resting_hr",
    "hrv_score",
]

ALL_FEATURE_COLS = (
    DEMOGRAPHIC_COLS
    + LIFESTYLE_COLS
    + MEDICAL_HISTORY_COLS
    + CLINICAL_MARKER_COLS
    + WEARABLE_COLS
)

# Variable Types for Pipelines
NOMINAL_COLS = ["sex", "region", "smoking_status", "alcohol_intake", "resting_ecg", "st_slope"]

ORDINAL_COLS = ["income_level", "physical_activity"]

ORDINAL_CATEGORIES = {
    "income_level": ["Low", "Middle", "High"],
    "physical_activity": ["Sedentary", "Light", "Active", "Very Active"],
}

BINARY_NUMERIC_COLS = [
    "diabetes",
    "hypertension",
    "family_history",
    "previous_heart_event",
    "fasting_blood_sugar",
    "exercise_angina",
]

CONTINUOUS_NUMERIC_COLS = [
    "age",
    "bmi",
    "diet_quality",
    "sleep_hours",
    "stress_level",
    "resting_bp",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
    "avg_resting_hr",
    "hrv_score",
]

# ==============================================================================
# 4. Visualization & Aesthetic Standards
# ==============================================================================
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
POS_COLOR = "#e34948"  # High Risk / Disease
NEG_COLOR = "#2a78d6"  # Low Risk / Healthy

PLOT_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8f8f8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
}


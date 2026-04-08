"""
Central configuration for the nutriscore-predictor pipeline.

All paths, feature lists, and hyperparameters live here so that any change
propagates automatically to every module that imports them.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"

DATA_RAW = DATA_DIR / "openfoodfacts_data_new.csv"
DATA_CLEAN = DATA_DIR / "openfoodfacts_clean.csv"

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
TARGET = "nutrition-score-fr_100g"

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
IDENTIFIERS = ["code", "product_name"]

# Columns kept in the cleaned dataset
SELECTED_FEATURES = [
    "code",
    "product_name",
    "pnns_groups_1",
    "nova_group",
    "energy_100g",
    "energy-kcal_100g",
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "salt_100g",
    "sodium_100g",
    "saturated-fat_100g",
    TARGET,
    "fiber_100g",
    "calcium_100g",
    "cholesterol_100g",
    "trans-fat_100g",
    "iron_100g",
    "additives_n",
]

# Components that must lie within [0, 100] — used for outlier removal
NUMERIC_COMPONENTS = [
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "salt_100g",
    "sodium_100g",
    "saturated-fat_100g",
    "fiber_100g",
    "calcium_100g",
    "cholesterol_100g",
    "trans-fat_100g",
    "iron_100g",
]

# Columns removed due to 100 % correlation with other features
REDUNDANT_COLS = ["energy-kcal_100g", "sodium_100g"]

# Feature matrix columns fed to ML models
MODEL_FEATURES = [
    "pnns_groups_1",
    "nova_group",
    "energy_100g",
    "proteins_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "salt_100g",
    "saturated-fat_100g",
    "fiber_100g",
    "calcium_100g",
    "cholesterol_100g",
    "trans-fat_100g",
    "iron_100g",
    "additives_n",
]

# ---------------------------------------------------------------------------
# PNNS group normalisation
# ---------------------------------------------------------------------------
PNNS_RENAME_MAP = {
    "fruits-and-vegetables": "Fruits and vegetables",
    "sugary-snacks": "Sugary snacks",
    "cereals-and-potatoes": "Cereals and potatoes",
    "salty-snacks": "Salty snacks",
}

PNNS_CATEGORIES = [
    "Fat and sauces",
    "Composite foods",
    "Sugary snacks",
    "Fruits and vegetables",
    "Fish Meat Eggs",
    "Beverages",
    "Milk and dairy products",
    "Cereals and potatoes",
    "Salty snacks",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

KNN_K_RANGE = range(1, 11)
KNN_BEST_K = 2

PCA_COMPONENTS = 2

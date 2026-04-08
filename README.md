# Nutriscore Predictor

> A machine learning pipeline that predicts the **Nutriscore** of food products from their nutritional composition, built on the open-source [OpenFoodFacts](https://world.openfoodfacts.org/) database.

[![CI](https://github.com/mhaegeman/nutriscore-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/mhaegeman/nutriscore-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

---

## What is Nutriscore?

Nutriscore is a European front-of-pack nutritional label that summarises the overall nutritional quality of a food product on a scale from **A** (healthiest) to **E** (least healthy). It is calculated from a numeric score (−15 to +40) derived from key nutrients:

| Grade | Score range | Meaning |
|-------|-------------|---------|
| **A** | −15 to −1  | Excellent nutritional quality |
| **B** | 0 to 2     | Good nutritional quality |
| **C** | 3 to 10    | Average nutritional quality |
| **D** | 11 to 18   | Poor nutritional quality |
| **E** | 19 to 40   | Very poor nutritional quality |

Positive contributors (lower score): fibre, protein, fruits & vegetables.
Negative contributors (higher score): saturated fat, sugar, sodium, energy.

### Why predict it?

Many food products in the OpenFoodFacts database lack a Nutriscore entry — either because manufacturers haven't submitted it or the score was introduced after the product was catalogued. This project builds a regression model that **estimates the missing Nutriscore from the nutritional composition that is already known**, enabling:

- Consumers to get a quick health estimate on unlabelled products.
- Nutrition researchers to fill gaps in large food datasets.
- Food-tech applications to integrate automated health scoring.

---

## Data Pipeline

The project processes raw OpenFoodFacts export data through a reproducible, modular pipeline:

```
Raw OpenFoodFacts TSV
        │
        ▼
┌─────────────────────┐
│  1. Data Cleaning   │  Remove duplicates, filter columns,
│  preprocessing.py   │  normalise labels, drop outliers
└────────┬────────────┘
         │  openfoodfacts_clean.csv
         ▼
┌─────────────────────┐
│  2. EDA & Visuals   │  Correlation matrix, PNNS distributions,
│  plots.py           │  Nutriscore by food category
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Feature Eng.    │  PNNS encoding, StandardScaler,
│  engineering.py     │  PCA (2 components)
└────────┬────────────┘
         │  X_scaled, PCA projections
         ▼
┌─────────────────────┐
│  4. Model Training  │  KNN Regressor (k-sweep 1–10)
│  train.py           │  Linear Regression (baseline)
└────────┬────────────┘
         │  ModelResult (MSE, MAE, R²)
         ▼
    Trained Model + Evaluation Metrics
```

### Features used (15 inputs)

| Feature | Description |
|---------|-------------|
| `pnns_groups_1` | Food category (9 PNNS groups, encoded) |
| `nova_group` | Processing level (1 = unprocessed → 4 = ultra-processed) |
| `energy_100g` | Energy content (kJ per 100 g) |
| `proteins_100g` | Protein content (g per 100 g) |
| `fat_100g` | Total fat (g per 100 g) |
| `carbohydrates_100g` | Total carbohydrates (g per 100 g) |
| `sugars_100g` | Sugars (g per 100 g) |
| `salt_100g` | Salt (g per 100 g) |
| `saturated-fat_100g` | Saturated fat (g per 100 g) |
| `fiber_100g` | Dietary fibre (g per 100 g) |
| `calcium_100g` | Calcium (g per 100 g) |
| `cholesterol_100g` | Cholesterol (g per 100 g) |
| `trans-fat_100g` | Trans fat (g per 100 g) |
| `iron_100g` | Iron (g per 100 g) |
| `additives_n` | Number of food additives |

---

## Model Results

Two regression models were evaluated on an 80/20 train-test split:

| Model | MAE | R² |
|-------|-----|----|
| **KNN Regressor (k=2)** | ~1.37 | ~0.92 |
| Linear Regression | higher | lower |

The **KNN Regressor with k=2** outperforms the linear baseline significantly, achieving a coefficient of determination of **0.92** — meaning 92 % of the variance in Nutriscore is explained by the nutritional features.

Key findings from the exploratory analysis:
- `fat_100g`, `carbohydrates_100g`, `sugars_100g`, and `additives_n` are the strongest correlates of Nutriscore.
- `energy-kcal_100g` and `sodium_100g` are perfectly correlated with `energy_100g` and `salt_100g` respectively and were removed.
- The **Sugary Snacks** PNNS category has the highest average Nutriscore (worst health profile), with a high proportion of Nova Group 4 products.
- **PCA** reveals that nutritional features cluster products along a healthiness axis, validating the use of distance-based methods like KNN.

---

## Project Structure

```
nutriscore-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI: lint, test, build (Python 3.10/3.11/3.12)
├── data/
│   └── openfoodfacts_data_new.csv   # Place raw data here (not tracked by git)
├── figures/                    # Auto-generated plots (not tracked by git)
├── notebooks/
│   ├── 01_data_cleaning.ipynb  # Interactive data cleaning walkthrough
│   └── 02_eda_and_modeling.ipynb   # EDA, PCA, and model comparison
├── scripts/
│   ├── run_pipeline.py         # End-to-end pipeline (clean → train → evaluate)
│   └── predict.py              # Predict Nutriscore for a single product
├── src/
│   └── nutriscore/
│       ├── config.py           # Central configuration (paths, features, hyperparams)
│       ├── data/
│       │   └── preprocessing.py    # Cleaning pipeline
│       ├── features/
│       │   └── engineering.py      # Encoding, scaling, PCA
│       ├── models/
│       │   └── train.py            # KNN and Linear Regression training + evaluation
│       └── visualization/
│           └── plots.py            # All plotting helpers
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic dataset)
│   ├── test_preprocessing.py   # Unit tests for data cleaning
│   └── test_models.py          # Unit tests for feature engineering and models
├── pyproject.toml              # Package metadata and tool config
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Development dependencies
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/mhaegeman/nutriscore-prediction.git
cd nutriscore-prediction

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Download the data

Download the OpenFoodFacts dataset and place it in the `data/` folder:

```bash
# Direct download from OpenFoodFacts (large file, ~2 GB)
curl -L "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz" \
     -o data/openfoodfacts_data_new.csv.gz && gunzip data/openfoodfacts_data_new.csv.gz
```

Or place your pre-filtered TSV export at `data/openfoodfacts_data_new.csv`.

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Clean the raw data → `data/openfoodfacts_clean.csv`
2. Generate all exploratory figures → `figures/`
3. Run PCA and model training
4. Print evaluation metrics for each model

### 4. Predict a single product

```bash
python scripts/predict.py \
    --energy 1800 \
    --proteins 6 \
    --fat 22 \
    --carbohydrates 55 \
    --sugars 28 \
    --salt 0.8 \
    --saturated-fat 10 \
    --fiber 1.5 \
    --additives 4 \
    --pnns 2 \
    --nova 4
```

Output:
```
  Predicted Nutriscore : +18.0
  Letter Grade        : D

  [A] [B] [C] [D] [E]
               ↑
```

### 5. Run tests

```bash
pytest tests/ -v
```

Tests run entirely offline using a synthetic dataset — no data download required.

---

## Generated Visualisations

After running the pipeline, the `figures/` directory contains:

| File | Description |
|------|-------------|
| `pnns_distribution.png` | Pie chart of product count by food category |
| `correlation_matrix.png` | Heatmap of pairwise correlations between all features |
| `nutriscore_by_pnns.png` | Box-plot of Nutriscore per PNNS food category |
| `nutriscore_by_nova.png` | Box-plot of Nutriscore per Nova processing group |
| `features_vs_nutriscore.png` | Line plots of fat, carbs, and sugar against Nutriscore |
| `pca_variance.png` | Bar chart of explained variance per principal component |
| `pca_scatter.png` | PCA scatter coloured by Nutriscore |
| `pca_circle.png` | PCA correlation circle for feature interpretation |
| `knn_tuning.png` | MAE and R² as a function of k (1–10) |
| `regression_line.png` | Actual vs. predicted Nutriscore for the best model |

---

## Development

### Code style

The project uses **ruff** for linting and formatting:

```bash
ruff check src/ tests/ scripts/    # Lint
ruff format src/ tests/ scripts/   # Auto-format
```

### CI pipeline

Every push triggers three GitHub Actions jobs:
1. **Code Quality** – ruff lint + format check
2. **Tests** – pytest on Python 3.10, 3.11, and 3.12
3. **Build** – verifies the package builds and installs from source

### Notebooks

The `notebooks/` folder contains the original interactive exploratory work:

- `01_data_cleaning.ipynb` – step-by-step data cleaning with commentary
- `02_eda_and_modeling.ipynb` – EDA, PCA visualisations, and model comparison

These notebooks serve as a narrative companion to the production code in `src/`.

---

## Extending the Project

Some directions to improve the pipeline further:

- **Better models**: try Random Forest, Gradient Boosting (XGBoost/LightGBM), or a neural network.
- **More features**: ingredient lists, additives taxonomy, or allergens.
- **Cross-validation**: replace single train-test split with k-fold CV for more reliable estimates.
- **Model persistence**: serialise the trained model with `joblib` for deployment.
- **REST API**: wrap `predict.py` with FastAPI for integration into food-tech applications.

---

## Data Source

This project uses data from [OpenFoodFacts](https://world.openfoodfacts.org/), a free, open, collaborative database of food products from around the world.

> OpenFoodFacts is licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1.0/).

---

## License

MIT — see [LICENSE](LICENSE) for details.

# Data-Driven Investment Model

> ML-based market return prediction comparing Random Forest, KNN, and Linear Regression in Python.

## Overview

This project trains and compares three machine learning models to forecast market returns from a tabular financial dataset. It is intended as an educational reference implementation: the pipeline covers data loading, feature engineering, model training, cross-validation, and side-by-side evaluation. Real performance numbers depend on the dataset you provide; this repository ships with a small synthetic sample so the pipeline can be run end-to-end without external data.

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core modeling language |
| Scikit-learn | ML model training and evaluation |
| Pandas / NumPy | Data wrangling and feature engineering |
| Matplotlib / Seaborn | Visualization |
| Jupyter | Exploratory notebooks |

Models compared: Linear Regression (baseline), K-Nearest Neighbors, and Random Forest Regressor.

## Repository Structure

```
data-driven-investment-model/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── README.md
│   └── sample_data.csv          # Small synthetic example dataset
├── docs/
│   └── data_schema.md           # Expected columns and types
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_comparison.ipynb
└── src/
    ├── __init__.py
    ├── data.py                  # Data loading helpers
    ├── features.py              # Feature engineering
    ├── models.py                # Model factory + training entrypoint
    └── evaluate.py              # Cross-validation and metrics
```

## Environment Setup

### Prerequisites
- Python 3.9+
- pip

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/JNKAnalyst/data-driven-investment-model.git
   cd data-driven-investment-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Provide data**
   - The repo ships with `data/sample_data.csv`, a small synthetic dataset that lets the pipeline run out of the box.
   - To use your own data, drop a CSV in `data/` matching the schema in `docs/data_schema.md` and update `data.data_path` in `config/config.yaml`.

## Configuration

See `config/config.yaml`. Key fields:

- `data.data_path` — path to the input CSV
- `data.target_column` — column to predict (default `returns`)
- `data.test_size` / `data.random_state` — train/test split
- `models.*` — hyperparameters per model
- `evaluation.cv_folds` / `evaluation.scoring` — cross-validation settings

## Running the Pipeline

```bash
# Train all three models and print cross-validated metrics
python -m src.models --config config/config.yaml

# Or work through the notebooks
jupyter notebook notebooks/
```

The training script loads the configured CSV, applies the feature engineering steps in `src/features.py`, fits each model, runs k-fold cross-validation, and prints a comparison table of RMSE and R².

## Results

This repository does not publish benchmark numbers, because results are entirely a function of the dataset you supply. Run the pipeline on your own data to obtain RMSE / R² for each model. The output of `src/models.py` is the canonical comparison.

## Author

**Joash** | MS Business Analytics
[GitHub](https://github.com/JNKAnalyst) | [Portfolio](https://jnkanalyst.github.io/portfolio/)

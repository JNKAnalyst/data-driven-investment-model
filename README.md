# Data-Driven Investment Model

> ML-based market return prediction using Random Forest, KNN, and Linear Regression in Python

## Overview

This project builds and compares three ML models with feature engineering across multi-variable financial datasets to optimize forecast accuracy. Random Forest outperformed baseline assumptions after cross-validation with **8.7% lower RMSE**, identifying key performance drivers across asset classes.

**Key Result:** Random Forest achieved the lowest prediction error after 5-fold cross-validation — **8.7% lower RMSE** vs. baseline.

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python | Core modeling language |
| Scikit-learn | ML model training and evaluation |
| Pandas / NumPy | Data wrangling and feature engineering |
| Matplotlib / Seaborn | Visualization |
| Random Forest | Best-performing model |
| KNN / Linear Regression | Comparison models |

## Repository Structure

```
data-driven-investment-model/
├── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   ├── models.py
│   ├── features.py
│   └── evaluate.py
├── data/
│   └── sample_data.csv          # Sample dataset (replace with full data)
├── config/
│   └── config.yaml
├── requirements.txt
└── .env.example
```

## Environment Setup

### Prerequisites
- Python 3.9+
- pip or conda

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/JNKAnalyst/data-driven-investment-model.git
   cd data-driven-investment-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your data**
   - Place your financial dataset CSV in the `data/` folder
   - Update the `data_path` in `config/config.yaml`

## Configuration

### `config/config.yaml`
```yaml
data:
  data_path: "data/financial_data.csv"
  target_column: "returns"
  test_size: 0.2
  random_state: 42

models:
  random_forest:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 5
  knn:
    n_neighbors: 7
  linear_regression:
    fit_intercept: true

evaluation:
  cv_folds: 5
  scoring: "neg_root_mean_squared_error"
```

## Running the Models

```bash
# Run full model pipeline
python src/models.py --config config/config.yaml

# Run individual notebooks
jupyter notebook notebooks/03_model_training.ipynb
```

## Requirements

```
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
matplotlib==3.8.0
seaborn==0.13.0
jupyter==1.0.0
pyyaml==6.0
python-dotenv==1.0.0
```

## Model Results Summary

| Model | RMSE | R² | Notes |
|-------|------|-----|-------|
| Random Forest | Lowest | Best | After cross-validation |
| KNN | Mid | Mid | Sensitive to feature scaling |
| Linear Regression | Highest | Baseline | Baseline comparison |

## Author

**Joash** | MS Business Analytics  
[GitHub](https://github.com/JNKAnalyst) | [Portfolio](https://jnkanalyst.github.io/portfolio/)

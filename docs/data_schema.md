# Data Schema

The pipeline expects a tabular CSV with one row per time period. The default target column is `returns`; any additional numeric columns become features after preprocessing.

## Columns in `data/sample_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `period` | int | Sequential period index (1..N). Used for ordering only. |
| `sp500_return` | float | Period return of an S&P 500 proxy. |
| `bond_yield_10y` | float | 10-year sovereign bond yield, in percent. |
| `vix` | float | Equity volatility index level. |
| `oil_price` | float | Crude oil spot price, USD per barrel. |
| `fx_dxy` | float | US dollar index level. |
| `returns` | float | **Target.** Period return to predict. |

## Required for any custom dataset

- One column matches `data.target_column` in `config/config.yaml` (default: `returns`).
- All other columns the model should learn from are numeric.
- Missing values are forward/backward filled, then dropped — keep gaps small.
- Date or string columns are dropped automatically by `src/features.py`. If you want to preserve a timestamp for ordering, sort the rows beforehand and drop the column.

## Optional rolling features

`src/features.add_rolling_features` can append rolling mean and rolling std for any column you specify. To enable, call it explicitly with your column list and window sizes — it is off by default to keep the pipeline dataset-agnostic.

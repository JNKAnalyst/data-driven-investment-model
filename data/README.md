# Data

This directory holds the input dataset.

## Sample data

`sample_data.csv` is a small **synthetic** dataset (120 rows). It is not real market data — it exists so the pipeline can be run end-to-end without external dependencies. Do not interpret any results from it as predictive of real markets.

## Bringing your own data

To use a real dataset, place a CSV in this directory matching the schema in `../docs/data_schema.md` and update `data.data_path` in `../config/config.yaml`.

CSV files larger than the sample are ignored by `.gitignore`; commit only files you intend to share.

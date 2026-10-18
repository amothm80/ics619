# Minimal Saudi Stock Forecasting Pipeline

This is a deliberately small, deployment-oriented baseline for ICS619. It downloads or accepts daily Tadawul OHLCV data, adds causal technical features, trains a GRU from scratch, saves artifacts, and produces a five-session closing-price forecast.

## Quick start

```powershell
cd work/saudi_forecasting
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python cli.py download --ticker 1120.SR --start 2021-01-01 --output data/alrajhi.csv
python cli.py train --input data/alrajhi.csv --output-dir artifacts --epochs 30
python cli.py predict --input data/alrajhi.csv --artifact-dir artifacts
```

Use a CSV instead of `download` when an official/licensed Tadawul source is available. Required columns are `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

## Method

- Uses actual trading sessions only; it does not fabricate weekend or holiday rows.
- Computes a compact causal feature set: OHLCV, one-day return, 5-day SMA/EMA, five-day volatility, high-low spread, open-close change, and volume change.
- Uses a chronological 70%/15%/15% train/validation/test split. Both scalers fit only on the training portion.
- Constructs direct multivariate 5-day input to 5-day close-forecast windows. A validation/test window may use earlier historical inputs, but all of its target days remain in that partition.
- Trains a 32-unit GRU with Adam/MSE, early stopping, no shuffling, and a fixed TensorFlow seed. It is a fast baseline, not a final claimed optimum.

`artifacts/` contains the Keras model, both scalers, metadata (configuration, features, metrics), and flattened test predictions. Extend `build_gru` only after recording this baseline; BiRNN, LSTM, averaging, and out-of-fold stacking belong in subsequent experiments.

## Reference relationship

The pipeline preserves the Baamer & Aljamaan methodological starting point—multivariate technical inputs, chronological preparation, and 5-day direct forecasting—while deliberately reducing the initial scope to one reproducible model. It does not reuse their older data or any pretrained model.

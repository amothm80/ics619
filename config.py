"""Project-wide settings. Every other module imports from here, never hard-codes these."""
import os
import random
from pathlib import Path

import numpy as np

# Trading name -> Tadawul symbol (one company per sector, as in the paper).
TICKERS = {
    "ADC": "2140",
    "ALRAJHI": "1120",
    "EMAAR_EC": "4220",
    "FITAIHI": "4180",
    "GACO": "6020",
    "SARCO": "2030",
    "SEC": "5110",
    "SPIMACO": "2070",
    "STC": "7010",
    "TASNEE": "2060",
}

# Chronological split (paper's original period). Bounds are inclusive.
TRAIN_END = "2015-12-31"
VAL_START, VAL_END = "2016-01-01", "2016-12-31"
TEST_START, TEST_END = "2017-01-01", "2020-03-05"

W = 5  # input window: past trading days
H = 5  # forecast horizon: next closes, predicted in one pass
SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "Tadawul_stcks.csv"  # Kaggle: salwaalzahrani/saudi-stock-exchange-tadawul
ARTIFACTS_DIR = ROOT / "artifacts"


def ticker_dir(ticker: str) -> Path:
    """artifacts/<ticker>/ holds that ticker's scalers, feature order, models and config."""
    if ticker not in TICKERS:
        raise KeyError(f"Unknown ticker: {ticker}")
    path = ARTIFACTS_DIR / ticker
    (path / "models").mkdir(parents=True, exist_ok=True)
    return path


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source: Python, NumPy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf  # imported here so importing config stays fast

    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

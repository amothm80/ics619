"""Project-wide settings. Every other module imports from here, never hard-codes these."""
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # logs stream in real time, even through tee or a file

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

# Data source and its chronological split, chosen with the DATA_SOURCE environment variable.
# Bounds are inclusive. "data" names the loader in data.py.
# Before the 2013-06-29 weekend change (Thu-Fri to Fri-Sat), Yahoo has no Saturday sessions
# and dates some Wednesday sessions as Thursday, so the Yahoo source starts after it.
YESTERDAY = str(date.today() - timedelta(days=1))  # today's Yahoo row may be an unfinished session
SOURCES = {
    "kaggle": {"data": "kaggle", "start": "2001-01-01", "train_end": "2015-12-31",
               "val": ("2016-01-01", "2016-12-31"), "test": ("2017-01-01", "2020-03-05")},
    "yfinance_recent": {"data": "yfinance", "start": "2013-06-29", "train_end": "2022-12-31",
                        "val": ("2023-01-01", "2023-12-31"), "test": ("2024-01-01", YESTERDAY)},
}
SOURCE = os.environ.get("DATA_SOURCE", "kaggle")
if SOURCE not in SOURCES:
    raise SystemExit(f"DATA_SOURCE must be one of {list(SOURCES)}, got {SOURCE!r}")
DATA = SOURCES[SOURCE]["data"]
START = SOURCES[SOURCE]["start"]
TRAIN_END = SOURCES[SOURCE]["train_end"]
VAL_START, VAL_END = SOURCES[SOURCE]["val"]
TEST_START, TEST_END = SOURCES[SOURCE]["test"]

W = 5  # input window: past trading days
H = 5  # forecast horizon: next closes, predicted in one pass
SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "Tadawul_stcks.csv"  # Kaggle: salwaalzahrani/saudi-stock-exchange-tadawul
YF_DIR = DATA_DIR / "yfinance"  # one cached download per ticker, shared by both yfinance splits
ARTIFACTS_DIR = ROOT / "artifacts" / SOURCE  # everything fitted or scored on this source


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

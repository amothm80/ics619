#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tasi_edl_pipeline.py
====================

Self-contained reproduction of

    Baamer R., Aljamaan H. (2026). "Ensemble deep learning models for stock price
    forecasting: evidence from the Saudi markets". PeerJ Comput. Sci. 12:e4064.
    Code:  https://github.com/ranabameer/saudi-stock-forecasting-ensembles
    Data:  https://www.kaggle.com/datasets/salwaalzahrani/saudi-stock-exchange-tadawul
           https://www.kaggle.com/datasets/paultimothymooney/stock-market-data  (US replication)

Pipeline (one file, end to end)
-------------------------------
 1. Acquire data from Kaggle (kagglehub) or from local CSVs.
 2. Per company: reindex to a *calendar-daily* grid, forward-fill, compute the
    34 technical indicators of Table 4 with pandas-ta-classic (window length 5).
    (Optional: re-run the Pearson |r| > 0.8 feature selection over ~130 TIs.)
 3. Chronological split  train <= 2015-12-31 | val = 2016 | test >= 2017-01-01,
    StandardScaler fitted on train only, sliding windows w = 5 -> h = 5 (direct
    multi-output).
 4. Base models: BiRNN (B), GRU (G), LSTM (L) – single recurrent layer + Dense(5),
    ReLU, Adam, MSE. Units per company are the ones hard-coded in the repo's final
    notebook (Baseline&WX.ipynb), or re-searched over the repo's unit grid.
 5. Ensembles: 4 averaging ensembles + 12 stacking ensembles
    (meta-learner ANN 128-64-32 / GRU 128 / LSTM 128) – Table 7.
 6. Baselines: Random Walk with drift, ARIMA(5,1,0), XGBoost, GRU(256) benchmark
    of Aseeri (2023).
 7. Metrics on de-scaled, 2-decimal-rounded prices: MAE, MSE, RMSE, MRE, R2
    (Tables 10 & 13), timing (Table 12), Wilcoxon signed-rank + Holm win/loss
    ranking (Table 11).

Paper vs. repository
--------------------
The paper text and the repository code disagree in several places. Because the
goal is reproducing the *reported numbers*, defaults follow the repository code
(which produced them). Every discrepancy is exposed as a switch:

  item                      repo (default)                     paper text
  ------------------------  ---------------------------------  -------------------------------
  base-model training       32 epochs, batch=5% of train,      100 epochs, batch=10%, lr=1e-4,
                            Adam lr=1e-3, no early stopping    early stopping 20, best val loss
                                                               -> --train-config paper
  stacking meta-learner     fit on VALIDATION-set base          rolling-origin out-of-fold
  training data             predictions (Baseline&WX.ipynb)     predictions on train
                                                               -> --stacking oof
                            (older notebook: fit on TRAIN-set   -> --stacking train
                             predictions)
  base-model unit choice    fixed table from repo (chosen on    chosen on validation
                            test RMSE in the original grid)     -> --units grid
  ARIMA                     one 5-step forecast from end of     (not specified)
                            train, repeated for every window    -> --fix-quirks gives rolling
  GRU(256) benchmark        no forward-fill (non-trading days   -> --fix-quirks
                            become 0), MinMax fitted per split
  Wilcoxon win/loss         every model-column x every          per-horizon comparison
                            model-column against each horizon   -> --wx-mode horizon

Note also that, exactly as in the repo, the series is reindexed to *calendar*
days (weekends/holidays forward-filled), so "5 days ahead" means 5 calendar rows.

Recommended environment (what the authors used on Colab)
--------------------------------------------------------
  python 3.10/3.11, tensorflow==2.15.*, numpy<2 (1.26), pandas 2.x,
  pandas-ta-classic, scikit-learn, scipy, statsmodels, xgboost, kagglehub

  pip install "tensorflow==2.15.*" "numpy<2" "pandas<3" pandas-ta-classic \
              scikit-learn scipy statsmodels xgboost kagglehub

Indicators come from pandas-ta-classic (import name `pandas_ta_classic`), the
maintained continuation of the pandas-ta 0.3.x line the authors used. The
original pandas-ta 0.3.14b0 is no longer available on PyPI. The script also runs
on TF 2.16+/Keras 3 and numpy 2.

Usage
-----
  python tasi_edl_pipeline.py                              # Saudi, repo-faithful
  python tasi_edl_pipeline.py --market both                # + US replication
  python tasi_edl_pipeline.py --companies ADC STC          # subset
  python tasi_edl_pipeline.py --train-config paper --stacking oof   # paper text
  python tasi_edl_pipeline.py --quick                      # tiny smoke run
  python tasi_edl_pipeline.py --saudi-csv path/Tadawul_stcks.csv --us-dir path/csvs

Kaggle: public datasets usually download anonymously; if kagglehub asks for
credentials, set KAGGLE_USERNAME / KAGGLE_KEY or place ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LOG = logging.getLogger("edl")

# =============================================================================
# 0. CONSTANTS (taken from the paper and the repository)
# =============================================================================

TRAIN_END, VAL_START, VAL_END, TEST_START = "2015-12-31", "2016-01-01", "2016-12-31", "2017-01-01"
DATA_END = "2020-03-05"
N_IN, N_OUT = 5, 5

# name -> (reindex start date, (BiRNN units, GRU units, LSTM units))   [Baseline&WX.ipynb]
SAUDI_COMPANIES: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    "ADC": ("2001-12-31", (200, 256, 500)),
    "ALRAJHI": ("2001-12-31", (900, 800, 400)),
    "GACO": ("2002-01-01", (900, 700, 1000)),
    "EMAAR EC": ("2006-10-06", (1024, 600, 1000)),
    "FITAIHI GROUP": ("2002-01-26", (200, 512, 900)),
    "SARCO": ("2002-01-23", (500, 800, 1024)),
    "SAUDI ELECTRICITY": ("2002-06-08", (300, 243, 1024)),
    "SPIMACO": ("2001-12-31", (900, 256, 600)),
    "STC": ("2003-01-25", (1024, 729, 1024)),
    "TASNEE": ("2001-12-31", (1024, 512, 1024)),
}
US_COMPANIES = ["AAPL", "BAC", "GOOG", "JNJ"]
US_START = "2001-12-31"

# unit grid of the standalone notebooks
UNIT_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3, 9, 27, 81, 243, 729,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Table 4 (Saudi, 34 TIs) and the US list from PrepareFeatursOfData.ipynb (33)
SELECTED_TI_SA = ["open", "high", "low", "close", "alma", "decay", "dema", "ema", "fwma", "hma",
                  "hl2", "hlc3", "hwma", "jma", "kama", "linreg", "median", "midpoint", "midprice",
                  "pwma", "quantile", "rma", "sinwma", "sma", "ssf", "swma", "t3", "tema", "trima",
                  "vwap", "vmwa", "wcp", "wma", "zlma"]
SELECTED_TI_US = ["Adjusted Close", "Close", "High", "Low", "Open", "alma", "decay", "dema", "ema",
                  "fwma", "hl2", "hlc3", "hma", "hwma", "linreg", "median", "midpoint", "midprice",
                  "pvi", "pwma", "quantile", "rma", "sinwma", "sma", "swma", "t3", "tema", "trima",
                  "vmwa", "vwap", "wcp", "wma", "zlma"]

COMBOS = {"B,G,L": [0, 1, 2], "B,G": [0, 1], "B,L": [0, 2], "G,L": [1, 2]}
BASE_NAMES = ["BRNN", "GRU", "LSTM"]
META_TYPES = ["ANN", "GRU", "LSTM"]

# Paper Table 10 (best model by RMSE per company) and Table 13 (US)
PAPER_BEST = {
    "ADC": ("STK_GRU(B,G)", 0.2498), "ALRAJHI": ("STK_GRU(B,G)", 1.0688),
    "EMAAR EC": ("AVG(B,G)", 0.3259), "FITAIHI GROUP": ("AVG(G,L)", 0.1908),
    "GACO": ("STK_LSTM(B,L)", 0.2637), "SARCO": ("AVG(B,G)", 1.2307),
    "SAUDI ELECTRICITY": ("STK_GRU(B,G)", 0.4651), "SPIMACO": ("AVG(B,G)", 0.5192),
    "STC": ("STK_LSTM(B,G)", 1.9293), "TASNEE": ("STK_LSTM(B,G)", 0.4401),
    "AAPL": ("STK_ANN(B,G,L)", 1.2127), "BAC": ("GRU", 0.6954),
    "GOOG": ("STK_ANN(B,L)", 1.1786), "JNJ": ("AVG(B,G)", 2.1984),
}
# models that appear in Table 11 (Wilcoxon ranking)
PAPER_WX_MODELS = ["AVG(B,G)", "AVG(B,G,L)", "STK_GRU(B,G)", "STK_GRU(B,G,L)", "AVG(B,L)", "GRU",
                   "STK_LSTM(B,G,L)", "AVG(G,L)", "STK_GRU(B,L)", "STK_LSTM(B,L)", "BRNN", "LSTM",
                   "XGBoost", "Random Walk", "Benchmark", "ARIMA"]
PAPER_TABLE11 = {"AVG(B,G)": 88.64, "AVG(B,G,L)": 88.36, "STK_GRU(B,G)": 87.66, "STK_GRU(B,G,L)": 86.82,
                 "AVG(B,L)": 85.69, "GRU": 82.62, "STK_LSTM(B,G,L)": 82.38, "AVG(G,L)": 81.44,
                 "STK_GRU(B,L)": 81.29, "STK_LSTM(B,L)": 80.41, "BRNN": 77.18, "LSTM": 62.74,
                 "XGBoost": 66.87, "Random Walk": 30.75, "Benchmark": 11.29, "ARIMA": 1.52}


# =============================================================================
# 1. RUN CONFIGURATION
# =============================================================================

@dataclass
class TrainCfg:
    epochs: int = 32
    batch_frac: float = 0.05
    lr: float = 1e-3
    patience: Optional[int] = None          # None -> no early stopping (repo)
    meta_epochs: int = 100
    meta_lr: float = 1e-3
    meta_patience: Optional[int] = None
    bench_epochs: int = 20


def make_train_cfg(name: str, quick: bool) -> TrainCfg:
    if name == "paper":   # Table 8
        cfg = TrainCfg(epochs=100, batch_frac=0.10, lr=1e-4, patience=20,
                       meta_epochs=100, meta_lr=1e-4, meta_patience=20)
    else:                 # repository notebooks
        cfg = TrainCfg()
    if quick:
        cfg.epochs, cfg.meta_epochs, cfg.bench_epochs = 2, 2, 2
        if cfg.patience:
            cfg.patience = cfg.meta_patience = 1
    return cfg


@dataclass
class RunCfg:
    data_dir: str
    out_dir: str
    train: TrainCfg
    stacking: str = "val"          # val | train | oof
    oof_folds: int = 5
    units: str = "repo"            # repo | grid | <int>
    unit_grid: List[int] = field(default_factory=lambda: list(UNIT_GRID))
    fix_quirks: bool = False
    feature_selection: str = "fixed"   # fixed | pearson
    pearson_threshold: float = 0.8
    skip_baselines: bool = False
    seed: int = 42
    resume: bool = False
    save_models: bool = False


# =============================================================================
# 2. UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "run.log"))):
        h.setFormatter(fmt)
        LOG.addHandler(h)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# =============================================================================
# 3. DATA ACQUISITION
# =============================================================================

def _kagglehub():
    try:
        import kagglehub  # noqa
        return kagglehub
    except ImportError:
        raise SystemExit("kagglehub is not installed: pip install kagglehub  "
                         "(or pass --saudi-csv / --us-dir with local files)")


def acquire_saudi(data_dir: str, override: Optional[str]) -> str:
    """Return path of Tadawul_stcks.csv (download from Kaggle if needed)."""
    if override:
        return override
    local = glob.glob(os.path.join(data_dir, "saudi", "**", "*.csv"), recursive=True)
    if not local:
        kh = _kagglehub()
        LOG.info("Downloading salwaalzahrani/saudi-stock-exchange-tadawul from Kaggle ...")
        src = kh.dataset_download("salwaalzahrani/saudi-stock-exchange-tadawul")
        dst = os.path.join(data_dir, "saudi")
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True):
            shutil.copy(f, dst)
        local = glob.glob(os.path.join(dst, "*.csv"))
    pref = [f for f in local if "tadawul" in os.path.basename(f).lower()]
    path = (pref or sorted(local, key=os.path.getsize, reverse=True))[0]
    LOG.info(f"Saudi data: {path}")
    return path


def acquire_us(data_dir: str, tickers: List[str], override_dir: Optional[str]) -> Dict[str, str]:
    """Return {ticker: csv path}. Downloads single files from the (large) Kaggle dataset."""
    out: Dict[str, str] = {}
    search_dirs = [override_dir] if override_dir else [os.path.join(data_dir, "us")]
    for t in tickers:
        for d in search_dirs:
            hits = glob.glob(os.path.join(d, "**", f"{t}.csv"), recursive=True)
            if hits:
                out[t] = hits[0]
                break
    missing = [t for t in tickers if t not in out]
    if missing and override_dir is None:
        kh = _kagglehub()
        dst = os.path.join(data_dir, "us")
        os.makedirs(dst, exist_ok=True)
        for t in missing:
            for sub in ("sp500", "nasdaq", "nyse", "forbes2000"):
                try:
                    LOG.info(f"Downloading {t} ({sub}) from paultimothymooney/stock-market-data ...")
                    p = kh.dataset_download("paultimothymooney/stock-market-data",
                                            path=f"stock_market_data/{sub}/csv/{t}.csv")
                    shutil.copy(p, os.path.join(dst, f"{t}.csv"))
                    out[t] = os.path.join(dst, f"{t}.csv")
                    break
                except Exception as e:  # noqa
                    LOG.info(f"   not in {sub}: {type(e).__name__}")
    for t in tickers:
        if t not in out:
            raise SystemExit(f"Could not obtain US data for {t}; pass --us-dir with {t}.csv")
    return out


_SA_RENAME = {"trading_name": "trading_name", "sectoer": "sector", "perc_change": "perc_change",
              "volume_traded": "volume_traded", "value_traded": "value_traded", "no_trades": "no_trades"}


def load_saudi_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]   # 'trading_name ' -> 'trading_name'
    df = df.rename(columns=_SA_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df["trading_name"] = df["trading_name"].astype(str).str.strip()
    return df.set_index("date").sort_index()


def load_us_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)   # dataset uses dd-mm-yyyy
    return df.set_index("Date").sort_index()


# =============================================================================
# 4. FEATURE ENGINEERING  (PrepareFeatursOfData.ipynb)
# =============================================================================

def _numeric_daily(df: pd.DataFrame, start: str, ffill: bool = True) -> pd.DataFrame:
    """Reindex to calendar days, '?' -> NaN, float32, forward fill (repo behaviour)."""
    df = df[~df.index.duplicated(keep="last")]
    df = df.reindex(pd.date_range(start=start, end=DATA_END, freq="D"))
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    return df.ffill() if ffill else df


def _first_col(x):
    return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x


def add_overlap_indicators(df: pd.DataFrame, o: str, h: str, l: str, c: str, v: str,
                           saudi: bool) -> pd.DataFrame:
    import pandas_ta_classic as ta
    C, H, L, V = df[c], df[h], df[l], df[v]
    df["alma"] = ta.alma(C, length=5)
    df["decay"] = ta.decay(C, length=5)
    df["dema"] = ta.dema(C, length=5)
    df["ema"] = ta.ema(C, length=5)
    df["fwma"] = ta.fwma(C, length=5)
    df["hma"] = ta.hma(C, length=5)
    df["hl2"] = ta.hl2(H, L)
    df["hlc3"] = ta.hlc3(H, L, C)
    df["hwma"] = ta.dema(C)                       # repo uses dema(default length) under the name 'hwma'
    if saudi:
        df["jma"] = ta.jma(C, length=5)
        df["kama"] = ta.kama(C, length=5)
    df["linreg"] = ta.linreg(C, length=5)
    df["median"] = ta.median(C, length=5)
    df["midpoint"] = ta.midpoint(C, length=5)
    df["midprice"] = ta.midprice(H, L, length=5)
    if not saudi:
        df["pvi"] = _first_col(ta.pvi(C, V, length=5))
    df["pwma"] = ta.pwma(C, length=5)
    df["quantile"] = ta.quantile(C, length=5)
    df["rma"] = ta.rma(C, length=5)
    df["sinwma"] = ta.sinwma(C, length=5)
    df["sma"] = ta.sma(C, length=5)
    if saudi:
        df["ssf"] = ta.ssf(C, length=5)
    df["swma"] = ta.swma(C, length=5)
    df["t3"] = ta.t3(C, length=5)
    df["tema"] = ta.tema(C, length=5)
    df["trima"] = ta.trima(C, length=5)
    df["vwap"] = ta.vwap(high=H, low=L, close=C, volume=V)
    df["vmwa"] = ta.vwma(volume=V, close=C, length=5)
    df["wcp"] = ta.wcp(H, L, C)
    df["wma"] = ta.wma(C, length=5)
    df["zlma"] = ta.zlma(C, length=5)
    return df


def engineer_saudi(raw: pd.DataFrame, company: str, start: str,
                   selected: List[str] = SELECTED_TI_SA) -> pd.DataFrame:
    co = raw[raw["trading_name"] == company.strip()]
    if co.empty:
        raise ValueError(f"company '{company}' not found in Saudi dataset")
    co = co.drop(columns=[c for c in ("symbol", "name", "trading_name", "sector") if c in co.columns])
    co = _numeric_daily(co, start)
    co = add_overlap_indicators(co, "open", "high", "low", "close", "volume_traded", saudi=True)
    co = co.fillna(0)
    return co[selected].replace([np.inf, -np.inf], 0)


def engineer_us(raw: pd.DataFrame, selected: List[str] = SELECTED_TI_US) -> pd.DataFrame:
    co = raw[[c for c in ["Low", "Open", "Volume", "High", "Close", "Adjusted Close"] if c in raw.columns]]
    co = _numeric_daily(co, US_START)
    co = add_overlap_indicators(co, "Open", "High", "Low", "Close", "Volume", saudi=False)
    co = co.fillna(0)
    # US notebook: df.replace([inf,-inf],0).fillna(0), selected_ti = all columns
    return co[selected].replace([np.inf, -np.inf], 0).fillna(0)


# ---- optional: full ~130 TI computation + Pearson selection (Standalone notebook get_company_df)
def all_indicators(df: pd.DataFrame, o: str, h: str, l: str, c: str, v: str) -> pd.DataFrame:
    import pandas_ta_classic as ta
    O, H, L, C, V = df[o], df[h], df[l], df[c], df[v]
    calls = {
        "aberration": lambda: ta.aberration(H, L, C, length=5),
        "accbands": lambda: ta.accbands(H, L, C, length=5),
        "ad": lambda: ta.ad(H, L, C, V), "adosc": lambda: ta.adosc(H, L, C, V, fast=1, slow=5),
        "alma": lambda: ta.alma(C, length=5), "amat": lambda: ta.amat(C, length=5),
        "aobv": lambda: ta.aobv(C, V), "aroon": lambda: ta.aroon(H, L, length=5),
        "atr": lambda: ta.atr(H, L, C, length=5), "bbands": lambda: ta.bbands(C, length=5),
        "bias": lambda: ta.bias(C, length=5), "bop": lambda: ta.bop(O, H, L, C),
        "brar": lambda: ta.brar(O, H, L, C, length=5), "cci": lambda: ta.cci(H, L, C, length=5),
        "cfo": lambda: ta.cfo(C, length=5), "cg": lambda: ta.cg(C, length=5),
        "chop": lambda: ta.chop(H, L, C, length=5), "cksp": lambda: ta.cksp(H, L, C),
        "cmf": lambda: ta.cmf(H, L, C, V, O, length=5), "cmo": lambda: ta.cmo(C, length=5),
        "coppock": lambda: ta.coppock(C, length=5, fast=6, slow=11), "cti": lambda: ta.cti(C, length=5),
        "decay": lambda: ta.decay(C, length=5), "decreasing": lambda: ta.decreasing(C, length=5),
        "dema": lambda: ta.dema(C, length=5), "dm": lambda: ta.dm(H, L, length=5),
        "donchian": lambda: ta.donchian(H, L, lower_length=5, upper_length=5),
        "dpo": lambda: ta.dpo(C, length=5), "ebsw": lambda: ta.ebsw(C, length=5),
        "efi": lambda: ta.efi(C, V, length=5), "ema": lambda: ta.ema(C, length=5),
        "entropy": lambda: ta.entropy(C, length=5), "eom": lambda: ta.eom(H, L, C, V, length=5),
        "er": lambda: ta.er(C, length=5), "eri": lambda: ta.eri(H, L, C, length=5),
        "fisher": lambda: ta.fisher(H, L, length=5), "fwma": lambda: ta.fwma(C, length=5),
        "hilo": lambda: ta.hilo(H, L, C, high_length=5, low_length=5), "hma": lambda: ta.hma(C, length=5),
        "hl2": lambda: ta.hl2(H, L), "hlc3": lambda: ta.hlc3(H, L, C), "hwma": lambda: ta.dema(C),
        "increasing": lambda: ta.increasing(C, length=5),
        "inertia": lambda: ta.inertia(C, H, L, length=5, rvi_length=5),
        "jma": lambda: ta.jma(C, length=5), "kama": lambda: ta.kama(C, length=5),
        "kc": lambda: ta.kc(H, L, C, length=5), "kdj": lambda: ta.kdj(H, L, C, length=5),
        "kurtosis": lambda: ta.kurtosis(C, length=5), "kvo": lambda: ta.kvo(H, L, C, V),
        "linreg": lambda: ta.linreg(C, length=5), "log_return": lambda: ta.log_return(C, length=5),
        "macd": lambda: ta.macd(C), "mad": lambda: ta.mad(C, length=5),
        "massi": lambda: ta.massi(H, L, fast=1, slow=5), "median": lambda: ta.median(C, length=5),
        "mfi": lambda: ta.mfi(H, L, C, V, length=5), "midpoint": lambda: ta.midpoint(C, length=5),
        "midprice": lambda: ta.midprice(H, L, length=5), "mom": lambda: ta.mom(C, length=5),
        "natr": lambda: ta.natr(H, L, C, length=5), "nvi": lambda: ta.nvi(C, V, length=5),
        "obv": lambda: ta.obv(C, V), "pdist": lambda: ta.pdist(O, H, L, C),
        "percent_return": lambda: ta.percent_return(C, length=5), "pgo": lambda: ta.pgo(H, L, C, length=5),
        "psl": lambda: ta.psl(C, O, length=5), "pvi": lambda: ta.pvi(C, V, length=5),
        "pvol": lambda: ta.pvol(C, V), "pvt": lambda: ta.pvt(C, V), "pwma": lambda: ta.pwma(C, length=5),
        "qqe": lambda: ta.qqe(C, length=5), "qstick": lambda: ta.qstick(O, C, length=5),
        "quantile": lambda: ta.quantile(C, length=5), "rma": lambda: ta.rma(C, length=5),
        "roc": lambda: ta.roc(C, length=5), "rsi": lambda: ta.rsi(C, length=5),
        "rsx": lambda: ta.rsx(C, length=5), "sinwma": lambda: ta.sinwma(C, length=5),
        "skew": lambda: ta.skew(C, length=5), "slope": lambda: ta.slope(C, length=5),
        "sma": lambda: ta.sma(C, length=5), "smi": lambda: ta.smi(C), "ssf": lambda: ta.ssf(C, length=5),
        "stc": lambda: ta.stc(C, tclength=5), "stdev": lambda: ta.stdev(C, length=5),
        "stochrsi": lambda: ta.stochrsi(C, length=5), "swma": lambda: ta.swma(C, length=5),
        "t3": lambda: ta.t3(C, length=5), "tema": lambda: ta.tema(C, length=5),
        "trima": lambda: ta.trima(C, length=5), "trix": lambda: ta.trix(C, length=5),
        "true_range": lambda: ta.true_range(H, L, C), "ui": lambda: ta.ui(C, length=5),
        "uo": lambda: ta.uo(H, L, C, fast=1, medium=3, slow=5),
        "variance": lambda: ta.variance(C, length=5), "vhf": lambda: ta.vhf(C, length=5),
        "vidya": lambda: ta.vidya(C, length=5), "vortex": lambda: ta.vortex(H, L, C, length=5),
        "vwap": lambda: ta.vwap(H, L, C, V), "vmwa": lambda: ta.vwma(C, V, length=5),
        "wcp": lambda: ta.wcp(H, L, C), "willr": lambda: ta.willr(H, L, C, length=5),
        "wma": lambda: ta.wma(C, length=5), "zlma": lambda: ta.zlma(C, length=5),
        "zscore": lambda: ta.zscore(C, length=5),
    }
    cols = {}
    for name, fn in calls.items():
        try:
            r = fn()
        except Exception:
            continue
        if r is None:
            continue
        if isinstance(r, pd.DataFrame):
            for k, col in enumerate(r.columns):
                cols[name if r.shape[1] == 1 else f"{name}_{k}"] = r[col]
        else:
            cols[name] = r
    out = pd.DataFrame(cols, index=df.index).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def pearson_feature_selection(frames: Dict[str, pd.DataFrame], close_col: str,
                              threshold: float) -> List[str]:
    """|r| > threshold with close on TRAIN rows of every company; intersection across companies."""
    sets = []
    for name, f in frames.items():
        tr = f.loc[:TRAIN_END]
        r = tr.corrwith(tr[close_col]).abs()
        keep = set(r[(r > threshold)].index) | {close_col}
        LOG.info(f"  {name}: {len(keep)} features with |r|>{threshold}")
        sets.append(keep)
    common = sorted(set.intersection(*sets))
    LOG.info(f"Pearson-selected common features ({len(common)}): {common}")
    return common


# =============================================================================
# 5. SPLITTING / SCALING / WINDOWING
# =============================================================================

def to_supervised(X: pd.DataFrame, y: pd.DataFrame, n_in=N_IN, n_out=N_OUT):
    xv, yv = X.values, y.values
    n = len(xv) - n_in - n_out + 1
    if n <= 0:
        return np.zeros((0, n_in, xv.shape[1]), "float32"), np.zeros((0, n_out, 1), "float32")
    Xs = np.stack([xv[i:i + n_in] for i in range(n)])
    ys = np.stack([yv[i + n_in:i + n_in + n_out] for i in range(n)])
    return Xs.astype("float32"), ys.astype("float32")


def windows_of(series: pd.Series, n_in=N_IN, n_out=N_OUT) -> np.ndarray:
    v = series.values
    return np.array([v[i + n_in:i + n_in + n_out] for i in range(len(v) - n_in - n_out + 1)])


@dataclass
class Split:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_y_eval: np.ndarray          # original-scale actuals (N,5)
    val_y_eval: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    target_scaler: object


def prepare_split(company: pd.DataFrame, features: List[str], target: str) -> Split:
    from sklearn.preprocessing import StandardScaler
    train = company.loc[:TRAIN_END]
    val = company.loc[VAL_START:VAL_END]
    test = company.loc[TEST_START:]
    fs = StandardScaler().fit(train[features])
    ts = StandardScaler().fit(train[[target]])

    def sx(d):
        return pd.DataFrame(fs.transform(d[features]), index=d.index, columns=features)

    def sy(d):
        return pd.DataFrame(ts.transform(d[[target]]), index=d.index, columns=[target])

    trx, tr_y = to_supervised(sx(train), sy(train))
    vx, v_y = to_supervised(sx(val), sy(val))
    tx, t_y = to_supervised(sx(test), sy(test))
    LOG.info(f"  split rows train/val/test = {len(train)}/{len(val)}/{len(test)}; "
             f"windows = {len(trx)}/{len(vx)}/{len(tx)}; features = {len(features)}")
    return Split(trx, tr_y, vx, v_y, tx, t_y, windows_of(test[target]), windows_of(val[target]),
                 train, val, test, ts)


# =============================================================================
# 6. METRICS  (Baseline&WX.ipynb evaluate_forecasts)
# =============================================================================

def inv(ts, pred_scaled: np.ndarray) -> np.ndarray:
    p = np.asarray(pred_scaled).reshape(len(pred_scaled), -1)
    return ts.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)


def metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import r2_score
    a = np.round(np.asarray(actual, dtype="float64"), 2).ravel()
    p = np.round(np.asarray(pred, dtype="float64"), 2).ravel()
    e = a - p
    mse = float(np.mean(e ** 2))
    return {"MAE": float(np.mean(np.abs(e))), "MSE": mse, "RMSE": math.sqrt(mse),
            "MRE": float(np.mean(np.abs(e / (a + 1e-8)))), "R2": float(r2_score(a, p))}


# =============================================================================
# 7. MODELS
# =============================================================================

def _keras():
    import tensorflow as tf
    return tf.keras


class BestWeights:
    """Keras callback factory: keep weights with the lowest val_loss, early stop with patience."""

    @staticmethod
    def make(patience: int):
        keras = _keras()

        class _CB(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best, self.wait, self.weights = np.inf, 0, None

            def on_epoch_end(self, epoch, logs=None):
                v = (logs or {}).get("val_loss")
                if v is None:
                    return
                if v < self.best:
                    self.best, self.wait, self.weights = v, 0, self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= patience:
                        self.model.stop_training = True

            def on_train_end(self, logs=None):
                if self.weights is not None:
                    self.model.set_weights(self.weights)

        return _CB()


def build_base(kind: str, units: int, input_shape, lr: float):
    keras = _keras()
    L = keras.layers
    if kind == "BRNN":
        rec = L.Bidirectional(L.SimpleRNN(units, activation="relu"))
    elif kind == "GRU":
        rec = L.GRU(units, activation="relu")
    elif kind == "LSTM":
        rec = L.LSTM(units, activation="relu")
    else:
        raise ValueError(kind)
    m = keras.Sequential([L.Input(shape=input_shape), rec, L.Dense(N_OUT)])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=lr))
    return m


def build_meta(kind: str, input_shape, lr: float):
    keras = _keras()
    L = keras.layers
    if kind == "ANN":
        layers = [L.Flatten(), L.Dense(128, activation="relu"), L.Dense(64, activation="relu"),
                  L.Dense(32, activation="relu")]
    elif kind == "GRU":
        layers = [L.GRU(128, activation="relu")]
    elif kind == "LSTM":
        layers = [L.LSTM(128, activation="relu")]
    else:
        raise ValueError(kind)
    m = keras.Sequential([L.Input(shape=input_shape)] + layers + [L.Dense(N_OUT)])
    m.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=lr))
    return m


def fit_base(kind: str, units: int, sp: Split, cfg: TrainCfg,
             train_x=None, train_y=None):
    tx = sp.train_x if train_x is None else train_x
    ty = sp.train_y if train_y is None else train_y
    m = build_base(kind, units, tx.shape[1:], cfg.lr)
    cbs = [BestWeights.make(cfg.patience)] if cfg.patience else []
    t0 = time.perf_counter()
    m.fit(tx, ty, epochs=cfg.epochs, batch_size=max(1, math.floor(len(tx) * cfg.batch_frac)),
          validation_data=(sp.val_x, sp.val_y), verbose=0, callbacks=cbs)
    return m, time.perf_counter() - t0


def predict_timed(m, x) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = m.predict(x, verbose=0)
    return np.asarray(p).reshape(len(x), N_OUT), time.perf_counter() - t0


def choose_units(kind: str, sp: Split, cfg: TrainCfg, grid: List[int], cache: dict, key: str):
    """Grid search over units, selecting on VALIDATION RMSE (original scale). Returns best model."""
    best = (np.inf, None, None, None)
    rows = cache.setdefault(key, {})
    for u in grid:
        m, tt = fit_base(kind, u, sp, cfg)
        pv, _ = predict_timed(m, sp.val_x)
        rmse = metrics(sp.val_y_eval, inv(sp.target_scaler, pv))["RMSE"]
        rows[str(u)] = rmse
        LOG.info(f"    grid {kind:4s} units={u:5d}  val RMSE={rmse:.4f}")
        if rmse < best[0]:
            best = (rmse, u, m, tt)
    return best[1], best[2], best[3]


# =============================================================================
# 8. ENSEMBLES
# =============================================================================

def stack_preds(preds: List[np.ndarray]) -> np.ndarray:
    """(N,5) x M  ->  (N,5,M)  (same as np.dstack in the repo)."""
    return np.stack(preds, axis=2)


def oof_base_predictions(sp: Split, units: Tuple[int, int, int], cfg: TrainCfg, folds: int):
    """Rolling-origin (expanding window) out-of-fold predictions on the TRAIN windows.
    A purge gap of n_in+n_out-1 windows avoids target overlap between fold-train and fold-valid."""
    n = len(sp.train_x)
    block = n // (folds + 1)
    gap = N_IN + N_OUT - 1
    ys = []
    preds = {k: [] for k in BASE_NAMES}
    for i in range(1, folds + 1):
        tr_end = i * block - gap
        va_start, va_end = i * block, (i + 1) * block if i < folds else n
        if tr_end < 50:
            continue
        for kind, u in zip(BASE_NAMES, units):
            m, _ = fit_base(kind, u, sp, cfg, sp.train_x[:tr_end], sp.train_y[:tr_end])
            p, _ = predict_timed(m, sp.train_x[va_start:va_end])
            preds[kind].append(p)
            _keras().backend.clear_session()
        ys.append(sp.train_y[va_start:va_end])
        LOG.info(f"    OOF fold {i}/{folds}: train<{tr_end} valid[{va_start}:{va_end})")
    return {k: np.concatenate(v) for k, v in preds.items()}, np.concatenate(ys)


def train_meta(kind: str, Xm: np.ndarray, ym: np.ndarray, cfg: TrainCfg,
               val: Optional[Tuple[np.ndarray, np.ndarray]]):
    m = build_meta(kind, Xm.shape[1:], cfg.meta_lr)
    cbs = [BestWeights.make(cfg.meta_patience)] if (cfg.meta_patience and val is not None) else []
    t0 = time.perf_counter()
    m.fit(Xm, ym, epochs=cfg.meta_epochs, validation_data=val, verbose=0, callbacks=cbs)
    return m, time.perf_counter() - t0


# =============================================================================
# 9. BASELINES
# =============================================================================

def random_walk_with_drift(seq: np.ndarray, h: int = N_OUT) -> np.ndarray:
    drift = np.mean(np.diff(seq))
    return np.array([seq[-1] + drift * (i + 1) for i in range(h)])


def baseline_random_walk(sp: Split, target: str):
    t0 = time.perf_counter()
    v = sp.test_df[target].values
    preds = np.array([random_walk_with_drift(v[i:i + N_IN]) for i in range(len(v) - N_IN - N_OUT + 1)])
    return preds, 0.0, time.perf_counter() - t0


def baseline_arima(sp: Split, target: str, order=(5, 1, 0), rolling: bool = False):
    from statsmodels.tsa.arima.model import ARIMA
    y_tr = sp.train_df[target].astype("float64").values
    t0 = time.perf_counter()
    fit = ARIMA(y_tr, order=order).fit()
    tt = time.perf_counter() - t0
    n = len(sp.test_y_eval)
    t1 = time.perf_counter()
    if not rolling:
        # repo: the same forecast from the end of training for every test window
        f = np.asarray(fit.forecast(steps=N_OUT)).ravel()
        preds = np.tile(f, (n, 1))
    else:
        # fixed: forecast from each window's origin, using the fitted ARIMA(p,1,0) recursion
        p = order[0]
        phi = np.asarray([fit.params[i] for i in range(len(fit.params))
                          if fit.model.param_names[i].startswith("ar.L")])
        hist = np.concatenate([y_tr, sp.test_df[target].astype("float64").values])
        off = len(y_tr)
        preds = np.zeros((n, N_OUT))
        for i in range(n):
            end = off + i + N_IN                      # first target index
            last = hist[end - 1]
            d = list(np.diff(hist[max(0, end - p - 1):end]))
            out = []
            for _ in range(N_OUT):
                nd = float(np.dot(phi, d[::-1][:p])) if len(d) >= p else 0.0
                last += nd
                out.append(last)
                d.append(nd)
            preds[i] = out
    return preds, tt, time.perf_counter() - t1


def baseline_xgboost(company: pd.DataFrame, features: List[str], target: str, quick: bool):
    """XGBoost with one regressor per horizon; features = all columns except the target (repo)."""
    from xgboost import XGBRegressor
    feats = [f for f in features if f != target]
    sp = prepare_split(company, feats, target)
    Xtr = sp.train_x.reshape(len(sp.train_x), -1)
    ytr = sp.train_y.reshape(len(sp.train_y), -1)
    models = []
    t0 = time.perf_counter()
    for h in range(N_OUT):
        m = XGBRegressor(n_estimators=20 if quick else 200, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
                         random_state=42)
        m.fit(Xtr, ytr[:, h])
        models.append(m)
    tt = time.perf_counter() - t0
    t1 = time.perf_counter()
    Xte = sp.test_x.reshape(len(sp.test_x), -1)
    p = np.stack([m.predict(Xte) for m in models], axis=1)
    return inv(sp.target_scaler, p), tt, time.perf_counter() - t1


def _zlema_repo(src: pd.Series, length: int = 5):
    import pandas_ta_classic as ta
    lag = round((length - 1) / 2)
    return ta.ema(src + (src - src.iloc[lag]), length=5)   # repo: src[lag] is a scalar


def baseline_benchmark(raw_company: pd.DataFrame, start: str, saudi: bool, cfg: TrainCfg,
                       fix_quirks: bool):
    """GRU(256, tanh, Nadam) benchmark of Aseeri (2023) as implemented in TestingResults.ipynb."""
    import pandas_ta_classic as ta
    from sklearn.preprocessing import MinMaxScaler
    keras = _keras()
    if saudi:
        cols = ["open", "high", "low", "close", "change", "perc_change", "volume_traded",
                "value_traded", "no_trades"]
        c = "close"
    else:
        cols = ["Open", "High", "Low", "Close"]          # get_company_df_bench_us
        c = "Close"
    co = _numeric_daily(raw_company[[x for x in cols if x in raw_company.columns]], start,
                        ffill=fix_quirks)   # repo bug: fill_missing() result discarded
    co["dema"] = ta.dema(co[c], length=5)
    co["hma"] = ta.hma(co[c], length=5)
    co["tema"] = ta.tema(co[c], length=5)
    co["wma"] = ta.wma(co[c], length=5)
    co["zlema"] = _zlema_repo(co[c])
    co = co.fillna(0)
    feats = list(co.columns)
    ci = feats.index(c)
    train, val, test = co.loc[:TRAIN_END], co.loc[VAL_START:VAL_END], co.loc[TEST_START:]

    if fix_quirks:
        sc = MinMaxScaler().fit(train)
        scs = (sc, sc, sc)
    else:                                   # repo: separate MinMax on each split (test included)
        scs = (MinMaxScaler().fit(train), MinMaxScaler().fit(val), MinMaxScaler().fit(test))

    def sup(d, s):
        a = s.transform(d)
        if fix_quirks:
            n = len(a) - N_IN - N_OUT + 1
        else:
            n = len(a) - N_IN - N_OUT        # repo: `if out_end < len(data)` drops last window
        X = np.stack([a[i:i + N_IN] for i in range(n)]).astype("float32")
        y = np.stack([a[i + N_IN:i + N_IN + N_OUT, ci] for i in range(n)]).astype("float32")
        return X, y

    trx, tr_y = sup(train, scs[0])
    vx, v_y = sup(val, scs[1])
    tx, _ = sup(test, scs[2])
    m = keras.Sequential([keras.layers.Input(shape=trx.shape[1:]),
                          keras.layers.GRU(256, activation="tanh"), keras.layers.Dense(N_OUT)])
    m.compile(loss="mse", optimizer=keras.optimizers.Nadam())
    t0 = time.perf_counter()
    m.fit(trx, tr_y, epochs=cfg.bench_epochs, batch_size=max(1, math.floor(len(trx) * 0.05)),
          validation_data=(vx, v_y), verbose=0)
    tt = time.perf_counter() - t0
    p, it = predict_timed(m, tx)
    actual = np.array([test[c].values[i + N_IN:i + N_IN + N_OUT] for i in range(len(tx))])
    if fix_quirks:
        mn, mx = scs[0].data_min_[ci], scs[0].data_max_[ci]
        pred = p * (mx - mn) + mn
    else:                                   # repo: MinMax fitted on the actual target windows
        pred = MinMaxScaler().fit(actual).inverse_transform(p)
    return pred, actual, tt, it


# =============================================================================
# 10. PER-COMPANY EXPERIMENT
# =============================================================================

def run_company(name: str, company: pd.DataFrame, features: List[str], target: str,
                units_fixed: Optional[Tuple[int, int, int]], raw_for_bench: Optional[pd.DataFrame],
                bench_start: str, saudi: bool, rc: RunCfg, quick: bool) -> Tuple[pd.DataFrame, Dict]:
    keras = _keras()
    set_seed(rc.seed)
    LOG.info(f"===== {name} =====")
    sp = prepare_split(company, features, target)
    ts = sp.target_scaler
    act = sp.test_y_eval
    preds: Dict[str, np.ndarray] = {}
    times: Dict[str, Tuple[float, float]] = {}   # model -> (train s, inference s)
    rows = []

    def record(model: str, pred: np.ndarray, tt: float, it: float, actual=None):
        preds[model] = pred
        times[model] = (tt, it)
        r = {"Company": name, "Model": model, **metrics(act if actual is None else actual, pred),
             "Train_Time_s": tt, "Inference_Time_s": it,
             "Inference_s_per_sample": it / max(1, len(pred))}
        rows.append(r)
        LOG.info(f"  {model:18s} RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  MRE={r['MRE']:.4f}  "
                 f"R2={r['R2']:.4f}  train={tt:.1f}s")

    # ---------------- base models ----------------
    cache_path = os.path.join(rc.out_dir, "unit_search.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    members, test_p, val_p, train_p, chosen = [], {}, {}, {}, []
    for k, kind in enumerate(BASE_NAMES):
        if units_fixed is not None:
            u = units_fixed[k]
            m, tt = fit_base(kind, u, sp, rc.train)
        else:
            LOG.info(f"  unit search for {kind} (selection on validation RMSE)")
            u, m, tt = choose_units(kind, sp, rc.train, rc.unit_grid, cache, f"{name}|{kind}")
            json.dump(cache, open(cache_path, "w"), indent=1)
        chosen.append(u)
        p, it = predict_timed(m, sp.test_x)
        test_p[kind] = p
        val_p[kind], _ = predict_timed(m, sp.val_x)
        if rc.stacking == "train":
            train_p[kind], _ = predict_timed(m, sp.train_x)
        record(kind, inv(ts, p), tt, it)
        if rc.save_models:
            d = os.path.join(rc.out_dir, "models", safe_name(name))
            os.makedirs(d, exist_ok=True)
            m.save(os.path.join(d, f"{kind}_{u}.keras"))
        members.append(m)
    LOG.info(f"  units (B,G,L) = {tuple(chosen)}")

    # ---------------- averaging ensembles ----------------
    for combo, idx in COMBOS.items():
        t0 = time.perf_counter()
        avg = np.mean(stack_preds([test_p[BASE_NAMES[i]] for i in idx]), axis=2)
        stage = time.perf_counter() - t0
        tt = sum(times[BASE_NAMES[i]][0] for i in idx)
        it = sum(times[BASE_NAMES[i]][1] for i in idx) + stage
        record(f"AVG({combo})", inv(ts, avg), tt, it)

    # ---------------- stacking ensembles ----------------
    if rc.stacking == "oof":
        LOG.info(f"  building rolling-origin OOF meta-features ({rc.oof_folds} folds)")
        oof_p, oof_y = oof_base_predictions(sp, tuple(chosen), rc.train, rc.oof_folds)
    for meta in META_TYPES:
        for combo, idx in COMBOS.items():
            ks = [BASE_NAMES[i] for i in idx]
            Xte = stack_preds([test_p[k] for k in ks])
            Xva = stack_preds([val_p[k] for k in ks])
            if rc.stacking == "val":          # Baseline&WX.ipynb (final repo)
                Xm, ym, vd = Xva, sp.val_y, None
            elif rc.stacking == "train":      # Ensemble_Stacking_in_Neural_Networks.ipynb
                Xm, ym, vd = stack_preds([train_p[k] for k in ks]), sp.train_y, (Xva, sp.val_y)
            else:                             # paper text: rolling-origin OOF
                Xm, ym, vd = stack_preds([oof_p[k] for k in ks]), oof_y, (Xva, sp.val_y)
            mm, mt = train_meta(meta, Xm, ym.reshape(len(ym), N_OUT), rc.train,
                                None if vd is None else (vd[0], vd[1].reshape(len(vd[1]), N_OUT)))
            p, st = predict_timed(mm, Xte)
            tt = sum(times[k][0] for k in ks) + mt
            it = sum(times[k][1] for k in ks) + st
            record(f"STK_{meta}({combo})", inv(ts, p), tt, it)
            del mm
    del members
    keras.backend.clear_session()

    # ---------------- baselines ----------------
    if not rc.skip_baselines:
        p, tt, it = baseline_random_walk(sp, target)
        record("Random Walk", p, tt, it)
        p, tt, it = baseline_arima(sp, target, rolling=rc.fix_quirks)
        record("ARIMA", p, tt, it)
        p, tt, it = baseline_xgboost(company, features, target, quick)
        record("XGBoost", p, tt, it)
        if raw_for_bench is not None:
            set_seed(rc.seed)
            p, a_b, tt, it = baseline_benchmark(raw_for_bench, bench_start, saudi, rc.train, rc.fix_quirks)
            # repo metrics are against the benchmark's own actuals; predictions are aligned to the
            # common windows for the statistical comparison
            record("Benchmark", p, tt, it, actual=a_b)
            keras.backend.clear_session()

    res = pd.DataFrame(rows)
    return res, {"actual": act, "preds": preds, "units": chosen}


# =============================================================================
# 11. WILCOXON SIGNED-RANK + HOLM WIN/LOSS RANKING  (Baseline&WX.ipynb)
# =============================================================================

def to_block_frame(actual: np.ndarray, preds: Dict[str, np.ndarray], models: List[str]) -> pd.DataFrame:
    n = min([len(actual)] + [len(preds[m]) for m in models])
    d = {f"Actual_t+{h + 1}": actual[:n, h] for h in range(N_OUT)}
    for m in models:
        for h in range(N_OUT):
            d[f"{m}_t+{h + 1}"] = np.asarray(preds[m])[:n, h]
    return pd.DataFrame(d)


def wilcoxon_ranking(per_company: Dict[str, Dict], models: List[str], mode: str, out_dir: str,
                     tag: str, window: int = N_OUT, alpha: float = 0.05):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    recs = []
    for comp, d in per_company.items():
        ms = [m for m in models if m in d["preds"]]
        df = to_block_frame(d["actual"], d["preds"], ms)
        df.to_csv(os.path.join(out_dir, "predictions", f"{safe_name(comp)}_block_predictions.csv"),
                  index=False)
        nb = len(df) // window
        A = {h: df[f"Actual_t+{h}"].values[:nb * window].reshape(nb, window) for h in range(1, N_OUT + 1)}
        mcols = [c for c in df.columns if not c.startswith("Actual")]
        P = {c: df[c].values[:nb * window].reshape(nb, window) for c in mcols}
        for h in range(1, N_OUT + 1):
            cols = mcols if mode == "repo" else [c for c in mcols if c.endswith(f"_t+{h}")]
            R = {c: np.sqrt(np.mean((P[c] - A[h]) ** 2, axis=1)) for c in cols}  # non-overlapping RMSE
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    r1, r2 = R[cols[i]], R[cols[j]]
                    try:
                        stat, p = wilcoxon(r1, r2)
                    except ValueError:          # identical samples
                        stat, p = np.nan, 1.0
                    recs.append((comp, h, cols[i], cols[j], stat, p, r1.mean(), r2.mean()))
    res = pd.DataFrame(recs, columns=["Company", "Horizon", "Model1", "Model2", "Statistic",
                                      "p_value", "mean_rmse_m1", "mean_rmse_m2"])
    res["p_value"] = res["p_value"].fillna(1.0)
    rej, pc, _, _ = multipletests(res["p_value"].values, alpha=alpha, method="holm")
    res["p_corrected"], res["Significant"] = pc, rej
    res.to_csv(os.path.join(out_dir, f"wilcoxon_pairwise_{tag}.csv"), index=False)

    strip = lambda c: re.sub(r"_t\+\d+$", "", c)  # noqa: E731
    wl = {}
    sig = res[res["Significant"]]
    m1b, m2b = sig["Model1"].map(strip), sig["Model2"].map(strip)
    better1 = sig["mean_rmse_m1"].values < sig["mean_rmse_m2"].values
    for a, b, w in zip(m1b, m2b, better1):
        win, lose = (a, b) if w else (b, a)
        wl.setdefault(win, [0, 0])[0] += 1
        wl.setdefault(lose, [0, 0])[1] += 1
    rank = pd.DataFrame([{"Model": k, "Wins": v[0], "Losses": v[1],
                          "WinRate_%": round(100 * v[0] / max(1, v[0] + v[1]), 2)} for k, v in wl.items()])
    rank = rank.sort_values("WinRate_%", ascending=False)
    rank["Paper_WinRate_%"] = rank["Model"].map(PAPER_TABLE11)
    rank.to_csv(os.path.join(out_dir, f"wilcoxon_ranking_{tag}.csv"), index=False)
    return rank


# =============================================================================
# 12. REPORTING
# =============================================================================

def best_table(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for comp, g in res.groupby("Company", sort=False):
        g = g[g["Model"] != "Benchmark"]
        row = {"Company": comp}
        for met in ("RMSE", "MAE", "MRE"):
            b = g.loc[g[met].idxmin()]
            row[f"Model({met})"], row[met] = b["Model"], round(b[met], 4)
        if comp in PAPER_BEST:
            pm, pr = PAPER_BEST[comp]
            row["Paper_Model(RMSE)"], row["Paper_RMSE"] = pm, pr
            hit = g[g["Model"] == pm]
            row["Ours_RMSE_of_paper_model"] = round(float(hit["RMSE"].iloc[0]), 4) if len(hit) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def timing_table(res: pd.DataFrame) -> pd.DataFrame:
    return (res.groupby("Model", sort=False)[["Train_Time_s", "Inference_Time_s", "Inference_s_per_sample"]]
            .mean().round(4))


def market_report(res: pd.DataFrame, per_company: Dict[str, Dict], market: str, rc: RunCfg,
                  wx_mode: str, wx_models: str, skip_wx: bool):
    od = rc.out_dir
    res.to_csv(os.path.join(od, f"results_{market}_all_models.csv"), index=False)
    bt = best_table(res)
    bt.to_csv(os.path.join(od, f"results_{market}_best_per_company.csv"), index=False)
    tt = timing_table(res)
    tt.to_csv(os.path.join(od, f"results_{market}_timing.csv"))
    pivot = res.pivot_table(index="Model", columns="Company", values="RMSE", sort=False).round(4)
    pivot.to_csv(os.path.join(od, f"results_{market}_rmse_matrix.csv"))
    LOG.info(f"\n===== {market.upper()}: best model per company (cf. paper Table {'10' if market == 'saudi' else '13'}) =====\n"
             + bt.to_string(index=False))
    LOG.info(f"\n===== {market.upper()}: RMSE of every model =====\n" + pivot.to_string())
    LOG.info(f"\n===== {market.upper()}: average computational cost (cf. Table 12) =====\n" + tt.to_string())
    if not skip_wx and len(per_company) > 0:
        all_models = list(dict.fromkeys(res["Model"]))
        models = [m for m in PAPER_WX_MODELS if m in all_models] if wx_models == "paper" else all_models
        LOG.info(f"Running Wilcoxon ({wx_mode} mode) over {len(models)} models ...")
        rank = wilcoxon_ranking(per_company, models, wx_mode, od, f"{market}_{wx_mode}")
        LOG.info(f"\n===== {market.upper()}: Wilcoxon win/loss ranking (cf. Table 11) =====\n"
                 + rank.to_string(index=False))


# =============================================================================
# 13. MAIN
# =============================================================================

def save_company(rc: RunCfg, market: str, name: str, res: pd.DataFrame, d: Dict):
    pdir = os.path.join(rc.out_dir, "predictions")
    os.makedirs(pdir, exist_ok=True)
    res.to_csv(os.path.join(pdir, f"{market}_{safe_name(name)}_metrics.csv"), index=False)
    np.savez_compressed(os.path.join(pdir, f"{market}_{safe_name(name)}.npz"),
                        actual=d["actual"], units=np.array(d["units"]),
                        **{f"pred::{k}": v for k, v in d["preds"].items()})


def load_company(rc: RunCfg, market: str, name: str):
    pdir = os.path.join(rc.out_dir, "predictions")
    f_m = os.path.join(pdir, f"{market}_{safe_name(name)}_metrics.csv")
    f_p = os.path.join(pdir, f"{market}_{safe_name(name)}.npz")
    if not (os.path.exists(f_m) and os.path.exists(f_p)):
        return None
    z = np.load(f_p, allow_pickle=False)
    preds = {k.split("::", 1)[1]: z[k] for k in z.files if k.startswith("pred::")}
    return pd.read_csv(f_m), {"actual": z["actual"], "preds": preds, "units": list(z["units"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["saudi", "us", "both"], default="saudi")
    ap.add_argument("--companies", nargs="*", default=None, help="subset of company names / tickers")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--saudi-csv", default=None, help="local Tadawul_stcks.csv (skip download)")
    ap.add_argument("--us-dir", default=None, help="folder with AAPL.csv, BAC.csv, ... (skip download)")
    ap.add_argument("--train-config", choices=["repo", "paper"], default="repo")
    ap.add_argument("--stacking", choices=["val", "train", "oof"], default="val",
                    help="meta-learner training data (val=final repo, train=older notebook, oof=paper text)")
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--units", default="repo",
                    help="'repo' (fixed Saudi table; US falls back to grid), 'grid', or an integer")
    ap.add_argument("--unit-grid", type=int, nargs="*", default=None)
    ap.add_argument("--feature-selection", choices=["fixed", "pearson"], default="fixed")
    ap.add_argument("--pearson-threshold", type=float, default=0.8)
    ap.add_argument("--fix-quirks", action="store_true",
                    help="rolling ARIMA forecasts + leak-free / ffilled GRU(256) benchmark")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-wilcoxon", action="store_true")
    ap.add_argument("--wx-mode", choices=["repo", "horizon"], default="repo")
    ap.add_argument("--wx-models", choices=["all", "paper"], default="paper",
                    help="'paper' = the 16 models of Table 11")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true", help="tf op determinism (slower)")
    ap.add_argument("--resume", action="store_true", help="reuse finished companies in --out-dir")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (2 epochs, 8 units)")
    a = ap.parse_args(argv)

    setup_logging(a.out_dir)
    import tensorflow as tf
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()
    try:
        import pandas_ta_classic
        pta = getattr(pandas_ta_classic, "version", getattr(pandas_ta_classic, "__version__", "?"))
    except Exception as e:
        raise SystemExit(f"pandas-ta-classic import failed ({e}). pip install pandas-ta-classic")
    LOG.info(f"python {sys.version.split()[0]} | tensorflow {tf.__version__} | numpy {np.__version__} | "
             f"pandas {pd.__version__} | pandas-ta-classic {pta}")
    LOG.info(f"args: {vars(a)}")

    rc = RunCfg(data_dir=a.data_dir, out_dir=a.out_dir, train=make_train_cfg(a.train_config, a.quick),
                stacking=a.stacking, oof_folds=a.oof_folds, units=a.units,
                fix_quirks=a.fix_quirks, feature_selection=a.feature_selection,
                pearson_threshold=a.pearson_threshold, skip_baselines=a.skip_baselines,
                seed=a.seed, resume=a.resume, save_models=a.save_models)
    if a.unit_grid:
        rc.unit_grid = a.unit_grid
    if a.quick:
        rc.unit_grid = [4, 8]
        if rc.oof_folds > 2:
            rc.oof_folds = 2
    os.makedirs(os.path.join(rc.out_dir, "predictions"), exist_ok=True)
    LOG.info(f"training config: {rc.train}")

    def units_for(market: str, name: str):
        if a.quick and rc.units != "grid":
            return (8, 8, 8)
        if rc.units.isdigit():
            return (int(rc.units),) * 3
        if rc.units == "repo" and market == "saudi":
            return SAUDI_COMPANIES[name][1]
        return None  # grid search

    markets = ["saudi", "us"] if a.market == "both" else [a.market]

    # ------------------------- SAUDI -------------------------
    if "saudi" in markets:
        raw = load_saudi_raw(acquire_saudi(rc.data_dir, a.saudi_csv))
        names = [c for c in SAUDI_COMPANIES if (not a.companies or c in a.companies)]
        features = list(SELECTED_TI_SA)
        if rc.feature_selection == "pearson":
            LOG.info("Pearson feature selection over the full indicator set (train rows only)")
            full = {}
            for n in names:
                st = SAUDI_COMPANIES[n][0]
                co = raw[raw["trading_name"] == n].drop(columns=["symbol", "name", "trading_name", "sector"],
                                                        errors="ignore")
                co = _numeric_daily(co, st)
                f = all_indicators(co, "open", "high", "low", "close", "volume_traded")
                full[n] = co.fillna(0).join(f, rsuffix="_ti")
            features = pearson_feature_selection(full, "close", rc.pearson_threshold)
            LOG.info(f"overlap with Table 4: {len(set(features) & set(SELECTED_TI_SA))}/34")
            json.dump(features, open(os.path.join(rc.out_dir, "pearson_features_saudi.json"), "w"))
        all_res, per_company = [], {}
        for n in names:
            got = load_company(rc, "saudi", n) if rc.resume else None
            if got is None:
                st = SAUDI_COMPANIES[n][0]
                if rc.feature_selection == "pearson":
                    co = full[n][features]
                else:
                    co = engineer_saudi(raw, n, st, features)
                rawc = raw[raw["trading_name"] == n]
                res, d = run_company(n, co, features, "close", units_for("saudi", n), rawc, st, True, rc, a.quick)
                save_company(rc, "saudi", n, res, d)
            else:
                LOG.info(f"resume: loaded {n}")
                res, d = got
            all_res.append(res)
            per_company[n] = d
        market_report(pd.concat(all_res, ignore_index=True), per_company, "saudi", rc,
                      a.wx_mode, a.wx_models, a.skip_wilcoxon)

    # ------------------------- US -------------------------
    if "us" in markets:
        tickers = [t for t in US_COMPANIES if (not a.companies or t in a.companies)] or US_COMPANIES
        paths = acquire_us(rc.data_dir, tickers, a.us_dir)
        all_res, per_company = [], {}
        for t in tickers:
            got = load_company(rc, "us", t) if rc.resume else None
            if got is None:
                raw_t = load_us_raw(paths[t])
                co = engineer_us(raw_t)
                res, d = run_company(t, co, list(SELECTED_TI_US), "Close", units_for("us", t), raw_t,
                                     US_START, False, rc, a.quick)
                save_company(rc, "us", t, res, d)
            else:
                LOG.info(f"resume: loaded {t}")
                res, d = got
            all_res.append(res)
            per_company[t] = d
        market_report(pd.concat(all_res, ignore_index=True), per_company, "us", rc,
                      a.wx_mode, a.wx_models, True)   # paper reports no Wilcoxon for US

    LOG.info(f"Done. All outputs in {os.path.abspath(rc.out_dir)}")


if __name__ == "__main__":
    main()
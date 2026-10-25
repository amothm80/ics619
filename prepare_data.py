#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py
===============

Data side of the reproduction of Baamer & Aljamaan (2026), PeerJ Comput. Sci. 12:e4064
("Ensemble deep learning models for stock price forecasting: evidence from the Saudi markets").
Moved out of tasi_edl_pipeline.py (the reference implementation) with the logic unchanged.

  1. Acquire data from Kaggle (kagglehub) or from local CSVs.
  2. Per company: reindex to a *calendar-daily* grid, forward-fill, compute the 34 technical
     indicators of Table 4 with pandas-ta-classic (length 5). Optional: re-run the Pearson
     |r| > 0.8 selection over ~130 TIs.
  3. Chronological split train <= 2015-12-31 | val = 2016 | test >= 2017-01-01, StandardScaler
     fit on train only, sliding windows w = 5 -> h = 5.

Outputs, one folder per company, data/prepared/{market}/{company}/:
  features.parquet  calendar-daily, ffilled, selected TIs (incl. target), date index
  windows.npz       train/val/test x,y (scaled), val_y_eval/test_y_eval (original scale), test_dates
  scalers.joblib    feature + target StandardScaler (fit on train only)
  raw.parquet       the company's raw rows before reindexing (for the GRU(256) benchmark)
  meta.json         feature list, target, start date, row/window counts, config, library versions
plus data/prepared/{market}/features.json (feature list; the Pearson result when used).

Run once per feature configuration; train_report.py consumes the folder.

Usage
-----
  python prepare_data.py                                  # Saudi, Table 4 features
  python prepare_data.py --companies ADC --saudi-csv path/Tadawul_stcks.csv
  python prepare_data.py --market both --us-dir path/csvs
  python prepare_data.py --feature-selection pearson --out data/prepared_pearson
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

warnings.filterwarnings("ignore")

LOG = logging.getLogger("edl")

# =============================================================================
# 0. DATA-SIDE CONSTANTS (taken from the paper and the repository)
# =============================================================================

TRAIN_END, VAL_START, VAL_END, TEST_START = "2015-12-31", "2016-01-01", "2016-12-31", "2017-01-01"
DATA_END = "2020-03-05"
# split boundaries as stored in meta.json["split"]; folders without that key use these
PAPER_SPLIT = {"train_end": TRAIN_END, "val_start": VAL_START, "val_end": VAL_END,
               "test_start": TEST_START, "end": DATA_END}
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
TARGET = {"saudi": "close", "us": "Close"}


# =============================================================================
# 2. UTILITIES
# =============================================================================

def setup_logging(out_dir: str, name: str = "run.log") -> None:
    os.makedirs(out_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, name))):
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

def _numeric_daily(df: pd.DataFrame, start: str, ffill: bool = True, end: str = DATA_END) -> pd.DataFrame:
    """Reindex to calendar days, '?' -> NaN, float32, forward fill (repo behaviour)."""
    df = df[~df.index.duplicated(keep="last")]
    df = df.reindex(pd.date_range(start=start, end=end, freq="D"))
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
                   selected: List[str] = SELECTED_TI_SA, end: str = DATA_END) -> pd.DataFrame:
    co = raw[raw["trading_name"] == company.strip()]
    if co.empty:
        raise ValueError(f"company '{company}' not found in Saudi dataset")
    co = co.drop(columns=[c for c in ("symbol", "name", "trading_name", "sector") if c in co.columns])
    co = _numeric_daily(co, start, end=end)
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
                              threshold: float, train_end: str = TRAIN_END) -> List[str]:
    """|r| > threshold with close on TRAIN rows of every company; intersection across companies."""
    sets = []
    for name, f in frames.items():
        tr = f.loc[:train_end]
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
    feature_scaler: object = None


def split_frames(company: pd.DataFrame, split: Optional[Dict] = None):
    s = split or PAPER_SPLIT
    return company.loc[:s["train_end"]], company.loc[s["val_start"]:s["val_end"]], company.loc[s["test_start"]:]


def prepare_split(company: pd.DataFrame, features: List[str], target: str, split: Optional[Dict] = None) -> Split:
    from sklearn.preprocessing import StandardScaler
    train, val, test = split_frames(company, split)
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
                 train, val, test, ts, fs)



# =============================================================================
# 6. PREPARED-DATA HANDOFF  (written here, read by train_report.py)
# =============================================================================

PREPARED_FILES = ("features.parquet", "windows.npz", "scalers.joblib", "raw.parquet", "meta.json")


def company_dir(root: str, market: str, company: str) -> str:
    return os.path.join(root, market, safe_name(company))


def library_versions() -> Dict[str, str]:
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for p in ("numpy", "pandas", "pandas-ta-classic", "scikit-learn", "pyarrow"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def write_prepared(d: str, company: str, market: str, co: pd.DataFrame, raw_c: pd.DataFrame,
                   sp: Split, features: List[str], target: str, start: str, config: Dict,
                   split: Optional[Dict] = None) -> None:
    import joblib
    os.makedirs(d, exist_ok=True)
    co.to_parquet(os.path.join(d, "features.parquet"))
    raw_c.to_parquet(os.path.join(d, "raw.parquet"))
    test_dates = np.array([t.strftime("%Y-%m-%d") for t in sp.test_df.index[:len(sp.test_x)]])
    np.savez_compressed(os.path.join(d, "windows.npz"),
                        train_x=sp.train_x, train_y=sp.train_y, val_x=sp.val_x, val_y=sp.val_y,
                        test_x=sp.test_x, test_y=sp.test_y, val_y_eval=sp.val_y_eval,
                        test_y_eval=sp.test_y_eval, test_dates=test_dates)
    joblib.dump({"feature": sp.feature_scaler, "target": sp.target_scaler}, os.path.join(d, "scalers.joblib"))
    meta = {"company": company, "market": market, "features": features, "target": target, "start": start,
            "rows": {"train": len(sp.train_df), "val": len(sp.val_df), "test": len(sp.test_df)},
            "windows": {"train": len(sp.train_x), "val": len(sp.val_x), "test": len(sp.test_x)},
            "config": config, "versions": library_versions(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if split is not None:                                  # absent = PAPER_SPLIT (keeps old meta hashes)
        meta["split"] = split
    with open(os.path.join(d, "meta.json"), "w") as f:     # written last: marks the folder complete
        json.dump(meta, f, indent=1)


def load_prepared(d: str) -> Tuple[Split, pd.DataFrame, pd.DataFrame, Dict]:
    """-> (Split with the original-scale split frames, features frame, raw rows, meta)."""
    import joblib
    missing = [f for f in PREPARED_FILES if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise SystemExit(f"{d}: missing {missing}; run prepare_data.py first")
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    co = pd.read_parquet(os.path.join(d, "features.parquet"))
    raw_c = pd.read_parquet(os.path.join(d, "raw.parquet"))
    sc = joblib.load(os.path.join(d, "scalers.joblib"))
    with np.load(os.path.join(d, "windows.npz"), allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    sp = Split(w["train_x"], w["train_y"], w["val_x"], w["val_y"], w["test_x"], w["test_y"],
               w["test_y_eval"], w["val_y_eval"], *split_frames(co, split_of(meta)), sc["target"], sc["feature"])
    return sp, co, raw_c, meta


def split_of(meta: Dict) -> Dict:
    return meta.get("split", PAPER_SPLIT)


def meta_hash(d: str) -> str:
    with open(os.path.join(d, "meta.json"), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


# =============================================================================
# 7. MAIN
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["saudi", "us", "both"], default="saudi")
    ap.add_argument("--companies", nargs="*", default=None, help="subset of company names / tickers")
    ap.add_argument("--data-dir", default="data", help="raw data (Kaggle downloads land here)")
    ap.add_argument("--saudi-csv", default=None, help="local Tadawul_stcks.csv (skip download)")
    ap.add_argument("--us-dir", default=None, help="folder with AAPL.csv, BAC.csv, ... (skip download)")
    ap.add_argument("--feature-selection", choices=["fixed", "pearson"], default="fixed")
    ap.add_argument("--pearson-threshold", type=float, default=0.8)
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--force", action="store_true", help="rebuild companies that are already prepared")
    a = ap.parse_args(argv)

    setup_logging(a.out, "prepare.log")
    try:
        import pandas_ta_classic
        pta = getattr(pandas_ta_classic, "version", getattr(pandas_ta_classic, "__version__", "?"))
    except Exception as e:
        raise SystemExit(f"pandas-ta-classic import failed ({e}). pip install pandas-ta-classic")
    LOG.info(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
             f"pandas-ta-classic {pta}")
    LOG.info(f"args: {vars(a)}")
    config = {"feature_selection": a.feature_selection, "pearson_threshold": a.pearson_threshold}

    def done(market, name):
        d = company_dir(a.out, market, name)
        return not a.force and all(os.path.exists(os.path.join(d, f)) for f in PREPARED_FILES)

    markets = ["saudi", "us"] if a.market == "both" else [a.market]

    # ------------------------- SAUDI -------------------------
    if "saudi" in markets:
        src = acquire_saudi(a.data_dir, a.saudi_csv)
        names = [c for c in SAUDI_COMPANIES if (not a.companies or c in a.companies)]
        info = {"selection": a.feature_selection, "source": src}
        features = list(SELECTED_TI_SA)
        todo = [n for n in names if not done("saudi", n)]
        # Pearson selection depends on every requested company, so it reruns them all
        if a.feature_selection == "pearson" and todo:
            todo = names
        if not todo:
            LOG.info(f"saudi: all {len(names)} companies already prepared in {a.out} (use --force)")
        else:
            raw = load_saudi_raw(src)
            if a.feature_selection == "pearson":
                LOG.info("Pearson feature selection over the full indicator set (train rows only)")
                full = {}
                for n in names:
                    st = SAUDI_COMPANIES[n][0]
                    co = raw[raw["trading_name"] == n].drop(columns=["symbol", "name", "trading_name", "sector"],
                                                            errors="ignore")
                    co = _numeric_daily(co, st)
                    f = all_indicators(co, "open", "high", "low", "close", "volume_traded")
                    full[n] = co.fillna(0).join(f, rsuffix="_ti")
                features = pearson_feature_selection(full, "close", a.pearson_threshold)
                LOG.info(f"overlap with Table 4: {len(set(features) & set(SELECTED_TI_SA))}/34")
                info.update({"pearson_threshold": a.pearson_threshold, "companies": names})
            for n in todo:
                LOG.info(f"===== {n} =====")
                st = SAUDI_COMPANIES[n][0]
                co = full[n][features] if a.feature_selection == "pearson" else engineer_saudi(raw, n, st, features)
                rawc = raw[raw["trading_name"] == n]
                sp = prepare_split(co, features, "close")
                write_prepared(company_dir(a.out, "saudi", n), n, "saudi", co, rawc, sp, features, "close", st,
                               {**config, "source": src})
            info["features"] = features
            with open(os.path.join(a.out, "saudi", "features.json"), "w") as f:
                json.dump(info, f, indent=1)

    # ------------------------- US -------------------------
    if "us" in markets:
        tickers = [t for t in US_COMPANIES if (not a.companies or t in a.companies)] or US_COMPANIES
        paths = acquire_us(a.data_dir, tickers, a.us_dir)
        for t in tickers:
            if done("us", t):
                LOG.info(f"us: {t} already prepared (use --force)")
                continue
            LOG.info(f"===== {t} =====")
            raw_t = load_us_raw(paths[t])
            co = engineer_us(raw_t)
            sp = prepare_split(co, list(SELECTED_TI_US), "Close")
            write_prepared(company_dir(a.out, "us", t), t, "us", co, raw_t, sp, list(SELECTED_TI_US), "Close",
                           US_START, {**config, "source": paths[t]})
        os.makedirs(os.path.join(a.out, "us"), exist_ok=True)
        with open(os.path.join(a.out, "us", "features.json"), "w") as f:
            json.dump({"selection": "fixed", "features": list(SELECTED_TI_US)}, f, indent=1)

    LOG.info(f"Done. Prepared data in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

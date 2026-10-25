#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_report.py
===============

Model side of the reproduction of Baamer & Aljamaan (2026), PeerJ Comput. Sci. 12:e4064.
Moved out of tasi_edl_pipeline.py (the reference implementation); it reads the folders written
by prepare_data.py instead of computing the split itself.

  4. Base models: BiRNN (B), GRU (G), LSTM (L) - single recurrent layer + Dense(5), ReLU, Adam, MSE.
  5. Ensembles: 4 averaging + 12 stacking (meta-learner ANN 128-64-32 / GRU 128 / LSTM 128).
  6. Baselines: Random Walk with drift, ARIMA(5,1,0), XGBoost, GRU(256) benchmark (Aseeri 2023).
  7. Metrics on de-scaled, 2-decimal-rounded prices, timing, Wilcoxon + Holm win/loss ranking.

Defaults follow the repository (Baseline&WX.ipynb); every paper-vs-repo difference is a switch
(--train-config paper, --stacking val/train/oof, --units repo/grid/<int>, --fix-quirks,
--wx-mode repo/horizon). Repo quirks are reproduced on purpose by default.

Outputs under --run-dir:
  base_preds/{company}.npz   scaled BRNN/GRU/LSTM preds on train/val/test, times, chosen units
  preds/{company}.npz        every model's original-scale preds, actuals and times
  models/{company}/*.keras   only with --save-models
  results_{market}_*.csv, wilcoxon_*.csv, predictions/*_block_predictions.csv, run.log,
  unit_search.json, run_config.json

Usage
-----
  python train_report.py --run-dir runs/repo                         # repo-faithful, Saudi
  python train_report.py --run-dir runs/adc --companies ADC --deterministic
  python train_report.py --run-dir runs/paper --train-config paper --stacking oof
  python train_report.py --run-dir runs/stk_train --reuse-base runs/repo --stacking train
  python train_report.py --run-dir runs/repo --report-only --wx-mode horizon
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

from prepare_data import (N_IN, N_OUT, PAPER_SPLIT, SAUDI_COMPANIES, US_COMPANIES, Split, _numeric_daily,
                          company_dir, load_prepared, meta_hash, safe_name, setup_logging, split_frames, split_of)

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LOG = logging.getLogger("edl")

# =============================================================================
# 0. MODEL-SIDE CONSTANTS (taken from the paper and the repository)
# =============================================================================

# unit grid of the standalone notebooks
UNIT_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3, 9, 27, 81, 243, 729,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

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
    force: bool = False                # redo companies that already have outputs
    reuse_base: Optional[str] = None   # run dir whose base_preds/ replace base-model training
    report_only: bool = False          # no training: rebuild reports from saved preds
    save_models: bool = False


# =============================================================================
# 2. UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)



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


PROGRESS = True   # --quiet turns the per-epoch progress lines off


def progress_cb(label: str, epochs: int):
    """Keras callback that logs loss / val_loss a few times per fit. Read-only: no effect on training."""
    keras = _keras()
    every = max(1, epochs // 4)

    class _CB(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.t0, self.last = time.perf_counter(), 0

        def on_epoch_end(self, epoch, logs=None):
            self.last = e = epoch + 1
            if e % every and e != epochs and e != 1:
                return
            logs = logs or {}
            vl = f"  val_loss={logs['val_loss']:.5f}" if "val_loss" in logs else ""
            LOG.info(f"    {label:20s} epoch {e:3d}/{epochs}  loss={logs.get('loss', float('nan')):.5f}{vl}"
                     f"  {time.perf_counter() - self.t0:6.1f}s")

        def on_train_end(self, logs=None):
            if self.last != epochs:
                LOG.info(f"    {label:20s} early stop at epoch {self.last}/{epochs}")

    return [_CB()] if PROGRESS else []


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
    cbs += progress_cb(f"{kind}({units})", cfg.epochs)
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
               val: Optional[Tuple[np.ndarray, np.ndarray]], label: Optional[str] = None):
    m = build_meta(kind, Xm.shape[1:], cfg.meta_lr)
    cbs = [BestWeights.make(cfg.meta_patience)] if (cfg.meta_patience and val is not None) else []
    cbs += progress_cb(label or f"meta {kind}", cfg.meta_epochs)
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


def baseline_xgboost(sp: Split, features: List[str], target: str, quick: bool):
    """XGBoost with one regressor per horizon; features = all columns except the target (repo).
    StandardScaler works column by column, so dropping the target column from the prepared windows
    gives exactly the arrays the repo builds with a separate prepare_split."""
    from xgboost import XGBRegressor
    keep = [i for i, f in enumerate(features) if f != target]
    Xtr = sp.train_x[:, :, keep].reshape(len(sp.train_x), -1)
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
    Xte = sp.test_x[:, :, keep].reshape(len(sp.test_x), -1)
    p = np.stack([m.predict(Xte) for m in models], axis=1)
    return inv(sp.target_scaler, p), tt, time.perf_counter() - t1


def _zlema_repo(src: pd.Series, length: int = 5):
    import pandas_ta_classic as ta
    lag = round((length - 1) / 2)
    return ta.ema(src + (src - src.iloc[lag]), length=5)   # repo: src[lag] is a scalar


def baseline_benchmark(raw_company: pd.DataFrame, start: str, saudi: bool, cfg: TrainCfg,
                       fix_quirks: bool, split: Dict = PAPER_SPLIT):
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
                        ffill=fix_quirks, end=split["end"])   # repo bug: fill_missing() result discarded
    co["dema"] = ta.dema(co[c], length=5)
    co["hma"] = ta.hma(co[c], length=5)
    co["tema"] = ta.tema(co[c], length=5)
    co["wma"] = ta.wma(co[c], length=5)
    co["zlema"] = _zlema_repo(co[c])
    co = co.fillna(0)
    feats = list(co.columns)
    ci = feats.index(c)
    train, val, test = split_frames(co, split)

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
          validation_data=(vx, v_y), verbose=0, callbacks=progress_cb("Benchmark GRU(256)", cfg.bench_epochs))
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

def result_row(name: str, model: str, actual: np.ndarray, pred: np.ndarray, tt: float, it: float) -> Dict:
    return {"Company": name, "Model": model, **metrics(actual, pred),
            "Train_Time_s": tt, "Inference_Time_s": it,
            "Inference_s_per_sample": it / max(1, len(pred))}


def save_base_preds(rc: RunCfg, name: str, test_p, val_p, train_p, times, units, prepared_hash: str):
    d = os.path.join(rc.out_dir, "base_preds")
    os.makedirs(d, exist_ok=True)
    arr = {"units": np.array(units),
           "meta": np.array(json.dumps({"company": name, "prepared_meta_hash": prepared_hash,
                                        "train": asdict(rc.train), "units_arg": rc.units, "seed": rc.seed}))}
    for kind in BASE_NAMES:
        arr[f"{kind}::train"], arr[f"{kind}::val"], arr[f"{kind}::test"] = train_p[kind], val_p[kind], test_p[kind]
        arr[f"{kind}::train_time"], arr[f"{kind}::infer_time"] = np.array(times[kind][0]), np.array(times[kind][1])
    np.savez_compressed(os.path.join(d, f"{safe_name(name)}.npz"), **arr)


def load_base_preds(run_dir: str, name: str):
    f = os.path.join(run_dir, "base_preds", f"{safe_name(name)}.npz")
    if not os.path.exists(f):
        raise SystemExit(f"--reuse-base: {f} not found")
    with np.load(f, allow_pickle=False) as z:
        test_p = {k: z[f"{k}::test"] for k in BASE_NAMES}
        val_p = {k: z[f"{k}::val"] for k in BASE_NAMES}
        train_p = {k: z[f"{k}::train"] for k in BASE_NAMES}
        times = {k: (float(z[f"{k}::train_time"]), float(z[f"{k}::infer_time"])) for k in BASE_NAMES}
        return test_p, val_p, train_p, times, [int(u) for u in z["units"]], json.loads(str(z["meta"]))


def run_company(name: str, prep_dir: str, units_fixed: Optional[Tuple[int, int, int]], saudi: bool,
                rc: RunCfg, quick: bool) -> Tuple[pd.DataFrame, Dict]:
    keras = _keras()
    set_seed(rc.seed)
    LOG.info(f"===== {name} =====")
    sp, company, raw_for_bench, pmeta = load_prepared(prep_dir)
    features, target, bench_start = pmeta["features"], pmeta["target"], pmeta["start"]
    split = split_of(pmeta)
    LOG.info(f"  prepared {prep_dir}: windows train/val/test = {len(sp.train_x)}/{len(sp.val_x)}/"
             f"{len(sp.test_x)}; features = {len(features)}; source = {pmeta['config'].get('source')}")
    LOG.info(f"  split: {bench_start} .. {split['train_end']} | {split['val_start']} .. {split['val_end']} | "
             f"{split['test_start']} .. {split['end']}")
    ts = sp.target_scaler
    act = sp.test_y_eval
    preds: Dict[str, np.ndarray] = {}
    actuals: Dict[str, np.ndarray] = {}
    times: Dict[str, Tuple[float, float]] = {}   # model -> (train s, inference s)
    rows = []

    def record(model: str, pred: np.ndarray, tt: float, it: float, actual=None):
        preds[model] = pred
        actuals[model] = act if actual is None else actual
        times[model] = (tt, it)
        r = result_row(name, model, actuals[model], pred, tt, it)
        rows.append(r)
        LOG.info(f"  {model:18s} RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  MRE={r['MRE']:.4f}  "
                 f"R2={r['R2']:.4f}  train={tt:.1f}s")

    # ---------------- base models ----------------
    test_p, val_p, train_p, chosen = {}, {}, {}, []
    if rc.reuse_base:
        test_p, val_p, train_p, base_times, chosen, bmeta = load_base_preds(rc.reuse_base, name)
        LOG.info(f"  base models reused from {rc.reuse_base} (no training)")
        if bmeta.get("prepared_meta_hash") != meta_hash(prep_dir):
            LOG.warning(f"  WARNING: base preds in {rc.reuse_base} were made from different prepared data "
                        f"({bmeta.get('prepared_meta_hash')} vs {meta_hash(prep_dir)})")
        for kind in BASE_NAMES:
            record(kind, inv(ts, test_p[kind]), *base_times[kind])
    else:
        cache_path = os.path.join(rc.out_dir, "unit_search.json")
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
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
            train_p[kind], _ = predict_timed(m, sp.train_x)   # saved for --stacking train re-runs
            record(kind, inv(ts, p), tt, it)
            if rc.save_models:
                d = os.path.join(rc.out_dir, "models", safe_name(name))
                os.makedirs(d, exist_ok=True)
                m.save(os.path.join(d, f"{kind}_{u}.keras"))
            del m
    save_base_preds(rc, name, test_p, val_p, train_p, {k: times[k] for k in BASE_NAMES}, chosen,
                    meta_hash(prep_dir))
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
    LOG.info(f"  stacking: training {len(META_TYPES) * len(COMBOS)} meta-learners (--stacking {rc.stacking})")
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
                                None if vd is None else (vd[0], vd[1].reshape(len(vd[1]), N_OUT)),
                                label=f"STK_{meta}({combo})")
            p, st = predict_timed(mm, Xte)
            tt = sum(times[k][0] for k in ks) + mt
            it = sum(times[k][1] for k in ks) + st
            record(f"STK_{meta}({combo})", inv(ts, p), tt, it)
            del mm
    keras.backend.clear_session()

    # ---------------- baselines ----------------
    if not rc.skip_baselines:
        LOG.info("  baselines: Random Walk, ARIMA(5,1,0), XGBoost, GRU(256) benchmark")
        p, tt, it = baseline_random_walk(sp, target)
        record("Random Walk", p, tt, it)
        p, tt, it = baseline_arima(sp, target, rolling=rc.fix_quirks)
        record("ARIMA", p, tt, it)
        p, tt, it = baseline_xgboost(sp, features, target, quick)
        record("XGBoost", p, tt, it)
        if raw_for_bench is not None:
            set_seed(rc.seed)
            p, a_b, tt, it = baseline_benchmark(raw_for_bench, bench_start, saudi, rc.train, rc.fix_quirks,
                                                split)
            # repo metrics are against the benchmark's own actuals; predictions are aligned to the
            # common windows for the statistical comparison
            record("Benchmark", p, tt, it, actual=a_b)
            keras.backend.clear_session()

    res = pd.DataFrame(rows)
    return res, {"actual": act, "preds": preds, "actuals": actuals, "times": times, "units": chosen}


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

def save_company(rc: RunCfg, name: str, d: Dict):
    pdir = os.path.join(rc.out_dir, "preds")
    os.makedirs(pdir, exist_ok=True)
    arr = {"actual": d["actual"], "units": np.array(d["units"]),
           "models": np.array(list(d["preds"]))}
    for m, p in d["preds"].items():
        # keep each array's own dtype (base-model preds are float32): Wilcoxon ties depend on the last bits
        arr[f"pred::{m}"] = np.asarray(p)
        arr[f"actual::{m}"] = np.asarray(d["actuals"][m])
        arr[f"train_time::{m}"], arr[f"infer_time::{m}"] = np.array(d["times"][m][0]), np.array(d["times"][m][1])
    tmp = os.path.join(pdir, f"{safe_name(name)}.tmp.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, os.path.join(pdir, f"{safe_name(name)}.npz"))   # atomic: never a half-written file


def load_company(rc: RunCfg, name: str):
    """Saved preds -> (metrics rows, per-company dict). Metrics are recomputed exactly as record() does."""
    f = os.path.join(rc.out_dir, "preds", f"{safe_name(name)}.npz")
    if not os.path.exists(f):
        return None
    with np.load(f, allow_pickle=False) as z:
        models = [str(m) for m in z["models"]]
        preds = {m: z[f"pred::{m}"] for m in models}
        actuals = {m: z[f"actual::{m}"] for m in models}
        times = {m: (float(z[f"train_time::{m}"]), float(z[f"infer_time::{m}"])) for m in models}
        d = {"actual": z["actual"], "preds": preds, "actuals": actuals, "times": times,
             "units": [int(u) for u in z["units"]]}
    rows = [result_row(name, m, actuals[m], preds[m], *times[m]) for m in models]
    return pd.DataFrame(rows), d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kaggle", "yahoo"], default="kaggle",
                    help="which prepared dataset --data defaults to: kaggle = prepare_data.py "
                         "(data/prepared), yahoo = prepare_data_yahoo.py (data/prepared_yahoo)")
    ap.add_argument("--data", default=None, help="prepared-data folder (overrides --source)")
    ap.add_argument("--run-dir", default="runs/default")
    ap.add_argument("--market", choices=["saudi", "us", "both"], default="saudi")
    ap.add_argument("--companies", nargs="*", default=None, help="subset of company names / tickers")
    ap.add_argument("--train-config", choices=["repo", "paper"], default="repo")
    ap.add_argument("--stacking", choices=["val", "train", "oof"], default="val",
                    help="meta-learner training data (val=final repo, train=older notebook, oof=paper text)")
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--units", default="repo",
                    help="'repo' (fixed Saudi table; US falls back to grid), 'grid', or an integer")
    ap.add_argument("--unit-grid", type=int, nargs="*", default=None)
    ap.add_argument("--fix-quirks", action="store_true",
                    help="rolling ARIMA forecasts + leak-free / ffilled GRU(256) benchmark")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-wilcoxon", action="store_true")
    ap.add_argument("--wx-mode", choices=["repo", "horizon"], default="repo")
    ap.add_argument("--wx-models", choices=["all", "paper"], default="paper",
                    help="'paper' = the 16 models of Table 11")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true", help="tf op determinism (slower)")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (2 epochs, 8 units)")
    ap.add_argument("--quiet", action="store_true", help="no per-epoch training progress lines")
    ap.add_argument("--reuse-base", default=None, metavar="RUN_DIR",
                    help="load base-model preds from RUN_DIR/base_preds instead of training them")
    ap.add_argument("--report-only", action="store_true",
                    help="no training: rebuild metrics, tables and Wilcoxon from saved preds")
    ap.add_argument("--force", action="store_true", help="redo companies that already have outputs")
    a = ap.parse_args(argv)
    if a.data is None:
        a.data = {"kaggle": "data/prepared", "yahoo": "data/prepared_yahoo"}[a.source]
    global PROGRESS
    PROGRESS = not a.quiet
    if a.report_only and (a.force or a.reuse_base):
        ap.error("--report-only cannot be combined with --force or --reuse-base")

    setup_logging(a.run_dir)
    import tensorflow as tf
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()
    LOG.info(f"python {sys.version.split()[0]} | tensorflow {tf.__version__} | numpy {np.__version__} | "
             f"pandas {pd.__version__}")
    LOG.info(f"args: {vars(a)}")

    rc = RunCfg(data_dir=a.data, out_dir=a.run_dir, train=make_train_cfg(a.train_config, a.quick),
                stacking=a.stacking, oof_folds=a.oof_folds, units=a.units, fix_quirks=a.fix_quirks,
                skip_baselines=a.skip_baselines, seed=a.seed, force=a.force, save_models=a.save_models,
                reuse_base=a.reuse_base, report_only=a.report_only)
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
    names_by_market = {
        "saudi": [c for c in SAUDI_COMPANIES if (not a.companies or c in a.companies)],
        "us": [t for t in US_COMPANIES if (not a.companies or t in a.companies)] or US_COMPANIES,
    }

    # run_config.json: args, prepared-data path and the hash of each company's meta.json
    hashes = {}
    for market in markets:
        for n in names_by_market[market]:
            d = company_dir(rc.data_dir, market, n)
            hashes[f"{market}/{n}"] = meta_hash(d) if os.path.exists(os.path.join(d, "meta.json")) else None
    with open(os.path.join(rc.out_dir, "run_config.json"), "w") as f:
        json.dump({"args": vars(a), "prepared": os.path.abspath(rc.data_dir), "prepared_meta_hash": hashes,
                   "train": asdict(rc.train), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)

    for market in markets:
        all_res, per_company = [], {}
        names = names_by_market[market]
        for i, n in enumerate(names, 1):
            got = None if rc.force else load_company(rc, n)
            if got is None:
                LOG.info(f"[{i}/{len(names)}] {n}")
                if rc.report_only:
                    raise SystemExit(f"--report-only: no saved preds for {n} in {rc.out_dir}/preds")
                res, d = run_company(n, company_dir(rc.data_dir, market, n), units_for(market, n),
                                     market == "saudi", rc, a.quick)
                save_company(rc, n, d)
            else:
                LOG.info(f"loaded saved preds for {n}")
                res, d = got
            all_res.append(res)
            per_company[n] = d
        # the paper reports no Wilcoxon for the US market
        market_report(pd.concat(all_res, ignore_index=True), per_company, market, rc,
                      a.wx_mode, a.wx_models, a.skip_wilcoxon or market == "us")

    LOG.info(f"Done. All outputs in {os.path.abspath(rc.out_dir)}")


if __name__ == "__main__":
    main()

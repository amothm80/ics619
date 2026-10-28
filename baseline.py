"""Step 12: remaining baselines. ARIMA (Close only), XGBoost (flattened windows) and GRU-256.

ARIMA:   order (p, 1, q), p, q in 0..2, picked by AIC on train closes; parameters fitted on train
         only, then frozen. Each val/test window forecasts its h closes from the closes up to the
         window's last input row (Kalman filter over the history, no refit).
XGBoost: the same scaled (w, 34) windows, flattened to w*34 columns, -> h scaled closes.
         Fitted on train windows; validation only drives early stopping, as for the RNNs.
GRU-256: Aseeri (2023) benchmark, one GRU(256) layer with the Table 8 settings. sweep.py already
         trains GRU_256 per ticker; it is reused, or trained here if missing.
Test windows are only scored. Forecasts are saved per ticker for the significance tests.
Usage: python baseline.py [TICKER ...]
"""
import argparse
import itertools
import warnings

import joblib
import keras
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

import config
import data
import features
import metrics
import preprocessing
import train

SPLITS = ("val", "test")
BOUNDS = {"val": (config.VAL_START, config.VAL_END), "test": (config.TEST_START, config.TEST_END)}


def window_origins(dates: pd.DatetimeIndex, w: int = config.W, h: int = config.H) -> dict:
    """{split: row index of each window's last input row}, in the same order as preprocessing.prepare."""
    i = np.arange(len(dates) - w - h + 1)
    return {s: preprocessing.split_windows(i, i, dates, *BOUNDS[s])[0] + w - 1 for s in SPLITS}


def arima_forecast(close: np.ndarray, n_train: int, origins: dict, h: int = config.H) -> tuple[tuple, dict]:
    """Pick the order by AIC on close[:n_train], fit there, then forecast h steps from every origin."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {}
        for o in itertools.product(range(3), [1], range(3)):
            fits[o] = ARIMA(close[:n_train], order=o).fit()
            print(f"  ARIMA{o} AIC {fits[o].aic:.1f}")
        order = min(fits, key=lambda o: fits[o].aic)
        res = fits[order].apply(close)  # same parameters, filtered over the whole series
        out = {s: np.stack([res.get_prediction(start=o + 1, end=o + h, dynamic=True).predicted_mean for o in idx])
               for s, idx in origins.items()}
    return order, out


def xgb_forecast(ticker: str, splits: dict) -> tuple[int, dict]:
    """Multi-output XGBoost on flattened windows. Returns (best iteration, {split: scaled forecasts})."""
    flat = {s: (X.reshape(len(X), -1), Y) for s, (X, Y) in splits.items()}
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, tree_method="hist",
                         early_stopping_rounds=train.PATIENCE, random_state=config.SEED)
    print(f"--- {ticker} XGBoost: training (train / val RMSE, scaled, every 10 trees)")
    model.fit(*flat["train"], eval_set=[flat["train"], flat["val"]], verbose=10)
    joblib.dump(model, config.ticker_dir(ticker) / "models" / "XGB.joblib")
    return model.best_iteration, {s: model.predict(flat[s][0]) for s in SPLITS}


def gru256_forecast(ticker: str, splits: dict) -> dict:
    """{split: scaled forecasts} from models/GRU_256.keras, trained first if absent."""
    path = config.ticker_dir(ticker) / "models" / "GRU_256.keras"
    if not path.exists():
        print(f"--- {ticker} GRU_256: training")
        train.fit_and_save("GRU", ticker, 256, splits)
    model = keras.models.load_model(path)
    return {s: model.predict(splits[s][0], verbose=0) for s in SPLITS}


def score_ticker(ticker: str, raw: pd.DataFrame) -> list[dict]:
    """One row per (split, baseline). Saves models and forecasts (price units) under artifacts/<ticker>/."""
    splits = preprocessing.prepare(ticker, raw)
    y_scaler = joblib.load(config.ticker_dir(ticker) / "y_scaler.joblib")
    df = features.add_indicators(data.load_ticker(ticker, raw))  # same rows prepare() windows
    close = df["close"].to_numpy()
    origins = window_origins(df.index)
    y = {s: metrics.to_price(y_scaler, splits[s][1]) for s in SPLITS}
    for s, idx in origins.items():  # origins line up with prepare()'s windows
        assert np.allclose(close[idx[:, None] + np.arange(1, config.H + 1)], y[s])

    order, arima = arima_forecast(close, len(df.loc[:config.TRAIN_END]), origins)
    best_it, xgb = xgb_forecast(ticker, splits)
    preds = {
        "ARIMA": arima,
        "XGBoost": {s: metrics.to_price(y_scaler, p) for s, p in xgb.items()},
        "GRU-256": {s: metrics.to_price(y_scaler, p) for s, p in gru256_forecast(ticker, splits).items()},
    }
    np.savez(config.ticker_dir(ticker) / "baselines.npz",
             **{f"{s}_y": y[s] for s in SPLITS}, **{f"{s}_{m}": p[s] for m, p in preds.items() for s in SPLITS})
    print(f"{ticker:8s} ARIMA{order}  XGBoost {best_it + 1} trees")
    return [{"ticker": ticker, "split": s, "model": m, **metrics.evaluate(y[s], p[s])} for m, p in preds.items() for s in SPLITS]


def ranking(res: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Every model so far, by mean per-ticker RMSE rank (1 = best). Ranks, because price levels differ by ticker."""
    rw = pd.read_csv(config.ARTIFACTS_DIR / "baseline_random_walk.csv").assign(model="Random walk")
    parts = [res, rw] + [pd.read_csv(config.ARTIFACTS_DIR / f) for f in ("ensembles_avg.csv", "ensembles_stk.csv")
                         if (config.ARTIFACTS_DIR / f).exists()]
    allres = pd.concat(parts)
    allres = allres[(allres.split == split) & allres.ticker.isin(res.ticker.unique())]
    rmse = allres.pivot(index="ticker", columns="model", values="RMSE")
    return pd.DataFrame({"mean_rank": rmse.rank(axis=1).mean(), "mean_RMSE": rmse.mean(), "MRE": allres.groupby("model").MRE.mean()}
                        ).sort_values("mean_rank")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", default=list(config.TICKERS))
    args = p.parse_args()

    raw = data.load_raw()
    res = pd.DataFrame([row for t in args.tickers for row in score_ticker(t, raw)])
    out = config.ARTIFACTS_DIR / "baselines.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    for s in SPLITS:
        print(f"\n{s} RMSE (price units):")
        print(res[res.split == s].pivot(index="ticker", columns="model", values="RMSE").to_string())
    print("\ntest ranking, all models:")
    print(ranking(res).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Steps 5-6: per-ticker scaling (fit on train only) and sliding windows of w inputs -> h targets.

Usage: python preprocessing.py   (prepares every ticker, saves scalers, runs the checks)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import data
import features

FEATURE_ORDER = config.ARTIFACTS_DIR / "feature_order.json"


def load_feature_order() -> list[str]:
    return json.loads(FEATURE_ORDER.read_text())["features"]


def fit_scalers(train: pd.DataFrame, cols: list[str]) -> tuple[StandardScaler, StandardScaler]:
    """One scaler for the features X, one for the target Close. Train rows only."""
    x_scaler = StandardScaler().fit(train[cols].to_numpy())
    y_scaler = StandardScaler().fit(train[["close"]].to_numpy())
    return x_scaler, y_scaler


def save_scalers(ticker: str, x_scaler, y_scaler, cols: list[str]) -> None:
    """Persist the scalers with the feature order they expect."""
    path = config.ticker_dir(ticker)
    joblib.dump(x_scaler, path / "x_scaler.joblib")
    joblib.dump(y_scaler, path / "y_scaler.joblib")
    (path / "features.json").write_text(json.dumps(cols, indent=2))


def make_windows(X: np.ndarray, y: np.ndarray, w: int = config.W, h: int = config.H):
    """Stride-1 windows: inputs X[i : i+w], targets y[i+w : i+w+h].

    Window i sees rows up to i+w-1 (its "today") and predicts the next h values.
    Returns X of shape (n, w, n_features) and Y of shape (n, h), n = len(X) - w - h + 1.
    """
    n = len(X) - w - h + 1
    Xw = np.stack([X[i:i + w] for i in range(n)])
    Yw = np.stack([y[i + w:i + w + h] for i in range(n)])
    return Xw, Yw


def split_windows(Xw, Yw, dates: pd.DatetimeIndex, start, end, w: int = config.W, h: int = config.H):
    """Windows whose h targets all fall in [start, end]. Inputs may reach back into the previous split."""
    i = np.arange(len(Xw))
    first, last = dates[i + w], dates[i + w + h - 1]
    keep = (first >= pd.Timestamp(start)) & (last <= pd.Timestamp(end))
    return Xw[keep], Yw[keep]


def prepare(ticker: str, raw: pd.DataFrame | None = None) -> dict:
    """Load -> indicators -> split -> fit/save scalers -> window. Returns {split: (X, Y)} in scaled units."""
    cols = load_feature_order()
    df = features.add_indicators(data.load_ticker(ticker, raw))[cols]
    train, _, _ = data.split(df)
    x_scaler, y_scaler = fit_scalers(train, cols)
    save_scalers(ticker, x_scaler, y_scaler, cols)

    # Transform the whole series with the train-fitted scalers, window once, then assign each
    # window to the split that holds all of its targets.
    Xw, Yw = make_windows(x_scaler.transform(df.to_numpy()), y_scaler.transform(df[["close"]].to_numpy())[:, 0])
    bounds = {
        "train": (df.index[0], config.TRAIN_END),
        "val": (config.VAL_START, config.VAL_END),
        "test": (config.TEST_START, config.TEST_END),
    }
    return {name: split_windows(Xw, Yw, df.index, *b) for name, b in bounds.items()}


def check_scaling(ticker: str) -> None:
    """Train X is ~N(0, 1) per column; val/test are not; inverse_transform recovers Close."""
    cols = json.loads((config.ticker_dir(ticker) / "features.json").read_text())
    x_scaler = joblib.load(config.ticker_dir(ticker) / "x_scaler.joblib")
    y_scaler = joblib.load(config.ticker_dir(ticker) / "y_scaler.joblib")
    df = features.add_indicators(data.load_ticker(ticker))[cols]
    train, val, test = data.split(df)

    Xt = x_scaler.transform(train.to_numpy())
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-6) and np.allclose(Xt.std(axis=0), 1, atol=1e-6)
    for name, part in (("val", val), ("test", test)):
        Xp = x_scaler.transform(part.to_numpy())
        print(f"  {name:4s} X mean {Xp.mean():+.2f}  std {Xp.std(axis=0).mean():.2f}")

    close = df[["close"]].to_numpy()
    assert np.allclose(y_scaler.inverse_transform(y_scaler.transform(close)), close)


def check_windows() -> None:
    """Hand-made series where row t holds t, so every value names its own row."""
    T, w, h = 12, 5, 5
    X = np.arange(T, dtype=float)[:, None] * np.array([1.0, 10.0])  # 2 features: t and 10t
    y = 100.0 + np.arange(T)                                         # target at row t is 100+t
    Xw, Yw = make_windows(X, y, w, h)
    assert Xw.shape == (T - w - h + 1, w, 2) and Yw.shape == (T - w - h + 1, h)
    assert Xw[0, :, 0].tolist() == [0, 1, 2, 3, 4]            # inputs: rows 0..4
    assert Yw[0].tolist() == [105, 106, 107, 108, 109]         # targets: rows 5..9
    assert Xw[-1, -1, 0] == 6 and Yw[-1, -1] == 111            # last window ends on the last row

    # Rows 0-5 "train", 6-11 "val": every val target lies in val; inputs may start in train.
    dates = pd.date_range("2020-01-01", periods=T)
    Xv, Yv = split_windows(Xw, Yw, dates, dates[6], dates[-1], w, h)
    assert len(Xv) == 2 and Yv[0, 0] == 106 and Xv[0, 0, 0] == 1


if __name__ == "__main__":
    check_windows()
    print("make_windows: hand-made check passed")
    raw = data.load_raw()
    for t in config.TICKERS:
        splits = prepare(t, raw)
        print(t, {k: (X.shape, Y.shape) for k, (X, Y) in splits.items()})
        check_scaling(t)

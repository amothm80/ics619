"""Step 7: error metrics on inverse-scaled prices, and the random-walk-with-drift baseline.

The random walk fits nothing, so it cannot leak validation or test data. It sees exactly
the same input windows as the learned models and forecasts the same h closes.
Usage: python metrics.py   (scores the baseline on val and test for every ticker)
"""
import joblib
import numpy as np
import pandas as pd

import config
import preprocessing


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def mre(y, p):
    """Mean relative error, |y - p| / |y|, as a fraction."""
    return float(np.mean(np.abs(y - p) / np.abs(y)))


def evaluate(y, p) -> dict:
    """All four metrics over every window and horizon step. y, p are prices, shape (n, h)."""
    return {"MAE": mae(y, p), "MSE": mse(y, p), "RMSE": rmse(y, p), "MRE": mre(y, p)}


def to_price(y_scaler, Y: np.ndarray) -> np.ndarray:
    """Undo the train-fitted Close scaler on an (n, h) array."""
    return y_scaler.inverse_transform(Y.reshape(-1, 1)).reshape(Y.shape)


def random_walk_drift(close_window: np.ndarray, h: int = config.H) -> np.ndarray:
    """Last close plus k times the mean daily change across the input window, k = 1..h.

    close_window: (n, w) closes in price units. Returns (n, h) forecasts.
    """
    last = close_window[:, -1:]
    drift = (close_window[:, -1:] - close_window[:, :1]) / (close_window.shape[1] - 1)
    return last + drift * np.arange(1, h + 1)


def check_random_walk() -> None:
    """A straight line is forecast exactly; a flat series stays flat."""
    line = np.array([[10.0, 11, 12, 13, 14]])
    assert random_walk_drift(line).tolist() == [[15, 16, 17, 18, 19]]
    assert random_walk_drift(np.full((1, 5), 7.0)).tolist() == [[7.0] * 5]
    assert evaluate(line, line) == {"MAE": 0, "MSE": 0, "RMSE": 0, "MRE": 0}


def score_random_walk(ticker: str, raw: pd.DataFrame | None = None) -> dict:
    """{split: metrics} for the random walk on this ticker's val and test windows."""
    splits = preprocessing.prepare(ticker, raw)
    y_scaler = joblib.load(config.ticker_dir(ticker) / "y_scaler.joblib")
    close_idx = preprocessing.load_feature_order().index("close")
    out = {}
    for name in ("val", "test"):
        X, Y = splits[name]
        close_window = to_price(y_scaler, X[:, :, close_idx])  # same train scaler as the target
        out[name] = evaluate(to_price(y_scaler, Y), random_walk_drift(close_window))
    return out


if __name__ == "__main__":
    check_random_walk()
    print("random_walk_drift: hand-made check passed")
    import data

    raw = data.load_raw()
    rows = [
        {"ticker": t, "split": s, **m}
        for t in config.TICKERS
        for s, m in score_random_walk(t, raw).items()
    ]
    res = pd.DataFrame(rows).set_index(["ticker", "split"])
    pd.set_option("display.float_format", "{:.4f}".format)
    print(res)
    print("\nmean over tickers:\n", res.groupby("split").mean())
    out = config.ARTIFACTS_DIR / "baseline_random_walk.csv"
    res.to_csv(out)
    print(f"saved {out}")

"""Step 13: significance testing and results tables on the test split.

Per ticker, every model's test forecasts are cut into non-overlapping 5-step windows (every H-th
stride-1 window), giving one RMSE per window. Each pair of models is compared with a Wilcoxon
signed-rank test on those paired RMSEs; the p-values of all pairs within a ticker are
Holm-Bonferroni corrected. A pair is a win for the model with the lower mean RMSE when the
corrected p < ALPHA, otherwise a tie. Counts are summed over tickers.
Nothing is fitted here. Forecasts come from forecasts.npz (base models), baselines.npz, the saved
stacking meta-learners and the random walk; their RMSEs are checked against the earlier CSVs.
Usage: python rankings.py [TICKER ...]
"""
import argparse
import itertools

import joblib
import keras
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import config
import data
import ensembles
import metrics
import preprocessing
import stacking

ALPHA = 0.05


def test_forecasts(ticker: str, raw: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Truth and {model: (n, H) test forecasts} in price units, for every model of Steps 7-12."""
    tdir = config.ticker_dir(ticker)
    base, bl = np.load(tdir / "forecasts.npz"), np.load(tdir / "baselines.npz")
    y = base["test_y"]
    assert np.allclose(y, bl["test_y"]), "base and baseline forecasts are on different windows"
    preds = {k: base[f"test_{k}"] for k in ensembles.KINDS}
    preds |= {ensembles.name(c): np.mean([preds[k] for k in c], axis=0) for c in ensembles.COMBOS}

    y_scaler = joblib.load(tdir / "y_scaler.joblib")
    scaled = {k: y_scaler.transform(preds[k].reshape(-1, 1)).reshape(y.shape) for k in ensembles.KINDS}
    for c in ensembles.COMBOS:  # meta-learners take (samples, H, n_base) scaled base forecasts
        for meta in stacking.META:
            model = keras.models.load_model(tdir / "models" / f"{stacking.name(meta, c)}.keras")
            p = model.predict(np.stack([scaled[k] for k in c], axis=-1), verbose=0)
            preds[stacking.name(meta, c)] = metrics.to_price(y_scaler, p)

    preds |= {m: bl[f"test_{m}"] for m in ("ARIMA", "XGBoost", "GRU-256")}
    X = preprocessing.prepare(ticker, raw)["test"][0]
    close_window = metrics.to_price(y_scaler, X[:, :, preprocessing.load_feature_order().index("close")])
    preds["Random walk"] = metrics.random_walk_drift(close_window)
    return y, preds


def check_against_csvs(ticker: str, y: np.ndarray, preds: dict) -> None:
    """Recomputed test RMSEs must match what Steps 7-12 saved."""
    saved = pd.concat([pd.read_csv(config.ARTIFACTS_DIR / f) for f in ("ensembles_avg.csv", "ensembles_stk.csv", "baselines.csv")])
    rw = pd.read_csv(config.ARTIFACTS_DIR / "baseline_random_walk.csv").assign(model="Random walk")
    saved = pd.concat([saved, rw]).query("ticker == @ticker and split == 'test'").set_index("model")["RMSE"]
    for m, p in preds.items():
        assert np.isclose(metrics.rmse(y, p), saved[m], rtol=1e-4), f"{ticker} {m}: RMSE differs from saved"


def window_rmse(y: np.ndarray, p: np.ndarray, h: int = config.H) -> np.ndarray:
    """RMSE of each non-overlapping h-step window: stride-1 windows 0, h, 2h, ..."""
    return np.sqrt(((y[::h] - p[::h]) ** 2).mean(axis=1))


def pairwise(ticker: str, rmse: dict) -> pd.DataFrame:
    """Wilcoxon on every model pair, Holm-corrected within the ticker. winner is None for a tie."""
    rows = []
    for a, b in itertools.combinations(rmse, 2):
        d = rmse[a] - rmse[b]
        p = 1.0 if np.allclose(d, 0) else wilcoxon(rmse[a], rmse[b]).pvalue  # identical models: no difference
        rows.append({"ticker": ticker, "a": a, "b": b, "mean_a": rmse[a].mean(), "mean_b": rmse[b].mean(), "p": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = multipletests(res.p, method="holm")[1]
    res["winner"] = np.where(res.p_holm >= ALPHA, None, np.where(res.mean_a < res.mean_b, res.a, res.b))
    return res


def win_loss(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per model, summed over tickers: wins, losses, ties, and win rate = wins / (wins + losses)."""
    rows = []
    for m in pd.unique(pairs[["a", "b"]].values.ravel()):
        mine = pairs[(pairs.a == m) | (pairs.b == m)]
        wins, ties = (mine.winner == m).sum(), mine.winner.isna().sum()
        losses = len(mine) - wins - ties
        rows.append({"model": m, "wins": wins, "losses": losses, "ties": ties,
                     "win_rate": wins / (wins + losses) if wins + losses else np.nan})
    return pd.DataFrame(rows).set_index("model").sort_values(["win_rate", "wins"], ascending=False)


def check_pairwise() -> None:
    """A model that is always 1 better wins; two copies of a model tie."""
    rng = np.random.default_rng(0)
    r = rng.uniform(1, 2, 50)
    res = pairwise("toy", {"good": r, "bad": r + 1, "copy": r.copy()})
    assert set(res.winner.dropna()) == {"good", "copy"} and res.winner.isna().sum() == 1
    wl = win_loss(res)
    assert wl.loc["good", ["wins", "losses", "ties"]].tolist() == [1, 0, 1] and wl.loc["bad", "win_rate"] == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", default=list(config.TICKERS))
    args = p.parse_args()
    check_pairwise()

    raw = data.load_raw()
    pairs, metric_rows = [], []
    for t in args.tickers:
        y, preds = test_forecasts(t, raw)
        check_against_csvs(t, y, preds)
        rmse = {m: window_rmse(y, f) for m, f in preds.items()}
        pairs.append(pairwise(t, rmse))
        metric_rows += [{"ticker": t, "model": m, **metrics.evaluate(y, f)} for m, f in preds.items()]
        print(f"{t:8s} {len(preds)} models, {len(rmse['GRU'])} non-overlapping test windows")
    pairs = pd.concat(pairs, ignore_index=True)
    wl = win_loss(pairs)
    table = pd.DataFrame(metric_rows).groupby("model")[["MAE", "MSE", "RMSE", "MRE"]].mean()
    table = wl.join(table)

    pairs.to_csv(config.ARTIFACTS_DIR / "wilcoxon_pairs.csv", index=False)
    table.to_csv(config.ARTIFACTS_DIR / "results_test.csv")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\ntest: Wilcoxon win/loss over {len(args.tickers)} tickers (Holm, alpha {ALPHA}), mean metrics over tickers")
    print(table.to_string())
    print("saved artifacts/wilcoxon_pairs.csv, artifacts/results_test.csv")


if __name__ == "__main__":
    main()

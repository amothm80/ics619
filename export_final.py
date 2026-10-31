"""Step 14: pick each ticker's final model on test RMSE and export it to final_model/ for the web app.

Candidates are every scored model except ARIMA (excluded by project decision), the random walk and
XGBoost (not Keras, so final_model/forecaster.py cannot run them; neither is close to the best).
The model with the lowest mean test RMSE (all stride-1 windows) is copied, with its scalers and
feature order, to final_model/<ticker>/. The test set chose these models, so their test scores
are optimistic. The export then reloads every bundle through forecaster.py and checks that it
reproduces the saved test RMSE, and how much a short input history moves the forecast.
Usage: DATA_SOURCE=yfinance_recent python export_final.py [TICKER ...]
"""
import argparse
import importlib
import json
import shutil
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd

import config
import data
import ensembles
import features
import metrics
import preprocessing
import stacking

OUT = config.ROOT / "final_model"
EXCLUDED = {"ARIMA", "Random walk", "XGBoost"}
MIN_HISTORY_ROWS = 250  # about one trading year; the forecast drift this causes is checked below
PACKAGES = ["tensorflow", "keras", "numpy", "pandas", "pandas-ta-classic", "scikit-learn", "joblib"]


def scores() -> pd.DataFrame:
    """Every model's val and test metrics, one row per (ticker, split, model)."""
    files = ("ensembles_avg.csv", "ensembles_stk.csv", "baselines.csv")
    rw = pd.read_csv(config.ARTIFACTS_DIR / "baseline_random_walk.csv").assign(model="Random walk")
    return pd.concat([pd.read_csv(config.ARTIFACTS_DIR / f) for f in files] + [rw]).drop(columns="epochs_run")


def spec(ticker: str, model: str) -> dict:
    """kind, member model paths and meta-learner path (relative to the ticker dir) of a named model."""
    chosen = {k: v["model"] for k, v in json.loads((config.ticker_dir(ticker) / "config.json").read_text())["models"].items()}
    specs = {k: ("single", [chosen[k]], None) for k in ensembles.KINDS}
    specs["GRU-256"] = ("single", ["models/GRU_256.keras"], None)
    for c in ensembles.COMBOS:
        specs[ensembles.name(c)] = ("average", [chosen[k] for k in c], None)
        for meta in stacking.META:
            specs[stacking.name(meta, c)] = ("stacking", [chosen[k] for k in c], f"models/{stacking.name(meta, c)}.keras")
    kind, members, meta = specs[model]
    return {"kind": kind, "members": members, "meta": meta}


def export_ticker(ticker: str, res: pd.DataFrame) -> dict:
    """Copy the test-best model's files to final_model/<ticker>/ and write its model.json."""
    mine = res[res.ticker == ticker].set_index(["split", "model"])[["MAE", "MSE", "RMSE", "MRE"]]
    test = mine.loc["test"].drop(index=EXCLUDED).sort_values("RMSE")
    best = test.index[0]
    s = spec(ticker, best)

    src, dst = config.ticker_dir(ticker), OUT / ticker
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for f in ("x_scaler.joblib", "y_scaler.joblib", "features.json"):
        shutil.copy(src / f, dst / f)
    for f in s["members"]:
        shutil.copy(src / f, dst / f.split("/")[-1])
    if s["meta"]:
        shutil.copy(src / s["meta"], dst / "meta.keras")

    record = {
        "ticker": ticker,
        "symbol": config.TICKERS[ticker],
        "model": best,
        "kind": s["kind"],
        "members": [f.split("/")[-1] for f in s["members"]],
        "meta": "meta.keras" if s["meta"] else None,
        "test": mine.loc[("test", best)].to_dict(),
        "val": mine.loc[("val", best)].to_dict(),
        "runner_up": {"model": test.index[1], "test_RMSE": test.RMSE.iloc[1]},
        "arima_test_RMSE": mine.loc[("test", "ARIMA"), "RMSE"],  # reference only, never selected
    }
    (dst / "model.json").write_text(json.dumps(record, indent=2))
    return record


def write_manifest(records: dict) -> None:
    """Settings shared by every ticker: data contract, features, windows, split dates, packages."""
    manifest = {
        "source": config.SOURCE,
        "selection": "lowest mean test RMSE per ticker; ARIMA, random walk and XGBoost excluded",
        "w": config.W,
        "h": config.H,
        "features": preprocessing.load_feature_order(),
        "indicator_length": features.LENGTH,
        "input_columns": data.COLUMNS,
        "min_history_rows": MIN_HISTORY_ROWS,
        "train_end": config.TRAIN_END,
        "val": [config.VAL_START, config.VAL_END],
        "test": [config.TEST_START, config.TEST_END],
        "packages": {p: version(p) for p in PACKAGES},
        "tickers": {t: {"symbol": r["symbol"], "yahoo": f"{r['symbol']}.SR", "model": r["model"], "kind": r["kind"]}
                    for t, r in records.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def verify(ticker: str, raw: pd.DataFrame, forecaster, saved_rmse: float) -> float:
    """Reproduce the saved test RMSE through forecaster.py; return the largest relative forecast change
    when only the last MIN_HISTORY_ROWS rows are given, over the last 20 test origins."""
    df = data.load_ticker(ticker, raw)
    X = forecaster.indicators(forecaster.clean(df))
    pd.testing.assert_frame_equal(X, features.add_indicators(df)[X.columns], check_freq=False)

    bundle = forecaster.load(ticker)
    Xw, Yw = preprocessing.make_windows(bundle["x_scaler"].transform(X.to_numpy()), X["close"].to_numpy())
    Xt, Yt = preprocessing.split_windows(Xw, Yw, X.index, config.TEST_START, config.TEST_END)
    rmse = metrics.rmse(Yt, forecaster.predict_windows(bundle, Xt))
    assert np.isclose(rmse, saved_rmse, rtol=1e-4), f"{ticker}: test RMSE {rmse:.6f} != saved {saved_rmse:.6f}"

    drift = 0.0
    for end in range(len(df) - 20, len(df)):
        full = forecaster.forecast(bundle, df.iloc[:end]).close.to_numpy()
        short = forecaster.forecast(bundle, df.iloc[end - MIN_HISTORY_ROWS:end]).close.to_numpy()
        drift = max(drift, float(np.max(np.abs(short - full) / full)))
    return drift


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", default=list(config.TICKERS))
    args = p.parse_args()

    res = scores()
    OUT.mkdir(exist_ok=True)
    records = {t: export_ticker(t, res) for t in args.tickers}
    write_manifest(records)

    sys.path.insert(0, str(OUT))
    forecaster = importlib.import_module("forecaster")  # reads the manifest just written
    raw = data.load_raw()
    print(f"{'ticker':8s} {'model':16s} {'test RMSE':>9s} {'runner-up':>16s} {'ARIMA':>7s} {'short-history drift':>20s}")
    for t, r in records.items():
        drift = verify(t, raw, forecaster, r["test"]["RMSE"])
        print(f"{t:8s} {r['model']:16s} {r['test']['RMSE']:9.4f} {r['runner_up']['model']:>16s} "
              f"{r['arima_test_RMSE']:7.4f} {drift:19.4%}")
    print(f"\nall bundles reproduce their saved test RMSE; exported to {OUT}")


if __name__ == "__main__":
    main()

"""Step 10: averaging ensembles. Equal-weight mean of the saved base-model forecasts.

Nothing is fitted here: the members are the validation-selected models in each ticker's
config.json, and the weights are fixed at 1/k. Test windows are only scored, never used to choose.
Forecasts are saved per ticker so later steps (stacking, significance tests) reuse them.
Usage: python ensembles.py [TICKER ...]
"""
import argparse
import json

import joblib
import keras
import numpy as np
import pandas as pd

import config
import metrics
import preprocessing

KINDS = ["BiRNN", "GRU", "LSTM"]
SHORT = {"BiRNN": "B", "GRU": "G", "LSTM": "L"}
COMBOS = [("BiRNN", "GRU", "LSTM"), ("BiRNN", "GRU"), ("BiRNN", "LSTM"), ("GRU", "LSTM")]
SPLITS = ("val", "test")


def name(combo) -> str:
    return f"AVG({','.join(SHORT[k] for k in combo)})"


def forecast(ticker: str, raw: pd.DataFrame | None = None) -> dict:
    """Base forecasts and truth in price units: {split: {"y": (n, h), kind: (n, h)}}. Saved to forecasts.npz."""
    tdir = config.ticker_dir(ticker)
    chosen = json.loads((tdir / "config.json").read_text())["models"]
    splits = preprocessing.prepare(ticker, raw)  # same train-fitted scalers as the models saw
    y_scaler = joblib.load(tdir / "y_scaler.joblib")
    out = {s: {"y": metrics.to_price(y_scaler, splits[s][1])} for s in SPLITS}
    for kind in KINDS:
        model = keras.models.load_model(tdir / chosen[kind]["model"])
        for s in SPLITS:
            out[s][kind] = metrics.to_price(y_scaler, model.predict(splits[s][0], verbose=0))
    np.savez(tdir / "forecasts.npz", **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    return out


def score_ticker(ticker: str, raw: pd.DataFrame | None = None) -> list[dict]:
    """One row per (split, model): the three base models and the four averages."""
    f = forecast(ticker, raw)
    rows = []
    for s in SPLITS:
        preds = {k: f[s][k] for k in KINDS}
        preds |= {name(c): np.mean([f[s][k] for k in c], axis=0) for c in COMBOS}
        rows += [{"ticker": ticker, "split": s, "model": m, **metrics.evaluate(f[s]["y"], p)} for m, p in preds.items()]
    return rows


def check_vs_members(res: pd.DataFrame) -> pd.DataFrame:
    """Per ticker, split and combo: does the average's RMSE beat its best member's?"""
    rmse = res.set_index(["ticker", "split", "model"])["RMSE"]
    rows = []
    for (t, s), r in rmse.groupby(level=["ticker", "split"]):
        r = r.droplevel(["ticker", "split"])
        for c in COMBOS:
            best = min(r[k] for k in c)
            rows.append({"ticker": t, "split": s, "combo": name(c), "avg": r[name(c)], "best_member": best,
                         "beats_best": r[name(c)] <= best})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", default=list(config.TICKERS))
    args = p.parse_args()

    import data

    raw = data.load_raw()
    res = pd.DataFrame([row for t in args.tickers for row in score_ticker(t, raw)])
    out = config.ARTIFACTS_DIR / "ensembles_avg.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    order = KINDS + [name(c) for c in COMBOS]
    for s in SPLITS:
        print(f"\n{s} RMSE (price units):")
        print(res[res.split == s].pivot(index="ticker", columns="model", values="RMSE")[order].to_string())
    print("\nmean over tickers:")
    print(res.groupby(["split", "model"])[["MAE", "RMSE", "MRE"]].mean().unstack("split").loc[order].to_string())

    chk = check_vs_members(res)
    print("\naverage RMSE <= best member's (count of tickers):")
    print(chk.pivot_table(index="combo", columns="split", values="beats_best", aggfunc="sum").to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

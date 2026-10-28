"""Step 9: unit-size sweep for BiRNN, GRU and LSTM on every ticker, scored on validation only.

The 2025 paper with the authors' unit sizes is not in hand, so each architecture tries a small
grid and keeps the size with the lowest validation loss. Test windows are never predicted here.
Each run is saved by train.fit_and_save; a run whose record exists is reused, so a stopped
sweep can resume. Usage: python sweep.py [TICKER ...] [--units 32 64 128 256]
"""
import argparse
import json

import pandas as pd

import config
import models
import preprocessing
import train

UNITS = [32, 64, 128, 256]


def sweep_ticker(ticker: str, units: list[int], raw: pd.DataFrame) -> list[dict]:
    """One row per (architecture, units): validation loss and validation price metrics."""
    splits = preprocessing.prepare(ticker, raw)  # refits and saves this ticker's scalers on train
    rows = []
    for kind in models.LAYERS:
        for u in units:
            path = config.ticker_dir(ticker) / "models" / f"{kind}_{u}.json"
            rec = json.loads(path.read_text()) if path.exists() else {}
            stale = (rec.get("epochs"), rec.get("patience")) != (train.EPOCHS, train.PATIENCE)
            if "best_val_loss" not in rec or stale:  # missing, from an older train.py, or other settings
                print(f"--- {ticker} {kind}_{u}: training")
                rec = train.fit_and_save(kind, ticker, u, splits, names=("val",))
            val = rec["results"]["val"]["model"]
            rows.append({"ticker": ticker, "kind": kind, "units": u, "epochs_run": rec["epochs_run"],
                         "val_loss": rec["best_val_loss"], **{f"val_{k}": v for k, v in val.items()}})
            print(f"{ticker:8s} {kind:5s} {u:4d}  epochs {rec['epochs_run']:3d}  val RMSE {val['RMSE']:.4f}")
    return rows


def save_choice(ticker: str, table: pd.DataFrame) -> dict:
    """Write artifacts/<ticker>/config.json: the chosen units per architecture and the settings used."""
    best = table.loc[table.groupby("kind")["val_loss"].idxmin()]
    choice = {r.kind: {"units": int(r.units), "model": f"models/{r.kind}_{r.units}.keras", "val_RMSE": r.val_RMSE}
              for r in best.itertuples()}
    cfg = {
        "ticker": ticker, "selected_on": "validation loss", "units_grid": sorted(table["units"].unique().tolist()),
        "models": choice, "scalers": ["x_scaler.joblib", "y_scaler.joblib"], "features": "features.json",
        "w": config.W, "h": config.H, "seed": config.SEED, "train_end": config.TRAIN_END,
        "lr": train.LR, "epochs": train.EPOCHS, "patience": train.PATIENCE, "batch_frac": train.BATCH_FRAC,
    }
    (config.ticker_dir(ticker) / "config.json").write_text(json.dumps(cfg, indent=2))
    return choice


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", default=list(config.TICKERS))
    p.add_argument("--units", type=int, nargs="+", default=UNITS)
    args = p.parse_args()

    import data

    raw = data.load_raw()
    rows = []
    for t in args.tickers:
        ticker_rows = sweep_ticker(t, args.units, raw)
        save_choice(t, pd.DataFrame(ticker_rows))
        rows += ticker_rows

    res = pd.DataFrame(rows)
    out = config.ARTIFACTS_DIR / "unit_sweep.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nval RMSE (price units), ticker x architecture x units:")
    print(res.pivot_table(index="ticker", columns=["kind", "units"], values="val_RMSE").to_string())

    best = res.loc[res.groupby(["ticker", "kind"])["val_loss"].idxmin()]
    print("\nbest units per ticker (val RMSE):")
    print(best.pivot(index="ticker", columns="kind", values="units").to_string())
    print(best.pivot(index="ticker", columns="kind", values="val_RMSE").to_string())
    winner = best.loc[best.groupby("ticker")["val_loss"].idxmin(), "kind"]
    print("\nbest architecture per ticker on val:", winner.value_counts().to_dict())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

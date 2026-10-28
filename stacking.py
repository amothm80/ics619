"""Step 11: stacking. Rolling-origin out-of-fold (OOF) base forecasts train the meta-learners.

11a. The first INITIAL share of the training windows is the first fold's past; the rest is cut
     into K chronological blocks. Fold k trains B, G and L on everything before block k and
     predicts block k. (With a short first past, fold 1 learned 2002-04 and extrapolated badly
     into the 2005-06 bubble; those OOF errors, ~40x the later folds', swamped the meta-learners.) The last H-1 windows before block k are dropped from
     its training set, so no training target shares a row with a predicted target. Fold models
     train for a fixed epoch count: the best epoch of the same ticker/architecture in Step 9
     (no data is held out of the fold for early stopping). Only training windows are used.
11b. Each meta-learner (ANN, GRU-128, LSTM-128) maps (samples, H, n_base) OOF forecasts to the
     H closes, with the Table 8 settings (early stopping on validation). At val/test time it is fed
     the forecasts of the Step 9 base models, which were trained on the full training set.
Everything stays in scaled-Close units; metrics are on inverse-scaled prices.
Usage: python stacking.py [TICKER ...] [--folds K]
"""
import argparse
import json

import joblib
import keras
import numpy as np
import pandas as pd

import config
import metrics
import models
import preprocessing
import train
from ensembles import COMBOS, KINDS, SHORT

K = 5
INITIAL = 0.5  # share of training windows every fold trains on
META = {
    "ANN": models.build_ann,
    "GRU": lambda n_base: models.build_rnn("GRU", 128, n_base),
    "LSTM": lambda n_base: models.build_rnn("LSTM", 128, n_base),
}


def name(meta: str, combo) -> str:
    return f"STK_{meta}({','.join(SHORT[k] for k in combo)})"


def folds(n: int, k: int = K, initial: float = INITIAL, h: int = config.H) -> list[tuple[np.ndarray, np.ndarray]]:
    """(fit, predict) window indices per fold. Window i's targets are rows i+w .. i+w+h-1."""
    edges = np.linspace(int(initial * n), n, k + 1).astype(int)
    return [(np.arange(0, a - (h - 1)), np.arange(a, b)) for a, b in zip(edges[:-1], edges[1:])]


def check_folds(fold_list, w: int = config.W, h: int = config.H) -> None:
    """Every OOF target comes from a model that never trained on that target row (or a later one)."""
    seen = set()
    for fit, pred in fold_list:
        last_fit_target = fit[-1] + w + h - 1
        first_pred_target = pred[0] + w
        assert last_fit_target < first_pred_target, "fold trains on a target row it predicts"
        assert not seen & set(pred), "window predicted twice"
        seen |= set(pred)


def best_epoch(ticker: str, kind: str, units: int) -> int:
    """Step 9's kept epoch (lowest validation loss) for this ticker's selected model."""
    hist = json.loads((config.ticker_dir(ticker) / "models" / f"{kind}_{units}.json").read_text())["history"]
    return int(np.argmin(hist["val_loss"])) + 1


def oof_predictions(ticker: str, splits: dict, k: int = K) -> dict:
    """{kind: (n_oof, H) forecasts, "Y": (n_oof, H) truth, "idx": window indices}, scaled units. Cached."""
    path = config.ticker_dir(ticker) / f"oof_k{k}_init{round(INITIAL * 100)}.npz"
    if path.exists():
        return dict(np.load(path))
    chosen = json.loads((config.ticker_dir(ticker) / "config.json").read_text())["models"]
    X, Y = splits["train"]
    fold_list = folds(len(X), k)
    check_folds(fold_list)
    out = {"idx": np.concatenate([p for _, p in fold_list])}
    out["Y"] = Y[out["idx"]]
    for kind in KINDS:
        units = chosen[kind]["units"]
        epochs = best_epoch(ticker, kind, units)
        preds = []
        for i, (fit, pred) in enumerate(fold_list):
            config.set_seeds()
            model = models.build_rnn(kind, units, X.shape[2])
            model.compile(optimizer=keras.optimizers.Adam(train.LR), loss="mse")
            print(f"--- {ticker} OOF {kind} fold {i + 1}/{k}: fit {len(fit)} windows x {epochs} epochs, predict {len(pred)}")
            model.fit(X[fit], Y[fit], epochs=epochs, batch_size=train.batch_size(len(fit)), verbose=2)
            preds.append(model.predict(X[pred], verbose=0))
        out[kind] = np.concatenate(preds)
    np.savez(path, **out)
    return out


def base_predictions(ticker: str, splits: dict) -> dict:
    """{split: {kind: (n, H)}} from the Step 9 models (full training set), scaled units."""
    tdir = config.ticker_dir(ticker)
    chosen = json.loads((tdir / "config.json").read_text())["models"]
    out = {s: {} for s in ("val", "test")}
    for kind in KINDS:
        model = keras.models.load_model(tdir / chosen[kind]["model"])
        for s in out:
            out[s][kind] = model.predict(splits[s][0], verbose=0)
    return out


def stack_ticker(ticker: str, raw: pd.DataFrame | None = None, k: int = K) -> list[dict]:
    """Train the 12 meta-learners on OOF forecasts; score them on val and test."""
    splits = preprocessing.prepare(ticker, raw)
    oof = oof_predictions(ticker, splits, k)
    base = base_predictions(ticker, splits)
    y_scaler = joblib.load(config.ticker_dir(ticker) / "y_scaler.joblib")

    def meta_input(preds: dict, combo) -> np.ndarray:  # (samples, H, n_base)
        return np.stack([preds[c] for c in combo], axis=-1)

    rows = []
    for combo in COMBOS:
        meta_splits = {"train": (meta_input(oof, combo), oof["Y"]), "val": (meta_input(base["val"], combo), splits["val"][1])}
        for meta, build in META.items():
            config.set_seeds()
            model = build(len(combo))
            print(f"--- {ticker} {name(meta, combo)}: training")
            hist = train.train(model, meta_splits)
            model.save(config.ticker_dir(ticker) / "models" / f"{name(meta, combo)}.keras")
            for s in ("val", "test"):
                y = metrics.to_price(y_scaler, splits[s][1])
                p = metrics.to_price(y_scaler, model.predict(meta_input(base[s], combo), verbose=0))
                rows.append({"ticker": ticker, "split": s, "model": name(meta, combo),
                             "epochs_run": len(hist.history["loss"]), **metrics.evaluate(y, p)})
            print(f"{ticker:8s} {name(meta, combo):18s} epochs {len(hist.history['loss']):3d}  "
                  f"val RMSE {rows[-2]['RMSE']:.4f}")
    return rows


def check_vs_avg(res: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE per split: each STK model next to the AVG of the same base models."""
    avg = pd.read_csv(config.ARTIFACTS_DIR / "ensembles_avg.csv")
    avg = avg[avg.ticker.isin(res.ticker.unique())]
    both = pd.concat([res, avg]).groupby(["model", "split"])["RMSE"].mean().unstack("split")
    rows = []
    for combo in COMBOS:
        c = ",".join(SHORT[x] for x in combo)
        for meta in META:
            stk = both.loc[f"STK_{meta}({c})"]
            rows.append({"combo": c, "meta": meta, **{f"{s}_stk": stk[s] for s in ("val", "test")},
                         **{f"{s}_avg": both.loc[f"AVG({c})", s] for s in ("val", "test")}})
    return pd.DataFrame(rows).set_index(["combo", "meta"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", default=list(config.TICKERS))
    p.add_argument("--folds", type=int, default=K)
    args = p.parse_args()

    import data

    raw = data.load_raw()
    res = pd.DataFrame([row for t in args.tickers for row in stack_ticker(t, raw, args.folds)])
    out = config.ARTIFACTS_DIR / "ensembles_stk.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    for s in ("val", "test"):
        print(f"\n{s} RMSE (price units):")
        print(res[res.split == s].pivot(index="ticker", columns="model", values="RMSE").to_string())
    print("\nmean RMSE over tickers, stacking vs averaging the same base models:")
    print(check_vs_avg(res).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

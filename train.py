"""Step 8: train one base model on one ticker with the paper's Table 8 settings.

Weights are fitted on train windows only. Validation loss drives early stopping and picks the
kept weights. Test windows are scored once, after training, and never feed back into it.
Usage: python train.py GRU ADC --units 64
"""
import argparse
import json

import joblib
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import metrics
import models
import preprocessing

LR = 1e-4
EPOCHS = 100
PATIENCE = 20
BATCH_FRAC = 0.10  # batch size as a fraction of the training windows


def batch_size(n_train: int) -> int:
    return max(1, round(BATCH_FRAC * n_train))


def train(model: keras.Model, splits: dict, verbose: int = 2) -> keras.callbacks.History:
    """Adam + MSE; early stopping on val loss, best-val-loss weights restored."""
    X, Y = splits["train"]
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=min(verbose, 1))
    return model.fit(
        X, Y,
        validation_data=splits["val"],
        epochs=EPOCHS,
        batch_size=batch_size(len(X)),
        callbacks=[stop],
        verbose=verbose,  # 2: one line per epoch: loss, val_loss, time
    )


def score(model: keras.Model, splits: dict, ticker: str, names=("val", "test")) -> dict:
    """{split: {model: metrics, random walk: metrics}} on inverse-scaled prices, for each split in names."""
    y_scaler = joblib.load(config.ticker_dir(ticker) / "y_scaler.joblib")
    close_idx = preprocessing.load_feature_order().index("close")
    out = {}
    for name in names:
        X, Y = splits[name]
        y = metrics.to_price(y_scaler, Y)
        out[name] = {
            "model": metrics.evaluate(y, metrics.to_price(y_scaler, model.predict(X, verbose=0))),
            "random_walk": metrics.evaluate(y, metrics.random_walk_drift(metrics.to_price(y_scaler, X[:, :, close_idx]))),
        }
    return out


def plot_loss(history: keras.callbacks.History, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    best = min(range(len(history.history["val_loss"])), key=history.history["val_loss"].__getitem__)
    ax.axvline(best, color="grey", ls="--", lw=1, label=f"kept (epoch {best + 1})")
    ax.set(title=title, xlabel="epoch", ylabel="MSE (scaled)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fit_and_save(kind: str, ticker: str, units: int, splits: dict, verbose: int = 2, names=("val", "test")) -> dict:
    """Seed, build, train, save model + record + loss plot under artifacts/<ticker>/models/. Returns the record."""
    config.set_seeds()  # same start for every run, whatever ran before it
    model = models.build_rnn(kind, units, splits["train"][0].shape[2])
    history = train(model, splits, verbose)

    out = config.ticker_dir(ticker) / "models"
    stem = f"{kind}_{units}"
    model.save(out / f"{stem}.keras")
    record = {
        "kind": kind, "units": units, "seed": config.SEED,
        "lr": LR, "epochs": EPOCHS, "patience": PATIENCE, "batch_size": batch_size(len(splits["train"][0])),
        "epochs_run": len(history.history["loss"]), "best_val_loss": min(history.history["val_loss"]),
        "history": history.history, "results": score(model, splits, ticker, names),
    }
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2))
    plot_loss(history, f"{ticker} {stem}", out / f"{stem}_loss.png")
    return record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=list(models.LAYERS))
    p.add_argument("ticker", choices=list(config.TICKERS))
    p.add_argument("--units", type=int, default=64)
    args = p.parse_args()

    splits = preprocessing.prepare(args.ticker)
    record = fit_and_save(args.kind, args.ticker, args.units, splits)
    print(f"{args.ticker} {args.kind}_{args.units}: {record['epochs_run']} epochs, windows "
          + ", ".join(f"{k} {len(v[0])}" for k, v in splits.items()))
    for name, r in record["results"].items():
        print(f"  {name:4s} RMSE model {r['model']['RMSE']:.4f}  random walk {r['random_walk']['RMSE']:.4f}")


if __name__ == "__main__":
    main()

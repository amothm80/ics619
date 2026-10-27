"""Feature selection: keep indicators with |Pearson r| > 0.8 against Close, on train rows, for all ten tickers.

Runs steps 1-3 per ticker (load/clean, indicators, split) over the full pandas-ta-classic
candidate pool, scores each candidate on the training split only, and writes the surviving
feature order to artifacts/feature_order.json. Usage: python select_features.py
"""
import json
import warnings

import numpy as np
import pandas as pd

import config
import data
import features

THRESHOLD = 0.8
OUT = config.ARTIFACTS_DIR / "feature_order.json"

SKIP = [
    # Not single-series indicators (same list pandas-ta's strategy() skips).
    "above", "above_value", "below", "below_value", "cross", "cross_value",
    "long_run", "short_run", "mavp", "td_seq", "tsignals", "vp", "xsignals",
    # Need a second series, are dispatchers, or return the input unchanged.
    "beta", "correl", "ma", "tos_stdevall",
    # Broken in pandas-ta-classic 0.8.32: NaN after the second row.
    "vidya",
]
# Long/short legs that are NaN by design on alternate regimes; the combined column is kept.
SPARSE = ("psarl_", "psars_", "supertl_", "superts_", "hilol_", "hilos_", "qqel_", "qqes_")


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV plus every pandas-ta-classic indicator (length=5 where it applies).

    Single-output indicators are named after the function ('sma', ...), as in the paper.
    lookahead=False keeps every column causal (uncentred dpo, no forward ichimoku spans),
    so values on the last training rows never use validation prices.
    """
    parts = [df[["open", "high", "low", "close", "volume"]]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in df.ta.indicators(as_list=True, exclude=SKIP):
            out = getattr(df.ta, name)(length=features.LENGTH, lookahead=False)
            if isinstance(out, tuple):  # ichimoku returns (spans, future spans)
                out = out[0]
            if out is not None:
                parts.append(out.rename(name).to_frame() if isinstance(out, pd.Series) else out)
    X = pd.concat(parts, axis=1)
    X.columns = X.columns.str.lower()
    X = X.loc[:, ~X.columns.duplicated() & ~X.columns.str.startswith(SPARSE)]
    # Leading warm-up NaNs stay NaN and are skipped pairwise by corrwith.
    return X.replace([np.inf, -np.inf], np.nan).ffill()


def train_abs_r(ticker: str, raw: pd.DataFrame) -> pd.Series:
    """|r| of every candidate with Close, on this ticker's training split only."""
    df = data.load_ticker(ticker, raw)
    train, _, _ = data.split(candidates(df))
    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns give NaN
        return train.corrwith(train["close"]).abs().drop("close")


def main() -> None:
    raw = data.load_raw()
    r = pd.DataFrame({t: train_abs_r(t, raw) for t in config.TICKERS})
    r = r.fillna(0.0)  # constant or missing column: no correlation
    worst = r.min(axis=1)
    passed = set(worst[worst > THRESHOLD].index) | {"close"}

    # Final order: the paper's 34 (features.FEATURES order) that pass on this data.
    # Extras are recorded, not used: most do not exist in the pandas-ta version the paper used.
    order = [c for c in features.FEATURES if c in passed]
    extra = sorted(passed - set(features.FEATURES))
    missing = [c for c in features.FEATURES if c not in passed]

    print(f"{len(r)} candidates, {len(passed)} pass |r| > {THRESHOLD} on train for all {r.shape[1]} tickers")
    print(f"paper's 34: {len(order)} pass, missing {missing}")
    print(f"extra ({len(extra)}):", " ".join(extra))
    print("weakest kept:", worst[order[4:]].nsmallest(5).round(3).to_dict())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "features": order,
        "threshold": THRESHOLD,
        "train_end": config.TRAIN_END,
        "min_abs_r": {c: round(float(worst.get(c, 1.0)), 4) for c in order},
        "vs_paper": {"missing": missing, "extra": {c: round(float(worst[c]), 4) for c in extra}},
    }, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

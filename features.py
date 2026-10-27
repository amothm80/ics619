"""Step 2: every pandas-ta-classic indicator (length=5 where it applies), warm-up rows dropped.

Each indicator is a fixed causal formula of past prices, so nothing is fitted here and
computing on the full series cannot leak validation or test data into training rows.
"""
import warnings

import numpy as np
import pandas as pd
import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)

LENGTH = 5

OHLC = ["open", "high", "low", "close"]
INDICATORS = [
    "alma", "decay", "dema", "ema", "fwma", "hma", "hl2", "hlc3", "hwma", "jma",
    "kama", "linreg", "median", "midpoint", "midprice", "pwma", "quantile", "rma",
    "sinwma", "sma", "ssf", "swma", "t3", "tema", "trima", "vwap", "vwma", "wcp",
    "wma", "zlma",
]
FEATURES = OHLC + INDICATORS  # fixed column order; persist with the scalers


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """The paper's 34 features (OHLC + 30 indicators, length=5), warm-up rows dropped."""
    parts = [df[OHLC]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in INDICATORS:
            out = getattr(df.ta, name)(length=LENGTH)
            parts.append(out.rename(name))

    X = pd.concat(parts, axis=1)[FEATURES]
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.ffill().dropna()

def check(df: pd.DataFrame, X: pd.DataFrame) -> None:
    """Raise if the feature frame is unusable. df is the step-1 frame X was built from."""
    assert not X.isna().any().any(), "NaNs remain"
    assert np.isfinite(X.to_numpy()).all(), "infinite values remain"
    assert X.index.isin(df.index).all() and X.index.equals(df.index[-len(X):]), "rows not a tail of df"
    assert (X[["open", "high", "low", "close"]] == df.loc[X.index, ["open", "high", "low", "close"]]).all().all()
    # hwma must be a real Holt-Winters MA, not the authors' ta.dema bug.
    assert np.allclose(X["hwma"],  X["dema"]), "hwma differs from the Holt-Winters recursion"
    assert not np.allclose(X["hwma"], X["dema"]), "hwma equals dema"
    assert list(X.columns) == FEATURES, "feature set/order differs from the paper's 34"
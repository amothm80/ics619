"""Load one ticker's daily prices from the Kaggle Tadawul CSV, on trading-day rows only."""
import pandas as pd

import config

COLUMNS = ["open", "high", "low", "close", "volume"]


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the CSV and tidy its headers (they carry trailing spaces)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"volume_traded": "volume"})


def load_ticker(ticker: str, raw: pd.DataFrame | None = None, end=config.TEST_END) -> pd.DataFrame:
    """One ticker, indexed by trading date, with gaps forward-filled.

    No calendar reindex: each row is a day the stock actually traded.
    Zero-volume rows (suspensions, some with close = 0) are dropped, not filled.
    """
    raw = load_raw() if raw is None else raw
    df = raw[raw["symbol"] == int(config.TICKERS[ticker])]
    df = df[(df["date"] <= end) & (df["volume"] > 0)]
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df.set_index("date")[COLUMNS]
    ohl = ["open", "high", "low"]
    df[ohl] = df[ohl].T.fillna(df["close"]).T  # no quote that day: O=H=L=C
    return df   # only missing open/high/low on real trading rows


def check(df: pd.DataFrame) -> None:
    """Raise if the frame breaks the trading-day contract."""
    assert not df.isna().any().any(), "NaNs remain"
    assert df.index.is_monotonic_increasing and df.index.is_unique, "dates not strictly increasing"
    dow = df.index.dayofweek  # Mon=0 ... Sun=6
    # Saudi weekend: Thu-Fri until 2013-06-28, Fri-Sat after (Thursday sessions ran until 2006).
    assert not (dow == 4).any(), "Friday rows found"
    assert not ((dow == 5) & (df.index >= "2013-06-29")).any(), "Saturday rows after weekend change"
    assert not ((dow == 3) & (df.index >= "2006-07-01") & (df.index < "2013-06-29")).any(), "Thursday rows in Thu-Fri weekend era"

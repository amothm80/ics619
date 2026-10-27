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


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test on the paper's dates (bounds inclusive). Never shuffled."""
    train = df.loc[:config.TRAIN_END]
    val = df.loc[config.VAL_START:config.VAL_END]
    test = df.loc[config.TEST_START:config.TEST_END]
    return train, val, test


def check_split(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if the splits overlap, are out of order, or lose rows; print their sizes."""
    assert len(train) and len(val) and len(test), "empty split"
    assert train.index.max() < val.index.min() and val.index.max() < test.index.min(), "splits overlap"
    assert len(train) + len(val) + len(test) == len(df.loc[:config.TEST_END]), "rows lost between splits"
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"{name:5s} {len(part):5d} rows  {part.index.min().date()} .. {part.index.max().date()}")


def check(df: pd.DataFrame) -> None:
    """Raise if the frame breaks the trading-day contract."""
    assert not df.isna().any().any(), "NaNs remain"
    assert df.index.is_monotonic_increasing and df.index.is_unique, "dates not strictly increasing"
    dow = df.index.dayofweek  # Mon=0 ... Sun=6
    # Saudi weekend: Thu-Fri until 2013-06-28, Fri-Sat after (Thursday sessions ran until 2006).
    assert not (dow == 4).any(), "Friday rows found"
    assert not ((dow == 5) & (df.index >= "2013-06-29")).any(), "Saturday rows after weekend change"
    assert not ((dow == 3) & (df.index >= "2006-07-01") & (df.index < "2013-06-29")).any(), "Thursday rows in Thu-Fri weekend era"

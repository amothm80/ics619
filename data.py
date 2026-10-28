"""Load one ticker's daily prices, on trading-day rows only, from the source config.DATA names.

kaggle:   the Kaggle Tadawul CSV (ends 2020-03-05).
yfinance: Yahoo Finance history for <symbol>.SR, cached per ticker in data/yfinance/.
          Prices are split-adjusted (auto_adjust=False: Close, not dividend-adjusted Adj Close).
Usage: python data.py [--refresh]   (downloads missing or, with --refresh, all Yahoo histories,
       then prints coverage and runs the checks for DATA_SOURCE)
"""
import argparse

import pandas as pd

import config

COLUMNS = ["open", "high", "low", "close", "volume"]

# Corporate actions Yahoo has no split record for: symbol -> [(effective date, ratio)]. Prices
# before the date are divided by the ratio, as Yahoo does for its own splits. Found by checking
# Yahoo against the Kaggle file 2010-2020.
MISSING_SPLITS = {
    "6020": [("2017-02-14", 0.6)],  # GACO 10:6 capital reduction (+66% jump on Yahoo)
}


def load_kaggle(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the CSV and tidy its headers (they carry trailing spaces)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"volume_traded": "volume"})


def download_yf(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """Full Yahoo history for one Tadawul symbol, cached as data/yfinance/<symbol>.SR.csv."""
    path = config.YF_DIR / f"{symbol}.SR.csv"
    if refresh or not path.exists():
        import yfinance as yf

        df = yf.Ticker(f"{symbol}.SR").history(period="max", auto_adjust=False, actions=True)
        if df.empty:
            raise SystemExit(f"Yahoo returned no data for {symbol}.SR")
        df.index = df.index.tz_localize(None).normalize().rename("date")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"downloaded {symbol}.SR: {len(df)} rows to {df.index[-1].date()}")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def load_yf(refresh: bool = False) -> pd.DataFrame:
    """Every ticker's Yahoo history in the Kaggle layout: symbol, date, open, high, low, close, volume."""
    parts = []
    for symbol in config.TICKERS.values():
        y = download_yf(symbol, refresh).astype({"Volume": float})
        for day, ratio in MISSING_SPLITS.get(symbol, []):
            before = y.index < day
            y.loc[before, ["Open", "High", "Low", "Close"]] /= ratio
            y.loc[before, "Volume"] *= ratio
        y = y.rename(columns=str.lower)[COLUMNS].reset_index()
        parts.append(y.assign(symbol=int(symbol)))
    return pd.concat(parts, ignore_index=True)


def load_raw() -> pd.DataFrame:
    """All tickers from the configured source, one row per symbol and date."""
    return load_kaggle() if config.DATA == "kaggle" else load_yf()


def load_ticker(ticker: str, raw: pd.DataFrame | None = None, end=config.TEST_END) -> pd.DataFrame:
    """One ticker, indexed by trading date, with gaps forward-filled.

    No calendar reindex: each row is a day the stock actually traded.
    Zero-volume rows (suspensions, some with close = 0; Yahoo's holiday padding) are dropped, not filled.
    """
    raw = load_raw() if raw is None else raw
    df = raw[raw["symbol"] == int(config.TICKERS[ticker])]
    df = df[(df["date"] >= config.START) & (df["date"] <= end) & (df["volume"] > 0)]
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df.set_index("date")[COLUMNS]
    ohl = ["open", "high", "low"]
    df[ohl] = df[ohl].T.fillna(df["close"]).T  # no quote that day: O=H=L=C
    return df   # only missing open/high/low on real trading rows


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test on the configured dates (bounds inclusive). Never shuffled."""
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


def jumps(df: pd.DataFrame, limit: float = 0.105) -> pd.Series:
    """Daily close moves beyond Tadawul's +-10% limit: likely corporate actions missing a split record."""
    r = df["close"].pct_change()
    return r[r.abs() > limit]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="download every Yahoo history again")
    args = p.parse_args()
    if args.refresh or config.DATA == "yfinance":
        load_yf(args.refresh)
    print(f"source {config.SOURCE}: start {config.START}, train to {config.TRAIN_END}, "
          f"val {config.VAL_START}..{config.VAL_END}, test {config.TEST_START}..{config.TEST_END}")
    raw = load_raw()
    for t in config.TICKERS:
        df = load_ticker(t, raw)
        print(f"\n{t}: {len(df)} rows {df.index[0].date()} .. {df.index[-1].date()}")
        check_split(df, *split(df))
        try:
            check(df)
        except AssertionError as e:
            print(f"  check failed: {e}")
        if len(j := jumps(df)):
            print("  moves > 10.5%:", {str(d.date()): round(v, 3) for d, v in j.items()})

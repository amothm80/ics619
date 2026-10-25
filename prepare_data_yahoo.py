#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data_yahoo.py
=====================

Fork of prepare_data.py that builds the same prepared-data folders from up-to-date Yahoo Finance
prices (yfinance) for the 10 TASI companies of the study, instead of the Kaggle snapshot that
ends on 2020-03-05. Feature engineering, scaling and windowing are the prepare_data.py functions,
unchanged; only acquisition, the date range and the split boundaries differ.

Differences from the Kaggle data (all recorded in meta.json["config"]):
  * Yahoo's Tadawul history starts in March 2010 (Kaggle: 2001/2002), so every company starts at
    its first Yahoo row and the default training set is 2010-03 .. 2015-12.
  * Prices: --adjust split (default) = Yahoo's Close, split/bonus-share adjusted to today;
    kaggle = undo the splits after 2020-03-05, which puts prices on the Kaggle file's scale
    (Kaggle was split-adjusted as of its snapshot; e.g. ALRAJHI x2.4, FITAIHI x5 vs Yahoo today);
    total = also dividend-adjusted (Adj Close).
  * Corporate actions missing on Yahoo are applied from MISSING_SPLITS (GACO 2017-02-14).
  * Benchmark raw columns: change / perc_change are computed from close, value_traded is
    approximated as close * volume, no_trades does not exist on Yahoo (the GRU(256) benchmark
    uses whichever columns are present).
  * Split: --train-end / --val-end (default the paper's 2015-12-31 / 2016-12-31); val and test
    start the next day; the test set runs to --end (default: the last date every requested
    company has). Stored in meta.json["split"], which train_report.py reads.

Raw downloads are cached in data/yahoo/{symbol}.csv (unadjusted for dividends, with split
actions); --refresh downloads again.

Outputs: the same files as prepare_data.py, in data/prepared_yahoo/saudi/{company}/.

Usage
-----
  python prepare_data_yahoo.py                                   # paper split, test 2017 -> latest
  python prepare_data_yahoo.py --companies ADC STC --refresh
  python prepare_data_yahoo.py --train-end 2021-12-31 --val-end 2022-12-31 --out data/prepared_yahoo_2022
  python prepare_data_yahoo.py --adjust kaggle --out data/prepared_yahoo_kscale
  python train_report.py --source yahoo --run-dir runs/yahoo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

from prepare_data import (DATA_END, LOG, PREPARED_FILES, SAUDI_COMPANIES, SELECTED_TI_SA, _numeric_daily,
                          all_indicators, company_dir, engineer_saudi, pearson_feature_selection,
                          prepare_split, setup_logging, write_prepared)

# trading_name -> Tadawul symbol (the 'symbol' column of Tadawul_stcks.csv) on Yahoo
YAHOO_TICKERS: Dict[str, str] = {
    "ADC": "2140.SR",
    "ALRAJHI": "1120.SR",
    "GACO": "6020.SR",
    "EMAAR EC": "4220.SR",
    "FITAIHI GROUP": "4180.SR",
    "SARCO": "2030.SR",
    "SAUDI ELECTRICITY": "5110.SR",
    "SPIMACO": "2070.SR",
    "STC": "7010.SR",
    "TASNEE": "2060.SR",
}

# corporate actions Yahoo has no 'Stock Splits' record for: symbol -> [(effective date, ratio)], ratio as
# Yahoo uses it (new shares per old share; prices before the date are divided by it). Found by checking
# Yahoo against the Kaggle file 2010-2020 and scanning for daily moves beyond Tadawul's +-10% limit.
MISSING_SPLITS: Dict[str, List[Tuple[str, float]]] = {
    "6020.SR": [("2017-02-14", 0.6)],     # GACO 10:6 capital reduction (+66% jump on Yahoo; Kaggle adjusts it)
}


# =============================================================================
# ACQUISITION
# =============================================================================

def download_yahoo(symbol: str, cache_dir: str, refresh: bool) -> pd.DataFrame:
    """Full daily history (auto_adjust=False, with Dividends / Stock Splits), cached as CSV."""
    path = os.path.join(cache_dir, f"{symbol}.csv")
    if refresh or not os.path.exists(path):
        try:
            import yfinance as yf
        except ImportError:
            raise SystemExit("yfinance is not installed: pip install yfinance")
        LOG.info(f"Downloading {symbol} from Yahoo Finance ...")
        df = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
        if df.empty:
            raise SystemExit(f"Yahoo returned no data for {symbol}")
        df.index = df.index.tz_localize(None).normalize()
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    LOG.info(f"  {symbol}: {len(df)} rows {df.index[0].date()} .. {df.index[-1].date()} ({path})")
    return df


def to_tadawul_raw(y: pd.DataFrame, company: str, adjust: str) -> pd.DataFrame:
    """Yahoo history -> rows shaped like load_saudi_raw() output (trading days only)."""
    y = y.astype({"Volume": "float64", "Stock Splits": "float64"})
    px = ["Open", "High", "Low", "Close"]
    for date, r in MISSING_SPLITS.get(YAHOO_TICKERS[company], []):
        LOG.info(f"  {company}: applying missing split {r} on {date}")
        before = y.index < pd.Timestamp(date)
        y.loc[before, px + ["Adj Close"]] /= r
        y.loc[before, "Volume"] *= r
        y.loc[y.index[~before][0], "Stock Splits"] = r     # so --adjust kaggle sees it too
    if adjust == "total":
        f = (y["Adj Close"] / y["Close"]).fillna(1.0)
        y[px] = y[px].mul(f, axis=0)
    elif adjust == "kaggle":
        # Yahoo divides all earlier prices by each split ratio; undo the splits after the Kaggle snapshot
        sp = y["Stock Splits"].where(y.index > pd.Timestamp(DATA_END), 0.0).replace(0.0, 1.0)
        f = sp[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
        y[px] = y[px].mul(f, axis=0)
        y["Volume"] = y["Volume"] / f
    # after the adjustment: split actions can sit on the rows dropped here.
    # Yahoo pads some holidays with 0-volume rows
    y = y[y["Close"].notna() & (y["Volume"] > 0)]
    out = pd.DataFrame({
        "trading_name": company,
        "open": y["Open"], "high": y["High"], "low": y["Low"], "close": y["Close"],
        "change": y["Close"].diff(), "perc_change": y["Close"].pct_change() * 100,
        "volume_traded": y["Volume"],
        "value_traded": y["Close"] * y["Volume"],              # approximation (Kaggle: actual turnover)
    }, index=y.index)
    out.index.name = "date"
    return out.sort_index()


def make_split(train_end: str, val_end: str, end: str) -> Dict[str, str]:
    te, ve, en = pd.Timestamp(train_end), pd.Timestamp(val_end), pd.Timestamp(end)
    if not te < ve < en:
        raise SystemExit(f"need train_end < val_end < end, got {te.date()} / {ve.date()} / {en.date()}")
    day = pd.Timedelta(days=1)
    f = lambda t: t.strftime("%Y-%m-%d")   # noqa: E731
    return {"train_end": f(te), "val_start": f(te + day), "val_end": f(ve), "test_start": f(ve + day),
            "end": f(en)}


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--companies", nargs="*", default=None, help="subset of company names")
    ap.add_argument("--data-dir", default="data", help="Yahoo downloads are cached in DATA_DIR/yahoo")
    ap.add_argument("--refresh", action="store_true", help="download again even if cached")
    ap.add_argument("--adjust", choices=["split", "kaggle", "total"], default="split",
                    help="split = Yahoo Close (split-adjusted to today); kaggle = split-adjusted as of "
                         "2020-03-05 like the Kaggle file; total = also dividend-adjusted")
    ap.add_argument("--train-end", default="2015-12-31")
    ap.add_argument("--val-end", default="2016-12-31")
    ap.add_argument("--end", default=None, help="last date (default: last date common to all companies)")
    ap.add_argument("--feature-selection", choices=["fixed", "pearson"], default="fixed")
    ap.add_argument("--pearson-threshold", type=float, default=0.8)
    ap.add_argument("--out", default="data/prepared_yahoo")
    ap.add_argument("--force", action="store_true", help="rebuild companies that are already prepared")
    a = ap.parse_args(argv)

    setup_logging(a.out, "prepare.log")
    LOG.info(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__}")
    LOG.info(f"args: {vars(a)}")
    names = [c for c in SAUDI_COMPANIES if (not a.companies or c in a.companies)]
    unknown = sorted(set(a.companies or []) - set(names))
    if unknown:
        raise SystemExit(f"unknown companies {unknown}; choose from {list(SAUDI_COMPANIES)}")

    cache = os.path.join(a.data_dir, "yahoo")
    raw = {n: to_tadawul_raw(download_yahoo(YAHOO_TICKERS[n], cache, a.refresh), n, a.adjust) for n in names}
    end = a.end or min(r.index[-1] for r in raw.values()).strftime("%Y-%m-%d")
    split = make_split(a.train_end, a.val_end, end)
    starts = {n: max(pd.Timestamp(SAUDI_COMPANIES[n][0]), raw[n].index[0]).strftime("%Y-%m-%d") for n in names}
    for n, st in starts.items():
        if pd.Timestamp(st) >= pd.Timestamp(split["train_end"]):
            raise SystemExit(f"{n}: Yahoo data starts {st}, after --train-end {split['train_end']}")
    LOG.info(f"split: start .. {split['train_end']} | {split['val_start']} .. {split['val_end']} | "
             f"{split['test_start']} .. {split['end']}")

    def done(n):
        d = company_dir(a.out, "saudi", n)
        if a.force or not all(os.path.exists(os.path.join(d, f)) for f in PREPARED_FILES):
            return False
        with open(os.path.join(d, "meta.json")) as f:
            m = json.load(f)
        same = m.get("split") == split and m["config"].get("adjust") == a.adjust
        if not same:
            LOG.info(f"  {n}: prepared with a different split/adjust; rebuilding")
        return same

    todo = [n for n in names if not done(n)]
    if a.feature_selection == "pearson" and todo:
        todo = names
    if not todo:
        LOG.info(f"all {len(names)} companies already prepared in {a.out} (use --force or --refresh)")
        return

    features = list(SELECTED_TI_SA)
    info: Dict = {"selection": a.feature_selection, "source": "yahoo", "adjust": a.adjust, "split": split}
    full: Optional[Dict[str, pd.DataFrame]] = None
    if a.feature_selection == "pearson":
        LOG.info("Pearson feature selection over the full indicator set (train rows only)")
        full = {}
        for n in names:
            co = _numeric_daily(raw[n].drop(columns=["trading_name"]), starts[n], end=end)
            full[n] = co.fillna(0).join(all_indicators(co, "open", "high", "low", "close", "volume_traded"),
                                        rsuffix="_ti")
        features = pearson_feature_selection(full, "close", a.pearson_threshold, split["train_end"])
        LOG.info(f"overlap with Table 4: {len(set(features) & set(SELECTED_TI_SA))}/34")
        info.update({"pearson_threshold": a.pearson_threshold, "companies": names})

    for n in todo:
        LOG.info(f"===== {n} ({YAHOO_TICKERS[n]}) =====")
        st = starts[n]
        co = full[n][features] if full is not None else engineer_saudi(raw[n], n, st, features, end=end)
        rawc = raw[n].loc[:end]
        sp = prepare_split(co, features, "close", split)
        downloaded = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(
            os.path.join(cache, f"{YAHOO_TICKERS[n]}.csv"))))
        config = {"feature_selection": a.feature_selection, "pearson_threshold": a.pearson_threshold,
                  "source": f"yahoo:{YAHOO_TICKERS[n]}", "adjust": a.adjust, "downloaded": downloaded,
                  "missing_splits": MISSING_SPLITS.get(YAHOO_TICKERS[n], [])}
        write_prepared(company_dir(a.out, "saudi", n), n, "saudi", co, rawc, sp, features, "close", st,
                       config, split=split)
    info["features"] = features
    with open(os.path.join(a.out, "saudi", "features.json"), "w") as f:
        json.dump(info, f, indent=1)
    LOG.info(f"Done. Prepared data in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

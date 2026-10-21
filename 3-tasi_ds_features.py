from pathlib import Path

import pandas as pd
import pandas_ta_classic as ta

INPUT_PATH = Path("data/tadawul_preprocessed.csv")
OUTPUT_PATH = Path("data/tadawul_features.csv")

WINDOW = 5

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close",
    "alma", "decay", "dema", "ema", "fwma", "hma",
    "hl2", "hlc3", "hwma", "jma", "kama", "linreg",
    "median", "midpoint", "midprice", "pwma", "quantile",
    "rma", "sinwma", "sma", "ssf", "swma", "t3", "tema",
    "trima", "vwap", "vmwa", "wcp", "wma", "zlma",
]


def add_repository_indicators(company_data: pd.DataFrame) -> pd.DataFrame:
    company_data = company_data.sort_values("Date").set_index("Date").copy()

    open_price = company_data["Open"]
    high = company_data["High"]
    low = company_data["Low"]
    close = company_data["Close"]
    volume = company_data["Volume"]

    company_data["alma"] = ta.alma(close, length=WINDOW)
    company_data["decay"] = ta.decay(close, length=WINDOW)
    company_data["dema"] = ta.dema(close, length=WINDOW)
    company_data["ema"] = ta.ema(close, length=WINDOW)
    company_data["fwma"] = ta.fwma(close, length=WINDOW)
    company_data["hma"] = ta.hma(close, length=WINDOW)

    company_data["hl2"] = ta.hl2(high, low)
    company_data["hlc3"] = ta.hlc3(high, low, close)

    # Kept exactly as implemented in the reference repository:
    # this is DEMA with pandas-ta-classic's default length, despite its hwma name.
    company_data["hwma"] = ta.dema(close)

    company_data["jma"] = ta.jma(close, length=WINDOW)
    company_data["kama"] = ta.kama(close, length=WINDOW)
    company_data["linreg"] = ta.linreg(close, length=WINDOW)
    company_data["median"] = ta.median(close, length=WINDOW)
    company_data["midpoint"] = ta.midpoint(close, length=WINDOW)
    company_data["midprice"] = ta.midprice(high, low, length=WINDOW)
    company_data["pwma"] = ta.pwma(close, length=WINDOW)
    company_data["quantile"] = ta.quantile(close, length=WINDOW)
    company_data["rma"] = ta.rma(close, length=WINDOW)
    company_data["sinwma"] = ta.sinwma(close, length=WINDOW)
    company_data["sma"] = ta.sma(close, length=WINDOW)
    company_data["ssf"] = ta.ssf(close, length=WINDOW)
    company_data["swma"] = ta.swma(close)
    company_data["t3"] = ta.t3(close, length=WINDOW)
    company_data["tema"] = ta.tema(close, length=WINDOW)
    company_data["trima"] = ta.trima(close, length=WINDOW)

    # Requires the Date index created above.
    company_data["vwap"] = ta.vwap(high, low, close, volume)
    company_data["vmwa"] = ta.vwma(close, volume, length=WINDOW)

    company_data["wcp"] = ta.wcp(high, low, close)
    company_data["wma"] = ta.wma(close, length=WINDOW)
    company_data["zlma"] = ta.zlma(close, length=WINDOW)

    # Matches the repository's treatment of indicator warm-up values.
    company_data[FEATURE_COLUMNS] = company_data[FEATURE_COLUMNS].fillna(0)

    return company_data.reset_index()


def main() -> None:
    data = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    required_columns = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    featured_data = pd.concat(
        [
            add_repository_indicators(company_data)
            for _, company_data in data.groupby("Ticker", sort=False)
        ],
        ignore_index=True,
    )

    featured_data = featured_data.sort_values(["Ticker", "Date"])
    featured_data.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")

    print(f"Saved {len(featured_data):,} rows to {OUTPUT_PATH}")
    print(f"Model feature columns: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
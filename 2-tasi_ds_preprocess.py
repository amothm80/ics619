from pathlib import Path
import warnings
import pandas as pd

RAW_PATH = Path("data/tadawul_raw_ohlcv.csv")
OUTPUT_PATH = Path("data/tadawul_preprocessed.csv")

REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_and_preprocess(raw_path: Path, output_path: Path) -> pd.DataFrame:
    data = pd.read_csv(raw_path, parse_dates=["Date"])

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = data[REQUIRED_COLUMNS].copy()
    data["Ticker"] = data["Ticker"].astype(str).str.strip()
    data = data.dropna(subset=["Date", "Ticker"])

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.duplicated(["Date", "Ticker"]).any():
        duplicates = data.loc[
            data.duplicated(["Date", "Ticker"], keep=False),
            ["Date", "Ticker"],
        ]
        raise ValueError(f"Duplicate Date–Ticker records found:\n{duplicates.head()}")

    invalid_ohlcv = (
        (data["Open"] <= 0)
        | (data["High"] <= 0)
        | (data["Low"] <= 0)
        | (data["Close"] <= 0)
        | (data["Volume"] < 0)
        | (data["High"] < data[["Open", "Close", "Low"]].max(axis=1))
        | (data["Low"] > data[["Open", "Close", "High"]].min(axis=1))
    )

    if invalid_ohlcv.any():
        invalid_rows = data.loc[invalid_ohlcv, REQUIRED_COLUMNS]

        warnings.warn(
            f"Found {len(invalid_rows)} OHLCV rows outside the expected "
            f"high/low range. Keeping them unchanged.\n"
            f"{invalid_rows.head()}",
            UserWarning,
            stacklevel=2,
        )

    prepared_frames = []

    for ticker, company_data in data.groupby("Ticker", sort=True):
        company_data = company_data.sort_values("Date").set_index("Date")

        all_dates = pd.date_range(
            start=company_data.index.min(),
            end=company_data.index.max(),
            freq="D",
        )

        # Matches the reference repository's daily reindex + forward-fill method.
        company_data = company_data[NUMERIC_COLUMNS].reindex(all_dates).ffill()

        if company_data.isna().any().any():
            raise ValueError(f"{ticker} has missing values at the start of its history.")

        company_data.insert(0, "Ticker", ticker)
        company_data.index.name = "Date"
        prepared_frames.append(company_data.reset_index())

    prepared = pd.concat(prepared_frames, ignore_index=True)
    prepared = prepared.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    prepared[NUMERIC_COLUMNS] = prepared[NUMERIC_COLUMNS].astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, float_format="%.6f")

    print(f"Saved {len(prepared):,} rows to {output_path}")
    print(prepared.groupby("Ticker").size())

    return prepared


if __name__ == "__main__":
    validate_and_preprocess(RAW_PATH, OUTPUT_PATH)
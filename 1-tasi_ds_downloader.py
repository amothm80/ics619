from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "2140.SR", "1120.SR", "4220.SR", "4180.SR", "6020.SR",
    "2280.SR", "5110.SR", "2070.SR", "7010.SR", "2060.SR",
]

frames = []

for ticker in TICKERS:
    data = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        multi_level_index=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.insert(1, "Ticker", ticker)
    frames.append(data)

raw_data = pd.concat(frames, ignore_index=True)

output_path = Path("data/tadawul_raw_ohlcv.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
raw_data.to_csv(output_path, index=False)

print(f"Saved raw data to: {output_path}")
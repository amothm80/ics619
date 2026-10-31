"""Streamlit app: next 5 trading-day closes for the ten Tadawul tickers, from the final_model/ bundles.

Usage: streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final_model"))
import forecaster  # noqa: E402

HISTORY_DAYS = 120  # trading days of history shown on the chart


@st.cache_resource
def bundle(ticker: str) -> dict:
    return forecaster.load(ticker)


@st.cache_data(ttl=3600)
def prices(ticker: str) -> pd.DataFrame:
    return forecaster.clean(forecaster.download(ticker))


st.set_page_config(page_title="Tadawul 5-day forecast", layout="wide")
st.title("Tadawul 5-day Close forecast")

tickers = forecaster.MANIFEST["tickers"]
ticker = st.sidebar.selectbox("Ticker", list(tickers), format_func=lambda t: f"{t} ({tickers[t]['symbol']})")
if st.sidebar.button("Refresh prices"):
    prices.clear()

spec = json.loads((forecaster.ROOT / ticker / "model.json").read_text())
st.sidebar.markdown(
    f"**Model:** {spec['model']} ({spec['kind']})  \n"
    f"**Test RMSE:** {spec['test']['RMSE']:.3f} SAR  \n"
    f"**Test MRE:** {spec['test']['MRE']:.2%}  \n"
    f"Trained to {forecaster.MANIFEST['train_end']}; picked on test results, so these scores are optimistic."
)

try:
    df = prices(ticker)
except Exception as e:  # network or Yahoo errors
    st.error(f"Could not download prices for {tickers[ticker]['yahoo']}: {e}")
    st.stop()

with st.spinner("Forecasting..."):
    fc = forecaster.forecast(bundle(ticker), df)

last = df["close"].iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric(f"Last close ({df.index[-1].date()})", f"{last:.2f} SAR")
c2.metric(f"Day 1 ({fc.date.iloc[0]})", f"{fc.close.iloc[0]:.2f} SAR", f"{fc.close.iloc[0] / last - 1:+.2%}")
c3.metric(f"Day 5 ({fc.date.iloc[-1]})", f"{fc.close.iloc[-1]:.2f} SAR", f"{fc.close.iloc[-1] / last - 1:+.2%}")

# History and forecast as two series; the forecast starts at the last close so the lines join.
hist = df["close"].iloc[-HISTORY_DAYS:].rename("history")
future = pd.Series(fc.close.to_numpy(), index=pd.to_datetime(fc.date), name="forecast")
future = pd.concat([hist.iloc[-1:].rename("forecast"), future])
st.line_chart(pd.concat([hist, future], axis=1), color=["#1f77b4", "#ff7f0e"])

st.subheader("Forecast")
st.dataframe(fc.rename(columns={"close": "close (SAR)"}).style.format({"close (SAR)": "{:.2f}"}), hide_index=True)
st.caption("Dates are the next Sunday-Thursday days; Tadawul holidays are not accounted for. "
           "Prices: Yahoo Finance, split-adjusted, not dividend-adjusted. Not investment advice.")

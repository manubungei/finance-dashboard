"""Financial Markets Analysis Dashboard — Streamlit app.

Pulls historical prices from Yahoo Finance (free, no API key) and shows
price performance, returns, rolling volatility, correlations and key
risk/return metrics for any tickers you choose.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="Financial Markets Dashboard", layout="wide")

st.title("Financial Markets Analysis Dashboard")
st.caption("Interactive analysis of asset prices, returns, risk and correlations.")

# ---- Sidebar controls ----
st.sidebar.header("Settings")
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", ", ".join(default_tickers))
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
years = st.sidebar.slider("Years of history", 1, 10, 3)

end = date.today()
start = end - timedelta(days=365 * years)


@st.cache_data(ttl=3600)
def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


if not tickers:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()

with st.spinner("Downloading market data..."):
    prices = load_prices(tickers, start, end)

if prices.empty:
    st.error("No data returned — check the ticker symbols.")
    st.stop()

prices = prices.dropna(axis=1, how="all")
returns = prices.pct_change().dropna()


# ---- Key metrics ----
def summary(returns):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return pd.DataFrame({
        "Annual return": ann_ret,
        "Annual volatility": ann_vol,
        "Sharpe (rf=0)": sharpe,
        "Max drawdown": max_dd,
    })


st.subheader("Key metrics")
st.dataframe(
    summary(returns).style.format({
        "Annual return": "{:.1%}",
        "Annual volatility": "{:.1%}",
        "Sharpe (rf=0)": "{:.2f}",
        "Max drawdown": "{:.1%}",
    })
)

# ---- Price performance ----
st.subheader("Price performance (rebased to 100)")
rebased = prices / prices.iloc[0] * 100
st.plotly_chart(
    px.line(rebased, labels={"value": "Rebased price", "variable": "Ticker", "index": "Date"}),
    width="stretch",
)

# ---- Cumulative return ----
st.subheader("Cumulative return")
cum = (1 + returns).cumprod() - 1
fig2 = px.line(cum, labels={"value": "Cumulative return", "variable": "Ticker", "index": "Date"})
fig2.update_yaxes(tickformat=".0%")
st.plotly_chart(fig2, width="stretch")

# ---- Rolling volatility ----
st.subheader("30-day rolling volatility (annualised)")
roll_vol = returns.rolling(30).std() * np.sqrt(252)
fig3 = px.line(roll_vol, labels={"value": "Volatility", "variable": "Ticker", "index": "Date"})
fig3.update_yaxes(tickformat=".0%")
st.plotly_chart(fig3, width="stretch")

# ---- Correlation heatmap ----
if returns.shape[1] > 1:
    st.subheader("Return correlations")
    st.plotly_chart(
        px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu",
                  zmin=-1, zmax=1, aspect="auto"),
        width="stretch",
    )

st.caption("Data: Yahoo Finance via yfinance. Educational use only — not investment advice.")

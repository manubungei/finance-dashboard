"""Financial Markets Analysis Dashboard — Streamlit app.

Pulls historical prices from Yahoo Finance (free, no API key) and shows
price performance, returns, rolling volatility, correlations, key
risk/return metrics, and a benchmark comparison (beta, alpha, correlation)
for any tickers you choose.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="Financial Markets Dashboard", layout="wide")

st.title("Financial Markets Analysis Dashboard")
st.caption("Interactive analysis of asset prices, returns, risk, correlations and market beta.")

TRADING_DAYS = 252

# ---- Sidebar controls ----
st.sidebar.header("Settings")
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", ", ".join(default_tickers))
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
benchmark = st.sidebar.text_input("Benchmark ticker", "^GSPC").strip().upper()
years = st.sidebar.slider("Years of history", 1, 10, 3)

end = date.today()
start = end - timedelta(days=365 * years)


@st.cache_data(ttl=3600)
def load_prices(symbols, start, end):
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    return data.dropna(how="all")


if not tickers:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()

# Fetch the assets and the benchmark together (dedupe if benchmark is also a ticker)
symbols = list(dict.fromkeys(tickers + ([benchmark] if benchmark else [])))
with st.spinner("Downloading market data..."):
    prices = load_prices(symbols, start, end)

if prices.empty:
    st.error("No data returned — check the ticker symbols.")
    st.stop()

prices = prices.dropna(axis=1, how="all")
returns = prices.pct_change().dropna()

asset_cols = [t for t in tickers if t in returns.columns]
has_benchmark = benchmark in returns.columns and benchmark not in tickers
if not asset_cols:
    st.error("None of the asset tickers returned data.")
    st.stop()


# ---- Core risk/return metrics ----
def summary(rets):
    ann_ret = rets.mean() * TRADING_DAYS
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return pd.DataFrame({
        "Annual return": ann_ret,
        "Annual volatility": ann_vol,
        "Sharpe (rf=0)": sharpe,
        "Max drawdown": max_dd,
    })


metrics = summary(returns[asset_cols])

# ---- Benchmark comparison: beta, alpha, correlation ----
if has_benchmark:
    bench_ret = returns[benchmark]
    bench_ann = bench_ret.mean() * TRADING_DAYS
    var_b = bench_ret.var()
    betas, alphas, corrs = {}, {}, {}
    for col in asset_cols:
        beta = np.cov(returns[col], bench_ret)[0, 1] / var_b
        betas[col] = beta
        alphas[col] = returns[col].mean() * TRADING_DAYS - beta * bench_ann  # CAPM alpha, rf=0
        corrs[col] = returns[col].corr(bench_ret)
    metrics["Beta"] = pd.Series(betas)
    metrics["Alpha (ann.)"] = pd.Series(alphas)
    metrics[f"Corr vs {benchmark}"] = pd.Series(corrs)

st.subheader("Key metrics")
fmt = {
    "Annual return": "{:.1%}", "Annual volatility": "{:.1%}",
    "Sharpe (rf=0)": "{:.2f}", "Max drawdown": "{:.1%}",
}
if has_benchmark:
    fmt.update({"Beta": "{:.2f}", "Alpha (ann.)": "{:.1%}", f"Corr vs {benchmark}": "{:.2f}"})
st.dataframe(metrics.style.format(fmt), width="stretch")
if has_benchmark:
    st.caption(
        f"Beta measures sensitivity to {benchmark}: >1 moves more than the market, <1 less. "
        "Alpha is annualised excess return after adjusting for that market exposure (rf=0)."
    )

# ---- Risk / return scatter ----
st.subheader("Risk vs return")
scatter_rows = summary(returns[asset_cols + ([benchmark] if has_benchmark else [])]).reset_index()
scatter_rows.columns = ["Ticker"] + list(scatter_rows.columns[1:])
scatter_rows["Type"] = np.where(scatter_rows["Ticker"] == benchmark, "Benchmark", "Asset")
fig_sc = px.scatter(
    scatter_rows, x="Annual volatility", y="Annual return", text="Ticker",
    color="Type", size=scatter_rows["Sharpe (rf=0)"].abs() + 0.1,
    labels={"Annual volatility": "Annual volatility", "Annual return": "Annual return"},
)
fig_sc.update_traces(textposition="top center")
fig_sc.update_xaxes(tickformat=".0%")
fig_sc.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_sc, width="stretch")

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
roll_vol = returns.rolling(30).std() * np.sqrt(TRADING_DAYS)
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

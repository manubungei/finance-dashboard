# Financial Markets Analysis Dashboard

An interactive dashboard that analyses asset prices, returns, risk and
correlations for any set of tickers, built with Streamlit and live data
from Yahoo Finance.

## Features
- Rebased price performance across selected assets
- Cumulative return over the chosen period
- 30-day rolling annualised volatility
- Return correlation heatmap
- Key metrics: annualised return, volatility, Sharpe (rf=0), max drawdown

## Run locally
```bash
conda activate ds
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser. Change the tickers and history length in the sidebar.

## Deploy (free)
1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, and point it at this repo's `app.py`.
3. Share the live URL on your CV and LinkedIn.

## Notes
Data comes from Yahoo Finance via the `yfinance` library. For educational
use only — nothing here is investment advice.

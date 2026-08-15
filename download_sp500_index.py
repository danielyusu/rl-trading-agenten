"""Download the S&P 500 price index (^GSPC) for the evaluation baseline.

Run this on Google Colab, then upload the resulting
``sp500_index.csv`` and place it at ``sp500/sp500_index.csv`` in the project,
overwriting the older file. The output format matches that file exactly:
two columns ``Date`` (YYYY-MM-DD) and ``S&P500`` (daily index level).

The end date (exclusive) is 2026-01-01 so the series covers the full test
period through 2025-12-31.
"""
import yfinance as yf
import pandas as pd

df = yf.download(
    "^GSPC",
    start="2014-12-22",      # matches the original file's start
    end="2026-01-01",        # exclusive -> includes 2025-12-31
    auto_adjust=False,
    progress=False,
)

# Recent yfinance returns a MultiIndex column frame; reduce to the Close series.
close = df["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

out = close.reset_index()
out.columns = ["Date", "S&P500"]
out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
out["S&P500"] = out["S&P500"].round(2)

out.to_csv("sp500_index.csv", index=False)
print(out.tail())
print(f"rows: {len(out)} | range: {out['Date'].iloc[0]} -> {out['Date'].iloc[-1]}")

# --- On Colab, trigger a browser download of the file: ---
try:
    from google.colab import files
    files.download("sp500_index.csv")
except Exception:
    print("Saved sp500_index.csv to the working directory.")

"""
Download historical data (2010-01-01 to 2025-12-31) for the top 100 S&P 500
companies ranked by their Weight in sp500_companies.csv, using yfinance.
Outputs:
  sp500_stocks_top100.csv
  sp500_companies_top100.csv
"""

import pandas as pd
import yfinance as yf

# ── 1. Identify top 100 tickers ───────────────────────────────────────────────
df_companies = pd.read_csv('sp500/sp500_companies.csv')
top100 = df_companies.sort_values('Weight', ascending=False).head(100).reset_index(drop=True)
tickers = top100['Symbol'].tolist()

print(f"Top 100 tickers by S&P 500 weight:")
print(tickers)

# ── 2. Download all at once from yfinance ─────────────────────────────────────
print(f"\nDownloading {len(tickers)} tickers from Yahoo Finance (2010-01-01 to 2025-12-31)...")
raw = yf.download(
    tickers,
    start='2010-01-01',
    end='2026-01-01',   # end is exclusive in yfinance, so 2026-01-01 gives us up to 2025-12-31
    auto_adjust=False,
    progress=True,
)

print(f"\nRaw download shape: {raw.shape}")

# ── 3. Reshape wide -> long ───────────────────────────────────────────────────
frames = []
for ticker in tickers:
    try:
        df_t = raw.xs(ticker, axis=1, level='Ticker')[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()
    except KeyError:
        print(f"  WARNING: {ticker} not found in download, skipping.")
        continue

    df_t = df_t.dropna(subset=['High', 'Low'])
    if df_t.empty:
        print(f"  WARNING: {ticker} has no valid rows after dropping NaN High/Low.")
        continue

    df_t.index.name = 'Date'
    df_t = df_t.reset_index()
    df_t.insert(1, 'Symbol', ticker)
    df_t['Date'] = pd.to_datetime(df_t['Date']).dt.strftime('%Y-%m-%d')
    frames.append(df_t)

df_stocks = pd.concat(frames, ignore_index=True)
df_stocks.columns = ['Date', 'Symbol', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
df_stocks['Volume'] = df_stocks['Volume'].astype('Int64')
df_stocks = df_stocks.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# ── 4. Companies metadata ─────────────────────────────────────────────────────
df_companies_top100 = top100.copy()
df_companies_top100['Fulltimeemployees'] = df_companies_top100['Fulltimeemployees'].astype('Int64')

# ── 5. Save ───────────────────────────────────────────────────────────────────
df_stocks.to_csv('sp500_stocks_top100.csv', index=False)
df_companies_top100.to_csv('sp500_companies_top100.csv', index=False)

print("\nSaved:")
print("  sp500_stocks_top100.csv")
print("  sp500_companies_top100.csv")

# ── 6. Coverage summary ───────────────────────────────────────────────────────
summary = (
    df_stocks.groupby('Symbol')['Date']
    .agg(first='min', last='max', rows='count')
    .join(df_companies_top100.set_index('Symbol')[['Shortname', 'Weight']])
    .sort_values('Weight', ascending=False)
)
pd.set_option('display.max_rows', 110)
pd.set_option('display.width', 130)
print(f"\nTotal rows: {len(df_stocks):,} | Symbols: {df_stocks['Symbol'].nunique()}")
print("\nCoverage per symbol:")
print(summary.to_string())

# Highlight any symbol starting after 2010-01-04
late_starters = summary[summary['first'] > '2010-01-05']
if not late_starters.empty:
    print("\nSymbols with data starting after 2010-01-04 (IPO / spinoff):")
    print(late_starters[['first', 'rows', 'Shortname']].to_string())

"""
01_fetch_data.py
----------------
PUBLIC DATA stage.

Downloads daily historical price data from Yahoo Finance for the 50 largest
Indonesian stocks by market capitalization (as of Aug 2026), listed in
data/tickers_idx_top50.csv.

Output:
    data/prices.csv        -> adjusted close price matrix (dates x tickers)
    data/valid_tickers.csv -> tickers that successfully returned data
"""
import os
import time
import pandas as pd
import yfinance as yf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

START = "2022-09-01"
END = "2026-08-31"   # per assignment: as of August 2026

def main():
    meta = pd.read_csv(os.path.join(DATA, "tickers_idx_top50.csv"))
    yahoo_symbols = [f"{t}.JK" for t in meta["ticker"]]
    print(f"Downloading {len(yahoo_symbols)} tickers from Yahoo Finance...")

    raw = yf.download(
        yahoo_symbols,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    # When multiple tickers are requested, yf returns a column-multiindex.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [yahoo_symbols[0]]

    # strip the .JK suffix back to plain tickers
    close.columns = [c.replace(".JK", "") for c in close.columns]

    # drop tickers with too much missing data (< 60% coverage)
    coverage = close.notna().mean()
    keep = coverage[coverage >= 0.60].index.tolist()
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        print(f"Dropped {len(dropped)} low-coverage/failed tickers: {dropped}")

    close = close[keep].sort_index()
    # forward/back fill small gaps, then drop any remaining all-NaN rows
    close = close.ffill().bfill()
    close = close.dropna(how="any")

    close.to_csv(os.path.join(DATA, "prices.csv"))
    valid = meta[meta["ticker"].isin(keep)].reset_index(drop=True)
    valid.to_csv(os.path.join(DATA, "valid_tickers.csv"), index=False)

    print(f"Saved prices.csv: {close.shape[0]} trading days x {close.shape[1]} tickers")
    print(f"Date range: {close.index.min().date()} -> {close.index.max().date()}")

if __name__ == "__main__":
    main()

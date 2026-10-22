"""
02_descriptive.py
-----------------
DESCRIPTIVE ANALYTICS stage.

Computes daily log returns, annualized return/volatility/Sharpe per stock,
correlation structure, cumulative performance and sector aggregates.

Outputs (in output/):
    descriptive_stats.csv
    sector_stats.csv
    corr_matrix.csv
    fig_cumulative_top.png
    fig_risk_return.png
    fig_corr_heatmap.png
    fig_vol_distribution.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

TRADING_DAYS = 252
RF_ANNUAL = 0.0575  # ~BI rate proxy as risk-free

def main():
    prices = pd.read_csv(os.path.join(DATA, "prices.csv"), index_col=0, parse_dates=True)
    meta = pd.read_csv(os.path.join(DATA, "valid_tickers.csv"))
    sector_map = dict(zip(meta["ticker"], meta["sector"]))

    # daily log returns
    rets = np.log(prices / prices.shift(1)).dropna()
    rets.to_csv(os.path.join(DATA, "returns.csv"))

    # per-stock annualized metrics
    mean_d = rets.mean()
    std_d = rets.std()
    ann_ret = mean_d * TRADING_DAYS
    ann_vol = std_d * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol

    # max drawdown per stock
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum / roll_max - 1).min()

    stats = pd.DataFrame({
        "sector": [sector_map.get(t, "NA") for t in rets.columns],
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "total_return": prices.iloc[-1] / prices.iloc[0] - 1,
    }).sort_values("sharpe", ascending=False)
    stats.to_csv(os.path.join(OUT, "descriptive_stats.csv"))
    print("Top 10 by Sharpe:\n", stats.head(10).round(3), "\n")

    # sector aggregates
    sector_stats = stats.groupby("sector").agg(
        n=("ann_return", "size"),
        avg_ann_return=("ann_return", "mean"),
        avg_ann_vol=("ann_volatility", "mean"),
        avg_sharpe=("sharpe", "mean"),
    ).sort_values("avg_sharpe", ascending=False)
    sector_stats.to_csv(os.path.join(OUT, "sector_stats.csv"))

    # correlation
    corr = rets.corr()
    corr.to_csv(os.path.join(OUT, "corr_matrix.csv"))

    # ---- plots ----
    # 1. cumulative performance of top-8 Sharpe stocks
    top8 = stats.head(8).index.tolist()
    plt.figure(figsize=(11, 6))
    for t in top8:
        plt.plot(cum.index, cum[t], label=t, linewidth=1.3)
    plt.title("Cumulative Growth of Rp1 - Top 8 Stocks by Sharpe Ratio")
    plt.xlabel("Date"); plt.ylabel("Growth multiple")
    plt.legend(ncol=4, fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_cumulative_top.png"), dpi=130); plt.close()

    # 2. risk-return scatter
    plt.figure(figsize=(9, 7))
    plt.scatter(stats["ann_volatility"], stats["ann_return"], s=35, alpha=0.7)
    for t in stats.index:
        plt.annotate(t, (stats.loc[t, "ann_volatility"], stats.loc[t, "ann_return"]),
                     fontsize=6, alpha=0.7)
    plt.axhline(0, color="grey", lw=0.8)
    plt.title("Risk vs Return (annualized) - 49 IDX Large-Cap Stocks")
    plt.xlabel("Annualized Volatility"); plt.ylabel("Annualized Return")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_risk_return.png"), dpi=130); plt.close()

    # 3. correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="correlation")
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr)), corr.columns, fontsize=6)
    plt.title("Daily Return Correlation Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_corr_heatmap.png"), dpi=130); plt.close()

    # 4. volatility distribution
    plt.figure(figsize=(9, 5))
    plt.hist(stats["ann_volatility"], bins=15, edgecolor="black", alpha=0.8)
    plt.title("Distribution of Annualized Volatility")
    plt.xlabel("Annualized Volatility"); plt.ylabel("Number of stocks")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_vol_distribution.png"), dpi=130); plt.close()

    print("Average pairwise correlation: %.3f" % corr.values[np.triu_indices_from(corr.values, 1)].mean())
    print("Descriptive stage done. Figures + CSVs saved to output/.")

if __name__ == "__main__":
    main()

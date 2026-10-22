"""
05_optimal_decision.py
----------------------
OPTIMAL DECISION stage.

Validates the prescriptive recommendation out-of-sample and produces the final
decision. We split the history into:
    TRAIN: start .. 2025-08-31   (used to estimate mu/cov and optimize weights)
    TEST : 2025-09-01 .. end     (held out, used to measure realized performance)

We compare the Max-Sharpe optimal portfolio against the Equal-Weight benchmark
on the TEST set (realized return, volatility, Sharpe, max drawdown). The final
recommended allocation is then re-estimated on the FULL history.

Outputs (in output/):
    backtest_results.csv
    final_recommendation.csv
    fig_backtest_equity.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")

TRADING_DAYS = 252
RF_ANNUAL = 0.0575
MAX_WEIGHT = 0.15
SPLIT_DATE = "2025-09-01"

def optimize_max_sharpe(mu, cov):
    n = len(mu)
    bounds = tuple((0.0, MAX_WEIGHT) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.ones(n) / n
    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        return -(r - RF_ANNUAL) / v if v > 0 else 0
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    return res.x

def perf_stats(daily_port_ret):
    ann_ret = daily_port_ret.mean() * TRADING_DAYS
    ann_vol = daily_port_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else 0
    cum = (1 + daily_port_ret).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    total = cum.iloc[-1] - 1
    return dict(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_drawdown=mdd, total_return=total)

def main():
    rets = pd.read_csv(os.path.join(DATA, "returns.csv"), index_col=0, parse_dates=True)
    exp = pd.read_csv(os.path.join(OUT, "expected_returns.csv"), index_col=0)
    tickers = [t for t in rets.columns if t in exp.index]
    rets = rets[tickers]

    train = rets[rets.index < SPLIT_DATE]
    test = rets[rets.index >= SPLIT_DATE]
    print(f"Train: {train.index.min().date()}..{train.index.max().date()} ({len(train)} days)")
    print(f"Test : {test.index.min().date()}..{test.index.max().date()} ({len(test)} days)")

    # Estimate inputs on TRAIN only (no look-ahead). Use historical mean on train
    # as the expected-return view to keep the backtest fully out-of-sample.
    mu_train = train.mean().values * TRADING_DAYS
    cov_train = LedoitWolf().fit(train.values).covariance_ * TRADING_DAYS

    w_opt = optimize_max_sharpe(mu_train, cov_train)
    w_eq = np.ones(len(tickers)) / len(tickers)

    # Realized daily portfolio returns on TEST (buy-and-hold weights)
    test_opt = test.values @ w_opt
    test_eq = test.values @ w_eq
    opt_series = pd.Series(test_opt, index=test.index)
    eq_series = pd.Series(test_eq, index=test.index)

    bt = pd.DataFrame({
        "Max Sharpe (optimized on train)": perf_stats(opt_series),
        "Equal Weight (benchmark)": perf_stats(eq_series),
    }).T
    bt.to_csv(os.path.join(OUT, "backtest_results.csv"))
    print("\nOut-of-sample backtest (TEST period):\n", bt.round(4), "\n")

    # equity curves
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, (1 + opt_series).cumprod(), label="Max-Sharpe Optimal", lw=2)
    plt.plot(test.index, (1 + eq_series).cumprod(), label="Equal Weight", lw=2, ls="--")
    plt.axhline(1.0, color="grey", lw=0.8)
    plt.title("Out-of-Sample Equity Curve (Test: Sep 2025 - Aug 2026)")
    plt.xlabel("Date"); plt.ylabel("Growth of Rp1")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_backtest_equity.png"), dpi=130); plt.close()

    # FINAL recommendation: re-estimate on full history using predictive expected returns
    mu_full = exp.loc[tickers, "expected_return"].values
    cov_full = LedoitWolf().fit(rets.values).covariance_ * TRADING_DAYS
    w_final = optimize_max_sharpe(mu_full, cov_full)
    final = pd.Series(w_final, index=tickers)
    final = final[final > 0.005].sort_values(ascending=False)
    final_df = final.rename("weight").to_frame()
    final_df["weight_pct"] = (final_df["weight"] * 100).round(2)
    final_df.to_csv(os.path.join(OUT, "final_recommendation.csv"))
    print("FINAL recommended allocation (holdings > 0.5%):\n", final_df.round(4))
    print("Sum of weights:", round(final.sum(), 4))
    print("Optimal decision stage done.")

if __name__ == "__main__":
    main()

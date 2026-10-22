"""
04_prescriptive.py
------------------
PRESCRIPTIVE ANALYTICS stage (optimal decision-making).

Mean-variance (Markowitz) portfolio optimization using:
  - expected returns from the predictive stage (output/expected_returns.csv)
  - covariance matrix from historical daily returns (annualized, Ledoit-Wolf
    shrinkage for stability)

Optimized portfolios under realistic constraints (long-only, max 15% weight):
  1. Maximum Sharpe ratio (tangency) portfolio
  2. Minimum variance portfolio
  3. Efficient frontier (for visualization)
  4. Equal-weight benchmark

Outputs (in output/):
    optimal_weights.csv
    frontier_points.csv
    portfolio_summary.csv
    fig_efficient_frontier.png
    fig_optimal_weights.png
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
MAX_WEIGHT = 0.15  # cap single-name concentration

def portfolio_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - RF_ANNUAL) / vol if vol > 0 else 0.0
    return ret, vol, sharpe

def main():
    rets = pd.read_csv(os.path.join(DATA, "returns.csv"), index_col=0, parse_dates=True)
    exp = pd.read_csv(os.path.join(OUT, "expected_returns.csv"), index_col=0)

    tickers = [t for t in rets.columns if t in exp.index]
    rets = rets[tickers]
    mu = exp.loc[tickers, "expected_return"].values

    # annualized covariance via Ledoit-Wolf shrinkage (robust, invertible)
    lw = LedoitWolf().fit(rets.values)
    cov = lw.covariance_ * TRADING_DAYS

    n = len(tickers)
    bounds = tuple((0.0, MAX_WEIGHT) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.ones(n) / n

    # 1. Max Sharpe (minimize negative Sharpe)
    def neg_sharpe(w):
        r, v, s = portfolio_perf(w, mu, cov)
        return -s
    res_sharpe = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                          options={"maxiter": 1000, "ftol": 1e-10})
    w_sharpe = res_sharpe.x

    # 2. Min variance
    def port_var(w):
        return w @ cov @ w
    res_minvar = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=cons,
                          options={"maxiter": 1000, "ftol": 1e-12})
    w_minvar = res_minvar.x

    # 3. Efficient frontier: minimize variance for a grid of target returns
    target_returns = np.linspace(mu.min() * 0.5, mu.max() * 0.95, 40)
    frontier = []
    for tr in target_returns:
        cons_t = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=tr: w @ mu - tr},
        )
        r = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=cons_t,
                     options={"maxiter": 1000, "ftol": 1e-10})
        if r.success:
            ret, vol, sh = portfolio_perf(r.x, mu, cov)
            frontier.append((vol, ret, sh))
    frontier = pd.DataFrame(frontier, columns=["volatility", "return", "sharpe"])
    frontier.to_csv(os.path.join(OUT, "frontier_points.csv"), index=False)

    # 4. Equal weight benchmark
    w_eq = np.ones(n) / n

    # assemble weights table
    weights = pd.DataFrame({
        "max_sharpe": w_sharpe,
        "min_variance": w_minvar,
        "equal_weight": w_eq,
    }, index=tickers)
    weights = weights.round(4)
    weights.to_csv(os.path.join(OUT, "optimal_weights.csv"))

    # performance summary
    summary = {}
    for name, w in [("Max Sharpe", w_sharpe), ("Min Variance", w_minvar), ("Equal Weight", w_eq)]:
        r, v, s = portfolio_perf(w, mu, cov)
        summary[name] = {"exp_return": r, "volatility": v, "sharpe": s,
                         "n_holdings": int((w > 0.005).sum())}
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(OUT, "portfolio_summary.csv"))
    print("Portfolio summary:\n", summary_df.round(4), "\n")

    # individual asset points for frontier plot
    ind_vol = np.sqrt(np.diag(cov))
    ind_ret = mu

    # ---- plots ----
    plt.figure(figsize=(10, 7))
    plt.scatter(ind_vol, ind_ret, s=25, alpha=0.4, color="grey", label="Individual stocks")
    plt.plot(frontier["volatility"], frontier["return"], "b-", lw=2, label="Efficient frontier")
    for name, w, c, mk in [("Max Sharpe", w_sharpe, "red", "*"),
                           ("Min Variance", w_minvar, "green", "D"),
                           ("Equal Weight", w_eq, "orange", "s")]:
        r, v, s = portfolio_perf(w, mu, cov)
        plt.scatter(v, r, color=c, marker=mk, s=180, edgecolor="black",
                    label=f"{name} (Sharpe={s:.2f})", zorder=5)
    plt.title("Efficient Frontier - IDX Top-50 Large Caps")
    plt.xlabel("Annualized Volatility"); plt.ylabel("Expected Annualized Return")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_efficient_frontier.png"), dpi=130); plt.close()

    # optimal weights bar chart (max sharpe, nonzero only)
    ms = weights["max_sharpe"]
    ms = ms[ms > 0.005].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(ms.index, ms.values, color="steelblue", edgecolor="black")
    plt.title("Max-Sharpe Optimal Portfolio Weights (holdings > 0.5%)")
    plt.ylabel("Weight"); plt.xticks(rotation=90)
    plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_optimal_weights.png"), dpi=130); plt.close()

    print("Max-Sharpe holdings (>0.5%):\n", ms.round(4))
    print("Prescriptive stage done.")

if __name__ == "__main__":
    main()

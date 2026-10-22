# Prescriptive Data Science — Assignment 1
## Optimal Portfolio Construction for the 50 Largest Indonesian Stocks (as of August 2026)

**Data Science Workflow:** Public Data → Descriptive Analytics → Predictive Analytics → Prescriptive Analytics → Optimal Decision

---

## 1. Executive Summary

This project applies the full data science workflow to a universe of the 50 Indonesian
stocks with the largest market capitalization on the Indonesia Stock Exchange (IDX) as of
August 2026, with the objective of producing an **optimal portfolio allocation decision**.

Using four years of daily adjusted-close prices (Sep 2022 – Aug 2026, 952 trading days,
47 usable tickers after data cleaning) sourced from Yahoo Finance, we:

1. **Described** the risk–return profile of each stock and sector. Healthcare, Properties
   and Energy delivered the strongest historical risk-adjusted returns, while Consumer
   Cyclicals, Industrials and Telco lagged. The average pairwise return correlation is low
   (**0.12**), signalling genuine diversification potential within the large-cap universe.
2. **Predicted** forward one-month returns with technical-factor models (Ridge, Random
   Forest) evaluated by walk-forward cross-validation. Return prediction is intrinsically
   hard: the best model (Random Forest) reached an information coefficient of only **0.084**,
   modestly positive but economically weak. We therefore use the model output only as a
   *shrinkage view* blended 50/50 with historical means.
3. **Prescribed** an optimal allocation via Markowitz mean-variance optimization with
   Ledoit-Wolf covariance shrinkage, long-only and a 15% single-name cap. The in-sample
   **Max-Sharpe** portfolio achieves an expected return of **58.0%**, volatility **19.3%**,
   and Sharpe **2.72**, versus an equal-weight Sharpe of **0.89**.
4. **Validated** the decision out-of-sample. Here lies the key finding: when the optimizer
   is trained only on 2022–2025 data and tested on Sep 2025 – Aug 2026, the concentrated
   Max-Sharpe portfolio **underperformed** the equal-weight benchmark (−28.4% vs −2.5%
   total return). The test window was a broad large-cap drawdown, and concentration
   amplified the loss. This is a textbook demonstration of **estimation error / overfitting**
   in naive mean-variance optimization.

**Optimal Decision.** We recommend the diversified, risk-controlled Max-Sharpe allocation
(15 holdings, 15% cap per name, spread across Energy, Basic Materials, Financials, Healthcare
and Consumer sectors) **as a strategic tilt, not a static buy-and-hold bet**. The out-of-sample
evidence mandates three guardrails: (i) keep the single-name and sector caps, (ii) rebalance
periodically rather than holding fixed weights, and (iii) treat the equal-weight portfolio as
a robust fallback. The honest conclusion is that disciplined diversification plus risk limits
matters more than chasing the highest in-sample Sharpe ratio.

---

## 2. Public Data

- **Universe:** 50 largest IDX stocks by market cap (Aug 2026), tickers in
  `data/tickers_idx_top50.csv`.
- **Price source:** Yahoo Finance via the `yfinance` API, symbol format `<TICKER>.JK`,
  `auto_adjust=True` (adjusted close, accounts for splits/dividends).
- **Period:** 2022-09-01 → 2026-08-28 (daily).
- **Cleaning:** tickers with <60% data coverage were dropped (AADI, CDIA, EMAS — recent
  IPOs/spin-offs with short price history). Small gaps were forward/back-filled; remaining
  incomplete rows removed. Final panel: **952 trading days × 47 tickers**, zero missing values.

Reproduced by `code/01_fetch_data.py` → `data/prices.csv`, `data/valid_tickers.csv`.

---

## 3. Descriptive Analytics

Daily **log returns** were computed and annualized (252 trading days). Metrics per stock:
annualized return, annualized volatility, Sharpe ratio (risk-free = 5.75%, a BI-rate proxy),
maximum drawdown, and cumulative total return.

**Key observations**
- **Top risk-adjusted performers:** SRAJ (Sharpe 1.64), CUAN, PTRO, MPRO, MLPT and DSSA —
  a mix of Healthcare, Energy, Properties and Technology, reflecting the strong commodity
  and selective growth cycle over the sample.
- **Sector ranking (avg Sharpe):** Healthcare (1.64) and Properties (0.66) lead, followed by
  Energy (0.47) and Technology (0.30); Consumer Cyclicals (−0.40), Industrials (−0.12) and
  Telecommunication (−0.06) trail.
- **Diversification:** mean pairwise correlation of only **0.12** across 47 names — a strong
  case for portfolio construction rather than single-stock bets.
- Several high-return names also carry very large drawdowns (e.g. MLPT −96%, DSSA −93%, CUAN
  −86%), underscoring that raw return is not the same as quality.

**Figures:** `fig_cumulative_top.png`, `fig_risk_return.png`, `fig_corr_heatmap.png`,
`fig_vol_distribution.png`. **Tables:** `descriptive_stats.csv`, `sector_stats.csv`,
`corr_matrix.csv`. Produced by `code/02_descriptive.py`.

---

## 4. Predictive Analytics

**Problem framing.** For each stock we engineer technical features from past prices and
predict the **next 21-trading-day (≈1 month) forward log return**:

| Feature | Description |
|---|---|
| mom_21 / mom_63 / mom_126 | price momentum over 1/3/6 months |
| vol_21 / vol_63 | rolling return volatility |
| ma_ratio | 21-day / 63-day moving-average ratio |
| rsi_14 | 14-day Relative Strength Index |

**Models & evaluation.** Ridge regression and Random Forest were trained on the pooled
cross-sectional panel and evaluated with `TimeSeriesSplit` (5 folds, no look-ahead),
compared against a historical-mean baseline.

| Model | RMSE | Directional Acc. | Information Coef. |
|---|---|---|---|
| Ridge | 0.194 | 0.49 | 0.049 |
| **Random Forest** | 0.194 | 0.49 | **0.084** |
| Baseline (hist-mean) | 0.191 | 0.48 | 0.000 |

**Honest interpretation.** Short-horizon return prediction is extremely noisy. RMSE is
essentially flat across models and the baseline, and directional accuracy sits near 0.49
(close to a coin flip). The Random Forest's positive information coefficient (0.084) is the
only edge, and it is small. Consequently we do **not** trust raw model forecasts; we **blend
them 50/50 with historical means (shrinkage)** to form the expected-return vector used
downstream.

**Figures/Tables:** `fig_pred_vs_actual.png`, `model_evaluation.csv`, `expected_returns.csv`.
Produced by `code/03_predictive.py`.

---

## 5. Prescriptive Analytics

**Method.** Markowitz mean-variance optimization (SLSQP), with:
- expected returns = blended predictive view (Section 4),
- covariance = **Ledoit-Wolf shrinkage** of the daily return covariance (annualized) for a
  stable, invertible matrix,
- constraints: long-only, fully invested (Σw = 1), **max 15% per stock** (concentration cap).

Three portfolios plus the efficient frontier were computed.

| Portfolio | Exp. Return | Volatility | Sharpe | # Holdings |
|---|---|---|---|---|
| **Max Sharpe** | 58.0% | 19.3% | **2.72** | 15 |
| Min Variance | 15.4% | 10.7% | 0.90 | 26 |
| Equal Weight | 22.4% | 18.6% | 0.89 | 47 |

The Max-Sharpe (tangency) portfolio dominates in-sample. **Figures:** `fig_efficient_frontier.png`,
`fig_optimal_weights.png`. **Tables:** `optimal_weights.csv`, `portfolio_summary.csv`,
`frontier_points.csv`. Produced by `code/04_prescriptive.py`.

---

## 6. Optimal Decision

**Out-of-sample validation.** To avoid presenting an in-sample illusion, we split:
- **Train:** 2022-09 → 2025-08 (714 days) — estimate μ, Σ, optimize weights.
- **Test:** 2025-09 → 2026-08 (237 days) — hold weights fixed, measure realized performance.

| Portfolio (test period) | Ann. Return | Ann. Vol | Sharpe | Max DD | Total Return |
|---|---|---|---|---|---|
| Max Sharpe (trained on train) | −28.7% | 36.6% | −0.94 | −62% | **−28.4%** |
| Equal Weight (benchmark) | +1.4% | 28.5% | −0.15 | −43% | **−2.5%** |

**Reading the result.** The test window was a broad large-cap drawdown — but while the
equal-weight benchmark was essentially flat (−2.5%), the concentrated optimized portfolio
lost **28.4%**. This is a classic manifestation of **estimation error**: mean-variance
optimizers over-weight assets with high *estimated* return/low *estimated* risk, which were
exactly the momentum names whose trend reversed. The impressive in-sample Sharpe of 2.72 did
not survive contact with new data.

**Final recommended allocation** (re-estimated on full history, holdings > 0.5%):

| Ticker | Weight | Ticker | Weight |
|---|---|---|---|
| SUPR | 15.00% | BYAN | 5.87% |
| DSSA | 15.00% | DNET | 5.66% |
| SRAJ | 8.68% | TPIA | 4.38% |
| INDF | 8.19% | MPRO | 4.05% |
| SMMA | 7.67% | CASA | 3.82% |
| CUAN | 7.50% | MORA | 3.10% |
| MLPT | 5.98% | DCII | 2.58% |
|  |  | ADRO | 2.52% |

(Weights sum to 100%; see `final_recommendation.csv`.)

**The decision, with guardrails.** We recommend this diversified, capped Max-Sharpe tilt as a
*strategic* allocation rather than a static bet, subject to:
1. **Keep concentration limits** (15% per name; monitor sector exposure, especially Energy
   and Financials, which dominate the allocation).
2. **Rebalance periodically** (e.g. monthly/quarterly) so the portfolio adapts to regime
   changes instead of riding a stale momentum signal.
3. **Equal-weight as fallback** — it proved materially more robust out-of-sample and is the
   prudent default when conviction in the forecast is low.

The overarching lesson for optimal decision-making: **robustness and risk control beat raw
optimization**. The highest in-sample Sharpe is not the safest decision.

**Figure/Table:** `fig_backtest_equity.png`, `backtest_results.csv`, `final_recommendation.csv`.
Produced by `code/05_optimal_decision.py`.

---

## 7. Limitations & Future Work

- Return prediction has weak signal; alternative data (fundamentals, macro, order-flow) could
  help.
- Single train/test split; a rolling-window walk-forward backtest with transaction costs would
  be more rigorous.
- No transaction costs, taxes, or liquidity constraints modelled.
- The ticker universe is taken directly from the official IDX "50 Biggest Market Capitalization
  — Aug 2026" file. Three names (AADI, CDIA, EMAS) were excluded for insufficient price history
  on Yahoo Finance, leaving 47 stocks analysed.

---

## 8. Reproducibility

```
pip install -r requirements.txt
python code/01_fetch_data.py       # Public Data
python code/02_descriptive.py      # Descriptive
python code/03_predictive.py       # Predictive
python code/04_prescriptive.py     # Prescriptive
python code/05_optimal_decision.py # Optimal Decision
```
All tables and figures are regenerated into `output/`.

*Risk-free rate, horizon, and concentration cap are configurable constants at the top of each
script. Note: results depend on live Yahoo Finance data and may shift slightly on re-run.*

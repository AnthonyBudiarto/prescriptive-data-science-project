"""
03_predictive.py
----------------
PREDICTIVE ANALYTICS stage.

Goal: predict each stock's forward return using technical features, with a
proper time-series (walk-forward) evaluation. The model's cross-sectional
forward-return forecast feeds the prescriptive optimizer as the "expected
return" view (instead of relying purely on historical mean).

Approach:
  - Build features per stock from past returns: momentum (21/63/126d),
    rolling volatility (21/63d), moving-average ratio, RSI.
  - Target: next 21-trading-day (≈1 month) forward log return.
  - Models: Ridge regression and RandomForest, compared against a naive
    "historical mean" baseline using TimeSeriesSplit.
  - Metric: directional accuracy + RMSE, information coefficient (rank corr).
  - Output: expected_returns.csv (annualized predicted return per stock).

Outputs (in output/):
    model_evaluation.csv
    expected_returns.csv
    fig_pred_vs_actual.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")

TRADING_DAYS = 252
HORIZON = 21  # ~1 month forward

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - 100 / (1 + rs)

def build_features(prices):
    """Return a long (date, ticker) feature frame with forward-return target."""
    frames = []
    logret = np.log(prices / prices.shift(1))
    for t in prices.columns:
        p = prices[t]
        r = logret[t]
        df = pd.DataFrame(index=p.index)
        df["mom_21"] = p / p.shift(21) - 1
        df["mom_63"] = p / p.shift(63) - 1
        df["mom_126"] = p / p.shift(126) - 1
        df["vol_21"] = r.rolling(21).std()
        df["vol_63"] = r.rolling(63).std()
        df["ma_ratio"] = p.rolling(21).mean() / p.rolling(63).mean() - 1
        df["rsi_14"] = rsi(p, 14) / 100.0
        # target: forward HORIZON-day log return
        df["target"] = np.log(p.shift(-HORIZON) / p)
        df["ticker"] = t
        frames.append(df)
    out = pd.concat(frames).dropna()
    return out

def main():
    prices = pd.read_csv(os.path.join(DATA, "prices.csv"), index_col=0, parse_dates=True)
    feat = build_features(prices)
    feature_cols = ["mom_21", "mom_63", "mom_126", "vol_21", "vol_63", "ma_ratio", "rsi_14"]

    # Pool all stocks cross-sectionally, ordered by date (panel regression).
    feat = feat.sort_index()
    X = feat[feature_cols].values
    y = feat["target"].values
    dates = feat.index

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            random_state=42, n_jobs=-1),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    last_pred_store = {}
    for name, model in models.items():
        rmses, dir_accs, ics = [], [], []
        for tr, te in tscv.split(X):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
            dir_accs.append(np.mean(np.sign(pred) == np.sign(y[te])))
            ic = spearmanr(pred, y[te]).correlation
            ics.append(ic if not np.isnan(ic) else 0.0)
        results.append({
            "model": name,
            "rmse": np.mean(rmses),
            "dir_accuracy": np.mean(dir_accs),
            "info_coef": np.mean(ics),
        })
        last_pred_store[name] = (y[te], model.predict(X[te]))

    # baseline: predict historical pooled mean target
    base_pred = np.full_like(y, y.mean())
    results.append({
        "model": "Baseline(hist-mean)",
        "rmse": np.sqrt(mean_squared_error(y, base_pred)),
        "dir_accuracy": np.mean(np.sign(base_pred) == np.sign(y)),
        "info_coef": 0.0,
    })
    eval_df = pd.DataFrame(results).set_index("model")
    eval_df.to_csv(os.path.join(OUT, "model_evaluation.csv"))
    print("Model evaluation:\n", eval_df.round(4), "\n")

    # pick best model by information coefficient
    best_name = eval_df.drop("Baseline(hist-mean)")["info_coef"].idxmax()
    best_model = models[best_name]
    best_model.fit(X, y)
    print(f"Best model by IC: {best_name}")

    # Generate current expected return per stock using the latest feature row.
    latest_feats = []
    tickers = []
    for t in prices.columns:
        sub = feat[feat["ticker"] == t]
        if len(sub) == 0:
            continue
        latest_feats.append(sub[feature_cols].iloc[-1].values)
        tickers.append(t)
    latest_feats = np.array(latest_feats)
    pred_fwd = best_model.predict(latest_feats)  # predicted HORIZON-day log return

    # annualize: scale HORIZON-day return to yearly
    ann_pred = pred_fwd * (TRADING_DAYS / HORIZON)

    # blend with historical mean (shrinkage) to reduce estimation noise
    hist_ann = (np.log(prices / prices.shift(1)).mean() * TRADING_DAYS).reindex(tickers).values
    blended = 0.5 * ann_pred + 0.5 * hist_ann

    exp_ret = pd.DataFrame({
        "ticker": tickers,
        "pred_ann_return_model": ann_pred,
        "hist_ann_return": hist_ann,
        "expected_return": blended,
    }).set_index("ticker").sort_values("expected_return", ascending=False)
    exp_ret.to_csv(os.path.join(OUT, "expected_returns.csv"))
    print("Top 10 expected returns (blended):\n", exp_ret.head(10).round(3))

    # plot predicted vs actual on last fold for best model
    y_true, y_hat = last_pred_store[best_name]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_hat, y_true, s=8, alpha=0.3)
    lim = [min(y_hat.min(), y_true.min()), max(y_hat.max(), y_true.max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("Predicted 21d fwd return"); plt.ylabel("Actual 21d fwd return")
    plt.title(f"Predicted vs Actual - {best_name} (last CV fold)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_pred_vs_actual.png"), dpi=130); plt.close()
    print("Predictive stage done.")

if __name__ == "__main__":
    main()

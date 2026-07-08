"""Fair, reproducible evaluation framework.

Goal (from the research brief): *no model or target proposal is accepted
without reproducible, leakage-free, out-of-sample evidence.* This module is the
referee. It is model-agnostic and needs no torch.

Core ideas
----------
* **Panel**: one tidy table of (date, stock, features..., targets..., fwd_ret).
* **Cross-sectional IC**: the standard quant measure of ranking skill — the
  per-date Spearman correlation between a score and realised forward return,
  averaged over dates. This is what a top-k stock ranker actually needs.
* **Walk-forward with embargo**: fit only on the past, predict the next block,
  purge an embargo of Y days at the boundary so no training label window
  overlaps the evaluation block (same anti-leak principle as the fixed
  chronological split).
* **Top-k backtest**: turn scores into a tradable portfolio and measure net
  Sharpe / drawdown / turnover under an explicit cost. Predictability is
  necessary but not sufficient — the target must also pay after costs.

The referee is deliberately simple (ridge / linear-probability model) so that
when we compare *targets* or *architectures*, the only thing that changes is the
thing under test.
"""

import numpy as np
import pandas as pd

from features import compute_features, FEATURE_COLS
from targets import TARGETS


# ----------------------------------------------------------------------------
# Panel construction
# ----------------------------------------------------------------------------
def build_panel(universe, Y=20):
    """Assemble the long panel from {stock_id: ohlcv_df}.

    Columns: date, stock, <FEATURE_COLS>, <target names>, fwd_ret (simple
    Y-day forward return used for PnL). All strictly point-in-time.
    """
    rows = []
    for sid, df in universe.items():
        df = df.sort_values("date").reset_index(drop=True)
        feats = compute_features(df)
        tgt = {name: fn(df, Y=Y) for name, fn in TARGETS.items()}
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n - Y] = c[Y:] / c[:n - Y] - 1.0

        block = feats.copy()
        block["stock"] = sid
        for name, arr in tgt.items():
            block[name] = arr
        block["fwd_ret"] = fwd_ret
        rows.append(block)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["date", "stock"]).reset_index(drop=True)
    return panel


# ----------------------------------------------------------------------------
# Information coefficient
# ----------------------------------------------------------------------------
def cross_sectional_ic(panel, score_col, target_col, min_names=5):
    """Mean/IR of the per-date cross-sectional Spearman IC between score and
    target. Returns (mean_ic, ic_ir, n_dates, series)."""
    def _ic(g):
        g = g[[score_col, target_col]].dropna()
        if len(g) < min_names or g[score_col].nunique() < 2 or g[target_col].nunique() < 2:
            return np.nan
        # Spearman = Pearson on ranks (avoids a scipy dependency).
        rs = g[score_col].rank()
        rt = g[target_col].rank()
        return rs.corr(rt)

    ics = panel.groupby("date", group_keys=False).apply(_ic).dropna()
    if len(ics) == 0:
        return np.nan, np.nan, 0, ics
    mean_ic = ics.mean()
    ic_ir = mean_ic / (ics.std() + 1e-12)  # information ratio of the IC series
    return mean_ic, ic_ir, len(ics), ics


# ----------------------------------------------------------------------------
# Simple, deterministic referee model (ridge / linear-probability)
# ----------------------------------------------------------------------------
def _xs_standardize(df, cols):
    """Per-date cross-sectional z-score of features (robust to regime scale)."""
    out = df.copy()
    for c in cols:
        g = out.groupby("date")[c]
        out[c] = (out[c] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return out


def _ridge_fit(X, y, lam=10.0):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam  # don't regularise the intercept
    w = np.linalg.solve(A, Xb.T @ y)
    return w


def _ridge_pred(X, w):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return Xb @ w


def walk_forward_scores(panel, target_col, feature_cols=FEATURE_COLS,
                        Y=20, min_train_dates=250, step=20, lam=10.0):
    """Expanding-window walk-forward. For each OOS block of `step` dates, fit a
    ridge model on all dates ending `Y` before the block (embargo) and predict
    the block. Returns the panel with an added `score` column (NaN outside OOS).
    """
    panel = _xs_standardize(panel, feature_cols)
    dates = np.array(sorted(panel["date"].unique()))
    date_rank = {d: i for i, d in enumerate(dates)}
    panel = panel.copy()
    panel["_r"] = panel["date"].map(date_rank)
    panel["score"] = np.nan

    start = min_train_dates
    while start < len(dates):
        block = range(start, min(start + step, len(dates)))
        train_cutoff = start - Y  # embargo
        if train_cutoff < min_train_dates // 2:
            start += step
            continue

        tr = panel[(panel["_r"] < train_cutoff)].dropna(
            subset=feature_cols + [target_col])
        oos = panel[panel["_r"].isin(list(block))].dropna(subset=feature_cols)
        if len(tr) < 200 or len(oos) == 0:
            start += step
            continue

        w = _ridge_fit(tr[feature_cols].to_numpy(float),
                       tr[target_col].to_numpy(float), lam=lam)
        pred = _ridge_pred(oos[feature_cols].to_numpy(float), w)
        panel.loc[oos.index, "score"] = pred
        start += step

    return panel.drop(columns="_r")


# ----------------------------------------------------------------------------
# Top-k backtest
# ----------------------------------------------------------------------------
def backtest_topk(panel, score_col="score", ret_col="fwd_ret", k=10,
                  holding=20, cost_bps=30.0):
    """Non-overlapping top-k long portfolio.

    Rebalance every `holding` trading dates: equal-weight the k highest scores,
    realise the Y-day forward return, charge round-trip cost. Returns a metrics
    dict and the per-rebalance return series.
    """
    df = panel.dropna(subset=[score_col, ret_col])
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]

    rets = []
    prev = set()
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < k:
            continue
        picks = day.nlargest(k, score_col)
        gross = picks[ret_col].mean()
        names = set(picks["stock"])
        turnover = len(names ^ prev) / max(len(names), 1)
        cost = (cost_bps / 1e4) * turnover
        rets.append(gross - cost)
        prev = names

    rets = np.array(rets, dtype=float)
    return _portfolio_metrics(rets, holding), rets


def cross_sectional_neutralize(panel, y_col, factor_cols):
    """Per-date OLS residual of y_col on factor_cols (+ intercept). Returns a
    Series aligned to panel.index: the part of y_col orthogonal to the factors,
    cross-sectionally, each date. Used to strip a factor (e.g. volatility) out of
    forward returns so we can ask whether a score predicts anything *beyond* it.
    """
    def _resid(g):
        out = pd.Series(np.nan, index=g.index)
        d = g[[y_col] + factor_cols].dropna()
        if len(d) < len(factor_cols) + 2:
            return out
        X = np.column_stack([np.ones(len(d))] +
                            [d[f].to_numpy(float) for f in factor_cols])
        y = d[y_col].to_numpy(float)
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[d.index] = y - X @ w
        return out

    return panel.groupby("date", group_keys=False).apply(_resid)


def backtest_long_short(panel, score_col="score", ret_col="fwd_ret", k=10,
                        holding=20, cost_bps=30.0):
    """Dollar-neutral long-short book: long the top-k scores, short the bottom-k,
    equal weight per leg. Return = mean(top) - mean(bottom) - costs (both legs).

    This nets out the common market/sector move each rebalance, so the result
    measures *ranking skill* (alpha) rather than long-only beta — essential when
    the universe is one highly-correlated sector. Metrics interpret the spread
    as the strategy return."""
    df = panel.dropna(subset=[score_col, ret_col])
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]

    rets = []
    prev = set()
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < 2 * k:
            continue
        ordered = day.sort_values(score_col)
        longs = ordered.tail(k)
        shorts = ordered.head(k)
        spread = longs[ret_col].mean() - shorts[ret_col].mean()
        names = set(longs["stock"]) | set(shorts["stock"])
        turnover = len(names ^ prev) / max(len(names), 1)
        cost = (cost_bps / 1e4) * turnover
        rets.append(spread - cost)
        prev = names

    rets = np.array(rets, dtype=float)
    return _portfolio_metrics(rets, holding), rets


def _portfolio_metrics(rets, holding):
    if len(rets) == 0:
        return {"n": 0, "mean_ret": float("nan"), "ann_ret": float("nan"),
                "sharpe": float("nan"), "hit_rate": float("nan"),
                "max_dd": float("nan")}
    periods_per_year = 252 / holding
    mean = rets.mean()
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = (mean / (std + 1e-12)) * np.sqrt(periods_per_year)
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1).min())
    ann_ret = (1 + mean) ** periods_per_year - 1
    return {
        "n": int(len(rets)),
        "mean_ret": float(mean),
        "ann_ret": float(ann_ret),
        "sharpe": float(sharpe),
        "hit_rate": float((rets > 0).mean()),
        "max_dd": max_dd,
    }

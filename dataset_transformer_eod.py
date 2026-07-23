"""TWSE EOD sequence dataset for the LSTM-Transformer daily-retrain system.

Builds, from the frozen research OHLCV cache (research/data_cache), everything the
GPU trainer needs:

  X            (N, seq_len, F) float32 sequence windows, strictly causal
  targets      {name: (N,) float32}, forward-looking, NaN where unknown
  meta         per-sample stock index, feature-date rank (global trading calendar),
               label-end rank per horizon (for matured-label filtering), sector

Anti-leakage rules implemented here (see brief §4/§13):
  * Execution lag: every forward target starts at close[t+1] — a prediction made
    from EOD features at t is only executable on t+1. No same-bar execution.
  * Matured labels: a sample enters a training set only if its label window ends
    on or before the refit date (`label_end_rank <= refit_rank`).
  * All per-day features are trailing-window ratios / z-scores (causal); the only
    cross-sectional transforms are per-date (no time direction). No scaler is ever
    fit on the full sample.
  * Recency weights are computed relative to the latest *matured label* rank.

Feature sets (brief §6) — `transaction` (trade count) is not in the reproducible
snapshot, so turnover is proxied by close*volume and trade-count features are
documented as unavailable offline.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "research"))

from data import load_universe, SECTOR_MAP  # noqa: E402


# ---------------------------------------------------------------------------
# Per-stock causal features
# ---------------------------------------------------------------------------
def _stock_features(df, feature_set):
    """Return a DataFrame (indexed like df) of per-day causal features for one
    stock. Cross-sectional (_xs) columns are placeholders filled at panel level."""
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    lr = np.log(c / c.shift(1))
    turn = c * v  # turnover proxy (TWD value traded); true field absent offline

    f = pd.DataFrame(index=df.index)

    def close_block():
        f["log_ret_1"] = lr
        f["mom_5"] = c.pct_change(5)
        f["mom_20"] = c.pct_change(20)
        f["mom_60"] = c.pct_change(60)
        # D1.2-style 126/5 momentum as a per-day input
        f["mom_126_5"] = c.shift(5) / c.shift(131) - 1.0
        f["vol_20"] = lr.rolling(20).std()
        f["vol_60"] = lr.rolling(60).std()
        f["dist_hi_60"] = c / c.rolling(60).max() - 1.0
        f["dist_lo_60"] = c / c.rolling(60).min() - 1.0
        f["px_over_ma20"] = c / c.rolling(20).mean() - 1.0

    def range_block():
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        f["hl_range"] = (h - l) / c
        f["true_range"] = tr / c
        f["atr_ratio"] = (tr / c) / ((tr / c).rolling(20).mean() + 1e-9) - 1.0
        f["close_loc"] = ((c - l) / ((h - l) + 1e-9)).clip(0, 1) - 0.5
        f["oc_ret"] = c / o - 1.0
        f["overnight_gap"] = o / prev_c - 1.0
        f["range_z"] = ((h - l) / c - ((h - l) / c).rolling(20).mean()) / (((h - l) / c).rolling(20).std() + 1e-9)

    def volume_block():
        f["vol_z"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
        f["turn_z"] = (turn - turn.rolling(20).mean()) / (turn.rolling(20).std() + 1e-9)
        f["vol_ma_ratio"] = v / (v.rolling(20).mean() + 1e-9) - 1.0
        f["log_vol_chg"] = np.log((v + 1.0) / (v.shift(1) + 1.0))
        # Amihud-style illiquidity, z-scored on a trailing year
        illiq = lr.abs() / (turn / 1e9 + 1e-9)
        f["illiq_z"] = (illiq - illiq.rolling(252).mean()) / (illiq.rolling(252).std() + 1e-9)

    fs = feature_set
    if fs in ("close_only", "close_d12", "curated_full", "full_d12",
              "sector_rel", "close_regime"):
        close_block()
    if fs in ("ohlc_range", "curated_full", "full_d12"):
        range_block()
    if fs in ("volume_block", "curated_full", "full_d12"):
        volume_block()
    if fs == "ohlc_range":
        # range-only set still needs a return anchor
        f["log_ret_1"] = lr
    if fs == "volume_block":
        f["log_ret_1"] = lr

    return f


# columns that get per-date cross-sectional transforms at panel level
_XS_SPECS = {
    "close_d12":   [("mom_126_5", "xs_d12_rank")],
    "full_d12":    [("mom_126_5", "xs_d12_rank")],
    "sector_rel":  [],  # handled specially below
    "curated_full": [],
}


def _panel_xs(panel, feature_set):
    """Add cross-sectional / sector-relative columns (per-date only)."""
    for src, dst in _XS_SPECS.get(feature_set, []):
        panel[dst] = panel.groupby("date")[src].rank(pct=True) - 0.5

    if feature_set == "close_regime":
        # market-state conditioning columns (per-date, causal, broadcast to all
        # names): equal-weight index drawdown, market vol, market momentum,
        # breadth. Warmup NaNs are excluded by the all-finite window rule.
        dm = panel.groupby("date")["log_ret_1"].mean().sort_index()
        ex = np.exp(dm.cumsum())
        mkt_dd = ex / ex.rolling(252, min_periods=60).max() - 1.0
        mkt_vol = dm.rolling(20).std()
        mkt_mom = ex.pct_change(20)
        breadth = (panel.assign(above=(panel["px_over_ma20"] > 0))
                        .groupby("date")["above"].mean() - 0.5)
        for name, ser in [("mkt_dd", mkt_dd), ("mkt_vol20", mkt_vol),
                          ("mkt_mom20", mkt_mom), ("breadth", breadth)]:
            panel[name] = panel["date"].map(ser)
        return FEATURE_COLS["close_only"] + ["mkt_dd", "mkt_vol20",
                                             "mkt_mom20", "breadth"]

    if feature_set in ("sector_rel", "curated_full", "full_d12"):
        g = panel.groupby(["date", "sector"])
        gu = panel.groupby("date")
        for col, dst in [("log_ret_1", "ret_vs_sector"), ("mom_20", "mom20_vs_sector"),
                         ("mom_60", "mom60_vs_sector")]:
            sec_mean = g[col].transform("mean")
            uni_mean = gu[col].transform("mean")
            n_sec = g[col].transform("count")
            base = sec_mean.where(n_sec >= 3, uni_mean)
            panel[dst] = panel[col] - base
        panel["vol20_vs_sector"] = panel["vol_20"] - panel.groupby(["date", "sector"])["vol_20"].transform("mean")
        if feature_set == "sector_rel":
            keep = ["ret_vs_sector", "mom20_vs_sector", "mom60_vs_sector",
                    "vol20_vs_sector", "mom_126_5", "vol_20", "log_ret_1", "mom_20"]
            return keep
    return None


FEATURE_COLS = {
    "close_only": ["log_ret_1", "mom_5", "mom_20", "mom_60", "mom_126_5",
                   "vol_20", "vol_60", "dist_hi_60", "dist_lo_60", "px_over_ma20"],
    "close_d12": ["log_ret_1", "mom_5", "mom_20", "mom_60", "mom_126_5",
                  "vol_20", "vol_60", "dist_hi_60", "dist_lo_60", "px_over_ma20",
                  "xs_d12_rank"],
    "close_regime": ["log_ret_1", "mom_5", "mom_20", "mom_60", "mom_126_5",
                     "vol_20", "vol_60", "dist_hi_60", "dist_lo_60",
                     "px_over_ma20", "mkt_dd", "mkt_vol20", "mkt_mom20",
                     "breadth"],
    "ohlc_range": ["log_ret_1", "hl_range", "true_range", "atr_ratio", "close_loc",
                   "oc_ret", "overnight_gap", "range_z"],
    "volume_block": ["log_ret_1", "vol_z", "turn_z", "vol_ma_ratio", "log_vol_chg",
                     "illiq_z"],
    "sector_rel": ["ret_vs_sector", "mom20_vs_sector", "mom60_vs_sector",
                   "vol20_vs_sector", "mom_126_5", "vol_20", "log_ret_1", "mom_20"],
    "curated_full": ["log_ret_1", "mom_5", "mom_20", "mom_126_5", "vol_20",
                     "dist_hi_60", "px_over_ma20", "hl_range", "close_loc",
                     "overnight_gap", "vol_z", "turn_z", "ret_vs_sector",
                     "mom20_vs_sector"],
    "full_d12": ["log_ret_1", "mom_5", "mom_20", "mom_126_5", "vol_20",
                 "dist_hi_60", "px_over_ma20", "hl_range", "close_loc",
                 "overnight_gap", "vol_z", "turn_z", "ret_vs_sector",
                 "mom20_vs_sector", "xs_d12_rank"],
}


# ---------------------------------------------------------------------------
# Targets (all start at close[t+exec_lag]; NaN where window incomplete)
# ---------------------------------------------------------------------------
def _fwd_ret(c, Y, exec_lag):
    n = len(c)
    out = np.full(n, np.nan)
    stop = n - Y - exec_lag
    if stop > 0:
        out[:stop] = c[Y + exec_lag:] / c[exec_lag:exec_lag + stop] - 1.0
    return out


def _barrier(c, Y, exec_lag, H=0.12, L=0.06):
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(n - Y - exec_lag):
        base = c[t + exec_lag]
        label = 0.0
        for p in c[t + exec_lag + 1:t + exec_lag + Y + 1]:
            if p >= base * (1 + H):
                label = 1.0
                break
            if p <= base * (1 - L):
                break
        out[t] = label
    return out


def build_dataset(feature_set="close_only", seq_len=40, horizons=(5, 10, 20),
                  exec_lag=1, universe=None, min_names_per_date=30,
                  include_barrier=False, verbose=True):
    """Build the full sequence dataset. Returns a dict; see module docstring."""
    ids = universe or [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    if verbose:
        print(f"[dataset] {len(uni)} stocks loaded, feature_set={feature_set}, "
              f"seq_len={seq_len}, exec_lag={exec_lag}")

    # ---- per-stock features + forward returns -> long panel
    blocks = []
    for sid, df in uni.items():
        # some cached CSVs carry duplicated dates (twstock refetch artifacts)
        df = (df.sort_values("date").drop_duplicates("date", keep="last")
                .reset_index(drop=True))
        feats = _stock_features(df, feature_set)
        c = df["close"].to_numpy(np.float64)
        blk = feats
        blk["date"] = df["date"].values
        blk["stock"] = sid
        blk["sector"] = SECTOR_MAP.get(sid, "other")
        blk["t_in_stock"] = np.arange(len(df))
        blk["n_stock"] = len(df)
        for Y in horizons:
            blk[f"fwd_{Y}"] = _fwd_ret(c, Y, exec_lag)
        # realized 20d vol at t (annualization-free; used by tgt_voladj_20)
        lr = np.log(c[1:] / c[:-1])
        blk["_vol20_t"] = pd.Series(np.concatenate([[np.nan], lr])).rolling(20).std().to_numpy()
        if include_barrier:
            blk["tgt_barrier"] = _barrier(c, 20, exec_lag)
        blocks.append(blk)
    panel = pd.concat(blocks, ignore_index=True)

    keep_override = _panel_xs(panel, feature_set)
    cols = keep_override or FEATURE_COLS[feature_set]

    # ---- cross-sectional targets per date
    gd = panel.groupby("date")
    date_count = gd["stock"].transform("count")
    for Y in horizons:
        r = panel[f"fwd_{Y}"]
        pct = panel.groupby("date")[f"fwd_{Y}"].rank(pct=True)
        tgt = 2.0 * (pct - 0.5)
        tgt[date_count < min_names_per_date] = np.nan
        panel[f"tgt_rank_{Y}"] = tgt.where(r.notna())
    # 20d excess vs universe / sector
    if 20 in horizons:
        uni_mean = panel.groupby("date")["fwd_20"].transform("mean")
        panel["tgt_exc_uni_20"] = panel["fwd_20"] - uni_mean
        sec_mean = panel.groupby(["date", "sector"])["fwd_20"].transform("mean")
        n_sec = panel.groupby(["date", "sector"])["fwd_20"].transform("count")
        panel["tgt_exc_sec_20"] = panel["fwd_20"] - sec_mean.where(n_sec >= 3, uni_mean)
        # vol-adjusted 20d forward return, cross-sectionally ranked to [-1, 1]
        raw_va = panel["fwd_20"] / panel["_vol20_t"].clip(lower=1e-4)
        pct_va = raw_va.groupby(panel["date"]).rank(pct=True)
        tva = 2.0 * (pct_va - 0.5)
        tva[date_count < min_names_per_date] = np.nan
        panel["tgt_voladj_20"] = tva.where(raw_va.notna())
        # avoid-bottom-quintile: discriminate only within the bottom 30% of the
        # 20d cross-section; everything above saturates at +1
        pct20 = panel.groupby("date")["fwd_20"].rank(pct=True)
        tab = (pct20.clip(upper=0.3) / 0.3) * 2.0 - 1.0
        tab[date_count < min_names_per_date] = np.nan
        panel["tgt_avoid_bot_20"] = tab.where(panel["fwd_20"].notna())

    # ---- global trading calendar ranks
    all_dates = np.array(sorted(panel["date"].unique()))
    rank_of = {d: i for i, d in enumerate(all_dates)}
    panel["date_rank"] = panel["date"].map(rank_of).astype(np.int32)

    # ---- assemble window samples per stock
    panel = panel.sort_values(["stock", "date"]).reset_index(drop=True)
    feat_mat = panel[cols].to_numpy(np.float32)
    np.clip(feat_mat, -10.0, 10.0, out=feat_mat)

    target_names = [f"tgt_rank_{Y}" for Y in horizons]
    if 20 in horizons:
        target_names += ["tgt_exc_uni_20", "tgt_exc_sec_20", "fwd_20",
                         "tgt_voladj_20", "tgt_avoid_bot_20"]
    for Y in horizons:
        if f"fwd_{Y}" not in target_names:
            target_names.append(f"fwd_{Y}")
    if include_barrier:
        target_names.append("tgt_barrier")

    X_list, meta = [], {"stock": [], "date_rank": [], "sector": []}
    tgt_list = {t: [] for t in target_names}
    label_end = {Y: [] for Y in horizons}

    stocks = panel["stock"].unique()
    stock_idx_of = {s: i for i, s in enumerate(stocks)}
    sectors = sorted(panel["sector"].unique())
    sector_idx_of = {s: i for i, s in enumerate(sectors)}

    for sid, g in panel.groupby("stock", sort=False):
        gi = g.index.to_numpy()
        F = feat_mat[gi]                       # (n, F) this stock, time-ordered
        dr = g["date_rank"].to_numpy()
        n = len(g)
        finite = np.isfinite(F).all(axis=1)
        # valid window end t: rows t-seq_len+1..t all finite
        ok = np.zeros(n, dtype=bool)
        csum = np.cumsum(finite.astype(np.int32))
        for t in range(seq_len - 1, n):
            prev = csum[t - seq_len] if t >= seq_len else 0
            ok[t] = (csum[t] - prev) == seq_len
        ts = np.nonzero(ok)[0]
        if len(ts) == 0:
            continue
        # windows via stride tricks
        win = np.lib.stride_tricks.sliding_window_view(F, (seq_len, F.shape[1]))[:, 0]
        X_list.append(win[ts - seq_len + 1].copy())
        meta["stock"].extend([stock_idx_of[sid]] * len(ts))
        meta["date_rank"].extend(dr[ts])
        meta["sector"].extend([sector_idx_of[g["sector"].iloc[0]]] * len(ts))
        for t_name in target_names:
            tgt_list[t_name].append(g[t_name].to_numpy(np.float32)[ts])
        for Y in horizons:
            le = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
            idx_end = np.arange(n) + exec_lag + Y
            has = idx_end < n
            le[has] = dr[idx_end[has]]
            label_end[Y].append(le[ts])

    X = np.concatenate(X_list, axis=0)
    out = {
        "X": X,
        "targets": {t: np.concatenate(tgt_list[t]) for t in target_names},
        "stock_idx": np.asarray(meta["stock"], np.int32),
        "date_rank": np.asarray(meta["date_rank"], np.int32),
        "sector_idx": np.asarray(meta["sector"], np.int32),
        "label_end_rank": {Y: np.concatenate(label_end[Y]) for Y in horizons},
        "stocks": list(stocks),
        "sectors": sectors,
        "dates": all_dates,
        "feature_set": feature_set,
        "feature_cols": cols,
        "seq_len": seq_len,
        "exec_lag": exec_lag,
        "horizons": tuple(horizons),
    }
    if verbose:
        print(f"[dataset] X {X.shape} ({X.nbytes/1e6:.0f} MB), "
              f"{len(all_dates)} dates {str(all_dates[0])[:10]}..{str(all_dates[-1])[:10]}")
    return out


# ---------------------------------------------------------------------------
# Recency weights + matured-label split
# ---------------------------------------------------------------------------
def recency_weights(label_end_rank, ref_rank, halflife=None, window=None):
    """Per-sample weight relative to the latest matured-label rank (<= ref_rank).

    halflife / window are in trading days. Samples outside `window` get weight 0.
    """
    age = ref_rank - label_end_rank.astype(np.float64)
    w = np.ones_like(age)
    if halflife:
        w = np.power(0.5, age / float(halflife))
    if window:
        w[age > window] = 0.0
    return w.astype(np.float32)


def matured_train_val(data, target, refit_rank, horizon, val_frac=0.10,
                      recency=None):
    """Indices + weights for one refit. Train/val are chronological with a purge
    of `horizon + exec_lag` trading days between them; both fully matured.

    recency: dict(halflife=, window=) or None. Weights reference the latest
    matured label rank, not the refit/feature date (brief §5).
    """
    le = data["label_end_rank"][horizon]
    y = data["targets"][target]
    usable = (le <= refit_rank) & np.isfinite(y)
    idx = np.nonzero(usable)[0]
    if len(idx) < 1000:
        raise ValueError(f"only {len(idx)} matured samples at rank {refit_rank}")
    dr = data["date_rank"][idx]
    dr_sorted = np.sort(np.unique(dr))
    v0 = dr_sorted[int(len(dr_sorted) * (1 - val_frac))]
    purge = horizon + data["exec_lag"]
    tr = idx[dr <= v0 - purge]
    va = idx[dr > v0]
    ref = int(le[idx].max())
    w = np.ones(len(tr), np.float32)
    if recency:
        w = recency_weights(le[tr], ref, recency.get("halflife"), recency.get("window"))
        keep = w > 1e-4
        tr, w = tr[keep], w[keep]
    # leakage assertions
    assert le[tr].max() <= refit_rank, "immature label leaked into train"
    assert le[va].max() <= refit_rank, "immature label leaked into val"
    assert data["date_rank"][tr].max() + purge <= data["date_rank"][va].min() + purge, "chronology violated"
    assert data["date_rank"][tr].max() <= v0 - purge < data["date_rank"][va].min(), "purge gap violated"
    return tr, va, w

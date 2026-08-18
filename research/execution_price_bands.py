"""v16 Stage B — empirical next-session price-band engine.

Implements the PRE-REGISTERED formulas in reports/continuous_research/
v16_next_session_execution/price_band_methodology.md (written before any
validation ran). Bands describe execution quality for an already-validated
signal (NEXT_OPEN_TIMING_VALIDATED); they never re-select stocks and never
produce short-entry levels. Reference prices are conditional research
estimates, not guaranteed fills. No orders, no broker APIs.

Leakage rule: for a signal dated T, calibration quantiles use observations
with date < T (validation) or <= T (production — nothing later exists at
generation time). Outcome columns (next_*) are used only as outcomes.
"""

import numpy as np
import pandas as pd

CONFIG = {
    # entry quantiles (of conditional next_open_gap): fresh vs existing
    "fresh":    {"ref": 0.50, "zone_lo": 0.25, "zone_hi": 0.60,
                 "ceil": 0.75, "chase": 0.90},
    "existing": {"ref": 0.40, "zone_lo": 0.20, "zone_hi": 0.50,
                 "ceil": 0.60, "chase": 0.75},
    # sell side
    "sell": {"ref": 0.50, "zone_hi_h": 0.60, "floor": 0.25,
             "panic_l": 0.10, "urgent_l": 0.05},
    # hold / misc
    "hold_hi_h": 0.95, "hold_lo_l": 0.05, "risk_lo_l": 0.10,
    "short_risk_h": 0.90,
    # ATR guardrails (single global set — NOT tuned; see methodology)
    "K_WIDTH": 0.25, "K_RISK": 1.50, "K_PANIC": 1.00, "K_HOLD": 1.00,
    # sample requirements + fallback
    "MIN_CELL_OBS": 400, "MIN_POOL": 750,
}

RANK_BUCKETS = ("TOP", "MID", "REST")      # <=20% / <=50% / rest
VOL_BUCKETS = ("LOW", "MED", "HIGH")       # cross-sectional vol20 terciles

# TWSE price-tick ladder (public documentation; NOT verified against a
# live feed in this repo — verify before operational reliance).
_TICKS = ((10, 0.01), (50, 0.05), (100, 0.10), (500, 0.50),
          (1000, 1.00), (float("inf"), 5.00))


def twse_tick(price):
    for lim, tick in _TICKS:
        if price < lim:
            return tick
    return 5.0


def round_to_tick(price, side):
    """Round a price to the TWSE tick ladder. side='buy' rounds DOWN
    (conservative for a buyer), side='sell' rounds UP."""
    if price is None or not np.isfinite(price) or price <= 0:
        return np.nan
    t = twse_tick(price)
    n = price / t
    n = np.floor(n + 1e-9) if side == "buy" else np.ceil(n - 1e-9)
    return round(n * t, 2)


def symbol_features(df):
    """Per-symbol daily features + next-session outcome columns.
    df = cache frame (date-sorted, deduped, columns date/open/high/low/
    close/volume). Outcomes are OUTCOME variables only (leakage rule)."""
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    n = len(c)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c),
                                       np.abs(lo - prev_c)))
    atr20 = pd.Series(tr).rolling(20).mean().to_numpy()
    lr = np.log(c / prev_c)
    out = pd.DataFrame({
        "date": df["date"].values, "close": c,
        "atr20_pct": atr20 / c,
        "vol20": pd.Series(lr).rolling(20).std().to_numpy(),
        "vol60": pd.Series(lr).rolling(60).std().to_numpy(),
    })
    nxt = np.full(n, np.nan)
    out["next_open_gap"] = np.concatenate([o[1:] / c[:-1] - 1.0, [np.nan]])
    out["next_high_from_close"] = np.concatenate([h[1:] / c[:-1] - 1.0,
                                                  [np.nan]])
    out["next_low_from_close"] = np.concatenate([lo[1:] / c[:-1] - 1.0,
                                                 [np.nan]])
    out["next_close_from_close"] = np.concatenate([c[1:] / c[:-1] - 1.0,
                                                   [np.nan]])
    del nxt
    return out


def build_history(cache_frames, score_panel=None):
    """Pooled calibration history: features+outcomes for every symbol-day,
    with cross-sectional vol terciles and (where the score panel covers
    the date) rank buckets. score_panel: DataFrame[date, stock, score] or
    None -> rank_bucket NaN everywhere (vol/global cells only)."""
    rows = []
    for sid, df in cache_frames.items():
        f = symbol_features(df)
        f["stock"] = sid
        rows.append(f)
    hist = pd.concat(rows, ignore_index=True)
    g = hist.groupby("date")["vol20"]
    q1 = g.transform(lambda s: s.quantile(1 / 3))
    q2 = g.transform(lambda s: s.quantile(2 / 3))
    hist["vol_bucket"] = np.where(hist["vol20"] <= q1, "LOW",
                          np.where(hist["vol20"] <= q2, "MED", "HIGH"))
    hist.loc[hist["vol20"].isna(), "vol_bucket"] = np.nan
    hist["rank_bucket"] = np.nan
    if score_panel is not None and len(score_panel):
        sp = score_panel[["date", "stock", "score"]].dropna()
        pct = sp.groupby("date")["score"].rank(ascending=False, pct=True)
        sp = sp.assign(rank_pct=pct)
        sp["rank_bucket"] = np.where(sp["rank_pct"] <= 0.2, "TOP",
                             np.where(sp["rank_pct"] <= 0.5, "MID", "REST"))
        hist = hist.merge(sp[["date", "stock", "rank_bucket"]],
                          on=["date", "stock"], how="left",
                          suffixes=("_drop", ""))
        hist = hist.drop(columns=["rank_bucket_drop"])
    return hist


def rank_bucket_of(rank_pct):
    if rank_pct is None or (isinstance(rank_pct, float) and np.isnan(rank_pct)):
        return None
    return "TOP" if rank_pct <= 0.2 else "MID" if rank_pct <= 0.5 else "REST"


class BandCalibrator:
    """Expanding-window conditional quantiles with hierarchical fallback:
    rank x vol -> vol -> global. `strict` controls the leakage boundary:
    strict=True (validation) uses date < asof; strict=False (production)
    uses date <= asof."""

    def __init__(self, hist, config=CONFIG):
        self.h = hist.dropna(subset=["next_open_gap"]).sort_values("date")
        self.cfg = config

    def _quants(self, sub):
        qs = {}
        for col, key in (("next_open_gap", "g"),
                         ("next_high_from_close", "h"),
                         ("next_low_from_close", "l")):
            v = sub[col].dropna()
            for q in (0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.75,
                      0.90, 0.95):
                qs[f"{key}{q:.2f}"] = float(v.quantile(q)) if len(v) else np.nan
        return qs

    def _samples(self, sub):
        # ORDERING INVARIANT (review 2026-08-19): the reach-confidence
        # layer slices the trailing RECENT_N observations, so the raw
        # arrays MUST be chronologically ordered oldest -> newest with a
        # deterministic within-date tiebreak — never incidental frame
        # order. Sorting here changes no cell membership, quantile,
        # fallback, or threshold (quantiles are order-invariant).
        sub = sub.sort_values(["date", "stock"], kind="stable")
        return {"gap": sub["next_open_gap"].to_numpy(np.float64),
                "low": sub["next_low_from_close"].dropna()
                .to_numpy(np.float64),
                "high": sub["next_high_from_close"].dropna()
                .to_numpy(np.float64)}

    def cell_full(self, asof, rank_bucket=None, vol_bucket=None,
                  strict=True):
        """-> (quantile dict, sample_count, fallback_level, samples dict)
        or (None, n, 'INSUFFICIENT', None). samples hold the raw
        calibration returns for range-reach estimation (B2)."""
        asof = pd.Timestamp(asof)
        base = self.h[self.h["date"] < asof] if strict \
            else self.h[self.h["date"] <= asof]
        if len(base) < self.cfg["MIN_POOL"]:
            return None, len(base), "INSUFFICIENT", None
        if rank_bucket in RANK_BUCKETS and vol_bucket in VOL_BUCKETS:
            sub = base[(base["rank_bucket"] == rank_bucket)
                       & (base["vol_bucket"] == vol_bucket)]
            if len(sub) >= self.cfg["MIN_CELL_OBS"]:
                return self._quants(sub), len(sub), "RANKxVOL", \
                    self._samples(sub)
        if vol_bucket in VOL_BUCKETS:
            sub = base[base["vol_bucket"] == vol_bucket]
            if len(sub) >= self.cfg["MIN_CELL_OBS"]:
                return self._quants(sub), len(sub), "VOL", \
                    self._samples(sub)
        return self._quants(base), len(base), "GLOBAL", \
            self._samples(base)

    def cell(self, asof, rank_bucket=None, vol_bucket=None, strict=True):
        """B1-compatible 3-tuple wrapper around cell_full."""
        q, n, fb, _ = self.cell_full(asof, rank_bucket, vol_bucket, strict)
        return q, n, fb


def entry_bands(prev_close, atr_pct, q, fresh, cfg=CONFIG):
    """Long-entry bands (fresh or existing-target flavor). Returns raw
    (unrounded) levels; NEVER a short entry."""
    p = cfg["fresh"] if fresh else cfg["existing"]
    A = atr_pct if np.isfinite(atr_pct) else 0.02
    g = lambda x: q[f"g{x:.2f}"]
    lo_q = q[f"l{cfg['risk_lo_l']:.2f}"]
    zone_lo = prev_close * (1 + g(p["zone_lo"]))
    zone_hi = prev_close * (1 + g(p["zone_hi"]))
    half = max((zone_hi - zone_lo) / 2, cfg["K_WIDTH"] * A * prev_close)
    mid = (zone_lo + zone_hi) / 2
    return {
        "reference": prev_close * (1 + g(p["ref"])),
        "ideal_zone_low": mid - half,
        "ideal_zone_high": mid + half,
        "acceptable_ceiling": prev_close * (1 + g(p["ceil"])),
        "do_not_chase_above": prev_close * (1 + g(p["chase"])),
        "risk_review_below": prev_close * (1 + min(lo_q, -cfg["K_RISK"] * A)),
    }


def sell_bands(prev_close, atr_pct, q, cfg=CONFIG):
    s = cfg["sell"]
    A = atr_pct if np.isfinite(atr_pct) else 0.02
    g = lambda x: q[f"g{x:.2f}"]
    return {
        "sell_reference": prev_close * (1 + max(g(s["ref"]), 0.0)),
        "ideal_sell_zone_low": prev_close * (1 + g(s["ref"])),
        "ideal_sell_zone_high": prev_close * (1 + q[f"h{s['zone_hi_h']:.2f}"]),
        "acceptable_sell_floor": prev_close * (1 + g(s["floor"])),
        "do_not_panic_sell_below": prev_close * (
            1 + max(q[f"l{s['panic_l']:.2f}"], -cfg["K_PANIC"] * A)),
        "urgent_risk_review_below": prev_close * (
            1 + min(q[f"l{s['urgent_l']:.2f}"], -cfg["K_RISK"] * A)),
    }


def hold_bands(prev_close, atr_pct, q, cfg=CONFIG):
    A = atr_pct if np.isfinite(atr_pct) else 0.02
    return {
        "no_action_zone_low": prev_close * (1 - cfg["K_HOLD"] * A),
        "no_action_zone_high": prev_close * (1 + cfg["K_HOLD"] * A),
        "review_below": prev_close * (
            1 + min(q[f"l{cfg['hold_lo_l']:.2f}"], -cfg["K_RISK"] * A)),
        "review_above": prev_close * (1 + q[f"h{cfg['hold_hi_h']:.2f}"]),
    }


def short_cover_bands(prev_close, atr_pct, q, cfg=CONFIG):
    """Risk/cover bands for an ACTUAL short. No short-entry bands exist
    anywhere in this module."""
    e = entry_bands(prev_close, atr_pct, q, fresh=False, cfg=cfg)
    A = atr_pct if np.isfinite(atr_pct) else 0.02
    return {
        "cover_reference": e["reference"],
        "cover_zone_low": e["ideal_zone_low"],
        "cover_zone_high": e["ideal_zone_high"],
        "risk_review_above": prev_close * (
            1 + max(q[f"h{cfg['short_risk_h']:.2f}"], cfg["K_RISK"] * A)),
    }


_BUY_KEYS = ("reference", "ideal_zone_low", "ideal_zone_high",
             "acceptable_ceiling", "do_not_chase_above",
             "cover_reference", "cover_zone_low", "cover_zone_high",
             "no_action_zone_low")
_SELL_KEYS = ("sell_reference", "ideal_sell_zone_low",
              "ideal_sell_zone_high", "acceptable_sell_floor",
              "do_not_panic_sell_below", "urgent_risk_review_below",
              "risk_review_below", "review_below", "review_above",
              "risk_review_above", "no_action_zone_high")


def round_levels(levels):
    """Tick-round every level: buy-intent levels down, sell/risk levels up."""
    out = {}
    for k, v in levels.items():
        side = "buy" if k in _BUY_KEYS else "sell"
        out[k] = round_to_tick(v, side)
    return out


# ------------------------------------------------------- B2 additions
# Range-reach probabilities (DAILY-RANGE statistics, never fill
# probabilities): see price_domain_methodology.md. `ref` = auction
# reference (normal-day assumption: previous close), so levels are
# comparable with the close-anchored calibration returns.

def reach_prob_buy(samples, ref, level):
    """Share of calibration sessions whose next-day LOW was at or below
    `level`. NOT a fill probability."""
    if samples is None or level is None or not np.isfinite(level) \
            or not np.isfinite(ref) or ref <= 0:
        return np.nan
    lr = samples["low"]
    return float((lr <= level / ref - 1 + 1e-12).mean()) if len(lr) \
        else np.nan


def reach_prob_sell(samples, ref, level):
    """Share of calibration sessions whose next-day HIGH was at or above
    `level`. NOT a fill probability."""
    if samples is None or level is None or not np.isfinite(level) \
            or not np.isfinite(ref) or ref <= 0:
        return np.nan
    hr = samples["high"]
    return float((hr >= level / ref - 1 - 1e-12).mean()) if len(hr) \
        else np.nan


def prob_open_beyond(samples, ref, level, side):
    """Share of calibration next-open gaps beyond `level` (side='above'
    or 'below'). Used for p_open_above_do_not_chase / below_panic."""
    if samples is None or level is None or not np.isfinite(level) \
            or not np.isfinite(ref) or ref <= 0:
        return np.nan
    g = samples["gap"]
    if not len(g):
        return np.nan
    x = level / ref - 1
    return float((g > x).mean()) if side == "above" else \
        float((g < x).mean())


# Range-reach calibration confidence (review patch 2, frozen by
# amendment A1 in price_domain_methodology.md). Leakage-safe: uses only
# the calibration sample itself (already restricted to < T / <= T);
# "recent" = the trailing RECENT_N observations of the date-sorted cell
# sample. Drift = recent reach minus full-sample reach at the row's
# primary level. NO year/date special-casing; describes historical
# calibration reliability ONLY — never changes actions or thresholds.
REACH_CONF = {"RECENT_N": 750, "RECENT_MIN": 200,
              "DRIFT_DEGRADED": 0.10, "DRIFT_HIGH": 0.05}


def reach_confidence(samples, ref, level, side, fallback,
                     cfg=REACH_CONF):
    """-> (confidence label, recent-vs-full drift). side='buy'|'sell'.
    HIGH / NORMAL / DEGRADED / INSUFFICIENT."""
    if samples is None or level is None or not np.isfinite(level) \
            or not np.isfinite(ref) or ref <= 0:
        return "INSUFFICIENT", np.nan
    arr = samples["low"] if side == "buy" else samples["high"]
    if not len(arr):
        return "INSUFFICIENT", np.nan
    x = level / ref - 1
    full = float((arr <= x + 1e-12).mean()) if side == "buy" \
        else float((arr >= x - 1e-12).mean())
    recent = arr[-cfg["RECENT_N"]:]
    if len(recent) < cfg["RECENT_MIN"]:
        return "DEGRADED", np.nan
    r = float((recent <= x + 1e-12).mean()) if side == "buy" \
        else float((recent >= x - 1e-12).mean())
    drift = r - full
    if fallback == "GLOBAL" or abs(drift) > cfg["DRIFT_DEGRADED"]:
        return "DEGRADED", drift
    if fallback == "RANKxVOL" and abs(drift) <= cfg["DRIFT_HIGH"]:
        return "HIGH", drift
    return "NORMAL", drift


def expected_price_quantiles(q, ref, domain=None):
    """Distribution display prices (nearest-tick, domain-clamped when
    known): expected open p10..p90, low p10..p75, high p25..p90."""
    import twse_price_domain as tpd
    out = {}
    spec = {"expected_open": ("g", (0.10, 0.25, 0.50, 0.75, 0.90)),
            "expected_low": ("l", (0.10, 0.25, 0.50, 0.75)),
            "expected_high": ("h", (0.25, 0.50, 0.75, 0.90))}
    for name, (k, qs) in spec.items():
        for p in qs:
            v = ref * (1 + q[f"{k}{p:.2f}"]) if q else np.nan
            if np.isfinite(v):
                v = domain.clamp(v, "nearest") if (domain is not None
                                                   and domain.known()) \
                    else tpd.legal_nearest(v)
            out[f"{name}_p{int(p * 100)}"] = v
    return out


def clamp_levels(levels, domain):
    """Project every band level into the legal domain (buy keys floor,
    sell keys ceil) and hard-validate each result. Raises
    twse_price_domain.PriceDomainValidationError on violation."""
    if domain is None:
        return levels
    out = {}
    for k, v in levels.items():
        side = "buy" if k in _BUY_KEYS else "sell"
        p = domain.clamp(v, side)
        domain.validate(p, context=k)
        out[k] = p
    return out


POSTURE = {
    "OPEN_LONG_NEW_SIGNAL": "EXECUTE_IN_IDEAL_ZONE",
    "OPEN_LONG_EXISTING_TARGET": "WAIT_FOR_PULLBACK",
    "ADD_LONG": "EXECUTE_WITHIN_LIMIT",
    "HOLD_LONG": "NO_ACTION",
    "REDUCE_LONG": "SELL_IF_REBOUND",
    "EXIT_LONG": "SELL_IN_IDEAL_ZONE",
    "WATCH_LONG": "WAIT_FOR_PULLBACK",
    "WATCH_NEUTRAL": "NO_ACTION",
    "BUY_TO_COVER": "RISK_REVIEW",
    "REDUCE_SHORT": "RISK_REVIEW",
    "HOLD_SHORT": "RISK_REVIEW",
    "NO_ACTION": "NO_ACTION",
    "NO_MODEL_OPINION": "NO_MODEL_OPINION",
    "POSITION_CONFLICT_REVIEW": "POSITION_CONFLICT",
}


def posture_for(user_action):
    return POSTURE.get(user_action, "NO_ACTION")


def bands_for_action(user_action, prev_close, atr_pct, q, cfg=CONFIG):
    """Dispatch: user_action -> rounded band dict (or {} when no bands
    apply). Night-time conditional levels only."""
    if q is None or prev_close is None or not np.isfinite(prev_close):
        return {}
    ua = user_action
    if ua == "OPEN_LONG_NEW_SIGNAL":
        lv = entry_bands(prev_close, atr_pct, q, fresh=True, cfg=cfg)
    elif ua in ("OPEN_LONG_EXISTING_TARGET", "ADD_LONG", "WATCH_LONG"):
        lv = entry_bands(prev_close, atr_pct, q, fresh=False, cfg=cfg)
    elif ua in ("REDUCE_LONG", "EXIT_LONG"):
        lv = sell_bands(prev_close, atr_pct, q, cfg=cfg)
    elif ua == "HOLD_LONG":
        lv = hold_bands(prev_close, atr_pct, q, cfg=cfg)
    elif ua in ("HOLD_SHORT", "REDUCE_SHORT", "BUY_TO_COVER"):
        lv = short_cover_bands(prev_close, atr_pct, q, cfg=cfg)
    else:   # NO_ACTION / NO_MODEL_OPINION / POSITION_CONFLICT_REVIEW / ...
        return {}
    return round_levels(lv)

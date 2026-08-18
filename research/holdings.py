"""Holdings normalization + position-aware user-action semantics (v16 Stage A).

Three cleanly separated concepts (v16 design):
  1. MODEL PORTFOLIO STATE — the decision book's BUY/HOLD/REDUCE/SELL/WATCH,
     which always refers to the model's own hypothetical paper book and is
     NEVER mutated here (SELL = long-book exit, NOT a short signal; WATCH =
     bullish-leaning near-miss, NOT bearish).
  2. ACTUAL USER POSITION — parsed from my_holdings.csv (new LONG/SHORT
     schema or legacy signed-shares schema), normalized so that negative
     share counts can never reach portfolio arithmetic.
  3. USER ACTION — a separate vocabulary mapped from (1) x (2).

Schema (new, preferred):
    symbol,side,shares,avg_cost,current_price,current_value,account,notes
    side  = LONG | SHORT ; shares = positive absolute quantity.
Legacy (no `side` column): shares > 0 -> LONG; shares < 0 -> SHORT with
qty = abs(shares); shares == 0 -> warned and dropped. Normalized
immediately after load. side present + negative shares = hard error
(never silently repaired).

Exposure conventions (documented in holdings_schema_design.md):
    market_value_abs      always positive (abs market value of the position)
    signed_exposure_value +value for LONG, -value for SHORT
    gross_long_value      sum of LONG market_value_abs
    gross_short_value     sum of SHORT market_value_abs
    gross_exposure        gross_long_value + gross_short_value
    net_exposure          gross_long_value - gross_short_value
Model-target comparisons for LONG targets use gross_long_value as the
denominator — a SHORT position can therefore never corrupt a long weight.

This module never generates orders. OPEN_SHORT is deliberately not part of
the vocabulary: no validated short-side model exists (v12 verdict:
short side rejected 6/6).
"""

import numpy as np
import pandas as pd

VALID_SIDES = ("LONG", "SHORT")

USER_ACTIONS = (
    "OPEN_LONG_NEW_SIGNAL", "OPEN_LONG_EXISTING_TARGET", "ADD_LONG",
    "HOLD_LONG", "REDUCE_LONG", "EXIT_LONG",
    "WATCH_LONG", "WATCH_NEUTRAL",
    "HOLD_SHORT", "REDUCE_SHORT", "BUY_TO_COVER",
    "NO_ACTION", "NO_MODEL_OPINION", "POSITION_CONFLICT_REVIEW",
)

# Tolerance defaults (documented in user_action_mapping.md):
# aligned iff |actual - target| <= max(ALIGN_REL * target, ALIGN_FLOOR).
# ALIGN_FLOOR = 2pp matches the overlay's long-standing medium_gap default;
# ALIGN_REL = 25% keeps the band proportional so an 8% target tolerates
# 6-10% actual (a realistic manual-execution corridor) while a 4% target
# tolerates 3-5%.
ALIGN_REL = 0.25
ALIGN_FLOOR = 0.02
# A REDUCE row with remaining target >= this is still a meaningful entry
# for a user holding nothing (half a typical equal-weight book slot of
# ~1/22 = 4.5%); below it the name is only worth watching.
REDUCE_ENTRY_MIN = 0.04

SHORT_DISCLAIMER = ("no validated short-side model exists (v12: short side "
                    "rejected); risk/conflict-based classification only, "
                    "never bearish alpha")


class HoldingsError(ValueError):
    """Raised for holdings-file contents that must not be silently repaired."""


def _num(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False)
                         .str.strip().replace({"": None}), errors="coerce")


def load_lots(path):
    """Read my_holdings.csv (either schema) -> (lots DataFrame, warnings).

    Lot columns: symbol, position_side (LONG/SHORT/UNKNOWN), position_qty
    (positive float or NaN), avg_cost, current_price, current_value (abs),
    account, notes, legacy_schema (bool). Zero-share lots are dropped with
    a warning. UNKNOWN side = shares unparseable (kept for REVIEW_MANUALLY).
    Raises HoldingsError on side/sign contradictions or invalid side values.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"symbol", "shares"}
    missing = need - set(df.columns)
    if missing:
        raise HoldingsError(
            f"holdings file {path} missing required columns: {sorted(missing)}")
    legacy = "side" not in df.columns
    for c in ("side", "avg_cost", "current_price", "current_value",
              "account", "notes"):
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].str.strip()
    df = df[df["symbol"] != ""].reset_index(drop=True)
    shares = _num(df["shares"])
    warnings = []

    if legacy:
        side = pd.Series(np.where(shares > 0, "LONG",
                         np.where(shares < 0, "SHORT", "UNKNOWN")),
                         index=df.index)
        qty = shares.abs()
        n_neg = int((shares < 0).sum())
        if n_neg:
            warnings.append(
                f"legacy schema: {n_neg} negative-share row(s) normalized to "
                "SHORT with positive quantity; prefer the explicit "
                "side=SHORT schema")
    else:
        side = df["side"].str.strip().str.upper()
        bad = sorted(set(side[~side.isin(VALID_SIDES) & (side != "")]))
        if bad:
            raise HoldingsError(
                f"invalid side value(s) {bad}: side must be LONG or SHORT")
        blank = side == ""
        if blank.any():
            raise HoldingsError(
                "side column present but blank for symbol(s): "
                f"{sorted(df.loc[blank, 'symbol'])} — fill LONG or SHORT")
        contradiction = shares < 0
        if contradiction.any():
            rows = df.loc[contradiction, "symbol"].tolist()
            raise HoldingsError(
                "negative shares with an explicit side column for "
                f"symbol(s) {rows}: in the side-based schema shares must be "
                "a positive absolute quantity. Not silently repaired — fix "
                "the file.")
        qty = shares

    zero = qty == 0
    if zero.any():
        warnings.append("zero-share row(s) dropped: "
                        + ", ".join(sorted(df.loc[zero, "symbol"])))
    unparse = shares.isna()
    lots = pd.DataFrame({
        "symbol": df["symbol"],
        "position_side": side.where(~unparse, "UNKNOWN"),
        "position_qty": qty,
        "avg_cost": _num(df["avg_cost"]),
        "current_price": _num(df["current_price"]),
        "current_value": _num(df["current_value"]).abs(),
        "account": df["account"],
        "notes": df["notes"],
    })
    lots["legacy_schema"] = legacy
    lots = lots[~zero.to_numpy()].reset_index(drop=True)
    return lots, warnings


def aggregate_positions(lots):
    """Aggregate lots -> one row per (symbol, side) + conflict flags.

    Returns (positions DataFrame, warnings). Columns: symbol, position_side,
    position_qty (summed), avg_cost (qty-weighted where every lot priced,
    else NaN), current_price (last explicit non-null; conflict warned),
    current_value (summed explicit abs values, NaN if none), account/notes
    (joined uniques), n_lots, both_sides (True when the symbol also has a
    position on the other side -> POSITION_CONFLICT_REVIEW downstream).
    UNKNOWN-side lots pass through un-aggregated (one row each) so the
    overlay can flag REVIEW_MANUALLY per input row.
    """
    warnings = []
    known = lots[lots["position_side"].isin(VALID_SIDES)]
    unknown = lots[~lots["position_side"].isin(VALID_SIDES)]
    rows = []
    for (sym, side), g in known.groupby(["symbol", "position_side"],
                                        sort=False):
        qty = float(g["position_qty"].sum())
        ac = np.nan
        if g["avg_cost"].notna().all() and qty > 0:
            ac = float((g["avg_cost"] * g["position_qty"]).sum() / qty)
        elif g["avg_cost"].notna().any():
            warnings.append(f"{sym} {side}: avg_cost missing on some lots — "
                            "aggregate avg_cost left blank")
        px = g["current_price"].dropna()
        if px.nunique() > 1:
            warnings.append(f"{sym} {side}: conflicting current_price across "
                            f"lots ({sorted(px.unique())}) — using the last")
        val = g["current_value"].dropna()
        rows.append({
            "symbol": sym, "position_side": side, "position_qty": qty,
            "avg_cost": ac,
            "current_price": float(px.iloc[-1]) if len(px) else np.nan,
            "current_value": float(val.sum()) if len(val) else np.nan,
            "account": "; ".join(sorted(set(g["account"]) - {""})),
            "notes": "; ".join(sorted(set(g["notes"]) - {""})),
            "n_lots": int(len(g)),
        })
        if len(g) > 1:
            warnings.append(f"{sym} {side}: {len(g)} lots aggregated "
                            f"(total qty {qty:g})")
    pos = pd.DataFrame(rows, columns=[
        "symbol", "position_side", "position_qty", "avg_cost",
        "current_price", "current_value", "account", "notes", "n_lots"])
    if len(pos):
        sides_per_sym = pos.groupby("symbol")["position_side"].nunique()
        both = set(sides_per_sym[sides_per_sym > 1].index)
        pos["both_sides"] = pos["symbol"].isin(both)
        for s in sorted(both):
            warnings.append(f"{s}: simultaneous LONG and SHORT lots — NOT "
                            "netted; flagged POSITION_CONFLICT_REVIEW")
    else:
        pos["both_sides"] = pd.Series(dtype=bool)
    if len(unknown):
        u = unknown.copy()
        u["n_lots"] = 1
        u["both_sides"] = False
        pos = pd.concat([pos, u[pos.columns.intersection(u.columns)]],
                        ignore_index=True)
        pos["both_sides"] = pos["both_sides"].fillna(False)
    return pos.reset_index(drop=True), warnings


def exposure_metrics(positions):
    """Exposure summary from positions carrying market_value_abs."""
    v = positions["market_value_abs"]
    long_v = float(v[positions["position_side"] == "LONG"].sum(skipna=True))
    short_v = float(v[positions["position_side"] == "SHORT"].sum(skipna=True))
    return {"gross_long_value": long_v, "gross_short_value": short_v,
            "gross_exposure": long_v + short_v,
            "net_exposure": long_v - short_v}


def map_user_action(*, position_side, model_action, model_target,
                    in_universe, in_book, cmp_weight,
                    universe_rank_pct=None, conflict=False, material=False,
                    align_rel=ALIGN_REL, align_floor=ALIGN_FLOOR,
                    reduce_entry_min=REDUCE_ENTRY_MIN):
    """Map (model state, actual position) -> (user_action, priority, reason).

    position_side: "LONG" | "SHORT" | "NONE" (no actual position).
    model_action:  book action string or "" (no book row).
    model_target:  book target weight (float; NaN/0 when absent).
    cmp_weight:    the user's comparison weight for LONG targets
                   (market_value_abs / gross_long_value), NaN if unknown.
    universe_rank_pct: rank/n over the scored universe (0=best), or None.
    conflict:      symbol holds both LONG and SHORT lots.
    material:      position is a material fraction of the portfolio (used
                   only to escalate NO_MODEL_OPINION priority).

    model_action semantics are the audited ones and are never reinterpreted:
    SELL = model long-book exit (NOT bearish), WATCH = bullish near-miss.
    OPEN_SHORT is not in the vocabulary (no validated short model).
    """
    act = (model_action or "").upper()
    tgt = float(model_target) if pd.notna(model_target) else 0.0

    if conflict:
        return ("POSITION_CONFLICT_REVIEW", "HIGH",
                "simultaneous LONG and SHORT lots for the same symbol — not "
                "netted; resolve the book before acting")
    if not in_universe:
        return ("NO_MODEL_OPINION", "HIGH" if material else "INFO",
                "outside the scored model universe — no model opinion, "
                "not bearish")

    if position_side == "NONE":
        if act == "BUY":
            return ("OPEN_LONG_NEW_SIGNAL", "HIGH",
                    f"fresh model entry (target {tgt:.1%}) and no actual "
                    "position — highest-freshness new-long candidate")
        if act == "HOLD" and tgt > 0:
            return ("OPEN_LONG_EXISTING_TARGET", "MEDIUM",
                    f"model already holds this (target {tgt:.1%}) but my "
                    "portfolio never entered / is absent — standing target, "
                    "less fresh than a new BUY signal")
        if act == "REDUCE":
            if tgt >= reduce_entry_min:
                return ("OPEN_LONG_EXISTING_TARGET", "LOW",
                        f"model trims but keeps a meaningful target "
                        f"({tgt:.1%} >= {reduce_entry_min:.0%}) — low-"
                        "priority standing entry")
            return ("WATCH_LONG", "LOW",
                    f"model trims to a small residual target ({tgt:.1%} < "
                    f"{reduce_entry_min:.0%}) — watch only")
        if act == "WATCH":
            return ("WATCH_LONG", "LOW",
                    "bullish-leaning near-miss (top-30% rank, not selected) "
                    "— watch, no position held")
        if act == "SELL":
            return ("NO_ACTION", "INFO",
                    "model exits its paper long and I hold nothing — "
                    "nothing to do (SELL is NOT a short signal)")
        return ("NO_ACTION", "INFO",
                "in the scored universe but not targeted and not held")

    if position_side == "LONG":
        band = max(align_rel * tgt, align_floor)
        if act == "SELL":
            return ("EXIT_LONG", "HIGH",
                    "model exits this name from its paper book and I hold "
                    "an actual long — exit review")
        if in_book and tgt > 0:
            if pd.isna(cmp_weight):
                return ("HOLD_LONG", "INFO",
                        f"model target {tgt:.1%} but my weight is not "
                        "computable (missing price/value) — comparison "
                        "unavailable")
            gap = cmp_weight - tgt
            if gap < -band:
                return ("ADD_LONG", "MEDIUM",
                        f"long {cmp_weight:.1%} vs target {tgt:.1%} "
                        f"(gap {gap * 100:+.1f}pp, band ±{band * 100:.1f}pp)"
                        " — materially underweight")
            if gap > band:
                pri = "HIGH" if gap > 2 * band else "MEDIUM"
                return ("REDUCE_LONG", pri,
                        f"long {cmp_weight:.1%} vs target {tgt:.1%} "
                        f"(gap {gap * 100:+.1f}pp, band ±{band * 100:.1f}pp)"
                        " — materially overweight")
            return ("HOLD_LONG", "LOW",
                    f"long {cmp_weight:.1%} ~ target {tgt:.1%} within "
                    f"±{band * 100:.1f}pp — aligned")
        if act == "WATCH":
            return ("HOLD_LONG", "MEDIUM",
                    "held long; model has it as a bullish near-miss (WATCH, "
                    "target 0) — hold under watch, review on further slip")
        # in universe, not in the book, not WATCH -> documented rank rule
        if universe_rank_pct is not None and universe_rank_pct <= 0.5:
            return ("REDUCE_LONG", "MEDIUM",
                    f"held long; unselected but still top-half rank "
                    f"({universe_rank_pct:.0%}) — soft reduce")
        if universe_rank_pct is None:
            return ("REDUCE_LONG", "MEDIUM",
                    "held long; unselected and rank unavailable — soft "
                    "reduce pending review")
        return ("EXIT_LONG", "HIGH",
                f"held long; unselected and bottom-half rank "
                f"({universe_rank_pct:.0%}) — exit review")

    if position_side == "SHORT":
        if in_book and tgt > 0:
            return ("BUY_TO_COVER", "HIGH",
                    f"actual SHORT against a positive model target "
                    f"({act} {tgt:.1%}) — model is bullish here; "
                    f"{SHORT_DISCLAIMER}")
        if act == "WATCH":
            return ("BUY_TO_COVER", "MEDIUM",
                    "actual SHORT against a bullish-leaning WATCH — "
                    f"conflict review; {SHORT_DISCLAIMER}")
        # SELL or unselected: SELL is a long-book exit, NEVER proof of
        # bearish alpha -> risk-based rank rule
        if universe_rank_pct is not None and universe_rank_pct <= 0.3:
            return ("BUY_TO_COVER", "MEDIUM",
                    f"actual SHORT on a top-tercile-ranked name "
                    f"({universe_rank_pct:.0%}) — model leans positive; "
                    f"{SHORT_DISCLAIMER}")
        if universe_rank_pct is not None and universe_rank_pct <= 0.5:
            return ("REDUCE_SHORT", "MEDIUM",
                    f"actual SHORT on a mid-ranked name "
                    f"({universe_rank_pct:.0%}); {SHORT_DISCLAIMER}")
        if universe_rank_pct is None:
            return ("REDUCE_SHORT", "MEDIUM",
                    f"actual SHORT, rank unavailable; {SHORT_DISCLAIMER}")
        return ("HOLD_SHORT", "LOW",
                f"actual SHORT on a bottom-half-ranked name "
                f"({universe_rank_pct:.0%}) — no conflict with the model, "
                f"but note: {SHORT_DISCLAIMER}")

    raise ValueError(f"unknown position_side {position_side!r}")

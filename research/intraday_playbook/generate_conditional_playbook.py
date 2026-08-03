"""v14 precomputed conditional playbook generator (research-only).

Reads the latest EOD decision book (+ blend ranking for short diagnostics,
+ selected_rules.json when present) and emits a per-symbol condition table
for the NEXT session. It does not trade and cannot: every row carries
live_trading_allowed=false; with EOD-only data every row carries
data_quality=DAILY_BAR_PROXY_ONLY; intermediate time checkpoints are
framework placeholders (confidence=none) until the intraday collector
exists. Personal holdings data is NOT read and never appears in outputs.

Usage: python research/intraday_playbook/generate_conditional_playbook.py
Outputs: reports/intraday_playbook/<asof>_conditional_playbook.{csv,md}
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V14_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v14_intraday_playbook")
PB_DIR = os.path.join(ROOT, "reports", "intraday_playbook")
GAP_BINS = ["<=-5", "-5..-3", "-3..-2", "-2..-1", "-1..0", "0..1", "1..2",
            "2..3", "3..5", ">=5"]
CHECKPOINTS = ["open+15m", "open+30m", "open+60m", "midday", "final-30m"]
EOD_TO_PSEUDO = {"BUY": "TOP_Q", "HOLD": "TOP_Q", "REDUCE": "TOP_Q",
                 "WATCH": "WATCH_BAND", "SELL": "MID"}
BASE = dict(data_source="EOD decision book + daily OHLCV",
            data_quality="DAILY_BAR_PROXY_ONLY", live_trading_allowed=False)


def _load_rules():
    p = os.path.join(V14_DIR, "selected_rules.json")
    if not os.path.exists(p):
        return {}
    rules = json.load(open(p, encoding="utf-8")).get("rules", [])
    return {(r["action"], r["gap_bin"]): r for r in rules}


def _heuristic(action, gap_bin, direction):
    """Conservative defaults for cells without a validated rule."""
    big_up = gap_bin in (">=5", "3..5")
    big_dn = gap_bin in ("<=-5", "-5..-3")
    if direction == "long":
        if big_up:
            return ("WATCH_ONLY", "med",
                    "avoid chasing large up-gaps (heuristic)")
        if big_dn and action in ("BUY", "HOLD"):
            return ("WATCH_ONLY", "high",
                    "large adverse gap against EOD stance — reassess before "
                    "any action (heuristic)")
        if action == "SELL":
            return ("EXIT_LONG", "med",
                    "EOD book exits this name; gap does not change that "
                    "(heuristic)")
        return ("NO_ACTION", "low", "no validated edge for this cell")
    # short_diagnostic
    if big_up:
        return ("AVOID_SHORT", "high",
                "up-gap squeeze risk on short candidate (heuristic)")
    if big_dn:
        return ("AVOID_SHORT", "med",
                "gap already moved; chasing shorts into gaps unvalidated "
                "(heuristic)")
    return ("WATCH_ONLY", "med", "short-diagnostic only; borrow unverified")


def main():
    pt = os.path.join(ROOT, "reports", "paper_trading")
    books = sorted(glob.glob(os.path.join(
        pt, "*_blend50_band10_decision_book.csv")))
    if not books:
        sys.exit("no decision book found")
    asof = os.path.basename(books[-1])[:10]
    db = pd.read_csv(books[-1], dtype={"symbol": str})
    rules = _load_rules()

    # short-diagnostic candidates: bottom blend quintile + v12 proxy flags
    from short_side_v12 import _ret20_frame, SQUEEZE_RET20, PRICE_FLOOR
    from queue_v9_lib import adv_frame
    preds = sorted(glob.glob(os.path.join(ROOT, "reports", "transformer_gpu",
                                          "*_predictions.csv")))
    shorts = pd.DataFrame()
    if preds:
        from transformer_hybrid import _cache_frames
        pr = pd.read_csv(preds[-1], dtype={"stock": str})
        mom = {sid: c["close"].to_numpy(float)[-6]
               / c["close"].to_numpy(float)[-132] - 1.0
               for sid, c in _cache_frames().items() if len(c) >= 132}
        pr = pr[pr["stock"].isin(mom)].copy()
        pr["z_tf"] = (pr["score"] - pr["score"].mean()) / (pr["score"].std() + 1e-9)
        mm = pd.Series(mom).reindex(pr["stock"]).to_numpy()
        pr["blend"] = 0.5 * pr["z_tf"] + 0.5 * (mm - np.nanmean(mm)) / (np.nanstd(mm) + 1e-9)
        bot = pr.nsmallest(max(3, round(0.2 * len(pr))), "blend")
        aux = _ret20_frame()
        aux = aux[aux["date"] == aux["date"].max()].set_index("stock")
        adv = adv_frame()
        adv = adv[adv["date"] == adv["date"].max()].set_index("stock")
        thr = adv["adv20"].quantile(1 / 3)
        ok = []
        for s in bot["stock"]:
            r20 = aux["ret20"].get(s, np.nan)
            px = aux["close"].get(s, np.nan)
            a = adv["adv20"].get(s, np.nan)
            flagged = ((np.isfinite(a) and a <= thr)
                       or (np.isfinite(r20) and r20 > SQUEEZE_RET20)
                       or (np.isfinite(px) and px < PRICE_FLOOR))
            ok.append(not flagged)
        shorts = bot[pd.Series(ok, index=bot.index)]

    rows = []

    def emit(sym, eod_action, w, rank, score, direction, pseudo):
        for gb in GAP_BINS:
            rule = rules.get((pseudo, gb))
            if rule and rule.get("oos_survives") and \
                    rule["direction"] == direction:
                label = ("ENTER_LONG_CONDITIONAL" if direction == "long"
                         else "ENTER_SHORT_CONDITIONAL")
                conf, risk = "validated", "med"
                reason = (f"proxy rule survived train/val/OOS "
                          f"(train t {rule['train']['t']:+.1f}); still "
                          "path-blind proxy evidence")
            else:
                label, risk, reason = _heuristic(eod_action, gb, direction)
                conf = "heuristic"
            rows.append({"date": asof, "symbol": sym,
                         "eod_action": eod_action, "eod_target_weight": w,
                         "eod_rank": rank, "eod_score": score,
                         "direction_bias": direction, "opening_gap_bin": gb,
                         "checkpoint": "open", "condition":
                         f"opening gap in {gb}% at 09:00 auction",
                         "suggested_action_label": label,
                         "confidence": conf, "risk_level": risk,
                         "reason": reason, **BASE})
        for cp in CHECKPOINTS:
            rows.append({"date": asof, "symbol": sym,
                         "eod_action": eod_action, "eod_target_weight": w,
                         "eod_rank": rank, "eod_score": score,
                         "direction_bias": direction, "opening_gap_bin": "any",
                         "checkpoint": cp, "condition":
                         "NO DATA — checkpoint rules pending intraday "
                         "collector", "suggested_action_label": "NO_ACTION",
                         "confidence": "none", "risk_level": "n/a",
                         "reason": "framework placeholder", **BASE})
        rows.append({"date": asof, "symbol": sym, "eod_action": eod_action,
                     "eod_target_weight": w, "eod_rank": rank,
                     "eod_score": score, "direction_bias": direction,
                     "opening_gap_bin": "any", "checkpoint": "pre-close",
                     "condition": "any open day-trade position still held",
                     "suggested_action_label": "FORCE_EXIT_EOD",
                     "confidence": "heuristic", "risk_level": "high",
                     "reason": "day-trade positions do not carry overnight "
                     "(policy)", **BASE})

    for _, r in db.iterrows():
        emit(r["symbol"], r["action"], r.get("target_weight"),
             r.get("rank"), r.get("model_score"), "long",
             EOD_TO_PSEUDO.get(r["action"], "MID"))
    held = set(db["symbol"])
    for _, r in shorts.iterrows():
        if r["stock"] in held:
            continue
        emit(r["stock"], "NOT_IN_BOOK", 0.0, None,
             round(float(r["blend"]), 3), "short_diagnostic", "BOTTOM_Q")

    os.makedirs(PB_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_p = os.path.join(PB_DIR, f"{asof}_conditional_playbook.csv")
    df.to_csv(csv_p, index=False)
    n_val = (df["confidence"] == "validated").sum()
    md = [f"# Conditional intraday playbook — {asof} (for the NEXT session)",
          "",
          "**RESEARCH ONLY. DAILY_BAR_PROXY_ONLY. live_trading_allowed="
          "false on every row. Not advice, not orders; checkpoint rows are "
          "unvalidated placeholders pending the intraday collector.**", "",
          f"- symbols: {df['symbol'].nunique()} "
          f"({len(db)} from the EOD book, "
          f"{df[df['direction_bias'] == 'short_diagnostic']['symbol'].nunique()} "
          "short-diagnostic)",
          f"- rows: {len(df)} · validated-cell rows: {n_val} · heuristic: "
          f"{(df['confidence'] == 'heuristic').sum()} · placeholders: "
          f"{(df['confidence'] == 'none').sum()}", "",
          "Open-checkpoint actions by gap bin (validated cells marked *):",
          ""]
    piv = (df[df["checkpoint"] == "open"]
           .assign(lab=lambda x: np.where(x["confidence"] == "validated",
                                          x["suggested_action_label"] + "*",
                                          x["suggested_action_label"]))
           .groupby(["direction_bias", "opening_gap_bin"], observed=True)["lab"]
           .agg(lambda s: s.value_counts().index[0]))
    md.append("```")
    md.append(piv.unstack(0).to_string())
    md.append("```")
    with open(csv_p.replace(".csv", ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[playbook {asof}] {len(df)} rows ({n_val} validated) -> {csv_p}")


if __name__ == "__main__":
    main()

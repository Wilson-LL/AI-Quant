"""Full-universe daily opportunity ranking (decision-support layer).

Exposes the COMPLETE cross-sectional production signal (the blend50
z-score written by blended_decision_book.py) that the validated
portfolio layer filters down to ~22 names — plus a user-priority view
that always puts ACTUAL holdings above unowned research candidates.

HARD GUARANTEES (tested):
- READ-ONLY toward the portfolio: no model action, book membership,
  target weight, WATCH state, or turnover behavior is altered.
- Ranking alone NEVER creates a BUY: unowned symbols only ever surface
  with their existing production action or as research context.
- No short-entry creation: OPEN_SHORT / SELL_SHORT / WATCH_SHORT do not
  exist here (the user_action vocabulary is holdings.USER_ACTIONS).
- model_agreement is DESCRIPTIVE seed dispersion (the preregistered
  seed-score-std tercile rule already used for book confidence),
  never a probability of profit.

Inputs (all read-only):
  reports/paper_trading/<asof>_blend50_universe_scores.csv
  reports/paper_trading/<asof>_blend50_band10_decision_book.csv
  reports/transformer_gpu/<asof>_predictions.csv         (coverage only)
  my_holdings.csv          (existing validated schema via holdings.py)
  reports/user_actions/history/*/*_universe_ranking.csv  (prev ranks)
Outputs:
  reports/user_actions/latest_universe_ranking.{csv,md}
  reports/user_actions/history/YYYY-MM/<asof>_universe_ranking.{csv,md}
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import holdings as hold  # noqa: E402
from data import SECTOR_MAP  # noqa: E402

STRATEGY = "blend50_band10"
OUT_DIR = os.path.join(ROOT, "reports", "user_actions")

# Preregistered DESCRIPTIVE rank-percentile bands (pct = rank/n, 0=best).
# Aligned with the existing production quantile semantics — the book is
# the top quintile (0.20) and WATCH extends to 0.30 — chosen BEFORE any
# outcome data and never optimized against returns.
SIGNAL_BANDS = (
    (0.10, "TOP_TIER"), (0.20, "STRONG"), (0.50, "POSITIVE"),
    (0.80, "NEUTRAL"), (1.01, "WEAK"),
)

# Existing preregistered seed-disagreement tercile rule (identical to the
# book's confidence field) relabeled descriptively. NOT a probability.
AGREEMENT_MAP = {"high": "HIGH", "medium": "NORMAL", "low": "LOW"}

# Tier-1 user actions = actual position requires a decision/review.
TIER1_ACTIONS = ("POSITION_CONFLICT_REVIEW", "EXIT_LONG", "BUY_TO_COVER",
                 "REDUCE_SHORT", "REDUCE_LONG")
# Severity order inside tier 1 (index = sort key).
T1_SEVERITY = {a: i for i, a in enumerate(TIER1_ACTIONS)}
ENTRY_ACTIONS = ("OPEN_LONG_NEW_SIGNAL", "OPEN_LONG_EXISTING_TARGET")

FORBIDDEN_ACTIONS = ("OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT")


def signal_strength(pct):
    if pd.isna(pct):
        return ""
    for hi, name in SIGNAL_BANDS:
        if pct <= hi:
            return name
    return "WEAK"


def _history_files(out_dir):
    """Dated ranking CSVs in history/, sorted by signal date ascending."""
    out = []
    for p in glob.glob(os.path.join(out_dir, "history", "*",
                                    "*_universe_ranking.csv")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_", os.path.basename(p))
        if m:
            out.append((m.group(1), p))
    return sorted(out)


def previous_rank_maps(out_dir, asof):
    """({sym: rank} for the most recent prior signal date, same for the
    5th-newest prior date, prior_date_or_None). Trading-session based —
    prior VALID ranking file, never calendar yesterday. Missing history
    stays missing (never faked to zero)."""
    prior = [(d, p) for d, p in _history_files(out_dir) if d < asof]
    if not prior:
        return {}, {}, None

    def load(p):
        try:
            df = pd.read_csv(p, dtype={"symbol": str})
            df = df[pd.to_numeric(df["universe_rank"],
                                  errors="coerce").notna()]
            return dict(zip(df["symbol"],
                            df["universe_rank"].astype(int)))
        except Exception:
            return {}

    m1 = load(prior[-1][1])
    m5 = load(prior[-5][1]) if len(prior) >= 5 else {}
    return m1, m5, prior[-1][0]


# Model-scope rule — mirrors the ACTUAL production training/scoring
# universe (dataset_transformer_eod.py:289:
#   ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]).
# An instrument outside this scope (e.g. an ETF) is INTENTIONALLY not
# scored — that is model scope, not a data failure, and it must never
# be confused with EOD publication coverage (the 99% production gate).
def outside_model_scope(sym):
    return SECTOR_MAP.get(sym) == "etf"


def _excluded_reason(sym, root, asof, preds_syms):
    if outside_model_scope(sym):
        return "OUTSIDE_MODEL_SCOPE"
    p = os.path.join(root, "research", "data_cache", f"{sym}.csv")
    if not os.path.isfile(p):
        return "NO_CACHE"
    try:
        df = (pd.read_csv(p, parse_dates=["date"])
              .drop_duplicates("date", keep="last"))
    except Exception:
        return "OTHER_VALIDATION_FAILURE"
    if sym in preds_syms:
        # scored by the model but dropped at the momentum join
        return "INSUFFICIENT_HISTORY" if len(df) < 132 else \
            "OTHER_VALIDATION_FAILURE"
    if str(df["date"].max())[:10] < asof:
        return "STALE_DATA"
    if len(df) < 132:
        return "INSUFFICIENT_HISTORY"
    return "OTHER_VALIDATION_FAILURE"


def _user_action_row(side, pos, book_row, in_uni, rank_pct, gl, gross):
    """Reuse the validated Stage-A mapper — no second holdings logic."""
    act = str(book_row["action"]) if book_row is not None else ""
    tgt = float(book_row["target_weight"]) if book_row is not None else 0.0
    mv = pos["market_value_abs"] if pos is not None else np.nan
    cmp_w = (mv / gl if pos is not None and side == "LONG"
             and pd.notna(mv) and gl > 0 else np.nan)
    conflict = bool(pos["both_sides"]) if pos is not None else False
    if side == "UNKNOWN":
        return ("NO_ACTION", "INFO", "shares unparseable — review manually")
    return hold.map_user_action(
        position_side=side if side in ("LONG", "SHORT") else "NONE",
        model_action=act, model_target=tgt, in_universe=in_uni,
        in_book=book_row is not None, cmp_weight=cmp_w,
        universe_rank_pct=rank_pct, conflict=conflict,
        material=(pd.notna(mv) and gross > 0 and mv / gross > 0.05))


def build_ranking(root=ROOT, holdings_path=None, asof=None, out_dir=None,
                  configured=None):
    """Build + write the full-universe ranking. Returns (df, meta).
    `configured` overrides the configured-universe list (tests only;
    production always uses SECTOR_MAP)."""
    out_dir = out_dir or OUT_DIR
    pt = os.path.join(root, "reports", "paper_trading")
    if holdings_path is None:
        holdings_path = os.path.join(root, "my_holdings.csv")

    scores_files = sorted(glob.glob(os.path.join(
        pt, "*_blend50_universe_scores.csv")))
    if asof is None:
        if not scores_files:
            raise FileNotFoundError(
                "no *_blend50_universe_scores.csv yet — run "
                "blended_decision_book.py first")
        asof = os.path.basename(scores_files[-1])[:10]
    sp = os.path.join(pt, f"{asof}_blend50_universe_scores.csv")
    if not os.path.isfile(sp):
        raise FileNotFoundError(f"no universe scores for {asof}: {sp}")
    uni = pd.read_csv(sp, dtype={"symbol": str})

    bp = os.path.join(pt, f"{asof}_{STRATEGY}_decision_book.csv")
    book = (pd.read_csv(bp, dtype={"symbol": str})
            if os.path.isfile(bp) else pd.DataFrame(
                columns=["symbol", "action", "target_weight", "rank"]))
    book_by_sym = {r["symbol"]: r for _, r in book.iterrows()}

    # coverage transparency: configured vs scored (never silently drop)
    configured = sorted(configured if configured is not None
                        else SECTOR_MAP)
    ranked_syms = set(uni["symbol"])
    pred_p = os.path.join(root, "reports", "transformer_gpu",
                          f"{asof}_predictions.csv")
    preds_syms = (set(pd.read_csv(pred_p, dtype={"stock": str})["stock"])
                  if os.path.isfile(pred_p) else set())
    excluded = [(s, _excluded_reason(s, root, asof, preds_syms))
                for s in configured if s not in ranked_syms]

    # holdings (existing validated schema; absent file -> research-only)
    positions = pd.DataFrame(columns=[
        "symbol", "position_side", "position_qty", "avg_cost",
        "market_value_abs", "both_sides"])
    if os.path.isfile(holdings_path):
        lots, _ = hold.load_lots(holdings_path)
        positions, _ = hold.aggregate_positions(lots)
    if len(positions):
        # Stage-A market-value convention (mirrors the nightly plan):
        # current_value -> qty x current_price -> qty x latest cache close
        vals = []
        for _, h in positions.iterrows():
            price = h["current_price"]
            if pd.isna(price):
                cp = os.path.join(root, "research", "data_cache",
                                  f"{h['symbol']}.csv")
                if os.path.isfile(cp):
                    try:
                        price = float(pd.read_csv(cp)["close"].iloc[-1])
                    except Exception:
                        price = np.nan
            v = h["current_value"]
            if pd.isna(v) and pd.notna(h["position_qty"]) \
                    and pd.notna(price):
                v = h["position_qty"] * price
            vals.append(v)
        positions = positions.copy()
        positions["market_value_abs"] = vals
    else:
        positions["market_value_abs"] = pd.Series(dtype=float)
    exp = (hold.exposure_metrics(positions) if len(positions)
           else {"gross_long_value": 0.0, "gross_exposure": 0.0})
    pos_by_symside = {(r["symbol"], r["position_side"]): r
                      for _, r in positions.iterrows()}
    held_syms = set(positions["symbol"])

    n = len(uni)
    m1, m5, prev_date = previous_rank_maps(out_dir, asof)

    rows = []
    all_syms = sorted(ranked_syms | held_syms)
    for sym in all_syms:
        u = uni[uni["symbol"] == sym]
        u = u.iloc[0] if len(u) else None
        b = book_by_sym.get(sym)
        rank = int(u["rank"]) if u is not None else np.nan
        pct = round(rank / n, 4) if u is not None else np.nan
        sides = [s for (s2, s) in pos_by_symside if s2 == sym] or ["NONE"]
        for side in sides:
            pos = pos_by_symside.get((sym, side))
            ua, pri, reason = _user_action_row(
                side, pos, b, u is not None, pct,
                exp["gross_long_value"], exp["gross_exposure"])
            assert ua not in FORBIDDEN_ACTIONS, \
                f"forbidden short-entry action {ua}"
            held = pos is not None
            short_risk = (side == "SHORT" and pd.notna(pct)
                          and pct <= 0.20)
            # Tier 1 = a VALIDATED position action/risk is required.
            # NO_MODEL_OPINION is manual review, never action-required —
            # a held out-of-scope asset stays prominent in tier 2 but
            # must not read as "trade needed".
            if held and (ua in TIER1_ACTIONS or short_risk):
                tier = 1
                if short_risk and ua not in TIER1_ACTIONS:
                    reason = ("held SHORT while the bullish model rank is "
                              f"in the top {pct:.0%} — position-risk "
                              "review (context only, NOT a validated "
                              "short-alpha rule); " + reason)
            elif held:
                tier = 2
                if ua == "NO_MODEL_OPINION":
                    reason = ("ACTUAL_HOLDING_OUTSIDE_MODEL_SCOPE — "
                              "模型未涵蓋，需要人工檢視（非減碼/交易訊號）； "
                              + reason)
            elif ua in ENTRY_ACTIONS:
                tier = 3
            elif ua == "WATCH_LONG" or (b is not None
                                        and b["action"] == "WATCH"):
                tier = 4
            else:
                tier = 5
            sev = T1_SEVERITY.get(ua, 8 if short_risk else 9)
            prev_rank = m1.get(sym, np.nan)
            rows.append({
                "symbol": sym, "signal_date": asof,
                "universe_rank": rank, "universe_size": n,
                "rank_percentile": pct,
                "signal_strength": signal_strength(pct),
                "model_agreement": AGREEMENT_MAP.get(
                    str(u["confidence"]) if u is not None else "", ""),
                "model_score": (round(float(u["blend_score"]), 4)
                                if u is not None else np.nan),
                "tf_score": (round(float(u["tf_score"]), 5)
                             if u is not None else np.nan),
                "seed_score_std": (round(float(u["seed_score_std"]), 5)
                                   if u is not None else np.nan),
                "sector": (u["sector"] if u is not None
                           else SECTOR_MAP.get(sym, "other")),
                "previous_rank": prev_rank,
                "rank_change_1d": (int(prev_rank - rank)
                                   if pd.notna(prev_rank)
                                   and pd.notna(rank) else np.nan),
                "rank_change_5d": (int(m5[sym] - rank)
                                   if sym in m5 and pd.notna(rank)
                                   else np.nan),
                "portfolio_member": bool(
                    b is not None and float(b["target_weight"]) > 0),
                "model_action": (str(b["action"]) if b is not None
                                 else ""),
                "target_weight": (float(b["target_weight"])
                                  if b is not None else 0.0),
                "watch_status": ("WATCH" if b is not None
                                 and b["action"] == "WATCH" else ""),
                "is_actual_holding": held,
                "holding_side": side if held else "",
                "shares": (float(pos["position_qty"]) if held else np.nan),
                "avg_cost": (float(pos["avg_cost"]) if held
                             and pd.notna(pos["avg_cost"]) else np.nan),
                "user_action": ua, "user_action_priority": pri,
                "priority_tier": tier, "_severity": sev,
                "priority_reason": reason,
            })

    df = pd.DataFrame(rows).sort_values(
        ["priority_tier", "_severity", "universe_rank"],
        na_position="last").drop(columns=["_severity"]).reset_index(
        drop=True)

    # invariants (also enforced by tests)
    ranked = df[pd.notna(df["universe_rank"])]
    assert ranked["symbol"].is_unique or \
        ranked[ranked.duplicated("symbol")]["is_actual_holding"].all(), \
        "duplicate non-holding ranked symbol"
    assert not set(df["user_action"]) & set(FORBIDDEN_ACTIONS)
    if len(ranked):
        assert int(ranked["universe_rank"].min()) == 1

    # three distinct universe scopes (never conflated):
    #   configured  — SECTOR_MAP watch universe
    #   model-eligible — configured minus intentional model-scope
    #                    exclusions (the production non-etf rule)
    #   scored      — model-eligible names actually ranked today
    # None of these ratios is the EOD publication coverage that the
    # PARTIAL_COVERAGE_MIN=0.99 production gate measures (that gate
    # keeps its own cached-universe denominator, untouched).
    n_eligible = sum(1 for s in configured if not outside_model_scope(s))
    meta = {
        "signal_date": asof, "prev_signal_date": prev_date,
        "configured_universe_count": len(configured),
        "model_eligible_count": n_eligible,
        "scored_count": n,
        "configured_coverage_ratio": round(n / len(configured), 4),
        "model_scored_coverage_ratio": (round(n / n_eligible, 4)
                                        if n_eligible else 0.0),
        "excluded": excluded,
    }
    csv_p, md_p = write_reports(df, meta, out_dir)
    meta["csv"], meta["md"] = csv_p, md_p
    return df, meta


# ------------------------------------------------------------ reports

DISCLAIMER = "以下為全 universe 模型排名，尚非正式買進訊號。"

ZH_STRENGTH = {"TOP_TIER": "極強", "STRONG": "強", "POSITIVE": "偏多",
               "NEUTRAL": "中性", "WEAK": "弱", "": "—"}
ZH_AGREE = {"HIGH": "高", "NORMAL": "中", "LOW": "低", "": "—"}


def _fmt_rank(r):
    return f"#{int(r)}" if pd.notna(r) else "—"


def ranking_md(df, meta):
    n, cfg = meta["scored_count"], meta["configured_universe_count"]
    md = [f"# AI-Quant 全市場候選排名 — {meta['signal_date']}", "",
          f"> {DISCLAIMER} 排名描述模型的相對看多強度，非報酬保證；"
          "系統不會自動下單。", "",
          f"涵蓋:關注清單 {cfg} 檔 → 模型適用 "
          f"{meta['model_eligible_count']} 檔（ETF 等非個股不在模型範圍）"
          f" → 本日已評分 {n} 檔 "
          f"(模型內涵蓋 {meta['model_scored_coverage_ratio']:.1%})。",
          "（此為排名層涵蓋統計，與夜間 99% EOD 發布完整性閘門無關。）"]
    scope_ex = [(s, r) for s, r in meta["excluded"]
                if r == "OUTSIDE_MODEL_SCOPE"]
    data_ex = [(s, r) for s, r in meta["excluded"]
               if r != "OUTSIDE_MODEL_SCOPE"]
    if scope_ex:
        md += ["模型範圍外(非錯誤):" + "、".join(
            f"{s} ({r})" for s, r in scope_ex)]
    if data_ex:
        md += ["資料未就緒:" + "、".join(
            f"{s} ({r})" for s, r in data_ex)]

    held = df[df["is_actual_holding"]]
    if len(held):
        md += ["", "## 我的實際持倉 — 優先處理", "",
               "| 股票 | 方向 | 模型排名 | 訊號強度 | 一致性 | 建議動作 "
               "| 原因 |", "|---|---|---|---|---|---|---|"]
        for _, r in held.iterrows():
            md.append(
                f"| {r['symbol']} | {r['holding_side']} "
                f"| {_fmt_rank(r['universe_rank'])}/{n} "
                f"| {ZH_STRENGTH[r['signal_strength']]} "
                f"| {ZH_AGREE[r['model_agreement']]} "
                f"| {r['user_action']} | {r['priority_reason']} |")

    entry = df[(~df["is_actual_holding"])
               & df["user_action"].isin(ENTRY_ACTIONS)]
    if len(entry):
        md += ["", "## 正式新進場候選(現有模型訊號)", "",
               "| 股票 | 模型排名 | 訊號強度 | 一致性 | 目標權重 |",
               "|---|---|---|---|---:|"]
        for _, r in entry.sort_values("universe_rank").iterrows():
            md.append(f"| {r['symbol']} | {_fmt_rank(r['universe_rank'])} "
                      f"| {ZH_STRENGTH[r['signal_strength']]} "
                      f"| {ZH_AGREE[r['model_agreement']]} "
                      f"| {r['target_weight']:.1%} |")

    research = df[(~df["is_actual_holding"]) & (~df["portfolio_member"])
                  & (~df["user_action"].isin(ENTRY_ACTIONS))
                  & pd.notna(df["universe_rank"])].nsmallest(
        15, "universe_rank")
    if len(research):
        md += ["", "## 最強但尚未正式進場(研究排名，尚非正式買進訊號)",
               "", "| 股票 | 模型排名 | 訊號強度 | 一致性 | 狀態 |",
               "|---|---|---|---|---|"]
        for _, r in research.iterrows():
            st = "觀察中" if r["watch_status"] else "未入選"
            md.append(f"| {r['symbol']} | {_fmt_rank(r['universe_rank'])} "
                      f"| {ZH_STRENGTH[r['signal_strength']]} "
                      f"| {ZH_AGREE[r['model_agreement']]} | {st} |")

    movers = df[pd.notna(df["rank_change_1d"])
                & (df["rank_change_1d"] >= 10)
                & pd.notna(df["universe_rank"])].nsmallest(
        10, "universe_rank")
    if len(movers):
        md += ["", "## 排名快速上升(排名變化非買進訊號)", ""]
        md += [f"- {r['symbol']}:{_fmt_rank(r['previous_rank'])} → "
               f"{_fmt_rank(r['universe_rank'])} (+{int(r['rank_change_1d'])})"
               for _, r in movers.iterrows()]
    elif meta["prev_signal_date"] is None:
        md += ["", "(尚無前一交易日排名紀錄,rank_change 為 unavailable。)"]

    top = df[pd.notna(df["universe_rank"])].nsmallest(30, "universe_rank")
    md += ["", "## 排名前 30", "",
           "| # | 股票 | 強度 | 一致性 | 持倉 | 模型狀態 |",
           "|---:|---|---|---|---|---|"]
    for _, r in top.iterrows():
        md.append(f"| {int(r['universe_rank'])} | {r['symbol']} "
                  f"| {ZH_STRENGTH[r['signal_strength']]} "
                  f"| {ZH_AGREE[r['model_agreement']]} "
                  f"| {r['holding_side'] or '—'} "
                  f"| {r['model_action'] or '—'} |")
    md += ["", f"完整 {n} 檔排名:latest_universe_ranking.csv", "",
           "---", "訊號強度 = 模型橫斷面排名百分位(前10% 極強、前20% 強、"
           "前50% 偏多);一致性 = 7-seed 分歧度三分位(描述性,非獲利機率)。"]
    return "\n".join(md)


def write_reports(df, meta, out_dir):
    import simplified_reports as sr
    os.makedirs(out_dir, exist_ok=True)
    hist = sr.history_dir(out_dir, meta["signal_date"])
    csv_p = os.path.join(hist,
                         f"{meta['signal_date']}_universe_ranking.csv")
    df.to_csv(csv_p, index=False)
    import shutil
    shutil.copyfile(csv_p,
                    os.path.join(out_dir, "latest_universe_ranking.csv"))
    text = ranking_md(df, meta)
    md_p = os.path.join(hist,
                        f"{meta['signal_date']}_universe_ranking.md")
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    shutil.copyfile(md_p,
                    os.path.join(out_dir, "latest_universe_ranking.md"))
    return csv_p, md_p


def top_nonportfolio(df, k=8):
    """Small candidate set for the simplified evening summary: strongest
    unowned, non-portfolio names (research context, never a BUY)."""
    c = df[(~df["is_actual_holding"]) & (~df["portfolio_member"])
           & (~df["user_action"].isin(ENTRY_ACTIONS))
           & df["signal_strength"].isin(("TOP_TIER", "STRONG"))
           & pd.notna(df["universe_rank"])]
    return c.nsmallest(k, "universe_rank")[
        ["symbol", "universe_rank", "universe_size", "signal_strength",
         "model_agreement", "watch_status"]]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--holdings", default=None)
    a = ap.parse_args(argv)
    df, meta = build_ranking(ROOT, holdings_path=a.holdings, asof=a.asof)
    print(f"universe ranking {meta['signal_date']}: "
          f"{meta['scored_count']}/{meta['model_eligible_count']} "
          f"model-eligible scored "
          f"({meta['configured_universe_count']} configured) "
          f"-> {meta['csv']}")
    if meta["excluded"]:
        print("excluded:", ", ".join(f"{s}({r})"
                                     for s, r in meta["excluded"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

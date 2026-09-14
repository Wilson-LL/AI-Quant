# -*- coding: utf-8 -*-
"""v16 — USER-FACING SIMPLIFIED REPORTS (Traditional Chinese, compact).

Presentation layer ONLY: renders the already-validated Stage B nightly
plan and Stage C1 live refresh into a ~5-10-second daily dashboard
(~20-35 lines on an ordinary day). It never computes prices, never
changes actions/thresholds/model logic; the full technical reports
remain untouched beside it.

One stock = ONE primary section (deterministic priority, documented per
the no-duplication rule):
  1. 注意        — hard risk/data issues (session/plan problems, domain
     conflicts, data errors, gapped/panic states, stale/missing/
     proxy-only quotes on transactional rows, HOLD outside review band)
     + 模型未涵蓋 (NO_MODEL_OPINION)
  2. 減碼 / 賣出參考 — valid REDUCE_LONG / EXIT_LONG (+ 空單回補)
  3. 買進參考    — valid genuine entry actions (OPEN_LONG_NEW_SIGNAL /
     OPEN_LONG_EXISTING_TARGET / ADD_LONG); WATCH_LONG NEVER here
  4. 價格偏高    — genuine entries at ABOVE_PREFERRED_EXECUTION_RANGE
  5. 觀察        — WATCH_LONG only (one line, not a table)
  6. (HOLD in range / NO_ACTION are omitted from the dashboard)

No automatic orders; prices are references, never guaranteed fills.
Internal codes (domain/proxy/state names, quantiles, reach percentages,
calibration confidence) never appear here — technical report only.
"""

import os
import shutil

import numpy as np
import pandas as pd

ENTRY_ACTIONS = ("OPEN_LONG_NEW_SIGNAL", "OPEN_LONG_EXISTING_TARGET",
                 "ADD_LONG")
SELL_ACTIONS = ("REDUCE_LONG", "EXIT_LONG")

SELL_ZH = {"REDUCE_LONG": "減碼", "EXIT_LONG": "賣出",
           "BUY_TO_COVER": "空單回補"}

# compact status vocabulary (buy side)
BUY_STATE_ZH = {"IN_IDEAL_ZONE": "可考慮",
                "BELOW_IDEAL_ZONE": "可考慮",
                "ABOVE_IDEAL_WITHIN_LIMIT": "尚可",
                "ABOVE_ACCEPTABLE_LIMIT": "偏高"}
SELL_STATE_ZH = {"IN_IDEAL_SELL_ZONE": "可考慮",
                 "ABOVE_IDEAL_SELL_ZONE": "可考慮",
                 "BELOW_IDEAL_WITHIN_FLOOR": "尚可",
                 "BELOW_ACCEPTABLE_SELL_FLOOR": "偏低"}

REAL_SOURCES = ("BEST_ASK", "BEST_BID", "TRADE_PRICE")


def _n(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{f:.2f}".rstrip("0").rstrip(".")


def _rng(lo, hi):
    if _n(lo) == "—" or _n(hi) == "—":
        return "—"
    return f"{_n(lo)}–{_n(hi)}"


def _live_price_display(r):
    if r.get("live_price_source") in REAL_SOURCES and \
            np.isfinite(float(r.get("live_price", np.nan))):
        return _n(r["live_price"])
    return "—"


# ------------------------------------------------------------ categorize

def categorize_live(r):
    """Single primary category per row (priority in module docstring)."""
    ua = r["user_action"]
    if ua == "NO_MODEL_OPINION":
        return "no_opinion"
    if ua == "WATCH_LONG":
        return "watch"
    state = str(r.get("live_execution_state") or "")
    hard_signal = str(r.get("signal_validity")) in (
        "SESSION_MISMATCH", "STALE_PLAN", "POSITION_CONFLICT")
    hard_domain = str(r.get("domain_validation_status")) in (
        "PRICE_DOMAIN_ASSUMPTION_CONFLICT", "DATA_VALIDATION_ERROR",
        "LIVE_PRICE_DOMAIN_ERROR")
    hard_state = state in ("GAPPED_THROUGH_RISK_REVIEW",
                           "URGENT_RISK_REVIEW", "REVIEW_BELOW",
                           "REVIEW_ABOVE")
    transactional = ua in ENTRY_ACTIONS + SELL_ACTIONS + ("BUY_TO_COVER",)
    bad_quote = (str(r.get("quote_freshness")) in ("STALE", "MISSING")
                 or str(r.get("live_price_source")) not in REAL_SOURCES)
    if hard_signal or hard_domain or hard_state or \
            (transactional and bad_quote):
        return "blocked"
    if ua in SELL_ACTIONS or ua == "BUY_TO_COVER":
        return "sell"
    if ua in ENTRY_ACTIONS:
        if state == "ABOVE_PREFERRED_EXECUTION_RANGE":
            return "expensive"
        return "buy"
    return "other"


def _attention_reason(r):
    """One short zh-TW line; internal codes never exposed."""
    ua = r["user_action"]
    if ua == "NO_MODEL_OPINION":
        return "模型未涵蓋"
    sv = str(r.get("signal_validity"))
    if sv == "SESSION_MISMATCH":
        return "計畫日期與今日不符，需重新產生計畫"
    if sv == "STALE_PLAN":
        return "昨夜計畫過期，需重新產生"
    if sv == "POSITION_CONFLICT":
        return "多空部位衝突，需人工確認"
    dv = str(r.get("domain_validation_status"))
    if dv == "PRICE_DOMAIN_ASSUMPTION_CONFLICT":
        return "今日價格基準可能特殊，需人工確認"
    if dv in ("DATA_VALIDATION_ERROR", "LIVE_PRICE_DOMAIN_ERROR"):
        return "市場資料異常，需人工確認"
    st = str(r.get("live_execution_state") or "")
    if st in ("GAPPED_THROUGH_RISK_REVIEW", "URGENT_RISK_REVIEW"):
        return "跌破風險區，需人工確認"
    if st in ("REVIEW_BELOW", "REVIEW_ABOVE"):
        return "價格超出檢視區間，需人工確認"
    if ua in SELL_ACTIONS:
        return "即時報價不足，今天暫不提供賣出參考"
    return "即時報價不足"


# ------------------------------------------------------------ LIVE

def live_summary_md(live, meta):
    status_zh = {"FRESH": "正常", "MIXED": "部分延遲",
                 "DEGRADED": "部分延遲"}
    data_state = status_zh.get(meta.get("market_data", ""), "不可用")
    if meta.get("mode") != "LIVE":
        data_state = "不可用（非即時）"
    md = [f"# AI-Quant 今日操作參考 — {meta['refresh_time'][:16]}", "",
          f"資料：{data_state}", "",
          "> 僅供價格與操作參考，不保證成交；系統不會自動下單。"]

    # Track A: EVERY actual position first, gated (A5/A6)
    rows = [r for _, r in live.iterrows()]
    expected = set(meta.get("holdings_symbols") or ()) | {
        str(r["symbol"]) for r in rows
        if str(r.get("position_side") or "") in HOLD_SIDES}
    sect, rendered = holdings_section_md(
        rows, price_key="live_price",
        universe_ranks=meta.get("universe_ranks"),
        expected_symbols=expected, live=True)
    check_holdings_coverage(expected, rendered, "live summary")
    md += sect

    cats = {k: [] for k in ("blocked", "sell", "buy", "expensive",
                            "watch", "no_opinion", "other")}
    for _, r in live.iterrows():
        cats[categorize_live(r)].append(r)

    if cats["buy"]:
        md += ["", "## 買進參考", "",
               "| 股票 | 參考買進 | 現價 | 狀態 |", "|---|---:|---:|---|"]
        for r in cats["buy"]:
            st = BUY_STATE_ZH.get(str(r.get("live_execution_state")),
                                  "可考慮")
            tag = " NEW" if r["user_action"] == "OPEN_LONG_NEW_SIGNAL" \
                else ""
            md.append(f"| {r['symbol']}{tag} "
                      f"| {_n(r.get('suggested_limit_reference'))} "
                      f"| {_live_price_display(r)} | {st} |")

    if cats["sell"]:
        md += ["", "## 減碼 / 賣出參考", "",
               "| 股票 | 操作 | 參考賣出 | 現價 | 狀態 |",
               "|---|---|---:|---:|---|"]
        for r in cats["sell"]:
            st = SELL_STATE_ZH.get(str(r.get("live_execution_state")),
                                   "可考慮")
            md.append(f"| {r['symbol']} | {SELL_ZH[r['user_action']]} "
                      f"| {_n(r.get('suggested_limit_reference'))} "
                      f"| {_live_price_display(r)} | {st} |")

    if cats["expensive"]:
        md += ["", "## 價格偏高", "",
               "| 股票 | 理想參考 | 現價 |", "|---|---:|---:|"]
        for r in cats["expensive"]:
            md.append(f"| {r['symbol']} | {_n(r.get('night_reference'))} "
                      f"| {_live_price_display(r)} |")
        md.append("")
        md.append("> 價格偏高代表進場成本較差，不代表模型訊號失效。")

    if cats["watch"]:
        names = []
        for r in cats["watch"]:
            st = str(r.get("live_execution_state") or "")
            mark = ("（偏高）" if st.startswith("ABOVE") else
                    "（價格合適）" if st in ("IN_IDEAL_ZONE",
                                             "BELOW_IDEAL_ZONE") else "")
            names.append(f"{r['symbol']}{mark}")
        md += ["", "## 觀察", "", "、".join(names), "",
               "> 觀察名單尚非正式買進訊號。"]

    attention = cats["blocked"] + cats["no_opinion"]
    if attention:
        md += ["", "## 注意", ""]
        for r in attention:
            md.append(f"- {r['symbol']}：{_attention_reason(r)}")

    rc = meta.get("ranking_context") or []
    if rc:
        md += ["", "## 其他高排名候選", "",
               "以下為全 universe 模型排名（研究候選），尚非正式買進訊號。",
               ""]
        for r in rc:
            st = "極強" if r["signal_strength"] == "TOP_TIER" else "強"
            px = (f"{r['live_price']:.2f}" if r.get("live_price")
                  else "無即時報價")
            md.append(f"- {r['symbol']}：排名 #{r['universe_rank']}"
                      f"/{r['universe_size']}，訊號{st}，現價 {px}")

    md += ["", "---", "完整技術資訊：", "latest_live_execution_plan.md"]
    return "\n".join(md)


# ------------------------------------------------------------ NIGHT

# ------------------------------------------------ Track A: holdings-first

class HoldingsCoverageError(RuntimeError):
    """A6 completeness gate: an actual position would be missing from a
    user-facing summary. Raised instead of silently dropping it."""


HOLD_SIDES = ("LONG", "SHORT", "UNKNOWN")

STATUS_ZH = {"IN_BOOK": "投組內", "WATCH": "觀察中",
             "RANKED_UNSELECTED": "未入選", "OUTSIDE_SCOPE": "模型未涵蓋",
             "DATA_UNAVAILABLE": "資料不足", "STALE_DATA": "資料過期",
             "PLAN_MISSING": "夜間計畫未含"}
PRI_ZH = {"HIGH": "高", "MEDIUM": "中", "LOW": "低", "INFO": "資訊"}
SIDE_ZH = {"LONG": "多", "SHORT": "空", "UNKNOWN": "?"}
ACTION_ZH = {"EXIT_LONG": "賣出", "REDUCE_LONG": "減碼", "ADD_LONG": "加碼",
             "HOLD_LONG": "續抱", "HOLD_SHORT": "續抱空單",
             "REDUCE_SHORT": "減碼空單", "BUY_TO_COVER": "空單回補",
             "POSITION_CONFLICT_REVIEW": "多空衝突檢視",
             "NO_MODEL_OPINION": "無模型意見", "NO_ACTION": "無動作",
             "WATCH_LONG": "觀察", "WATCH_NEUTRAL": "觀察"}


def _pct(v):
    return f"{v * 100:+.1f}%" if v is not None and pd.notna(v) else "—"


def _model_status(r):
    ua = str(r.get("user_action") or "")
    ma = str(r.get("model_action") or "")
    if ua == "NO_MODEL_OPINION":
        return "OUTSIDE_SCOPE"
    if ma in ("BUY", "HOLD", "REDUCE") and _f(r.get("model_rank")) is not None:
        return "IN_BOOK"
    if ma == "WATCH":
        return "WATCH"
    if ma == "SELL" or _f(r.get("universe_rank")) is not None:
        return "RANKED_UNSELECTED"
    return "DATA_UNAVAILABLE"


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def holdings_section_md(rows, price_key, universe_ranks=None,
                        expected_symbols=None, live=False):
    """`# 我的實際持倉` — EVERY actual position exactly once (A5/A6).

    rows: plan/live rows (dicts or Series) carrying position_side,
    position_qty, avg_cost, previous_close, model_action, model_rank,
    user_action, user_action_priority, user_action_reason and the
    price column `price_key` (live_price intraday).
    universe_ranks: {symbol: universe_rank} from the ranking layer.
    expected_symbols: the holdings file's symbols; any not present in
    rows is still rendered (PLAN_MISSING) — never dropped. Returns
    (md_lines, rendered_symbols)."""
    universe_ranks = universe_ranks or {}
    held = [r for r in rows if str(r.get("position_side") or "")
            in HOLD_SIDES]
    md = ["", "# 我的實際持倉", ""]
    if not held and not expected_symbols:
        md += ["（my_holdings.csv 無持倉或不存在）"]
        return md, set()
    md += ["| 股票 | 方向 | 成本 | 目前價 | 未實現損益% | 模型排名 "
           "| 模型狀態 | 正式動作 | 優先級 | 原因 |",
           "|---|---|---:|---:|---:|---:|---|---|---|---|"]
    rendered = set()
    seen = set()
    for r in held:
        sym = str(r["symbol"])
        side = str(r.get("position_side"))
        key = (sym, side)
        if key in seen:
            continue
        seen.add(key)
        rendered.add(sym)
        cost = _f(r.get("avg_cost"))
        px = _f(r.get(price_key)) if price_key else None
        px_note = ""
        if px is None:
            px = _f(r.get("previous_close"))
            px_note = "（前收）" if px is not None and live else ""
        pnl = None
        if cost and px is not None and side in ("LONG", "SHORT"):
            pnl = (px / cost - 1.0) * (1 if side == "LONG" else -1)
        status = _model_status(r)
        rk = _f(r.get("universe_rank"))
        if rk is None:
            rk = universe_ranks.get(sym)
        if rk is None:
            rk = _f(r.get("model_rank"))
        rank_txt = f"#{int(rk)}" if rk is not None else "N/A"
        if status == "DATA_UNAVAILABLE" and rk is not None:
            status = "RANKED_UNSELECTED"   # rank came from the universe map
        ua = str(r.get("user_action") or "")
        pri = PRI_ZH.get(str(r.get("user_action_priority") or ""), "—")
        if ua == "NO_MODEL_OPINION":
            reason = "需要人工檢視，不代表買進/賣出訊號"
            rank_txt = "N/A"
        elif ua in ACTION_REASON_ZH and ua != "NO_ACTION":
            reason = ACTION_REASON_ZH[ua]       # validated action stands
        elif status == "DATA_UNAVAILABLE":
            reason = "資料不足，需要人工檢視"
        else:
            reason = ACTION_REASON_ZH.get(ua, "")
        md.append(f"| {sym} | {SIDE_ZH.get(side, side)} "
                  f"| {_n(cost)} | {_n(px)}{px_note} | {_pct(pnl)} "
                  f"| {rank_txt} | {STATUS_ZH[status]} "
                  f"| {ACTION_ZH.get(ua, ua)} | {pri} | {reason} |")
    for sym in sorted(set(expected_symbols or ()) - rendered):
        rendered.add(sym)
        md.append(f"| {sym} | ? | — | — | — | N/A "
                  f"| {STATUS_ZH['PLAN_MISSING']} | 無動作 | — "
                  "| 持倉檔在夜間計畫後變更？請重跑 daily_ops；需要人工檢視 |")
    md += ["", "> 成本/損益僅為持倉背景資訊，不改變模型動作；"
           "系統不會自動下單。"]
    return md, rendered


ACTION_REASON_ZH = {
    "EXIT_LONG": "模型已將此檔移出投組", "REDUCE_LONG": "高於模型目標權重",
    "ADD_LONG": "低於模型目標權重", "HOLD_LONG": "與模型目標一致",
    "HOLD_SHORT": "模型無多方意見", "REDUCE_SHORT": "模型偏多，注意風險",
    "BUY_TO_COVER": "模型看多此檔，空單風險",
    "POSITION_CONFLICT_REVIEW": "同時持有多空部位", "NO_ACTION": "無需動作",
    "WATCH_LONG": "觀察中", "WATCH_NEUTRAL": "觀察中",
}


def check_holdings_coverage(expected_symbols, rendered_symbols, where):
    missing = sorted(set(expected_symbols) - set(rendered_symbols))
    if missing:
        raise HoldingsCoverageError(
            f"{where}: actual holdings missing from the summary: "
            f"{missing} — refusing to publish an incomplete holdings view")


def _universe_teaser(universe_top):
    """Compact evening section: strongest non-portfolio research names.
    NEVER promoted to a buy — descriptive ranking context only."""
    if universe_top is None or not len(universe_top):
        return []
    md = ["", "## 全市場強勢候選摘要", "",
          "以下為全 universe 模型排名，尚非正式買進訊號。", ""]
    for _, r in universe_top.iterrows():
        st = "極強" if r["signal_strength"] == "TOP_TIER" else "強"
        tag = "（觀察中）" if r.get("watch_status") else ""
        md.append(f"- {r['symbol']}：排名 #{int(r['universe_rank'])}"
                  f"/{int(r['universe_size'])}，訊號{st}{tag}")
    md += ["", "完整排名：latest_universe_ranking.md"]
    return md


def night_summary_md(plan, meta, universe_top=None, universe_ranks=None,
                     holdings_symbols=None):
    md = [f"# AI-Quant 明日操作參考 — {meta['intended_execution_date']}",
          "",
          "> 僅供價格與操作參考，不保證成交；系統不會自動下單。"]

    # Track A: EVERY actual position first, gated (A5/A6)
    rows = [r for _, r in plan.iterrows()]
    expected = set(holdings_symbols or ()) | {
        str(r["symbol"]) for r in rows
        if str(r.get("position_side") or "") in HOLD_SIDES}
    sect, rendered = holdings_section_md(
        rows, price_key=None, universe_ranks=universe_ranks,
        expected_symbols=expected)
    check_holdings_coverage(expected, rendered, "nightly summary")
    md += sect

    buys = plan[plan["user_action"].isin(ENTRY_ACTIONS)]
    if len(buys):
        md += ["", "## 明日買進參考", "",
               "| 股票 | 理想買進 | 可接受上限 |", "|---|---:|---:|"]
        for _, r in buys.iterrows():
            tag = " NEW" if r["user_action"] == "OPEN_LONG_NEW_SIGNAL" \
                else ""
            md.append(f"| {r['symbol']}{tag} "
                      f"| {_rng(r.get('ideal_zone_low'), r.get('ideal_zone_high'))} "
                      f"| {_n(r.get('acceptable_ceiling'))} |")

    sells = plan[plan["user_action"].isin(SELL_ACTIONS +
                                          ("BUY_TO_COVER",))]
    if len(sells):
        md += ["", "## 明日減碼 / 賣出參考", "",
               "| 股票 | 操作 | 參考賣出 |", "|---|---|---:|"]
        for _, r in sells.iterrows():
            ref = (_n(r.get("cover_reference"))
                   if r["user_action"] == "BUY_TO_COVER"
                   else _n(r.get("sell_reference")))
            md.append(f"| {r['symbol']} | {SELL_ZH[r['user_action']]} "
                      f"| {ref} |")

    watch = plan[plan["user_action"] == "WATCH_LONG"]
    if len(watch):
        md += ["", "## 明日觀察", "",
               "、".join(watch["symbol"]), "",
               "> 觀察名單尚非正式買進訊號。"]

    attention = []
    for _, r in plan[plan["user_action"] == "NO_MODEL_OPINION"].iterrows():
        attention.append(f"- {r['symbol']}：模型未涵蓋")
    if attention:
        md += ["", "## 注意", ""] + attention

    md += _universe_teaser(universe_top)

    md += ["", "---", "完整技術資訊：",
           "latest_next_session_action_plan.md"]
    return "\n".join(md)


# ------------------------------------------------------------ writers

def history_dir(out_dir, date):
    """Dated user-action outputs live under history/YYYY-MM/ (2026-08-25
    cleanup) so the user-facing folder shows only the latest_* files.
    Shared by every dated-report writer."""
    p = os.path.join(out_dir, "history", str(date)[:7])
    os.makedirs(p, exist_ok=True)
    return p


def _write(out_dir, dated_name, latest_name, text):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(history_dir(out_dir, dated_name[:10]), dated_name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    shutil.copyfile(p, os.path.join(out_dir, latest_name))
    return p


def write_night_summary(plan, meta, out_dir, universe_top=None,
                        universe_ranks=None, holdings_symbols=None):
    return _write(out_dir,
                  f"{meta['signal_date']}_next_session_summary.md",
                  "latest_next_session_summary.md",
                  night_summary_md(plan, meta, universe_top=universe_top,
                                   universe_ranks=universe_ranks,
                                   holdings_symbols=holdings_symbols))


def write_live_summary(live, meta, out_dir):
    return _write(out_dir,
                  f"{meta['session_date']}_live_execution_summary.md",
                  "latest_live_execution_summary.md",
                  live_summary_md(live, meta))

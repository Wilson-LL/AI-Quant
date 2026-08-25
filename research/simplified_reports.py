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

    md += ["", "---", "完整技術資訊：", "latest_live_execution_plan.md"]
    return "\n".join(md)


# ------------------------------------------------------------ NIGHT

def night_summary_md(plan, meta):
    md = [f"# AI-Quant 明日操作參考 — {meta['intended_execution_date']}",
          "",
          "> 僅供價格與操作參考，不保證成交；系統不會自動下單。"]

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


def write_night_summary(plan, meta, out_dir):
    return _write(out_dir,
                  f"{meta['signal_date']}_next_session_summary.md",
                  "latest_next_session_summary.md",
                  night_summary_md(plan, meta))


def write_live_summary(live, meta, out_dir):
    return _write(out_dir,
                  f"{meta['session_date']}_live_execution_summary.md",
                  "latest_live_execution_summary.md",
                  live_summary_md(live, meta))

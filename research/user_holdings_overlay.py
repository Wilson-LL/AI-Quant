"""User real-holdings overlay vs the model target book.

Compares the user's actual holdings (my_holdings.csv at repo root) against the
latest model decision book and reports coverage, model actions, and weight
gaps. This is an operational review tool ONLY:

  - it never generates orders and never says "place this order";
  - it reads data_cache, predictions, decision books, and daily diffs
    READ-ONLY; it writes only under reports/user_holdings/;
  - NOT_IN_MODEL_UNIVERSE means "no model coverage", not "bearish".

Universe / book sources (fallback order, most authoritative first):
  1. model universe  = symbols in the latest reports/transformer_gpu/
     <date>_predictions.csv (the actual scored universe);
     fallback: symbols in the latest blend50_band10 decision book;
     fallback: union of the latest per-strategy paper books.
  2. decision book:
     - blend50_band10: reports/paper_trading/<date>_blend50_band10_decision_book.csv
       (native actions BUY/HOLD/REDUCE/SELL/WATCH + previous weights);
     - blend50 / d12 / tf: reports/paper_trading/books/<date>_<strategy>.csv
       (weights only); actions are derived by comparing with the previous
       dated book of the same strategy (BUY = new, HOLD = weight kept or up,
       REDUCE = weight down, SELL = dropped); no WATCH rows exist for these.
     - d7b: no daily books are generated for the D7b variant today; the
       script says so and lists what is available instead of crashing.

Classification rules (first match wins):
  1. shares missing/unparseable                     -> REVIEW_MANUALLY
  2. symbol has no data_cache CSV                   -> NOT_IN_DATA_CACHE
  3. symbol not in the model universe               -> NOT_IN_MODEL_UNIVERSE
  4. book action SELL / WATCH (action dominates)    -> MODEL_SELL / MODEL_WATCH
  5. in book but my weight unknown                  -> PRICE_MISSING (no price
     found anywhere) / VALUE_MISSING (price known but value not computable) /
     MODEL_BUY / MODEL_HOLD / MODEL_REDUCE by action when a price exists
  6. in book, my weight known:
     action BUY -> MODEL_BUY; REDUCE -> MODEL_REDUCE; HOLD -> compare weights:
       |gap| <= medium_gap -> MATCH_TARGET
       gap  >  medium_gap  -> OVERWEIGHT_VS_MODEL
       gap  < -medium_gap  -> UNDERWEIGHT_VS_MODEL
  7. in universe, not in book                       -> IN_UNIVERSE_NOT_SELECTED

Usage:
  python research/user_holdings_overlay.py
  python research/user_holdings_overlay.py --strategy blend50_band10
  python research/user_holdings_overlay.py --date 2026-07-29 --strategy blend50_band10
  python research/user_holdings_overlay.py --holdings my_holdings.csv
  python research/user_holdings_overlay.py --large-gap 0.05 --medium-gap 0.02
  python research/user_holdings_overlay.py --dry-run
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

STRATEGIES = ("blend50_band10", "blend50", "d12", "tf", "d7b")
BOOK_ACTIONS = ("BUY", "HOLD", "REDUCE", "SELL", "WATCH")
CAVEATS = [
    "This is not trading advice.",
    "This does not generate orders.",
    "AI-Quant is currently in paper-trading / monitoring mode.",
    "NOT_IN_MODEL_UNIVERSE does not mean bearish; it means no model coverage.",
    "IN_UNIVERSE_NOT_SELECTED means covered but not currently targeted.",
    "REDUCE compares against the model's previous paper book, not necessarily "
    "my real portfolio.",
]

OUT_COLUMNS = [
    "symbol", "my_shares", "my_avg_cost", "latest_price", "my_current_value",
    "my_current_weight", "in_data_cache", "in_model_universe",
    "in_latest_decision_book", "model_strategy_used", "model_action",
    "model_target_weight", "model_previous_weight", "model_weight_change",
    "weight_gap", "model_rank", "model_score", "model_confidence", "sector",
    "classification", "review_priority", "interpretation", "notes",
]


# ---------------------------------------------------------------- inputs

def read_holdings(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"symbol", "shares"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"holdings file {path} missing required columns: {sorted(missing)}")
    for c in ("avg_cost", "current_price", "current_value", "account", "notes"):
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].str.strip()
    df = df[df["symbol"] != ""].reset_index(drop=True)
    for c in ("shares", "avg_cost", "current_price", "current_value"):
        df[c + "_num"] = pd.to_numeric(df[c].str.replace(",", "", regex=False),
                                       errors="coerce")
    return df


def cache_symbols(root):
    d = os.path.join(root, "research", "data_cache")
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(d, "*.csv"))}


def latest_close(root, sym):
    """Latest close from data_cache (read-only; dedupes duplicate dates)."""
    p = os.path.join(root, "research", "data_cache", f"{sym}.csv")
    if not os.path.isfile(p):
        return None, None
    try:
        df = pd.read_csv(p, usecols=["date", "close"])
        df = df.drop_duplicates("date", keep="last")
        row = df.iloc[-1]
        return float(row["close"]), str(row["date"])[:10]
    except Exception:
        return None, None


def model_universe(root):
    """(set of symbols, human-readable source description)."""
    pred_dir = os.path.join(root, "reports", "transformer_gpu")
    preds = sorted(glob.glob(os.path.join(pred_dir, "*_predictions.csv")))
    if preds:
        df = pd.read_csv(preds[-1], dtype={"stock": str})
        return set(df["stock"]), f"latest predictions {os.path.basename(preds[-1])}"
    dbs = sorted(glob.glob(os.path.join(root, "reports", "paper_trading",
                                        "*_blend50_band10_decision_book.csv")))
    if dbs:
        df = pd.read_csv(dbs[-1], dtype={"symbol": str})
        return set(df["symbol"]), f"fallback: decision book {os.path.basename(dbs[-1])}"
    books = glob.glob(os.path.join(root, "reports", "paper_trading", "books", "*.csv"))
    syms = set()
    for p in books:
        try:
            syms |= set(pd.read_csv(p, dtype={"stock": str})["stock"])
        except Exception:
            pass
    return syms, "fallback: union of paper-trading books" if syms else "none found"


def available_books(root):
    """{strategy: sorted list of dates with a book file}."""
    pt = os.path.join(root, "reports", "paper_trading")
    out = {}
    dates = sorted(os.path.basename(p)[:10] for p in
                   glob.glob(os.path.join(pt, "*_blend50_band10_decision_book.csv")))
    if dates:
        out["blend50_band10"] = dates
    for strat in ("blend50", "d12", "tf", "d7b"):
        ds = sorted(os.path.basename(p)[:10] for p in
                    glob.glob(os.path.join(pt, "books", f"*_{strat}.csv")))
        if ds:
            out[strat] = ds
    return out


def load_book(root, strategy, date):
    """Book as DataFrame indexed by symbol: action, target_weight,
    previous_weight, weight_change, rank, score, sector, confidence."""
    pt = os.path.join(root, "reports", "paper_trading")
    if strategy == "blend50_band10":
        p = os.path.join(pt, f"{date}_blend50_band10_decision_book.csv")
        db = pd.read_csv(p, dtype={"symbol": str})
        db = db.rename(columns={"model_score": "score"})
        cols = ["action", "target_weight", "previous_weight", "weight_change",
                "rank", "score", "sector", "confidence"]
        return db.set_index("symbol")[cols]

    # weight-only strategies: derive actions vs the previous dated book
    try:
        from data import SECTOR_MAP
    except Exception:
        SECTOR_MAP = {}
    p = os.path.join(pt, "books", f"{date}_{strategy}.csv")
    cur = pd.read_csv(p, dtype={"stock": str}).set_index("stock")
    prev_dates = [d for d in available_books(root).get(strategy, []) if d < date]
    prev_w = pd.Series(dtype=float)
    if prev_dates:
        pp = os.path.join(pt, "books", f"{prev_dates[-1]}_{strategy}.csv")
        prev_w = pd.read_csv(pp, dtype={"stock": str}).set_index("stock")["weight"]
    rows = []
    for sym, r in cur.iterrows():
        w1, w0 = float(r["weight"]), float(prev_w.get(sym, 0.0))
        action = ("BUY" if w0 == 0 else "HOLD" if w1 >= w0 - 1e-6 else "REDUCE")
        rows.append((sym, action, w1, w0, r.get("rank"), r.get("score")))
    for sym, w0 in prev_w.items():
        if sym not in cur.index:
            rows.append((sym, "SELL", 0.0, float(w0), np.nan, np.nan))
    db = pd.DataFrame(rows, columns=["symbol", "action", "target_weight",
                                     "previous_weight", "rank", "score"])
    db["weight_change"] = db["target_weight"] - db["previous_weight"]
    db["sector"] = db["symbol"].map(lambda s: SECTOR_MAP.get(s, "other"))
    db["confidence"] = "n/a"
    return db.set_index("symbol")


def diff_context(root, date, strategy, symbols):
    """Which of my symbols appear in today's daily diff for this strategy."""
    p = os.path.join(root, "reports", "continuous_research", "daily_diffs",
                     f"DIFF_{date}.md")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        text = f.read()
    m = re.search(rf"^## {re.escape(strategy)}\s.*?(?=^## |\Z)", text,
                  re.M | re.S)
    if not m:
        return {"file": p, "found_section": False}
    sec = m.group(0)
    out = {"file": p, "found_section": True, "entries": [], "exits": [],
           "deltas": []}
    for key, label in (("entries", "entries"), ("exits", "exits"),
                       ("deltas", "weight deltas")):
        lm = re.search(rf"- {re.escape(label)}[^:]*:\s*(.+)", sec)
        if lm:
            syms = set(re.findall(r"\b\d{3,4}[A-Z]?\b", lm.group(1)))
            out[key] = sorted(syms & symbols)
    return out


# ---------------------------------------------------------- classification

def classify(row, medium_gap, large_gap):
    """-> (classification, review_priority, interpretation)."""
    sym = row["symbol"]
    w_my, gap = row["my_current_weight"], row["weight_gap"]
    action = row["model_action"]
    big_holding = pd.notna(w_my) and w_my > large_gap

    if pd.isna(row["my_shares"]):
        return ("REVIEW_MANUALLY", "INFO",
                "shares missing or unparseable — row not classifiable.")
    if not row["in_data_cache"]:
        pri = "HIGH" if big_holding else "INFO"
        return ("NOT_IN_DATA_CACHE", pri,
                "Not in the data cache — AI-Quant has no data and no model "
                "opinion on this stock.")
    if not row["in_model_universe"]:
        pri = "HIGH" if big_holding else "INFO"
        return ("NOT_IN_MODEL_UNIVERSE", pri,
                "AI-Quant currently has no model opinion on this stock.")

    if action == "SELL":
        return ("MODEL_SELL", "HIGH",
                "Model paper book exits this name (rank fell out of the "
                "band). Review — this is not an order.")
    if action == "WATCH":
        pri = "HIGH" if pd.notna(gap) and abs(gap) > large_gap else "MEDIUM"
        return ("MODEL_WATCH", pri,
                "Model has this inside the widened band but does not hold "
                "it. Watch, do not trade automatically.")

    in_book = bool(row["in_latest_decision_book"])
    if in_book and pd.isna(w_my):
        if pd.isna(row["latest_price"]):
            return ("PRICE_MISSING", "INFO",
                    "No price available (holdings file blank, no data_cache "
                    "close) — membership known but weights not comparable.")
        cls = {"BUY": "MODEL_BUY", "HOLD": "MODEL_HOLD",
               "REDUCE": "MODEL_REDUCE"}.get(action, "VALUE_MISSING")
        return (cls, "INFO",
                f"Model action is {action}; my portfolio value could not be "
                "computed, so no weight comparison.")
    if in_book:
        gtxt = f"my {w_my:.1%} vs model target {row['model_target_weight']:.1%}"
        if action == "BUY":
            pri = "HIGH" if abs(gap) > large_gap else \
                  "MEDIUM" if abs(gap) > medium_gap else "LOW"
            return ("MODEL_BUY", pri,
                    f"Model book newly enters this name; {gtxt}.")
        if action == "REDUCE":
            pri = "HIGH" if gap > medium_gap else "LOW"
            return ("MODEL_REDUCE", pri,
                    "Model paper book trims this name vs its previous paper "
                    f"weight (not vs my real position); {gtxt}.")
        # HOLD -> weight comparison
        if gap > large_gap:
            return ("OVERWEIGHT_VS_MODEL", "HIGH",
                    f"Overweight vs model by {gap * 100:.1f}pp ({gtxt}). Review.")
        if gap > medium_gap:
            return ("OVERWEIGHT_VS_MODEL", "MEDIUM",
                    f"Overweight vs model by {gap * 100:.1f}pp ({gtxt}).")
        if gap < -large_gap:
            return ("UNDERWEIGHT_VS_MODEL", "HIGH",
                    f"Underweight vs model by {-gap * 100:.1f}pp ({gtxt}). Review.")
        if gap < -medium_gap:
            return ("UNDERWEIGHT_VS_MODEL", "MEDIUM",
                    f"Underweight vs model by {-gap * 100:.1f}pp ({gtxt}).")
        return ("MATCH_TARGET", "LOW",
                f"My weight is close to the model target ({gtxt}).")

    # in universe, not in the book (weight_gap is vs a 0 target)
    if pd.notna(w_my) and w_my > large_gap:
        return ("IN_UNIVERSE_NOT_SELECTED", "HIGH",
                "The model covers this stock, but it is not currently part "
                f"of the target portfolio (my weight {w_my:.1%}). Review.")
    if pd.notna(w_my) and w_my > 0.03:
        return ("IN_UNIVERSE_NOT_SELECTED", "MEDIUM",
                "The model covers this stock, but it is not currently part "
                f"of the target portfolio (my weight {w_my:.1%}).")
    pri = "INFO" if pd.isna(w_my) else "LOW"
    return ("IN_UNIVERSE_NOT_SELECTED", pri,
            "The model covers this stock, but it is not currently part of "
            "the target portfolio.")


# ------------------------------------------------------------------ core

def build_overlay(root, holdings, strategy, date, medium_gap, large_gap):
    cache = cache_symbols(root)
    universe, universe_src = model_universe(root)
    book = load_book(root, strategy, date)
    in_book_syms = set(book.index)

    rows = []
    for _, h in holdings.iterrows():
        sym = h["symbol"]
        price, price_date = (h["current_price_num"], "holdings file")
        if pd.isna(price):
            price, price_date = latest_close(root, sym)
            price = np.nan if price is None else price
        value = h["current_value_num"]
        if pd.isna(value) and pd.notna(h["shares_num"]) and pd.notna(price):
            value = h["shares_num"] * price
        b = book.loc[sym] if sym in in_book_syms else None
        rows.append({
            "symbol": sym,
            "my_shares": h["shares_num"],
            "my_avg_cost": h["avg_cost_num"],
            "latest_price": price if pd.notna(price) else np.nan,
            "my_current_value": value,
            "in_data_cache": sym in cache,
            "in_model_universe": sym in universe,
            "in_latest_decision_book": b is not None,
            "model_strategy_used": strategy,
            "model_action": b["action"] if b is not None else "",
            "model_target_weight": float(b["target_weight"]) if b is not None else np.nan,
            "model_previous_weight": float(b["previous_weight"]) if b is not None else np.nan,
            "model_weight_change": float(b["weight_change"]) if b is not None else np.nan,
            "model_rank": b["rank"] if b is not None else np.nan,
            "model_score": b["score"] if b is not None else np.nan,
            "model_confidence": b["confidence"] if b is not None else "",
            "sector": b["sector"] if b is not None else "",
            "notes": h["notes"],
            "account": h["account"],
        })
    ov = pd.DataFrame(rows)

    total_value = ov["my_current_value"].sum(skipna=True)
    ov["my_current_weight"] = (ov["my_current_value"] / total_value
                               if total_value and total_value > 0 else np.nan)
    # gap vs model target; target counts as 0 for covered-but-not-selected,
    # and stays undefined (NaN) where the model has no coverage at all
    tgt = ov["model_target_weight"].where(ov["in_latest_decision_book"], 0.0)
    tgt = tgt.where(ov["in_model_universe"], np.nan)
    ov["weight_gap"] = ov["my_current_weight"] - tgt

    cls = ov.apply(lambda r: classify(r, medium_gap, large_gap), axis=1)
    ov["classification"] = [c[0] for c in cls]
    ov["review_priority"] = [c[1] for c in cls]
    ov["interpretation"] = [c[2] for c in cls]
    return ov, total_value, universe_src


def write_outputs(root, ov, total_value, universe_src, strategy, date,
                  medium_gap, large_gap):
    out_dir = os.path.join(root, "reports", "user_holdings")
    os.makedirs(out_dir, exist_ok=True)
    csv_p = os.path.join(out_dir, f"{date}_user_holdings_overlay.csv")
    md_p = os.path.join(out_dir, f"{date}_user_holdings_overlay.md")
    json_p = os.path.join(out_dir, f"{date}_user_holdings_summary.json")

    out = ov.copy()
    for c in ("my_current_weight", "model_target_weight",
              "model_previous_weight", "model_weight_change", "weight_gap"):
        out[c] = out[c].round(5)
    out[OUT_COLUMNS].to_csv(csv_p, index=False)

    # ------- markdown
    pri_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    covered = ov[ov["in_model_universe"]]
    not_covered = ov[~ov["in_model_universe"]]
    in_book = ov[ov["in_latest_decision_book"]]
    not_sel = ov[ov["classification"] == "IN_UNIVERSE_NOT_SELECTED"]
    missing_px = ov[ov["latest_price"].isna()]
    gaps = ov.dropna(subset=["weight_gap"])
    fmt_w = lambda v: "n/a" if pd.isna(v) else f"{v:.1%}"
    fmt_g = lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.1f}pp"

    md = [f"# User Holdings Overlay — {date}", "",
          f"Strategy compared: **{strategy}** · model universe source: "
          f"{universe_src} · thresholds: medium {medium_gap:.0%} / large "
          f"{large_gap:.0%}", "",
          "## Portfolio summary", "",
          f"- holdings: {len(ov)}",
          f"- total known portfolio value: "
          f"{total_value:,.0f}" if total_value else
          "- total known portfolio value: n/a (no values computable)",
          f"- covered by model universe: {len(covered)}",
          f"- not covered: {len(not_covered)}",
          f"- in target/decision book: {len(in_book)}",
          f"- in universe but not selected: {len(not_sel)}",
          f"- missing prices: {len(missing_px)}"]
    if len(gaps):
        top_o = gaps.loc[gaps["weight_gap"].idxmax()]
        top_u = gaps.loc[gaps["weight_gap"].idxmin()]
        md.append(f"- largest overweight vs model: {top_o['symbol']} "
                  f"{fmt_g(top_o['weight_gap'])}"
                  if top_o["weight_gap"] > 0 else
                  "- largest overweight vs model: none")
        md.append(f"- largest underweight vs model: {top_u['symbol']} "
                  f"{fmt_g(top_u['weight_gap'])}"
                  if top_u["weight_gap"] < 0 else
                  "- largest underweight vs model: none")
    md += ["", "## Action summary", ""]
    groups = [("MODEL_BUY", ov[ov["model_action"] == "BUY"]),
              ("MODEL_HOLD", ov[ov["model_action"] == "HOLD"]),
              ("MODEL_REDUCE", ov[ov["model_action"] == "REDUCE"]),
              ("MODEL_SELL", ov[ov["model_action"] == "SELL"]),
              ("MODEL_WATCH", ov[ov["model_action"] == "WATCH"]),
              ("IN_UNIVERSE_NOT_SELECTED", not_sel),
              ("NOT_IN_MODEL_UNIVERSE",
               ov[ov["classification"].isin(("NOT_IN_MODEL_UNIVERSE",
                                             "NOT_IN_DATA_CACHE"))])]
    for label, g in groups:
        syms = ", ".join(g["symbol"]) if len(g) else "—"
        md.append(f"- **{label}** ({len(g)}): {syms}")

    hi = ov[ov["review_priority"] == "HIGH"]
    md += ["", "## High priority review", ""]
    if len(hi):
        md += ["| symbol | my weight | model target | gap | action | "
               "classification | interpretation |",
               "|---|---|---|---|---|---|---|"]
        for _, r in hi.iterrows():
            md.append(f"| {r['symbol']} | {fmt_w(r['my_current_weight'])} | "
                      f"{fmt_w(r['model_target_weight'])} | "
                      f"{fmt_g(r['weight_gap'])} | {r['model_action'] or '—'} | "
                      f"{r['classification']} | {r['interpretation']} |")
    else:
        md.append("None.")

    md += ["", "## Holdings not covered by model", ""]
    if len(not_covered):
        for _, r in not_covered.iterrows():
            md.append(f"- **{r['symbol']}** (my weight "
                      f"{fmt_w(r['my_current_weight'])}): "
                      "AI-Quant currently has no model opinion on this stock.")
    else:
        md.append("None — all holdings are in the model universe.")

    md += ["", "## Holdings covered but not selected", ""]
    if len(not_sel):
        for _, r in not_sel.iterrows():
            md.append(f"- **{r['symbol']}** (my weight "
                      f"{fmt_w(r['my_current_weight'])}): The model covers "
                      "this stock, but it is not currently part of the "
                      "target portfolio.")
    else:
        md.append("None.")

    md += ["", "## Large mismatches (|gap| > "
           f"{medium_gap:.0%})", ""]
    mm = gaps[gaps["weight_gap"].abs() > medium_gap].sort_values(
        "weight_gap", key=abs, ascending=False)
    if len(mm):
        md += ["| symbol | my weight | model target | gap | direction |",
               "|---|---|---|---|---|"]
        for _, r in mm.iterrows():
            side = "overweight" if r["weight_gap"] > 0 else "underweight"
            md.append(f"| {r['symbol']} | {fmt_w(r['my_current_weight'])} | "
                      f"{fmt_w(r['model_target_weight'])} | "
                      f"{fmt_g(r['weight_gap'])} | {side} |")
    else:
        md.append("None.")

    md += ["", "## Daily diff context", ""]
    dc = diff_context(root, date, strategy, set(ov["symbol"]))
    if dc is None:
        md.append(f"No daily diff found for {date}.")
    elif not dc.get("found_section"):
        md.append(f"Daily diff exists but has no `{strategy}` section.")
    else:
        any_hit = False
        for key, label in (("entries", "entries"), ("exits", "exits"),
                           ("deltas", "weight deltas >1pp")):
            if dc[key]:
                md.append(f"- my holdings in today's {label}: "
                          + ", ".join(dc[key]))
                any_hit = True
        if not any_hit:
            md.append("None of my holdings appeared in today's entries, "
                      "exits, or >1pp weight deltas.")

    md += ["", "## Important caveats", ""]
    md += [f"- {c}" for c in CAVEATS]

    with open(md_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    summary = {
        "date": date, "strategy": strategy, "universe_source": universe_src,
        "n_holdings": int(len(ov)),
        "total_known_value": float(total_value) if total_value else None,
        "n_covered": int(len(covered)), "n_not_covered": int(len(not_covered)),
        "n_in_book": int(len(in_book)), "n_not_selected": int(len(not_sel)),
        "n_missing_price": int(len(missing_px)),
        "priority_counts": ov["review_priority"].value_counts().to_dict(),
        "classification_counts": ov["classification"].value_counts().to_dict(),
        "thresholds": {"medium_gap": medium_gap, "large_gap": large_gap},
    }
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    shutil.copyfile(csv_p, os.path.join(out_dir, "latest_user_holdings_overlay.csv"))
    shutil.copyfile(md_p, os.path.join(out_dir, "latest_user_holdings_overlay.md"))
    return csv_p, md_p, json_p


def run(root, holdings_path, strategy, date, medium_gap, large_gap,
        dry_run=False):
    if not os.path.isabs(holdings_path):
        holdings_path = os.path.join(root, holdings_path)
    if not os.path.isfile(holdings_path):
        print(f"my_holdings.csv not found ({holdings_path}). Skipping user "
              "holdings overlay. Copy my_holdings.example.csv to "
              "my_holdings.csv to enable.")
        return 0

    books = available_books(root)
    if strategy not in books:
        print(f"No decision book files found for strategy '{strategy}'.")
        if strategy == "d7b":
            print("D7b (band15 + cap7.5) is a documented deployment variant; "
                  "no daily D7b books are generated today.")
        print("Available latest decision books:")
        for s, ds in books.items():
            print(f"  {s}: latest {ds[-1]} ({len(ds)} dates)")
        return 0
    if date is None:
        date = books[strategy][-1]
    elif date not in books[strategy]:
        print(f"No {strategy} book for {date}. Available dates: "
              + ", ".join(books[strategy]))
        return 0

    if dry_run:
        universe, universe_src = model_universe(root)
        out_dir = os.path.join(root, "reports", "user_holdings")
        pt = os.path.join(root, "reports", "paper_trading")
        book_p = (os.path.join(pt, f"{date}_blend50_band10_decision_book.csv")
                  if strategy == "blend50_band10"
                  else os.path.join(pt, "books", f"{date}_{strategy}.csv"))
        print("[dry-run] would read:")
        print(f"  holdings:      {holdings_path}")
        print(f"  decision book: {book_p}")
        print(f"  universe:      {universe_src} ({len(universe)} symbols)")
        print(f"  data_cache:    research/data_cache/<symbol>.csv (read-only)")
        print("[dry-run] would write:")
        for name in (f"{date}_user_holdings_overlay.csv",
                     f"{date}_user_holdings_overlay.md",
                     f"{date}_user_holdings_summary.json",
                     "latest_user_holdings_overlay.csv",
                     "latest_user_holdings_overlay.md"):
            print(f"  {os.path.join(out_dir, name)}")
        print("[dry-run] no files written.")
        return 0

    holdings = read_holdings(holdings_path)
    if holdings.empty:
        print(f"holdings file {holdings_path} has no rows — nothing to "
              "overlay. Add your positions to it first.")
        return 0
    ov, total_value, universe_src = build_overlay(
        root, holdings, strategy, date, medium_gap, large_gap)
    csv_p, md_p, json_p = write_outputs(
        root, ov, total_value, universe_src, strategy, date,
        medium_gap, large_gap)
    n_hi = int((ov["review_priority"] == "HIGH").sum())
    print(f"[overlay {date} vs {strategy}] {len(ov)} holdings, "
          f"{int(ov['in_latest_decision_book'].sum())} in book, "
          f"{n_hi} HIGH priority -> {csv_p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--holdings", default="my_holdings.csv")
    ap.add_argument("--strategy", default="blend50_band10", choices=STRATEGIES)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest)")
    ap.add_argument("--large-gap", type=float, default=0.05)
    ap.add_argument("--medium-gap", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(ROOT, a.holdings, a.strategy, a.date,
               a.medium_gap, a.large_gap, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())

"""P2 — daily diff report for the paper-trading books (operational tooling).

For an as-of date with saved books (reports/paper_trading/books/<date>_<strat>.csv)
emit reports/continuous_research/daily_diffs/DIFF_<date>.md with: entries/exits
vs the previous snapshot, weight deltas >1pp, turnover + cost estimate, sector
shares, cross-book top-quintile overlap, and anomaly flags.

Usage: python research/daily_diff_report.py [--asof YYYY-MM-DD]
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import SECTOR_MAP  # noqa: E402

BOOK_DIR = os.path.join(ROOT, "reports", "paper_trading", "books")
DIFF_DIR = os.path.join(ROOT, "reports", "continuous_research", "daily_diffs")
STRATS = ("d12", "tf", "blend50", "blend50_band10")


def _load(asof, strat):
    p = os.path.join(BOOK_DIR, f"{asof}_{strat}.csv")
    return pd.read_csv(p, dtype={"stock": str}) if os.path.exists(p) else None


def main(asof=None):
    dates = sorted({f[:10] for f in os.listdir(BOOK_DIR)})
    if not dates:
        sys.exit("no books found")
    if asof is None:
        asof = dates[-1]
    prior = [d for d in dates if d < asof]
    prev_date = prior[-1] if prior else None

    lines = [f"# Daily book diff — {asof}",
             f"(previous snapshot: {prev_date or 'none — first book'})", ""]
    anomalies = []
    books = {}
    for strat in STRATS:
        cur = _load(asof, strat)
        if cur is None:
            continue
        books[strat] = cur
        w1 = cur.set_index("stock")["weight"]
        lines += [f"## {strat}  ({len(cur)} names, maxW {w1.max():.1%})", ""]
        if prev_date:
            prev = _load(prev_date, strat)
            if prev is not None:
                w0 = prev.set_index("stock")["weight"]
                alln = w0.index.union(w1.index)
                dw = w1.reindex(alln, fill_value=0) - w0.reindex(alln, fill_value=0)
                turn = float(dw.abs().sum() / 2)
                entries = sorted(set(w1.index) - set(w0.index))
                exits = sorted(set(w0.index) - set(w1.index))
                big = dw[dw.abs() > 0.01].sort_values()
                lines += [f"- turnover (one-way): **{turn:.3f}** · est cost of "
                          f"today's trades: {2*turn*60/1e4:.2%} @60bps / "
                          f"{2*turn*150/1e4:.2%} @150bps",
                          f"- entries ({len(entries)}): {', '.join(entries) or '—'}",
                          f"- exits ({len(exits)}): {', '.join(exits) or '—'}"]
                if len(big):
                    lines.append("- weight deltas >1pp: " + ", ".join(
                        f"{s} {d:+.1%}" for s, d in big.items()))
                if turn > 0.5:
                    anomalies.append(f"{strat}: turnover {turn:.2f} > 0.50")
        sec = pd.Series([SECTOR_MAP.get(s, "other") for s in w1.index], index=w1.index)
        shares = w1.groupby(sec).sum().sort_values(ascending=False)
        lines.append("- sector shares: " + ", ".join(
            f"{k} {v:.0%}" for k, v in shares.head(4).items()))
        if shares.iloc[0] > 0.5:
            anomalies.append(f"{strat}: top sector {shares.index[0]} at "
                             f"{shares.iloc[0]:.0%} > 50%")
        lines.append("")
    if len(books) >= 2:
        lines += ["## Cross-book agreement (holdings overlap, Jaccard)", ""]
        names = {k: set(v["stock"]) for k, v in books.items()}
        keys = list(names)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                j = len(names[a] & names[b]) / max(len(names[a] | names[b]), 1)
                lines.append(f"- {a} vs {b}: {j:.2f}")
        lines.append("")
    lines += ["## Anomalies", ""] + ([f"- **{a}**" for a in anomalies] or ["- none"])

    os.makedirs(DIFF_DIR, exist_ok=True)
    out = os.path.join(DIFF_DIR, f"DIFF_{asof}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[diff] {out} ({len(anomalies)} anomalies)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    main(ap.parse_args().asof)

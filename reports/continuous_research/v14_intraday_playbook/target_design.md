# v14 Target / Label Design (Task 3)

## If true intraday data existed (NOT the current state)

Forward 15/30/60-min returns from each checkpoint; entry-to-close;
first-passage labels (TP-before-stop vs stop-before-TP); MAE/MFE per
entry; short-side mirrors. These become definable ONLY after the collector
has accumulated history.

## Actual labels built this sprint (daily-bar WEAK PROXY)

Per (symbol, day t), conditioned on information available before t's open
(EOD score/action from t−1, gap known at t's open):

- `oc_ret` = close_t/open_t − 1 — the open-to-close outcome. The ONLY
  label the proxy rule search optimizes.
- `gap` = open_t/close_{t−1} − 1 — the conditioning variable (binned).
- `hi_ret`/`lo_ret` = high_t/open_t − 1, low_t/open_t − 1 — range envelope
  used ONLY for risk-envelope descriptive statistics.
- `stop_tp_ambiguous` — any stop/TP construction from daily bars is
  PATH-AMBIGUOUS (cannot know whether high or low came first) and is
  therefore reported as a bounded pair (best-case/worst-case), never a
  point estimate, and never fed to the rule search.

## Non-decision-grade declaration

These labels cannot validate time-of-day rules, fills, slippage, or
stop/TP ordering. Anything derived from them is labeled
DAILY_BAR_PROXY_ONLY and cannot be promoted beyond
paper-watchlist-candidate status regardless of measured edge. See
daily_bar_proxy_limitations.md.

# v16 Stage B2 — Do-Not-Chase Outcome Analysis (Task 26)

Date: 2026-08-18 · Data: do_not_chase_outcome_analysis.csv · ANALYSIS
ONLY — the chase threshold was frozen before this ran and is NOT retuned
by this result. Question: when the next OPEN already exceeds the frozen
do-not-chase level, does the validated 20-session strategy return
materially degrade? Forward return = open(T+1)→open(T+21) per name
(the validated next-open convention); max adverse excursion = min of
daily lows over the 20-session window vs entry — a LEVEL statistic,
no intraday ordering claimed. Strict historical separation (bands from
< T data; outcomes from ≥ T+1).

## Results

| Kind | Open bucket | n | mean fwd | median fwd | hit | ~Sharpe | med. max adverse |
|---|---|---|---|---|---|---|---|
| existing | open ≤ ceiling | 676 | +2.77% | +0.66% | 52.2% | 0.66 | −6.72% |
| existing | ceiling < open ≤ chase | 158 | +5.72% | +1.80% | 55.7% | 1.15 | −6.37% |
| existing | open > chase | 301 | +4.48% | +0.54% | 52.2% | 0.79 | −6.82% |
| fresh | open ≤ ceiling | 238 | +1.45% | 0.00% | 49.6% | 0.36 | −6.95% |
| fresh | ceiling < open ≤ chase | 39 | +1.33% | +0.16% | 51.3% | 0.36 | −6.12% |
| fresh | open > chase | 22 | +3.47% | +5.57% | 68.2% | 1.25 | −3.42% |

(Per-name observations are cross-sectionally correlated within a
rebalance; the Sharpe column is approximate context, not a tradable
statistic.)

## Honest verdict

**The do-not-chase interpretation does NOT have 20-session outcome
support.** Names opening beyond the chase threshold did not
underperform — if anything, above-threshold opens (a momentum
continuation signature) carried equal or better forward returns, and
the fresh-signal above-chase bucket (n=22, small) was the strongest.
This is fully consistent with the Task 1–2 timing audit: the signal is
slow, its alpha accrues over the 20-day hold, and entry-price variation
of ~1–2% is second-order for the HOLDING outcome.

What do-not-chase legitimately remains: an **execution-quality
discipline** — it bounds the price paid relative to the conditional
open distribution (you avoid paying a top-decile gap for a position you
could enter more cheaply on most comparable days, and the waiting-
tradeoff table quantifies the odds). It should NOT be read as "entering
above this level destroys the trade", and the nightly report must not
imply that. Per pre-registration nothing is retuned; whether to soften
the do-not-chase LANGUAGE (not the levels) is a user decision for the
Stage B2 review.

**Review outcome (2026-08-19, accepted):** the threshold is officially
reframed as "above preferred execution range" — an execution-quality
level flagging an unusually expensive entry vs the historical next-open
distribution. User-facing reports state explicitly that B2 analysis did
NOT show the validated 20-session signal fails above it, and that
waiting improves price quality at the cost of a higher
T+1_RANGE_DID_NOT_REACH_LEVEL probability. Numeric thresholds unchanged
(methodology amendment A1).

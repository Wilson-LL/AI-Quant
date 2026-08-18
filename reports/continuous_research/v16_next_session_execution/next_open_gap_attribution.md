# v16 Task 2 — Overnight-Gap Attribution

Date: 2026-08-18 · Data: next_open_gap_attribution.csv,
extreme_gaps_top20.csv. Long-book roles reconstructed per rebalance with
the engine's own selection logic (entrant = newly selected, incumbent =
retained, exit = dropped this rebalance). gap1 = open(T+1)/close(T) − 1;
intra1 = close(T+1)/open(T+1) − 1. Units: bps.

## Gap by book role

| Window | Role | n | gap mean | gap median | intra(T+1) mean | gap p10/p90 |
|---|---|---|---|---|---|---|
| CH | entrant | 181 | +67.8 | +26.3 | +35.0 | −90 / +250 |
| CH | incumbent | 706 | +55.1 | +33.8 | −1.6 | −137 / +278 |
| CH | exit | 155 | +82.7 | +32.2 | +19.7 | −77 / +208 |
| BR | entrant | 293 | +15.3 | +11.5 | −51.2 | −125 / +151 |
| BR | incumbent | 1119 | +20.3 | +27.5 | −44.9 | −137 / +173 |
| BR | exit | 264 | +36.4 | 0.0 | −33.9 | −111 / +176 |

By score quintile: top-quintile (Q5, the book) gaps +19 (BR) to +60 (CH)
bps mean; the tiny Q1/Q2 samples among book-role rows are exits with
large positive gaps (momentum names exiting after spikes). Sectors with
n≥200 (electronics, semis) match the overall pattern; no sector anomaly.

## Answers to the five pre-set questions

1. **Are BUY names already gapping up before entry?** Yes, slightly:
   median +26 bps (CH) / +12 bps (BR) on entrants. Real but small — one
   such gap per 20-session hold against a typical per-rebalance net
   return of +200–370 bps.
2. **Is the edge lost to overnight repricing?** No. The total drag is
   tens of bps per rebalance and is partly offset elsewhere (exits also
   gap up — selling at the open after a favorable gap recovers some);
   the headline A-vs-B comparison nets to ≈0.
3. **Does most alpha occur after T+1 open?** Yes. The same-close
   diagnostic F is not better than executing a session later; the 20-day
   hold, not the first overnight, carries the return.
4. **Does waiting until T+1 close avoid bad gaps?** No systematic
   benefit: B ≥ A in 3 of 4 window/mode combos, and where A is ahead (CH
   long-only) the margin (0.07) is noise-level.
5. **Are there avoidable opening-gap regimes?** None found at this
   horizon. The extreme gaps (top-20 list) are dominated by ±10%
   limit-move days — 14 of 20 are the 2025-04-09 market-wide limit-up
   rebound — and the pre-registered |gap|>5% filter shows those
   entries carried *positive* signal (dropping them lowers both
   conventions). "Do not chase" logic belongs at the per-name price-band
   level (Task 9), where this gap distribution becomes the empirical
   input — not at the portfolio-timing level.

## Feed-forward to the price advisor (Task 9, not yet approved)

The conditional gap distributions computed here (by role, quintile,
window) are exactly the raw material for provisional band construction:
entrant median gap +12–26 bps, p90 ≈ +150–280 bps (a natural
"do-not-chase" anchor), p10 ≈ −90 to −137 bps (a natural pullback-entry
anchor). Nothing in this file is an operational recommendation yet.

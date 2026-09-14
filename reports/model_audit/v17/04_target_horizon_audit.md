# 04 — Target / horizon audit (B3)

Data: `ic.csv`, `buckets.csv`, `stage1_holding_sector.csv` (from
`research/audit_v17_signal.py`, frozen panels CH 2023→ / BR 2021→).

## Rank IC of the production blend vs realized forward return, by horizon

| Panel | 1d | 5d | 10d | 20d (target) | 30d | 40d |
|---|---|---|---|---|---|---|
| CH blend | 0.017 | 0.035 | 0.048 | **0.067** | 0.077 | **0.087** |
| CH tf only | 0.015 | 0.030 | 0.041 | 0.057 | 0.068 | 0.082 |
| CH mom only | 0.019 | 0.036 | 0.047 | 0.063 | 0.071 | 0.079 |
| BR blend | 0.008 | 0.017 | 0.022 | **0.032** | 0.036 | **0.043** |

Bucket forward returns (CH, mean per date): top5 / 10-20 / bottom50 at
20d = 5.6% / 3.4% / 0.8%; at 40d = 11.4% / 7.4% / 1.5%. Top20−bottom20
spread: 20d 3.7% → 40d 7.9% (CH); 2.6% → 5.9% (BR). **Alpha keeps
accruing beyond the 20-session target and the ordering is monotone in
every bucket.**

## Consistency with the rest of the system

- **Live holding duration ≫ 20 sessions.** blend50_band10 turnover is
  0.09 per 20-session rebalance live (0.25 in the backtest): average
  position life ≈ 4–11 rebalances ≈ 80–220 sessions. The model is trained
  to rank 20-session returns while the book behaves like a multi-month
  holder — the band, not the target, sets the effective horizon.
- **Simply holding the 20d-trained signal longer does NOT raise Sharpe**
  (Stage-1 CPU test, long-only net60): CH 1.989 (hold 20) → 1.626 (40) →
  1.288 (60); BR 1.455 → 1.212 → 1.139. Annualized return is similar
  (0.71 / 0.57 / 0.66) but fewer, larger blocks raise per-block variance
  and per-rebalance turnover (0.25 → 0.38 → 0.49). The 20-session
  rebalance with hysteresis already harvests the longer-horizon
  information while diversifying entry timing.
- Prior research: 5d/10d targets and holds REJECTED (IC decays at short
  horizons; rank-10 target collapsed on BR); **40d/60d targets were never
  trained** (only proposed in the D1.1 assumptions audit).

## Verdict

The 20-session target is not obviously wrong — it is the shortest
horizon at which the signal is strong, and short rebalance + hysteresis
is a sound way to hold a slow signal. But the evidence (IC and spread
rising to 40d; effective holding of months) makes a **40-session-target
challenger (H1)** the single best-supported model hypothesis in this
audit, evaluated under BOTH 20- and 40-session rebalances so the
comparison is apples-to-apples. Expectations tempered: the gain must
show up as better rank quality at the same rebalance cadence, not merely
as "longer holds".

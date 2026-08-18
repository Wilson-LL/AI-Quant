# v16 Task 2 — Signal Decay Report

Date: 2026-08-18 · Data: signal_decay_report.csv (net Sharpe at
0/60/100/150 bps round-trip; common-coverage subset; identical books).
F is a diagnostic only — a 22:00 workflow can never execute at T's close.

## net60 Sharpe by execution point

| Execution point | CH L/S | CH LO | BR L/S | BR LO |
|---|---|---|---|---|
| F: T close (diagnostic, unavailable) | 2.115 | 1.719 | 1.438 | 1.474 |
| **B: T+1 open (user workflow)** | **2.158** | **1.918** | **1.463** | **1.505** |
| A/E: T+1 close (validated reference) | 2.147 | 1.989 | 1.443 | 1.455 |
| D: T+2 open | 2.048 | 1.870 | 1.348 | 1.392 |
| T+2 close | 1.921 | 1.921 | 1.447 | 1.452 |

Full cost ladder in the CSV; ordering is stable at net0/net100/net150.

## Readings

1. **There is no meaningful decay between T close and T+1 close.** The
   diagnostic same-close point F is not better than A or B anywhere (it
   is *worse* for CH long-only by 0.27) — the model's edge is not an
   overnight-repricing effect that execution delay destroys. The signal
   is slow; alpha accrues across the 20-day hold.
2. **T+1 open ≈ T+1 close.** Differences of ±0.02–0.07 Sharpe, sign
   inconsistent across windows/modes — indistinguishable from zero at
   these sample sizes (1 SE ≈ 0.5).
3. **One extra session costs ~0.1 (open) to ~0.23 (close, CH L/S).**
   D (T+2 open) loses 0.05–0.11 vs B. The T+2-close point reproduces the
   v9 X3 delay cost exactly (2.147→1.921 = 0.226 on CH L/S), confirming
   the two independent implementations agree.
4. **Practical instruction:** execute on T+1 (any point in the session is
   supportable); avoid slipping to T+2 — the cost of one full extra
   session is real though survivable, consistent with v9's
   "delay-robust" verdict.

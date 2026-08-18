# v16 Stage B — Price-Band OOS Validation Results

Date: 2026-08-18 · Runner: research/next_session_price_band_validation.py
· Data: price_band_validation.csv. Formulas/quantiles/constants were
pre-registered in price_band_methodology.md BEFORE this run and were not
modified afterward. Signals: frozen BR 7-seed walkforward panel (2021→,
1,703 role-observations over 68 rebalances, 0 skipped); calibration
strictly expanding (observations < T only). Daily-OHLC path ordering is
never interpreted (no stop-before-target claims).

## Headline coverage (share of next-session OPENS by band region)

| Band | n | below ideal | inside ideal | zone→ceiling | above ceiling | above chase |
|---|---|---|---|---|---|---|
| FRESH entry | 299 | 19.4% | **61.2%** | 2.7% | 10.0% | **6.7%** |
| EXISTING entry | 1,135 | 14.7% | **54.7%** | 0.6% | 6.8% | **23.2%** |
| SELL | 269 | below floor 28.3% | **inside 43.1%** | floor→zone 15.2% | above ideal 13.4% | below panic **0.4%** |

Reference pricing error (|reference − next open|): FRESH median 58 bps,
q75 126, q90 215; EXISTING median 73 bps; SELL median 56 bps.

Interpretation against the design:
- FRESH inside-zone coverage (61%) exceeds the nominal q25–q60 span
  (35%) because the K_WIDTH ATR floor widens thin empirical zones —
  conservative in the right direction. Above-chase 6.7% vs nominal 10%
  (q90): the tail threshold behaves as designed.
- EXISTING above-chase 23.2% vs nominal 25% (chase = q75 by design —
  deliberately stricter so standing targets are entered on pullbacks,
  not chased): matches the quantile design almost exactly.
- SELL panic-breach 0.4%: the K_PANIC ATR floor dominates the empirical
  q10, so panic levels are rarely touched by the open — again
  conservative.

## Stability

By year: coverage is stable 2021–2025 (inside-zone 47–74% fresh,
48–65% existing; reference error medians 34–98 bps). **2026 YTD is the
outlier** (fresh n=17, median error 186 bps; sell below-floor 44%) —
consistent with the 2026 high-vol regime and small samples; flagged for
re-check once 2026 completes. By vol regime: LOW-vol errors are
~half the HIGH-vol errors (35–43 vs 69–94 bps median), as expected;
inside-zone coverage is regime-stable (50–63%).

## Conditional behavior after the open (outcome description only)

After inside-zone opens vs above-chase opens, the same-day high/low
medians (vs the open) are reported in the CSV (`after_inside` /
`after_chase` rows) — used later by the Stage-C live advisor design,
not acted on here.

## Honest limitations

- Coverage percentages describe OPEN placement only; fills inside a zone
  are not guaranteed (limit orders at zone prices may not execute).
- Bands inherit the dividend-unadjusted cache (timing_audit F5): ex-date
  gaps sit in the empirical distributions; they widen tails slightly but
  are the same data the user will face at 22:00.
- 2026-and-later drift must be monitored; the calibrator is expanding,
  so new regimes enter the distributions automatically but with lag.
- No profitability claim: this validates that the price geometry matches
  its design historically, nothing more.

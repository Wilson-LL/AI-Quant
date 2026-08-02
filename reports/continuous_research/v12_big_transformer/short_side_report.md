# v12 short-side / long-short research

Panel-based (frozen 7-seed A8 panels), 20d rebalance, production long caps; short leg: bottom quintile after the conservative squeeze/borrow PROXY (low-ADV tercile, 20d return > +30%, price < 10 excluded), name cap 5%, sector cap 20%, costs 2x the long leg. REVIEW-ONLY — no orders.

## Window CH

### tf (net60 long / net120 short)
- L100 1.86 (DD -28.8%) · L100_S50 1.82 (DD -22.4%) · L100_S100 1.52 (DD -21.6%) · short leg alone -1.00 (DD -45.4%)
- short hit rate 48.7% · MAE(mean worst 20d move among shorts) +15.9% · avg shorts held 14 (excluded by proxy 7) · short turnover 0.49
- short-leg yearly Sharpe: 2023 -1.98, 2024 1.15, 2025 -1.28, 2026 -3.41

### d12 (net60 long / net120 short)
- L100 1.79 (DD -28.9%) · L100_S50 1.72 (DD -19.1%) · L100_S100 1.35 (DD -17.1%) · short leg alone -0.80 (DD -45.1%)
- short hit rate 49.5% · MAE(mean worst 20d move among shorts) +18.9% · avg shorts held 14 (excluded by proxy 7) · short turnover 0.40
- short-leg yearly Sharpe: 2023 -1.62, 2024 -0.04, 2025 -0.39, 2026 -2.27

### blend50 (net60 long / net120 short)
- L100 1.92 (DD -26.7%) · L100_S50 2.00 (DD -17.6%) · L100_S100 1.74 (DD -13.4%) · short leg alone -0.81 (DD -44.2%)
- short hit rate 50.9% · MAE(mean worst 20d move among shorts) +19.1% · avg shorts held 14 (excluded by proxy 7) · short turnover 0.38
- short-leg yearly Sharpe: 2023 -1.91, 2024 0.57, 2025 -0.7, 2026 -2.31

## Window BR

### tf (net60 long / net120 short)
- L100 1.26 (DD -30.2%) · L100_S50 1.15 (DD -24.6%) · L100_S100 0.84 (DD -23.2%) · short leg alone -0.84 (DD -60.7%)
- short hit rate 47.6% · MAE(mean worst 20d move among shorts) +16.6% · avg shorts held 14 (excluded by proxy 7) · short turnover 0.44
- short-leg yearly Sharpe: 2021 -1.75, 2022 -0.27, 2023 -1.3, 2024 0.68, 2025 -0.73, 2026 -3.08

### d12 (net60 long / net120 short)
- L100 1.36 (DD -36.0%) · L100_S50 1.12 (DD -38.6%) · L100_S100 0.64 (DD -43.9%) · short leg alone -1.02 (DD -72.4%)
- short hit rate 46.6% · MAE(mean worst 20d move among shorts) +18.1% · avg shorts held 14 (excluded by proxy 7) · short turnover 0.42
- short-leg yearly Sharpe: 2021 -1.19, 2022 -0.65, 2023 -1.34, 2024 -0.23, 2025 -0.99, 2026 -5.36

### blend50 (net60 long / net120 short)
- L100 1.36 (DD -31.1%) · L100_S50 1.32 (DD -29.9%) · L100_S100 1.05 (DD -32.9%) · short leg alone -0.64 (DD -53.8%)
- short hit rate 48.8% · MAE(mean worst 20d move among shorts) +15.7% · avg shorts held 14 (excluded by proxy 7) · short turnover 0.39
- short-leg yearly Sharpe: 2021 -1.64, 2022 -0.3, 2023 -0.76, 2024 0.72, 2025 -0.57, 2026 -2.93

## Gate evaluation (pre-registered)

- CH/tf: short adds value after 2x costs: **NO** (L100 1.86 -> S50 1.82 / S100 1.52); short leg standalone positive: NO (-1.00)
- CH/d12: short adds value after 2x costs: **NO** (L100 1.79 -> S50 1.72 / S100 1.35); short leg standalone positive: NO (-0.80)
- CH/blend50: short adds value after 2x costs: **NO** (L100 1.92 -> S50 2.00 / S100 1.74); short leg standalone positive: NO (-0.81)
- BR/tf: short adds value after 2x costs: **NO** (L100 1.26 -> S50 1.15 / S100 0.84); short leg standalone positive: NO (-0.84)
- BR/d12: short adds value after 2x costs: **NO** (L100 1.36 -> S50 1.12 / S100 0.64); short leg standalone positive: NO (-1.02)
- BR/blend50: short adds value after 2x costs: **NO** (L100 1.36 -> S50 1.32 / S100 1.05); short leg standalone positive: NO (-0.64)

## Caveats

- Panel universe is survivorship-biased; short-side results are OPTIMISTIC (delisted losers absent).
- Borrow availability/fees are a proxy; TWSE short-sale rules (uptick, quota) NOT modeled.
- This report never constitutes trading advice or orders.

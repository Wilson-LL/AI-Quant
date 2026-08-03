# v14 daily-bar proxy backtest (NON-DECISION-GRADE)

**DAILY_BAR_PROXY_ONLY — path-blind, assumed costs (30/60 bps round trip), survivorship-biased. Cannot validate time-of-day, stops, or fills. See daily_bar_proxy_limitations.md.**

Day-frame: 142,667 symbol-days, 2021-01-05 → 2026-07-23; signal = previous session blend z (frozen OOS panel); outcome = open→close net of assumed costs.

All-days long baseline: mean net -0.3938%, win rate 36.5% (day-trading the open-to-close with costs is NEGATIVE on average — the proxy edge, if any, must come from conditioning).

## Notable long cells (n>=100, |t|>=2)

| action | gap_bin | n | mean net | win | t | p5 | p95 |
|---|---|---|---|---|---|---|---|
| TOP_Q | <=-5 | 378 | +0.580% | 43% | +3.0 | -4.35% | +7.50% |
| BOTTOM_Q | <=-5 | 115 | +0.904% | 38% | +2.5 | -4.13% | +8.55% |
| BOTTOM_Q | 2..3 | 445 | -0.265% | 41% | -2.0 | -4.12% | +5.28% |
| BOTTOM_Q | 3..5 | 213 | -0.502% | 43% | -2.1 | -6.16% | +5.45% |
| WATCH_BAND | 3..5 | 196 | -0.635% | 39% | -2.7 | -5.81% | +5.13% |
| MID | 3..5 | 516 | -0.408% | 41% | -2.9 | -5.34% | +5.34% |
| WATCH_BAND | 2..3 | 392 | -0.459% | 39% | -3.2 | -5.02% | +5.13% |
| MID | 2..3 | 1201 | -0.273% | 41% | -3.5 | -4.19% | +4.94% |
| BOTTOM_Q | >=5 | 136 | -1.308% | 24% | -4.7 | -8.09% | +3.33% |
| WATCH_BAND | -2..-1 | 903 | -0.337% | 44% | -4.7 | -3.55% | +2.81% |
| BOTTOM_Q | -2..-1 | 1306 | -0.301% | 43% | -5.0 | -3.89% | +2.84% |
| TOP_Q | 2..3 | 1559 | -0.453% | 41% | -5.1 | -5.89% | +6.48% |
| TOP_Q | 3..5 | 1056 | -0.698% | 42% | -5.8 | -7.41% | +5.35% |
| TOP_Q | >=5 | 463 | -1.047% | 40% | -5.9 | -8.39% | +3.86% |
| TOP_Q | -2..-1 | 2154 | -0.339% | 45% | -6.0 | -4.64% | +3.75% |
| MID | >=5 | 238 | -1.455% | 28% | -6.6 | -8.09% | +3.56% |
| WATCH_BAND | 1..2 | 1564 | -0.429% | 37% | -7.3 | -3.79% | +3.57% |
| MID | -2..-1 | 3492 | -0.286% | 44% | -9.1 | -3.27% | +2.54% |
| BOTTOM_Q | 1..2 | 2471 | -0.426% | 35% | -10.1 | -3.29% | +3.25% |
| TOP_Q | 1..2 | 4402 | -0.589% | 38% | -13.4 | -5.15% | +4.49% |
| WATCH_BAND | -1..0 | 5652 | -0.362% | 40% | -16.9 | -2.92% | +1.96% |
| MID | 1..2 | 5727 | -0.505% | 35% | -18.6 | -3.56% | +2.98% |
| TOP_Q | -1..0 | 8888 | -0.448% | 40% | -19.1 | -3.94% | +2.92% |
| WATCH_BAND | 0..1 | 4721 | -0.495% | 34% | -19.8 | -3.14% | +2.19% |
| TOP_Q | 0..1 | 9280 | -0.503% | 37% | -21.1 | -4.05% | +3.12% |
| BOTTOM_Q | 0..1 | 10309 | -0.421% | 33% | -27.1 | -2.77% | +2.08% |
| BOTTOM_Q | -1..0 | 12593 | -0.405% | 34% | -31.9 | -2.61% | +1.72% |
| MID | 0..1 | 25626 | -0.397% | 33% | -45.0 | -2.56% | +1.83% |
| MID | -1..0 | 32622 | -0.323% | 36% | -45.5 | -2.27% | +1.57% |

## Short-diagnostic cells (BOTTOM_Q, 2x costs, n>=100, |t|>=2)

| gap_bin | n | mean net | win | t |
|---|---|---|---|---|
| -5..-3 | 175 | -0.785% | 38% | -3.4 |
| 2..3 | 445 | -0.635% | 43% | -4.9 |
| <=-5 | 115 | -1.804% | 23% | -4.9 |
| -3..-2 | 376 | -0.896% | 32% | -7.2 |
| -2..-1 | 1306 | -0.599% | 34% | -9.9 |
| 1..2 | 2471 | -0.474% | 43% | -11.2 |
| 0..1 | 10309 | -0.479% | 34% | -30.8 |
| -1..0 | 12593 | -0.495% | 30% | -39.1 |

## Risk envelopes by gap bin (descriptive only)

| gap_bin | n | mean hi from open | mean lo from open | worst oc |
|---|---|---|---|---|
| <=-5 | 928 | +3.10% | -1.35% | -8.2% |
| -5..-3 | 1302 | +2.69% | -1.96% | -10.3% |
| -3..-2 | 2223 | +2.21% | -1.79% | -8.0% |
| -2..-1 | 7855 | +1.60% | -1.47% | -8.9% |
| -1..0 | 59755 | +0.99% | -1.02% | -10.0% |
| 0..1 | 49936 | +1.08% | -1.16% | -10.9% |
| 1..2 | 14164 | +1.62% | -1.72% | -11.6% |
| 2..3 | 3597 | +2.26% | -2.21% | -12.4% |
| 3..5 | 1981 | +2.56% | -2.87% | -13.7% |
| >=5 | 926 | +1.54% | -3.27% | -16.4% |

Full cell table: backtest_results.csv. Rule selection with train/val/OOS splits: search_playbook_rules.py output.

## Rule search result (pre-registered splits; DAILY_BAR_PROXY_ONLY)

**No cell survived the train+validation gates** — the conditional day-trade proxy shows no robust exploitable open-to-close edge after costs at the pre-registered bars.

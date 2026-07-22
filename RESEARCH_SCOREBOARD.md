# AI-Quant Continuous Research Scoreboard

Branch: `research/continuous-alpha-loop-4060ti` · loop started 2026-07-22 20:39
Protocol: chronological walk-forward, exec-lag-1, matured labels, purge seq+horizon,
quintile books, hard 10% name cap, min_names=60. Panels cached under
`reports/transformer_gpu/panels/` (walk-forward OOS predictions — reusable without GPU).

## Baselines (fixed reference points)

| # | baseline | window | L/S net60 | net100 | maxDD | LO net60 | notes |
|---|---|---|--:|--:|--:|--:|---|
| B1 | D1.2 mom126/5 | 2023-01→2026-07 | 1.64 | 1.49 | −14.2% | 1.77 | BASE_d12_mom126_5 |
| B2 | mom20 | 2023-01→2026-07 | 0.40 | — | — | — | far behind |
| B3 | Champion Transformer (preset B, close-only, seq60, h64, 5-seed, equal-all) | 2023-01→2026-07 | **1.91** | 1.77 | −15.0% | **1.93** | G9_presetB_equal_all |
| B4 | 50/50 blend (score-level z) | 2023-01→2026-07 | 1.75 | 1.61 | −14.9% | 1.87 | CHAMP_blend50_tf_d12 |
| B5 | equal-weight universe | 2023-01→2026-07 | ~1.0 gross | — | −33% | — | benchmark |
| B1b | D1.2 | 2021-01→2026-07 (bear window) | 1.09 | — | — | — | 2022: −1.55 |
| B3b | Transformer | 2021-01→2026-07 | 1.00 | — | — | — | 2022: +0.58 |
| B4b | 50/50 blend | 2021-01→2026-07 | **1.37** | — | — | — | 2022: −0.29 |

Bear-window caveat: early refits train on 2018–20 only → absolute levels lower; use
for *relative* comparisons on the same panel only.

## Leaderboard (updated 2026-07-22 21:05)

1. **Best standalone model:** Champion Transformer preset B — L/S 1.91 net60 (2023–26)
2. **Best long-only model:** Champion Transformer LO quintile — 1.93 net60; blend50+band10 LO 1.83 with better bear profile
3. **Best long-short model / best overall book:** **blend50 + 10% no-trade band — L/S 1.95 net60, DD −12.3% (2023–26); 1.42, DD −26.4% (2021–26)** [D1, cycle 2]
4. **Best blend:** 50/50 score-level z-blend + band10 (score-level blending > return-level blending by ~0.2 Sharpe [C2])
5. **Best drawdown reducer:** blend50+band10 (−12.3% champion window); preset C (−10.5% at Sharpe 1.58) still holds for standalone
6. **Best 2022/bear performer:** Transformer standalone (+0.58 in 2022); blend50 −0.29 vs D1.2 −1.55
7. **Best low-turnover candidate:** blend50+band10 (turn 0.30) pending A1 rank-10 result
8. **Rejected overfit ideas:** recency weighting (all schemes), warm-start daily/weekly retrain, full-field features (naive), OHLC-range block, 1–1.5y rolling windows, **adaptive blend weights from trailing data (C1: all 5 variants ≤ static)**, **inverse-vol weighting on blend book (D1)**
9. **Promising but unproven:** 10d-rank target at champion strength (A1 running), excess-vs-sector target (LO 1.78 screen)

## Experiment log (continuous loop)

| ID | ts | hypothesis | verdict | key metric | vs B3 | vs B4 | commit |
|---|---|---|---|---|---|---|---|
| (prior sprint G1–G9 → see reports/transformer_gpu/CONSOLIDATED.md) | | | | | | | 52e8467 |
| C1 | 07-22 21:00 | adaptive TF/D1.2 blend weight from trailing matured data beats static 50/50 | **REJECT** | best adaptive 1.28 (AD5) vs static 1.37, 2021–26 L/S net60 | n/a | worse | — |
| C2 | 07-22 21:00 | static blend frontier, score vs return level | **KEEP (finding)** | score-blend50 1.37 / blend70 1.33; return-level 1.14–1.18 — score-level dominates; frontier flat 50–70 | blend complements | confirms B4 | — |
| D1 | 07-22 21:05 | band/invvol/mode construction transfer to blend book | **PROMOTE** | blend50+band10 L/S **1.95** net60 DD −12.3% (2023–26); 1.42 DD −26.4% (bear); consistent all 4 panel×mode combos; band 10–20 plateau; invvol hurts everywhere | beats 1.91 | beats 1.75 | pending |

<!-- new experiments appended below by the loop -->

## Hypothesis queue (ranked by expected value)

1. **C1 adaptive blend** — walk-forward adaptive TF/D1.2 blend weight (rolling sleeve
   Sharpe / IC / market-drawdown regime switch) beats static 50/50 on 2021–26 without
   losing on 2023–26. Cost: CPU-only on cached panels. [Track C]
2. **C2 static blend grid on bear window** — 30/70…70/30 score-level + return-level
   sleeves on the 2021 bear panel; establishes blend frontier. [Track C]
3. **D1 construction transfer** — no-trade band + inverse-vol + 10d hold on the blend
   book (band already helped standalone). [Track D]
4. **A1 rank-10 target champion rerun** — 10d-rank target at full 5-seed preset B
   (screen said 1.59 with 2 seeds; champion effect was +0.4). GPU ~2h. [Track A/B]
5. **C3 vol-regime switch** — market vol / drawdown detector gates D1.2 weight. [Track C]
6. **F1 paper-trading scaffold** — daily shadow book tracker for D1.2 / TF / blends. [Track F]
7. **A2 confidence filtering** — ensemble seed-disagreement as uncertainty veto. [Track A]
8. **B1 vol-adjusted target** — volatility-adjusted 20d forward return target. [Track B]

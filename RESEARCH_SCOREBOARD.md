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
| B3b | Transformer | 2021-01→2026-07 | 1.00 | — | — | — | 2022: +0.58 (shallow) |
| B4b | 50/50 blend | 2021-01→2026-07 | **1.37** | — | — | — | 2022: −0.29 |
| B3c | Transformer, deep 2015+ cache | 2021-01→2026-07 | 1.14 | 0.97 | −17.1% | — | 2022: −0.05; NEW reference |
| B4c | **blend50+band10, deep cache** | 2021-01→2026-07 | **1.47** | 1.32 | **−18.7%** | — | 2022: −0.15; 2023+: 2.02; NEW reference |
| B3d | Transformer, deep cache (REF23) | 2023-01→2026-07-22 | 1.91 | 1.74 | −17.4% | 1.91 | champion window, deep = shallow |
| B4d | **blend50+band10, deep (REF23)** | 2023-01→2026-07-22 | **2.06** | 1.92 | **−10.7%** | 1.95 | standing champion-window reference |

Bear-window caveat (superseded 07-23): rows B1b–B4b used the shallow 2018+ cache;
B3c/B4c on the 2015-backfilled cache are the standing references.

## Leaderboard (updated 2026-07-22 21:05)

1. **Best standalone model:** Champion Transformer preset B — L/S 1.91 net60 (2023–26)
2. **Best long-only model:** Champion Transformer LO quintile — 1.93 net60; blend50+band10 LO 1.83 with better bear profile
3. **Best long-short model / best overall book:** **blend50 + 10% no-trade band — L/S 1.95 net60, DD −12.3% (2023–26); 1.42, DD −26.4% (2021–26)** [D1, cycle 2]
4. **Best blend:** 50/50 score-level z-blend + band10 (score-level blending > return-level blending by ~0.2 Sharpe [C2])
5. **Best drawdown reducer:** blend50+band10 (−12.3% champion window); preset C (−10.5% at Sharpe 1.58) still holds for standalone
6. **Best 2022/bear performer:** Transformer standalone (+0.58 in 2022); blend50 −0.29 vs D1.2 −1.55
7. **Best low-turnover candidate:** blend50+band10 (turn 0.30) pending A1 rank-10 result
8. **Rejected overfit ideas:** recency weighting (all schemes), warm-start daily/weekly retrain, full-field features (naive), OHLC-range block, 1–1.5y rolling windows, **adaptive blend weights from trailing data (C1)**, **inverse-vol weighting on blend book (D1)**, **rank-10 target line — standalone AND all blends (A1/A1B: bull-window LO 1.97 collapsed to 0.96 on bear window)**, **multi-horizon r20+r10 ensembles (B2: fail bear gate)**, **short holds h5/h10 for the 20d-target blend (D3)**, **exposure-scaling gates EX1/EX2 (C3)**
9. **Promising but unproven:** excess-vs-sector target (LO 1.78 screen), EX3 own-equity DD gate (monitor), 7.5% name cap (Sharpe-neutral concentration cut), close+D1.2 features at 5 seeds (E1 running)

## Experiment log (continuous loop)

| ID | ts | hypothesis | verdict | key metric | vs B3 | vs B4 | commit |
|---|---|---|---|---|---|---|---|
| (prior sprint G1–G9 → see reports/transformer_gpu/CONSOLIDATED.md) | | | | | | | 52e8467 |
| C1 | 07-22 21:00 | adaptive TF/D1.2 blend weight from trailing matured data beats static 50/50 | **REJECT** | best adaptive 1.28 (AD5) vs static 1.37, 2021–26 L/S net60 | n/a | worse | — |
| C2 | 07-22 21:00 | static blend frontier, score vs return level | **KEEP (finding)** | score-blend50 1.37 / blend70 1.33; return-level 1.14–1.18 — score-level dominates; frontier flat 50–70 | blend complements | confirms B4 | — |
| D1 | 07-22 21:05 | band/invvol/mode construction transfer to blend book | **PROMOTE** | blend50+band10 L/S **1.95** net60 DD −12.3% (2023–26); 1.42 DD −26.4% (bear); consistent all 4 panel×mode combos; band 10–20 plateau; invvol hurts everywhere | beats 1.91 | beats 1.75 | 121bd54 |
| F1 | 07-22 21:25 | paper-trading scaffold (backfill/snapshot/evaluate) | **KEEP (tool)** | 76 shadow books 2025-01→2026-07; 17 matured rebalances/strategy; ledger Sharpes consistent with backtest | n/a | n/a | 739732d |
| C3 | 07-22 21:35 | exposure scaling reduces DD ≥20% both panels | **REJECT** (EX3 L/S → monitor) | EX3 bear 1.46/−19.5% vs base 1.42/−26.4%; champion DD unchanged, −0.05 Sharpe; EX1/EX2 fail | — | filter only | 739732d |
| D1b | 07-22 21:40 | blend60/70+band10 beats blend50+band10 | **REJECT** | 1.83/1.87 champ, 1.37/1.31 bear — below blend50+band10 both panels (better 2022/DD noted) | — | worse | — |
| F2 | 07-22 21:50 | blended decision-book generator | **KEEP (tool)** | 2026-07-07 book: 22 names, maxW 10.0%, 21 HOLD/1 BUY/1 SELL/9 WATCH | n/a | n/a | 739732d |
| D3 | 07-22 21:55 | shorter holds / tighter cap on blend book | **REJECT h5/h10; ADOPT-OPTIONAL cap7.5** | h10 1.66–1.80 champ, 1.18 bear (< h20 1.95/1.42); IC decays at short horizons; cap 7.5%: 1.92/1.45 champ/bear ≈ free concentration cut | — | h20 stays | — |
| R1 | 07-22 22:05 | blend50+band10 universe bootstrap (200×, drop 20%) | **PASS** | champ p5 1.52 / p50 1.75, bear p5 1.15 / p50 1.38, 100% positive; drop-top-3: 1.77 / 1.16 | robust | robust | — |
| A1 | 07-22 22:10 | rank-10 target at full champion strength beats champion | **REJECT standalone; blend → A1B** | standalone L/S 1.51 / LO 1.78, DD −28.5% (vs champ 1.91/1.93); but blend h10+band10: L/S 1.93 / **LO 1.97, net100 1.91, turn 0.16** (2023–26 only) | loses | LO blend intriguing | — |
| B2 | 07-22 22:55 | multi-horizon (r20+r10) ensemble beats blend50+band10 | **REJECT** (bear gate) | champion window MH1 LO 1.93 (+0.10) passed screen; bear window MH1 1.11/1.22, MH2 1.39/1.43 < blend 1.42/1.48 | ties LO champ window only | fails bear | — |
| A1B | 07-22 23:40 | rank-10 blend LO 1.97 generalizes to bear window | **REJECT — rank-10 line closed** | bear: standalone 1.08/1.21; blend h10+band10 L/S 0.92 / LO 0.96 vs blend50+band10 1.42/1.48; 2022 −0.83 | worse | much worse | — |
| V1 | 07-22 22:50 | Section-6 validation profile for blend50+band10 | **PASS** | no losing year either panel (worst 2022 −0.04, else ≥1.37); IC>0 every yr except 2022; rank-ac 0.993; avg max-sector share of longs 37% (p95 57%, book capped soft-20%) | — | — | 34f9bf6 |
| A2a | 07-23 00:55 | champion reproducibility (score_std edit runtime test) | **PASS — exact** | rerun L/S 1.91 / LO 1.93 / IC 0.0716 = original to 2 decimals; production Edit 1 validated | = | = | — |
| E1 | 07-23 00:55 | close+D1.2 features win at 5 seeds (screen said LO 1.82) | **REJECT** | L/S 1.37 / LO 1.50, IC 0.057 ≪ close-only 1.91/1.93/0.072; close-only reconfirmed; 2-seed screens again shown unreliable | loses | loses | — |
| A2b | 07-23 01:05 | seed-disagreement confidence filtering improves books | **REJECT** | CF1/CF2 drop-uncertain: Sharpe halves (blend 1.95→~1.0), IC 0.072→0.032–0.053, turnover ~doubles — high-std names ARE the high-signal names; CF3 shrinkage ≤ base everywhere | worse | worse | — |
| BF | 07-23 03:25 | 2015-backfill of cache | **DONE (tool)** | 106 stocks +77k rows, 0 failures; cache 2015-01→2026-07 | n/a | n/a | bb28a5b |
| BD | 07-23 05:50 | deep training improves bear window; blend survives | **PASS — new bear reference** | deep tf 1.14 (was 1.00); **blend50+band10 1.47 / DD −18.7% (was 1.42/−26.4%), 2023+ 2.02**; 2022 tf −0.05 (deep loses some crash-adaptivity vs +0.58 shallow) but blend −0.15 ≈ flat | blend > tf | improved | — |
| B3 | 07-23 05:50 | vol-adjusted 20d target beats rank-20 | **REJECT** | 1.44 / 1.72 net60, IC 0.057 — loses ~0.5 despite deeper training data | loses | loses | — |
| B4 | 07-23 05:55 | avoid-bottom target as book or veto | **REJECT — instructive** | val IC **+0.146** but OOS IC **−0.069**; L/S −1.95; veto10/20 on champion LO: 1.45/0.93 vs 1.86 base. Textbook val/OOS inversion: learnable low-risk proxy anti-predicts returns | much worse | much worse | — |
| A3 | 07-23 05:55 | champion regularization is at an optimum | **CONFIRMED** | dropout0.3 1.86 (val IC .048), wd5e-4 1.90 (val IC .051) ≈ champion; no val-IC case for change; deep cache leaves champion-window results intact (~1.90) | ≈ | — | — |
| F3 | 07-23 06:15 | live daily cadence on fresh 2026-07-22 data | **PASS (tool)** | refresh→retrain(267s)→inference→books→decision book ≈ 11 min end-to-end | n/a | n/a | 677e864 |
| C4 | 07-23 06:30 | sector-neutral scoring as risk control | **REJECT** | SN1/SN2 cost 0.4–1.1 Sharpe on both panels for ~10pp share cut — sector tilt is load-bearing | worse | worse | 677e864 |
| WD1 | 07-23 07:40 | deep+3y-window ensemble recovers 2022 adaptivity | **REJECT — line closed** | roll3y member collapses on deep cache (L/S −0.03, IC .003); ens-blend 1.30 < 1.47 ref (footnote: its DD −13.3% / 2022 +0.05 were better — tradeoff persists, not worth −0.17 Sharpe) | worse | worse | — |
| REF | 07-23 09:50 | deep-cache champion-window references | **SET** | standalone 1.91/1.91 (= shallow — champion window depth-insensitive); blend50+band10 **2.06 / −10.7% L/S, 1.95 LO** (B4d) | = | new ref | 0f292c8 |
| B5 | 07-23 10:10 | preset-C architecture-diversity blend beats refs | **REJECT — line closed** | C-deep standalone 1.82/1.88 (2023), 1.13/1.28 (2021, DD −14.0% best-in-class); 3-way blend 2.05/−9.75% ties champion window but loses bear (1.37 vs 1.47, 2022 −0.51); BC50 worse everywhere | ties one window | fails dual gate | — |

<!-- new experiments appended below by the loop -->

## Hypothesis queue v2 (regenerated 2026-07-23 01:10 after queue v1 exhausted)

Champion weak spots driving the new queue: (1) 2022 only ~flat; (2) LO book carries
−27–31% DD market beta; (3) bear-panel evidence used shallow 2018–20 early training;
(4) single-target (rank-20) signal, IC-IR 0.33–0.40.

1. **BEAR-DEEP** — rerun bear-window champion + blend with the 2015-backfilled cache
   (early refits get full-depth training). Validates the promoted candidate's weakest
   evidence. GPU ~40 min after backfill lands. [Track C/validation]
2. **B3 vol-adjusted target** — fwd20 / realized vol20 target at champion strength;
   different alpha axis, may lift 2022. GPU ~20 min batched. [Track B]
3. **B4 avoid-bottom-quintile target** — asymmetric bad-tail avoidance; candidate
   drawdown reducer for the LO book. Needs new target column (additive) + GPU. [Track B]
4. **A3 regularization one-knob check** — dropout 0.3 & wd 5e-4 at champion config
   (5 seeds): is the champion under/over-regularized? GPU ~30 min batched. [Track A]
5. **F3 paper-trading cadence** — daily snapshot + weekly evaluate; realized-vs-backtest
   IC tracking for blend50+band10. CPU, ongoing. [Track F]
6. **E2 true full-fields revisit** — blocked until data_cache_full accumulates
   (~6 months); keep collecting via --full-fields. [Track E, deferred]

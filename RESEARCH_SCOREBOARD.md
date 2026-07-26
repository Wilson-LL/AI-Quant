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
| B4d | blend50+band10, deep (REF23) | 2023-01→2026-07-22 | 2.06 | 1.92 | −10.7% | 1.95 | superseded by B4e |
| B4e | **blend50+band10, 7-seed (A8)** | 2023-01→2026-07-23 | **2.147** (seeds 0–6) / 1.843 (seeds 10–16) | 2.00/1.70 | −10.6/−10.2% | 1.99/1.81 | reference restated as SEED-ROBUST RANGE **≈1.85–2.15** (bootstrap median 1.92 = planning number) |
| B4f | **blend50+band10, 7-seed bear** | 2021-01→2026-07-23 | **1.443** / 1.301 (seeds 10–16) | 1.30/1.15 | −18.0/−26.8% | 1.46/1.32 | range **≈1.30–1.45** (bootstrap median 1.37); 2022 −0.15/−0.81 seed-dependent |
| B4g | crash-first window (W22) | 2022-01→2026-07-23 | 1.321 | 1.17 | −14.2% | 1.24 | third-window descriptive ref; 2022 −0.72 (refit-grid alignment matters) |

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
| ES | 07-23 11:45 | excess-vs-sector target at 5 seeds (screen LO 1.78) | **REJECT — promising list emptied** | standalone 1.66/1.76 (2023), 1.17/1.29 (2021), 2022 −0.86; blend 1.54/1.17 ≪ refs 2.06/1.47 | loses | loses | — |
| RL1 | 07-23 14:30 | pairwise ranking loss beats MSE-on-rank | **REJECT — line closed by val-IC gate** | val IC 0.036 < champion 0.050 (selection metric decides); OOS confirms: 1.80/0.99 standalone, blends 1.80/1.24 ≪ 2.06/1.47; smoke promise didn't scale | loses | loses | — |
| E3 | 07-23 17:35 | market-regime features fix 2022 (queue v4) | **REJECT — regime-feature line closed; A4/B6 cancelled** | standalone 1.26/0.96, IC diluted (0.060/0.017); blend 1.89/1.27 < refs; **2022 −0.30 worse than −0.15** — opposite of the target effect. Note: val IC marginally favored E3 (0.052 vs 0.050) — marginal val-IC edges are not decision-grade | loses | loses | — |
| D5 | 07-24 | variance-parity leg weighting (queue v5 CPU) | **REJECT** | 1.87/1.28 L/S vs refs 2.06/1.47 — risk-based weighting loses again (invvol precedent) | — | worse | — |
| D6 | 07-24 | concentrated top-10/15 LO books | **REJECT** | top10 LO 1.94/−35% (champ), 1.32/−43% (bear); Sharpe ≈ quintile at far worse DD | — | worse risk | — |
| D7 | 07-24 | conservative spec band15+cap7.5 | **DOCUMENTED OPTION (near-gate)** | 1.98/−10.6% champ (−0.08 vs ref, misses 0.05 tol), 1.46/−16.1% bear (+2.6pp DD, turn 0.28/0.22) — the conservative production variant | — | ≈, safer | — |
| V5s | 07-24 | queue v5 screens (3-seed, scheduler run 1) | **screens read** | family val ICs: wd1e3 .0485 > L1 .0477 > do1 .0464 > wd1e5 .0459 > L3 .0427 > listwise .0345; OOS(not selective): L3 1.89, wd1e5 1.81, L1 1.75. **REJECT at screen: E4_range (LS 0.24), E5_liq (1.01, IC .020), RL2 listwise (val-IC gate — ranking-loss family closed for good)**. B7_spread excluded from family comparison (own line, 1.46 unremarkable) | — | — | 94c52c6 |
| B6s | 07-24 | dd-adjusted target screen | **INVERSION #2 (B4 pattern)** | val IC +0.158 / OOS LS **−1.98** — risk-proxy targets inflate val IC and invert OOS; P5 inversion check queued with pre-declared expected REJECT; promotion rule fixed (val IC within-family only) | — | — | — |
| A8 | 07-24 | 7-seed ensemble (queue v5) | **ADOPTED — new production spec** | champion window **2.147 / −10.6%** (+0.088 vs ref), bear window **1.443 / −18.0%, 2022 −0.15** (−0.027 vs ref, within 0.05 tol; DD 0.7pp better). Cost: +2 seeds ≈ +40% retrain (~6 min/day, trivial). New refs B4e/B4f set | +0.09 | supersedes | — |
| P5s | 07-24 | 5-seed promotions (run 2) | **all REJECT** | A5_L1 blend 1.86/−9.9%; **seq90 blend 1.815 despite family-best val IC 0.054 — val-IC edges ≠ OOS wins (again)**; B6 5-seed inversion CONFIRMED (val 0.159/OOS −1.94, blend −0.46) — B4/B6 pattern now twice 5-seed-confirmed | lose | lose | — |
| S120 | 07-24 18:00 | seq120 at 5 seeds | **REJECT — sequence axis closed** | blend 1.659/1.747 vs 2.147/1.99; decision-grade ordering 40 < 60 > 90 > 120 | loses | loses | 89c586e |
| D7b | 07-24 18:10 | conservative spec on 7-seed panels | **ADOPT as recommended deployment spec** | band15+cap7.5 on 7-seed signal: **2.15 champ (= ref), 1.44 bear (= ref), bear DD −15.2% (+2.8pp), 2022 −0.08, turn 0.29/0.23, maxW 7.5%** (bull DD −11.2%, 0.6pp worse — noted). Shadow books stay band10 until quarter boundary | = , safer | = , safer | 4341476 |
| A9 | 07-24 20:00 | 9-seed: does the seed curve saturate? | **SATURATION — keep 7** | 2023 blend 2.173 (+0.026, within tol); bear 1.437 (tie) with DD −20.4% (2.4pp worse) and 2022 −0.32 (worse). Curve 5→2.06, 7→2.147, 9→2.173/flat-bear; 9 costs +28% retrain for no dual-window gain. (Redundant auto-spawned BEAR_A9_seeds9_2023 failed on a None-wd bug — fixed; its window was already covered by A9_seeds9_2021, marked redundant, not hidden) | — | keep 7 | — |
| E4p | 07-24 20:10 | production Edit 4: daily retrain 5→7 seeds | **DONE — tested** | 7 checkpoints in 304 s, val ICs +0.131…+0.156, inference consumed unchanged; 07-24 books regenerated on adopted spec (22 held, maxW 10.0%) | — | — | 83afcb8 |
| R2 | 07-24 21:00 | 7-seed blend universe bootstrap (200×, drop 20%) | **PASS** | champ p5 1.61 / p50 1.92 / p95 2.17; bear p5 1.14 / p50 1.37; 100% positive both. Note: base ≈ p95 on both — point estimates are name-composition-optimistic; bootstrap medians (1.92/1.37) are the conservative planning numbers | robust | robust | 8821975 |
| V7 | 07-25 00:30 | validation battery (queue v7) | **SEED-SENSITIVITY FLAG; refit-robust; 3rd window set** | SR1 disjoint-seed 2023: **1.843 vs 2.147 — outside ±0.15 → the 7-seed "+0.09 win" was partly seed-set luck; refs restated as ranges (B4e/B4f)**. SR2 bear 1.301 (marginal pass; 2022 seed-dependent −0.81 vs −0.15). RF1 refit-63: 2.085 = protocol-robust. W22 crash-first: 1.321/−14.2% (B4g). Bonus: accidental exact rerun (spawn bug #3, fixed) reproduced BEAR_A8 bit-identically — walkforward determinism confirmed | honest ranges | honest ranges | — |
| V8 | 07-26 05:25 | 14-seed (0–6 + 10–16) converges both windows to seed-set midpoints (queue v8; crash-interrupted 07-25 01:42 by CUDA/driver wedge + shutdown, resumed after recovery audit — see RECOVERY_AFTER_SHUTDOWN.md) | **CONVERGENCE CONFIRMED; KEEP 7 (user decision 07-26) — 14-seed line closed** | champ blend 2.125 vs midpoint 1.995 (+0.13, clears); bear blend 1.383 vs midpoint 1.372 (+0.011 = parity, within seed noise). But bear DD −22.6% (7-seed: −18.0%), 2022 −0.43 (7-seed: −0.15), 2× retrain cost — the gate's own cost-discipline rationale argues keep-7. 2022 stabilization hypothesis held (−0.43 ∈ [−0.81, −0.15]). Auto-spawned BEAR_ENS14_2023 = exact duplicate of ENS14_2021 (bear spawn inherits oos_start-invariant config): bit-identical books — determinism reconfirmed, ~2.4h GPU wasted (spawn-dedup gap noted). Pre-crash ENS14_2021 partials discarded as INVALID; full rerun used | at range top | parity, worse DD | — |
| V9 | 07-26 | deployment validation queue (13 CPU experiments, signal frozen to 7-seed panels; user-approved tracks: construction grid, execution realism, paper readiness, bear overlays, robustness audit) | **12 PASS/REJECT-as-registered, 1 FLAG — deployment case execution-validated; D7b stays; EX3 disarmed** | C1: D7b non-dominated (band15_cap10 2.21/1.45/1.52 misses 1pp-DD bar; band15 drives bear DD). X1: net150 1.81/1.13, break-even 632/463 bps. X2: 10M TWD capacity trivial (p95 participation 0.4% ADV). X3: same-bar audit clean, T+2 cost 0.226 ≤0.30. X4: settled-cash cost ≈0. P1/P2 OPERATIONAL (ledger 17/20 matured obs — evidence gate arms in ~3 rebalances; daily diff live). B2 REJECT → **EX3 disarmed** (d7b +2.2pp <3pp); B3 REJECT (sleeve +1.6pp); B4 REJECT as pre-declared (bear improved — blend 2022 −0.15→+0.04 — but CH flat; one-crash-sample, recorded not acted). R3 PASS (d7b p5 1.70/1.11/1.02, 100% positive). R4: 30% sector cap costs 0.001 — premium not load-bearing. R5 **FLAG**: BR/d7b drop-top5 retention 0.667 <0.70; top name 1519 = 9–12% PnL — watch in ledger | validated | validated; no crash overlay | — |
| V10s | 07-26 | Short v10 (user-approved Option B): cross-sectional attention, multi-task heads, augmentation-vs-seed-variance, 5d-reversal leg; flag-gated training edits, champion path verified bit-identical flags-OFF | **3 lines closed; 1 promotion — MT heads produce the best bear/2022 profile on record; adoption HELD pending disjoint-seed validation** | A1 CS-attention REJECT decisive (val IC 0.037 vs 0.048 — line closed at this scale). C1 reversal REJECT decisive (rev5 standalone **negative** −0.51/−1.05 — no short-horizon reversal on TWSE; line closed). D1 augmentation REJECT (val-IC cross-set spread already 0.00025 at baseline; noise/drop make it worse — **v7 seed variance lives in OOS book space, not val IC**). B1_MT3 (aux heads rank_5/10, 1.0/0.3/0.3) val IC 0.05026 cleared bar by 0.0001 (thin — E3 caveat) → 7-seed dual-window: **2.011 / 1.515, bear DD −13.0% (def −18.0%), 2022 +0.23 (def −0.15)** — bear point above range top, champ −0.136 below default (inside ±0.15 seed-noise band). One 7-seed draw; seed-set sensitivity unknown → **recommend v10b MT validation battery (disjoint seeds + refit-63) before any adoption** | −0.14 champ | +0.07 bear, +5pp DD, 2022 positive | — |

<!-- new experiments appended below by the loop -->

## Loop state (2026-07-23 14:40)

All experimental levers in Tracks A–E have been tested under pre-registered
dual-window gates; **13 challengers rejected, zero survived against
blend50+band10**. The loop is in OPERATIONAL CADENCE: daily refresh → (monthly
retrain) → inference → blended decision book → paper ledger; monitors armed
(EX3 bear gate, 7.5% cap option, preset C DD-alternative). Experimental work
reopens on: new market regimes in the paper ledger, true full-field data
(~2027-01), or new external information.

## Hypothesis queue v5 (GPU research mode, 2026-07-24; supersedes v3/v4 idle state)

User-directed GPU-saturated phase. 13 GPU configs registered in
`reports/continuous_research/gpu_scheduler/queue_v5.json` (+auto-promotions,
+bear spawns), run by `research/gpu_research_scheduler.py`:
layers 1/3 · dropout 0.1 · wd 1e-5/1e-3 · spread target · dd-adjusted target ·
listwise loss · seq 90/120 · close+range · close+liquidity · 7-seed ensemble.
CPU items: ERC weighting, top-N LO, conservative spec (band15+cap7.5).
Closed lines from v1–v4 are NOT re-run. Gates unchanged (dual-window vs refs
2.06/−10.7% and 1.47/−18.7%, val-IC selection, hard 10% cap).

## Hypothesis queue v3 (regenerated 2026-07-23 11:50; v2 fully executed)

Closed lines (do not reopen without new information): adaptive weights, return-level
blending, invvol, exposure gates EX1/EX2, sector-neutral scoring, short holds,
rank-10 / voladj / avoid-bottom / excess-sector targets, close_d12 & full-field
features, confidence filtering, window-diversity ensembles, architecture-diversity
blends, recency weighting, warm-start retrains.

1. **RL1 pairwise ranking loss** — champion trains MSE on rank targets; pairwise
   logistic loss on same-date pairs is the last allowed, untested Track-A lever.
   Needs fit_one loss option (Edit 2) + dual-window run. GPU ~1.5h. [Track A]
2. **F3 daily cadence (operational)** — refresh → retrain(monthly)/inference →
   books → ledger each TWSE close; realized-vs-backtest IC after 20d maturity.
   [Track F, time-gated]
3. **E2 true full-fields revisit** — when data_cache_full has ~6 months of real
   turnover/transaction. [Track E, calendar-gated ~2027-01]
4. **Monitors:** EX3 own-equity DD gate (attach if bear regime emerges); 7.5% cap
   (concentration-sensitive deployments); preset C standalone (DD-priority use).

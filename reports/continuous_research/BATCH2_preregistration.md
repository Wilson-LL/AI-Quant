# Pre-registration — Cycles 11–14 (BATCH2, post-backfill)

Registered: 2026-07-23 01:30 (before running; launches after the 2015-backfill
completes so every config trains on the same deep cache)

Cache note: all five configs (and any future run) use the 2015-01+ backfilled
cache. Champion-window results are therefore comparable *within* this batch and
to future runs, but early-2023 refits now train on 3 extra years vs the original
champion panel — the BEARDEEP config doubles as the new reference for that.

## Cycle 11 — BEAR-DEEP (validation of the promoted candidate)

Hypothesis: with full-depth training for early refits, the bear-window (2021–26)
results improve; the blend50+band10 conclusion (best book) survives.
Config: champion rank-20, OOS 2021-01→. Panel LOOP_BEARDEEP_rank20_2021.
Report: standalone + blend50+band10 (CPU post-pass) vs shallow-cache references
(tf 1.00/D1.2 1.09/blend 1.37/blend+band 1.42 L/S net60; 2022: +0.58/−1.55/−0.29).
Gate: this REPLACES the bear reference panel going forward regardless of direction
(more training data = more realistic protocol); a materially WORSE result
(> 0.2 Sharpe drop) flags training-depth sensitivity on the scoreboard.

## Cycle 12 — B3 vol-adjusted target

Hypothesis: tgt_voladj_20 (fwd20 / realized vol20, cross-sectionally ranked)
shifts the book toward risk-efficient names; better Calmar/2022 than rank-20.
Gate: beats champion books (1.91/1.93) or blend books (1.95/1.83) on net60 or
maxDD by pre-set margins (≥0.05 Sharpe or ≥3pp DD with Sharpe within 0.1) →
bear-window validation next batch; else reject/research-only.

## Cycle 13 — B4 avoid-bottom target

Hypothesis: tgt_avoid_bot_20 (discriminates only the bottom 30%) produces a
useful *short leg* or LO veto, improving DD.
Evaluation: standalone books + "champion longs minus avoid-bot bottom-decile
veto" (CPU post-pass). Gate: LO DD improves ≥3pp at ≤0.1 Sharpe cost → filter
candidate (category C); else reject.

## Cycle 14 — A3 regularization one-knob checks

Hypothesis: champion is near a regularization optimum; dropout 0.3 or wd 5e-4
will NOT beat it (confirmation test; if one wins by ≥0.1 with val-IC support,
the champion config updates — val IC decides, per protocol).

Runtime: one dataset build + 5 configs × ~15 min ≈ 1.7h GPU.

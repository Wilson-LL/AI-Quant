# Pre-registration — Cycle 2 (A1 rank-10 champion rerun · D1 blend construction)

Registered: 2026-07-22 20:52 (before running)

## A1 — 10d-rank target at full champion strength [Track B/A]

Prior evidence: G4 screen (2 seeds, preset A/B-screen) gave tgt_rank_10 L/S 1.59 /
LO 1.83 with turnover 0.19 vs tgt_rank_20 1.53. The champion upgrade (preset B,
5 seeds) lifted rank_20 from 1.53 → 1.91. Hypothesis: the same upgrade lifts
rank_10 to ≥ champion level, at ~half the rebalance turnover... or reveals the
screen ordering was 2-seed noise.

Config (fixed in advance): close_only, seq 60, preset B, seeds [0,1,2,3,4],
equal-all history, refit 126d, target tgt_rank_10, holding 10, OOS 2023-01→2026-07.
Panel saved as `LOOP_A1_rank10_presetB.csv.gz`.

Success criteria:
- Beats B3 champion (L/S 1.91 / LO 1.93 net60) → replacement-track, needs bear-window rerun.
- Within 0.15 of champion but with materially lower turnover / better net150 → low-turnover candidate.
- Else → reject "rank_10 at scale" and keep rank_20.
Also report: blend with D1.2 (z-avg) on the resulting panel.

## D1 — construction transfer to the blend book [Track D]

Hypothesis: the constructions that helped the standalone champion (5% no-trade
band; inverse-vol weighting) also help the 50/50 score-blend book, improving
net Sharpe and/or DD without reducing net60 by more than 0.05.

Grid (pre-declared, all reported): panel = champion `G9_presetB_equal_all` blended
50/50 with D1.2 z-mom; constructions: band ∈ {0, 0.05, 0.10} × weighting ∈
{equal, invvol} × mode ∈ {LS, LO}, hold 20. Robustness gate: any cell claimed as
an improvement must ALSO improve (or tie within 0.05) on the bear panel
(`BEAR_presetB_2021`) blend — no single-panel cherry-picks.

Both CPU-only on cached panels.

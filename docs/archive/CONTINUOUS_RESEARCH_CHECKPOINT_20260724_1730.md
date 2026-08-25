# Continuous Research Checkpoint — 2026-07-24 17:30 (GPU research mode)

Covers 2026-07-23 12:30 → 2026-07-24 17:30 · commits 830bee1 → this.

## 1. Experiments run

Queue v4 tail: RL1 pairwise loss REJECT (val-IC gate); E3 market-regime
features REJECT (2022 worsened). Daily ops: 2026-07-23 cycle + laggard-row
regeneration.

Queue v5 (GPU mode, user-directed): scheduler built and ran 18 GPU configs
across 3 runs (1 smoke, 12 screens, 5 five-to-seven-seed confirmations/bear)
plus 3 CPU construction cells (D5/D6/D7). One config pending (P5_A7_seq120).

## 2. GPU utilization summary

**The GPU was meaningfully used:** ~9h of training across runs 1–3; utilization
83–95% during training (shared with FFXIV/desktop ~3 GB; peak training VRAM
~2.9 GB; no OOM events; 2 config failures from a preset/seq bug — fixed,
rerun, both completed). Dataset builds (~35 min each ×4) remain the main
non-GPU wall-clock; grouped by the scheduler to build each once.

## 3. Best new candidate — PRODUCTION SPEC UPGRADED

**7-seed ensemble blend50+band10 (A8):**
- 2023–26: L/S **2.147 net60 / DD −10.6%** (prior ref 2.06/−10.7%), net100 2.00, LO 1.99
- 2021–26: L/S **1.443 / DD −18.0%**, 2022 −0.15 (prior 1.47/−18.7% — tie within tolerance, DD better)
- Cost: +2 seeds ≈ +40% retrain time (~6 min/day all-in — trivial).
New standing references B4e/B4f. First reference improvement since the D1 band
discovery.

## 4. Rejected this window

RL1 pairwise + RL2 listwise (ranking-loss family closed), E3 regime features,
E4 close+range (LS 0.24), E5 close+liquidity (IC 0.020), A5 1-layer (blend
1.86), seq90 at 5 seeds (blend 1.82 despite family-best val IC — val-IC edges
≠ OOS wins), B6 dd-adjusted target (val/OOS inversion #2, confirmed at 5
seeds: val +0.159 / OOS −1.94), B7 spread target (unremarkable screen), D5
variance-parity, D6 top-N concentration. D7 band15+cap7.5 documented as the
conservative variant (1.98/−10.6 · 1.46/−16.1).

## 5. Comparison vs champion

Every rejected line lost to blend50+band10 on at least one window; the only
survivor (7-seed) is an upgrade OF the champion spec, not a rival model.
D1.2 1.64 · champion tf 1.91 · mom20 0.40 · equal-weight ~1.0 all unchanged.

## 6. Next queue (v6 draft)

1. P5_A7_seq120 (running — last v5 item).
2. Ensemble-size curve continuation: 9-seed check (does the seed curve
   saturate at 7?) — cheap, directly follows the A8 win.
3. Seed-count × bear-window interaction for the conservative D7 spec.
4. Daily ops each TWSE close (scheduler-integrated).
5. Time-gated: full-fields (~2027-01), paper-ledger maturation.

## 7. Was the GPU meaningfully used?

Yes — see §2; and the mode produced a production-spec upgrade (§3) plus ten
disciplined rejections that tightened the champion's evidence.

## Methodology hardening (from run 1)

Cross-target val ICs are NOT comparable (risk-proxy targets inflate val IC and
invert OOS — B4/B6 pattern, twice 5-seed-confirmed). Scheduler promotion is
now within-family only. seq_len now propagates into derived presets.

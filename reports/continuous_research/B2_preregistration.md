# Pre-registration — Cycle 9 (B2 multi-horizon transformer ensemble)

Registered: 2026-07-22 22:50 (before running)

Hypothesis: averaging the rank-20 and rank-10 transformer scores (two models
trained on different horizons, same architecture/data) diversifies model noise
and beats either alone; the 3-way mix with D1.2 beats blend50+band10.

Pre-declared variants (champion window 2023–26; h20 book, band10, equal weight):
- **MH1** tf_multi = ½·z(score_rank20) + ½·z(score_rank10), standalone
- **MH2** 50/50 tf_multi × D1.2 (3-way: ¼ r20 + ¼ r10 + ½ mom)
- **MH3** ⅓ r20 + ⅓ r10 + ⅓ mom
All L/S + LO, vs baselines: champion standalone 1.91/1.93, blend50+band10 1.95/1.83.

Gates: any variant beating blend50+band10 by ≥ 0.05 (either mode) on the champion
window → MUST be re-validated on the bear window (BEAR_presetB_2021 × LOOP_A1B
panel, available when A1B lands) before promotion. Champion-window-only wins are
"monitor". Losses → reject.

CPU-only; panels cached.

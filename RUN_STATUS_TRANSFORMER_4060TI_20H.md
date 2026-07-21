# Run Status — Transformer RTX 4060 Ti 20h Sprint

Branch: `research/transformer-4060ti-daily-retrain-20h` (created from
`research/d1-1-momentum-prototype-clean` HEAD d41fdd4 — see note below).

> **Branch provenance note**: the prompt said this branch was created from
> `research/transformer-gpu-confirmation-20h`, but that branch (and the other
> transformer sprint branches/artifacts it references) does not exist in this
> repository. Only `main`, `list`, and `research/d1-1-momentum-prototype-clean`
> exist. The sprint branch was created from the D1.1 clean milestone HEAD, and
> the transformer EOD pipeline is being built fresh this sprint, reusing
> model.py / train.py / research/ as the prompt requires.

---

## 2026-07-22 00:42 — Sprint start (elapsed 0:00)

- Current experiment: G0 CUDA readiness.
- Device: RTX 4060 Ti confirmed, CUDA True, torch 2.13 nightly cu132, venv Python 3.12.3.
- AMP benchmark: 10.6× speedup vs fp32; epoch @200k samples ≈ 17 s; inference 119 stocks ≈ 6 ms; peak VRAM 317 MB.
- Files changed: RTX4060TI_ENVIRONMENT_CHECK.md (new), this file (new).
- Commands run: nvidia-smi, torch CUDA check, GPU smoke benchmark.
- Blockers: none. Prior-sprint artifacts referenced by the prompt don't exist; documented and proceeding.
- Next: write RTX4060TI_SPRINT_PLAN.md, then build dataset_transformer_eod.py.
- Commits: none yet.

## 2026-07-22 01:15 — Pipeline built, G1 running (elapsed 0:33)

- Current experiment: **G1** (close-only seq40/seq60 vs baselines) running on GPU in background.
- Built and committed (da1a54c): dataset_transformer_eod.py, train_transformer_eod.py,
  inference_transformer_eod.py, research/transformer_portfolio.py,
  research/transformer_experiments.py, research/refresh_data.py + docs.
- G0 done: smoke train on real data (201k samples, 3.7 s/epoch, VRAM 1.3 GB), checkpoint saved.
- Fixed: duplicate dates in 18 cached CSVs (dedupe at load); leg-aware cost model;
  min-names guard for boundary rebalances.
- Device: cuda (RTX 4060 Ti), AMP on.
- Latest metrics: OOS 2023→2026 baselines — D1.2-style mom126/5 L/S net@60 Sharpe **1.64**
  (LO 1.77); mom20 L/S 0.40. High bar for the transformer.
- G1 first refits: val rank IC +0.075 / −0.046 (noisy), ~50 s per 3-seed refit.
- Blockers: none.
- Next: G1 results → launch G2 (recency sweep); write hybrid/portfolio analysis (G6/G7).
- Estimated remaining: experiments ~8–10 h GPU + analysis + reports.

## 2026-07-22 01:40 — G1 done, G2 running (elapsed 0:58)

- **G1 result (OOS 2023-01→2026-07, 42 rebalances, quintile books):**
  - D1.2-style mom126/5 baseline: L/S net@60 Sharpe **1.64** (LO 1.77), IC +0.074.
  - mom20 baseline: L/S 0.40 (LO 1.22), IC +0.061.
  - Transformer close-only seq40 HL126 (3 seeds): **IC −0.021, L/S −1.21, LO 1.00** — fails vs baseline.
  - Transformer seq60: IC +0.008, L/S −0.21, LO 1.09 — also fails.
  - Val ICs unstable, strongly negative in 2026 refits (−0.09/−0.12): the recency-weighted
    model latches onto a recent pattern that anti-correlates OOS.
- ~40 s per 3-seed refit; VRAM ~1.3 GB; GPU experiments much faster than planned.
- G2 (9 recency configs × 2 seeds) running; ~30 min expected.
- Commits: da1a54c (pipeline). Files added since: research/transformer_hybrid.py,
  research/transformer_diagnostics.py.
- Next: G2 → G4 (targets) → G5 (features) → G3 (cadence) → diagnostics/hybrids → champion.
- Blockers: none.

## 2026-07-22 02:10 — G2 + G6 + diagnostics done; G4/G5/G3 running (elapsed 1:28)

- **G2 (recency, 9 configs): equal-weight ALL history wins** — IC +0.071, L/S net@60 1.54,
  LO 1.75, nearly matching D1.2 (1.64/1.77). Monotone: hl_63 → −2.50 L/S; hl_126 → −1.35;
  hl_252 → +0.19; hl_378 → +0.71; roll_3y 1.44. Aggressive recency weighting is HARMFUL
  on this data; the prior-sprint claim "HL126 best" is refuted under full GPU training.
  Selection metric (val IC) agrees with OOS ranking (equal_all val IC +0.062 top) — honest pick.
- **G6 (hybrid): 50/50 z-rank blend transformer+D1.2 beats D1.2** — L/S 1.75 vs 1.64,
  LO 1.87 vs 1.77, net@100 1.60 vs 1.49, similar DD, lower turnover. All blend weights
  (30/50/70) non-harmful. Transformer earns its keep as ensemble component.
- **Diagnostics**: tf score is 71% rank-correlated with mom126/5, 48% with vol20;
  residual IC after stripping mom+vol = +0.016 (small real independent component);
  IC positive every year 2023–2026; score rank autocorr 0.998.
- G4 (targets) → G5 (features) → G3 (cadence) queued sequentially on GPU (~2.5 h).
- Blockers: none. Next: results consolidation, presets A/B/C timing, champion run, G8 workflow.

## 2026-07-22 02:50 — G4 done, G5 running (elapsed 2:08)

- **G4 (targets, equal-all recency, close-only):** 20d rank best L/S (1.54, IC 0.071);
  10d rank close behind (L/S 1.58, LO 1.82, IC 0.054, 84 rebalances); 5d rank weaker
  (L/S 1.05); excess-vs-universe 1.19; excess-vs-sector 1.46 (LO 1.78); original
  barrier target as diagnostic: IC 0.063 but worse books (DD −36%). Keep 20d rank
  primary; 10d rank is a viable secondary.
- G4 wall time: 23 min for 6 targets × 7 refits × 2 seeds.
- G5 (7 feature sets) running; G3 (cadence) queued next.
- Commits: b96fd1c (G1/G2/G6/G7 results + tooling). New: research/transformer_presets.py.

## 2026-07-22 03:35 — G5 done, G3 in progress (elapsed 2:53)

- **G5 (feature sets, all at equal-all recency): CLOSE-ONLY WINS.**
  close_only L/S 1.54 / LO 1.75; close+D1.2-rank L/S 1.53 / **LO 1.84 (best LO)**;
  ohlc_range −0.65; volume_block 1.23; sector_rel 0.97; curated_full 0.13;
  full_d12 1.16. Full TWSE fields hurt even curated + regularized + GPU-trained —
  prior sprint's "close-only beats full-field" conclusion CONFIRMED honestly.
- **G3 partial (OOS 2024-07→2026-07):** frozen L/S 1.30 (IC 0.076); quarterly 1.16
  (IC 0.060) — retraining more often is so far NOT better; monthly running,
  weekly-warm and daily-warm queued (~1.5 h remaining).
- Device: cuda; ~1.3–2 GB VRAM; all runs AMP.
- Blockers: none. Next after G3: presets A/B/C, consolidation, champion daily
  workflow (G8), final report.

## 2026-07-22 05:30 — G3/G9/G8 done, consolidation + robustness done (elapsed 4:48)

- **G3 (cadence, OOS 2024-07→2026-07): frozen 1.30 ≥ monthly 1.25 ≥ quarterly 1.19
  ≫ daily-warm −1.33 / weekly-warm −2.60 (L/S net@60).** Daily/weekly warm-start
  fine-tuning actively degrades the signal. Daily retrain does NOT beat frozen/monthly.
  Daily-scratch sample over 2026 running as supplement.
- **G9 (presets): B (h64/seq60/5 seeds) champion — L/S 1.91 / LO 1.93 net@60, net@100
  1.77, IC 0.072, DD −15.0%. BEATS D1.2 (1.64/1.77) on all cost levels.** Val-IC
  selection agrees (B val IC 0.074 highest) — honest pick. A: 1.48; C: 1.58 (DD −10.5%).
- **G8 (daily workflow) end-to-end PASS**: refresh (dry-run) → daily-retrain 134.6 s
  (5 seeds, val IC +0.209) → inference 1.8 s → decision book 22 names, max weight
  exactly 10.0%, all outputs written. ~5–10 min/day vs 12 h budget.
- **Cap bug found+fixed**: clip-and-redistribute ping-pong let single-name-sector
  names stabilize at 11.1% > 10% cap; replaced with monotone water-fill; all
  consolidated books now show maxW = 10.0% exactly.
- **Consolidation (uniform min_names=60)**: CONSOLIDATED.md written; champion B
  standalone 1.91/1.93; blend50 on 2-seed panel 1.75; D1.2 1.64/1.77.
- **Robustness (champion)**: universe bootstrap (200 draws, drop 20%):
  p5/p50/p95 = 1.57/1.83/2.05, 100% positive; drop-top-3-names → 1.60.
- Files: docs/transformer_daily/RTX4060TI_DAILY_BUDGET.md, research/transformer_presets.py,
  research/transformer_robustness.py, research/transformer_daily_scratch.py, G8 outputs.
- Next: daily-scratch supplement finishes (~1 h), final report, commits.

## 2026-07-22 08:00 — Sprint complete (elapsed 7:18)

- **Daily-scratch supplement**: 121 fresh daily fits over 2026 → OOS IC +0.190 vs
  frozen champion +0.209 on the same window; 32.6 s/seed/day. Daily full retrain
  buys nothing — cadence conclusion unchanged (monthly recommended).
- **Bear-regime stress (OOS 2021→2026 incl. 2022 crash)**: transformer L/S 1.00,
  D1.2 1.09, **50/50 blend 1.37**. Crash year 2022: **tf +0.58 vs D1.2 −1.55**.
  The two signals are regime-complementary → blend recommendation upgraded from
  "safe" to "positively justified".
- Final report written: TRANSFORMER_4060TI_DAILY_RETRAIN_20H_REPORT.md.
- All queue items complete: G0 ✓ G1 ✓ G2 ✓ G3(+scratch) ✓ G4 ✓ G5 ✓ G6 ✓ G7 ✓
  G8 ✓ presets ✓ robustness ✓ diagnostics ✓ bear stress ✓ consolidation ✓
  budget doc ✓. Queue finished early because measured GPU throughput was ~10×
  the planning estimate.
- No blockers. Nothing pushed; no protected branch touched; production files
  untouched.

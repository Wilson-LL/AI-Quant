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

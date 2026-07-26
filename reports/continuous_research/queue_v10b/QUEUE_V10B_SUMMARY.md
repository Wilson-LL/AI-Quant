# Queue v10b Summary — MT Validation Battery + Full-v10 Continuation

Completed 2026-07-27 (run started 07-26). All items done, 0 failed, ~4.3 h GPU
(battery ~2.1 h, phase 2 ~0.5 h, B2 promotion ~1.7 h). Champion path
anchor-verified bit-identical before each phase. Production unchanged; cache
read-only throughout; CUDA stable.

## Phase 1 — MT validation battery: NOT VALIDATED (2/3 gates)

| Run | Result | Gate | Pass |
|---|---|---|---|
| SR_MT1 disjoint seeds 10–16, champ | blend 1.947, DD −10.4% | in [1.85, 2.15] | ✓ |
| SR_MT2 disjoint seeds, bear | **blend 1.259, DD −23.5%, 2022 −0.38** | ≥1.44 ∧ 2022 ≥ 0 ∧ DD ≥ −16% | **✗ (all three)** |
| RF_MT refit-63, champ | blend 1.881 | in range | ✓ |

The original MT draw's bear case (1.515 / DD −13.0% / 2022 +0.23) was
**seed-set luck** — on disjoint seeds MT lands *below* the plain spec's own
disjoint bear reference (1.259 vs SR2's 1.301) and the 2022 sign flips.
Same failure mode v7 caught on the 7-seed "+0.09" claim. **MT line closed;
champion (plain-trunk 7-seed blend) stands.** Side-note: MT's val ICs in
these runs were strong (0.056–0.060) while books collapsed — val-IC
selection remains blind to book-space fragility.

## Phase 2 — remaining full-v10 screens

| ID | Result | Verdict |
|---|---|---|
| B2_Q3 quantile head | val IC **0.06299** (bar 0.05013), standalone LS 1.888, blend 1.964 | **won the promotion slot** (wide margin, healthy books — no B4/B6 inversion signature) |
| A3_TCN3 anchor | val IC 0.04995 (≈ parity!), but books collapsed (LS 0.303) | expected-lose confirmed, with a lesson: the attention trunk's advantage is in book space, not val IC |
| C2_RES3 residual target | val IC 0.032, OOS IC −0.002, LS −0.81 | REJECT (plain failure, not an inversion); residual-target line closed |
| D2_DIST distillation | student = **96.1%** of 7-teacher ensemble val IC (bar 95%), 40 s train | **PASS** — daily retrain could drop 7×→1×; production adoption = user decision, and should first get an OOS walkforward student-vs-ensemble check (this was a single-refit val test) |

## B2 promotion (7 seeds, dual-window): REJECT

| Window | Standalone LS | Blend LS | Gate |
|---|---|---|---|
| 2023→ | **2.036** (champ tf: 1.672) | 1.910, DD −13.3% | in range (low end; default 2.147) |
| 2021→ | 1.24 (champ tf: 1.19) | **1.274**, DD −20.2%, 2022 −0.10 | **BELOW 1.30 floor → REJECT** |

Honest residue worth keeping: pinball-loss training **genuinely improves the
standalone transformer** (+0.36 champ / +0.05 bear vs the champion
standalone, with a real val-IC gain) — but the improvement does not survive
the 50/50 momentum blend, and the blend is the deployment object. Recorded
as a lead (`quantile` flag stays available, default OFF), not a challenger.
Any revisit (e.g. blend-weight re-tuning around the quantile signal) is
adaptive-weight-adjacent and would need its own pre-registration.

## Net effect of v10 + v10b (the whole GPU research arc)

Nine new-axis experiments + validation battery, zero adoptions — and that is
the system working: five lines closed decisively (cross-sectional attention,
5d reversal, input augmentation, residual target, MT heads), one anchor
confirmed (TCN), one seed-luck promotion caught before adoption (MT), one
rejected at the dual-window gate with a documented lead (quantile), and one
practical PASS awaiting a user decision (distillation). The champion spec
survives everything thrown at it; its evidence base is now materially deeper.

Open user decisions: (1) D2 distillation follow-up (OOS walkforward check →
possible 7×→1× daily retrain), (2) whether the quantile-standalone lead ever
gets a pre-registered revisit. Flags in the training module: cs_attn, mt_aux,
aug_noise, aug_datedrop, quantile, tcn — all preset-key gated, default OFF,
champion path anchor-verified.

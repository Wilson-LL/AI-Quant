# 24 — H-RESIDUAL-SIGNAL and H-FACTOR-PREMIUM: results

**Preregistration:** `23_h_residual_factor_prereg.md` and `h_residual_factor_spec.json`, committed and pushed as `5012d5b` before any result was computed. **Code:** `research/h_residual_factor.py` (zero GPU, no retraining). **Machine-readable:** `h_residual_signal_result.json`, `h_factor_premium_result.json`. Per-date tables are in `h_rf/`, untracked.
**Evidence label:** BURNED MECHANISM DIAGNOSTIC. It covers 131 decision dates in 2026-H1 and 27 refit blocks. The primary SE is Newey–West, lag 4, on the block series. The prospective slice is descriptive only, about one label window.

## Headline

- **H-RESIDUAL-SIGNAL: RESIDUAL_SIGNAL_WEAK** (primary arm B). The network carries a positive, fairly consistent residual beyond any linear mix of its own inputs, but the 95% interval still includes zero. The daily-refit arm (production cadence) and the 126-session arm meet the PRESENT rule; they are secondary and do not set the label.
- **H-FACTOR-PREMIUM: FACTOR_PREMIUM_REGIME_DEPENDENT** for `mom_126_5`, 12-1 momentum and `vol_60`. All three paid strongly from January to April, turned negative in June and July, and paid again in the prospective slice.
- **Long-training attribution: FACTOR_DEEXPOSURE_EXPLAINS_MOST.** The early-minus-late deterioration in P4-B0-LONG is almost entirely the loss of momentum exposure, priced by each block's momentum payoff.
- **Decision tree: Case C.** Residual WEAK: no architecture scaling, no GPU. Build independent sample and prospective shadow history first.

## H-RESIDUAL-SIGNAL (arm B, 27 blocks)

| quantity | mean | HAC SE | 95% CI | blocks > 0 | leave-one-block-out |
|---|---|---|---|---|---|
| Transformer IC | 0.115 | 0.072 | −0.033, +0.262 | 74% | 0.102–0.134 |
| `mom_126_5` IC (= D1.2 momentum) | 0.076 | 0.081 | −0.090, +0.242 | 63% | |
| 12-1 momentum IC | **0.134** | 0.085 | −0.041, +0.309 | 74% | |
| ridge on the 10 inputs, IC | 0.083 | 0.072 | −0.064, +0.231 | 70% | |
| **residual IC beyond the 10 inputs (primary)** | **+0.056** | 0.031 | **−0.008, +0.120** | 67% | 0.049–0.064 |
| residual IC beyond `mom_126_5` | +0.082 | 0.045 | −0.010, +0.174 | 74% | |
| residual IC beyond the ridge score | +0.089 | 0.060 | −0.035, +0.213 | 74% | |
| residual IC beyond 12-1 momentum | +0.028 | 0.045 | −0.064, +0.120 | 59% | |
| residual IC beyond 10 inputs + 12-1 momentum | +0.038 | 0.030 | −0.024, +0.100 | 70% | |
| paired Transformer minus ridge IC | +0.031 | 0.033 | −0.037, +0.099 | 63% | |
| **placebo floor** (a pure function of two inputs) | +0.010 | 0.016 | −0.023, +0.044 | 59% | |

- **Exposure:**
  - Spearman correlation of the Transformer with `mom_126_5` is 0.75, with 12-1 momentum 0.69, and with the ridge score 0.67.
  - Rank variance explained is 56% by `mom_126_5` alone and 67% by the 10 inputs.
- **How much of the OOS value is factor exposure:** 69% of the Transformer's rank-Pearson IC is carried by the linear span of its inputs, and 51% by `mom_126_5` alone. The remaining 31% is the residual.
- **Spread:** the top-minus-bottom quintile 20-session spread is 6.8% for the Transformer and 3.3% for its input-residual. The residual spread's interval is +0.5% to +6.1%.
- **Placebo:** the floor of +0.010 trips the preregistered METHOD_SENSITIVE flag by 0.0001. It does not change the label, which failed PRESENT on its lower bound, and the residual exceeds the floor by 0.046.

**Secondary arms:**

| arm | refit cadence | residual IC beyond inputs | 95% CI | rule outcome |
|---|---|---|---|---|
| B | 5-session (primary) | +0.056 | −0.008, +0.120 | WEAK |
| C | daily (production cadence) | +0.069 | +0.008, +0.130 | PRESENT |
| A | 126-session | +0.073 | +0.022, +0.124 | PRESENT |

All three arms share the same 3-seed recipe, dates and labels, so they are not independent evidence.

**Prospective slice** (production 7-seed scores, 11 matured dates 2026-07-24 → 08-11, one window, descriptive):
- The Transformer IC is 0.248.
- The input-residual IC is +0.009: nothing beyond the inputs.
- The ridge scored 0.319, and `vol_60` alone 0.357.

**Two findings the preregistration did not anticipate:**
1. **Plain 12-1 momentum, which is not a network input, beat the Transformer in this window:** 0.134 vs 0.115. Adding it to the 10 inputs cuts the residual from +0.056 to +0.038, so about a third is absorbed (like-for-like; the +0.028 figure is the residual beyond 12-1 momentum alone). Part of what the network adds beyond its last-step inputs therefore looks like longer-horizon momentum rebuilt from its 60-day input sequence.
2. The residual's leave-one-block-out range (0.049–0.064) shows it is not driven by one block. The uncertainty comes from the overlap of 20-session labels. The naive clustered SE is 0.022, against a Newey–West SE of 0.031.

### Feature contribution (diagnostic; correlation is not causation)

| feature | Spearman with Transformer | univariate R² | drop-one ΔR² | standardised coefficient |
|---|---|---|---|---|
| `mom_126_5` | +0.74 | 0.56 | **0.109** | **+0.58** |
| `mom_60` | +0.61 | 0.38 | 0.009 | +0.13 |
| `vol_60` | +0.59 | 0.36 | 0.007 | +0.13 |
| `dist_lo_60` | +0.58 | 0.35 | 0.004 | +0.02 |
| `vol_20` | +0.56 | 0.32 | 0.004 | +0.01 |
| `mom_20` | +0.26 | 0.10 | 0.008 | −0.02 |
| `px_over_ma20` | +0.16 | 0.08 | 0.011 | −0.09 |
| `mom_5`, `log_ret_1`, `dist_hi_60` | ≤ +0.04 | ≤ 0.08 | ≤ 0.008 | small |

The network is mostly a **momentum ranker with a volatility tilt**. `mom_126_5` is the only input with a large unique contribution. The other momentum, volatility and range inputs are collinear with it and add little on their own.

## H-FACTOR-PREMIUM (131 dates; Newey–West lag 20; 5 degrees of freedom)

| factor | mean IC | 95% CI | days > 0 | blocks > 0 | Jan–Apr 17 | Apr 20–Jul | prospective (15 dates) | label |
|---|---|---|---|---|---|---|---|---|
| `mom_126_5` | +0.074 | −0.125, +0.274 | 65% | 67% | **+0.222** | **−0.071** | +0.217 | REGIME_DEPENDENT |
| 12-1 momentum | +0.136 | −0.075, +0.347 | 73% | 74% | **+0.304** | **−0.029** | +0.230 | REGIME_DEPENDENT |
| `vol_60` | +0.077 | −0.126, +0.281 | 61% | 70% | **+0.168** | **−0.013** | +0.260 | REGIME_DEPENDENT |

- **Monthly IC for `mom_126_5`:**

  | Jan | Feb | Mar | Apr | May | Jun | Jul |
  |---|---|---|---|---|---|---|
  | +0.29 | +0.13 | +0.21 | +0.23 | +0.02 | −0.25 | −0.11 |

- **Context, 2016–2025, descriptive:** yearly `mom_126_5` IC ranged from −0.056 to +0.075. So the 2026 January–April momentum payoff was about three to ten times a typical year. The burned window was an unusually strong momentum regime followed by a sharp reversal.
- **Other inputs:** every momentum, volatility and range input shows the same pattern, positive in the first half and negative in the second.

## Long-training attribution

**LONG refits** (epoch 3 vs mean of epochs 50/75/100; additive rank-Pearson split):

| refit | total early−late gap | carried by the input span | carried by `mom_126_5` alone | residual part | momentum exposure, early → late |
|---|---|---|---|---|---|
| 2026-01-05 | 0.403 | 0.293 (73%) | 0.261 | 0.110 | 0.74 → −0.04 |
| 2026-04-20 | 0.097 | 0.179 (184%) | 0.138 | −0.082 | 0.69 → 0.12 |
| 2026-07-23 | −0.005 | 0.029 | −0.005 | −0.034 | 0.81 → 0.29 |
| **pooled** | | **101%** | **80%** | | |

**P4-B0 cross-refit** (9 refits): the epoch 3 → 15 OOS drop tracks momentum's payoff in each block.
- Correlation with payoff: r = 0.82, one-sided permutation p = 0.003.
- Correlation with exposure × payoff: r = 0.81, p = 0.003.
- Examples: 2026-03-26 had payoff +0.45 and epoch 3 won by 0.19. 2026-07-01 had payoff −0.32 and epoch 15 won by 0.22.

**Classification: FACTOR_DEEXPOSURE_EXPLAINS_MOST.** Long training strips the momentum exposure. Whether that hurts depends entirely on whether momentum pays in the block. In momentum-reversal blocks, the de-exposed late model did better. H-EPOCH-LONG's rejection is therefore mostly a statement about momentum exposure in a momentum-rich window, not about late-epoch learning as such.

## Decision tree outcome

**Case C: RESIDUAL_SIGNAL_WEAK.**
- No architecture scaling and no GPU experiment.
- The early LR-decay experiment (H-LR-SCHEDULE) is **not yet justified** by the frozen primary label. The two secondary arms reaching PRESENT is the strongest argument to revisit it once more independent sample exists. The Council takes this up in Session 03.
- Next: grow the independent sample. That means the prospective shadow history, a pre-2026 historical residual panel if one can be built without retraining, and a longer-history factor-premium test.

## Limits

- The window is burned and momentum-extreme.
- The benchmarks are deliberately simple and linear.
- The residual removes only date-t linear use of the last-step inputs. It keeps the network's nonlinear and temporal use of those same series, which partly rebuilds longer-horizon momentum.
- The three arms share seeds, recipe and dates.
- The prospective slice is one window, and its production run history has gaps (see the operational note in the final report).
- Survivor universe; unadjusted prices.

## Council Session 03 revisions (frozen labels unchanged)

See `COUNCIL_03_residual_signal_synthesis.md`. The frozen labels stand, and the operative reading is stricter:

- **H-RESIDUAL-SIGNAL = RESIDUAL_SIGNAL_WEAK (frozen, arm B, 2026-H1 burned). No residual signal is demonstrated beyond the network's own inputs.**
  - The label depends on the uncertainty convention: PRESENT only at HAC lag 0–1, and WEAK at lag 2 and above or with df = 5.
  - The PRESENT readings in arms C and A are not independent (block correlation 0.80–0.92 with arm B). Arm C drops to WEAK under the factor stage's df = 5.
  - The residual co-moves with momentum payoff (r ≈ 0.62–0.74).
  - On the post-peek 2021–25 historical panel, the residual is +0.014 (95% CI −0.012 to +0.040).
  - Adding raw-scale inputs and 12-1 momentum brings the residual to about zero in both samples (post-hoc).
- **Correction to the Case-B trigger:** the secondary arms reaching PRESENT is not a reason to revisit H-LR-SCHEDULE. Every member's gate fails.
- **Data-integrity defect found.** The EOD cache has whole-month holes: 70 of 110 stocks since 2016, 24 in 2025–26. The model builds features on each stock's own rows, so it splices across them. This happens in training and in live inference, with no look-ahead. In 2026-H1, spliced-window rows carry 42–60% of the residual.
- **The calibration test used Gaussian inputs, and the placebo floor moves with the regime.** The single +0.010 floor understates the method's noise.

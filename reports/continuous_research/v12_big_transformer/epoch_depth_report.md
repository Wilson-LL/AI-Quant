# v12 epoch/training-depth report

| item | blend LS net60 | dd | val_ic | mean epochs run | mean val-IC-peak epoch |
|---|---|---|---|---|---|
| P2_recipe_Wall_baseline | 1.969 | -0.1054 | 0.04813 | 6.0 | n/a (25-ep run) |
| P2E_Wall_baseline_E50p5 | 1.838 | -0.1225 | 0.05025 | 8.4 | 2.4 |
| P2E_Wall_baseline_E50p10m10 | 1.798 | -0.1516 | 0.05164 | 15.2 | 4.2 |
| P2E_Wall_baseline_E100p10 | 1.798 | -0.1516 | 0.05164 | 15.2 | 4.2 |
| P2E_Wall_baseline_E100p20m25 | 1.326 | -0.1948 | 0.0628 | 35.0 | 12.8 |

Baseline row = P2_recipe_Wall_baseline (max_epochs 25 / patience 3, no per-epoch curves). A cell is only interesting if it beats the baseline at BOOK level — longer training is never evidence by itself (plan §5).

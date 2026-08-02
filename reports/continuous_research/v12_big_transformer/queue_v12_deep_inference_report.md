# v12 deep inference report — 2026-07-30

Production 7-seed ensemble, read-only. {'seed': 7, 'mc_dropout': 30, 'feat_noise': 60} passes, runtime 16.6s (budget 5.0h), peak VRAM 0.66 GB.

## Rank stability (today)

- truncated-history rank correlation: t-1 0.9986, t-3 0.9563, t-5 0.9373
- deterministic top-quintile members' mean inclusion probability across all passes: 0.954
- uncertainty source agreement: corr(seed_std, mc_std) = 0.887, corr(seed_std, noise_std) = 0.773
- classes: 0 HIGH_CONF_LONG, 0 HIGH_CONF_SHORT_DIAGNOSTIC (review-only), 16 UNSTABLE
- full per-stock panel: C:\Users\wilso\source\code\AI-Quant\reports\transformer_gpu\v12_big_transformer\deep_scores_2026-07-30.csv

## Historical uncertainty value (frozen panels — the gate)

### CH
- corr(seed_std, realized rank error): 0.0618 (positive on 64% of dates)
- blend book plain 2.147 (DD -10.6%) vs drop-high-std -1.768 (DD -46.5%)

### BR
- corr(seed_std, realized rank error): 0.0695 (positive on 66% of dates)
- blend book plain 1.443 (DD -18.0%) vs drop-high-std -0.393 (DD -39.6%)

## Notes

- Multi-snapshot/multi-refit ensembles need checkpoints collected over time; production daily retrain overwrites — if ever wanted, a snapshot-retention policy would be a separate (operational) proposal.
- Confidence filtering remains a CLOSED line; the book comparison above is evidence about uncertainty value, not an adoption proposal.
- Review-only; no orders; short list is diagnostic.

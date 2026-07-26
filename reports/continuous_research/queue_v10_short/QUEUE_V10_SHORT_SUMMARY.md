# Queue v10-short Summary — New Signal/Model Research (Option B)

Completed 2026-07-26. 10/10 done, 0 failed, ~4.4 h GPU (within the 4–6 h
budget). Champion path verified bit-identical with flags OFF before any run
(pre/post anchor diff empty; model.py untouched — wrappers only). Production
defaults unchanged. No data_cache mutation (runner never calls daily_ops).

## Screen results (3-seed, champion window, val-IC selection)

| ID | val IC | vs baseline 0.04813 | Verdict |
|---|---|---|---|
| BASE3_s012 (seeds 0–2) | 0.04813 | — | baseline / promotion bar |
| BASE3_s101112 (seeds 10–12) | 0.04838 | +0.00025 | D1 spread reference |
| A1_CS3 cross-sectional attention | **0.03704** | −0.011 | **REJECT — decisive.** CS attention over each date's names hurts val IC by 23%. The cross-sectional-attention line closes at this trunk/scale. |
| B1_MT3 multi-task heads | **0.05026** | +0.00213 | **cleared the +0.002 bar → promoted** (margin is thin — noted) |
| D1_NOISE (both seed-sets) | 0.05021 / 0.04764 | spread 0.00257 | **REJECT** — cross-set spread LARGER than baseline's 0.00025 |
| D1_DROP (both seed-sets) | 0.04852 / 0.04671 | spread 0.00181 | **REJECT** — same; also mean slightly worse |
| C1_REV_CPU 5d-reversal leg | — | — | **REJECT — decisive.** rev5 standalone Sharpe **−0.51 (CH) / −1.05 (BR)**: short-horizon reversal does not exist in this universe (continuation instead). Any blend weight hurts (rev10: 1.55/1.34 vs 2.15/1.44). GPU branch dead; reversal line closed. |

D1 side-finding worth keeping: at 3 seeds the **val-IC cross-set spread is
already negligible (0.00025)** — the v7 seed-set variance (1.843 vs 2.147)
lives in OOS book space, not in val IC. Augmentation aimed at val-IC variance
was aimed at the wrong layer; the fragility is downstream of selection.

## Promotion (PROMOTE_TOP_K=1): B1_MT3 at 7 seeds, dual-window

| Window | MT blend LS net60 | vs blend default | vs D7b | max DD | 2022 |
|---|---|---|---|---|---|
| 2023→ champion | **2.011** | −0.136 | −0.144 | −11.4% (def −10.6%) | — |
| 2021→ bear | **1.515** | **+0.072** | +0.074 | **−13.0%** (def −18.0%) | **+0.23** (def −0.15) |

Both windows inside/above the seed-robust ranges (1.85–2.15 / 1.30–1.45);
the bear point is ABOVE the range top. Per pre-registration the result is
recorded and **adoption is the user's decision**.

## Honest assessment of the MT result

The profile is exactly what v9's Track 4 tried and failed to buy at book
level: +5pp bear DD, a positive 2022, +0.07 bear Sharpe — paid for with
−0.14 champion-window Sharpe. Two reasons NOT to adopt yet:

1. **Seed-set sensitivity unknown.** −0.136 is inside the ±0.15 seed-set
   noise band v7 established for 7-seed books. The bear-side gains could be
   partly seed luck; MT has exactly one 7-seed draw per window.
2. **The screen edge was marginal** (0.05026 vs bar 0.05013), and the E3
   lesson says marginal val-IC edges are not decision-grade — the promotion
   rule was honored, but the margin deserves suspicion.

**Recommendation: hold adoption; run a targeted MT validation battery as the
first item of v10b** — disjoint-seed replication (seeds 10–16, both windows,
mirroring SR1/SR2) + refit-63 protocol check. If the bear/2022 profile
survives disjoint seeds, MT becomes either a champion challenger or the
natural "defensive spec"; if it does not, it closes cleanly.

## Ledger

- Runtimes: screens 0.9–2.6 ks each (A1's per-date batching ≈ 4× slower);
  promotion 7.0 ks (one bear refit hit GPU contention, 1.8 ks).
- Panels: SCHED_BASE3_*, SCHED_A1_CS3, SCHED_B1_MT3, SCHED_D1_*,
  SCHED_P7_B1_MT3_{champ_2023,bear_2021} (gitignored as usual).
- Flags added (default OFF, preset-key gated): cs_attn, mt_aux, aug_noise,
  aug_datedrop. model.py untouched.
- Not run (per approval): A2, A3, B2, C2, C1-GPU, D2 — available for v10b.

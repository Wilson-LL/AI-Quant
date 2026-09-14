# 03 — Survivorship bias (B2)

## Verdict: **SURVIVORSHIP_BIAS_PRESENT**

The universe is a hand-curated list of 112 currently-liquid TWSE names
(`research/data.py:64-124`, snapshot 2026-07-06, "push toward 100+ names")
applied **retroactively to all history from 2015**. There is no
point-in-time constituent file anywhere in the repo, no delisted names,
and no mechanism to add names that were liquid in 2018 but are not
today. Every backtest, panel, and reference number (2.147 / 1.443 /
1.989 / 1.455) is computed on this survivor set. The system discloses
this on every emitted artifact ("survivorship-biased research universe").

## How material is it here?

Mechanism matters. Survivorship inflates results mainly through (a)
names that collapsed/delisted being absent from the *short/bottom* side
and from the long side's tail risk, and (b) selecting today's winners
whose past momentum was, in hindsight, "right".

Mitigating facts specific to this universe:
- All names are large/mid-cap TWSE constituents (TW50/TW100-class);
  delisting frequency in that tier over 2015–2026 is low, and the
  strategy is **long-only in production** (the short leg is research-only,
  so mechanism (a) affects only the L/S numbers).
- The signal is cross-sectional *relative* rank within the set; a survivor
  set tilts the universe's mean upward but not obviously the within-set
  ordering.
- The 2021–2022 bear window (BR) shows the strategy approximately flat
  (2022 net60 ≈ −0.15) rather than heroic — a survivor-only bias would
  more often show implausible crash resilience.

Aggravating facts:
- Hindsight selection of ~112 names in 2026 is itself a lookahead on
  *which* names to consider — names that are liquid today because they
  outperformed. Long-only momentum on a "winners" set can look better
  than on a point-in-time set. The size of this effect is **not
  estimable from inside the repo** (no counterfactual universe).
- Concentration: v9-R5 found one name (1519) contributes 9–12% of
  positive PnL and bear-window Sharpe retention after dropping the top 5
  contributors is 0.667 — the results lean on a handful of survivors.

## Estimate (bounded, not measured)

Literature on survivorship in equity cross-sectional strategies typically
finds annualized return inflation on the order of 1–4 pp for long-only
large-cap sets and materially more for small caps or short legs. For a
strategy quoting ~50% annualized (CH) the *relative* inflation of the
Sharpe is probably modest (order 5–15%), but the honest statement is:
**the reference numbers are upper bounds; a point-in-time universe is
required before any of them can be called unbiased.**

## Data required to reconstruct point-in-time eligibility (P2)

| Need | Why | Candidate source |
|---|---|---|
| Listing date + delisting date for every TWSE/TPEx common stock since 2014 | membership existence | TWSE/TPEx listed-company registers; FinMind `TaiwanStockInfo` + delisting list |
| Historical index constituents with effective dates (TW50, TW Mid-Cap 100, or TWSE market-cap ranks) if a size screen is used | the current 112 are a size/liquidity-curated set | TWSE index constituent announcements (quarterly reviews) |
| Daily volume/turnover for ALL candidates, not just survivors | liquidity screen at T | twstock/TWSE daily quotes for the full list (bulk fetch) |
| Corporate-action history for the same names | consistent with P1 | as in 02/P1 |

Rule to apply once available: scoreable at T iff listed ≥ 132 sessions,
in the size/liquidity screen as of T, and not within 20 sessions of a
delisting announcement. Until then every historical figure keeps the
label **SURVIVOR-UNIVERSE UPPER-BOUND REFERENCE**; research is not
blocked, but no historical number is a production expectation.

## Proposed proper solution (separate work; not done here)

1. Obtain historical TWSE index constituents (TW50/TW100/TWSE Mid-Cap)
   with effective dates, plus the listing/delisting register.
2. Rebuild the panel with a **point-in-time eligibility mask**: a name is
   scoreable on date T iff it was a constituent / listed and had ≥132
   sessions of history on T.
3. Re-run the frozen champion on the masked panel (CPU, minutes) and
   report the delta as the survivorship cost. Only then re-run training
   with the historical universe (GPU).
4. Until then, keep the disclaimer and treat OOS numbers as optimistic.

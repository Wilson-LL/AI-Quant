# v16 Stage B2 — TWSE Price-Domain Rule Audit (Task 1)

Date: 2026-08-18. Requirements per the Stage B2 specification; verified
against official TWSE sources where possible on 2026-08-18:

- twse.com.tw "Trading Mechanism Introduction" (fetched 2026-08-18):
  daily price fluctuation limit for stocks and domestic ETFs is
  "10 percent above and below the **auction reference price at market
  opening** for the given day" — confirming the limit anchors to the
  opening-auction REFERENCE price, not the actual 09:00 opening trade.
- twse.com.tw search results (2026-08-18): IPO securities are not
  subject to daily price fluctuation limits during their first five
  trading days; volatility-interruption / special-instrument carve-outs
  exist (warrants, sub-NTD1 reference prices, bonds at 5%).
- The per-instrument tick tables on the official page differ by
  security type (ETFs/ETNs/warrants have their own ladders); the
  extraction was not reliable enough to certify non-stock ladders, so
  they are treated as UNVERIFIED here.

## Rules implemented (ordinary listed stocks)

1. Standard daily limit: ±10% around the day's **market opening auction
   reference price** — NOT the actual opening transaction price.
2. On an ordinary session with a valid previous closing price, the
   auction reference is normally the previous close. This repo can only
   ASSUME that (see limitations), so the domain status is
   `NORMAL_DAY_ASSUMPTION`, never `CONFIRMED_STANDARD_LIMIT`.
3. Tick ladder for ordinary stocks (user-specified requirement,
   consistent with the pre-existing Stage B1 helper):

   | price P | tick |
   |---|---|
   | P < 10 | 0.01 |
   | 10 ≤ P < 50 | 0.05 |
   | 50 ≤ P < 100 | 0.10 |
   | 100 ≤ P < 500 | 0.50 |
   | 500 ≤ P < 1000 | 1.00 |
   | P ≥ 1000 | 5.00 |

4. Legal limit prices are direction-aware tick projections that never
   exceed ±10%: `legal_limit_up` = greatest legal tick ≤ ref×1.10;
   `legal_limit_down` = smallest legal tick ≥ ref×0.90. Reference
   regression case: ref 40.60 → **limit up 44.65, limit down 36.55**.
5. The ladder is NOT generalized to ETFs, warrants, ETNs, bonds, or
   other instrument types (their ladders/limits differ and were not
   certified) — those return `UNKNOWN` domain status with NA limits.

## Special cases (Task 21 audit result: repo data is INSUFFICIENT)

The repo holds raw OHLCV only — no corporate-action calendar, no
listing-date table, no suspension records, and prices are known to be
dividend/split-UNADJUSTED (timing_audit F5). Therefore:

- Ex-right / ex-dividend reference adjustments: NOT detectable ex ante.
- IPO first-five-sessions (no limit): NOT detectable (no listing dates).
- Resumption after suspension / no-previous-close: partially detectable
  (missing prior row → `UNKNOWN`).

Consequence (conservative, implemented): every ordinary-stock domain is
labeled `NORMAL_DAY_ASSUMPTION` with `reference_source=PREVIOUS_CLOSE`
and MEDIUM confidence, and every nightly report carries the caveat that
an ex-date or special day may use an adjusted auction reference the
system cannot see. Special-day references are NEVER inferred from price
movement. Missing/invalid previous close → `UNKNOWN`, limits NA,
confidence LOW.

## Session-structure semantics (Task 22)

TWSE's regular session = opening auction (~09:00) + continuous matching
+ closing auction (13:25–13:30). Stage B has daily OHLC outcomes only,
so B2 estimates distributions of the next OPEN, daily HIGH, daily LOW,
and CLOSE. It cannot estimate time-of-day prices, time-to-level, or
high-before-low ordering — those wait for Stage C and the accumulating
v15 intraday dataset.

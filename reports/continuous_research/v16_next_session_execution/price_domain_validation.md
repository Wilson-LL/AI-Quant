# v16 Stage B2 — Legal Domain + Next-Open Distribution Validation

Date: 2026-08-18 · Data: price_domain_validation.csv · Methodology frozen
beforehand in price_domain_methodology.md. 1,703 role-observations, 68
rebalances, BR panel 2021→, expanding windows (< T).

## Legal-domain hard gate: PASS

**0 violations** across every emitted price of every evaluated
observation — all entry/sell band levels and all distribution display
prices are legal ticks inside [legal_limit_down, legal_limit_up] under
the normal-day assumption (reference = previous close). Unit tests
additionally pin the official-style anchor (ref 40.60 → 44.65 / 36.55)
and the tick-boundary transitions (9.99/10, 49.95/50, 99.9/100,
499.5/500, 999/1000).

## Next-open distribution coverage (target = nominal quantile)

| Group | n | ≤p10 | ≤p25 | ≤p50 | ≤p75 | ≤p90 | med \|p50 err\| |
|---|---|---|---|---|---|---|---|
| ALL | 1,703 | 12.3% | 28.4% | 52.4% | 74.9% | 90.8% | 65 bps |
| 2021–2025 | 1,538 | 7–15% | 22–35% | 44–66% | 69–88% | 87–97% | 55–77 bps |
| **2026 YTD** | 165 | 21.2% | 30.9% | 44.2% | 63.0% | 83.6% | **125 bps** |
| vol LOW / MED / HIGH | 167/459/1,077 | 12/10/13% | 27/26/30% | 56/52/52% | 69/72/77% | 91/87/92% | 38/54/81 bps |

Reading: coverage tracks the nominal quantiles within a few points
overall — the conditional distributions are usable as stated. The known
regime caveat repeats: **2026 YTD is mildly miscalibrated** (medians sit
too high → actual opens land lower than predicted; p50 error doubles).
This mirrors the B1 finding; per pre-registration nothing is retuned —
the expanding window will absorb the regime with lag, and the row-level
`range_reach_data_quality` / fallback flags stay the honest signal.

No profitability claim; this validates distribution geometry only.

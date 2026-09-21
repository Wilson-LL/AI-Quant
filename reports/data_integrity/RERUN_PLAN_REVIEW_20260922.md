# Review of RERUN_PLAN_AFTER_DATA_BASELINE_V2 (2026-09-22). Plan only; nothing executed.

**Research stays frozen** until PRODUCTION_BASELINE_V2 is recorded and the user lifts the freeze. No GPU work was run for this review.

## How large was the V1 input contamination? (measured, CPU, read-only)

This is the share of (symbol, decision-date) model inputs whose required 191-session window crossed a missing session. Source: `tmp/contamination.py` logic, V2 calendar.

| interval | V1 contaminated inputs | V2 | symbols affected (V1) | decision dates affected (V1) |
|---|---|---|---|---|
| 2026-H1 burned interval (P0, P4, residual audits) | **12.6%** | 3.3% | 21 | 131 / 131 |
| 2023–2026 reference window | **10.0%** | 2.5% | 44 | all 856 |
| 2021–2026 reference window | **9.5%** | 2.3% | 58 | all 1,346 |
| 2015+ full history | 7.8% | 2.5% | 68 | 2,550 / 2,850 |

**V2 remainder.** The V2 figures are registered or unresolved symbol no-trade days (6 confirmed, 109 unresolved historical). The training gap guard excludes them. They are not spliced.

**Implication.** Every prior result touched roughly 1 in 8–10 of its inputs, on every date. Relative (same-data, A-vs-B) conclusions are much more robust than absolute levels, because both arms saw the same contaminated inputs. Knife-edge labels are the exception.

## Revised order (supersedes the order in RERUN_PLAN_AFTER_DATA_BASELINE_V2.md)

1. **Rebuild the champion and simple-baseline references on V2.** This covers champion blend50 + band10 (7 seeds), momentum 126_5 alone, ridge panel and equal-weight, with a frozen recipe and the gap guard on. It gates everything below.
2. **Recompute Transformer vs momentum / ridge** on identical V2 cross-sections, with HAC intervals.
3. **Re-run H-RESIDUAL-SIGNAL on V2**, using the unchanged preregistration (5012d5b) and the V2 champion scores from step 1.
4. **Quantify how much the cache repair changes the prior conclusions.** Compare V1 and V2 for each result below, and decide whether V2 "materially changes the relevant inputs or baseline behaviour" using a rule fixed now:
   - the change exceeds the V1 seed-dispersion band, **or**
   - it flips a preregistered label.
5. **Rerun P0 / P4 experiments only if step 4 says so.** Do not rerun every historical experiment blindly.

## Categorisation of prior results

| result (source) | V1 finding | category | reason / trigger |
|---|---|---|---|
| Champion reference levels: L/S ≈ 1.85–2.15 (2023–26), 1.30–1.45 (2021–26); planning numbers 1.92 / 1.37 | absolute performance | **FULL_RERUN_REQUIRED** | Absolute level on inputs 10% / 9.5% contaminated. Step 1. |
| Simple baselines: momentum 126_5, ridge, EW (v17 `11_baseline_comparison`) | transformer residual IC ≈ 0.02 / 0.00 over the baselines | **FULL_RERUN_REQUIRED** | Both sides of the comparison were contaminated unevenly (momentum reads only 2 closes; sequences read 191). Step 2. CPU for the baselines. |
| H-RESIDUAL-SIGNAL (`24_h_residual_factor_result`) | Case C, residual WEAK; placebo floor tripped the METHOD_SENSITIVE flag by 0.0001 | **SENSITIVITY_RERUN** | CPU-only given V2 scores. The label sits near its thresholds, and the 2026-H1 inputs were 12.6% contaminated. Step 3. |
| P4-B0-LONG, H-EPOCH-LONG (`22_p4_b0_long_oos_verdict`) | REJECTED at minimum margins (7/9 fits; mechanism short by 0.002–0.004) | **SENSITIVITY_RERUN, conditional** | The label is knife-edge by its own report, and the burned interval was 12.6% contaminated. Rerun only if step 4 shows V2 moves the 2026-H1 champion or baseline beyond seed dispersion. Costs GPU hours: needs explicit approval. |
| P0-A′ / P0-A refit cadence (`16`, `17`) | 126-session refit not beaten by 5-session or daily; dense arms val IC +0.027 but OOS −0.013 to −0.016 | **STILL_VALID (relative), conditional** | A paired A-vs-B comparison on identical data, where the contamination hits all arms. Reopen only if step 4 finds a material V1→V2 change on 2026-H1. |
| Validation-policy diagnosis (`18`) | val/OOS anti-correlation is a time-trend confound | **STILL_VALID** | A methodological finding about the sliding validation window, independent of data holes. |
| Epoch-selection diagnosis (`19`) | keep validation-IC early stopping until a challenger passes the frozen bar | **STILL_VALID** | Same reasoning as `18`. Revisit only via P4 if step 4 triggers it. |
| Continuous-loop closed lines (~20 challengers, v10 lines, 14-seed line) | rejected on preregistered dual-window gates | **STILL_VALID (relative)** | Each is relative to the champion on the same data. Reopen a specific line only with new information, e.g. a champion shift in step 1 larger than that line's rejection margin. |
| v9 deployment validation (break-even 632 / 463 bps, capacity, delay, settlement) | deployment case passes | **SENSITIVITY_RERUN (CPU)** | Break-even scales with the gross spread, so recompute it from the step 1 V2 reference. The execution mechanics are unaffected. |
| v17 data audits: leakage, survivorship, target / horizon, cost (`02`–`08`) | as reported | **STILL_VALID, plus addendum** | Add the cache-hole finding (H-DATA-INTEGRITY) as a new data-validity item. Survivorship and unadjusted prices remain open caveats. |

## Notes on the original plan

- The original plan put "reassess P0/P4 invalidation" last, as a blanket review. The revision above makes P0/P4 reruns **conditional** on a measured, preregistered materiality rule (step 4). That avoids spending GPU on experiments whose relative conclusions V2 cannot change.
- Burned-interval discipline is unchanged: 2026-H1 results remain diagnostics, never promotion evidence. The first prospective evidence after PRODUCTION_BASELINE_V2 is the matured paper ledger on V2 data.

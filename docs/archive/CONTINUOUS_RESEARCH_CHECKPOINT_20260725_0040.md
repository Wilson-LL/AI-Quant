# Continuous Research Checkpoint — 2026-07-25 00:40 (GPU research mode)

Covers 2026-07-24 17:30 → 2026-07-25 00:40 · commits 3f392bf → this.

## 1. Experiments run

Queue v5 tail (seq120 REJECT — sequence axis closed), queue v6 (9-seed:
saturation, keep 7; conservative spec ADOPTED on 7-seed panels), Edit 4
(7-seed daily retrain wired + live-tested), daily 07-24 cycle ×2 (books on
adopted spec), R2 bootstrap (PASS, 100% positive), queue v7 validation battery
(4 GPU configs + 1 accidental exact replication).

## 2. GPU utilization summary

Near-continuous training 17:30→00:30 (~6.5h GPU work): 9-seed ×2 windows,
14 refit-63 fits, disjoint-seed ×2 windows, crash-first window, 7-seed daily
retrain. VRAM 2.2–2.9 GB (alongside FFXIV ~3 GB), 0 OOM, 1 config failed on a
spawn-inheritance bug (fixed; the failure doubled as a determinism check —
bit-identical reproduction of BEAR_A8).

## 3. Best candidate / MAJOR HONESTY CORRECTION

Spec unchanged: **7-seed blend50+band10** (+ conservative band15/cap7.5
deployment variant). But the validation battery **retracted the "+0.09
improvement" claim**: disjoint seeds 10–16 give 1.843 (champion window) vs
2.147 for seeds 0–6 — outside tolerance. References are now stated as
seed-robust ranges: **≈1.85–2.15 (2023–26), ≈1.30–1.45 (2021–26)**, with
bootstrap medians (1.92 / 1.37) as planning numbers. Keep-7 rationale: never
worse than 5, trivial cost — but the honest claim is parity-with-smoothing,
not outperformance. Third window (crash-first 2022→) recorded: 1.321/−14.2%.

## 4. Rejected

seq120 (axis closed), 9-seed (saturation), and the v7 battery produced no new
candidates by design (validation, not search).

## 5. Comparison vs champion

All GPU-mode alternatives remain rejected; the champion book's evidence is now
range-stated, bootstrap-backed, protocol-robust (refit-63 2.085), and
determinism-verified.

## 6. Next queue (v8, running)

14-seed ensemble (both seed sets) — principled response to the seed-variance
discovery: does averaging both sets converge both windows to their midpoints?
Adopt-14 only if both windows land at/above midpoint; else 7 stays.
After v8: paper-ledger maturation readout; daily ops; full-fields ~2027-01.

## 7. Was the GPU meaningfully used?

Yes — and the most valuable output was negative: the seed-sensitivity flag
that corrected our own headline number before it could mislead deployment.

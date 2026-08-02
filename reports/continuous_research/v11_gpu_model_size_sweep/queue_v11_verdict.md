# Queue v11 verdict — GPU model-size sweep

## VERDICT: KEEP_CURRENT_PRODUCTION / REJECT_ALL_CHALLENGERS

User decision 2026-07-30, closing the queue after Phase 0 + Phase 1
(9/9 items done, 0 failed, 0 OOM). Phase 2 (V11_B 7-seed dual-window
confirmation) was staged per the pre-registered rule but **cancelled by
user decision** — a +0.002 screen margin is seed-noise and does not justify
~4 h of confirmation GPU.

## Reasons (as accepted by the user)

1. **Phase 0 baseline reproduced exactly:** blend L/S net60 **2.147** (CH,
   2023→) / **1.443** (BR, 2021→) — byte-level agreement with the standing
   A8 references, so every candidate delta is attributable to architecture.
2. **Phase 1 showed no meaningful challenger.** Screen ranking (3 seeds,
   CH, bar = baseline's own screen 1.969): B 1.971 · D 1.918 · C 1.807 ·
   E 1.789 · F 1.711 · G 1.680.
3. **V11_B wider96 cleared the bar by +0.002** — seed-noise thin — with
   slightly worse max DD (−10.8% vs −10.5%), higher turnover (0.305 vs
   0.301), ~1.3× train time and ~1.3× VRAM: parity at higher operational
   cost is a rejection.
4. **Every larger model degraded book-level performance**, monotonically
   with distance from the production shape (top-quintile overlap fell
   0.88 → 0.68 as Sharpe fell 1.97 → 1.68).
5. **Seq90 replicated its previously closed rejection** (E 1.789; F worst
   DD −15.8%) — the sequence axis stays closed.
6. **Validation IC again dissociated from book quality** (dissociations
   #5–7: C val 0.0537/book 1.807; F best-of-sweep val 0.0557/book 1.711;
   B's val edge 0.0518 bought nothing). Standing rule re-confirmed:
   **do not adopt based on validation IC alone.**
7. **More VRAM did not improve throughput or books:** epoch time flat from
   batch 2048 upward (launch-bound workload); the 9–11 GB target was
   reachable only at batch 16384, where the 1-epoch val IC degraded
   (fewer optimizer steps). GPU-memory targeting is **recorded, not
   adopted**; **batch 1024 remains the practical recipe**; all sweep models
   fit in ≤4.2 GB.

## Standing state after v11

- **Production architecture unchanged:** preset B (h64 / 2 layers / 4 heads
  / ff128 / seq60), close_only, tgt_rank_20, 7-seed, batch 1024. No
  challenger promoted, to production or to paper.
- **Closed at book level by this queue:** hidden 96 (parity-at-cost),
  hidden 128 (heads 8), depth 3 (both widths), seq90 (re-confirmed),
  width×length combination. Do not reopen without new evidence
  (new data ~2027-01, or a new regime in the paper ledger).
- The only training-module change was the flag-gated `batch` preset key
  (anchor-verified byte-identical when absent). Daily retrain, inference,
  decision book, paper ledger, holdings overlay, data_cache: untouched.

## Artifacts

queue_v11.json (P2 items marked cancelled) · queue_v11_results.{csv,md} ·
queue_v11_book_metrics.csv · queue_v11_gpu_usage.csv (probe ladder +
per-run rows) · queue_v11_run_manifest.json (closure block) ·
logs/<item>.log · panels SCHED_P*.csv.gz (gitignored) ·
checkpoints/v11_sweep/ (gitignored).

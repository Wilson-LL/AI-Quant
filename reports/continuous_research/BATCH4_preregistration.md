# Pre-registration — Cycle 18 (BATCH4 excess-vs-sector target at champion strength)

Registered: 2026-07-23 10:15 (before running)

Last untested "promising" item: tgt_exc_sec_20 scored LO 1.78 / L/S 1.46 in the
2-seed G4 screen — the best non-champion LO. A1/E1 showed 2-seed orderings are
unreliable in both directions, so this needs the 5-seed dual-window treatment
before the line is closed.

Configs: tgt_exc_sec_20, preset B, 5 seeds, deep cache, OOS 2023-01→ and 2021-01→.
Post-pass: standalone books + 50/50 z-blend with D1.2 + band10, both windows.

Gates vs standing refs (B4d 2.06/−10.7%; B4c 1.47/−18.7%; champion standalone
1.91; bear tf 1.14):
- Standalone must beat champion standalone on BOTH windows to become the model;
  blend must beat BOTH blend references to become the book. Margin ≥ 0.05.
- Partial wins → monitor with explicit windows. Losses → reject, line closed,
  "promising but unproven" list emptied.

Runtime: ~1.5h GPU (5 seeds × 18 refits + build).

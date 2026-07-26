# Queue v10b Proposal — Full-v10 Continuation, Reshaped by Short-v10 Results

Status: **PROPOSED — NOT STARTED** (2026-07-26). Awaiting user approval.

Short v10 changed the priority order. The continuation is no longer "the rest
of full v10": two of its items are now dead, and one new item — validating
the MT-heads result — is more valuable than everything else combined.

## Dropped permanently (evidence, not budget)

- **A2 inverted attention** — A1 showed cross-sectional attention hurts
  val IC by 23% at this trunk/scale; the pure-CS variant is strictly more
  radical along the failed axis. Line closed.
- **C1 GPU branch (reversal feature)** — rev5 standalone Sharpe is negative
  (−0.51/−1.05); TWSE shows continuation, not reversal. Line closed.

## Proposed v10b (priority order)

### 1. MT validation battery (NEW — the reason v10b exists) ~4–5 h
Mirror of the v7 battery that caught the 7-seed seed-luck inflation:
- **SR-MT1**: MT heads, disjoint seeds 10–16, champion window (~1 h)
- **SR-MT2**: MT heads, disjoint seeds 10–16, bear window (~1.5 h)
- **RF-MT**: MT heads, seeds 0–6, refit-63 protocol check, champ window (~1.5 h)

Pre-registered gates: the bear/2022 profile must survive disjoint seeds —
bear blend ≥ 1.44 AND 2022 ≥ 0.0 AND bear DD ≤ −16% or better; champ within
the 1.85–2.15 range. All three hold → MT becomes a formal champion
challenger / defensive-spec candidate (adoption still user-gated). Any fails
→ MT recorded as seed-luck, line closed, current champion stands with no
residual ambiguity. Either outcome is decisive.

### 2. B2 quantile head screen (~30 min + CPU book pass)
Unchanged from the v10 proposal; overfit-flagged (sizing rule), single
pre-registered damping rule, two-stage gate.

### 3. C2 market-residualized target (~25 min)
Unchanged; carries the pre-registered inversion tripwire (val IC ≥0.055 with
OOS IC <0.02 → inversion #3, line closed permanently).

### 4. A3 TCN anchor (~20 min)
Now MORE interesting after A1: attention-hostility of the cross-section is
established; the question "is temporal attention itself earning its keep?"
gains weight. Expected-lose anchor; promotes only on an upset.

### 5. D2 distillation (~30 min)
Runs last, and only against whichever spec is champion after item 1
resolves (distilling a spec that's about to change would be wasted).

## Budget

Screens + battery ≈ 6–7 h GPU; plus at most ONE promotion slot
(PROMOTE_TOP_K=1, same +0.002 val-IC bar) for B2/C2/A3 winners ≈ +3.5 h
worst case. Expected total: **6–8 h**. The MT battery is not a "promotion"
(it validates an already-promoted result) and does not consume the slot.

## Guardrails (unchanged)

Signal edits stay flag-gated with the champion path re-anchored bit-identical
before the first run; production defaults untouched; no data_cache mutation;
cache read-only; config-level dedup; dual-window gates vs seed-robust ranges;
CUDA-error = stop and write a stability note; every adoption requires
explicit user sign-off. NOT started until approval.

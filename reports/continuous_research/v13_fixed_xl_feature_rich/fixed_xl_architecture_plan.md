# v13 Fixed-XL Architecture Plan (Task 3)

**Fixed research constraint: XL2-class — h1024 / trans_layers 24 / heads 16
/ ff4096 / seq60, LSTM_CondTransformer.** Confirmed numbers (v12 measured +
estimator, updated for feature width):

1. **Parameters:** 311,906,369 at input_dim 10; ~+139k per +34 input
   features (4·1024·ΔF for the LSTM input matrix + 1024·ΔF for
   cross_kv_proj) → ~312.1M at F6(44). 1 GiB fp32 = 268.4M — target held.
2. **Checkpoint:** 1189.9 MB measured (fp32 state_dict); +<1 MB with F6.
3. **VRAM:** 6.45 GB measured (micro 64 × accum 16 + gradient
   checkpointing, AMP) + X tensor 0.66→2.91 GB by feature set → ≤ ~9.5 GB;
   fp16 X halves that if needed. Fits the 16 GB card.
4. **Training time:** ~22.5 min/epoch full-history (measured 1348 s);
   5 h budget ⇒ ~8 epochs fixed + best-by-val, single refit.
5. **Inference:** seconds at eval batch ≤ 512 (scaled eval batch is
   mandatory — the fixed-8192 eval batch OOMed at h1024 in v12).
6–8. **Wider input is structurally trivial:** input_dim flows only into
   `lstm.weight_ih` and `cross_kv_proj`; positional embedding (seq×hidden)
   and cross-attention are width-agnostic; the input projection is NOT a
   bottleneck (0.04% of parameters at F6).
9–10. **Required and retained from the v12 kit** (`fit_big` in
   run_queue_v12_big_transformer.py): AMP + gradient checkpointing
   (encoder layers, `use_reentrant=False`) + micro-batch 64 / accumulation
   16 + message-based OOM classification (`AcceleratorError` counts) +
   subprocess isolation per run (hard OOM poisons the CUDA context) +
   wall-clock budget guard. v13 checkpoints go ONLY to
   `checkpoints/v13_fixed_xl_feature_rich/` (gitignored).

## Known failure mode to test against

v12's XL2 on close_only **collapsed to the mean** (train loss pinned at the
rank-target variance 0.3332, constant per-date output, val IC NaN). The
Phase-3 screen's primary diagnostic is therefore collapse detection:
train-loss trajectory vs 0.3332, per-date score dispersion, rank
non-degeneracy — richer features preventing collapse is the minimum
interesting outcome. Contingency documented but NOT enabled without
separate approval: lr warmup / lower lr for XL (a recipe change, out of
current pre-registered scope).

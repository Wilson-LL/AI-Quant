# Transformer Config Guide — Where Every Parameter Lives

Read-only inspection map (2026-07-29). No code was modified. Line numbers
refer to the current working tree at commit `795e13c`.

The one-sentence version: **the architecture is `model.py`, every
architectural hyperparameter comes from `PRESETS` in
`train_transformer_eod.py` (production = preset `B`), optimizer settings are
`fit_one()` keyword defaults, the 7-seed list is hardcoded in
`mode_daily_retrain()`, checkpoints live in `checkpoints/transformer_eod/`,
and the 50/50 blend weights are literals in `research/paper_trading.py` and
`research/blended_decision_book.py`.**

## 1. Architecture definition

- `model.py:4-71` — `class LSTM_CondTransformer`: LSTM trunk → linear
  projection + learned positional embedding (`pos_emb`, `model.py:28`) →
  cross-attention (query = LSTM states, key/value = projected raw inputs,
  `model.py:39-44,63-65`) → `nn.TransformerEncoder` (`model.py:30-37`) →
  last-timestep MLP head `Linear(hidden,32)→ReLU→Dropout→Linear(32,1)`
  (`model.py:46-51`).
- `train_transformer_eod.py:55-60` — `build_net(input_dim, cfg)` maps preset
  keys onto the constructor: `lstm_hidden = trans_hidden = cfg["hidden"]`.
- `train_transformer_eod.py:68-166` — v10 flag-gated wrappers (`CSAttnNet`,
  `MTNet`, `QNet`, `TCNNet` + `_embed()`), **default OFF**, activated only by
  preset keys `cs_attn / mt_aux / quantile / tcn / aug_noise / aug_datedrop`
  that are absent from presets A/B/C. `model.py` itself was never touched.

## 2–6. hidden (d_model), layers, heads, dropout, feedforward

All five come from one place — `PRESETS`, `train_transformer_eod.py:38-45`:

```python
PRESETS = {
    "A": dict(hidden=64,  trans_layers=2, lstm_layers=1, heads=4, ff=128,
              dropout=0.2, seq_len=40, max_epochs=20, patience=3, seeds=3),
    "B": dict(hidden=64,  trans_layers=2, lstm_layers=1, heads=4, ff=128,
              dropout=0.2, seq_len=60, max_epochs=25, patience=3, seeds=5),
    "C": dict(hidden=128, trans_layers=2, lstm_layers=1, heads=4, ff=256,
              dropout=0.2, seq_len=60, max_epochs=20, patience=3, seeds=3),
}
```

**Production is preset `B`** (h64, 2 encoder layers, 4 heads, ff 128,
dropout 0.2, seq60). Where each key lands:

| Key | Consumed at | Effect |
|---|---|---|
| `hidden` | `train_transformer_eod.py:57-58` → `model.py:21,27,31` | LSTM hidden size AND transformer d_model (always equal) |
| `trans_layers` | `model.py:37` | `TransformerEncoder(num_layers=...)` — 2 |
| `lstm_layers` | `model.py:22` | 1 (so LSTM-internal dropout is disabled, `model.py:24`) |
| `heads` | `model.py:32,41` | encoder `nhead` AND cross-attention heads — 4 |
| `ff` | `model.py:33` | `dim_feedforward` — 128 |
| `dropout` | `model.py:24,34,49` | LSTM (inactive at 1 layer), encoder layers, fc head — 0.2 |

## 7. seq_len

- Defined per preset (`B`: 60, `train_transformer_eod.py:42-43`).
- Consumed: positional embedding shape `model.py:28`; dataset windowing via
  `build_dataset(feature_set, seq_len=PRESETS[preset]["seq_len"], ...)` in
  `mode_daily_retrain` (`train_transformer_eod.py:535`).
- At inference it is NOT re-derived from the preset — it is read back from
  the checkpoint's saved `cfg` (`inference_transformer_eod.py:145`), so
  scoring always matches whatever the ensemble was trained with.

## 8. feature_set

- Column lists: `FEATURE_COLS` dict, `dataset_transformer_eod.py:152-167+`.
  Production `close_only` = 10 features: `log_ret_1, mom_5, mom_20, mom_60,
  mom_126_5, vol_20, vol_60, dist_hi_60, dist_lo_60, px_over_ma20`
  (`dataset_transformer_eod.py:153-154`).
- Default `"close_only"`: `mode_daily_retrain` signature
  (`train_transformer_eod.py:525`) and CLI `--feature-set`
  (`train_transformer_eod.py:584`).
- Inference recovers it from the checkpoint (`ck["feature_set"]`,
  `inference_transformer_eod.py:58-61`) to size `input_dim`.

## 9. Target / horizon

- Targets built as cross-sectional rank of forward returns:
  `tgt_rank_{Y}` in `build_dataset`, `dataset_transformer_eod.py:260-267`
  (horizons default `(5, 10, 20)`, `dataset_transformer_eod.py:213`).
- Production defaults `target="tgt_rank_20"`, `horizon=20`:
  `mode_daily_retrain` (`train_transformer_eod.py:525-526`), CLI
  (`train_transformer_eod.py:586-587`), and `walkforward`
  (`train_transformer_eod.py:390`).

## 10. Learning rate / weight decay / batch size / epochs

- `fit_one()` keyword defaults, `train_transformer_eod.py:217-219`:
  **`lr=3e-4, weight_decay=1e-4, batch=1024`**; optimizer is AdamW
  (`train_transformer_eod.py:249`).
- Epochs: per preset `max_epochs` (B: 25) with `patience=3` early stopping
  on val rank-IC and `min_epochs=2` (`train_transformer_eod.py:289-292,372`).
- `walkforward()` re-exposes `lr=3e-4, weight_decay=1e-4`
  (`train_transformer_eod.py:392-393`); warm-start cadence multiplies lr by
  0.3 (`train_transformer_eod.py:444`) — research option, not production
  (production daily retrain is full refit, equal-weight history).

## 11. Seed list / 7-seed ensemble

- **Production: `seeds or list(range(7))` — seeds 0–6 — hardcoded in
  `mode_daily_retrain`, `train_transformer_eod.py:544`** (comment cites the
  A8/A9 2026-07-24 adoption). One checkpoint per seed; inference averages
  all of them.
- The per-preset `seeds` field (B: 5) is the legacy *screening* count used
  only when `walkforward()` is called without an explicit list
  (`train_transformer_eod.py:403`). Research runs pass seeds explicitly
  (e.g. the A8 7-seed panels and the disjoint-seed battery 10–16 via
  `research/gpu_research_scheduler.py`).
- Reproducibility: `torch.manual_seed(seed)` + `np.random.seed(seed)` in
  `fit_one` (`train_transformer_eod.py:229-230`).

## 12. Checkpoint save / load

- Directory: `CKPT_DIR = checkpoints/transformer_eod/`
  (`train_transformer_eod.py:35`) — gitignored (`.gitignore:18`).
- Save: `daily_seed{s}.pt` per seed (state_dict + cfg + feature_set/target/
  horizon metadata), `train_transformer_eod.py:550-557`, plus
  `daily_manifest.json` (`train_transformer_eod.py:575`).
- Load: `inference_transformer_eod.py:38-55` (`load_ensemble()` globs
  `daily_seed*.pt`, rebuilds each net from the saved cfg via
  `build_net_from_ck`, `inference_transformer_eod.py:58-61`); the D2
  distillation study read the same files read-only
  (`research/d2_distill_oos.py`).

## 13. blend50 / d12-vs-transformer ensemble weights

- **Paper books:** `research/paper_trading.py:91` and `:124` —
  `0.5 * z_tf + 0.5 * z_mom` (both literals; there is no config knob).
  Strategies tracked: `STRATS = ("d12", "tf", "blend50", "blend50_band10")`
  (`research/paper_trading.py:39`); no-trade band `BAND = 0.10`
  (`research/paper_trading.py:41`).
- **Decision book:** `research/blended_decision_book.py:51-53` — same 50/50
  z-blend; the d12 leg is mom126 skip-5 computed inline as
  `c[-6] / c[-132] - 1.0` (`research/blended_decision_book.py:48-49`).
- Book construction caps: `NAME_CAP = 0.10`, `SECTOR_CAP = 0.20`,
  `CAP_TOL = 1e-6` in `research/transformer_portfolio.py:22-24`
  (`cap_weights` at `:52`). The D7b variant (band15 + cap7.5) is **not** a
  default anywhere — v9 built it by patching `cap_weights` arguments at
  call time (`research/queue_v9_lib.py`, `caps()` contextmanager).

## 14. Production defaults vs research overrides

**Production path** (`--mode daily-retrain` → inference → blended book):

| Parameter | Value | Source of truth |
|---|---|---|
| preset | B (h64/2L/4H/ff128/do0.2/seq60) | `PRESETS["B"]` + CLI default `--preset B` |
| feature_set | close_only (10 features) | `mode_daily_retrain` / CLI default |
| target / horizon | tgt_rank_20 / 20d | `mode_daily_retrain` / CLI default |
| seeds | 0–6 (7-seed) | hardcoded `train_transformer_eod.py:544` |
| lr / wd / batch | 3e-4 / 1e-4 / 1024 | `fit_one` defaults |
| history weighting | equal (recency=None) | `mode_daily_retrain` default |
| book | blend50 + band10, cap10/sector20 | `paper_trading.py`, `transformer_portfolio.py` |

**Research overrides** (never touch the production path):

- `research/gpu_research_scheduler.py` — experiment configs override on top
  of `preset_base="B"` via an `overrides` dict; `_RUN_DEFAULTS` pins
  `weight_decay=1e-4, loss="mse", holding=20, refit_every=126,
  oos_start="2023-01-01", horizon=20` for dedup keying.
- `walkforward()` kwargs (`loss`, `cadence`, `recency`, seeds, oos window) —
  research-only entry point; daily retrain does not use it.
- v10 flag keys (`cs_attn`, `mt_aux`, `quantile`, `tcn`, `aug_noise`,
  `aug_datedrop`) — exist in `fit_one`/`walkforward`, absent from all
  presets, champion path verified bit-identical with flags off. All
  associated research lines are CLOSED (see RESEARCH_LINES_CLOSED.md).
- `inference_transformer_eod.py` CLI `--top-frac 0.2 --band 0.05` shape the
  standalone tf target book only; the production blend book is built by
  `research/blended_decision_book.py` with its own band10 logic.

## Search commands used

ripgrep (what I actually ran, via the repo tooling):

```
rg -n "class \w+|PRESET|def build_net" model.py
rg -n "PRESETS|preset|seq_len|feature_set|lr|weight_decay|batch|epoch|seed|d_model|nhead|dropout|dim_feedforward|hidden" train_transformer_eod.py
rg -n "CKPT_DIR|checkpoints" train_transformer_eod.py
rg -n "def build_dataset|FEATURE|close_only|tgt_rank|horizons" dataset_transformer_eod.py
rg -n "NAME_CAP|SECTOR_CAP|CAP_TOL|def cap_weights" research/transformer_portfolio.py
rg -n "BAND|STRATEGIES|blend|mom126|0\.5" research/paper_trading.py
```

Windows findstr equivalents:

```
findstr /n /r "class PRESET build_net" model.py
findstr /n "PRESETS seq_len feature_set weight_decay batch epoch seed dropout hidden" train_transformer_eod.py
findstr /n "CKPT_DIR checkpoints" train_transformer_eod.py
findstr /n "build_dataset FEATURE close_only tgt_rank horizons" dataset_transformer_eod.py
findstr /n "NAME_CAP SECTOR_CAP CAP_TOL cap_weights" research\transformer_portfolio.py
findstr /n "BAND blend mom126" research\paper_trading.py
```

Plus full reads of `model.py`, `inference_transformer_eod.py`,
`research/blended_decision_book.py`, and targeted reads of
`dataset_transformer_eod.py:150-167` and `train_transformer_eod.py:580-593`.

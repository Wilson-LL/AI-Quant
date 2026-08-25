# Legacy LSTM stack (superseded, frozen)

The pre-transformer implementation (moved here from the repo root,
2026-08-25 cleanup). **Not used by production** — the live system is the
LSTM-CondTransformer pipeline driven by `daily_ops.bat` (root
`model.py` / `dataset_transformer_eod.py` / `train_transformer_eod.py` /
`inference_transformer_eod.py`). Retained for historical /
reproducibility context of the original D1-era experiments
(see `docs/archive/d1_1/`).

Files: `dataset.py`, `train.py`, `inference.py`, `test_dataset.py`
(hand-rolled runner, not unittest), `generate_stocks_json.py`.

Runtime assumptions (unchanged from its original root-level life —
deliberately NOT modernized):

- Must be run **from the repository root** with this directory on the
  path, e.g.
  `python -c "import sys; sys.path.insert(0,'research/legacy_lstm'); import train"`
  or by temporarily copying the files back to root. The scripts use
  CWD-relative paths (`./checkpoints/stocks.json`) and always did.
- `train.py` / `inference.py` import root `model.py` (still at root —
  shared with the production stack) and sibling `dataset.py`.
- `generate_stocks_json.py` writes `./checkpoints/stocks.json`, which
  only this legacy stack reads.
- Dependencies include `tqdm` (legacy-only; the production pipeline
  does not use it).

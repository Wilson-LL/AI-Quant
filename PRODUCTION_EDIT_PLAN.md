# Production Edit Plan

Status as of 2026-07-22 (RTX 4060 Ti 20h sprint):

**No production file edits planned or made.** `model.py`, `train.py`, `inference.py`,
`dataset.py` remain untouched. The transformer EOD daily-retrain system is built as
new, additive files (`dataset_transformer_eod.py`, `train_transformer_eod.py`,
`inference_transformer_eod.py`, `research/transformer_portfolio.py`,
`research/refresh_data.py`) that **import** `model.LSTM_CondTransformer` and reuse
train.py's training patterns (AdamW, gradient clipping, early stopping, checkpoint
format) without modifying them.

Rationale: zero risk to the validated D1.1 lineage and the original barrier-target
pipeline; rollback is file deletion.

If a production edit becomes necessary, this file must first be updated with:
reason, expected benefit, rollback, and tests — per sprint rules.

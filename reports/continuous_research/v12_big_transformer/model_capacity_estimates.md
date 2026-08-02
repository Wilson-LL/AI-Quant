# v12 model capacity estimates

Exact parameter counts from instantiating `model.LSTM_CondTransformer` (input_dim=10, close_only). Memory/runtime columns are analytic estimates (AdamW fp32 states, AMP fp16 activations at micro-batch 256, 6.0-12.0 effective TFLOPS on the RTX 4060 Ti 16 GB); the v12 Phase-1 feasibility smoke measures them for real before any long run.

| band | params | ckpt fp32 | ckpt fp16 | opt (GB) | act (GB) | train VRAM est | fits 16GB | min/epoch est |
|---|---|---|---|---|---|---|---|---|
| S_baseline | 113,857 | 0.4 MB | 0.2 MB | 0.0 | 0.06 | 0.96 GB | yes | 0-0 |
| M1_h128L4 | 960,577 | 3.7 MB | 1.8 MB | 0.01 | 0.25 | 1.17 GB | yes | 0-0 |
| M2_h256L4 | 3,788,865 | 14.5 MB | 7.2 MB | 0.03 | 0.5 | 1.46 GB | yes | 1-1 |
| M3_h256L6 | 5,368,385 | 20.5 MB | 10.2 MB | 0.04 | 0.71 | 1.69 GB | yes | 1-1 |
| L1_h512L8 | 27,658,305 | 105.5 MB | 52.8 MB | 0.21 | 1.82 | 3.13 GB | yes | 4-8 |
| L2_h512L12 | 40,267,841 | 153.6 MB | 76.8 MB | 0.3 | 2.64 | 4.14 GB | yes | 6-11 |
| L3_h768L8 | 62,131,265 | 237.0 MB | 118.5 MB | 0.46 | 2.73 | 4.55 GB | yes | 9-17 |
| XL1_h1024L16 | 211,136,577 | 805.4 MB | 402.7 MB | 1.57 | 6.92 | 10.96 GB | yes | 29-58 |
| XL2_h1024L24 | 311,906,369 | 1189.8 MB | 594.9 MB | 2.32 | 10.2 | 15.75 GB | NO | 43-86 |
| XL3_h768L24 | 175,537,217 | 669.6 MB | 334.8 MB | 1.31 | 7.65 | 11.16 GB | yes | 24-49 |
| XL4_h1536L12 | 361,451,585 | 1378.8 MB | 689.4 MB | 2.69 | 7.91 | 14.2 GB | NO | 50-100 |

1 GiB fp32 checkpoint = 268,435,456 parameters (2^30 / 4).

Module split of the baseline (exact): lstm 19,456, input_proj 4,160, pos_emb 3,840, cross_attn 17,344, encoder 66,944, head 2,113

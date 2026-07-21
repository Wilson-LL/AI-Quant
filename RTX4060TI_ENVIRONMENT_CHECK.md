# RTX 4060 Ti Environment Check

Date: 2026-07-22 00:45 (local) · Branch: `research/transformer-4060ti-daily-retrain-20h`

## Verdict

**PASS — CUDA is available and the machine is suitable for the 12h daily workflow.**
Training must use the repo venv (`.venv\Scripts\python.exe`); the system Python has no torch.

## Hardware / software inventory

| item | value |
|---|---|
| Python (venv) | 3.12.3 (MSC v.1938, 64-bit) |
| PyTorch | 2.13.0.dev20260522+cu132 (nightly, CUDA 13.2 build) |
| CUDA available | **True** |
| GPU | NVIDIA GeForce RTX 4060 Ti, compute capability 8.9 (Ada) |
| VRAM | 16 GB (16,380 MiB per nvidia-smi; ~3.4 GiB already used by desktop apps) |
| Driver | 591.86, CUDA 13.1 runtime per nvidia-smi, WDDM mode |
| CPU count | 16 logical cores |
| RAM | 34.1 GB |
| device_count | 1 |

## AMP / mixed precision

Benchmarked with the real `LSTM_CondTransformer` (input_dim=10, hidden 64, 2+2 layers,
seq_len 40 — Preset A scale, 145,857 params), batch 512, 100 optimizer steps:

| mode | 100 steps | est. epoch @200k samples |
|---|---|---|
| fp32 | 45.4 s | ~178 s |
| **AMP (autocast+GradScaler)** | **4.27 s** | **~17 s** |

- AMP speedup: **10.6×**, losses finite, GradScaler stable → **AMP is usable and mandatory** (fp32 is anomalously slow on this nightly build).
- Peak VRAM during training: **317 MB** — enormous headroom; batch 512–2048 is safe even with the desktop using ~3.4 GB.
- Inference on a full 119-stock cross-section (one sequence each): **~6 ms**.

## 12h daily budget assessment

At ~17 s/epoch on the full panel, a 20-epoch, 5-seed retrain is **~30 min**; full-universe
inference is milliseconds; data refresh (twstock, 1 request/stock for the current month)
is ~2–10 min network-bound. The complete daily collect → retrain → inference → decision
book cycle fits in **well under 1 hour**, far inside the 12 h budget. Suitability: **YES**.

## Caveats

1. **Exit-code quirk**: after heavy AMP training, the nightly torch build sometimes exits
   with a nonzero code (observed: 9) during CUDA teardown *after* all work completed.
   Pipeline scripts therefore persist results to files and success is judged by output
   artifacts, not exit codes.
2. Torch is a **nightly dev build** — acceptable for research; pin a stable release
   (torch ≥ 2.13 stable, cu13x) before any production hardening.
3. GPU is shared with an active desktop session (games/browsers observed on it).
   Training at these model sizes is unaffected (VRAM headroom ~12 GB), but wall-clock
   timings can jitter.
4. Device priority honored: CUDA for all real training; CPU only for smoke tests.

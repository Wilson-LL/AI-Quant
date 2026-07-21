# Run Status — Transformer RTX 4060 Ti 20h Sprint

Branch: `research/transformer-4060ti-daily-retrain-20h` (created from
`research/d1-1-momentum-prototype-clean` HEAD d41fdd4 — see note below).

> **Branch provenance note**: the prompt said this branch was created from
> `research/transformer-gpu-confirmation-20h`, but that branch (and the other
> transformer sprint branches/artifacts it references) does not exist in this
> repository. Only `main`, `list`, and `research/d1-1-momentum-prototype-clean`
> exist. The sprint branch was created from the D1.1 clean milestone HEAD, and
> the transformer EOD pipeline is being built fresh this sprint, reusing
> model.py / train.py / research/ as the prompt requires.

---

## 2026-07-22 00:42 — Sprint start (elapsed 0:00)

- Current experiment: G0 CUDA readiness.
- Device: RTX 4060 Ti confirmed, CUDA True, torch 2.13 nightly cu132, venv Python 3.12.3.
- AMP benchmark: 10.6× speedup vs fp32; epoch @200k samples ≈ 17 s; inference 119 stocks ≈ 6 ms; peak VRAM 317 MB.
- Files changed: RTX4060TI_ENVIRONMENT_CHECK.md (new), this file (new).
- Commands run: nvidia-smi, torch CUDA check, GPU smoke benchmark.
- Blockers: none. Prior-sprint artifacts referenced by the prompt don't exist; documented and proceeding.
- Next: write RTX4060TI_SPRINT_PLAN.md, then build dataset_transformer_eod.py.
- Commits: none yet.

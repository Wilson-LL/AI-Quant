"""v12 Phase 5 — deep GPU inference / uncertainty / rank-stability research.

Research-only; reads the production ensemble READ-ONLY and writes only under
the v12 research paths. Never generates orders; short-side lists are
review-only diagnostics.

Part A (GPU, current production 7-seed ensemble, latest date):
  - deterministic 7-seed baseline (production replica)
  - MC-dropout passes per seed (uncertainty beyond seed disagreement)
  - feature-space noise perturbations (robustness)
  - truncated-history scoring (as-of t-1/t-3/t-5: missing-days robustness)
  -> per-stock: mean/std by source, top/bottom-quintile inclusion
     probability, stability classification, high-confidence and unstable
     lists (long + short-diagnostic).

Part B (CPU, frozen A8 panels CH/BR — the actual value gate):
  - does per-date seed disagreement (score_std) predict realized rank error?
  - would excluding the highest-uncertainty tercile from the top quintile
    have improved the book? (EVIDENCE ONLY: confidence filtering is a
    CLOSED line; no adoption is proposed here.)

Usage:
  python research/deep_inference_v12.py [--mc-samples 30] [--noise-draws 20]
      [--max-inference-hours 5]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V12_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v12_big_transformer")
DEEP_DIR = os.path.join(ROOT, "reports", "transformer_gpu",
                        "v12_big_transformer")


def _score(nets, Xg, idx, train_mode=False, noise=None, gen=None):
    import torch
    out = []
    for net in nets:
        net.train() if train_mode else net.eval()
        with torch.no_grad():
            xb = Xg[idx]
            if noise is not None:
                xb = xb + noise * torch.randn(xb.shape, device=xb.device,
                                              generator=gen)
            with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                out.append(net(xb).float())
        net.eval()
    import torch as _t
    return _t.stack(out)


def part_a(mc_samples, noise_draws, budget_s):
    import torch
    from dataset_transformer_eod import build_dataset
    from inference_transformer_eod import load_ensemble
    import train_transformer_eod as T
    T.require_cuda()
    t0 = time.time()
    nets, cfg, manifest = load_ensemble()          # READ-ONLY
    data = build_dataset(manifest["feature_set"], seq_len=cfg["seq_len"],
                         horizons=(manifest["horizon"],))
    Xg = T.to_gpu(data)
    torch.cuda.reset_peak_memory_stats()
    dr = np.asarray(data["date_rank"])
    rmax = int(dr.max())
    stocks_all = np.asarray(data["stocks"])
    gen = torch.Generator(device=Xg.device.type)
    gen.manual_seed(123)

    def _idx_at(rank):
        return np.nonzero(dr == rank)[0]

    idx = _idx_at(rmax)
    idx_t = torch.as_tensor(idx, device=Xg.device)
    stocks = stocks_all[np.asarray(data["stock_idx"])[idx]]
    n = len(idx)
    asof = str(np.asarray(data["dates"])[rmax])[:10]
    feat_std = float(Xg.std())

    passes = {}          # source -> (n_passes, n_stocks) score array
    # 1. deterministic per-seed (production baseline)
    det = _score(nets, Xg, idx_t).cpu().numpy()
    passes["seed"] = det
    # 2. MC dropout
    mc = []
    for _ in range(mc_samples):
        if time.time() - t0 > budget_s * 0.5:
            break
        mc.append(_score(nets, Xg, idx_t, train_mode=True).mean(0)
                  .cpu().numpy())
    passes["mc_dropout"] = np.stack(mc)
    # 3. feature noise
    nz = []
    for sig in (0.005, 0.01, 0.02):
        for _ in range(noise_draws):
            if time.time() - t0 > budget_s * 0.8:
                break
            nz.append(_score(nets, Xg, idx_t, noise=sig * feat_std, gen=gen)
                      .mean(0).cpu().numpy())
    passes["feat_noise"] = np.stack(nz)
    # 4. truncated history (score as-of earlier dates; stability of ranks)
    trunc_ranks = {}
    base_rank = pd.Series(det.mean(0), index=stocks).rank(ascending=False)
    for k in (1, 3, 5):
        i2 = _idx_at(rmax - k)
        s2 = stocks_all[np.asarray(data["stock_idx"])[i2]]
        sc2 = _score(nets, Xg, torch.as_tensor(i2, device=Xg.device)) \
            .mean(0).cpu().numpy()
        r2 = pd.Series(sc2, index=s2).rank(ascending=False)
        common = base_rank.index.intersection(r2.index)
        # inputs are already ranks -> pearson on ranks == spearman (no scipy)
        trunc_ranks[k] = float(base_rank[common].corr(r2[common]))

    # aggregate per stock
    allp = np.concatenate([passes["seed"], passes["mc_dropout"],
                           passes["feat_noise"]])
    ranks = pd.DataFrame(allp, columns=stocks).rank(axis=1, ascending=False)
    k5 = max(3, round(0.2 * n))
    topq = (ranks <= k5).mean(0)
    botq = (ranks > n - k5).mean(0)
    rows = pd.DataFrame({
        "stock": stocks, "asof": asof,
        "score_mean": allp.mean(0), "seed_std": passes["seed"].std(0),
        "mc_std": passes["mc_dropout"].std(0),
        "noise_std": passes["feat_noise"].std(0),
        "rank_det": base_rank.reindex(stocks).values,
        "rank_std_all_passes": ranks.std(0).reindex(stocks).values,
        "topq_prob": topq.reindex(stocks).values,
        "botq_prob": botq.reindex(stocks).values})
    comb = rows[["seed_std", "mc_std", "noise_std"]].rank(pct=True).mean(1)
    rows["stability_class"] = np.select(
        [(rows["topq_prob"] >= 0.9) & (comb <= 0.5),
         (rows["botq_prob"] >= 0.9) & (comb <= 0.5),
         ((rows["topq_prob"] > 0.2) & (rows["topq_prob"] < 0.8))
         | (comb >= 0.85)],
        ["HIGH_CONF_LONG", "HIGH_CONF_SHORT_DIAGNOSTIC", "UNSTABLE"],
        default="MID")
    os.makedirs(DEEP_DIR, exist_ok=True)
    panel_p = os.path.join(DEEP_DIR, f"deep_scores_{asof}.csv")
    rows.sort_values("rank_det").to_csv(panel_p, index=False)

    stats = {
        "asof": asof, "n_stocks": int(n),
        "n_passes": {k: int(v.shape[0]) for k, v in passes.items()},
        "trunc_rank_spearman": {f"t-{k}": round(v, 4)
                                for k, v in trunc_ranks.items()},
        "corr_seedstd_mcstd": round(float(rows["seed_std"].corr(rows["mc_std"])), 3),
        "corr_seedstd_noisestd": round(float(rows["seed_std"]
                                             .corr(rows["noise_std"])), 3),
        "n_high_conf_long": int((rows["stability_class"] == "HIGH_CONF_LONG").sum()),
        "n_high_conf_short_diag": int((rows["stability_class"]
                                       == "HIGH_CONF_SHORT_DIAGNOSTIC").sum()),
        "n_unstable": int((rows["stability_class"] == "UNSTABLE").sum()),
        "topq_mean_inclusion_of_det_book": round(float(
            rows.nsmallest(k5, "rank_det")["topq_prob"].mean()), 3),
        "runtime_s": round(time.time() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "panel": panel_p,
    }
    return rows, stats


def part_b():
    """Historical uncertainty-value gate on the frozen panels (CPU)."""
    from queue_v9_lib import get_merged
    import transformer_portfolio as tp
    out = {}
    for win in ("CH", "BR"):
        m = get_merged(win).copy()
        m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
        # 1. does seed std predict realized rank error?
        def _day(g):
            if len(g) < 30:
                return np.nan
            re = (g["score"].rank() - g["fwd_20"].rank()).abs()
            return g["score_std"].rank().corr(re.rank())  # spearman, no scipy
        cs = m.groupby("date", group_keys=False).apply(_day,
                                                       include_groups=False)
        # 2. book evidence: drop highest-std tercile from top quintile
        plain = tp.backtest_scores(m.assign(score=m["blend"]), holding=20,
                                   mode="long_short", no_trade_band=0.10)
        mf = m.copy()
        thr = mf.groupby("date")["score_std"].transform(
            lambda s: s.quantile(2 / 3))
        mf.loc[mf["score_std"] > thr, "blend"] = (
            mf.loc[mf["score_std"] > thr, "blend"] - 1e3)  # push out of topQ
        filt = tp.backtest_scores(mf.assign(score=mf["blend"]), holding=20,
                                  mode="long_short", no_trade_band=0.10)
        out[win] = {
            "corr_std_rank_error_mean": round(float(cs.mean()), 4),
            "corr_std_rank_error_frac_pos": round(float((cs > 0).mean()), 3),
            "book_plain_net60": plain["net60"]["sharpe"],
            "book_dropstd_net60": filt["net60"]["sharpe"],
            "book_plain_dd": plain["net60"]["max_dd"],
            "book_dropstd_dd": filt["net60"]["max_dd"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--noise-draws", type=int, default=20)
    ap.add_argument("--max-inference-hours", type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(V12_DIR, exist_ok=True)
    rows, stats = part_a(a.mc_samples, a.noise_draws,
                         a.max_inference_hours * 3600)
    hist = part_b()

    md = ["# v12 deep inference report — " + stats["asof"], "",
          f"Production 7-seed ensemble, read-only. {stats['n_passes']} "
          f"passes, runtime {stats['runtime_s']}s (budget "
          f"{a.max_inference_hours}h), peak VRAM {stats['peak_vram_gb']} GB.",
          "",
          "## Rank stability (today)", "",
          f"- truncated-history rank correlation: "
          + ", ".join(f"{k} {v}" for k, v in
                      stats["trunc_rank_spearman"].items()),
          f"- deterministic top-quintile members' mean inclusion probability "
          f"across all passes: {stats['topq_mean_inclusion_of_det_book']}",
          f"- uncertainty source agreement: corr(seed_std, mc_std) = "
          f"{stats['corr_seedstd_mcstd']}, corr(seed_std, noise_std) = "
          f"{stats['corr_seedstd_noisestd']}",
          f"- classes: {stats['n_high_conf_long']} HIGH_CONF_LONG, "
          f"{stats['n_high_conf_short_diag']} HIGH_CONF_SHORT_DIAGNOSTIC "
          f"(review-only), {stats['n_unstable']} UNSTABLE",
          f"- full per-stock panel: {stats['panel']}", "",
          "## Historical uncertainty value (frozen panels — the gate)", ""]
    for win, h in hist.items():
        md += [f"### {win}",
               f"- corr(seed_std, realized rank error): "
               f"{h['corr_std_rank_error_mean']} (positive on "
               f"{h['corr_std_rank_error_frac_pos']:.0%} of dates)",
               f"- blend book plain {h['book_plain_net60']} "
               f"(DD {h['book_plain_dd']:.1%}) vs drop-high-std "
               f"{h['book_dropstd_net60']} (DD {h['book_dropstd_dd']:.1%})",
               ""]
    md += ["## Notes", "",
           "- Multi-snapshot/multi-refit ensembles need checkpoints "
           "collected over time; production daily retrain overwrites — "
           "if ever wanted, a snapshot-retention policy would be a "
           "separate (operational) proposal.",
           "- Confidence filtering remains a CLOSED line; the book "
           "comparison above is evidence about uncertainty value, not an "
           "adoption proposal.",
           "- Review-only; no orders; short list is diagnostic."]
    with open(os.path.join(V12_DIR, "queue_v12_deep_inference_report.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps({**stats, "historical": hist}, indent=2))
    print("[deep-inference] -> queue_v12_deep_inference_report.md")


if __name__ == "__main__":
    main()

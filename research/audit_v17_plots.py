"""Dependency-free SVG line plots for the v17 research curves (matplotlib
is not installed in the production venv and adding it is out of scope).

  python research/audit_v17_plots.py reports/model_audit/v17/p4/b0_long_epoch_curves.csv \
         reports/model_audit/v17/p4/plots  --marks 3,100

Writes one SVG per metric: individual (refit, seed) curves in light
strokes, the aggregate mean in bold, vertical marks at the given epochs.
"""

import os
import sys

import pandas as pd

W, H, PAD = 900, 420, 55


def svg_lines(series, title, ylabel, marks, path, xmax):
    ys = [v for s in series.values() for v in s.values()] or [0, 1]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    def X(e):
        return PAD + (e - 1) / max(xmax - 1, 1) * (W - 2 * PAD)
    def Y(v):
        return H - PAD - (v - lo) / (hi - lo) * (H - 2 * PAD)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2}" y="20" text-anchor="middle" font-size="15">{title}</text>',
           f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#333"/>',
           f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#333"/>',
           f'<text x="{W/2}" y="{H-12}" text-anchor="middle">epoch</text>',
           f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">{ylabel}</text>']
    for e in (1, 10, 20, 30, 50, 75, 100):
        if e <= xmax:
            out.append(f'<text x="{X(e)}" y="{H-PAD+14}" text-anchor="middle" font-size="10">{e}</text>')
    for v in (lo, (lo + hi) / 2, hi):
        out.append(f'<text x="{PAD-4}" y="{Y(v)+4}" text-anchor="end" font-size="10">{v:.3f}</text>')
    for e, label, col in marks:
        if e <= xmax:
            out.append(f'<line x1="{X(e)}" y1="{PAD}" x2="{X(e)}" y2="{H-PAD}" stroke="{col}" stroke-dasharray="4 3"/>'
                       f'<text x="{X(e)+3}" y="{PAD+12}" fill="{col}" font-size="10">{label}</text>')
    for name, s in series.items():
        bold = name == "mean"
        pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in sorted(s.items()))
        out.append(f'<polyline fill="none" stroke="{"#c0392b" if bold else "#7f8c8d"}" '
                   f'stroke-width="{2.5 if bold else 0.8}" stroke-opacity="{1 if bold else 0.55}" points="{pts}"/>')
    out.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main(csv_p, out_dir, marks_arg="3,100", prod_epochs=None):
    os.makedirs(out_dir, exist_ok=True)
    c = pd.read_csv(csv_p)
    xmax = int(c["epoch"].max())
    marks = [(int(m), f"epoch {m}", "#2980b9") for m in marks_arg.split(",") if m]
    if prod_epochs:
        marks.append((int(round(float(prod_epochs))), f"prod-selected ~{prod_epochs}", "#27ae60"))
    for metric, ylabel in (("train_loss", "train loss"), ("val_ic", "validation rank IC"),
                           ("oos_ic", "future OOS rank IC (diagnostic)"), ("oos_pred_std", "prediction dispersion"),
                           ("val_loss", "validation MSE"), ("grad_norm_mean", "mean grad norm"), ("weight_norm", "weight norm")):
        if metric not in c.columns:
            continue
        series = {}
        for (rd, sd), g in c.groupby(["refit_date", "seed"]):
            series[f"{rd}_s{sd}"] = dict(zip(g["epoch"], g[metric]))
        series["mean"] = c.groupby("epoch")[metric].mean().to_dict()
        svg_lines(series, f"{metric} vs epoch (thin = refit/seed, bold = mean)", ylabel, marks,
                  os.path.join(out_dir, f"{metric}.svg"), xmax)
    print("plots written to", out_dir)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else "3,100", a[3] if len(a) > 3 else None)

"""Numerical check of the label-shift approximation of anchor trust (Proposition 1 of the JAS manuscript).

For logistic regression the gradient is g = mean_i x_i (p_i - y_i). If an institution differs from the anchor only in
its outcome base rate, its residuals shift by about -Delta_k = -(ybar_k - ybar_0) on every record, so its gradient is
about g_k = g_0 - Delta_k m, with m the mean feature vector (the intercept coordinate of m is 1). Then
    cos(g_k, g_0) = (1 - b d_k) / sqrt(1 - 2 b d_k + d_k^2),   b = g_0'm / (|g_0| |m|),   d_k = Delta_k |m| / |g_0|,
which falls monotonically in |d_k| when b = 0. Version 'intercept' uses m = e_b (the shift only in the intercept).
This script runs clean FedAvg on the ACS state federation (20 seeds, 30 rounds) and, at rounds 1, 5, 10, 20 and 30,
compares the measured cosine between each state's local gradient and the anchor gradient (the reference of the anchor
rule) with both predictions, computed from the data's base rates only. It also reports the cosine after removing the
base-rate direction (each gradient minus its intercept coordinate times its own mean feature vector).

Run from the project root:  TF_OUT=results/revision_jas python code/jas_mechanism.py
Output: <TF_OUT>/mechanism_check.json, <TF_OUT>/mechanism_scatter.dat and <TF_OUT>/mechanism_points.dat
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402


def predicted(g0, delta, m=None):
    """cos(g0 - delta m, g0). m = e_b (intercept only) by default, or the mean feature vector."""
    if m is None:
        m = np.zeros(len(g0)); m[-1] = 1.
    G, M = np.linalg.norm(g0), np.linalg.norm(m)
    b, d = g0 @ m / (G * M), delta * M / G
    return (1 - b * d) / np.sqrt(1 - 2 * b * d + d ** 2)


assert abs(predicted(np.array([1., 0.]), 0.) - 1) < 1e-12  # no gap, full trust
assert abs(predicted(np.array([1., 0.]), 1.) - 2 ** -.5) < 1e-12  # g_k = (1, -1): cos 1/sqrt(2)
assert abs(predicted(np.array([3., 1.]), 2., np.array([1., 2.])) - np.cos(np.arctan2(-3, 1) - np.arctan2(1, 3))) < 1e-12


def spearman(x, y):
    rx_, ry_ = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx_, ry_)[0, 1])


def main():
    rows, CHECK = [], {1, 5, 10, 20, 30}
    for seed in rx.SEEDS:
        cl, (ax, ay, _) = rx.acs(seed)
        w = np.zeros(cl[0]['xtr'].shape[1])
        for t in range(1, 31):
            if t in CHECK:
                g0 = rx.grad_model(w, ax, ay)
                for k, c in enumerate(cl):
                    gk = rx.grad_model(w, c['xtr'], c['ytr'])
                    gap = float(c['ytr'].mean() - ay.mean())
                    rows.append(dict(seed=seed, round=t, state=rx.ACS_CLIENTS[k], gap=gap, measured=rx.cos(gk, g0),
                                     pred=float(predicted(g0, gap)), pred_mean=float(predicted(g0, gap, ax.mean(0))),
                                     centered=rx.cos(gk - gk[-1] * c['xtr'].mean(0), g0 - g0[-1] * ax.mean(0)),
                                     resid_k=float(gk[-1]), resid_0=float(g0[-1])))
            U = np.array([rx.train_model(w, c['xtr'], c['ytr']) - w for c in cl])
            w = w + U.mean(0)
    col = lambda c: np.array([r[c] for r in rows])  # noqa: E731
    m, p, pm, cc, gap = col('measured'), col('pred'), col('pred_mean'), col('centered'), col('gap')
    dr = col('resid_k') - col('resid_0')
    out = dict(n=len(rows),
               intercept_only=dict(corr=float(np.corrcoef(m, p)[0, 1]), spearman=spearman(m, p), mae=float(np.abs(m - p).mean())),
               mean_feature=dict(corr=float(np.corrcoef(m, pm)[0, 1]), spearman=spearman(m, pm), mae=float(np.abs(m - pm).mean())),
               residual_shift=dict(slope_on_gap=float(np.polyfit(gap, dr, 1)[0]), corr=float(np.corrcoef(gap, dr)[0, 1])),
               mean_measured=float(m.mean()), mean_centered=float(cc.mean()),
               corr_gap_measured=float(np.corrcoef(np.abs(gap), m)[0, 1]), corr_gap_centered=float(np.corrcoef(np.abs(gap), cc)[0, 1]))
    out['by_round'] = {}
    for t in sorted(CHECK):
        s = col('round') == t
        out['by_round'][t] = dict(corr_mean=float(np.corrcoef(m[s], pm[s])[0, 1]), mae_mean=float(np.abs(m[s] - pm[s]).mean()),
                                  mean_measured=float(m[s].mean()), mean_pred_mean=float(pm[s].mean()))
    last = [r for r in rows if r['round'] == 30]
    out['state_round30'] = {s: {c: float(np.mean([r[c] for r in last if r['state'] == s]))
                                for c in ('gap', 'measured', 'pred', 'pred_mean', 'centered')} for s in rx.ACS_CLIENTS}
    st = out['state_round30']
    out['state_rank_agreement'] = dict(pred=spearman([v['measured'] for v in st.values()], [v['pred'] for v in st.values()]),
                                       pred_mean=spearman([v['measured'] for v in st.values()], [v['pred_mean'] for v in st.values()]))
    rx.OUT.mkdir(parents=True, exist_ok=True)
    (rx.OUT / 'mechanism_check.json').write_text(json.dumps(out, indent=1))
    with open(rx.OUT / 'mechanism_scatter.dat', 'w') as fh:
        fh.write('pred predmean measured round\n')
        for r in rows:
            fh.write(f"{r['pred']:.4f} {r['pred_mean']:.4f} {r['measured']:.4f} {r['round']}\n")
    # per-point file with state index (states sorted by employment rate, 1 = lowest) for Figure 2 of the manuscript
    srt = sorted(rx.ACS_CLIENTS, key=lambda s: st[s]['gap'])
    with open(rx.OUT / 'mechanism_points.dat', 'w') as fh:
        fh.write('state gap predmean measured centered round\n')
        for r in rows:
            fh.write(f"{srt.index(r['state']) + 1} {r['gap']:.4f} {r['pred_mean']:.4f} {r['measured']:.4f} "
                     f"{r['centered']:.4f} {r['round']}\n")
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

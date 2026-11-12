"""Tables and numbers added for the JAS revision. Run after revision_analysis.py (both with TF_RES set):

    TF_RES=results/revision_jas python code/revision_analysis.py
    TF_RES=results/revision_jas python code/jas_analysis.py

Reads the run files in TF_RES and writes LaTeX rows to TF_RES/tables/ and every quoted number to
TF_RES/jas_numbers.json. Unit of analysis: the seed (rotated attackers averaged within a seed first).
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('TF_RES', ROOT / 'results' / 'revision_jas'))
TAB = RES / 'tables'
ST = ['wv', 'me', 'nm', 'ut', 'nd', 'ms']
HO = ['wy', 'ak', 'sd', 'de', 'ri', 'mt']
RULES = {'fedavg': 'FedAvg', 'median': 'Median', 'trimmed': 'Trimmed mean', 'multikrum': 'Multi-Krum',
         'fltrust': 'FLTrust', 'fedval': 'FedVal', 'anchor': 'Anchor rule', 'cpq': 'CPQ-R', 'cpqj': 'CPQ-B'}
BOOT = np.random.default_rng(0)
N = {}


def write(name, text):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(text, encoding='utf-8')


def num(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    if float(s) == 0:
        s = f'{0:.{d}f}'
    return s.replace('-', '$-$').replace('+', '$+$')


def boot_ci(v, B=10000):
    v = np.asarray(v, float)
    m = v[BOOT.integers(0, len(v), size=(B, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def t_ci(v):
    v = np.asarray(v, float); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def ci_txt(c, d=3):
    return f'[{num(c[0], d)}, {num(c[1], d)}]'


def load(name, K=6):
    d = pd.read_csv(RES / name)
    d['j'] = d['recall_support'] - d['fpr_support']
    if 'rel_recall_support' in d:
        d['rel_j'] = d['rel_recall_support'] - d['rel_fpr_support']
    if 'aud_recall_support' in d:
        d['aud_j'] = d['aud_recall_support'] - d['aud_fpr_support']
    return d


def by_seed(d, ds, fam, meth):
    x = d[(d.dataset == ds) & (d.family == fam) & (d.method == meth)]
    return x.groupby('seed').mean(numeric_only=True)


def weights(d, ds, meth, K=6, fam='none'):
    return by_seed(d, ds, fam, meth)[[f'w{k}' for k in range(K)]].mean().to_numpy()


def low(d, ds, meth, states, K=6):
    w = weights(d, ds, meth, K)
    return float(w.min()), states[int(w.argmin())]


def rate(state, weighted=False):
    dd = np.load(ROOT / 'data' / 'acs' / f'acs_{state}.npz'); y = dd['y']
    if not weighted:
        return float(y.mean())
    return float(np.average(y, weights=np.load(ROOT / 'data' / 'acs' / f'acs_{state}_w.npz')['w']))


def attacker_readmitted(d, ds, meth, fam='signflip'):
    x = d[(d.dataset == ds) & (d.family == fam) & (d.method == meth)].copy()
    x['ar'] = [r[f"ra{int(r['attackers'])}"] for _, r in x.iterrows()]
    return x.groupby('seed')['ar'].mean()


def attacker_weight(d, ds, meth, fam):
    x = d[(d.dataset == ds) & (d.family == fam) & (d.method == meth)].copy()
    x['aw'] = [r[f"w{int(r['attackers'])}"] for _, r in x.iterrows()]
    return x.groupby('seed')['aw'].mean()


# --------------------------------------------------------------------------- Table 1 with survey weights
def states_table():
    lines, out = [], {}
    order = sorted(ST, key=rate) + ['vt']
    for s in order:
        dd = np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz'); y, g = dd['y'], dd['g']
        ww = np.load(ROOT / 'data' / 'acs' / f'acs_{s}_w.npz'); w, rw = ww['w'], ww['rw'].astype(float)

        def est(v, m=None):
            m = np.ones(len(v), bool) if m is None else m
            th = np.average(v[m], weights=w[m])
            reps = np.array([np.average(v[m], weights=rw[m, r]) for r in range(80)])
            return float(th), float(np.sqrt(4 / 80 * ((reps - th) ** 2).sum()))  # successive difference replication
        e, ese = est(y); p, pse = est(g.astype(float)); ed, edse = est(y, g == 1)
        out[s] = dict(n=int(len(y)), emp=float(y.mean()), emp_w=e, emp_w_se=ese, prev=float(g.mean()), prev_w=p,
                      prev_w_se=pse, emp_dis=float(y[g == 1].mean()), emp_dis_w=ed, emp_dis_w_se=edse)
        lab = s.upper() + (' (anchor)' if s == 'vt' else '')
        lines.append(f"{lab} & {len(y):,} & {y.mean():.3f} & {e:.3f} ({ese:.3f}) & {g.mean():.3f} & {p:.3f} ({pse:.3f}) & "
                     f"{y[g == 1].mean():.3f} \\\\")
    write('tab_states_w.tex', '\n'.join(lines) + '\n')
    lines = []
    for s in sorted(HO, key=rate):
        dd = np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz'); y, g = dd['y'], dd['g']
        out[s] = dict(n=int(len(y)), emp=float(y.mean()), prev=float(g.mean()), emp_dis=float(y[g == 1].mean()))
        lines.append(f"{s.upper()} & {len(y):,} & {y.mean():.3f} & {g.mean():.3f} & {y[g == 1].mean():.3f} \\\\")
    write('tab_states_heldout.tex', '\n'.join(lines) + '\n')
    N['states'] = out


# --------------------------------------------------------------------------- R1-02: exact permutation test, six states
def spearman_perm(gap, w):
    """One-sided exact permutation p-value for a negative Spearman correlation (n! orderings)."""
    rg = stats.rankdata(gap); rw = stats.rankdata(w)
    obs = np.corrcoef(rg, rw)[0, 1]
    perms = [np.corrcoef(rg, rw[list(p)])[0, 1] for p in itertools.permutations(range(len(w)))]
    return float(obs), float(np.mean(np.array(perms) <= obs + 1e-12))


def perm_tests(main, ds='acs', states=ST, key='perm'):
    gap = np.array([abs(rate(s) - rate('vt')) for s in states])
    out = {}
    for m in ['multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']:
        rho, p = spearman_perm(gap, weights(main, ds, m))
        out[m] = dict(rho=rho, p=p)
    N[key] = out


# --------------------------------------------------------------------------- R1-03: PUMA regression with clusters
def cr1(X, e, cl):
    n, k = X.shape; G = len(np.unique(cl)); XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[cl == g].T @ e[cl == g], X[cl == g].T @ e[cl == g]) for g in np.unique(cl))
    V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    return np.sqrt(np.diag(V))


def wild_cluster_p(y, X, cl, j):
    """Wild cluster bootstrap-t with the null imposed (Cameron, Gelbach and Miller 2008), Rademacher weights,
    all 2^G sign patterns enumerated (G = 6 states, 64 patterns)."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    t0 = b[j] / cr1(X, e, cl)[j]
    Xr = np.delete(X, j, 1); br = np.linalg.lstsq(Xr, y, rcond=None)[0]; er = y - Xr @ br
    groups = np.unique(cl); ts = []
    for signs in itertools.product([-1., 1.], repeat=len(groups)):
        v = np.array(signs)[np.searchsorted(groups, cl)]
        ys = Xr @ br + er * v
        bs = np.linalg.lstsq(X, ys, rcond=None)[0]
        ts.append(bs[j] / cr1(X, ys - X @ bs, cl)[j])
    return float(np.mean(np.abs(ts) >= abs(t0) - 1e-12))


def vif(Z):
    out = []
    for j in range(Z.shape[1]):
        X = np.column_stack([np.ones(len(Z)), np.delete(Z, j, 1)])
        r = Z[:, j] - X @ np.linalg.lstsq(X, Z[:, j], rcond=None)[0]
        out.append(float(Z[:, j].var() / r.var()))
    return out


def puma_regression():
    meta = json.loads((RES / 'puma_meta.json').read_text())
    K = len(meta); emp = np.array([m['employment'] for m in meta]); pv = np.array([m['prevalence'] for m in meta])
    cl = np.array([ST.index(m['state']) for m in meta])
    dist = np.abs(emp - rate('vt'))
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    D = np.column_stack([z(dist), z(pv)])
    FE = np.column_stack([(cl == g).astype(float) for g in range(1, 6)])
    Xa = np.column_stack([np.ones(K), D]); Xb = np.column_stack([np.ones(K), D, FE])
    within = D - np.array([D[cl == g].mean(0) for g in cl])
    N['puma_diag'] = dict(vif_pooled=vif(D), vif_within_state=vif(within), corr=float(np.corrcoef(dist, pv)[0, 1]),
                          corr_within=float(np.corrcoef(within[:, 0], within[:, 1])[0, 1]))
    pu = load('runs_puma.csv'); pu = pu[(pu.dataset == 'acs-puma') & (pu.family == 'none')]
    sets = [(m, pu, 'acs-puma', RULES[m]) for m in ['multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']]
    if (RES / 'runs_center_puma.csv').exists():
        pc = load('runs_center_puma.csv'); pc = pc[pc.family == 'none']
        sets += [(m, pc, 'acs-puma-center', RULES[m] + ', centered') for m in ['fltrust', 'anchor', 'cpqj']]
    lines, reg = [], {}
    for m, d, ds, lab in sets:
        y = K * d[(d.dataset == ds) & (d.method == m)].groupby('seed')[[f'w{k}' for k in range(K)]].mean().mean().to_numpy()
        row = {}
        for spec, X in [('pooled', Xa), ('fe', Xb)]:
            b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; se = cr1(X, e, cl)
            row[spec] = dict(dist=float(b[1]), prev=float(b[2]), se_dist=float(se[1]), se_prev=float(se[2]),
                             p_dist=wild_cluster_p(y, X, cl, 1), p_prev=wild_cluster_p(y, X, cl, 2),
                             r2=float(1 - e.var() / y.var()), resid_sd=float(e.std(ddof=X.shape[1])))
        yl = np.log(y + 0.1)  # log outcome check; 0.1 of a uniform share keeps excluded institutions finite
        bl = np.linalg.lstsq(Xb, yl, rcond=None)[0]; el = yl - Xb @ bl
        row['log_fe'] = dict(dist=float(bl[1]), prev=float(bl[2]), p_dist=wild_cluster_p(yl, Xb, cl, 1),
                             p_prev=wild_cluster_p(yl, Xb, cl, 2), se_dist=float(cr1(Xb, el, cl)[1]))
        reg[lab] = row
        a, f = row['pooled'], row['fe']
        lines.append(f"{lab} & {num(a['dist'], 2, True)} ({a['se_dist']:.2f}) & {a['p_dist']:.3f} & "
                     f"{num(a['prev'], 2, True)} ({a['se_prev']:.2f}) & {a['p_prev']:.3f} & "
                     f"{num(f['dist'], 2, True)} ({f['se_dist']:.2f}) & {f['p_dist']:.3f} & "
                     f"{num(f['prev'], 2, True)} ({f['se_prev']:.2f}) & {f['p_prev']:.3f} \\\\")
    write('tab_puma_reg2.tex', '\n'.join(lines) + '\n')
    lines = [f"{lab} & {num(r['log_fe']['dist'], 3, True)} & {r['log_fe']['p_dist']:.3f} & {num(r['log_fe']['prev'], 3, True)} & "
             f"{r['log_fe']['p_prev']:.3f} & {r['pooled']['r2']:.2f} & {r['fe']['r2']:.2f} & {r['fe']['resid_sd']:.3f} \\\\"
             for lab, r in reg.items()]
    write('tab_puma_reg_diag.tex', '\n'.join(lines) + '\n')
    N['puma_reg'] = reg


# --------------------------------------------------------------------------- R1-10: t-intervals next to percentile intervals
def t_intervals(main):
    lines, out = [], {}
    for fam, fl in [('none', 'Clean'), ('signflip', 'Sign-flip'), ('targeted', 'Targeted')]:
        a = by_seed(main, 'acs', fam, 'cpqj')
        for b in ['fedavg', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq']:
            dlt = (a['j'] - by_seed(main, 'acs', fam, b)['j']).dropna().to_numpy()
            pc, tc = boot_ci(dlt), t_ci(dlt)
            out[f'{fam}|{b}'] = dict(mean=float(dlt.mean()), pct=pc, t=tc)
            lines.append(f"{fl if b == 'fedavg' else ''} & {RULES[b]} & {num(dlt.mean(), 3, True)} & {ci_txt(pc)} & {ci_txt(tc)} \\\\")
    write('tab_tint.tex', '\n'.join(lines) + '\n')
    N['tint'] = out


# --------------------------------------------------------------------------- R1-13 and the estimand: calibration
def calibration(main):
    lines, out = [], {}
    rates = np.array([rate(s) for s in ST])
    for m in ['fedavg', 'median', 'trimmed', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']:
        x = by_seed(main, 'acs', 'none', m)
        has_w = 'w0' in x and x['w0'].notna().all()
        pred = float((weights(main, 'acs', m) - 1 / 6) @ rates) if has_w else np.nan
        out[m] = dict(f1=float(x.f1.mean()), j=float(x.j.mean()), brier=float(x.brier.mean()), logloss=float(x.logloss.mean()),
                      citl=float(x.citl.mean()), citl_support=float(x.citl_support.mean()), pred_shift=pred,
                      pos_rate=float(x.pos_rate.mean()))
        lines.append(f"{RULES[m]} & {x.f1.mean():.3f} & {x.j.mean():.3f} & {x.brier.mean():.4f} & {x.logloss.mean():.3f} & "
                     f"{num(x.citl.mean(), 3, True)} & {num(x.citl_support.mean(), 3, True)} & {num(pred, 3, True)} \\\\")
    write('tab_calib.tex', '\n'.join(lines) + '\n')
    ok = [v for v in out.values() if not np.isnan(v['pred_shift'])]
    N['calib'] = out
    N['calib_corr_pred_citl'] = float(np.corrcoef([v['pred_shift'] for v in ok], [v['citl'] for v in ok])[0, 1])


# --------------------------------------------------------------------------- R1-20, R1-21, centered cosine, pooled anchor
def fixes(main, anchor_runs):
    var = [('Standard cosine, Vermont anchor', main, 'acs'),
           ('Intercept left out of the cosine', load('runs_variants.csv'), 'acs-icpt'),
           ('Base-rate direction removed (centered)', load('runs_center.csv'), 'acs-center'),
           ('Anchor reweighted to the federation rate', load('runs_variants.csv'), 'acs-prior'),
           ('Pooled anchor', anchor_runs, 'acs-anchor-pooled')]
    lines, out = [], {}
    fa = by_seed(main, 'acs', 'none', 'fedavg')
    for lab, d, ds in var:
        cells, row = [], {}
        for m in ['fltrust', 'anchor', 'cpqj']:
            w, s = low(d, ds, m, ST); x = by_seed(d, ds, 'none', m)
            row[m] = dict(low=w, state=s, f1=float(x.f1.mean()), j=float(x.j.mean()),
                          citl=float(x.citl.mean()) if 'citl' in x and x.citl.notna().any() else None)
            sf = by_seed(d, ds, 'signflip', m)
            row[m]['sf_f1'] = float(sf.f1.mean()) if len(sf) else None
            cells += [f'{w:.3f}', f'{x.j.mean():.3f}']
        f1s = [row[m]['f1'] for m in row]; sfs = [row[m]['sf_f1'] for m in row if row[m]['sf_f1'] is not None]
        lines.append(f"{lab} & " + ' & '.join(cells) + f" & {min(f1s):.3f}--{max(f1s):.3f} & "
                     + (f"{min(sfs):.3f}--{max(sfs):.3f}" if sfs else '--') + ' \\\\')
        out[lab] = row
    lines.append('\\midrule')
    lines.append(f"FedAvg (reference) & 0.167 & {fa.j.mean():.3f} & 0.167 & {fa.j.mean():.3f} & 0.167 & {fa.j.mean():.3f} & "
                 f"{fa.f1.mean():.3f} & {by_seed(main, 'acs', 'signflip', 'fedavg').f1.mean():.3f} \\\\")
    write('tab_fix.tex', '\n'.join(lines) + '\n')
    N['fixes'] = out
    # centered cosine versus standard cosine, paired by seed (clean)
    c = load('runs_center.csv'); pair = {}
    for m in ['fltrust', 'anchor', 'cpq', 'cpqj']:
        a = by_seed(c, 'acs-center', 'none', m); b = by_seed(main, 'acs', 'none', m)
        lw = lambda x: x[[f'w{k}' for k in range(6)]].min(1)  # noqa: E731
        dl, dj, df1 = (lw(a) - lw(b)).to_numpy(), (a.j - b.j).to_numpy(), (a.f1 - b.f1).to_numpy()
        pair[m] = dict(low=dict(mean=float(dl.mean()), ci=boot_ci(dl)), j=dict(mean=float(dj.mean()), ci=boot_ci(dj)),
                       f1=dict(mean=float(df1.mean()), ci=boot_ci(df1)))
        dvf = (a.j - fa.j).to_numpy(); pair[m]['j_vs_fedavg'] = dict(mean=float(dvf.mean()), ci=boot_ci(dvf))
    N['center_vs_standard'] = pair
    if (RES / 'runs_center_anchor.csv').exists():
        ca = load('runs_center_anchor.csv'); rows, out2 = [], {}
        for a in ['pooled'] + sorted(ST, key=rate):
            cells = []
            for m in ['fltrust', 'anchor', 'cpq', 'cpqj']:
                w0, s0 = low(anchor_runs, f'acs-anchor-{a}', m, ST)
                w1, s1 = low(ca, f'acs-anchor-{a}-center', m, ST)
                out2[f'{a}|{m}'] = dict(standard=w0, centered=w1, state_std=s0, state_ctr=s1)
                cells.append(f'{w0:.3f} / {w1:.3f}')
            lab = 'Pooled' if a == 'pooled' else f'{a.upper()} ({rate(a):.3f})'
            rows.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
        w0 = {m: low(main, 'acs', m, ST)[0] for m in ['fltrust', 'anchor', 'cpq', 'cpqj']}
        w1 = {m: low(c, 'acs-center', m, ST)[0] for m in ['fltrust', 'anchor', 'cpq', 'cpqj']}
        rows.insert(0, f'VT ({rate("vt"):.3f}, main) & ' + ' & '.join(f'{w0[m]:.3f} / {w1[m]:.3f}' for m in w0) + ' \\\\')
        write('tab_center_anchor.tex', '\n'.join(rows) + '\n')
        N['center_anchor'] = out2


# --------------------------------------------------------------------------- R1-05a, R2-05: null federation; gate
def null_and_gate(main):
    nl = load('runs_null.csv'); out = {}
    for m in RULES:
        x = by_seed(nl, 'acs-iid', 'none', m)
        out[m] = dict(f1=float(x.f1.mean()), gate=float(x.gate_blocks.mean()),
                      low=float(x[[f'w{k}' for k in range(6)]].min(1).mean()) if x['w0'].notna().all() else None)
    c = nl[nl.method == 'cpqj']
    q = c[[f'q{k}' for k in range(6)]].sum(1); ra = c[[f'ra{k}' for k in range(6)]].sum(1)
    lt = c[[f'lt{k}' for k in range(6)]].sum(1)
    cr = nl[nl.method == 'cpq']; q2 = cr[[f'q{k}' for k in range(6)]].sum(1); ra2 = cr[[f'ra{k}' for k in range(6)]].sum(1)
    per_seed = (c.null_rejects / c.null_tests).to_numpy()
    out['cpqj_null'] = dict(tests=int(c.null_tests.sum()), rejects=int(c.null_rejects.sum()),
                            rate=float(c.null_rejects.sum() / c.null_tests.sum()), rate_ci=boot_ci(per_seed),
                            runs_any=float((c.null_rejects > 0).mean()), lowtrust_per_run=float(lt.mean()),
                            quarantined_per_run=float(q.mean()), readmitted_per_run=float(ra.mean()),
                            runs_with_quarantine=float((q > 0).mean()))
    out['cpq_null'] = dict(quarantined_per_run=float(q2.mean()), readmitted_per_run=float(ra2.mean()),
                           runs_with_quarantine=float((q2 > 0).mean()))
    N['null'] = out
    lines = []
    for m in ['fedavg', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']:
        lo_real = low(main, 'acs', m, ST)[0]
        lines.append(f"{RULES[m]} & {lo_real:.3f} & {out[m]['low']:.3f} & {out[m]['f1']:.3f} & {out[m]['gate']:.2f} \\\\")
    write('tab_null.tex', '\n'.join(lines) + '\n')
    # release gate on the ACS state federation
    lines, g = [], {}
    v = load('runs_variants.csv')
    for m in ['fedavg', 'median', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']:
        row = {}
        for d, ds, fam in [(main, 'acs', 'none'), (main, 'acs', 'signflip'), (main, 'acs', 'targeted'),
                           (main, 'acs', 'minmax'), (main, 'acs', 'liar'), (v, 'acs-slow', 'slowflip')]:
            x = by_seed(d, ds, fam, m)
            if len(x):
                row[fam] = dict(blocks=float(x.gate_blocks.mean()), f1=float(x.f1.mean()), rel_f1=float(x.rel_f1.mean()),
                                j=float(x.j.mean()), rel_j=float(x.rel_j.mean()))
        g[m] = row
        c0, s, sl = row['none'], row['signflip'], row['slowflip']
        lines.append(f"{RULES[m]} & {c0['blocks']:.2f} & {s['blocks']:.2f} & {s['f1']:.3f} & {s['rel_f1']:.3f} & "
                     f"{sl['blocks']:.2f} & {sl['f1']:.3f} & {sl['rel_f1']:.3f} \\\\")
    write('tab_gate.tex', '\n'.join(lines) + '\n')
    N['gate'] = g


# --------------------------------------------------------------------------- R1-05b guard, R2-03 slow attacker, R2-08 min-max, R2-07 FedVal
def variants(main):
    v = load('runs_variants.csv'); out = {}
    a, b = by_seed(v, 'acs-guardci', 'none', 'cpqj'), by_seed(main, 'acs', 'none', 'cpqj')
    out['guard'] = dict(
        point=dict(low=low(main, 'acs', 'cpqj', ST)[0], j=float(b.j.mean()), f1=float(b.f1.mean()),
                   sf_readmit=float(attacker_readmitted(main, 'acs', 'cpqj').mean()),
                   push_w=float(attacker_weight(main, 'acs', 'cpqj', 'pushpos').mean()),
                   tg_j=float(by_seed(main, 'acs', 'targeted', 'cpqj').j.mean())),
        ci=dict(low=low(v, 'acs-guardci', 'cpqj', ST)[0], j=float(a.j.mean()), f1=float(a.f1.mean()),
                sf_readmit=float(attacker_readmitted(v, 'acs-guardci', 'cpqj').mean()),
                push_w=float(attacker_weight(v, 'acs-guardci', 'cpqj', 'pushpos').mean()),
                tg_j=float(by_seed(v, 'acs-guardci', 'targeted', 'cpqj').j.mean())))
    lines = []
    for m in RULES:
        sl, mm = by_seed(v, 'acs-slow', 'slowflip', m), by_seed(v, 'acs-minmaxstd', 'minmax_std', m)
        ref = by_seed(main, 'acs', 'minmax', m)
        aw = attacker_weight(v, 'acs-slow', m, 'slowflip') if m not in ('median', 'trimmed') else None
        ra = attacker_readmitted(v, 'acs-slow', m, 'slowflip').mean() if m in ('cpq', 'cpqj') else None
        out[f'slow|{m}'] = dict(f1=float(sl.f1.mean()), attacker_w=None if aw is None else float(aw.mean()),
                                readmit=None if ra is None else float(ra))
        out[f'minmaxstd|{m}'] = dict(f1=float(mm.f1.mean()), j=float(mm.j.mean()), unit_f1=float(ref.f1.mean()))
        lines.append(f"{RULES[m]} & {sl.f1.mean():.3f} & {'--' if aw is None else f'{aw.mean():.3f}'} & "
                     f"{'--' if ra is None else f'{ra:.2f}'} & {ref.f1.mean():.3f} & {mm.f1.mean():.3f} & {mm.j.mean():.3f} \\\\")
    write('tab_attacks2.tex', '\n'.join(lines) + '\n')
    lines = []
    for ds0, ds1, lab, K, sts in [('synthetic', 'synthetic-fedval2', 'Synthetic proxy', 4, ['1', '2', '3', '4']),
                                  ('acs', 'acs-fedval2', 'ACS states', 6, ST)]:
        for reading, d, ds in [('Client-level MAD (main)', main, ds0), ('Per-record MAD (Eq. 4)', v, ds1)]:
            c0, s, t = by_seed(d, ds, 'none', 'fedval'), by_seed(d, ds, 'signflip', 'fedval'), by_seed(d, ds, 'targeted', 'fedval')
            lw = low(d, ds, 'fedval', sts, K)[0]
            out[f'fedval|{ds}'] = dict(f1=float(c0.f1.mean()), sf_f1=float(s.f1.mean()), tg_j=float(t.j.mean()), low=lw)
            lines.append(f"{lab if reading.startswith('Client') else ''} & {reading} & {c0.f1.mean():.3f} & {lw:.3f} & "
                         f"{s.f1.mean():.3f} & {t.j.mean():.3f} \\\\")
    write('tab_fedval2.tex', '\n'.join(lines) + '\n')
    N['variants'] = out


# --------------------------------------------------------------------------- R2-06: tau and delta
def sens2():
    s = pd.read_csv(RES / 'cpq_sensitivity2.csv'); s['j'] = s.recall_support - s.fpr_support
    lines, out = [], {}
    for ds in ['synthetic', 'acs']:
        R, K = (16, 4) if ds == 'synthetic' else (30, 6)
        for (tau, delta), g in s[s.dataset == ds].groupby(['tau', 'delta'], sort=False):
            c = g[g.family == 'none'].groupby('seed').mean(numeric_only=True)
            f = g[g.family == 'signflip'].groupby('seed').mean(numeric_only=True)
            t = g[g.family == 'targeted'].groupby('seed').mean(numeric_only=True)
            lw = c[[f'w{k}' for k in range(K)]].mean().min()
            out[f'{ds}|{tau}|{delta}'] = dict(f1=float(c.f1.mean()), j=float(c.j.mean()), low=float(lw),
                                              sf_f1=float(f.f1.mean()), sf_readmit=float(f.attacker_readmitted.mean()))
            lines.append(f"{'Synthetic' if ds == 'synthetic' else 'ACS'} & {tau:g} & {delta:g} & {c.f1.mean():.3f} & "
                         f"{c.j.mean():.3f} & {lw:.3f} & {c.readmitted.mean():.2f} & {f.f1.mean():.3f} & "
                         f"{f.attacker_readmitted.mean():.2f} & {t.j.mean():.3f} \\\\")
    write('tab_sens2.tex', '\n'.join(lines) + '\n')
    N['sens2'] = out


# --------------------------------------------------------------------------- R1-01c, R1-24, R3-03, MLP: robustness summary
def robustness(main):
    mlp = load('runs_mlp.csv'); pu = load('runs_puma.csv')
    sets = [('Main (logistic, sample)', main, 'acs', ST, 6), ('Survey-weighted', load('runs_weighted.csv'), 'acs-weighted', ST, 6),
            ('Ancestry added', load('runs_anc.csv'), 'acs-anc', ST, 6),
            ('MLP learner', mlp, 'acs-mlp', ST, 6), ('Held-out states', load('runs_heldout.csv'), 'acs-heldout', HO, 6),
            ('89 PUMAs', pu, 'acs-puma', None, 89)]
    lines, out = [], {}
    for lab, d, ds, sts, K in sets:
        row, cells = {}, []
        for m in ['fltrust', 'anchor', 'cpq', 'cpqj', 'fedval', 'multikrum']:
            w = weights(d, ds, m, K); row[m] = float(w.min())
            cells.append(f'{w.min():.3f}')
        a, b = by_seed(d, ds, 'none', 'cpqj'), by_seed(d, ds, 'none', 'fedavg')
        dj = (a.j - b.j).dropna().to_numpy()
        row['j_cpqj_minus_fedavg'] = dict(mean=float(dj.mean()), ci=boot_ci(dj))
        row['f1_fedavg'] = float(b.f1.mean()); row['f1_cpqj'] = float(a.f1.mean())
        out[lab] = row
        lines.append(f"{lab} & " + ' & '.join(cells) + f" & {num(dj.mean(), 3, True)} {ci_txt(row['j_cpqj_minus_fedavg']['ci'])} \\\\")
    write('tab_robust.tex', '\n'.join(lines) + '\n')
    N['robust'] = out
    tune = json.loads((RES / 'mlp_tuning.json').read_text())
    c4 = by_seed(mlp, 'acs-mlp', 'none', 'central4'); c1 = by_seed(mlp, 'acs-mlp', 'none', 'central')
    N['mlp'] = dict(tuning=tune, central_f1=float(c1.f1.mean()), central4_f1=float(c4.f1.mean()) if len(c4) else None,
                    fedavg_f1=float(by_seed(mlp, 'acs-mlp', 'none', 'fedavg').f1.mean()))


# --------------------------------------------------------------------------- R1-23, R3-03: pre-registered held-out hypotheses
def heldout():
    h = load('runs_heldout.csv'); out = {}
    perm_tests(h, 'acs-heldout', HO, 'perm_heldout')
    lw = lambda x: x[[f'w{k}' for k in range(6)]].min(1)  # noqa: E731
    d2 = (lw(by_seed(h, 'acs-heldout', 'none', 'cpqj')) - lw(by_seed(h, 'acs-heldout', 'none', 'anchor'))).to_numpy()
    r_b = attacker_readmitted(h, 'acs-heldout', 'cpqj'); r_r = attacker_readmitted(h, 'acs-heldout', 'cpq')
    d3 = (r_b - r_r).to_numpy()
    d4 = (by_seed(h, 'acs-heldout', 'none', 'cpqj').j - by_seed(h, 'acs-heldout', 'none', 'fedavg').j).to_numpy()
    out['H2'] = dict(mean=float(d2.mean()), ci=boot_ci(d2))
    out['H3'] = dict(cpqj=float(r_b.mean()), cpq=float(r_r.mean()), diff=float(d3.mean()), ci=boot_ci(d3))
    out['H4'] = dict(mean=float(d4.mean()), ci=boot_ci(d4))
    out['gap'] = {s: abs(rate(s) - rate('vt')) for s in HO}
    out['weights'] = {m: dict(zip(HO, weights(h, 'acs-heldout', m).round(4).tolist())) for m in ['fltrust', 'anchor', 'cpq', 'cpqj', 'fedval', 'multikrum']}
    if (RES / 'runs_heldout_center.csv').exists():
        hc = load('runs_heldout_center.csv')
        for m in ['fltrust', 'anchor', 'cpqj']:
            a, b = by_seed(hc, 'acs-heldout-center', 'none', m), by_seed(h, 'acs-heldout', 'none', m)
            dl, df = (lw(a) - lw(b)).to_numpy(), (a.f1 - b.f1).to_numpy()
            sf = by_seed(hc, 'acs-heldout-center', 'signflip', m)
            out[f'H5|{m}'] = dict(low_std=float(lw(b).mean()), low_ctr=float(lw(a).mean()), dlow=float(dl.mean()),
                                  dlow_ci=boot_ci(dl), df1=float(df.mean()), df1_ci=boot_ci(df))
            out[f'H6|{m}'] = dict(clean_f1=float(a.f1.mean()), sf_f1=float(sf.f1.mean()))
    N['heldout'] = out


# --------------------------------------------------------------------------- R0-02: voluntary disclosure
def disclosure(main):
    d = load('runs_disclose.csv'); out, lines = {}, []
    age_p = []
    for s in ST:
        dd = np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz')
        age_p.append((0.2 + 0.8 * (dd['X'][dd['g'] == 1, 0] * 100 - 16) / 74))
    rate_age = float(np.concatenate(age_p).mean())
    scen = [('Full disclosure', main, 'acs', 1.0), ('80\\% at random', d, 'acs-disc-mcar80', .8),
            ('60\\% at random', d, 'acs-disc-mcar60', .6), ('40\\% at random', d, 'acs-disc-mcar40', .4),
            ('Rises with age', d, 'acs-disc-age', rate_age)]
    for lab, dd, ds, p in scen:
        row = {}
        for m in ['fedavg', 'fltrust', 'cpqj']:
            x = by_seed(dd, ds, 'none', m)
            aud = float(x.aud_j.mean()) if 'aud_j' in x else float(x.j.mean())
            row[m] = dict(j=float(x.j.mean()), aud_j=aud, low=low(dd, ds, m, ST)[0], f1=float(x.f1.mean()))
        xb = dd[(dd.dataset == ds) & (dd.method == 'cpqj') & (dd.family == 'none')]
        row['cpqj_quar_per_run'] = float(xb[[f'q{k}' for k in range(6)]].sum(1).mean())
        row['cpqj_readm_per_run'] = float(xb[[f'ra{k}' for k in range(6)]].sum(1).mean())
        row['cpqj_sf_readmit'] = float(attacker_readmitted(dd, ds, 'cpqj').mean())
        row['p'] = p; out[lab] = row
        lines.append(f"{lab} & {p:.2f} & {row['fedavg']['j']:.3f} & {row['fedavg']['aud_j']:.3f} & {row['cpqj']['j']:.3f} & "
                     f"{row['cpqj']['aud_j']:.3f} & {row['cpqj']['low']:.3f} & {row['cpqj_readm_per_run']:.2f} & "
                     f"{row['cpqj_sf_readmit']:.2f} \\\\")
    write('tab_disclose.tex', '\n'.join(lines) + '\n')
    N['disclose'] = out


def singh(main):
    """R3-02: the microaggregation detector of Singh et al. (2023), k = 3, and its distance baseline (k = 1)."""
    d = load('runs_singh.csv'); pu = load('runs_puma.csv'); out, lines = {}, []
    rows = [('fedavg', main, RULES['fedavg']), ('multikrum', main, RULES['multikrum']), ('fltrust', main, RULES['fltrust']),
            ('cpqj', main, RULES['cpqj']), ('singh1', d, 'Distance rule (Singh et al., $k=1$)'),
            ('singh3', d, 'Microaggregation (Singh et al., $k=3$)')]
    for m, dd, lab in rows:
        c, s, t = by_seed(dd, 'acs', 'none', m), by_seed(dd, 'acs', 'signflip', m), by_seed(dd, 'acs', 'targeted', m)
        lw = low(dd, 'acs', m, ST)[0]
        x = dd[(dd.dataset == 'acs') & (dd.family == 'signflip') & (dd.method == m)].copy()
        x['ax'] = [r[f"ex{int(r['attackers'])}"] for _, r in x.iterrows()]
        ax = float(x.groupby('seed')['ax'].mean().mean())
        pdd = d if m.startswith('singh') else pu
        pc, ps = by_seed(pdd, 'acs-puma', 'none', m), by_seed(pdd, 'acs-puma', 'signflip', m)
        plw = float(weights(pdd, 'acs-puma', m, 89).min()) if pc['w0'].notna().all() else None
        out[m] = dict(f1=float(c.f1.mean()), j=float(c.j.mean()), low=lw, sf_f1=float(s.f1.mean()), sf_att_excl=ax,
                      tg_j=float(t.j.mean()), puma_f1=float(pc.f1.mean()), puma_low=plw, puma_sf_f1=float(ps.f1.mean()))
        lines.append(f"{lab} & {c.f1.mean():.3f} & {lw:.3f} & {c.j.mean():.3f} & {s.f1.mean():.3f} & {ax:.3f} & {t.j.mean():.3f} & "
                     f"{pc.f1.mean():.3f} & {'--' if plw is None else f'{89 * plw:.2f}'} & {ps.f1.mean():.3f} \\\\")
    write('tab_singh.tex', '\n'.join(lines) + '\n')
    N['singh'] = out


def partial():
    """R3-06: partial participation (4 of 6 states, half of the 89 PUMAs per round), standard and centered cosine.
    Weights are relative to a uniform share among the participants of a round, scaled so that uniform is 1/K."""
    if not (RES / 'runs_partial.csv').exists():
        return
    s = load('runs_partial.csv')
    pu = load('runs_partial_puma.csv') if (RES / 'runs_partial_puma.csv').exists() else None
    out, lines = {}, []
    for ctr, meths in [(False, ['fedavg', 'fltrust', 'anchor', 'cpqj']), (True, ['fltrust', 'anchor', 'cpqj'])]:
        if ctr:
            lines.append('\\midrule')
        for m in meths:
            ds = 'acs-partial' + ('-center' if ctr else '')
            c, sf = by_seed(s, ds, 'none', m), by_seed(s, ds, 'signflip', m)
            lw = float(weights(s, ds, m).min())
            row = dict(f1=float(c.f1.mean()), j=float(c.j.mean()), low=lw, sf_f1=float(sf.f1.mean()))
            cells = [f'{row["f1"]:.3f}', f'{lw:.3f}', f'{row["j"]:.3f}', f'{row["sf_f1"]:.3f}']
            if pu is not None:
                pds = 'acs-puma-partial' + ('-center' if ctr else '')
                pc, ps = by_seed(pu, pds, 'none', m), by_seed(pu, pds, 'signflip', m)
                plw = float(weights(pu, pds, m, 89).min())
                row.update(puma_f1=float(pc.f1.mean()), puma_low=plw, puma_sf_f1=float(ps.f1.mean()))
                cells += [f'{row["puma_f1"]:.3f}', f'{89 * plw:.2f}', f'{row["puma_sf_f1"]:.3f}']
            out[m + ('_center' if ctr else '')] = row
            lines.append(RULES[m] + (', centered' if ctr else '') + ' & ' + ' & '.join(cells) + ' \\\\')
    write('tab_partial.tex', '\n'.join(lines) + '\n')
    N['partial'] = out


def main():
    main_runs = load('runs.csv')
    anchor_runs = load('runs_anchor.csv')
    states_table(); perm_tests(main_runs); puma_regression(); t_intervals(main_runs); calibration(main_runs)
    fixes(main_runs, anchor_runs); null_and_gate(main_runs); variants(main_runs); sens2(); robustness(main_runs)
    heldout(); disclosure(main_runs); singh(main_runs); partial()
    N['mechanism'] = json.loads((RES / 'mechanism_check.json').read_text())
    (RES / 'jas_numbers.json').write_text(json.dumps(N, indent=1))
    print('written', len(N), 'groups')


if __name__ == '__main__':
    main()

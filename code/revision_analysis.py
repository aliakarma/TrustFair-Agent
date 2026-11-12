"""Summaries, paired bootstrap intervals, LaTeX tables and pgfplots data for all experiments.

Run after revision_experiments.py:  python code/revision_analysis.py
Unit of analysis: the seed. Rotated attacker identities are averaged within a seed first.
Intervals: 95% percentile bootstrap over the 20 seeds (10,000 resamples, generator seed 0).
J_S = TPR_S - FPR_S is the support group's Youden index (true-positive rate minus false-positive rate).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('TF_RES', ROOT / 'results' / 'revision'))  # JAS revision: TF_RES=results/revision_jas
TAB = RES / 'tables'
STATES = ['wv', 'me', 'nm', 'ut', 'nd', 'ms']
METRICS = ['f1', 'tpr_gap', 'worst_client_f1', 'recall_support', 'worst_group_recall', 'recall_majority',
           'fpr_support', 'fpr_majority', 'pos_rate', 'auc', 'auc_support', 'mean_prob', 'j_support', 'j_majority',
           'brier', 'logloss', 'citl', 'citl_support', 'rel_f1', 'rel_j_support', 'gate_blocks']
NAMES = {'fedavg': 'FedAvg', 'median': 'Coordinate median', 'trimmed': 'Trimmed mean', 'multikrum': 'Multi-Krum',
         'fltrust': 'FLTrust', 'fltrust_noscale': 'FLTrust without rescaling', 'fedval': 'FedVal',
         'anchor': 'Anchor rule', 'cpq': 'CPQ-R', 'cpqj': 'CPQ-B',
         'anchor+rw': 'Anchor + 1.5 weight', 'anchor+int': 'Anchor + intercept',
         'anchor+rw+int': 'Anchor + weight + intercept', 'cpq+rw+int': 'CPQ-R + weight + intercept',
         'cpqj+rw+int': 'CPQ-B + weight + intercept', 'central': 'Centralized reference',
         'local': 'Local-only reference', 'bayes': 'Bayes rule of the oracle'}
SHORT = {'fedavg': 'FedAvg', 'median': 'Median', 'trimmed': 'Trimmed mean', 'multikrum': 'Multi-Krum',
         'fltrust': 'FLTrust', 'fedval': 'FedVal', 'anchor': 'Anchor rule', 'cpq': 'CPQ-R', 'cpqj': 'CPQ-B'}
MAIN = ['fedavg', 'median', 'trimmed', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']
TRUST = ['fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']
BOOT = np.random.default_rng(0)


def K_of(ds):
    return 4 if ds.startswith('synthetic') else (89 if ds == 'acs-puma' else 6)


def rounds_of(ds):
    return 16 if ds.startswith('synthetic') else 30


def boot_ci(v, B=10000):
    v = np.asarray(v, float)
    idx = BOOT.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(m, s=None, d=3):
    return f'{m:.{d}f}' if s is None else f'{m:.{d}f} ({s:.{d}f})'


def _num(x, d, sign):
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    if float(s) == 0:  # no signed zero after rounding
        s = f'{0:.{d}f}'
    return s.replace('+', '$+$').replace('-', '$-$')


def sfmt(x, d=3):
    return _num(x, d, True)


def ci_cell(p, d=3):  # bounds carry a sign only when negative
    return f"{sfmt(p['mean'], d)} [{_num(p['ci'][0], d, False)}, {_num(p['ci'][1], d, False)}]"


def write(name, text):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(text, encoding='utf-8')


def load():
    files = ['runs.csv', 'runs_anchor.csv', 'runs_match.csv', 'runs_puma.csv', 'runs_mlp.csv']
    df = pd.concat([pd.read_csv(RES / f) for f in files if (RES / f).exists()], ignore_index=True)
    df['j_support'] = df['recall_support'] - df['fpr_support']
    df['j_majority'] = df['recall_majority'] - df['fpr_majority']
    if 'rel_recall_support' in df:
        df['rel_j_support'] = df['rel_recall_support'] - df['rel_fpr_support']
    for c in ['brier', 'logloss', 'citl', 'citl_support', 'rel_f1', 'rel_j_support', 'gate_blocks']:
        if c not in df:
            df[c] = np.nan
    return df


def detection(df):
    out = []
    for r in df[df['family'] != 'none'].to_dict('records'):
        att = [int(a) for a in str(r['attackers']).split('-')]
        k = K_of(r['dataset'])
        if pd.isna(r.get('w0', np.nan)):
            continue
        ex = [r[f'ex{j}'] for j in range(k)]; w = [r[f'w{j}'] for j in range(k)]
        ra = [r.get(f'ra{j}', np.nan) for j in range(k)]; q = [r.get(f'q{j}', np.nan) for j in range(k)]
        ben = [j for j in range(k) if j not in att]
        out.append(dict(dataset=r['dataset'], family=r['family'], method=r['method'], seed=r['seed'],
                        attacker_excluded=np.mean([ex[a] for a in att]), benign_excluded=np.mean([ex[j] for j in ben]),
                        attacker_weight=np.mean([w[a] for a in att]), attacker_readmitted=np.sum([ra[a] for a in att]),
                        benign_quarantined=np.sum([q[j] for j in ben]), benign_readmitted=np.sum([ra[j] for j in ben])))
    return pd.DataFrame(out).groupby(['dataset', 'family', 'method', 'seed'], as_index=False).mean()


def summarize(sl, det):
    summary = []
    for (ds, fam, meth), g in sl.groupby(['dataset', 'family', 'method']):
        item = dict(dataset=ds, family=fam, method=meth, n_seeds=len(g))
        for m in METRICS:
            v = g[m].dropna().to_numpy()
            if len(v):
                item[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0., ci=boot_ci(v))
        kk, R = K_of(ds), rounds_of(ds)
        if 'w0' in g and g['w0'].notna().any():
            item['weights'] = [float(g[f'w{j}'].mean()) for j in range(kk)]
            item['excluded'] = [float(g[f'ex{j}'].mean()) for j in range(kk)]
        if meth in ('cpq', 'cpqj') and 'q0' in g and g['q0'].notna().any():
            for c, col in [('quarantined', 'q'), ('lowtrust', 'lt'), ('readmitted', 'ra')]:
                item[c + '_per_round'] = float(sum(g[f'{col}{j}'].mean() for j in range(kk)) / R)
            item['evals_per_round'] = float(g['evals'].mean() / R)
            for c in ['multi_readmit', 'joint_violation', 'joint_blocked']:
                item[c + '_per_run'] = float(g[c].mean())
        d = det[(det.dataset == ds) & (det.family == fam) & (det.method == meth)] if len(det) else det
        if len(d):
            for c in ['attacker_excluded', 'benign_excluded', 'attacker_weight', 'attacker_readmitted',
                      'benign_quarantined', 'benign_readmitted']:
                item[c] = float(d[c].mean())
        summary.append(item)
    return summary


COMPS = [('cpqj', b) for b in ['fedavg', 'median', 'trimmed', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq']]
COMPS += [('cpq', b) for b in ['fedavg', 'median', 'multikrum', 'fltrust', 'fedval', 'anchor']]
COMPS += [('anchor+rw', 'anchor'), ('anchor+int', 'anchor'), ('anchor+rw+int', 'anchor'), ('cpq+rw+int', 'cpq'),
          ('cpqj+rw+int', 'cpqj'), ('fltrust', 'anchor'), ('fltrust_noscale', 'fltrust'), ('fedval', 'fltrust')]
PAIR_METRICS = ['f1', 'recall_support', 'fpr_support', 'j_support', 'tpr_gap', 'auc', 'pos_rate', 'brier']


def paired_all(sl):
    out = []
    for (ds, fam), g in sl.groupby(['dataset', 'family']):
        by = {m: x.set_index('seed') for m, x in g.groupby('method')}
        for a, b in COMPS:
            if a not in by or b not in by:
                continue
            item = dict(dataset=ds, family=fam, a=a, b=b)
            for m in PAIR_METRICS:
                dlt = (by[a][m] - by[b][m]).dropna().to_numpy()
                if len(dlt):
                    item[m] = dict(mean=float(dlt.mean()), ci=boot_ci(dlt), n_pos=int((dlt > 0).sum()),
                                   n_neg=int((dlt < 0).sum()), n=len(dlt))
            out.append(item)
    return out


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return beta, 1 - resid.var() / y.var()


def main():
    runs = load()
    sl = runs.groupby(['dataset', 'family', 'method', 'seed'], as_index=False).mean(numeric_only=True)
    det = detection(runs)
    summary = summarize(sl, det)
    S = {(s['dataset'], s['family'], s['method']): s for s in summary}
    paired = paired_all(sl)
    P = {(p['dataset'], p['family'], p['a'], p['b']): p for p in paired}
    (RES / 'summary.json').write_text(json.dumps(summary, indent=1))
    (RES / 'paired.json').write_text(json.dumps(paired, indent=1))
    rate = {s: float(np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz')['y'].mean()) for s in STATES + ['vt']}
    prev = {s: float(np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz')['g'].mean()) for s in STATES + ['vt']}
    order = sorted(range(6), key=lambda j: rate[STATES[j]])  # states sorted by employment rate
    numbers = dict(employment=rate, prevalence=prev)
    lines = []
    for s in [STATES[j] for j in order] + ['vt']:
        d = np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz'); y, g = d['y'], d['g']
        lines.append(f"{s.upper()}{' (anchor)' if s == 'vt' else ''} & {len(y):,} & {y.mean():.3f} & {g.mean():.3f} & "
                     f"{y[g == 1].mean():.3f} \\\\")
    write('tab_states.tex', '\n'.join(lines) + '\n')

    def m_(ds, fam, meth, metric):
        s = S.get((ds, fam, meth))
        return s[metric]['mean'] if s and metric in s else np.nan

    # ---- main comparison tables (means; SDs in the supplement tables)
    for ds in ['synthetic', 'acs']:
        lines = []
        for meth in MAIN + ['central', 'local'] + (['bayes'] if ds == 'synthetic' else []):
            cells = []
            for fam in ['none', 'signflip', 'targeted']:
                for m in ['f1', 'recall_support', 'fpr_support']:
                    v = m_(ds, fam, meth, m)
                    cells.append('--' if np.isnan(v) else fmt(v))
            if meth == 'central':
                lines.append('\\midrule')
            lines.append(NAMES[meth] + ' & ' + ' & '.join(cells) + ' \\\\')
        write(f'tab_main_{ds}.tex', '\n'.join(lines) + '\n')

    # ---- supplement: mean (SD) for every attack family
    fams = [('none', 'Clean'), ('signflip', 'Sign-flip'), ('targeted', 'Targeted'), ('minmax', 'Min-max'),
            ('pushpos', 'Positive push')]
    for ds in ['synthetic', 'acs', 'acs-puma', 'acs-mlp']:
        lines = []
        for fam, label in fams:
            meths = [m for m in MAIN + ['central', 'local', 'bayes'] if (ds, fam, m) in S]
            if not meths:
                continue
            lines.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
            for meth in meths:
                s = S[(ds, fam, meth)]
                cells = [fmt(s[m]['mean'], s[m]['sd']) if m in s else '--' for m in
                         ['f1', 'recall_support', 'fpr_support', 'j_support', 'auc']]
                label = {'central': 'Centralized', 'local': 'Local-only', 'bayes': 'Bayes rule'}.get(meth, SHORT.get(meth))
                lines.append(label + ' & ' + ' & '.join(cells) + ' \\\\')
        write(f'tab_full_{ds}.tex', '\n'.join(lines) + '\n')

    # ---- paired differences: balanced CPQ minus comparator in J_S (clean F1 differences stay in paired.json)
    lines = []
    for ds, label in [('synthetic', 'Synthetic proxy'), ('acs', 'ACS state federation')]:
        lines.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for b in ['fedavg', 'median', 'trimmed', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq']:
            cells = [ci_cell(P[(ds, fam, 'cpqj', b)]['j_support']) for fam in ['none', 'signflip', 'targeted']]
            lines.append(SHORT[b] + ' & ' + ' & '.join(cells) + ' \\\\')
    write('tab_paired.tex', '\n'.join(lines) + '\n')

    # ---- supplement: recall-test CPQ minus comparator (recall and J_S), one row per comparison and condition
    lines = []
    for ds, label in [('synthetic', 'Synthetic proxy'), ('acs', 'ACS state federation')]:
        lines.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for b in ['fedavg', 'median', 'multikrum', 'fltrust', 'fedval', 'anchor']:
            for fam, fl in [('none', 'Clean'), ('signflip', 'Sign-flip'), ('targeted', 'Targeted')]:
                p = P[(ds, fam, 'cpq', b)]
                lines.append(f"{'vs ' + SHORT[b] if fam == 'none' else ''} & {fl} & {ci_cell(p['recall_support'])} & "
                             f"{ci_cell(p['j_support'])} \\\\")
    write('tab_paired_recall.tex', '\n'.join(lines) + '\n')

    # ---- exclusion of attacker and benign clients
    lines = []
    for ds, label in [('synthetic', 'Synthetic proxy'), ('acs', 'ACS state federation')]:
        lines.append(f'\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for meth in ['fedavg', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']:
            c, sf, tg = S[(ds, 'none', meth)], S[(ds, 'signflip', meth)], S[(ds, 'targeted', meth)]
            ra = fmt(sf['attacker_readmitted'], d=2) if meth in ('cpq', 'cpqj') else '--'
            lines.append(f"{NAMES[meth]} & {fmt(np.mean(c['excluded']))} & {fmt(sf['attacker_excluded'])} & "
                         f"{fmt(sf['benign_excluded'])} & {ra} & {fmt(tg['attacker_excluded'])} & {fmt(tg['benign_excluded'])} \\\\")
    write('tab_detect.tex', '\n'.join(lines) + '\n')

    # ---- adaptive attacks: min-max, positive push, lying reporter
    lines = []
    for ds, label in [('synthetic', 'Synthetic proxy'), ('acs', 'ACS state federation')]:
        lines.append(f'\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for meth in MAIN:
            mm, pp, lr = S[(ds, 'minmax', meth)], S[(ds, 'pushpos', meth)], S.get((ds, 'liar', meth))
            liar = (f"{fmt(lr['f1']['mean'])} & {fmt(lr['benign_excluded'])}" if lr else '-- & --')
            aw = fmt(pp['attacker_weight']) if 'attacker_weight' in pp else '--'
            lines.append(f"{NAMES[meth]} & {fmt(mm['f1']['mean'])} & {fmt(mm['j_support']['mean'])} & "
                         f"{fmt(pp['f1']['mean'])} & {fmt(pp['fpr_support']['mean'])} & {aw} & {liar} \\\\")
    write('tab_adaptive.tex', '\n'.join(lines) + '\n')

    # ---- base rate versus prevalence: matched samples
    lines = []
    for ds, label in [('acs-match-rate', 'Employment rate set to the anchor\'s (0.595), prevalence kept'),
                      ('acs-match-prev', 'Prevalence set to the six-state mean, employment rate kept')]:
        lines.append(f'\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for meth in ['multikrum'] + TRUST:
            s = S[(ds, 'none', meth)]
            lines.append(NAMES[meth] + ' & ' + ' & '.join(f"{s['weights'][j]:.3f}" for j in order)
                         + f" & {fmt(s['f1']['mean'])} \\\\")
    write('tab_match.tex', '\n'.join(lines) + '\n')
    lines = []  # same layout for the unmatched main grid, for the reader's comparison
    for meth in ['fedavg', 'multikrum'] + TRUST:
        s = S[('acs', 'none', meth)]
        lines.append(NAMES[meth] + ' & ' + ' & '.join(f"{s['weights'][j]:.3f}" for j in order)
                     + f" & {fmt(s['f1']['mean'])} \\\\")
    write('tab_weights_acs.tex', '\n'.join(lines) + '\n')

    # ---- anchor sensitivity: lowest state weight per rule
    lines, anc_rows = [], [('vt', 'acs')] + [(a, f'acs-anchor-{a}') for a in [STATES[j] for j in order] + ['pooled']]
    for a, ds in anc_rows:
        label = 'Pooled' if a == 'pooled' else (a.upper() + (' (main)' if a == 'vt' else ''))
        r = 'mixed' if a == 'pooled' else f'{rate[a]:.3f}'
        cells = []
        for meth in TRUST:
            w = S[(ds, 'none', meth)]['weights']
            j = int(np.argmin(w)); cells.append(f'{w[j]:.3f} ({STATES[j].upper()})')
        lines.append(f"{label} & {r} & " + ' & '.join(cells) + ' \\\\')
        f1s = [S[(ds, 'none', meth)]['f1']['mean'] for meth in TRUST]
        numbers.setdefault('anchor_f1_range', {})[a] = [min(f1s), max(f1s)]
    write('tab_anchor.tex', '\n'.join(lines) + '\n')
    lines = []
    for a, ds in anc_rows:
        label = 'Pooled' if a == 'pooled' else a.upper()
        for meth in TRUST:
            s = S[(ds, 'none', meth)]
            lines.append(f"{label} & {SHORT[meth]} & " + ' & '.join(f"{s['weights'][j]:.3f}" for j in order)
                         + f" & {fmt(s['f1']['mean'])} & {fmt(s['j_support']['mean'])} \\\\")
        lines.append('\\addlinespace')
    write('tab_anchor_full.tex', '\n'.join(lines[:-1]) + '\n')
    lines = []
    for fam, label in [('signflip', 'Sign-flip'), ('targeted', 'Targeted')]:
        for meth in TRUST:
            s = S[('acs-anchor-pooled', fam, meth)]
            lines.append(f"{label} & {SHORT[meth]} & {fmt(s['f1']['mean'], s['f1']['sd'])} & "
                         f"{fmt(s['recall_support']['mean'])} & {fmt(s['fpr_support']['mean'])} & "
                         f"{fmt(s['attacker_excluded'])} \\\\")
    write('tab_anchor_pooled_attack.tex', '\n'.join(lines) + '\n')

    # ---- PUMA federation: main results and weight regression
    meta = json.loads((RES / 'puma_meta.json').read_text())
    emp = np.array([m['employment'] for m in meta]); pv = np.array([m['prevalence'] for m in meta])
    dist = np.abs(emp - rate['vt'])
    numbers['puma'] = dict(K=len(meta), corr_dist_prev=float(np.corrcoef(dist, pv)[0, 1]),
                           corr_emp_prev=float(np.corrcoef(emp, pv)[0, 1]),
                           emp_range=[float(emp.min()), float(emp.max())], prev_range=[float(pv.min()), float(pv.max())])
    lines = []
    for meth in MAIN + ['central', 'local']:
        cells = []
        for fam, m in [('none', 'f1'), ('none', 'recall_support'), ('none', 'fpr_support'), ('signflip', 'f1'),
                       ('targeted', 'recall_support'), ('targeted', 'fpr_support'), ('minmax', 'f1')]:
            v = m_('acs-puma', fam, meth, m)
            cells.append('--' if np.isnan(v) else fmt(v))
        if meth == 'central':
            lines.append('\\midrule')
        lines.append(NAMES[meth] + ' & ' + ' & '.join(cells) + ' \\\\')
    write('tab_puma.tex', '\n'.join(lines) + '\n')
    pu = sl[(sl.dataset == 'acs-puma') & (sl.family == 'none')]
    zs = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    X = np.column_stack([zs(dist), zs(pv)])
    wcols = [f'w{j}' for j in range(len(meta))]
    reg, lines = {}, []
    for meth in ['multikrum'] + TRUST:
        g = pu[pu.method == meth].sort_values('seed')
        fits = [ols(len(meta) * r[wcols].to_numpy(float), X) for _, r in g.iterrows()]
        B = np.array([f[0][1:] for f in fits]); r2 = float(np.mean([f[1] for f in fits]))
        reg[meth] = dict(dist=dict(mean=float(B[:, 0].mean()), ci=boot_ci(B[:, 0])),
                         prev=dict(mean=float(B[:, 1].mean()), ci=boot_ci(B[:, 1])), r2=r2)
        lines.append(f"{NAMES[meth]} & {ci_cell(reg[meth]['dist'])} & {ci_cell(reg[meth]['prev'])} & {r2:.2f} \\\\")
    write('tab_puma_reg.tex', '\n'.join(lines) + '\n')
    numbers['puma_regression'] = reg
    with open(RES / 'puma_scatter.dat', 'w') as fh:
        fh.write('employment prevalence ' + ' '.join(['multikrum'] + TRUST) + '\n')
        for i in range(len(meta)):
            fh.write(f'{emp[i]:.4f} {pv[i]:.4f} ' + ' '.join(
                f"{len(meta) * S[('acs-puma', 'none', m)]['weights'][i]:.4f}" for m in ['multikrum'] + TRUST) + '\n')

    # ---- MLP learner on the state federation
    lines = []
    for meth in MAIN + ['central', 'local']:
        cells = []
        for fam, m in [('none', 'f1'), ('none', 'recall_support'), ('none', 'fpr_support'), ('none', 'auc'),
                       ('signflip', 'f1'), ('targeted', 'f1'), ('targeted', 'j_support')]:
            v = m_('acs-mlp', fam, meth, m)
            cells.append('--' if np.isnan(v) else fmt(v))
        if meth == 'central':
            lines.append('\\midrule')
        lines.append(NAMES[meth] + ' & ' + ' & '.join(cells) + ' \\\\')
    write('tab_mlp.tex', '\n'.join(lines) + '\n')
    if ('acs-mlp', 'none', 'fltrust') in S:
        numbers['mlp_weights'] = {m: [S[('acs-mlp', 'none', m)]['weights'][j] for j in order] for m in TRUST + ['multikrum']}
        numbers['mlp_tuning'] = json.loads((RES / 'mlp_tuning.json').read_text())

    # ---- operating point mechanism (clean condition)
    lines = []
    for ds, label in [('synthetic', 'Synthetic proxy'), ('acs', 'ACS state federation')]:
        lines.append(f'\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for meth in ['fedavg', 'multikrum', 'fltrust', 'fltrust_noscale', 'fedval', 'anchor', 'cpq', 'cpqj']:
            s = S[(ds, 'none', meth)]
            lines.append(NAMES[meth] + ' & ' + ' & '.join(fmt(s[m]['mean']) for m in
                         ['f1', 'pos_rate', 'mean_prob', 'auc', 'recall_support', 'fpr_support']) + ' \\\\')
    write('tab_mechanism.tex', '\n'.join(lines) + '\n')

    # ---- cost and review load of the two CPQ variants
    lines = []
    for ds, label in [('synthetic', 'Synthetic'), ('acs', 'ACS states'), ('acs-puma', 'ACS PUMAs')]:
        for fam, fl in [('none', 'clean'), ('signflip', 'sign-flip')]:
            for meth in ['cpq', 'cpqj']:
                s = S.get((ds, fam, meth))
                if not s:
                    continue
                lines.append(f"{label}, {fl} & {SHORT[meth]} & {s['lowtrust_per_round']:.2f} & {s['quarantined_per_round']:.2f} & "
                             f"{s['readmitted_per_round']:.2f} & {s['evals_per_round']:.2f} & "
                             f"{s['multi_readmit_per_run']:.2f} & {s['joint_violation_per_run']:.2f} & {s['joint_blocked_per_run']:.2f} \\\\")
    write('tab_cost.tex', '\n'.join(lines) + '\n')

    # ---- ablation of the fairness components
    lines = []
    for ds, label in [('synthetic', 'Synthetic proxy'), ('acs', 'ACS state federation')]:
        lines.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for meth in ['anchor', 'anchor+rw', 'anchor+int', 'anchor+rw+int', 'cpq', 'cpq+rw+int', 'cpqj', 'cpqj+rw+int']:
            cells = [fmt(S[(ds, fam, meth)][m]['mean'], S[(ds, fam, meth)][m]['sd'])
                     for fam, m in [('none', 'tpr_gap'), ('none', 'recall_support'), ('signflip', 'tpr_gap')]]
            lines.append(NAMES[meth] + ' & ' + ' & '.join(cells) + ' \\\\')
    write('tab_ablation.tex', '\n'.join(lines) + '\n')

    # ---- two attackers (synthetic only)
    lines = []
    for meth in MAIN:
        s = S[('synthetic', 'signflip2', meth)]
        lines.append(f"{NAMES[meth]} & {fmt(s['f1']['mean'], s['f1']['sd'])} & "
                     f"{fmt(s['recall_support']['mean'], s['recall_support']['sd'])} & {fmt(s['fpr_support']['mean'])} \\\\")
    write('tab_two_attackers.tex', '\n'.join(lines) + '\n')

    # ---- anchor representativeness (synthetic, mixture anchor)
    lines = []
    for meth in TRUST:
        a = S[('synthetic', 'none', meth)]; b = S[('synthetic-mixanchor', 'none', meth)]
        bs = S[('synthetic-mixanchor', 'signflip', meth)]
        lines.append(f"{SHORT[meth]} & " + ' & '.join(f'{x:.3f}' for x in a['weights']) + ' & '
                     + ' & '.join(f'{x:.3f}' for x in b['weights']) + f" & {fmt(bs['f1']['mean'], bs['f1']['sd'])} \\\\")
    write('tab_anchor_mix.tex', '\n'.join(lines) + '\n')

    # ---- g in the input (synthetic, 20 seeds), with clean weights
    ga = pd.read_csv(RES / 'g_input_ablation.csv')
    lines, gsum = [], []
    for fam in ['none', 'signflip']:
        for meth in ['fedavg', 'anchor', 'anchor+rw+int']:
            cells = []
            for keep in (1, 0):
                g = ga[(ga.family == fam) & (ga.method == meth) & (ga.g_in_input == keep)]
                cells += [fmt(g['f1'].mean(), g['f1'].std(ddof=1)), fmt(g['tpr_gap'].mean(), g['tpr_gap'].std(ddof=1))]
            a = ga[(ga.family == fam) & (ga.method == meth) & (ga.g_in_input == 0)].set_index('seed')
            b = ga[(ga.family == fam) & (ga.method == meth) & (ga.g_in_input == 1)].set_index('seed')
            dg = (a['tpr_gap'] - b['tpr_gap']).to_numpy()
            gsum.append(dict(family=fam, method=meth, gap_diff_mean=float(dg.mean()), gap_diff_ci=boot_ci(dg),
                             weights_g_in=[float(b[f'w{k}'].mean()) for k in range(4)],
                             weights_g_out=[float(a[f'w{k}'].mean()) for k in range(4)]))
            lines.append(f"{'Clean' if fam == 'none' else 'Sign-flip'} & {NAMES[meth]} & " + ' & '.join(cells) + ' \\\\')
    write('tab_g_input.tex', '\n'.join(lines) + '\n')
    (RES / 'g_input_summary.json').write_text(json.dumps(gsum, indent=1))

    # ---- CPQ threshold sensitivity
    sens = pd.read_csv(RES / 'cpq_sensitivity.csv')
    sens['j_support'] = sens['recall_support'] - sens['fpr_support']
    lines = []
    for ds in ['synthetic', 'acs']:
        R = rounds_of(ds)
        for keys, g in sens[sens.dataset == ds].groupby(['method', 'a0', 'nu', 'z', 'eta1'], sort=False):
            meth, a0, nu, zz, e1 = keys
            c = g[g.family == 'none'].groupby('seed').mean(numeric_only=True)
            f = g[g.family == 'signflip'].groupby('seed').mean(numeric_only=True)
            t = g[g.family == 'targeted'].groupby('seed').mean(numeric_only=True)
            thr = f'$\\nu={nu:g}$' if meth == 'cpq' else f'$z={zz:g}$'
            lines.append(f"{'Synthetic' if ds == 'synthetic' else 'ACS'} & {SHORT[meth]} & {a0:.1f} & {thr} & {e1:g} & "
                         f"{c['f1'].mean():.3f} & {c['j_support'].mean():.3f} & {c['lowtrust'].mean() / R:.2f} & "
                         f"{c['readmitted'].mean():.2f} & {f['f1'].mean():.3f} & {f['attacker_readmitted'].mean():.2f} & "
                         f"{t['j_support'].mean():.3f} \\\\")
    write('tab_sensitivity.tex', '\n'.join(lines) + '\n')

    # ---- pgfplots data: weights (clean, states sorted by employment rate) and sign-flip curves
    wm = ['multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']
    for ds in ['synthetic', 'acs']:
        idx = order if ds == 'acs' else range(4)
        with open(RES / f'weights_{ds}.dat', 'w') as fh:
            fh.write('client ' + ' '.join(wm) + '\n')
            for x, j in enumerate(idx):
                fh.write(f'{x + 1} ' + ' '.join(f"{S[(ds, 'none', m)]['weights'][j]:.4f}" for m in wm) + '\n')
    numbers['state_order'] = [STATES[j] for j in order]
    curves = json.loads((RES / 'curves.json').read_text())
    for ds in ['synthetic', 'acs']:
        acc = {m: [] for m in MAIN}
        for key, v in curves.items():
            run_ds, fam, att, meth, seed = key.split('|')
            if run_ds == ds and fam == 'signflip' and meth in acc:
                acc[meth].append(v)
        with open(RES / f'curves_signflip_{ds}.dat', 'w') as fh:
            fh.write('round ' + ' '.join(MAIN) + '\n')
            for t in range(rounds_of(ds)):
                fh.write(f'{t + 1} ' + ' '.join(f'{np.mean([c[t] for c in acc[m]]):.4f}' for m in MAIN) + '\n')

    # ---- prototype audit and numbers quoted in the text
    audit = json.loads((RES / 'original_fallback_audit.json').read_text())
    orig = dict(fallback_rounds_attack=sum(a['fallback_rounds'] for a in audit if a['attack'] == 1),
                fallback_rounds_clean=sum(a['fallback_rounds'] for a in audit if a['attack'] == 0),
                mean_attacker_weight={m: float(np.mean([a['attacker_weight'] for a in audit if a['attack'] == 1
                                                        and a['method'] == m])) for m in ['trusted', 'trustfair']},
                clean_weights_trusted=np.mean([a['weights'] for a in audit if a['attack'] == 0 and
                                               a['method'] == 'trusted'], 0).round(3).tolist())
    for ds in ['synthetic', 'acs']:
        r = runs[runs.dataset == ds]
        orig[f'share_support_lower_recall_{ds}'] = float((r['recall_support'] <= r['recall_majority'] + 1e-12).mean())
    (RES / 'original_audit_summary.json').write_text(json.dumps(orig, indent=1))
    numbers['prototype_audit'] = orig
    # quantities quoted in the text that no table holds
    numbers['trust_weighted_employment_rate'] = {
        m: float(sum(w * rate[STATES[j]] for j, w in enumerate(S[('acs', 'none', m)]['weights'])))
        for m in ['fedavg', 'multikrum', 'fltrust', 'fltrust_noscale', 'fedval', 'anchor', 'cpq', 'cpqj']}
    numbers['joint_checks_per_round'] = {ds: float(S[(ds, 'none', 'cpqj')]['evals_per_round']
                                               - 2 * S[(ds, 'none', 'cpqj')]['lowtrust_per_round'])
                                         for ds in ['synthetic', 'acs', 'acs-puma']}
    by_att = {}
    for fam in ['signflip', 'liar']:
        x = runs[(runs.dataset == 'acs') & (runs.method == 'cpqj') & (runs.family == fam)].copy()
        x['att'] = x['attackers'].astype(int)
        x['att_readmitted'] = [r[f"ra{r['att']}"] for _, r in x.iterrows()]
        g = x.groupby('att')[['att_readmitted', 'f1', 'recall_support', 'fpr_support']].mean()
        by_att[fam] = {STATES[a]: {k: float(v) for k, v in row.items()} for a, row in g.iterrows()}
    numbers['cpqj_acs_by_attacker'] = by_att
    pu = S[('acs-puma', 'signflip', 'cpqj')]
    numbers['puma_cpqj_signflip_readmissions_per_attacker'] = pu['attacker_readmitted'] / 9
    (RES / 'text_numbers.json').write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:3000])


if __name__ == '__main__':
    main()

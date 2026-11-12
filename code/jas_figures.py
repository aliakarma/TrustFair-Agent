"""Plot data for Figures 3 and 4 of the JAS manuscript: mean aggregation weight per institution with the standard and
the centered cosine, clean condition, mean over 20 seeds. Run after the experiments (TF_RES as for jas_analysis.py):

    TF_RES=results/revision_jas python code/jas_figures.py

Writes TF_RES/weights_center_acs.dat (six states, sorted by employment rate) and TF_RES/puma_center.dat (89 PUMAs).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('TF_RES', ROOT / 'results' / 'revision_jas'))
ST = ['wv', 'me', 'nm', 'ut', 'nd', 'ms']  # client order of the state federation (revision_experiments.ACS_CLIENTS)
RULES = ['fltrust', 'anchor', 'cpqj']


def mean_weights(d, ds, meth, K):
    """Seed mean of the per-run mean normalized weights, clean condition."""
    x = d[(d.dataset == ds) & (d.family == 'none') & (d.method == meth)]
    assert x.seed.nunique() == 20, (ds, meth, x.seed.nunique())
    return x.groupby('seed')[[f'w{k}' for k in range(K)]].mean().mean().to_numpy()


def main():
    std, cen = pd.read_csv(RES / 'runs.csv'), pd.read_csv(RES / 'runs_center.csv')
    emp = {s: float(np.load(ROOT / 'data' / 'acs' / f'acs_{s}.npz')['y'].mean()) for s in ST}
    order = sorted(range(6), key=lambda j: emp[ST[j]])
    cols = {f'{m}_std': mean_weights(std, 'acs', m, 6) for m in RULES}
    cols |= {f'{m}_cen': mean_weights(cen, 'acs-center', m, 6) for m in RULES}
    # consistency with the weight file that revision_analysis.py writes for the supplement
    old = pd.read_csv(RES / 'weights_acs.dat', sep=' ')
    for m in RULES:
        assert np.allclose(old[m].to_numpy(), cols[f'{m}_std'][order], atol=6e-5), m
    # TikZ label anchor (angle) per state for Figure 3: WV, NM and MS lie within 0.02 in employment rate, so their
    # labels go left (0 = east anchor), below (90), and up-right (225); ME up-left (315), ND above (270), UT up-right
    lpos = {'WV': 0, 'NM': 90, 'MS': 225, 'ME': 315, 'ND': 270, 'UT': 225}
    with open(RES / 'weights_center_acs.dat', 'w') as fh:
        fh.write('client state lpos employment ' + ' '.join(cols) + '\n')
        for i, j in enumerate(order, 1):
            s = ST[j].upper()
            fh.write(f'{i} {s} {lpos[s]} {emp[ST[j]]:.4f} ' + ' '.join(f'{v[j]:.4f}' for v in cols.values()) + '\n')

    meta = json.loads((RES / 'puma_meta.json').read_text())
    K = len(meta)
    pstd, pcen = pd.read_csv(RES / 'runs_puma.csv'), pd.read_csv(RES / 'runs_center_puma.csv')
    pcols = {f'{m}_std': K * mean_weights(pstd, 'acs-puma', m, K) for m in RULES}
    pcols |= {f'{m}_cen': K * mean_weights(pcen, 'acs-puma-center', m, K) for m in RULES}
    old = pd.read_csv(RES / 'puma_scatter.dat', sep=' ')
    for m in RULES:
        assert np.allclose(old[m].to_numpy(), pcols[f'{m}_std'], atol=6e-4), m
    with open(RES / 'puma_center.dat', 'w') as fh:
        fh.write('employment prevalence ' + ' '.join(pcols) + '\n')
        for i, mm in enumerate(meta):
            fh.write(f"{mm['employment']:.4f} {mm['prevalence']:.4f} " + ' '.join(f'{v[i]:.4f}' for v in pcols.values()) + '\n')
    print(open(RES / 'weights_center_acs.dat').read())
    print('PUMA lowest relative weight:', {k: round(float(v.min()), 3) for k, v in pcols.items()})


if __name__ == '__main__':
    main()

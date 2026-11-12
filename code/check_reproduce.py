"""Check that the JAS-revision code reproduces the original results/revision/runs.csv for a sample of runs.
Run from the project root:  python code/check_reproduce.py   (exit 1 on any mismatch)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
runs = pd.read_csv(ROOT / 'results' / 'revision' / 'runs.csv')
COLS = ['f1', 'recall_support', 'fpr_support', 'auc', 'mean_prob', 'tpr_gap', 'w0', 'w1', 'w2', 'w3']
cases = [('acs', 1001, 'none', (), 'fedavg'), ('acs', 1001, 'none', (), 'cpqj'), ('acs', 1002, 'signflip', (2,), 'cpqj'),
         ('acs', 1001, 'none', (), 'fltrust'), ('acs', 1003, 'liar', (1,), 'cpqj'), ('acs', 1001, 'none', (), 'fedval'),
         ('synthetic', 1001, 'signflip', (3,), 'anchor'), ('synthetic', 1004, 'none', (), 'cpq')]
bad = 0
for ds, seed, fam, att, meth in cases:
    cl, anc = rx.synth(seed) if ds == 'synthetic' else rx.acs(seed)
    m, _, log = rx.run_one(cl, anc, meth, att, rx.kind_of(fam), 16 if ds == 'synthetic' else 30)
    row = rx.make_row(ds, seed, fam, att, meth, m, log, len(cl))
    ref = runs[(runs.dataset == ds) & (runs.seed == seed) & (runs.family == fam) & (runs.method == meth)
               & (runs.attackers.astype(str) == ('-'.join(map(str, att)) or 'none'))].iloc[0]
    for c in COLS:
        if not np.isclose(row[c], ref[c], rtol=0, atol=1e-12):
            print('MISMATCH', ds, seed, fam, meth, c, row[c], ref[c]); bad += 1
    print(ds, seed, fam, meth, 'f1', round(row['f1'], 4), 'rel_f1', round(row['rel_f1'], 4), 'gate', row['gate_blocks'])
print('mismatches:', bad)
sys.exit(1 if bad else 0)

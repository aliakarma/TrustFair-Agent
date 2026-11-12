"""Build the ACS PUMS employment benchmark used in the revision (public census data, no private records).

Input : data/acs/raw/csv_p<state>.zip, ACS 2018 1-year person files downloaded from
        https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/ (hashes in data/acs/raw/SHA256SUMS).
Output: data/acs/acs_<state>.npz with X (model features), y (employed, ESR==1), g (DIS==1, audit only)
        and puma (public use microdata area, used only to split states into institutions).
Task  : the ACSEmployment definition of Ding et al. (2021), restricted to ages 16 to 90.
        The disability indicators DIS, DEAR, DEYE and DREM are kept out of X (audit-only labels).
Run   : python code/prepare_acs.py   (from the project root)
JAS revision: existing acs_<state>.npz files are never rewritten. The run also writes acs_<state>_w.npz (person
        weight PWGTP and the 80 replicate weights PWGTP1-80, same row order) and acs_<state>_anc.npz (the same
        features plus ancestry ANC, one-hot), and builds the six held-out states WY, AK, SD, DE, RI, MT.
"""
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'acs' / 'raw'
STATES = ['wv', 'me', 'nm', 'ut', 'nd', 'ms', 'vt', 'wy', 'ak', 'sd', 'de', 'ri', 'mt']
WCOLS = ['PWGTP'] + [f'PWGTP{i}' for i in range(1, 81)]
COLS = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'NATIVITY', 'SEX', 'RAC1P', 'ESR', 'PUMA']
# Fixed category lists so every state gets the same columns without sharing statistics between clients.
CATS = {'MAR': range(1, 6), 'RELP': range(0, 18), 'ESP': range(0, 9), 'CIT': range(1, 6),
        'MIG': range(0, 4), 'MIL': range(0, 5), 'NATIVITY': range(1, 3), 'SEX': range(1, 3), 'RAC1P': range(1, 10)}


def features(df, anc=False):
    age = df['AGEP'].to_numpy(float) / 100.0
    schl = df['SCHL'].fillna(0).to_numpy(float) / 24.0
    parts = [age, age ** 2, schl]
    for col, levels in CATS.items():
        v = df[col].fillna(0).to_numpy(int)
        parts += [(v == lv).astype(float) for lv in list(levels)[1:]]  # first level is the reference
    if anc:  # ancestry recode of folktables' ACSEmployment: 1 single, 2 multiple, 3 unclassified, 4 not reported, 8 suppressed
        v = df['ANC'].fillna(0).to_numpy(int)
        parts += [(v == lv).astype(float) for lv in (2, 3, 4, 8)]
    parts.append(np.ones(len(df)))  # bias last, as grad() in experiment.py expects
    return np.column_stack(parts)


def main():
    for s in STATES:
        with zipfile.ZipFile(RAW / f'csv_p{s}.zip') as z:
            name = [n for n in z.namelist() if n.endswith('.csv')][0]
            df = pd.read_csv(z.open(name), usecols=COLS + ['ANC'] + WCOLS)
        df = df[(df['AGEP'] >= 16) & (df['AGEP'] <= 90)].reset_index(drop=True)
        X = features(df)
        y = (df['ESR'] == 1).to_numpy(float)
        g = (df['DIS'] == 1).to_numpy(int)
        assert X.shape[0] == y.shape[0] == g.shape[0] and np.isfinite(X).all()
        base = ROOT / 'data' / 'acs' / f'acs_{s}.npz'
        if base.exists():  # never rewrite a file the published results were computed from; check it instead
            d = np.load(base)
            assert np.array_equal(d['X'], X) and np.array_equal(d['y'], y) and np.array_equal(d['g'], g), s
        else:
            np.savez_compressed(base, X=X, y=y, g=g, puma=df['PUMA'].to_numpy(int))
        np.savez_compressed(ROOT / 'data' / 'acs' / f'acs_{s}_w.npz', w=df['PWGTP'].to_numpy(float),
                            rw=df[WCOLS[1:]].to_numpy(np.float32))
        np.savez_compressed(ROOT / 'data' / 'acs' / f'acs_{s}_anc.npz', X=features(df, anc=True), y=y, g=g,
                            puma=df['PUMA'].to_numpy(int))
        print(s, 'sha256', hashlib.sha256((RAW / f'csv_p{s}.zip').read_bytes()).hexdigest(),
              'weighted population', int(df['PWGTP'].sum()))
        print(f'{s}: n={len(y)} d={X.shape[1]} employed={y.mean():.3f} disability={g.mean():.3f} '
              f'employed|dis={y[g == 1].mean():.3f} pumas={df["PUMA"].nunique()}')


if __name__ == '__main__':
    main()

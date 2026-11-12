"""Experiments for TrustFair-Agent (synthetic proxy, ACS PUMS state federation, ACS PUMA federation).

Relative to code/experiment.py (the FedAgent-Chain prototype script, kept unchanged in behavior):
  * the audit-only disability label g is removed from the model input, but kept for auditing;
  * 20 seeds fixed in advance, attacker identity rotated over every client;
  * robust baselines: coordinate median, trimmed mean, Multi-Krum, FLTrust, FedVal;
  * the prototype anchor rule keeps theta when every trust weight is zero (the prototype used uniform weights);
  * attacks: sign-flip, targeted support-group suppression, min-max (Shejwalkar and Houmansadr 2021),
    a positive-push attack aimed at the coverage test, a sign-flip attacker that lies in validation reports,
    and a two-attacker sign-flip stress test;
  * coverage-preserving quarantine with a recall test (cpq) and with a balanced test (cpqj): paired flip counts,
    a Bonferroni z bound on the per-group gain in TPR - FPR, and a joint check of simultaneous readmissions;
  * ablation of the fixed 1.5 support-group weight and the local intercept;
  * centralized, local-only and Bayes-rule references;
  * anchor sensitivity, base-rate and prevalence matching, an 89-institution PUMA federation and an MLP learner.

Run from the project root:  python code/revision_experiments.py [--part audit,main,anchor,match,puma,mlp,sens,g]
Outputs go to results/revision/. Figures are drawn by pgfplots from the .dat files written by revision_analysis.py.
"""
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import sigmoid, gen, grad, train, f1  # noqa: E402  (prototype generator and learner)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('TF_OUT', ROOT / 'results' / 'revision'))  # JAS revision: TF_OUT=results/revision_jas
SEEDS = list(range(1001, 1021))          # 20 seeds, fixed before any revision run
ORIG_SEEDS = [41, 107, 233, 509, 811]    # seeds of the prototype
TUNE_SEED = 999                          # used only to choose the MLP learning rate, never reported
G_COL = 4                                # column of g in experiment.gen() output
MIN_CELL = 20                            # minimum validation records per cell before a group statistic is used
CPQ = dict(c=2.0, a0=0.2, eta1=10.0, eta2=5.0, tau=0.5, nu=0.01, delta=0.01, z=1.96)
FEDVAL = dict(s1=3.0, C=3.0, s1_avg=5.0, s2=3.0, s2_group=30.0)  # values of Valadi et al. (2023)
PUSH = 3.0                               # positive-push attack: bias shift in units of the attacker's update norm
MODEL = dict(kind='logreg', hidden=16, lr=0.25)
ACS_CLIENTS = ['wv', 'me', 'nm', 'ut', 'nd', 'ms']
ACS_ANCHOR = 'vt'
VALIDATION_RULES = ['cpq', 'cpqj']      # the only rules that read client validation reports
# Variants added for the JAS revision. Defaults reproduce the original runs exactly.
#   nobias: trust cosine computed without the intercept coordinate (last parameter)
#   prior: anchor records reweighted so the anchor's employment rate equals the federation's pooled rate
#   guard: 'point' (Eq. 7) or 'ci' (readmit only if every group's lower bound dJ - z se is at least -delta)
#   fedval_mad: 'clients' (MAD over client mean losses) or 'samples' (Eq. 4 of Valadi et al.: per-sample deviations
#               from the dimension's average loss, summed and divided by the number of clients)
#   test_all: diagnostic, test every institution's inclusion each round (null calibration of the balanced test)
#   data: suffix of the ACS files ('' main, '_anc' with ancestry)
FLAGS = dict(nobias=False, center=False, prior=False, guard='point', fedval_mad='clients', test_all=False, data='')
GATE_Z = 1.96                            # release gate: one-sided Bonferroni bound over the two groups at 5%


# ----------------------------------------------------------------------------- learner
def _unpack(w, d):
    H = MODEL['hidden']
    return w[:d * H].reshape(d, H), w[d * H:d * H + H], w[-1]


def logit(w, X):
    if MODEL['kind'] == 'logreg':
        return X @ w
    W1, w2, b2 = _unpack(w, X.shape[1])
    return np.maximum(X @ W1, 0.) @ w2 + b2


def grad_model(w, X, y, weights=None):
    if MODEL['kind'] == 'logreg':
        return grad(w, X, y, weights)
    W1, w2, b2 = _unpack(w, X.shape[1])
    hp = X @ W1; h = np.maximum(hp, 0.)
    e = sigmoid(h @ w2 + b2) - y
    if weights is not None:
        e = e * weights / np.mean(weights)
    e = e / len(y)
    gW1 = X.T @ (np.outer(e, w2) * (hp > 0)); gW1[:-1] += 0.004 * W1[:-1]  # ridge, bias row of X excluded
    return np.concatenate([gW1.ravel(), h.T @ e + 0.004 * w2, [e.sum()]])


def train_model(w, X, y, steps=6, weights=None):
    if MODEL['kind'] == 'logreg':
        return train(w, X, y, steps=steps, weights=weights)
    w = w.copy()
    for _ in range(steps):
        w -= MODEL['lr'] * grad_model(w, X, y, weights)
    return w


def init_params(d, seed):
    if MODEL['kind'] == 'logreg':
        return np.zeros(d)
    H, rng = MODEL['hidden'], np.random.default_rng(seed + 7)
    return np.concatenate([rng.normal(0, 1 / np.sqrt(d), d * H), rng.normal(0, 1 / np.sqrt(H), H), [0.]])


# ----------------------------------------------------------------------------- data
def oracle_prob(xg, k):
    """Label probability of the synthetic oracle (same rule as experiment.gen)."""
    s, g, lang, needs, acc, accb = xg[:, 0:4], xg[:, 4], xg[:, 5], xg[:, 6], xg[:, 7], xg[:, 8]
    score = 1.05 * s[:, 0] + .70 * s[:, 1] - .42 * s[:, 2] + .35 * s[:, 3]
    score += .55 * acc + .50 * accb - .65 * needs * (1 - acc) - .25 * lang * (1 - accb)
    score += .20 * g + .12 * k - 0.2
    return sigmoid(score)


def synth(seed, anchor='client2', keep_g=False):
    """Same random stream as experiment.one(): train/test per client, then the anchor, then validation sets."""
    rng = np.random.default_rng(seed)
    cut = (lambda x: x) if keep_g else (lambda x: np.delete(x, G_COL, 1))
    cl = []
    for k in range(4):
        x, y, g = gen(rng, 480, k)
        xt, yt, gt = gen(rng, 220, k)
        cl.append(dict(xtr=cut(x), ytr=y, gtr=g.astype(int), xte=cut(xt), yte=yt, gte=gt.astype(int),
                       bayes=(oracle_prob(xt, k) >= .5).astype(int)))
    ax, ay, ag = gen(rng, 360, 1)  # anchor follows client 2's distribution, as in the prototype
    for k, c in enumerate(cl):
        x, y, g = gen(rng, 120, k)
        c.update(xva=cut(x), yva=y, gva=g.astype(int))
    if anchor == 'mixture':  # sensitivity: anchor drawn equally from all four client distributions
        parts = [gen(rng, 90, k) for k in range(4)]
        ax, ay, ag = (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                      np.concatenate([p[2] for p in parts]))
    return cl, (cut(ax), ay, ag.astype(int))


_ACS = {}


def _acs(state):
    key = state + FLAGS['data']
    if key not in _ACS:
        d = np.load(ROOT / 'data' / 'acs' / f'acs_{key}.npz')
        _ACS[key] = (d['X'], d['y'], d['g'].astype(int), d['puma'])
    return _ACS[key]


def _acs_weights(state):
    """Person weight PWGTP of each record (same row order as acs_<state>.npz), written by prepare_acs.py --weights."""
    return np.load(ROOT / 'data' / 'acs' / f'acs_{state}_w.npz')['w']


def _logit_p(p):
    return np.log(p / (1 - p))


def matched_index(rng, y, g, n, prev, rate):
    """Stratified sample of n records with disability prevalence prev and employment rate rate. Both groups'
    employment log-odds shift by one constant, so the state's disability employment gap keeps its log-odds size."""
    p1, p0 = y[g == 1].mean(), y[g == 0].mean()
    lo, hi = -10., 10.
    for _ in range(60):
        c = (lo + hi) / 2
        r = prev * sigmoid(_logit_p(p1) + c) + (1 - prev) * sigmoid(_logit_p(p0) + c)
        lo, hi = (c, hi) if r < rate else (lo, c)
    q1, q0 = sigmoid(_logit_p(p1) + c), sigmoid(_logit_p(p0) + c)
    n1 = int(round(n * prev)); e1 = int(round(n1 * q1)); e0 = int(round((n - n1) * q0))
    counts = {(1, 1): e1, (1, 0): n1 - e1, (0, 1): e0, (0, 0): n - n1 - e0}
    idx = np.concatenate([rng.choice(np.where((g == a) & (y == b))[0], m, replace=False)
                          for (a, b), m in counts.items()])
    assert abs(g[idx].mean() - prev) <= 1 / n and abs(y[idx].mean() - rate) <= 2 / n
    return rng.permutation(idx)


def match_target(state, match):
    """'rate': keep the state's prevalence, set its employment rate to the anchor's.
       'prev': set every state to the mean prevalence of the six, keep its employment rate."""
    y, g = _acs(state)[1:3]
    if match == 'rate':
        return g.mean(), _acs(ACS_ANCHOR)[1].mean()
    return np.mean([_acs(s)[2].mean() for s in ACS_CLIENTS]), y.mean()


def acs(seed, anchor=ACS_ANCHOR, match=None, ntr=2000, nte=1000, nva=500, nanchor=500, weighted=False):
    """Six states as institutions. anchor: 'vt' (Vermont, main), a client state (500 of its records held out from
    every split), or 'pooled' (about 83 held-out records from each client state). match: None, 'rate' or 'prev'.
    weighted: attach the ACS person weights (training, test and anchor); the random splits are unchanged."""
    rng = np.random.default_rng(seed)
    cl, perms, n0 = [], {}, ntr + nte + nva
    for s in ACS_CLIENTS:
        X, y, g, _ = _acs(s)
        i = rng.permutation(len(y)) if match is None else matched_index(rng, y, g, n0, *match_target(s, match))
        perms[s] = i
        tr, te, va = i[:ntr], i[ntr:n0 - nva], i[n0 - nva:n0]
        cl.append(dict(xtr=X[tr], ytr=y[tr], gtr=g[tr], xte=X[te], yte=y[te], gte=g[te],
                       xva=X[va], yva=y[va], gva=g[va]))
        if weighted:
            W = _acs_weights(s); cl[-1].update(swtr=W[tr], swte=W[te])
    if anchor == ACS_ANCHOR:
        X, y, g, _ = _acs(ACS_ANCHOR)
        i = rng.permutation(len(y))[:nanchor]
        return cl, ((X[i], y[i], g[i], _acs_weights(ACS_ANCHOR)[i]) if weighted else (X[i], y[i], g[i]))
    if anchor == 'pooled':
        sizes = np.diff(np.linspace(0, nanchor, len(ACS_CLIENTS) + 1).round().astype(int))
        idx = {s: perms[s][n0:n0 + m] for s, m in zip(ACS_CLIENTS, sizes)}
    else:
        idx = {anchor: perms[anchor][n0:n0 + nanchor]}
    parts = [tuple(a[i] for a in _acs(s)[:3]) for s, i in idx.items()]
    return cl, tuple(np.concatenate([p[j] for p in parts]) for j in range(3))


def acs_puma(seed, nanchor=500):
    """Every PUMA of the six states is an institution (89 in total): 50% train, 30% test, 20% validation."""
    rng = np.random.default_rng(seed)
    cl = []
    for s in ACS_CLIENTS:
        X, y, g, pu = _acs(s)
        for p in np.unique(pu):
            j = rng.permutation(np.where(pu == p)[0])
            a, b = int(.5 * len(j)), int(.8 * len(j))
            cl.append(dict(xtr=X[j[:a]], ytr=y[j[:a]], gtr=g[j[:a]], xte=X[j[a:b]], yte=y[j[a:b]], gte=g[j[a:b]],
                           xva=X[j[b:]], yva=y[j[b:]], gva=g[j[b:]]))
    X, y, g, _ = _acs(ACS_ANCHOR)
    i = rng.permutation(len(y))[:nanchor]
    return cl, (X[i], y[i], g[i])


def acs_iid(seed, K=6, ntr=2000, nte=1000, nva=500, nanchor=500):
    """Null federation: the six client states' records pooled and dealt at random to K institutions and the anchor,
    so institutions and anchor share one distribution and no institution differs in base rate."""
    rng = np.random.default_rng(seed)
    X, y, g = (np.concatenate([_acs(s)[j] for s in ACS_CLIENTS]) for j in range(3))
    i, n0, cl = rng.permutation(len(y)), ntr + nte + nva, []
    for k in range(K):
        j = i[k * n0:(k + 1) * n0]
        tr, te, va = j[:ntr], j[ntr:ntr + nte], j[ntr + nte:]
        cl.append(dict(xtr=X[tr], ytr=y[tr], gtr=g[tr], xte=X[te], yte=y[te], gte=g[te],
                       xva=X[va], yva=y[va], gva=g[va]))
    a = i[K * n0:K * n0 + nanchor]
    return cl, (X[a], y[a], g[a])


def disclose(cl, anc, rng, scen):
    """Voluntary disclosure of the audit label. A person with a disability (g = 1) discloses with probability p and an
    undisclosed person is recorded as g = 0, because the system cannot tell them apart from persons without a
    disability. Applies to validation reports, the anchor and the audited test labels; the true test labels are kept in
    gte for the true metrics. scen: ('mcar', p) or ('age', None), where p rises linearly from 0.2 at age 16 to 1.0 at
    age 90 (AGEP/100 is column 0 of X)."""
    def mask(X, g):
        p = np.full(len(g), scen[1]) if scen[0] == 'mcar' else 0.2 + 0.8 * (X[:, 0] * 100 - 16) / 74
        return g * (rng.random(len(g)) < p)
    out = []
    for c in cl:
        c = dict(c); c['gva'] = mask(c['xva'], c['gva']); c['gte_aud'] = mask(c['xte'], c['gte']); out.append(c)
    return out, (anc[0], anc[1], mask(anc[0], anc[2]))


def puma_meta():
    out = []
    for s in ACS_CLIENTS:
        _, y, g, pu = _acs(s)
        for p in np.unique(pu):
            m = pu == p
            out.append(dict(state=s, puma=int(p), n=int(m.sum()), employment=float(y[m].mean()),
                            prevalence=float(g[m].mean())))
    return out


# ----------------------------------------------------------------------------- evaluation
def auc(s, y):
    """Mann-Whitney AUC with average ranks for ties."""
    y = np.asarray(y) == 1
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = np.empty(len(s)); ranks[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


assert auc(np.array([.1, .4, .35, .8]), np.array([0, 0, 1, 1])) == .75
assert auc(np.array([.5, .5]), np.array([0, 1])) == .5


def evaluate(w, clients, adapters=None, preds=None, gkey='gte'):
    """F1, within-client TPR gap and worst-client F1 as in experiment.metric(), plus pooled group TPR and FPR,
    positive rate, AUC, mean predicted probability, Brier score, log loss and calibration-in-the-large (mean predicted
    probability minus observed rate). With survey weights (c['swte']) the group rates, F1, Brier score and log loss are
    weighted. gkey selects the audit labels (gte true, gte_aud as disclosed)."""
    ys, ps, ss, gs, locs, gaps, ws = [], [], [], [], [], [], []
    tp, pos, fp, neg = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    for k, c in enumerate(clients):
        if preds is not None:
            p, sc = preds[k], np.full(len(c['yte']), np.nan)
        else:
            b = 0. if adapters is None else adapters[k]
            sc = sigmoid(logit(w if np.ndim(w) == 1 else w[k], c['xte']) + b)
            p = (sc >= .5).astype(int)
        y, g = c['yte'], c[gkey]
        wt = c.get('swte', np.ones(len(y)))
        locs.append(f1(y, p)); ys.append(y); ps.append(p); ss.append(sc); gs.append(g); ws.append(wt)
        a, b1 = (y == 1) & (g == 0), (y == 1) & (g == 1)
        if a.sum() > 0 and b1.sum() > 0:
            gaps.append(abs(np.average(p[a], weights=wt[a]) - np.average(p[b1], weights=wt[b1])))
        for gv, m in ((0, a), (1, b1)):
            tp[gv] += (p * wt)[m].sum(); pos[gv] += wt[m].sum()
        for gv in (0, 1):
            m = (y == 0) & (g == gv)
            fp[gv] += (p * wt)[m].sum(); neg[gv] += wt[m].sum()
    rec, fpr = tp / np.maximum(pos, 1), fp / np.maximum(neg, 1)
    Y, Pp, S, G, Wt = map(np.concatenate, (ys, ps, ss, gs, ws))
    has = preds is None
    if 'swte' in clients[0]:  # weighted F1 from weighted counts
        TP, FP, FN = (Wt * (Y == 1) * Pp).sum(), (Wt * (Y == 0) * Pp).sum(), (Wt * (Y == 1) * (1 - Pp)).sum()
        F1 = float(2 * TP / (2 * TP + FP + FN + 1e-12))
    else:
        F1 = f1(Y, Pp)
    Sc = np.clip(S, 1e-12, 1 - 1e-12)
    return dict(f1=F1, tpr_gap=float(np.mean(gaps)), worst_client_f1=float(min(locs)),
                recall_support=float(rec[1]), recall_majority=float(rec[0]), worst_group_recall=float(rec.min()),
                fpr_support=float(fpr[1]), fpr_majority=float(fpr[0]), pos_rate=float(Pp.mean()),
                auc=auc(S, Y) if has else np.nan, auc_support=auc(S[G == 1], Y[G == 1]) if has else np.nan,
                mean_prob=float(S.mean()) if has else np.nan,
                brier=float(np.average((S - Y) ** 2, weights=Wt)) if has else np.nan,
                logloss=float(np.average(-(Y * np.log(Sc) + (1 - Y) * np.log(1 - Sc)), weights=Wt)) if has else np.nan,
                citl=float(np.average(S, weights=Wt) - np.average(Y, weights=Wt)) if has else np.nan,
                citl_support=float(S[G == 1].mean() - Y[G == 1].mean()) if has else np.nan)


def val_index(clients):
    """Validation splits stacked once per run; each record keeps the index of the institution that holds it."""
    return dict(X=np.vstack([c['xva'] for c in clients]), y=np.concatenate([c['yva'] for c in clients]).astype(int),
                g=np.concatenate([c['gva'] for c in clients]).astype(int), K=len(clients),
                cid=np.concatenate([np.full(len(c['yva']), k) for k, c in enumerate(clients)]))


def val_pred(w, V):
    return (sigmoid(logit(w, V['X'])) >= .5).astype(int)


def val_table(p, V, base=None):
    """Integer counts each institution reports for a candidate model. Without base: cells (g, y, prediction),
    8 per institution. With base: cells (g, y, base prediction, prediction), 16 per institution."""
    code = V['g'] * 4 + V['y'] * 2 + p if base is None else V['g'] * 8 + V['y'] * 4 + base * 2 + p
    m = 8 if base is None else 16
    return np.bincount(V['cid'] * m + code, minlength=V['K'] * m).reshape(V['K'], m)


def val_stats(T, skip, liars=frozenset(), lie=None):
    """Pooled group recall and F1 from the 8-cell reports of every institution outside skip. A lying institution
    reports every positive missed and every negative flagged (lie='low') or a perfect model (lie='high')."""
    T = T.copy()
    for j in set(liars) - set(skip):
        if lie is None:
            continue
        C = T[j].reshape(2, 2, 2); n = C.sum(2); C[:] = 0
        C[:, 1, 1 if lie == 'high' else 0] = n[:, 1]; C[:, 0, 0 if lie == 'high' else 1] = n[:, 0]
    S = np.delete(T, sorted(skip), 0).sum(0).reshape(2, 2, 2)  # [g, y, p]
    tp, pos = S[:, 1, 1], S[:, 1].sum(1)
    TP, FP, FN = tp.sum(), S[:, 0, 1].sum(), (pos - tp).sum()
    rec = np.where(pos >= MIN_CELL, tp / np.maximum(pos, 1), np.nan)
    return rec, 2 * TP / (2 * TP + FP + FN + 1e-12)


def flip_stats(T, skip, liars=frozenset()):
    """Paired comparison of a candidate with the base model from 16-cell reports. Returns the per-group change in
    TPR - FPR, its standard error from the discordant counts, the cells with enough records, and both F1 values.
    A lying institution reports that the candidate flips every positive to negative and every negative to positive."""
    T = T.copy()
    for j in set(liars) - set(skip):
        C = T[j].reshape(2, 2, 2, 2); n = C.sum((2, 3)); C[:] = 0
        C[:, 1, 1, 0] = n[:, 1]; C[:, 0, 0, 1] = n[:, 0]
    S = np.delete(T, sorted(skip), 0).sum(0).reshape(2, 2, 2, 2)  # [g, y, base, candidate]
    P, N = np.maximum(S[:, 1].sum((1, 2)), 1), np.maximum(S[:, 0].sum((1, 2)), 1)
    up_p, dn_p, up_n, dn_n = S[:, 1, 0, 1], S[:, 1, 1, 0], S[:, 0, 0, 1], S[:, 0, 1, 0]
    dJ = (up_p - dn_p) / P - (up_n - dn_n) / N
    se = np.sqrt((up_p + dn_p) / P ** 2 + (up_n + dn_n) / N ** 2)
    ok = (P >= MIN_CELL) & (N >= MIN_CELL)
    tp_n, fp_n, fn_n = S[:, 1, :, 1].sum(), S[:, 0, :, 1].sum(), S[:, 1, :, 0].sum()
    tp_b, fp_b, fn_b = S[:, 1, 1, :].sum(), S[:, 0, 1, :].sum(), S[:, 1, 0, :].sum()
    return (dJ, se, ok, 2 * tp_n / (2 * tp_n + fp_n + fn_n + 1e-12), 2 * tp_b / (2 * tp_b + fp_b + fn_b + 1e-12))


def fit_intercept(w, c):
    """Local adapter of the prototype: a client-local intercept fitted after training."""
    bias = 0.
    for _ in range(15):
        p = sigmoid(logit(w, c['xtr']) + bias)
        bias -= .20 * ((p - c['ytr']).mean() + 0.30 * bias)
    return bias


# ----------------------------------------------------------------------------- aggregation rules
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


CENTER = {}  # mean feature vector of the anchor, set by run_one when FLAGS['center']


def tcos(a, b):
    """Trust cosine. With FLAGS['nobias'] the intercept coordinate (last parameter) is left out. With FLAGS['center']
    the base-rate direction is removed first: u - u_b m, with u_b the intercept coordinate and m the anchor's mean
    feature vector (Proposition 1: a base-rate gap moves a logistic gradient along m)."""
    if FLAGS['center']:
        m = CENTER['m']; return cos(a - a[-1] * m, b - b[-1] * m)
    return cos(a[:-1], b[:-1]) if FLAGS['nobias'] else cos(a, b)


def anchor_weights(ctx):
    """Record weights of the anchor: survey weights if attached, times the prior-shift factor if FLAGS['prior']
    (pi / pi_0 for employed and (1 - pi) / (1 - pi_0) for not employed, pi the federation's pooled training rate)."""
    anc = ctx['anchor']; ay = anc[1]
    aw = anc[3] if len(anc) > 3 else np.ones(len(ay))
    if FLAGS['prior']:
        pi = ctx['fed_rate']; pi0 = np.average(ay, weights=aw)
        aw = aw * np.where(ay == 1, pi / pi0, (1 - pi) / (1 - pi0))
    return None if (len(anc) == 3 and not FLAGS['prior']) else aw


def anchor_update(w, ctx):
    ax, ay = ctx['anchor'][:2]
    return train_model(w, ax, ay, weights=anchor_weights(ctx)) - w  # FLTrust server update: same local steps as clients


def agg_fedavg(w, U, ctx):
    return U.mean(0), np.full(len(U), 1 / len(U))


def agg_median(w, U, ctx):
    return np.median(U, 0), None


def agg_trimmed(w, U, ctx):
    b = ctx['f']
    return np.sort(U, 0)[b:len(U) - b].mean(0), None


def agg_multikrum(w, U, ctx):
    K, f = len(U), ctx['f']
    d = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    nb = max(K - f - 2, 1)
    score = np.array([np.sort(np.delete(d[i], i))[:nb].sum() for i in range(K)])
    sel = np.argsort(score)[:K - f]
    wts = np.zeros(K); wts[sel] = 1 / len(sel)
    return U[sel].mean(0), wts


def agg_fltrust(w, U, ctx, rescale=True):
    r = anchor_update(w, ctx); nr = np.linalg.norm(r)
    ts = np.array([max(0., tcos(u, r)) for u in U])
    if ts.sum() < 1e-12:
        ctx['log']['skips'] += 1
        return np.zeros_like(w), np.zeros(len(U))
    scaled = U * (nr / (np.linalg.norm(U, axis=1) + 1e-12))[:, None] if rescale else U
    return (ts[:, None] * scaled).sum(0) / ts.sum(), ts / ts.sum()


def agg_fltrust_noscale(w, U, ctx):
    """Mechanism check: FLTrust trust scores without rescaling each update to the anchor update's norm."""
    return agg_fltrust(w, U, ctx, rescale=False)


def logloss(w, X, y):
    p = sigmoid(logit(w, X))
    return -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


def agg_fedval(w, U, ctx):
    """FedVal (Valadi et al. 2023, Eqs. 2, 3, 5, 6 and Algorithm 1) on the server's anchor data. Dimensions: the two
    labels (adaptive s2), the positives of each audit group (the paper's group-recall extension, s2 fixed at 30) and
    the overall loss (s1_avg). MAD_k is read as the mean absolute deviation of client losses on dimension k."""
    F = FEDVAL; ax, ay, ag = ctx['anchor'][:3]  # anchor survey weights, if any, are not used by FedVal
    L = np.array([logloss(w + u, ax, ay) for u in U])  # clients x anchor records
    dims = [(ay == 0, None), (ay == 1, None), ((ay == 1) & (ag == 0), F['s2_group']),
            ((ay == 1) & (ag == 1), F['s2_group'])]
    Lavg = L.mean()

    def score(s2):
        S = np.zeros(len(U))
        for m, fixed in dims:
            if m.sum() == 0:
                continue
            Lk = L[:, m].mean(1); div = Lk.mean() - Lk
            mad = (np.abs(div).mean() if FLAGS['fedval_mad'] == 'clients'
                   else np.abs(L[:, m] - L[:, m].mean()).sum() / len(U)) + 1e-12
            S += max(1., Lk.mean() / Lavg) ** (s2 if fixed is None else fixed) * (F['s1'] * div / mad + F['C'] * F['s1'])
        La = L.mean(1); div = La.mean() - La
        mad = np.abs(div).mean() if FLAGS['fedval_mad'] == 'clients' else np.abs(L - L.mean()).sum() / len(U)
        return S + F['s1_avg'] * div / (mad + 1e-12) + F['C'] * F['s1_avg']

    best = None
    s0 = ctx['s2']
    for s2 in (s0, s0 + .5, s0 - .5, s0 + 5, s0 - 5):
        wts = np.maximum(0., score(s2))
        if wts.sum() <= 0:
            continue
        delta = (wts[:, None] * U).sum(0) / wts.sum()
        loss = logloss(w + delta, ax, ay).mean()
        if best is None or loss < best[0]:
            best = (loss, s2, delta, wts)
    if best is None:
        ctx['log']['skips'] += 1
        return np.zeros_like(w), np.zeros(len(U))
    ctx['s2'] = best[1]
    return best[2], best[3] / best[3].sum()


def agg_anchor(w, U, ctx):
    """Prototype rule of experiment.one(): single anchor gradient, ReLU cosine, norm cap c=2."""
    ax, ay = ctx['anchor'][:2]
    ref = -grad_model(w, ax, ay, anchor_weights(ctx)); nref = np.linalg.norm(ref)
    ws = np.array([max(0, tcos(u, ref)) * min(1., 2. / max(np.linalg.norm(u) / (nref + 1e-8), 1e-8)) for u in U])
    if ws.sum() < 1e-8:
        ctx['log']['skips'] += 1
        if ctx['fallback'] == 'uniform':  # prototype behavior: plain FedAvg, attacker included
            ws[:] = 1
        else:                             # revision: keep theta for this round
            return np.zeros_like(w), np.zeros(len(U))
    return np.average(U, axis=0, weights=ws), ws / ws.sum()


def cpq_trust(w, U, ctx):
    """Trust score Q_k from anchor alignment and norm anomaly, clipped updates and the accepted weights."""
    P = CPQ
    r = anchor_update(w, ctx); nr = np.linalg.norm(r)
    norms = np.linalg.norm(U, axis=1)
    A = np.array([tcos(u, r) for u in U])
    anomaly = np.maximum(0., np.log(np.maximum(norms / (nr + 1e-12), 1e-12)) - np.log(P['c']))
    Q = sigmoid(P['eta1'] * (A - P['a0']) - P['eta2'] * anomaly)
    clipped = U * np.minimum(1., P['c'] * nr / (norms + 1e-12))[:, None]
    return Q, clipped, np.where(Q >= P['tau'], Q, 0.)


def agg_cpq(w, U, ctx):
    """Coverage-preserving quarantine with the recall test (the first version). A low-trust client is quarantined
    when adding it raises some group's validation recall by more than nu, and readmitted if it lowers neither pooled
    validation F1 nor any group's recall by more than delta. Readmissions are tested one at a time."""
    P, log, K, V = CPQ, ctx['log'], len(U), ctx['val']
    Q, clipped, wts = cpq_trust(w, U, ctx)

    def cand(weights):
        return w + ((weights[:, None] * clipped).sum(0) / weights.sum() if weights.sum() > 0 else 0.)

    base = cand(wts); Tb = val_table(val_pred(base, V), V); final = wts.copy()
    for k in np.where(Q < P['tau'])[0]:
        log['lowtrust'][k] += 1
        trial = wts.copy(); trial[k] = P['tau']
        r_with, f_with = val_stats(val_table(val_pred(cand(trial), V), V), {k}, ctx['liars'], 'low')
        r_wo, f_wo = val_stats(Tb, {k}, ctx['liars'], 'high')
        diff = r_with - r_wo
        Vk = float(np.nanmax(np.maximum(diff, 0))) if np.isfinite(diff).any() else 0.
        log['evals'] += 2
        if Vk > P['nu']:
            log['quarantined'][k] += 1
            if f_with >= f_wo - P['delta'] and np.isfinite(diff).any() and np.nanmin(diff) >= -P['delta']:
                final[k] = P['tau']; log['readmitted'][k] += 1
    readm = [k for k in range(K) if final[k] > 0 and wts[k] == 0]
    if len(readm) > 1:  # diagnostic only, with truthful reports: does the joint readmission respect delta?
        log['multi_readmit'] += 1
        r_j, f_j = val_stats(val_table(val_pred(cand(final), V), V), set(readm))
        r_b, f_b = val_stats(Tb, set(readm))
        d = r_j - r_b
        if f_j < f_b - P['delta'] or (np.isfinite(d).any() and np.nanmin(d) < -P['delta']):
            log['joint_violation'] += 1
    if final.sum() == 0:
        log['skips'] += 1
        return np.zeros_like(w), np.zeros(K)
    return cand(final) - w, final / final.sum()


def agg_cpqj(w, U, ctx):
    """Coverage-preserving quarantine with the balanced test. Each reporting institution compares the base model
    with the candidate record by record and returns discordant counts per group and label. A low-trust client is
    quarantined only if some group's TPR - FPR rises by more than z standard errors (z = 1.96 bounds the two groups
    together at 5%). It is readmitted if pooled F1 and every group's TPR - FPR fall by at most delta. Readmissions
    are then applied greedily, strongest evidence first, each checked jointly against the current aggregate."""
    P, log, K, V = CPQ, ctx['log'], len(U), ctx['val']
    Q, clipped, wts = cpq_trust(w, U, ctx)

    def cand(weights):
        return w + ((weights[:, None] * clipped).sum(0) / weights.sum() if weights.sum() > 0 else 0.)

    def guard(dJ, ok, f_new, f_base, se=None):
        lo = dJ if FLAGS['guard'] == 'point' or se is None else dJ - P['z'] * se
        return f_new >= f_base - P['delta'] and (not ok.any() or lo[ok].min() >= -P['delta'])

    pb, passed = val_pred(cand(wts), V), []
    if FLAGS['test_all']:  # null diagnostic: would the balanced test claim a gain from including institution k?
        for k in range(K):
            w_in, w_out = wts.copy(), wts.copy(); w_in[k] = max(wts[k], P['tau']); w_out[k] = 0.
            if w_out.sum() == 0:
                continue
            dJ, se, ok, _, _ = flip_stats(val_table(val_pred(cand(w_in), V), V, val_pred(cand(w_out), V)), {k})
            ev = (dJ - P['z'] * se)[ok]
            log['null_tests'] += 1; log['null_rejects'] += int(ev.size > 0 and ev.max() > 0)
    for k in np.where(Q < P['tau'])[0]:
        log['lowtrust'][k] += 1
        trial = wts.copy(); trial[k] = P['tau']
        dJ, se, ok, f_new, f_base = flip_stats(val_table(val_pred(cand(trial), V), V, pb), {k}, ctx['liars'])
        log['evals'] += 2
        ev = (dJ - P['z'] * se)[ok]
        if ev.size and ev.max() > 0:
            log['quarantined'][k] += 1
            if guard(dJ, ok, f_new, f_base, se):
                passed.append((float(ev.max()), int(k)))
    final, readm = wts.copy(), []
    for _, k in sorted(passed, reverse=True):
        trial = final.copy(); trial[k] = P['tau']
        if readm:
            dJ, se, ok, f_new, f_base = flip_stats(val_table(val_pred(cand(trial), V), V, val_pred(cand(final), V)),
                                                   set(readm) | {k}, ctx['liars'])
            log['evals'] += 1
            if not guard(dJ, ok, f_new, f_base, se):
                log['joint_blocked'] += 1
                continue
        final = trial; readm.append(k); log['readmitted'][k] += 1
    if final.sum() == 0:
        log['skips'] += 1
        return np.zeros_like(w), np.zeros(K)
    return cand(final) - w, final / final.sum()


def mdav(q, k):
    """MDAV microaggregation (Singh et al. 2023, Algorithm 1) of the 1-D attributes q into clusters of size k to 2k-1."""
    R, out = list(range(len(q))), []
    q = np.asarray(q, float)
    def closest(r, pool):
        return sorted(pool, key=lambda i: abs(q[i] - q[r]))[:k]
    while len(R) >= 3 * k:
        xbar = q[R].mean(); xr = max(R, key=lambda i: abs(q[i] - xbar))
        xs = max(R, key=lambda i: abs(q[i] - q[xr]))
        c1 = closest(xr, R); R = [i for i in R if i not in c1]
        c2 = closest(xs, R); R = [i for i in R if i not in c2]
        out += [c1, c2]
    if 2 * k <= len(R) <= 3 * k - 1:
        xbar = q[R].mean(); xr = max(R, key=lambda i: abs(q[i] - xbar))
        c1 = closest(xr, R); out += [c1, [i for i in R if i not in c1]]
    elif R:
        out.append(R)
    return out


assert sorted(len(c) for c in mdav([.1, .2, .3, .7, .8, .9], 3)) == [3, 3]
assert sorted(sum(mdav(list(range(20)), 3), [])) == list(range(20))


def agg_singh(w, U, ctx):
    """Fair attack detection by microaggregation (Singh, Blanco-Justicia and Domingo-Ferrer 2023, Protocol 2):
    institutions publish a demographic attribute (here their disability prevalence), MDAV groups them into clusters of
    at least k = SINGH_K, and within each cluster an update is accepted if its Euclidean distance to the cluster's mean
    update lies in [Q1 - 1.5 IQR, Q3 + 1.5 IQR] (Eq. 2, lambda = 1.5). Accepted updates get alpha_u = 1/m and rejected
    ones 0, as in Protocol 2 (the step is not renormalized). SINGH_K = 1 gives the paper's FL baseline (one cluster)."""
    K = len(U); q = ctx['prev']
    clusters = [list(range(K))] if SINGH['k'] == 1 else mdav(q, SINGH['k'])
    acc = np.zeros(K)
    for C in clusters:
        d = np.linalg.norm(U[C] - U[C].mean(0), axis=1)
        q1, q3 = np.percentile(d, [25, 75]); iqr = q3 - q1
        for i, di in zip(C, d):
            acc[i] = float(q1 - 1.5 * iqr <= di <= q3 + 1.5 * iqr)
    if acc.sum() == 0:
        ctx['log']['skips'] += 1
        return np.zeros_like(w), np.zeros(K)
    return (acc[:, None] * U).sum(0) / K, acc / acc.sum()


SINGH = dict(k=3)
PART = dict(frac=None, rng=None)  # partial participation, set only by partial_grid


AGG = dict(fedavg=agg_fedavg, median=agg_median, trimmed=agg_trimmed, multikrum=agg_multikrum,
           fltrust=agg_fltrust, fltrust_noscale=agg_fltrust_noscale, fedval=agg_fedval, anchor=agg_anchor,
           cpq=agg_cpq, cpqj=agg_cpqj, singh=agg_singh)


# ----------------------------------------------------------------------------- attacks
def minmax_update(B, direction='unit'):
    """Min-max attack (Shejwalkar and Houmansadr 2021): the largest step against the benign mean whose distance to
    every benign update stays within the largest benign pairwise distance. direction 'unit' is the inverse unit
    vector, 'std' the inverse coordinate-wise standard deviation."""
    mu = B.mean(0)
    p = -mu / (np.linalg.norm(mu) + 1e-12) if direction == 'unit' else -B.std(0) / (np.linalg.norm(B.std(0)) + 1e-12)
    dmax = max(np.linalg.norm(a - b) for a in B for b in B)
    lo, hi = 0., dmax + max(np.linalg.norm(mu - b) for b in B)
    for _ in range(50):
        g = (lo + hi) / 2
        lo, hi = (g, hi) if max(np.linalg.norm(mu + g * p - b) for b in B) <= dmax else (lo, g)
    out = mu + lo * p
    assert max(np.linalg.norm(out - b) for b in B) <= dmax + 1e-9
    return out


def poison(U, attackers, attack):
    U, att = U.copy(), list(attackers)
    if attack in ('signflip', 'liar'):
        U[att] = -5 * U[att]
    elif attack == 'targeted':
        U[att] = 3 * U[att]  # boosted update trained on flipped support-group labels
    elif attack == 'minmax':
        U[att] = minmax_update(np.delete(U, att, 0))
    elif attack == 'minmax_std':
        U[att] = minmax_update(np.delete(U, att, 0), 'std')
    elif attack == 'pushpos':  # honest update plus a large shift of the output bias (last parameter)
        for a in att:
            U[a, -1] += PUSH * np.linalg.norm(U[a])
    return U


# ----------------------------------------------------------------------------- one federated run
def gate_check(w_new, w_rel, ctx):
    """Group-wise release gate: every institution reports the 16 paired counts of the candidate release against the
    last released model on its validation split. The release is blocked if some group's Youden index falls
    significantly, i.e. dJ_h + z se_h < 0 for a group with enough records (z = 1.96, one-sided, Bonferroni over the two
    groups). se_h is the null standard error of Eq. 5."""
    V = ctx['val']
    dJ, se, ok, _, _ = flip_stats(val_table(val_pred(w_new, V), V, val_pred(w_rel, V)), set(), ctx['liars'])
    return bool(ok.any() and (dJ + GATE_Z * se)[ok].min() < 0)


def run_one(clients, anchor, method, attackers=(), attack=None, rounds=16, rw=False, intercept=False,
            fallback='skip', f=1, w0=None):
    """One federated run. attack 'slowflip' sends honest updates for the first half of the rounds and sign-flipped
    updates afterwards. Every run also passes each round's model through the release gate and reports the last released
    model (rel_*) next to the final model."""
    K = len(clients); D = clients[0]['xtr'].shape[1]
    w = np.zeros(D) if w0 is None else w0.copy()
    excl_thr = 0.05 if K == 6 else 0.3 / K  # 0.3 of a uniform share for every K (0.3/6 is not exactly 0.05 in floats)
    log = dict(skips=0, evals=0, lowtrust=np.zeros(K), quarantined=np.zeros(K), readmitted=np.zeros(K),
               wsum=np.zeros(K), excl=np.zeros(K), nweights=0, multi_readmit=0, joint_violation=0, joint_blocked=0,
               null_tests=0, null_rejects=0, gate_blocks=0, npart=np.zeros(K))
    sws = [c.get('swtr') for c in clients]
    fed_rate = float(sum((c['ytr'] * (1 if s is None else s)).sum() for c, s in zip(clients, sws))
                     / sum((len(c['ytr']) if s is None else s.sum()) for c, s in zip(clients, sws)))
    ctx = dict(anchor=anchor, clients=clients, log=log, fallback=fallback, f=f, val=val_index(clients),
               liars=frozenset(attackers) if attack == 'liar' else frozenset(), s2=FEDVAL['s2'], fed_rate=fed_rate,
               prev=[float(c['gtr'].mean()) for c in clients])
    curve, w_rel = [], w.copy()
    CENTER['m'] = anchor[0].mean(0)
    for t in range(rounds):
        att_now = attack if attack != 'slowflip' else ('signflip' if t >= rounds // 2 else None)
        U = []
        for k, c in enumerate(clients):
            y = np.where(c['gtr'] == 1, 0., c['ytr']) if (k in attackers and att_now == 'targeted') else c['ytr']
            sw = np.where(c['gtr'] == 1, 1.5, 1.0) if rw else None
            if sws[k] is not None:
                sw = sws[k] if sw is None else sw * sws[k]
            U.append(train_model(w, c['xtr'], y, weights=sw) - w)
        U = poison(np.array(U), attackers, att_now) if (attackers and att_now) else np.array(U)
        if PART['frac']:  # partial participation: only a random subset of institutions is aggregated this round
            P = np.sort(PART['rng'].choice(K, max(2, int(round(PART['frac'] * K))), replace=False))
            sub = dict(ctx, clients=[clients[i] for i in P], val=val_index([clients[i] for i in P]),
                       prev=[ctx['prev'][i] for i in P], liars=frozenset())
            delta, ws = AGG[method](w, U[P], sub)
            ctx['s2'], wts = sub['s2'], None
            if ws is not None:  # weight relative to a uniform share among the participants, logged when sampled
                r = ws * len(P) / K
                log['wsum'][P] += r; log['excl'][P] += (r < excl_thr); log['npart'][P] += 1
        else:
            delta, wts = AGG[method](w, U, ctx)
        w = w + delta
        if wts is not None:
            log['wsum'] += wts; log['excl'] += (wts < excl_thr); log['nweights'] += 1
        if gate_check(w, w_rel, ctx):
            log['gate_blocks'] += 1
        else:
            w_rel = w.copy()
        curve.append(evaluate(w, clients)['f1'])
    if PART['frac'] and log['npart'].any():  # make_row divides by nweights: store means over sampled rounds
        n = np.maximum(log['npart'], 1)
        log['wsum'], log['excl'], log['nweights'] = log['wsum'] / n, log['excl'] / n, 1
    adapters = [fit_intercept(w, c) for c in clients] if intercept else None
    m = evaluate(w, clients, adapters)
    rel = evaluate(w_rel, clients, adapters)
    m.update({f'rel_{k}': rel[k] for k in ('f1', 'recall_support', 'fpr_support', 'recall_majority', 'fpr_majority')})
    m['gate_blocks'] = log['gate_blocks']
    if 'gte_aud' in clients[0]:
        aud = evaluate(w, clients, adapters, gkey='gte_aud')
        m.update({f'aud_{k}': aud[k] for k in ('recall_support', 'fpr_support', 'recall_majority', 'fpr_majority')})
    if FLAGS['test_all']:
        m.update(null_tests=log['null_tests'], null_rejects=log['null_rejects'])
    return m, curve, log


def run_reference(clients, kind, rounds=16, w0=None):
    """Centralized (pooled data) and local-only references with the same number of gradient steps."""
    steps, D = 6 * rounds, clients[0]['xtr'].shape[1]
    w0 = np.zeros(D) if w0 is None else w0
    if kind == 'central':
        X = np.vstack([c['xtr'] for c in clients]); Y = np.concatenate([c['ytr'] for c in clients])
        return evaluate(train_model(w0, X, Y, steps=steps), clients)
    if kind == 'local':
        return evaluate(np.array([train_model(w0, c['xtr'], c['ytr'], steps=steps) for c in clients]), clients)
    return evaluate(None, clients, preds=[c['bayes'] for c in clients])  # Bayes rule of the synthetic oracle


# ----------------------------------------------------------------------------- experiment grids
MAIN = ['fedavg', 'median', 'trimmed', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']
ABL = {'anchor+rw': ('anchor', True, False), 'anchor+int': ('anchor', False, True),
       'anchor+rw+int': ('anchor', True, True), 'cpq+rw+int': ('cpq', True, True), 'cpqj+rw+int': ('cpqj', True, True)}
TRUST = ['fltrust', 'fedval', 'anchor', 'cpq', 'cpqj']  # rules that use the server's anchor data
FAMILIES = ['signflip', 'targeted', 'minmax', 'pushpos', 'liar']


def kind_of(fam):
    return None if fam == 'none' else ('signflip' if fam == 'signflip2' else fam)


def attacks(K, two=True, families=FAMILIES):
    out = [('none', ())]
    for fam in families:
        out += [(fam, (a,)) for a in range(K)]
    if two:
        out += [('signflip2', (a, (a + 1) % K)) for a in range(K)]
    return out


def make_row(dataset, seed, fam, att, name, m, log, K):
    row = dict(dataset=dataset, seed=seed, family=fam, attackers='-'.join(map(str, att)) or 'none', method=name,
               **m)
    if log is None:
        return row
    nw = log['nweights']
    row.update(skips=log['skips'], evals=log['evals'], multi_readmit=log['multi_readmit'],
               joint_violation=log['joint_violation'], joint_blocked=log['joint_blocked'])
    for k in range(K):
        row[f'w{k}'] = log['wsum'][k] / nw if nw else np.nan
        row[f'ex{k}'] = log['excl'][k] / nw if nw else np.nan
        row[f'q{k}'] = log['quarantined'][k]; row[f'ra{k}'] = log['readmitted'][k]; row[f'lt{k}'] = log['lowtrust'][k]
    return row


def grid(dataset, seeds=SEEDS):
    K = 4 if dataset == 'synthetic' else len(ACS_CLIENTS)
    rounds = 16 if dataset == 'synthetic' else 30
    rows, curves = [], {}
    for seed in seeds:
        cl, anc = synth(seed) if dataset == 'synthetic' else acs(seed)
        for fam, att in attacks(K, two=(dataset == 'synthetic')):
            jobs = [(m, m, False, False) for m in (VALIDATION_RULES if fam == 'liar' else MAIN)]
            if fam in ('none', 'signflip', 'targeted'):
                jobs += [(name, base, rw, it) for name, (base, rw, it) in ABL.items()]
            if fam == 'none':
                jobs += [('fltrust_noscale', 'fltrust_noscale', False, False)]
            for name, base, rw, it in jobs:
                m, curve, log = run_one(cl, anc, base, att, kind_of(fam), rounds, rw, it)
                rows.append(make_row(dataset, seed, fam, att, name, m, log, K))
                curves[(dataset, fam, rows[-1]['attackers'], name, seed)] = curve
        for kind in ['central', 'local'] + (['bayes'] if dataset == 'synthetic' else []):
            rows.append(make_row(dataset, seed, 'none', (), kind, run_reference(cl, kind, rounds), None, K))
        if dataset == 'synthetic':  # anchor representativeness
            cl2, anc2 = synth(seed, anchor='mixture')
            for fam, att in attacks(K, two=False, families=['signflip']):
                for meth in TRUST:
                    m, _, log = run_one(cl2, anc2, meth, att, kind_of(fam), rounds)
                    rows.append(make_row('synthetic-mixanchor', seed, fam, att, meth, m, log, K))
        print(f'  {dataset} seed {seed} done', flush=True)
    return rows, curves


def anchor_grid(seeds=SEEDS):
    """ACS anchor sensitivity: the same client splits with the anchor taken from Vermont (main grid), from 500
    held-out records of each client state, or pooled from all six. Pooled anchor also runs the rotated attacks."""
    rows, K = [], len(ACS_CLIENTS)
    for seed in seeds:
        for a in ['pooled'] + ACS_CLIENTS:
            cl, anc = acs(seed, anchor=a)
            fams = attacks(K, two=False, families=['signflip', 'targeted']) if a == 'pooled' else [('none', ())]
            for fam, att in fams:
                for meth in TRUST:
                    m, _, log = run_one(cl, anc, meth, att, kind_of(fam), 30)
                    rows.append(make_row(f'acs-anchor-{a}', seed, fam, att, meth, m, log, K))
        print(f'  anchor seed {seed} done', flush=True)
    return rows


def match_grid(seeds=SEEDS):
    """Separate employment base rate from disability prevalence: 'rate' sets every state's employment rate to the
    anchor's and keeps its prevalence, 'prev' sets every state's prevalence to the six-state mean and keeps its rate."""
    rows, K = [], len(ACS_CLIENTS)
    for seed in seeds:
        for match in ['rate', 'prev']:
            cl, anc = acs(seed, match=match)
            for meth in ['fedavg', 'multikrum'] + TRUST:
                m, _, log = run_one(cl, anc, meth, (), None, 30)
                rows.append(make_row(f'acs-match-{match}', seed, 'none', (), meth, m, log, K))
        print(f'  match seed {seed} done', flush=True)
    return rows


def puma_grid(seeds=SEEDS, frac=0.1):
    """89 PUMA institutions. Attack conditions place a random 10% of institutions under attacker control
    (9 attackers, drawn per seed); trimmed mean and Multi-Krum are told f = 9."""
    rows = []
    for seed in seeds:
        cl, anc = acs_puma(seed)
        K = len(cl); nat = int(round(frac * K))
        att = tuple(sorted(np.random.default_rng(seed + 50).choice(K, nat, replace=False).tolist()))
        for fam in ['none', 'signflip', 'targeted', 'minmax']:
            a = () if fam == 'none' else att
            for meth in MAIN:
                m, _, log = run_one(cl, anc, meth, a, kind_of(fam), 30, f=nat)
                rows.append(make_row('acs-puma', seed, fam, a, meth, m, log, K))
        for kind in ['central', 'local']:
            rows.append(make_row('acs-puma', seed, 'none', (), kind, run_reference(cl, kind, 30), None, K))
        print(f'  puma seed {seed} done', flush=True)
    return rows


def mlp_grid(seeds=SEEDS):
    """ACS state federation with a one-hidden-layer MLP (16 ReLU units). The learning rate is chosen on TUNE_SEED
    by clean FedAvg test F1 before any reported seed is run."""
    rows, K = [], len(ACS_CLIENTS)
    MODEL.update(kind='mlp')
    try:
        tune = {}
        for lr in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):  # JAS revision: grid extended past 1.0
            MODEL['lr'] = lr
            cl, anc = acs(TUNE_SEED)
            w0 = init_params(cl[0]['xtr'].shape[1], TUNE_SEED)
            tune[lr] = run_one(cl, anc, 'fedavg', (), None, 30, w0=w0)[0]['f1']
        MODEL['lr'] = max(tune, key=tune.get)
        (OUT / 'mlp_tuning.json').write_text(json.dumps(dict(f1_by_lr=tune, chosen=MODEL['lr']), indent=1))
        for seed in seeds:
            cl, anc = acs(seed)
            w0 = init_params(cl[0]['xtr'].shape[1], seed)
            for fam, att in attacks(K, two=False, families=['signflip', 'targeted']):
                for meth in MAIN:
                    m, _, log = run_one(cl, anc, meth, att, kind_of(fam), 30, w0=w0)
                    rows.append(make_row('acs-mlp', seed, fam, att, meth, m, log, K))
            for kind in ['central', 'local']:
                rows.append(make_row('acs-mlp', seed, 'none', (), kind, run_reference(cl, kind, 30, w0), None, K))
            rows.append(make_row('acs-mlp', seed, 'none', (), 'central4', run_reference(cl, 'central', 120, w0), None, K))
            print(f'  mlp seed {seed} done', flush=True)
    finally:
        MODEL.update(kind='logreg', lr=0.25)
    return rows


def cpq_sensitivity(seeds=SEEDS):
    """Threshold sweep. Recall test: a0 x nu as before. Balanced test: a0 x z, plus eta1 in {5, 20} at the defaults."""
    rows, base = [], dict(CPQ)
    settings = [('cpq', dict(a0=a0, nu=nu)) for a0 in (0.0, 0.2, 0.4) for nu in (0.005, 0.01, 0.02)]
    settings += [('cpqj', dict(a0=a0, z=z)) for a0 in (0.0, 0.2, 0.4) for z in (1.645, 1.96, 2.576)]
    settings += [(m, dict(eta1=e)) for m in VALIDATION_RULES for e in (5.0, 20.0)]
    for ds in ['synthetic', 'acs']:
        K, rounds = (4, 16) if ds == 'synthetic' else (len(ACS_CLIENTS), 30)
        for seed in seeds:
            cl, anc = synth(seed) if ds == 'synthetic' else acs(seed)
            for meth, upd in settings:
                CPQ.clear(); CPQ.update(base); CPQ.update(upd)
                for fam, att in attacks(K, two=False, families=['signflip', 'targeted']):
                    m, _, log = run_one(cl, anc, meth, att, kind_of(fam), rounds)
                    rows.append(dict(dataset=ds, seed=seed, method=meth, a0=CPQ['a0'], nu=CPQ['nu'], z=CPQ['z'],
                                     eta1=CPQ['eta1'], family=fam, attackers='-'.join(map(str, att)) or 'none', **m,
                                     quarantined=int(log['quarantined'].sum()), readmitted=int(log['readmitted'].sum()),
                                     lowtrust=int(log['lowtrust'].sum()),
                                     attacker_readmitted=int(sum(log['readmitted'][a] for a in att))))
        print(f'  sensitivity {ds} done', flush=True)
    CPQ.clear(); CPQ.update(base)
    return rows


def g_input_ablation(seeds=SEEDS):
    """Same 20 seeds and revised code, with and without the audit label g in the model input, with weights."""
    rows = []
    for seed in seeds:
        for keep in (True, False):
            cl, anc = synth(seed, keep_g=keep)
            for fam, att in [('none', ()), ('signflip', (3,))]:
                for name, (base, rw, it) in {'fedavg': ('fedavg', False, False), 'anchor': ('anchor', False, False),
                                             'anchor+rw+int': ('anchor', True, True)}.items():
                    m, _, log = run_one(cl, anc, base, att, 'signflip' if att else None, 16, rw, it)
                    rows.append(dict(seed=seed, g_in_input=int(keep), family=fam, method=name, **m,
                                     **{f'w{k}': log['wsum'][k] / max(log['nweights'], 1) for k in range(4)}))
    write_csv('g_input_ablation.csv', rows)


def fallback_audit():
    """Prototype configuration (g in the input, uniform fallback, 5 prototype seeds): count all-zero-weight rounds
    and check that this code path reproduces results/summary.json."""
    out = []
    for seed in ORIG_SEEDS:
        cl, anc = synth(seed, keep_g=True)
        for attack in [False, True]:
            for name, (base, rw, it) in {'trusted': ('anchor', False, False),
                                         'trustfair': ('anchor', True, True)}.items():
                m, _, log = run_one(cl, anc, base, (3,) if attack else (), 'signflip' if attack else None,
                                    16, rw, it, fallback='uniform')
                out.append(dict(seed=seed, attack=int(attack), method=name, fallback_rounds=log['skips'],
                                attacker_weight=float(log['wsum'][3] / 16), f1=m['f1'], tpr_gap=m['tpr_gap'],
                                weights=(log['wsum'] / 16).round(4).tolist()))
    return out


def _with_flags(upd, fn):
    old = dict(FLAGS); FLAGS.update(upd)
    try:
        return fn()
    finally:
        FLAGS.clear(); FLAGS.update(old); _ACS.clear()


def _conds(K, fams):
    return [('none', ())] + [(f, (a,)) for f in fams for a in range(K)]


def variants_grid(seeds=SEEDS, only=None):
    """JAS revision variants on the ACS state federation (Vermont anchor, 30 rounds) and, for FedVal, the synthetic
    proxy: intercept-free trust cosine, prior-shift-corrected anchor, interval guard, FedVal with the per-sample MAD of
    Eq. 4, a slow-start sign-flip attacker and min-max with the inverse standard deviation."""
    rows, K = [], len(ACS_CLIENTS)
    specs = [('acs-icpt', dict(nobias=True), ['fltrust', 'anchor', 'cpq', 'cpqj'], ['signflip']),
             ('acs-center', dict(center=True), ['fltrust', 'anchor', 'cpq', 'cpqj'], ['signflip']),
             ('acs-prior', dict(prior=True), ['fltrust', 'anchor', 'cpq', 'cpqj'], ['signflip']),
             ('acs-guardci', dict(guard='ci'), ['cpqj'], ['signflip', 'targeted', 'pushpos']),
             ('acs-fedval2', dict(fedval_mad='samples'), ['fedval'], ['signflip', 'targeted']),
             ('acs-slow', {}, MAIN, ['slowflip']),
             ('acs-minmaxstd', {}, MAIN, ['minmax_std'])]
    specs = [x for x in specs if only is None or x[0] == only]
    for seed in seeds:
        for ds, upd, meths, fams in specs:
            def job(ds=ds, meths=meths, fams=fams):
                out = []
                cl, anc = acs(seed)
                conds = _conds(K, fams) if ds not in ('acs-slow', 'acs-minmaxstd') else \
                    [(f, (a,)) for f in fams for a in range(K)]
                for fam, att in conds:
                    for meth in meths:
                        m, _, log = run_one(cl, anc, meth, att, kind_of(fam), 30)
                        out.append(make_row(ds, seed, fam, att, meth, m, log, K))
                return out
            rows += _with_flags(upd, job)

        def job_syn():
            out = []
            cl, anc = synth(seed)
            for fam, att in _conds(4, ['signflip', 'targeted']):
                m, _, log = run_one(cl, anc, 'fedval', att, kind_of(fam), 16)
                out.append(make_row('synthetic-fedval2', seed, fam, att, 'fedval', m, log, 4))
            return out
        if only is None:
            rows += _with_flags(dict(fedval_mad='samples'), job_syn)
        print(f'  variants seed {seed} done', flush=True)
    return rows


def null_grid(seeds=SEEDS):
    """Null federation (acs_iid): no institution differs from the others or from the anchor. Clean runs of every rule;
    CPQ-B also tests each institution's inclusion every round (FLAGS['test_all']) to estimate the balanced test's
    rejection rate when inclusion cannot help in expectation."""
    rows = []
    for seed in seeds:
        cl, anc = acs_iid(seed)
        for meth in MAIN:
            upd = dict(test_all=True) if meth == 'cpqj' else {}
            m, _, log = _with_flags(upd, lambda: run_one(cl, anc, meth, (), None, 30))
            rows.append(make_row('acs-iid', seed, 'none', (), meth, m, log, len(cl)))
        print(f'  null seed {seed} done', flush=True)
    return rows


def sens2_grid(seeds=SEEDS):
    """Sensitivity of CPQ-B to tau (acceptance threshold and readmission weight) and to the guard tolerance delta."""
    rows, base = [], dict(CPQ)
    settings = [dict(tau=t) for t in (0.3, 0.7)] + [dict(delta=d) for d in (0.005, 0.02)]
    for ds in ['synthetic', 'acs']:
        K, rounds = (4, 16) if ds == 'synthetic' else (len(ACS_CLIENTS), 30)
        for seed in seeds:
            cl, anc = synth(seed) if ds == 'synthetic' else acs(seed)
            for upd in settings:
                CPQ.clear(); CPQ.update(base); CPQ.update(upd)
                for fam, att in attacks(K, two=False, families=['signflip', 'targeted']):
                    m, _, log = run_one(cl, anc, 'cpqj', att, kind_of(fam), rounds)
                    rows.append(dict(dataset=ds, seed=seed, method='cpqj', a0=CPQ['a0'], nu=CPQ['nu'], z=CPQ['z'],
                                     eta1=CPQ['eta1'], tau=CPQ['tau'], delta=CPQ['delta'], family=fam,
                                     attackers='-'.join(map(str, att)) or 'none', **m,
                                     quarantined=int(log['quarantined'].sum()), readmitted=int(log['readmitted'].sum()),
                                     lowtrust=int(log['lowtrust'].sum()),
                                     attacker_readmitted=int(sum(log['readmitted'][a] for a in att)),
                                     **{f'w{k}': log['wsum'][k] / max(log['nweights'], 1) for k in range(K)}))
        print(f'  sens2 {ds} done', flush=True)
    CPQ.clear(); CPQ.update(base)
    return rows


def simple_grid(ds, make, fams, meths=MAIN, seeds=SEEDS, upd=None):
    """Clean plus rotated attacks on an ACS state federation built by make(seed)."""
    rows, K = [], len(ACS_CLIENTS)

    def job(seed):
        out = []
        cl, anc = make(seed)
        for fam, att in _conds(K, fams):
            for meth in meths:
                m, _, log = run_one(cl, anc, meth, att, kind_of(fam), 30)
                out.append(make_row(ds, seed, fam, att, meth, m, log, K))
        return out
    for seed in seeds:
        rows += _with_flags(upd or {}, lambda: job(seed))
        print(f'  {ds} seed {seed} done', flush=True)
    return rows


DISCLOSE = {'mcar80': ('mcar', 0.8), 'mcar60': ('mcar', 0.6), 'mcar40': ('mcar', 0.4), 'age': ('age', None)}


def disclose_grid(seeds=SEEDS):
    """Voluntary disclosure of the audit label (see disclose()); same splits and seeds as the main grid."""
    rows = []
    for name, scen in DISCLOSE.items():
        def make(seed, scen=scen):
            cl, anc = acs(seed)
            return disclose(cl, anc, np.random.default_rng(seed + 77), scen)
        rows += simple_grid(f'acs-disc-{name}', make, ['signflip'],
                            ['fedavg', 'multikrum', 'fltrust', 'fedval', 'anchor', 'cpq', 'cpqj'], seeds)
    return rows


HELDOUT = ['wy', 'ak', 'sd', 'de', 'ri', 'mt']  # six least populous states not used before (2018 estimates)


def heldout_grid(seeds=SEEDS):
    """Confirmatory replication on six states never used during method development, same anchor (Vermont)."""
    global ACS_CLIENTS
    old = ACS_CLIENTS; ACS_CLIENTS = HELDOUT
    try:
        rows, _ = grid('acs', seeds)
    finally:
        ACS_CLIENTS = old
    for r in rows:
        r['dataset'] = 'acs-heldout'
    return rows


def heldout_center_grid(seeds=SEEDS):
    """Centered trust cosine on the six held-out states (pre-registered as H5 and H6)."""
    global ACS_CLIENTS
    old = ACS_CLIENTS; ACS_CLIENTS = HELDOUT
    try:
        rows = variants_grid(seeds, only='acs-center')
    finally:
        ACS_CLIENTS = old
    for r in rows:
        r['dataset'] = 'acs-heldout-center'
    return rows


def center_anchor_grid(seeds=SEEDS):
    """Centered trust cosine under every anchor of the anchor-sensitivity grid (clean condition)."""
    def job():
        rows, K = [], len(ACS_CLIENTS)
        for seed in seeds:
            for a in ['pooled'] + ACS_CLIENTS:
                cl, anc = acs(seed, anchor=a)
                for meth in ['fltrust', 'anchor', 'cpq', 'cpqj']:
                    m, _, log = run_one(cl, anc, meth, (), None, 30)
                    rows.append(make_row(f'acs-anchor-{a}-center', seed, 'none', (), meth, m, log, K))
            print(f'  center anchor seed {seed} done', flush=True)
        return rows
    return _with_flags(dict(center=True), job)


def center_puma_grid(seeds=SEEDS, frac=0.1):
    """Centered trust cosine on the 89-PUMA federation, clean and with the same 9 sign-flip attackers."""
    def job():
        rows = []
        for seed in seeds:
            cl, anc = acs_puma(seed)
            K = len(cl); nat = int(round(frac * K))
            att = tuple(sorted(np.random.default_rng(seed + 50).choice(K, nat, replace=False).tolist()))
            for fam in ['none', 'signflip']:
                for meth in ['fltrust', 'anchor', 'cpq', 'cpqj']:
                    m, _, log = run_one(cl, anc, meth, () if fam == 'none' else att, kind_of(fam), 30, f=nat)
                    rows.append(make_row('acs-puma-center', seed, fam, () if fam == 'none' else att, meth, m, log, K))
            print(f'  center puma seed {seed} done', flush=True)
        return rows
    return _with_flags(dict(center=True), job)


def partial_grid(seeds=SEEDS, puma=False, att_frac=0.1):
    """Partial participation: each round a random subset trains and is aggregated (4 of the 6 states, or half of the
    89 PUMAs). Every rule of a seed sees the same sampling sequence. Standard and centered trust cosine, clean and
    sign-flip (rotated single attacker for the states, the 9 attackers of the PUMA grid). Per-client quarantine
    counters are not reported, because the aggregators index the participants of each round."""
    rows = []
    for seed in seeds:
        cl, anc = acs_puma(seed) if puma else acs(seed)
        K = len(cl); frac = 0.5 if puma else 4 / 6
        if puma:
            nat = int(round(att_frac * K))
            att = tuple(sorted(np.random.default_rng(seed + 50).choice(K, nat, replace=False).tolist()))
            conds, f = [('none', ()), ('signflip', att)], nat
        else:
            conds, f = _conds(K, ['signflip']), 1
        ds = 'acs-puma-partial' if puma else 'acs-partial'
        for center, meths in [(False, ['fedavg', 'fltrust', 'anchor', 'cpqj']), (True, ['fltrust', 'anchor', 'cpqj'])]:
            def job():
                out = []
                for fam, a in conds:
                    for meth in meths:
                        PART.update(frac=frac, rng=np.random.default_rng(seed + 7000))
                        try:
                            m, _, log = run_one(cl, anc, meth, a, kind_of(fam), 30, f=f)
                        finally:
                            PART.update(frac=None, rng=None)
                        out.append(make_row(ds + ('-center' if center else ''), seed, fam, a, meth, m, log, K))
                return out
            rows += _with_flags(dict(center=center), job)
        print(f'  {ds} seed {seed} done', flush=True)
    return rows


def singh_grid(seeds=SEEDS, frac=0.1):
    """Singh et al. (2023) microaggregation detector (k = 3) and its distance baseline (k = 1) on the ACS state
    federation (clean, rotated sign-flip and targeted) and the PUMA federation (clean and the 9 sign-flip attackers)."""
    rows, K = [], len(ACS_CLIENTS)
    for seed in seeds:
        cl, anc = acs(seed)
        for fam, att in _conds(K, ['signflip', 'targeted']):
            for kk in (1, 3):
                SINGH['k'] = kk
                m, _, log = run_one(cl, anc, 'singh', att, kind_of(fam), 30)
                rows.append(make_row('acs', seed, fam, att, f'singh{kk}', m, log, K))
        cl, anc = acs_puma(seed); Kp = len(cl); nat = int(round(frac * Kp))
        att = tuple(sorted(np.random.default_rng(seed + 50).choice(Kp, nat, replace=False).tolist()))
        for fam in ['none', 'signflip']:
            for kk in (1, 3):
                SINGH['k'] = kk
                a = () if fam == 'none' else att
                m, _, log = run_one(cl, anc, 'singh', a, kind_of(fam), 30, f=nat)
                rows.append(make_row('acs-puma', seed, fam, a, f'singh{kk}', m, log, Kp))
        print(f'  singh seed {seed} done', flush=True)
    SINGH['k'] = 3
    return rows


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, keys); wr.writeheader(); wr.writerows(rows)


def main(parts):
    OUT.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}
    if 'audit' in parts:
        (OUT / 'original_fallback_audit.json').write_text(json.dumps(fallback_audit(), indent=1))
    if 'main' in parts:
        s = time.time(); rows, curves = [], {}
        for ds in ['synthetic', 'acs']:
            r, c = grid(ds); rows += r; curves.update(c)
        write_csv('runs.csv', rows)
        (OUT / 'curves.json').write_text(json.dumps({'|'.join(map(str, k)): v for k, v in curves.items()
                                                     if k[1] == 'signflip' and k[3] in MAIN}))
        times['main'] = round(time.time() - s)
    for part, fn, name in [('anchor', anchor_grid, 'runs_anchor.csv'), ('match', match_grid, 'runs_match.csv'),
                           ('puma', puma_grid, 'runs_puma.csv'), ('mlp', mlp_grid, 'runs_mlp.csv'),
                           ('sens', cpq_sensitivity, 'cpq_sensitivity.csv'), ('variants', variants_grid, 'runs_variants.csv'),
                           ('null', null_grid, 'runs_null.csv'),
                           ('center', lambda: variants_grid(only='acs-center'), 'runs_center.csv'),
                           ('heldout_center', heldout_center_grid, 'runs_heldout_center.csv'),
                           ('center_anchor', center_anchor_grid, 'runs_center_anchor.csv'),
                           ('center_puma', center_puma_grid, 'runs_center_puma.csv'),
                           ('singh', singh_grid, 'runs_singh.csv'), ('sens2', sens2_grid, 'cpq_sensitivity2.csv'),
                           ('partial', partial_grid, 'runs_partial.csv'),
                           ('partial_puma', lambda: partial_grid(puma=True), 'runs_partial_puma.csv'),
                           ('weighted', lambda: simple_grid('acs-weighted', lambda s: acs(s, weighted=True), ['signflip']),
                            'runs_weighted.csv'),
                           ('anc', lambda: simple_grid('acs-anc', acs, ['signflip'], upd=dict(data='_anc')), 'runs_anc.csv'),
                           ('heldout', heldout_grid, 'runs_heldout.csv'), ('disclose', disclose_grid, 'runs_disclose.csv')]:
        if part in parts:
            s = time.time(); write_csv(name, fn()); times[part] = round(time.time() - s)
            print(f'{part}: {times[part]}s', flush=True)
    if 'puma' in parts:
        (OUT / 'puma_meta.json').write_text(json.dumps(puma_meta(), indent=1))
    if 'g' in parts:
        g_input_ablation()
    env = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(), processor=platform.processor(),
               seeds=SEEDS, cpq=CPQ, fedval=FEDVAL, push=PUSH, min_cell=MIN_CELL, acs_clients=ACS_CLIENTS,
               acs_anchor=ACS_ANCHOR, parts=sorted(parts), part_seconds=times, seconds=round(time.time() - t0))
    (OUT / f'env_{"_".join(sorted(parts))}.json').write_text(json.dumps(env, indent=1))


ALL_PARTS = ['audit', 'main', 'anchor', 'match', 'puma', 'mlp', 'sens', 'g']
# JAS revision parts, run by name: variants, null, sens2, weighted, anc, heldout, disclose

if __name__ == '__main__':
    parts = set(sys.argv[sys.argv.index('--part') + 1].split(',')) if '--part' in sys.argv else set(ALL_PARTS)
    main(parts)

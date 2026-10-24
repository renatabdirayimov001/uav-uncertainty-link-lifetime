"""Prediction-level coverage with trace-level confidence intervals (8 independent test traces)."""
import sys, json
sys.argv = ['x', '0.1']
import numpy as np
from scipy import stats
from exp2 import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

TEST = [3, 4, 5, 6, 23, 24, 25, 26]
ALS = [0.05, 0.1, 0.2]
n = len(Yca)
q = {a: HGB(loss='quantile', quantile=a, max_iter=300, learning_rate=0.08).fit(Xtr, Ytr) for a in ALS}
corr = {a: np.quantile(q[a].predict(Xca) - Yca, min(1, (1 - a) * (1 + 1 / n))) for a in ALS}
mpca = mp.predict(Xca)
cpq = {a: np.quantile(mpca - Yca, min(1, (1 - a) * (1 + 1 / n))) for a in ALS}
z = {0.05: 1.6449, 0.1: 1.2816, 0.2: 0.8416}
per = []
for sd in TEST:
    X, Y = ds(sd)
    let = X[:, 3]; mpx = mp.predict(X)
    pr = np.array([m.predict(X) for m in ens]); mu, sg = pr.mean(0), pr.std(0)
    row = dict(seed=sd, n=int(len(Y)), meanY=float(Y.mean()), LET=float(np.mean(Y >= let)), LET_meanL=float(np.clip(let, 0, H).mean()),
               ML=float(np.mean(Y >= mpx)), ML_meanL=float(np.clip(mpx, 0, H).mean()),
               ML_mae=float(np.abs(mpx - Y).mean()), ML_bias=float((mpx - Y).mean()))
    for a in ALS:
        qa = q[a].predict(X)
        for nm, L in [('ENS', mu - z[a] * sg), ('QR', qa), ('CP', mpx - cpq[a]), ('CQR', qa - corr[a])]:
            L = np.clip(L, 0, H)
            row[f'{nm}@{a}'] = float(np.mean(Y >= L)); row[f'{nm}@{a}_meanL'] = float(L.mean())
            if a == 0.1: row[f'{nm}@{a}_zero'] = float(np.mean(L == 0)); row[f'{nm}@{a}_short_of_long'] = float(np.mean(L[Y > 10] <= 2))
    # conditional (alpha = 0.1) by HELLO age and distance
    Lc = np.clip(q[0.1].predict(X) - corr[0.1], 0, H); age = X[:, 4]; dR = X[:, 0]
    for lab, m in [('age0', age == 0), ('age1', age == 1), ('age2-4', (age >= 2) & (age <= 4)),
                   ('d<0.5', dR < 0.5), ('d0.5-0.8', (dR >= 0.5) & (dR < 0.8)), ('d>=0.8', dR >= 0.8)]:
        row[f'cond_{lab}'] = dict(n=int(m.sum()), meanY=float(Y[m].mean()), LET=float(np.mean(Y[m] >= let[m])),
                                  CQR=float(np.mean(Y[m] >= Lc[m])), CQR_meanL=float(Lc[m].mean()))
    per.append(row); print('trace', sd, 'done', flush=True)

def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(m), float(m - h), float(m + h)]
summ = {}
for k in per[0]:
    if k in ('seed',) or isinstance(per[0][k], dict): continue
    summ[k] = ci([r[k] for r in per])
for k in [k for k in per[0] if k.startswith('cond_')]:
    summ[k] = {kk: ci([r[k][kk] for r in per]) for kk in per[0][k]}
json.dump(dict(per_trace=per, summary=summ, corr={str(a): float(v) for a, v in corr.items()},
               cp_margin={str(a): float(v) for a, v in cpq.items()}), open('results5/pred_ci.json', 'w'), indent=1)
print('DONE')

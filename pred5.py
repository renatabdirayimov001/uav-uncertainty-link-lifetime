"""Coverage with sub-step (continuous-time) lifetime labels: the exact range-exit instant is found by linear
interpolation of the trajectories inside the step in which the link breaks (resolution 0.01 s).
Checks that the CQR coverage is not an artefact of integer lifetimes (ties)."""
import sys, json
sys.argv = ['x', '0.1']
import numpy as np
from scipy import stats
from exp2 import mk, known, R, H, Xtr, Ytr, Xca, Yca
from sim import features_age
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

def cont_life(P, t, i, j, Y):
    """continuous residual lifetime: Y integer steps up; exit happens between t+Y-1 and t+Y."""
    out = Y.astype(float).copy()
    m = (Y < H) & (Y > 0)                          # Y = 0: link already down when sampled -> lifetime 0
    tt = t + Y[m].astype(int)                       # first step where the link is down
    a = P[tt - 1, j[m]] - P[tt - 1, i[m]]; b = P[tt, j[m]] - P[tt, i[m]]
    s = np.linspace(0, 1, 101)[None, :, None]
    d = np.linalg.norm(a[:, None, :] + s * (b - a)[:, None, :], axis=2)
    first = np.argmax(d > R, axis=1) / 100.0
    out[m] = Y[m] - 1 + first
    return out

def ds_cont(seed, T=1500, step=3):
    P, up, life, Po, A = mk(seed, T); N = P.shape[1]; iu = np.triu_indices(N, 1); Xs, Yi, Yc = [], [], []
    for t in range(20, T - H - 1, step):
        m = known(A, t)[iu]; i = iu[0][m]; j = iu[1][m]
        X, _ = features_age(Po, A, t, i, j, R, H); Y = life[t, i, j]
        Xs.append(X); Yi.append(Y); Yc.append(cont_life(P, t, i, j, Y))
    return np.vstack(Xs), np.concatenate(Yi), np.concatenate(Yc)

q = HGB(loss='quantile', quantile=0.1, max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, Ytr)
n = len(Yca)
corr_int = np.quantile(q.predict(Xca) - Yca, 0.9 * (1 + 1 / n))
_, _, Ycal_c = ds_cont(2)
corr_c = np.quantile(q.predict(Xca) - Ycal_c, 0.9 * (1 + 1 / n))
res = {'corr_integer': float(corr_int), 'corr_continuous': float(corr_c), 'per_trace': []}
for sd in [3, 4, 5, 6, 23, 24, 25, 26]:
    X, Yi, Yc = ds_cont(sd); qq = q.predict(X)
    r = dict(seed=sd, raw_int=float(np.mean(Yi >= np.clip(qq, 0, H))), raw_cont=float(np.mean(Yc >= np.clip(qq, 0, H))),
             cqr_int=float(np.mean(Yi >= np.clip(qq - corr_int, 0, H))), cqr_cont_intcal=float(np.mean(Yc >= np.clip(qq - corr_int, 0, H))),
             cqr_cont=float(np.mean(Yc >= np.clip(qq - corr_c, 0, H))))
    res['per_trace'].append(r); print(r, flush=True)
def ci(v):
    v = np.asarray(v); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return [float(v.mean()), float(h)]
res['summary'] = {k: ci([r[k] for r in res['per_trace']]) for k in ['raw_int', 'raw_cont', 'cqr_int', 'cqr_cont_intcal', 'cqr_cont']}
json.dump(res, open('results5/pred_cont.json', 'w'), indent=1); print(json.dumps(res['summary']), res['corr_integer'], res['corr_continuous'])

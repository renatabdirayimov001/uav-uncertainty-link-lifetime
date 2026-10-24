"""Continuous-time labels: calibration on one trace (seed 2) versus pooled calibration on four traces (2, 27, 28, 29)."""
import sys, json
sys.argv = ['x', '0.1']
import numpy as np
from scipy import stats
exec(open('pred5.py').read().split("res = {")[0])
cal = [ds_cont(s) for s in [27, 28, 29]]
Xp = np.vstack([Xca] + [c[0] for c in cal]); Yp = np.concatenate([Ycal_c] + [c[2] for c in cal]); npool = len(Yp)
corr_p = np.quantile(q.predict(Xp) - Yp, 0.9 * (1 + 1 / npool))
out = {'corr_single': float(corr_c), 'corr_pooled': float(corr_p), 'n_pooled': int(npool), 'per_trace': []}
for sd in [3, 4, 5, 6, 23, 24, 25, 26]:
    X, Yi, Yc = ds_cont(sd); qq = q.predict(X)
    r = dict(seed=sd, single=float(np.mean(Yc >= np.clip(qq - corr_c, 0, H))), pooled=float(np.mean(Yc >= np.clip(qq - corr_p, 0, H))),
             meanL_single=float(np.clip(qq - corr_c, 0, H).mean()), meanL_pooled=float(np.clip(qq - corr_p, 0, H).mean()))
    out['per_trace'].append(r); print(r, flush=True)
def ci(v):
    v = np.asarray(v); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return [float(v.mean()), float(h)]
out['summary'] = {k: ci([r[k] for r in out['per_trace']]) for k in ['single', 'pooled', 'meanL_single', 'meanL_pooled']}
json.dump(out, open('results5/pred_cont_pooled.json', 'w'), indent=1); print(json.dumps(out['summary']), corr_c, corr_p)

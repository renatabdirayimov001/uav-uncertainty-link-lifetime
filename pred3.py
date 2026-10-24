"""Ablation: role of the conformal correction when the quantile model is weaker (prediction level, 8 test traces)."""
import sys, json
sys.argv = ['x', '0.1']
import numpy as np
from scipy import stats
from exp2 import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

TEST = [3, 4, 5, 6, 23, 24, 25, 26]; a = 0.1; n = len(Yca)
rng = np.random.default_rng(0)
sub = rng.choice(len(Ytr), int(0.02 * len(Ytr)), replace=False)
noage = [c for c in range(Xtr.shape[1]) if c != 4]
variants = {
    'full model': (np.arange(Xtr.shape[1]), np.arange(len(Ytr)), dict(max_iter=300, learning_rate=0.08)),
    'without HELLO-age feature': (np.array(noage), np.arange(len(Ytr)), dict(max_iter=300, learning_rate=0.08)),
    '2% of training data': (np.arange(Xtr.shape[1]), sub, dict(max_iter=300, learning_rate=0.08)),
    'under-fitted (20 iterations)': (np.arange(Xtr.shape[1]), np.arange(len(Ytr)), dict(max_iter=20, learning_rate=0.08)),
}
models = {}
for k, (cols, idx, kw) in variants.items():
    m = HGB(loss='quantile', quantile=a, **kw).fit(Xtr[idx][:, cols], Ytr[idx])
    corr = float(np.quantile(m.predict(Xca[:, cols]) - Yca, (1 - a) * (1 + 1 / n)))
    models[k] = (m, cols, corr)
per = {k: [] for k in variants}
for sd in TEST:
    X, Y = ds(sd)
    for k, (m, cols, corr) in models.items():
        q = m.predict(X[:, cols])
        per[k].append([np.mean(Y >= np.clip(q, 0, H)), np.mean(Y >= np.clip(q - corr, 0, H)), np.clip(q, 0, H).mean(), np.clip(q - corr, 0, H).mean()])
    print('trace', sd, flush=True)
def ci(v):
    v = np.asarray(v); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]
out = {k: dict(correction_s=models[k][2], raw_cov=ci(np.array(v)[:, 0]), cqr_cov=ci(np.array(v)[:, 1]),
               raw_meanL=ci(np.array(v)[:, 2]), cqr_meanL=ci(np.array(v)[:, 3])) for k, v in per.items()}
json.dump(out, open('results5/ablation_cqr.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

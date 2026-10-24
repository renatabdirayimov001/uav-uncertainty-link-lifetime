"""Prediction-level quality of the published lifetime estimators on the 8 prediction test traces (same as pred2)."""
import sys, json, pickle
sys.argv = ['x', '0.1']
import numpy as np
from scipy import stats
from exp2 import mk, known, HOLD, R, H, mp, Xtr, Ytr, Xca, Yca
from sim import features_age
from pubbase import lbopar_lifetime, olsrp_lifetime, link_seq, link_runs
from nplstm import LSTMReg
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
lstm = pickle.load(open('results6/lstm.pkl', 'rb'))
qm = HGB(loss='quantile', quantile=0.1, max_iter=300, learning_rate=0.08).fit(Xtr, Ytr)
corr = np.quantile(qm.predict(Xca) - Yca, 0.9 * (1 + 1 / len(Yca)))
TEST = [3, 4, 23, 24]
per = {k: [] for k in ['LB-OPAR', 'OLSR+', 'LSTM', 'XGB', 'LET', 'CQR']}
for sd in TEST:
    P, up, life, Po, A = mk(sd, 1500); run, nbr = link_runs(A, HOLD); N = P.shape[1]; iu = np.triu_indices(N, 1)
    acc = {k: ([], []) for k in per}
    for t in range(20, 1500 - H - 1, 6):
        m = known(A, t)[iu]; i = iu[0][m]; j = iu[1][m]; Y = life[t, i, j]
        X, let = features_age(Po, A, t, i, j, R, H)
        est = {'LB-OPAR': lbopar_lifetime(Po, A, t, i, j, R, H), 'OLSR+': olsrp_lifetime(Po, A, t, i, j, R, H),
               'LSTM': lstm.predict(link_seq(Po, A, run, t, i, j, nbr)), 'XGB': mp.predict(X), 'LET': let, 'CQR': qm.predict(X) - corr}
        for k, L in est.items(): acc[k][0].append(np.clip(L, 0, H)); acc[k][1].append(Y)
    for k, (Ls, Ys) in acc.items():
        L = np.concatenate(Ls); Y = np.concatenate(Ys)
        per[k].append([np.mean(Y >= L), L.mean(), np.abs(L - Y).mean(), np.mean(L[Y > 10] <= 2), np.mean(L <= 2)])
    print('trace', sd, {k: round(v[-1][0], 3) for k, v in per.items()}, flush=True)
def ci(v):
    v = np.asarray(v); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return [float(v.mean()), float(h)]
out = {k: {nm: ci(np.array(v)[:, q]) for q, nm in enumerate(['cov', 'meanL', 'mae', 'short_of_long', 'short'])} for k, v in per.items()}
json.dump(out, open('results6/pred_pub.json', 'w'), indent=1); print(json.dumps(out, indent=0))

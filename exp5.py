"""Revision experiments (all routing results of the revised paper).
usage: python3 exp5.py <mode> <seeds>
  main     : base scenario, full estimator set incl. raw QR, new metrics (val seeds 7-10, test 11-22)
  shift    : trained at 10-25 m/s, tested at 25-40 m/s (no retraining); static vs adaptive calibration
  noretrain: trained in base scenario, tested in degraded-information scenario (no retraining)
  N30/N80  : swarm-size sensitivity with the N=50 models
"""
import sys, os, json, time, heapq
mode = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(',')]
sys.argv = ['x', '0.1']
os.environ['SCEN'] = 'base'
import numpy as np
from exp2 import *                       # trains point model mp, ensemble ens on base scenario (seeds 1, 2)
from route3 import run_routing3
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

n = len(Yca)
GRID = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
qm = {g: HGB(loss='quantile', quantile=g, max_iter=300, learning_rate=0.08).fit(Xtr, Ytr) for g in GRID}
qca = {g: qm[g].predict(Xca) for g in GRID}
corr = {g: np.quantile(qca[g] - Yca, min(1, (1 - g) * (1 + 1 / n))) for g in [0.05, 0.1, 0.2, 0.3]}
mpca = mp.predict(Xca)
cpq = {a: np.quantile(mpca - Yca, min(1, (1 - a) * (1 + 1 / n))) for a in [0.1, 0.3]}
Ecal = np.sort(qca[0.1] - Yca)
def Qhat(al):
    al = float(np.clip(al, 1e-4, 0.9999)); k = int(np.ceil((1 - al) * (len(Ecal) + 1))) - 1
    return Ecal[min(max(k, 0), len(Ecal) - 1)]
# quantile-level ACI: initial level beta0 = grid level whose raw calibration coverage is 0.9 (interpolated)
covg = np.array([np.mean(Yca >= qca[g]) for g in GRID])
BETA0 = float(np.interp(0.9, covg[::-1], np.array(GRID)[::-1]))
def qlevel(X, beta):
    beta = float(np.clip(beta, GRID[0], GRID[-1])); k = np.searchsorted(GRID, beta)
    if k == 0: return qm[GRID[0]].predict(X)
    g1, g2 = GRID[k - 1], GRID[k]; w = (beta - g1) / (g2 - g1)
    return (1 - w) * qm[g1].predict(X) + w * qm[g2].predict(X)
INFO = dict(beta0=BETA0, raw_cov_grid=dict(zip(map(str, GRID), covg.tolist())), corr={str(k): float(v) for k, v in corr.items()},
            cp_margin={str(k): float(v) for k, v in cpq.items()}, n_train=int(len(Ytr)), n_cal=int(n),
            hgb_params=qm[0.1].get_params())

def estimate(name, X, let):
    if name == 'LET': return let
    if name == 'ML': return mp.predict(X)
    if name == 'ENS':
        pr = np.array([m.predict(X) for m in ens]); return pr.mean(0) - 1.2816 * pr.std(0)
    if name == 'QRW': return qm['weak'].predict(X)
    if name == 'CQRW': return qm['weak'].predict(X) - corr['weak']
    if name.startswith('QR@'): return qm[float(name[3:])].predict(X)
    if name.startswith('CP@'): return mp.predict(X) - cpq[float(name[3:])]
    if name.startswith('CQR@'):
        g = float(name[4:]); return qm[g].predict(X) - corr[g]
    raise ValueError(name)

def obs_life(A, H, hold):
    """observable neighbour lifetime: time until the entry expires from the neighbour table"""
    known = A <= hold; out = np.zeros(A.shape); cur = np.zeros(A.shape[1:])
    for t in range(A.shape[0] - 1, -1, -1):
        cur = np.where(known[t], np.minimum(cur + 1, H), 0.0); out[t] = cur
    return out

GAM = 0.005
def evaluate5(seed, policies, N=50, speed=(10, 25), scen='base', T=int(os.environ.get('TT', 600 if mode.startswith('N') else 900)), nflows=15, Tu=2):
    os.environ['SCEN'] = scen
    P, up, life, Po, A = mk(seed, T, N=N, speed=speed)
    os.environ['SCEN'] = 'base'
    olife = obs_life(A, H, HOLD)
    iu = np.triu_indices(N, 1)
    rng = np.random.default_rng(seed + 5); flows = []
    while len(flows) < nflows:
        s, d = rng.choice(N, 2, replace=False)
        if np.linalg.norm(P[10, s] - P[10, d]) > 600: flows.append((s, d))
    feats = {}
    def F(t):
        if t not in feats:
            m = known(A, t)[iu]; i = iu[0][m]; j = iu[1][m]; X, let = features_age(Po, A, t, i, j, R, H); feats[t] = (i, j, X, let)
        return feats[t]
    res = {}
    for lab, (nm, margin) in policies.items():
        cache = {}; st = dict(at=0.1 if 'CQR+' in nm else BETA0, pend=[], cov_n=0, cov_hit=0, hist=[])
        def pf(t, nm=nm, cache=cache, st=st):
            if t not in cache:
                i, j, X, let = F(t)
                if nm == 'ORACLE': w = life[t, i, j]
                elif nm == 'HOP': w = let
                elif nm in ('CQR+ACI-ideal', 'CQR+ACI-obs', 'QL+ACI-obs'):
                    while st['pend'] and st['pend'][0][0] <= t:
                        _, e = heapq.heappop(st['pend'])
                        if nm == 'QL+ACI-obs': st['at'] = float(np.clip(st['at'] + GAM * (0.1 - e), GRID[0], GRID[-1]))
                        else: st['at'] = float(np.clip(st['at'] + GAM * (0.1 - e), 1e-4, 0.9999))
                    w = qlevel(X, st['at']) if nm == 'QL+ACI-obs' else qm[0.1].predict(X) - Qhat(st['at'])
                    w = np.clip(w, 0, H)
                    if nm == 'CQR+ACI-ideal':          # idealized feedback: true break time, known when it happens
                        for yy, ll in zip(life[t, i, j], w): heapq.heappush(st['pend'], (t + min(yy, ll), 1.0 if yy < ll else 0.0))
                    else:                               # observable feedback: neighbour-table expiry, break back-dated to last HELLO
                        yo = olife[t, i, j]; yb = np.where(yo < H, np.maximum(yo - HOLD, 0), yo)
                        for yy, bb, ll in zip(yo, yb, w):
                            e = 1.0 if bb < ll else 0.0
                            heapq.heappush(st['pend'], (t + (yy if e else ll), e))
                    st['hist'].append((int(t), st['at']))
                else: w = estimate(nm, X, let)
                w = np.clip(w, 0, H)
                if nm != 'HOP' and t <= T - H - 1:
                    st['cov_n'] += len(w); st['cov_hit'] += int(np.sum(life[t, i, j] >= w))
                W = np.full((N, N), np.nan); W[i, j] = w; W[j, i] = w; cache[t] = W
            return cache[t]
        M = run_routing3(up, pf, flows, T, policy='hop' if nm == 'HOP' else 'max-min', Tu=Tu, margin=margin, life=life)
        mins = nflows * (T - 20) / 60
        res[lab] = dict(PDR=M['deliv'] / M['sec'], PDR_uncond=M['deliv'] / M['slots'], connected=M['sec'] / M['slots'],
                        breaks=M['breaks'] / mins, disc=M['disc'] / mins, disc_re=M['disc_re'] / mins, disc_pro=M['disc_pro'] / mins,
                        hops=M['hops'] / max(1, M['hopsn']), cov_all=st['cov_hit'] / max(1, st['cov_n']),
                        link_cov_used=M['lc_hit'] / max(1, M['lc_n']), path_cov=M['pc_hit'] / max(1, M['pc_n']),
                        kept=M['kept'], kept_fail_rate=M['kept_fail'] / max(1, M['kept']), hopbins=M['hopbins'])
        if st['hist']: res[lab]['level_final'] = st['at']; res[lab]['level_trace'] = st['hist'][::25]
    return res

SHIFT_POL = {'ORACLE': ('ORACLE', 2), 'LET m=4': ('LET', 4), 'ML m=9': ('ML', 9), 'QR a=0.1': ('QR@0.1', 2),
             'CQR static a=0.1': ('CQR@0.1', 2), 'CQR+ACI ideal': ('CQR+ACI-ideal', 2), 'CQR+ACI obs': ('CQR+ACI-obs', 2),
             'QL+ACI obs': ('QL+ACI-obs', 2)}
if mode == 'main':
    pol = {'ORACLE': ('ORACLE', 2), 'HOP': ('HOP', 2)}
    for m in [2, 4, 6, 9]: pol[f'LET m={m}'] = ('LET', m)
    for m in [4, 9]: pol[f'ML m={m}'] = ('ML', m)
    pol['Ens m=4'] = ('ENS', 4); pol['QR a=0.1'] = ('QR@0.1', 2)
    pol['CP a=0.1'] = ('CP@0.1', 2); pol['CP a=0.3'] = ('CP@0.3', 2)
    for a in [0.05, 0.1, 0.2, 0.3]: pol[f'CQR a={a}'] = (f'CQR@{a}', 2)
    kw = {}
elif mode == 'shift': pol = SHIFT_POL; kw = dict(speed=(25, 40))
elif mode == 'shiftb': pol = {'QR fixed beta=0.02': ('QR@0.02', 2)}; kw = dict(speed=(25, 40))
elif mode == 'weak':
    # quantile model trained on 2% of the training data (same subsample as the ablation), raw vs conformal
    _sub = np.random.default_rng(0).choice(len(Ytr), int(0.02 * len(Ytr)), replace=False)
    qm['weak'] = HGB(loss='quantile', quantile=0.1, max_iter=300, learning_rate=0.08).fit(Xtr[_sub], Ytr[_sub])
    corr['weak'] = np.quantile(qm['weak'].predict(Xca) - Yca, min(1, 0.9 * (1 + 1 / n)))
    INFO['weak_corr'] = float(corr['weak'])
    pol = {'QR weak a=0.1': ('QRW', 2), 'CQR weak a=0.1': ('CQRW', 2)}; kw = {}
elif mode == 'noretrain': pol = SHIFT_POL; kw = dict(scen='stale')
else:
    pol = {'ORACLE': ('ORACLE', 2), 'HOP': ('HOP', 2), 'LET m=4': ('LET', 4), 'ML m=9': ('ML', 9), 'Ens m=4': ('ENS', 4),
           'CQR a=0.1': ('CQR@0.1', 2), 'CQR a=0.2': ('CQR@0.2', 2)}
    kw = dict(N=int(mode[1:]))
os.makedirs('results5', exist_ok=True)
json.dump(INFO, open(f'results5/info{"_" + mode if mode in ("weak", "shiftb") else ""}.json', 'w'), indent=1, default=str)
for sd in seeds:
    t0 = time.time(); r = evaluate5(sd, pol, **kw)
    json.dump(r, open(f'results5/{mode}_s{sd}.json', 'w'), indent=1)
    print(mode, 'seed', sd, 'done', round(time.time() - t0), 's', flush=True)

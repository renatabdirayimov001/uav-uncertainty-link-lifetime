"""Comparison with re-implemented published schemes on the same traces as exp5 (usage: python3 exp6.py <mode> <seeds>).
mode: main (base scenario), N80, noretrain. Output: results6/<mode>_s<seed>.json"""
import sys, os, json, time, pickle
mode = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(',')]
sys.argv = ['x', '0.1']; os.environ['SCEN'] = 'base'
import numpy as np
from exp2 import *                        # mk, known, ds, HOLD, R, H, mp (point model), Xtr...
from route import hop_widest, _comp
from pubbase import *
from nplstm import LSTMReg
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

# ---------------- proposed CQR (alpha = 0.1) and reference LET, identical to exp5
qm = HGB(loss='quantile', quantile=0.1, max_iter=300, learning_rate=0.08).fit(Xtr, Ytr)
corr = np.quantile(qm.predict(Xca) - Yca, (1 - 0.1) * (1 + 1 / len(Yca)))

# ---------------- LSTM of LPMD-GPSR, trained on the training trace (seed 1)
LP = 'results6/lstm.pkl'; os.makedirs('results6', exist_ok=True)
def seq_dataset(seed, T=1500, step=3):
    P, up, life, Po, A = mk(seed, T); run, nbr = link_runs(A, HOLD); N = P.shape[1]; iu = np.triu_indices(N, 1); Xs = []; Ys = []
    for t in range(20, T - H - 1, step):
        m = known(A, t)[iu]; i = iu[0][m]; j = iu[1][m]
        Xs.append(link_seq(Po, A, run, t, i, j, nbr)); Ys.append(life[t, i, j])
    return np.concatenate(Xs), np.concatenate(Ys)
if os.path.exists(LP): lstm = pickle.load(open(LP, 'rb'))
else:
    t0 = time.time(); Xs, Ys = seq_dataset(1)
    lstm = LSTMReg(Xs.shape[-1], nh=32, layers=2, seed=0).fit(Xs, Ys, epochs=9, bs=256, lr=2e-3)
    Xv, Yv = seq_dataset(2); pv = lstm.predict(Xv)
    lstm.val = dict(mae=float(np.abs(pv - Yv).mean()), bias=float((pv - Yv).mean()), n=int(len(Yv)), train_s=time.time() - t0)
    print('LSTM val', lstm.val, flush=True); pickle.dump(lstm, open(LP, 'wb'))

def run_routing_gen(up, pf, select, flows, T, proactive, Tu=2, Trd=2, margin=2, life=None, Hh=60):
    st = [dict(route=None, outage=0) for _ in flows]
    M = dict(sec=0, deliv=0, breaks=0, disc=0, disc_re=0, disc_pro=0, hops=0, hopsn=0, slots=0, pc_n=0, pc_hit=0)
    for t in range(20, T):
        upd = (t % Tu == 0)
        if upd: W = pf(t)
        for k, (s, d) in enumerate(flows):
            f = st[k]; r = f['route']
            if r is not None and not all(up[t, r[q], r[q + 1]] for q in range(len(r) - 1)):
                M['breaks'] += 1; f['route'] = None; f['outage'] = Trd
            if f['route'] is None and f['outage'] <= 0:
                p = select(pf(t), s, d)
                if p is not None: f['route'] = p; M['disc'] += 1; M['disc_re'] += 1
            elif f['route'] is not None and upd and proactive != 'none':
                r = f['route']; b = min(W[r[q], r[q + 1]] if not np.isnan(W[r[q], r[q + 1]]) else 0 for q in range(len(r) - 1))
                if proactive == 'better' or b <= margin:
                    p = select(W, s, d)
                    if p is not None and p != r:
                        bp = min(W[p[q], p[q + 1]] for q in range(len(p) - 1))
                        if proactive != 'better' or bp > b:
                            f['route'] = p; M['disc'] += 1; M['disc_pro'] += 1
            M['slots'] += 1
            comp = _comp(up, t)
            if comp[s] == comp[d]:
                M['sec'] += 1
                r = f['route']
                if r is not None and all(up[t, r[q], r[q + 1]] for q in range(len(r) - 1)):
                    M['deliv'] += 1; M['hops'] += len(r) - 1; M['hopsn'] += 1
            f['outage'] -= 1
    return M

def evaluate6(seed, N=50, scen='base', T=int(os.environ.get('TT', 900)), nflows=15, Tu=2, speed=(10, 25)):
    os.environ['SCEN'] = scen
    P, up, life, Po, A = mk(seed, T, N=N, speed=speed)
    os.environ['SCEN'] = 'base'
    run, nbr = link_runs(A, HOLD); iu = np.triu_indices(N, 1)
    rng = np.random.default_rng(seed + 5); flows = []
    while len(flows) < nflows:
        s, d = rng.choice(N, 2, replace=False)
        if np.linalg.norm(P[10, s] - P[10, d]) > 600: flows.append((s, d))
    lk = {}
    def links(t):
        if t not in lk:
            m = known(A, t)[iu]; lk[t] = (iu[0][m], iu[1][m])
        return lk[t]
    def mkW(i, j, w):
        W = np.full((N, N), np.nan); W[i, j] = w; W[j, i] = w; return W
    lstm_cache = {}
    def lstm_dur(t):
        if t not in lstm_cache:
            i, j = links(t); lstm_cache[t] = lstm.predict(link_seq(Po, A, run, t, i, j, nbr)) if len(i) else np.zeros(0)
        return lstm_cache[t]
    est = {
        'CQR': lambda t, i, j: np.clip(qm.predict(features_age(Po, A, t, i, j, R, H)[0]) - corr, 0, H),
        'LET': lambda t, i, j: features_age(Po, A, t, i, j, R, H)[1],
        'LBOPAR': lambda t, i, j: lbopar_lifetime(Po, A, t, i, j, R, H),
        'OLSRP': lambda t, i, j: olsrp_lifetime(Po, A, t, i, j, R, H),
        'OLSRP2': lambda t, i, j: olsrp_lifetime(Po, A, t, i, j, R, H, qth=0.2),
        'LSTM': lambda t, i, j: np.clip(lstm_dur(t), 0, H),
    }
    res = {}; mins = nflows * (T - 20) / 60
    pol = {'CQR a=0.1 (proposed)': ('CQR', hop_widest, 'margin', 2), 'LET m=4': ('LET', hop_widest, 'margin', 4)}
    for w2 in [0.5, 0.8, 0.95]:
        pol[f'LB-OPAR w2={w2}'] = ('LBOPAR', (lambda W, s, d, w2=w2: lbopar_path(W, s, d, w1=1 - w2, w2=w2)), 'none', 0)
    pol['OLSR+ qth=0.05'] = ('OLSRP', hop_widest, 'better', 0); pol['OLSR+ qth=0.2'] = ('OLSRP2', hop_widest, 'better', 0)
    for nm, lab in [('LBOPAR', 'LB-OPAR LLT'), ('OLSRP', 'OLSR+ LLT'), ('LSTM', 'LPMD LSTM')]:
        for m in [2, 4, 6, 9]: pol[f'{lab} in RCLR m={m}'] = (nm, hop_widest, 'margin', m)
    for lab, (nm, sel, pro, mg) in pol.items():
        cache = {}; cov = [0, 0]
        def pf(t, nm=nm, cache=cache, cov=cov):
            if t not in cache:
                i, j = links(t); w = np.clip(est[nm](t, i, j), 0, H); cache[t] = mkW(i, j, w)
                if t <= T - H - 1: cov[0] += len(w); cov[1] += int(np.sum(life[t, i, j] >= w))
            return cache[t]
        t0 = time.time(); M = run_routing_gen(up, pf, sel, flows, T, pro, Tu=Tu, margin=mg)
        res[lab] = dict(PDR=M['deliv'] / M['sec'], PDR_uncond=M['deliv'] / M['slots'], connected=M['sec'] / M['slots'],
                        breaks=M['breaks'] / mins, disc=M['disc'] / mins, disc_re=M['disc_re'] / mins, disc_pro=M['disc_pro'] / mins,
                        hops=M['hops'] / max(1, M['hopsn']), cov_all=cov[1] / max(1, cov[0]), sec=time.time() - t0)
        print(' ', lab, round(res[lab]['PDR'], 3), round(time.time() - t0), 's', flush=True)
    # ---- LPMD-GPSR protocol (stateless greedy forwarding, LSTM durations predicted every slot)
    for delta in [1, 2, 4]:
        t0 = time.time(); M = dict(sec=0, deliv=0, slots=0, hops=0, hopsn=0, drop=0); covl = [0, 0]
        for t in range(20, T):
            i, j = links(t); dur = lstm_dur(t)
            if delta == 1 and t <= T - H - 1: covl[0] += len(dur); covl[1] += int(np.sum(life[t, i, j] >= dur))
            pdur = {}
            for a_, b_, v in zip(i, j, dur): pdur[(a_, b_)] = v; pdur[(b_, a_)] = v
            kt = known(A, t); comp = _comp(up, t)
            for s, d in flows:
                M['slots'] += 1
                if comp[s] != comp[d]: continue
                M['sec'] += 1
                p = gpsr_path(t, s, d, Po, A, up, kt, pdur, delta=delta)
                if p is None: M['drop'] += 1; continue
                if all(up[t, p[q], p[q + 1]] for q in range(len(p) - 1)):
                    M['deliv'] += 1; M['hops'] += len(p) - 1; M['hopsn'] += 1
        res[f'LPMD-GPSR delta={delta}'] = dict(PDR=M['deliv'] / M['sec'], PDR_uncond=M['deliv'] / M['slots'], connected=M['sec'] / M['slots'],
                                breaks=float('nan'), disc=float('nan'), disc_re=float('nan'), disc_pro=float('nan'),
                                hops=M['hops'] / max(1, M['hopsn']), drop_rate=M['drop'] / M['sec'], sec=time.time() - t0,
                                **({'cov_all': covl[1] / max(1, covl[0])} if delta == 1 else {}))
        print('  LPMD-GPSR', delta, round(res[f'LPMD-GPSR delta={delta}']['PDR'], 3), round(time.time() - t0), 's', flush=True)
    return res

kw = dict(N=80, T=600) if mode == 'N80' else dict(scen='stale') if mode == 'noretrain' else dict(speed=(25, 40)) if mode == 'shift' else {}
for sd in seeds:
    t0 = time.time(); r = evaluate6(sd, **kw)
    json.dump(r, open(f'results6/{mode}_s{sd}.json', 'w'), indent=1)
    print(mode, 'seed', sd, 'done', round(time.time() - t0), 's', flush=True)

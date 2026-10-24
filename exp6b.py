"""LPAR (Li & Sun 2025) on the same traces: AODV-style TILS routes, greedy forwarding while a broken route is repaired."""
import sys, os, json, time
mode = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(',')]
sys.argv = ['x', '0.1']; os.environ['SCEN'] = 'base'
import numpy as np
from exp2 import mk, known, HOLD, R, H
from route import _comp
from pubbase import *

def evaluate_lpar(seed, N=50, scen='base', T=900, nflows=15, Tu=2, Trd=2, speed=(10, 25)):
    os.environ['SCEN'] = scen; P, up, life, Po, A = mk(seed, T, N=N, speed=speed); os.environ['SCEN'] = 'base'
    iu = np.triu_indices(N, 1)
    rng = np.random.default_rng(seed + 5); flows = []
    while len(flows) < nflows:
        s, d = rng.choice(N, 2, replace=False)
        if np.linalg.norm(P[10, s] - P[10, d]) > 600: flows.append((s, d))
    cache = {}
    def Wt(t):
        if t not in cache:
            m = known(A, t)[iu]; i = iu[0][m]; j = iu[1][m]; w = lpar_tils(Po, A, t, i, j, R)
            W = np.full((N, N), np.nan); W[i, j] = w; W[j, i] = w; cache[t] = W
        return cache[t]
    st = [dict(route=None, outage=0) for _ in flows]; M = dict(sec=0, deliv=0, deliv_greedy=0, breaks=0, disc=0, slots=0, hops=0, hopsn=0)
    big = {}
    for t in range(20, T):
        kt = known(A, t); comp = _comp(up, t)
        for k, (s, d) in enumerate(flows):
            f = st[k]; r = f['route']
            if r is not None and not all(up[t, r[q], r[q + 1]] for q in range(len(r) - 1)):
                M['breaks'] += 1; f['route'] = None; f['outage'] = Trd
            if f['route'] is None and f['outage'] <= 0:
                p = lpar_path(Wt(t), s, d)
                if p is not None: f['route'] = p; M['disc'] += 1
            M['slots'] += 1
            if comp[s] == comp[d]:
                M['sec'] += 1; r = f['route']
                if r is not None and all(up[t, r[q], r[q + 1]] for q in range(len(r) - 1)):
                    M['deliv'] += 1; M['hops'] += len(r) - 1; M['hopsn'] += 1
                elif r is None:                         # greedy forwarding of packets while the route is repaired
                    g = gpsr_path(t, s, d, Po, A, up, kt, big, delta=-1, w=(0, 1, 0))
                    if g is not None and all(up[t, g[q], g[q + 1]] for q in range(len(g) - 1)):
                        M['deliv'] += 1; M['deliv_greedy'] += 1; M['hops'] += len(g) - 1; M['hopsn'] += 1
            f['outage'] -= 1
    mins = nflows * (T - 20) / 60
    return {'LPAR': dict(PDR=M['deliv'] / M['sec'], PDR_uncond=M['deliv'] / M['slots'], connected=M['sec'] / M['slots'],
                         breaks=M['breaks'] / mins, disc=M['disc'] / mins, disc_re=M['disc'] / mins, disc_pro=0.0,
                         hops=M['hops'] / max(1, M['hopsn']), greedy_share=M['deliv_greedy'] / max(1, M['deliv']))}

kw = dict(N=80, T=600) if mode == 'N80' else dict(scen='stale') if mode == 'noretrain' else dict(speed=(25, 40)) if mode == 'shift' else {}
if 'TT' in os.environ: kw['T'] = int(os.environ['TT'])
for sd in seeds:
    t0 = time.time(); r = evaluate_lpar(sd, **kw)
    json.dump(r, open(f'results6/lpar_{mode}_s{sd}.json', 'w'), indent=1)
    print(mode, 'seed', sd, round(r['LPAR']['PDR'], 3), 'greedy', round(r['LPAR']['greedy_share'], 3), round(time.time() - t0), 's', flush=True)

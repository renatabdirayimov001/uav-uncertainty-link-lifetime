"""Simplified re-implementation of RCPLA-FANET (Mohammadzadeh Dogaheh et al., Research Square preprint, 2026,
doi:10.21203/rs.3.rs-10921375/v1) in the common slot-level simulator, on the same traces and flows as exp5/exp6.

  * hybrid lifetime estimate (their Eq. 3): T_hat = max{0, T_geo + rho * T_res * clip(theta^T phi, -1, 1)},
    T_geo = kinematic LET from the (stale, dead-reckoned) HELLO state, theta learned by recursive least squares from
    completed link episodes, rho = n / (n + n0) limits the early residual influence;
  * rolling one-sided calibration: residuals r = [T_hat - T]_+ of completed episodes, window W = 80,
    q_t = higher empirical (1 - alpha) quantile, bound T_tilde = max{0, T_hat - q_t} (0.7 T_hat while |W| < 15);
  * hop-by-hop route construction: admissible next hops are unvisited, make geographic progress and have T_tilde >= tau (if none is admissible,
    all progress-making neighbours are ranked by the same score as a recovery fallback);
    score = 0.45 progress + 0.30 survival + 0.15 packet success (HELLO freshness proxy), normalised (no energy model);
    learning automata act on the candidates within delta = 0.08 of the best score (softmax prior blended with
    experience, zeta = 0.18; best action with prob. 0.75, otherwise sampled; reward 0.12 / penalty 0.08);
  * pre-emptive repair: when the bound of a hop of the active route falls below 0.8 s, the upstream UAV splices in an
    admissible two-hop detour that avoids the threatened hop.
Simplifications: one network-wide RLS model and calibration window (instead of per-UAV), no energy, no queueing
delay in tau, slot-level delivery as for all other schemes. Usage: python3 exp7.py <mode> <seeds>"""
import sys, os, json, time
from collections import deque
mode = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(',')]
sys.argv = ['x', '0.1']; os.environ['SCEN'] = 'base'
import numpy as np
from exp2 import mk, known, HOLD, R, H
from sim import features_age
from route import _comp

T_RES, N0, WIN, WMIN = 10.0, 20, 80, 15
W_PROG, W_SURV, W_PKT = 0.45, 0.30, 0.15
DELTA, ZETA, P_BEST, A_R, A_P, T_REPAIR = 0.08, 0.18, 0.75, 0.12, 0.08, 0.8

class RLS:
    def __init__(self, n, lam=0.995):
        self.th = np.zeros(n); self.P = np.eye(n) * 100.0; self.lam = lam; self.n = 0
    def update(self, phi, y):
        Pp = self.P @ phi; k = Pp / (self.lam + phi @ Pp)
        self.th += k * (y - phi @ self.th); self.P = (self.P - np.outer(k, Pp)) / self.lam; self.n += 1

def phi_of(X):
    """six residual features: d/R, radial speed, relative speed, HELLO age, |dz|, manoeuvre indicator (+ bias)"""
    return np.c_[X[:, 0], X[:, 1] / 30, X[:, 2] / 30, X[:, 4] / HOLD, X[:, 7] / 100, X[:, 8] / 5, np.ones(len(X))]

def evaluate7(seed, N=50, scen='base', T=int(os.environ.get('TT', 900)), nflows=15, Tu=2, Trd=2, speed=(10, 25),
              variants=((0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.2, 0.5))):
    os.environ['SCEN'] = scen
    P, up, life, Po, A = mk(seed, T, N=N, speed=speed)
    os.environ['SCEN'] = 'base'
    iu = np.triu_indices(N, 1)
    rng0 = np.random.default_rng(seed + 5); flows = []
    while len(flows) < nflows:
        s, d = rng0.choice(N, 2, replace=False)
        if np.linalg.norm(P[10, s] - P[10, d]) > 600: flows.append((s, d))
    feat = {}
    def F(t):
        if t not in feat:
            m = known(A, t)[iu]; i, j = iu[0][m], iu[1][m]
            X, let = features_age(Po, A, t, i, j, R, H) if len(i) else (np.zeros((0, 10)), np.zeros(0))
            feat[t] = (i, j, X, let, phi_of(X))
        return feat[t]
    res = {}; mins = nflows * (T - 20) / 60
    for alpha, tau in variants:
        rng = np.random.default_rng(seed * 100 + int(alpha * 100) + int(tau * 10))
        rls = RLS(7); win = deque(maxlen=WIN); pending = {}        # completion time -> list of (phi, Tgeo, That, Ytrue)
        la = {}                                                   # (node, dest) -> {neighbour: experience prob}
        cov = [0, 0]; B = None
        def bounds(t):
            i, j, X, let, ph = F(t)
            rho = rls.n / (rls.n + N0)
            th = np.maximum(0, let + rho * T_RES * np.clip(ph @ rls.th, -1, 1))
            if len(win) >= WMIN:
                q = np.quantile(np.array(win), 1 - alpha, method='higher'); Lb = np.maximum(0, th - q)
            else: Lb = 0.7 * th
            Lb = np.minimum(Lb, H)
            # schedule completed-episode feedback (a random subset of the current links, outcome known at t + Y)
            Y = life[t, i, j]; pick = rng.random(len(i)) < 0.1
            for k in np.nonzero(pick & (Y < H))[0]:
                pending.setdefault(t + int(Y[k]), []).append((ph[k], let[k], th[k], Y[k]))
            if t <= T - H - 1: cov[0] += len(Lb); cov[1] += int(np.sum(Y >= Lb))
            Bm = {}
            for a_, b_, v in zip(i, j, Lb): Bm[(a_, b_)] = v; Bm[(b_, a_)] = v
            return Bm
        def feedback(t):
            for ph, tg, th, y in pending.pop(t, []):
                win.append(max(th - y, 0.0)); rls.update(ph, float(np.clip((y - tg) / T_RES, -1, 1)))
        def candidates(t, k, d, path, kt, Bm):
            nb = np.nonzero(kt[k])[0]; nb = nb[(nb != k) & ~np.isin(nb, path)]
            if len(nb) == 0: return nb, None
            pd_ = Po[t, d]; a = A[t, k, nb]; tj = t - a
            vj = (Po[tj, nb] - Po[tj - 3, nb]) / 3; pj = Po[tj, nb] + vj * a[:, None]
            dkd = np.linalg.norm(Po[t, k] - pd_); djd = np.linalg.norm(pj - pd_, axis=1)
            surv = np.array([Bm.get((k, j), 0.0) for j in nb])
            ok = (djd < dkd) & (surv >= tau)
            if d in nb: ok |= (nb == d) & (surv >= tau)
            if not ok.any():                                # empty risk-feasible set: fall back to progress-making
                ok = (djd < dkd)                             # neighbours, ranked by the same score (recovery mode)
            nb, djd, surv, a = nb[ok], djd[ok], surv[ok], a[ok]
            if len(nb) == 0: return nb, None
            prog = (dkd - djd) / R; pkt = 1 - a / HOLD
            norm = lambda v: v / max(np.max(v), 1e-9)
            sc = (W_PROG * norm(prog) + W_SURV * norm(surv) + W_PKT * norm(np.maximum(pkt, 0))) / (W_PROG + W_SURV + W_PKT)
            return nb, sc
        def choose(k, d, nb, sc):
            if d in nb: return int(d)
            keep = sc >= sc.max() - DELTA; nb, sc = nb[keep], sc[keep]
            prior = np.exp(sc / 0.05); prior /= prior.sum()
            ex = la.setdefault((k, d), {}); e = np.array([ex.get(int(j), 1.0 / len(nb)) for j in nb]); e /= e.sum()
            p = (1 - ZETA) * prior + ZETA * e; p /= p.sum()
            return int(nb[np.argmax(p)]) if rng.random() < P_BEST else int(rng.choice(nb, p=p))
        def reinforce(k, d, j, good):
            ex = la.setdefault((k, d), {}); ks = list(ex) or [j]
            if j not in ex: ex[j] = 1.0 / max(1, len(ks)); ks = list(ex)
            for m in ks:
                if good: ex[m] = ex[m] + A_R * (1 - ex[m]) if m == j else ex[m] * (1 - A_R)
                else: ex[m] = ex[m] * (1 - A_P) if m == j else ex[m] * (1 - A_P) + A_P / max(1, len(ks) - 1)
        def build(t, s, d, Bm):
            kt = known(A, t); path = [s]; k = s
            for _ in range(15):
                nb, sc = candidates(t, k, d, path, kt, Bm)
                if sc is None: return None
                j = choose(k, d, nb, sc); path.append(j); k = j
                if j == d: return path
            return None
        st = [dict(route=None, outage=0) for _ in flows]
        M = dict(sec=0, deliv=0, breaks=0, disc=0, disc_re=0, disc_pro=0, hops=0, hopsn=0, slots=0, repairs=0, fail=0, noroute=0, outage=0)
        t0 = time.time()
        for t in range(20, T):
            feedback(t)
            upd = (t % Tu == 0)
            if upd or B is None: B = bounds(t)
            for fi, (s, d) in enumerate(flows):
                f = st[fi]; r = f['route']
                if r is not None and not all(up[t, r[q], r[q + 1]] for q in range(len(r) - 1)):
                    q = next(q for q in range(len(r) - 1) if not up[t, r[q], r[q + 1]])
                    reinforce(r[q], d, r[q + 1], False)
                    M['breaks'] += 1; f['route'] = None; f['outage'] = Trd
                if f['route'] is None and f['outage'] <= 0:
                    p = build(t, s, d, B)
                    if p is not None: f['route'] = p; M['disc'] += 1; M['disc_re'] += 1
                    else: M['fail'] += 1
                elif f['route'] is not None and upd:
                    r = f['route']
                    for q in range(len(r) - 1): reinforce(r[q], d, r[q + 1], True)
                    for q in range(len(r) - 1):                     # pre-emptive two-hop repair of a threatened hop
                        if B.get((r[q], r[q + 1]), 0.0) < T_REPAIR:
                            k, nxt = r[q], r[q + 1]; kt = known(A, t); best = None
                            for m in np.nonzero(kt[k])[0]:
                                if m in r or not kt[m, nxt]: continue
                                b = min(B.get((k, m), 0.0), B.get((m, nxt), 0.0))
                                if b >= max(tau, T_REPAIR) and (best is None or b > best[0]): best = (b, int(m))
                            if best is not None:
                                f['route'] = r[:q + 1] + [best[1]] + r[q + 1:]; M['disc'] += 1; M['disc_pro'] += 1; M['repairs'] += 1
                            break
                M['slots'] += 1
                comp = _comp(up, t)
                if comp[s] == comp[d]:
                    M['sec'] += 1; r = f['route']
                    if r is None: M['noroute' if f['outage'] <= 0 else 'outage'] += 1
                    if r is not None and all(up[t, r[q], r[q + 1]] for q in range(len(r) - 1)):
                        M['deliv'] += 1; M['hops'] += len(r) - 1; M['hopsn'] += 1
                f['outage'] -= 1
        lab = f'RCPLA a={alpha} tau={tau}'
        res[lab] = dict(PDR=M['deliv'] / M['sec'], PDR_uncond=M['deliv'] / M['slots'], connected=M['sec'] / M['slots'],
                        breaks=M['breaks'] / mins, disc=M['disc'] / mins, disc_re=M['disc_re'] / mins, disc_pro=M['disc_pro'] / mins,
                        hops=M['hops'] / max(1, M['hopsn']), cov_all=cov[1] / max(1, cov[0]), sec=time.time() - t0,
                        noroute_frac=M['noroute'] / max(1, M['sec']), outage_frac=M['outage'] / max(1, M['sec']), repairs=M['repairs'] / mins)
        print(' ', lab, {k: round(v, 3) for k, v in res[lab].items()}, flush=True)
    return res

if __name__ == '__main__':
    kw = dict(N=80, T=600) if mode == 'N80' else dict(scen='stale') if mode == 'noretrain' else dict(speed=(25, 40)) if mode == 'shift' else {}
    os.makedirs('results7', exist_ok=True)
    for sd in seeds:
        t0 = time.time(); r = evaluate7(sd, **kw)
        json.dump(r, open(f'results7/{mode}_s{sd}.json', 'w'), indent=1)
        print(mode, 'seed', sd, 'done', round(time.time() - t0), 's', flush=True)

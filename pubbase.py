"""Re-implementations of published link-lifetime-aware FANET routing schemes in the common simulator.

LB-OPAR  : Gharib et al., Ad Hoc Networks 132 (2022) 102878. Link lifetime from three consecutive positions
           (speed + acceleration, first exit time from range), path cost w1*hops + w2/path-lifetime (+ w3*load),
           optimal path by the edge-removal/BFS procedure of their Algorithm 1; re-routing when the route fails.
OLSR+    : Rahmani et al., Vehicular Communications 36 (2022) 100489. Link quality Q(t)=Q0*exp(s*t),
           s = a1*D + a2*V' + a3*phi (distance, relative-velocity and direction scores, a1=a2=a3=1/3),
           link lifetime = time until Q falls to Q_th; route = maximum route lifetime (min link lifetime);
           link-state routing table recomputed every update period.
LPMD-GPSR: Dong & Tang, Computer Communications 250 (2026) 108458. Two-layer LSTM link-duration predictor
           on L=5 consecutive observations; greedy geographic forwarding: candidates must stay in range at
           t+delta (predicted duration > delta) and be closer to the destination; score
           0.35*E + 0.4*(d_id/d_jd) + 0.25*cos(theta_j); packets without a candidate are dropped.
"""
import numpy as np
from collections import deque

R_ = 350.; H_ = 60

# ------------------------------------------------------------------ helpers on stale HELLO information
def hello_state(Po, A, t, i, j):
    """per link: own position/velocity of i at t, and j's position/velocity/acceleration from the latest HELLO (age a)."""
    a = A[t, i, j]; tj = t - a
    pi2, pi1, pi0 = Po[t, i], Po[t - 1, i], Po[t - 2, i]
    pj2, pj1, pj0 = Po[tj, j], Po[tj - 1, j], Po[tj - 2, j]
    return a, (pi0, pi1, pi2), (pj0, pj1, pj2)

# ------------------------------------------------------------------ LB-OPAR link lifetime (three positions)
def lbopar_lifetime(Po, A, t, i, j, R=R_, H=H_, dtau=0.5):
    a, (pi0, pi1, pi2), (pj0, pj1, pj2) = hello_state(Po, A, t, i, j)
    vi = pi2 - pi1; ai = pi2 - 2 * pi1 + pi0
    vj = pj2 - pj1; aj = pj2 - 2 * pj1 + pj0
    taus = np.arange(0, H + dtau, dtau)
    out = np.full(len(a), float(H))
    alive = np.ones(len(a), bool)
    for tau in taus:                                     # first time the predicted distance exceeds R
        Pi = pi2 + vi * tau + 0.5 * ai * tau ** 2
        ta = (a + tau)[:, None]
        Pj = pj2 + vj * ta + 0.5 * aj * ta ** 2
        d = np.linalg.norm(Pj - Pi, axis=1)
        hit = alive & (d > R)
        out[hit] = tau; alive &= ~hit
        if not alive.any(): break
    return out

def lbopar_path(W, s, d, w1=0.5, w2=0.5, Hmax=10):
    """Algorithm 1 of LB-OPAR (load weight w3=0, light traffic): BFS shortest path, then repeatedly remove all
    links whose cost 1/lifetime is >= the worst link of the current path and search again; keep the path with
    the lowest w1*hops + w2/path-lifetime."""
    N = W.shape[0]; A = ~np.isnan(W)
    lt = np.where(A, np.maximum(W, 1e-3), np.nan)
    best = None; bestv = np.inf; thr = -np.inf
    while True:
        Aa = A & (lt > thr)
        p = bfs(Aa, s, d, Hmax)
        if p is None: break
        plt = min(lt[p[k], p[k + 1]] for k in range(len(p) - 1))
        v = w1 * (len(p) - 1) + w2 / plt
        if v < bestv: best, bestv = p, v
        thr = plt                                        # remove links with lifetime <= path lifetime
    return best

def bfs(Aa, s, d, Hmax=10):
    N = Aa.shape[0]; prev = np.full(N, -1); seen = np.zeros(N, bool); seen[s] = True; q = deque([(s, 0)])
    while q:
        u, h = q.popleft()
        if u == d: break
        if h >= Hmax: continue
        for v in np.nonzero(Aa[u] & ~seen)[0]:
            seen[v] = True; prev[v] = u; q.append((v, h + 1))
    if not seen[d]: return None
    p = [d]
    while p[-1] != s: p.append(prev[p[-1]])
    return p[::-1]

# ------------------------------------------------------------------ OLSR+ link lifetime (quality decay)
def olsrp_lifetime(Po, A, t, i, j, R=R_, H=H_, qth=0.05):
    a, (pi0, pi1, pi2), (pj0, pj1, pj2) = hello_state(Po, A, t, i, j)
    vi = pi2 - pi1; vj = pj2 - pj1
    d = np.linalg.norm(pj2 - pi2, axis=1)               # distance at HELLO reception (as in OLSR+)
    dhm, dlm = 0.75 * R, 1.5 * R                         # D_H-Min = 3R/4, D_L-Max = 3/4 * 2R
    D = np.where(d <= dhm, 1.0, np.where(d >= dlm, -1.0, -2 * (d - dhm) / (dlm - dhm) + 1))
    si, sj = np.linalg.norm(vi, axis=1), np.linalg.norm(vj, axis=1); vij = np.linalg.norm(vi - vj, axis=1)
    V = 1 - 2 * vij / np.maximum(si + sj, 1e-6); V = np.clip(V, -1, 1)
    cth = (vi * vj).sum(1) / np.maximum(si * sj, 1e-6); th = np.arccos(np.clip(cth, -1, 1))
    phi = 1 - 2 * th / np.pi
    s = (D + V + phi) / 3
    q0 = np.clip(1 - d / R, 1e-3, 1)                     # initial quality from the received signal (disk model: distance proxy)
    T = np.where(s < 0, np.log(qth / q0) / np.where(s < 0, s, -1) - a, H)
    return np.clip(T, 0, H)

# ------------------------------------------------------------------ LPMD-GPSR: LSTM sequence features
SEQ = 5
def link_seq(Po, A, run, t, i, j, nbr):
    """sequence (len SEQ) of per-link features observed by i at t-SEQ+1..t:
    relative position (3), relative velocity (3), |v_j|, azimuth and elevation of v_j, elapsed link duration, neighbour count of i."""
    seq = []
    for k in range(SEQ - 1, -1, -1):
        tt = t - k; a = A[tt, i, j]; tj = tt - a
        pi = Po[tt, i]; vi = (Po[tt, i] - Po[tt - 3, i]) / 3
        vj = (Po[tj, j] - Po[tj - 3, j]) / 3; pj = Po[tj, j] + vj * a[:, None]
        dp = (pj - pi) / R_; dv = (vj - vi) / 20; sj = np.linalg.norm(vj, axis=1)
        az = np.arctan2(vj[:, 1], vj[:, 0]); el = np.arctan2(vj[:, 2], np.linalg.norm(vj[:, :2], axis=1) + 1e-9)
        seq.append(np.c_[dp, dv, sj / 20, np.cos(az), np.sin(az), el, np.minimum(run[tt, i, j], 60) / 60, nbr[tt, i] / 20])
    return np.stack(seq, 1)

def link_runs(A, hold=4):
    """elapsed duration (s) of each link as seen in the neighbour table, and neighbour counts."""
    known = A <= hold; T = A.shape[0]; run = np.zeros(A.shape); cur = np.zeros(A.shape[1:])
    for t in range(T):
        cur = np.where(known[t], cur + 1, 0); run[t] = cur
    nbr = known.sum(2) - 1
    return run, nbr

def gpsr_path(t, s, d, Po, A, up, known_t, pred_dur, delta=1.0, Hmax=15, w=(0.35, 0.4, 0.25)):
    """LPMD-GPSR greedy forwarding of one packet at slot t; pred_dur[(k,j)] = LSTM-predicted link duration seen by k.
    Returns the hop list or None when the packet is dropped (no candidate)."""
    path = [s]; k = s
    pd_ = Po[t, d]                                       # destination position from a location service (noisy GPS)
    for _ in range(Hmax):
        if known_t[k, d]: path.append(d); return path
        nb = np.nonzero(known_t[k])[0]; nb = nb[nb != k]
        if len(nb) == 0: return None
        a = A[t, k, nb]; tj = t - a
        vj = (Po[tj, nb] - Po[tj - 3, nb]) / 3; pj = Po[tj, nb] + vj * a[:, None]
        dkd = np.linalg.norm(Po[t, k] - pd_); djd = np.linalg.norm(pj - pd_, axis=1)
        dur = np.array([pred_dur.get((k, j), 0.0) for j in nb])
        ok = (dur > delta) & (djd < dkd) & ~np.isin(nb, path)
        if not ok.any(): return None
        cos = ((vj * (pd_ - pj)).sum(1)) / np.maximum(np.linalg.norm(vj, axis=1) * np.maximum(djd, 1e-6), 1e-6)
        sc = w[0] * 1.0 + w[1] * dkd / np.maximum(djd, 1e-6) + w[2] * cos
        sc[~ok] = -np.inf
        k = int(nb[np.argmax(sc)]); path.append(k)
        if k == d: return path
    return None

# ------------------------------------------------------------------ LPAR (Li & Sun, Wireless Networks 31, 2025)
def lpar_tils(Po, A, t, i, j, R=R_, k=2, vmax=30.0):
    """time-interval link stability L_ij = sum_{n=0..k} D^n * Y^n with acceleration-based position prediction (Eqs. 2-10)."""
    a, (pi0, pi1, pi2), (pj0, pj1, pj2) = hello_state(Po, A, t, i, j)
    vi = pi2 - pi1; ai = vi - (pi1 - pi0)
    vj = pj2 - pj1; aj = vj - (pj1 - pj0)
    dprev = np.linalg.norm((pj1 + vj * a[:, None]) - pi1, axis=1)
    L = np.zeros(len(a))
    for n in range(k + 1):
        Pi = pi2 + vi * n + 0.5 * ai * n ** 2; Vi = vi + ai * n
        ta = (a + n)[:, None]; Pj = pj2 + vj * ta + 0.5 * aj * ta ** 2; Vj = vj + aj * ta
        d = np.linalg.norm(Pj - Pi, axis=1)
        D = np.where(d <= R, (R - d) / R, 0.0)
        rv = np.linalg.norm(Vi - Vj, axis=1)
        mu = np.sign(d - dprev)                         # +1 moving apart, -1 approaching
        Y = np.clip(0.5 - mu * rv / (4 * vmax), 0, 1)   # approaching links are more stable (intent of their Fig. 2)
        L += D * Y; dprev = d
    return L

def lpar_path(W, s, d, Hmax=10):
    """AODV-style discovery with the TILS metric: RREQs propagate hop by hop; each node keeps the reverse route
    through the previous-hop neighbour with the largest link stability; RREP follows the reverse route."""
    N = W.shape[0]; Aa = ~np.isnan(W); lvl = np.full(N, -1); lvl[s] = 0; par = np.full(N, -1); fr = [s]
    for h in range(1, Hmax + 1):
        nxt = []
        for v in range(N):
            if lvl[v] != -1: continue
            cand = [u for u in fr if Aa[u, v]]
            if cand:
                par[v] = max(cand, key=lambda u: W[u, v]); nxt.append(v)
        for v in nxt: lvl[v] = h
        if lvl[d] != -1 or not nxt: break
        fr = nxt
    if lvl[d] == -1: return None
    p = [d]
    while p[-1] != s: p.append(par[p[-1]])
    return p[::-1]

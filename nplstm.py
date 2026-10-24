"""Minimal two-layer LSTM regressor in NumPy (forward + BPTT + Adam), used to re-implement the
LSTM link-duration predictor of LPMD-GPSR (Dong & Tang, Computer Communications 2026) without a DL framework."""
import numpy as np

def sig(x): return 1 / (1 + np.exp(-x))

class LSTMReg:
    def __init__(self, nin, nh=32, layers=2, seed=0):
        rng = np.random.default_rng(seed); self.nh = nh; self.P = {}
        for l in range(layers):
            ni = nin if l == 0 else nh
            s = 1 / np.sqrt(ni + nh)
            self.P[f'W{l}'] = rng.normal(0, s, (ni + nh, 4 * nh))
            b = np.zeros(4 * nh); b[nh:2 * nh] = 1.0           # forget-gate bias 1
            self.P[f'b{l}'] = b
        self.P['Wy'] = rng.normal(0, 1 / np.sqrt(nh), (nh, 1)); self.P['by'] = np.zeros(1)
        self.L = layers
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}; self.v = {k: np.zeros_like(v) for k, v in self.P.items()}; self.it = 0

    def forward(self, X, keep=False):
        B, T, _ = X.shape; nh = self.nh; inp = X; caches = []
        for l in range(self.L):
            W, b = self.P[f'W{l}'], self.P[f'b{l}']
            h = np.zeros((B, nh)); c = np.zeros((B, nh)); hs = []; cl = []
            for t in range(T):
                z = np.concatenate([inp[:, t], h], 1); g = z @ W + b
                i, f, o, gg = sig(g[:, :nh]), sig(g[:, nh:2 * nh]), sig(g[:, 2 * nh:3 * nh]), np.tanh(g[:, 3 * nh:])
                cp = c; c = f * c + i * gg; tc = np.tanh(c); h = o * tc
                hs.append(h)
                if keep: cl.append((z, i, f, o, gg, cp, tc))
            caches.append(cl); inp = np.stack(hs, 1)
        y = h @ self.P['Wy'] + self.P['by']
        if keep: self.cache = (caches, h, X.shape)
        return y[:, 0]

    def backward(self, dy):
        caches, hlast, (B, T, _) = self.cache; nh = self.nh; G = {k: np.zeros_like(v) for k, v in self.P.items()}
        dy = dy[:, None]; G['Wy'] = hlast.T @ dy; G['by'] = dy.sum(0)
        dh_out = [None] * T; dh_out[T - 1] = dy @ self.P['Wy'].T
        for l in range(self.L - 1, -1, -1):
            W = self.P[f'W{l}']; ni = W.shape[0] - nh
            dh_next = np.zeros((B, nh)); dc_next = np.zeros((B, nh)); dinp = [None] * T
            for t in range(T - 1, -1, -1):
                z, i, f, o, gg, cp, tc = caches[l][t]
                dh = dh_next + (dh_out[t] if dh_out[t] is not None else 0)
                do = dh * tc; dc = dc_next + dh * o * (1 - tc ** 2)
                di = dc * gg; dg = dc * i; df = dc * cp; dc_next = dc * f
                dgate = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - gg ** 2)], 1)
                G[f'W{l}'] += z.T @ dgate; G[f'b{l}'] += dgate.sum(0)
                dz = dgate @ W.T; dinp[t] = dz[:, :ni]; dh_next = dz[:, ni:]
            dh_out = dinp
        return G

    def step(self, G, lr=1e-3, b1=0.9, b2=0.999, clip=5.0):
        self.it += 1
        nrm = np.sqrt(sum((g ** 2).sum() for g in G.values()))
        sc = min(1.0, clip / (nrm + 1e-12))
        for k in self.P:
            g = G[k] * sc
            self.m[k] = b1 * self.m[k] + (1 - b1) * g; self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.it); vh = self.v[k] / (1 - b2 ** self.it)
            self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, X, y, epochs=8, bs=256, lr=2e-3, seed=0, verbose=True):
        self.mu = X.reshape(-1, X.shape[-1]).mean(0); self.sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        self.ym, self.ys = y.mean(), y.std()
        Xn = (X - self.mu) / self.sd; yn = (y - self.ym) / self.ys
        rng = np.random.default_rng(seed); n = len(y)
        for ep in range(epochs):
            idx = rng.permutation(n); tot = 0
            for k in range(0, n, bs):
                b = idx[k:k + bs]; p = self.forward(Xn[b], keep=True); e = p - yn[b]
                tot += (e ** 2).sum(); G = self.backward(2 * e / len(b)); self.step(G, lr=lr * (0.5 ** (ep // 3)))
            if verbose: print(f'  LSTM epoch {ep + 1}: mse(norm)={tot / n:.4f}', flush=True)
        return self

    def predict(self, X, bs=4096):
        out = [self.forward((X[k:k + bs] - self.mu) / self.sd) for k in range(0, len(X), bs)]
        return np.concatenate(out) * self.ys + self.ym

if __name__ == '__main__':      # gradient check
    m = LSTMReg(3, nh=4, layers=2, seed=1); X = np.random.default_rng(0).normal(size=(5, 4, 3)); y = np.random.default_rng(1).normal(size=5)
    p = m.forward(X, keep=True); G = m.backward(2 * (p - y) / 5)
    for k in ['W0', 'W1', 'b1', 'Wy']:
        idx = (0,) * m.P[k].ndim; old = m.P[k][idx]; eps = 1e-6
        m.P[k][idx] = old + eps; lp = ((m.forward(X) - y) ** 2).mean(); m.P[k][idx] = old - eps; lm = ((m.forward(X) - y) ** 2).mean(); m.P[k][idx] = old
        print(k, G[k][idx], (lp - lm) / (2 * eps))

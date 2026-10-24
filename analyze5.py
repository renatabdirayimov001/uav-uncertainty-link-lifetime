"""Summaries, paired CIs and figures for the revision (reads results5/)."""
import json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R5 = 'results5'; OUT = 'figures'; os.makedirs(OUT, exist_ok=True)
VAL = [7, 8, 9, 10]; TEST = list(range(11, 23))
C = dict(CQR='#2a78d6', LET='#eb6834', ML='#1baf7a', ENS='#eda100', CP='#e87ba4', QR='#6250d6', STATIC='#008300', ORACLE='#55534e', HOP='#9a988f')
MK = dict(CQR='o', LET='s', ML='^', ENS='D', CP='v', QR='P', STATIC='h', ORACLE='*', HOP='x')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e4e3df',
                     'grid.linewidth': 0.6, 'axes.edgecolor': '#8a887f', 'legend.frameon': False, 'lines.linewidth': 2, 'pdf.fonttype': 42})

def load(mode, seeds=None):
    fs = sorted(glob.glob(f'{R5}/{mode}_s*.json'))
    out = {}
    for f in fs:
        sd = int(f.split('_s')[-1].split('.')[0])
        if seeds is None or sd in seeds: out[sd] = json.load(open(f))
    return out

def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2: return [float(v.mean()), float('nan'), float('nan')]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]

METRICS = ['PDR', 'PDR_uncond', 'connected', 'breaks', 'disc', 'disc_re', 'disc_pro', 'hops', 'cov_all', 'link_cov_used', 'path_cov', 'kept_fail_rate']
def summarize(D, keys=None):
    keys = keys or list(next(iter(D.values())).keys())
    return {k: {m: ci([D[s][k][m] for s in D]) for m in METRICS} for k in keys}
def paired(D, a, b, m):
    return ci([D[s][a][m] - D[s][b][m] for s in D]), int(sum((D[s][a][m] > D[s][b][m]) if m in ('PDR', 'PDR_uncond') else (D[s][a][m] < D[s][b][m]) for s in D)), len(D)
def hopbins(D, k):
    agg = {h: np.zeros(4) for h in ['1', '2', '3', '4', '5+']}
    for s in D:
        for h, v in D[s][k]['hopbins'].items(): agg[h] += np.array(v)
    return {h: dict(n=int(v[0]), path_cov=float(v[1] / max(1, v[0])), kept=int(v[2]), kept_fail=float(v[3] / max(1, v[2]))) for h, v in agg.items()}

res = {}
V = load('main', VAL); T = load('main', TEST)
if len(V) >= 1 and len(T) >= 2:
    fam = {'LET': [f'LET m={m}' for m in [2, 4, 6, 9]], 'ML': ['ML m=4', 'ML m=9'], 'CP': ['CP a=0.1', 'CP a=0.3'],
           'CQR': [f'CQR a={a}' for a in [0.05, 0.1, 0.2, 0.3]]}
    chosen = {f: max(ks, key=lambda k: np.mean([V[s][k]['PDR'] for s in V])) for f, ks in fam.items()} if V else {}
    chosen.update({'ENS': 'Ens m=4', 'QR': 'QR a=0.1', 'ORACLE': 'ORACLE', 'HOP': 'HOP'})
    res['main'] = dict(n_val=len(V), n_test=len(T), chosen=chosen, summary=summarize(T))
    cq = chosen['CQR']; pr = {}
    for f in ['LET', 'ML', 'ENS', 'QR', 'CP']:
        b = chosen[f]; pr[b] = {m: paired(T, cq, b, m) for m in ['PDR', 'breaks', 'disc', 'PDR_uncond']}
    for b in ['LET m=2', 'LET m=4', 'LET m=6', 'LET m=9']:
        pr.setdefault(b, {m: paired(T, cq, b, m) for m in ['PDR', 'breaks', 'disc', 'PDR_uncond']})
    res['main']['paired_vs_' + cq] = pr
    res['main']['hopbins'] = {k: hopbins(T, k) for k in ['LET m=4', 'ML m=9', 'QR a=0.1', cq, 'CQR a=0.05']}
    S = res['main']['summary']
    # ---- figure: PDR with paired CI
    rows = [('Shortest hop', 'HOP', 'HOP'), (f"CP on point ({chosen['CP'].split()[-1]})", chosen['CP'], 'CP'), ('Bootstrap ens., m=4', 'Ens m=4', 'ENS'),
            (f"ML point, {chosen['ML'].split()[-1]}", chosen['ML'], 'ML'), ('Raw QR, $\\alpha$=0.1', 'QR a=0.1', 'QR'),
            (f"LET, {chosen['LET'].split()[-1]}", chosen['LET'], 'LET'), (f"CQR, $\\alpha$={cq.split('=')[-1]}", cq, 'CQR'), ('Oracle', 'ORACLE', 'ORACLE')]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for i, (lab, k, c) in enumerate(rows):
        m, lo, hi = S[k]['PDR']
        ax.barh(i, m, height=0.6, color=C[c]); ax.text(m + 0.01, i, f'{m:.3f}', va='center', fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 0.85)
    ax.set_xlabel('Conditional PDR (12 test traces)'); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/pdr_main.pdf'); plt.close(fig)
    # ---- figure: pareto
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for key, ks, lab in [('LET', fam['LET'], 'LET (margin 2–9 s)'), ('ML', fam['ML'], 'ML point (margin 4, 9 s)'),
                         ('CQR', fam['CQR'][::-1], 'CQR ($\\alpha$=0.3–0.05)')]:
        ax.plot([S[k]['disc'][0] for k in ks], [S[k]['breaks'][0] for k in ks], color=C[key], marker=MK[key], label=lab)
    OFF = {'CQR a=0.3': (-30, -9), 'CQR a=0.2': (-10, -14), 'CQR a=0.1': (-8, -17), 'CQR a=0.05': (-10, -14),
           'LET m=2': (-4, 6), 'LET m=4': (3, -11), 'LET m=6': (-6, -11), 'LET m=9': (-6, -11)}
    for k in fam['CQR'] + fam['LET']:   # label every CQR point with its risk level and every LET point with its margin
        v = k.split('=')[-1]; lab = f'$\\alpha$={v}' if k.startswith('CQR') else f'm={v} s'
        ax.annotate(lab, (S[k]['disc'][0], S[k]['breaks'][0]), xytext=OFF[k], textcoords='offset points', fontsize=6.5,
                    color=C['CQR' if k.startswith('CQR') else 'LET'])
    for key, k, lab in [('ENS', 'Ens m=4', 'Bootstrap ens., m=4'), ('QR', 'QR a=0.1', 'Raw QR, $\\alpha$=0.1'), ('ORACLE', 'ORACLE', 'Oracle')]:
        ax.plot(S[k]['disc'][0], S[k]['breaks'][0], color=C[key], marker=MK[key], ls='none', ms=8 if key == 'ORACLE' else (11 if key == 'QR' else 6), mfc='none' if key == 'QR' else C[key], mew=1.5, label=lab)
    ax.set_xlabel('Route discoveries per flow-minute'); ax.set_ylabel('Unexpected breaks per flow-minute')
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.3)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(f'{OUT}/pareto_main.pdf'); plt.close(fig)
    # ---- figure: coverage levels (all / used links / used paths) and kept-route failure by hop count
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    keys = [('LET', chosen['LET']), ('ML', chosen['ML']), ('QR', 'QR a=0.1'), ('CQR', cq)]
    x = np.arange(3); w = 0.2
    for q, (c, k) in enumerate(keys):
        v = [S[k]['cov_all'][0], S[k]['link_cov_used'][0], S[k]['path_cov'][0]]
        axs[0].bar(x + (q - 1.5) * w, v, width=w, color=C[c], label={'LET':'LET, m=4 s','ML':'ML point, m=9 s','QR':'Raw QR, $\\alpha$=0.1','CQR':'CQR, $\\alpha$=0.1'}[c])
    axs[0].axhline(0.9, color='#8a887f', ls=':', lw=1); axs[0].set_xticks(x); axs[0].set_xticklabels(['all links', 'links on routes', 'routes (min)'])
    axs[0].set_ylabel('Coverage $\\Pr\\{Y\\geq L\\}$'); axs[0].set_ylim(0, 1); axs[0].grid(axis='x', visible=False)
    hb = res['main']['hopbins']; hs = ['2', '3', '4', '5+']
    for c, k in keys:
        axs[1].plot(hs, [hb[k][h]['kept_fail'] for h in hs], color=C[c], marker=MK[c])
    axs[1].set_xlabel('Route length (hops)'); axs[1].set_ylabel('Kept routes breaking\nbefore next check')
    hh, ll = axs[0].get_legend_handles_labels(); fig.legend(hh, ll, loc='lower center', ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f'{OUT}/selection_effect.pdf'); plt.close(fig)

for mode in ['shift', 'noretrain', 'N30', 'N80']:
    if len(glob.glob(f'{R5}/{mode}_s*.json')) >= 2:
        D = load(mode); res[mode] = dict(n=len(D), summary=summarize(D))
        ks = list(next(iter(D.values())).keys())
        ref = 'CQR static a=0.1' if 'CQR static a=0.1' in ks else 'CQR a=0.1'
        res[mode]['paired_vs_' + ref] = {b: {m: paired(D, ref, b, m) for m in ['PDR', 'breaks', 'disc']} for b in ks if b != ref}
json.dump(res, open(f'{R5}/summary.json', 'w'), indent=1)
print('ok', list(res))

# ---- prediction-level coverage with trace-level CIs (8 test traces)
if os.path.exists(f'{R5}/pred_ci.json'):
    P = json.load(open(f'{R5}/pred_ci.json'))['summary']; al = [0.05, 0.1, 0.2]; tg = [1 - a for a in al]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for nm, c, lab in [('ENS', 'ENS', 'Bootstrap ensemble $\\mu-z\\sigma$'), ('QR', 'QR', 'Quantile reg. (uncalibrated)'),
                       ('CP', 'CP', 'Split CP on point'), ('CQR', 'CQR', 'CQR (proposed)')]:
        for ax, suf in [(axs[0], ''), (axs[1], '_meanL')]:
            v = np.array([P[f'{nm}@{a}{suf}'] for a in al])
            ax.errorbar(tg, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=C[c], marker=MK[c], capsize=2, label=lab)
    for nm, c, lab in [('LET', 'LET', 'LET (kinematic)'), ('ML', 'ML', 'ML point')]:
        axs[0].plot([0.8, 0.95], [P[nm][0]] * 2, '--', color=C[c], marker=MK[c], label=lab)
        axs[1].plot([0.8, 0.95], [P[nm + '_meanL'][0]] * 2, '--', color=C[c], marker=MK[c])
    axs[0].plot([0.8, 0.95], [0.8, 0.95], ':', color='#8a887f', lw=1, label='Ideal')
    axs[0].set_xlabel('Target coverage $1-\\alpha$'); axs[0].set_ylabel('Empirical $\\Pr\\{Y\\geq L\\}$')
    axs[1].set_xlabel('Target coverage $1-\\alpha$'); axs[1].set_ylabel('Mean bound $L$ (s)')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f'{OUT}/coverage_alpha.pdf'); plt.close(fig)
    print('coverage fig ok')

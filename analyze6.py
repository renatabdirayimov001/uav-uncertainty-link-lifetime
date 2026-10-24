"""Comparison with re-implemented published schemes (results6 + results5) -> tables and figure."""
import json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R5, R6, R7 = 'results5', 'results6', 'results7'
OUTF = os.environ.get('OUTF', 'figures'); OUTT = os.environ.get('OUTT', 'tables')
os.makedirs(OUTF, exist_ok=True); os.makedirs(OUTT, exist_ok=True)
VAL, TEST = [7, 8, 9, 10], list(range(11, 23))
SC = {'main': ('Base', TEST), 'N80': ('Dense ($N=80$)', list(range(41, 47))), 'noretrain': ('Degraded info.', list(range(51, 59))),
      'shift': ('Speed shift', list(range(31, 39)))}

def load(mode, seeds):
    D = {}
    for s in seeds:
        f6, fl, f5 = f'{R6}/{mode}_s{s}.json', f'{R6}/lpar_{mode}_s{s}.json', f'{R5}/{mode}_s{s}.json'
        if not (os.path.exists(f6) and os.path.exists(fl) and os.path.exists(f5)): continue
        r = json.load(open(f6)); r.update(json.load(open(fl)))
        f7 = f'{R7}/{mode}_s{s}.json'
        if os.path.exists(f7): r.update(json.load(open(f7)))          # RCPLA-FANET (exp7.py)
        r5 = json.load(open(f5)); cq5 = r5['CQR a=0.1'] if 'CQR a=0.1' in r5 else r5['CQR static a=0.1']
        # RCLR, LET and the gradient-boosted point predictor are taken from the main experiment (results5) on the same traces,
        # so that every table of the paper reports identical numbers for them
        r['Rocha XGB'] = r5['ML m=9']; r['Rocha ref CQR'] = cq5; r['CQR a=0.1 (proposed)'] = cq5; r['LET m=4'] = r5['LET m=4']
        D[s] = r
    return D
def ci(v):
    v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return float(v.mean()), float(h)

V = load('main', VAL)
fam = {'LB-OPAR': [k for k in V[7] if k.startswith('LB-OPAR w2')], 'OLSR+': [k for k in V[7] if k.startswith('OLSR+ qth')],
       'LPMD-GPSR': [k for k in V[7] if k.startswith('LPMD-GPSR')],
       'LB-OPAR LLT': [k for k in V[7] if k.startswith('LB-OPAR LLT')], 'OLSR+ LLT': [k for k in V[7] if k.startswith('OLSR+ LLT')],
       'LPMD LSTM': [k for k in V[7] if k.startswith('LPMD LSTM')], 'RCPLA': [k for k in V[7] if k.startswith('RCPLA')]}
ch = {f: max(ks, key=lambda k: np.mean([V[s][k]['PDR'] for s in V])) for f, ks in fam.items()}
ch.update({'LPAR': 'LPAR', 'Rocha XGB': 'Rocha XGB', 'LET': 'LET m=4', 'CQR': 'CQR a=0.1 (proposed)'})
print('chosen', ch)
REF = 'CQR a=0.1 (proposed)'
proto = [('LB-OPAR', 'LB-OPAR~\\cite{gharib2022lbopar}'), ('OLSR+', 'OLSR+~\\cite{rahmani2022olsr}'), ('LPAR', 'LPAR~\\cite{li2025predictionbased}'),
         ('LPMD-GPSR', 'LPMD-GPSR~\\cite{dong2026lpmdgpsr}'), ('RCPLA', 'RCPLA-FANET~\\cite{dogaheh2026rcpla}'), ('Rocha XGB', 'GB point predictor~\\cite{rocha2022evaluating} in RCLR routing'),
         ('LET', 'LET ($m=4$~s) in RCLR routing'), ('CQR', '\\textbf{RCLR (CQR, $\\alpha=0.1$)}')]
summary = {'chosen': ch}
# ---------------- base scenario table
T = load('main', TEST); n = len(T); summary['n_test'] = n
rows = ''
for f, lab in proto:
    k = ch[f]
    pdr = ci([T[s][k]['PDR'] for s in T]); pu = ci([T[s][k]['PDR_uncond'] for s in T])
    br = ci([T[s][k]['breaks'] for s in T]); di = ci([T[s][k]['disc'] for s in T]); hp = np.mean([T[s][k]['hops'] for s in T])
    if f == 'CQR': dp = '--'
    else:
        ref = 'Rocha ref CQR' if f == 'Rocha XGB' else REF
        dd = ci([100 * (T[s][ref]['PDR'] - T[s][k]['PDR']) for s in T]); w = sum(T[s][ref]['PDR'] > T[s][k]['PDR'] for s in T)
        rel = 100 * np.mean([T[s][ref]['PDR'] for s in T]) / pdr[0] - 100
        dp = f'${dd[0]:+.2f}\\pm{dd[1]:.2f}$ ({w}/{n}) & ${rel:+.1f}\\%$'
        summary.setdefault('base', {})[f] = dict(PDR=pdr, dPDR=dd, wins=int(w), rel=rel, breaks=br, disc=di)
    fmt = lambda x: '--' if np.isnan(x[0]) else f'{x[0]:.2f}'
    cells = [f'{pdr[0]:.3f}', f'{pu[0]:.3f}', fmt(br), fmt(di), f'{hp:.2f}', dp] if f != 'CQR' else [f'{pdr[0]:.3f}', f'{pu[0]:.3f}', fmt(br), fmt(di), f'{hp:.2f}', '--', '--']
    if f == 'CQR': cells = ['\\textbf{' + c + '}' if c != '--' else c for c in cells]
    rows += lab + ' & ' + ' & '.join(cells) + ' \\\\\n'
    if f == 'RCPLA': rows += '\\midrule\n'
summary['base_cqr'] = ci([T[s][REF]['PDR'] for s in T])
open(f'{OUTT}/tab_published.tex', 'w').write(rows)
# ---------------- predictor level (prediction quality on 8 test traces + use in RCLR routing)
PP = json.load(open(f'{R6}/pred_pub.json')) if os.path.exists(f'{R6}/pred_pub.json') else None
rows = ''
for f, lab, pk in [('LB-OPAR LLT', 'LB-OPAR three-position LLT', 'LB-OPAR'), ('OLSR+ LLT', 'OLSR+ quality-decay LLT', 'OLSR+'),
                   ('LPMD LSTM', 'LPMD-GPSR LSTM', 'LSTM'), ('Rocha XGB', 'Gradient-boosted point predictor', 'XGB'),
                   ('LET', 'LET (Su et al.)', 'LET'), ('CQR', '\\textbf{CQR lower bound (proposed)}', 'CQR')]:
    k = ch[f]; pdr = ci([T[s][k]['PDR'] for s in T])
    m = k.split('m=')[-1].split(' ')[0] if 'm=' in k else ('9' if f == 'Rocha XGB' else '2')
    if f == 'CQR': dp = '--'
    else:
        ref = 'Rocha ref CQR' if f == 'Rocha XGB' else REF
        dd = ci([100 * (T[s][ref]['PDR'] - T[s][k]['PDR']) for s in T]); w = sum(T[s][ref]['PDR'] > T[s][k]['PDR'] for s in T)
        dp = f'${dd[0]:+.2f}\\pm{dd[1]:.2f}$ ({w}/{n})'
        summary.setdefault('pred', {})[f] = dict(PDR=pdr, dPDR=dd, wins=int(w), m=m)
    q = PP[pk] if PP else None
    qc = [f"{q['cov'][0]:.3f}", f"{q['meanL'][0]:.1f}", f"{q['mae'][0]:.1f}", f"{100*q['short_of_long'][0]:.1f}"] if q else ['--'] * 4
    rows += f'{lab} & ' + ' & '.join(qc) + f' & {m} & {pdr[0]:.3f} & {dp} \\\\\n'
open(f'{OUTT}/tab_predlevel.tex', 'w').write(rows)
# ---------------- across scenarios
rows = ''; fig_data = {}
for f, lab in proto:
    k = ch[f]; cells = []
    for mode, (nm, seeds) in SC.items():
        D = load(mode, seeds)
        if len(D) == 0: cells += ['--']; continue
        p = np.mean([D[s][k]['PDR'] for s in D]); fig_data.setdefault(f, []).append(p)
        if f == 'CQR': cells.append(f'\\textbf{{{p:.3f}}}')
        else:
            ref = 'Rocha ref CQR' if f == 'Rocha XGB' else ('CQR a=0.1 (proposed)')
            rel = 100 * np.mean([D[s][ref]['PDR'] for s in D]) / p - 100; w = sum(D[s][ref]['PDR'] > D[s][k]['PDR'] for s in D)
            cells.append(f'{p:.3f} (${rel:+.1f}\\%$)')
            summary.setdefault('scen', {}).setdefault(mode, {})[f] = dict(PDR=p, rel=rel, wins=int(w), n=len(D))
    rows += lab.replace(' in RCLR routing', '$^{\\dagger}$') + ' & ' + ' & '.join(cells) + ' \\\\\n'
open(f'{OUTT}/tab_pub_scen.tex', 'w').write(rows)
json.dump(summary, open(f'{R6}/summary6.json', 'w'), indent=1, default=float)
# ---------------- figure
# same colours as in analyze5.py for LET / GB point (ML) / CQR; published schemes get colours used nowhere else
C = {'RCPLA': '#b03a2e', 'LB-OPAR': '#8c564b', 'OLSR+': '#17a2b8', 'LPAR': '#b5b82a', 'LPMD-GPSR': '#a3367a', 'Rocha XGB': '#1baf7a', 'LET': '#eb6834', 'CQR': '#2a78d6'}
LBL = {'RCPLA': 'RCPLA-FANET', 'LB-OPAR': 'LB-OPAR', 'OLSR+': 'OLSR+', 'LPAR': 'LPAR', 'LPMD-GPSR': 'LPMD-GPSR', 'Rocha XGB': 'GB point predictor', 'LET': 'LET', 'CQR': 'RCLR (proposed)'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e4e3df',
                     'grid.linewidth': 0.6, 'axes.edgecolor': '#8a887f', 'legend.frameon': False, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.0, 2.9)); keys = [f for f, _ in proto if f in fig_data]; nS = len(next(iter(fig_data.values())))
w = 0.8 / len(keys); x = np.arange(nS)
for q, f in enumerate(keys):
    ax.bar(x + (q - (len(keys) - 1) / 2) * w, fig_data[f], width=w * 0.92, color=C[f], label=LBL[f])
for xi, v in zip(x, fig_data['CQR']): ax.text(xi + (len(keys) - 1) / 2 * w, v + 0.01, f'{v:.3f}', ha='center', fontsize=6.5, color='#1f5fae')
ax.set_xticks(x); ax.set_xticklabels([SC[m][0].replace('$N=80$', 'N=80') for m in SC][:nS]); ax.set_ylabel('Conditional PDR'); ax.set_ylim(0, 1.0)
ax.grid(axis='x', visible=False); fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=8, fontsize=6.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f'{OUTF}/published_comparison.pdf'); plt.close(fig)
print(json.dumps({k: v for k, v in summary.items() if k != 'chosen'}, indent=0, default=float)[:3000])

"""Generate LaTeX tables for the revised paper from results5/ (no hand transcription)."""
import json, os
import numpy as np
R5 = 'results5'; OUT = 'tables'
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f'{R5}/summary.json'))
P = json.load(open(f'{R5}/pred_ci.json')); PS = P['summary']
A = json.load(open(f'{R5}/ablation_cqr.json'))

def f3(v): return f'{v:.3f}'
def ci3(v, d=3): return f'{v[0]:.{d}f}\\,$\\pm$\\,{(v[2]-v[1])/2:.{d}f}'
def pm(v, k=1, d=2): return f'${k*v[0]:+.{d}f}\\pm{(v[2]-v[1])/2*abs(k):.{d}f}$'
def w(name, s): open(f'{OUT}/{name}.tex', 'w').write(s); print('wrote', name)

# ---------- prediction-level calibration (alpha = 0.1) ----------
rows = [('LET (kinematic)', 'LET', 'LET_meanL', None, None), ('ML point', 'ML', 'ML_meanL', None, None),
        ('Bootstrap ensemble $\\mu-z\\sigma$', 'ENS@0.1', 'ENS@0.1_meanL', 'ENS@0.1_zero', 'ENS@0.1_short_of_long'),
        ('Quantile regression (raw)', 'QR@0.1', 'QR@0.1_meanL', 'QR@0.1_zero', 'QR@0.1_short_of_long'),
        ('Split CP on point', 'CP@0.1', 'CP@0.1_meanL', 'CP@0.1_zero', 'CP@0.1_short_of_long'),
        ('\\textbf{CQR (proposed)}', 'CQR@0.1', 'CQR@0.1_meanL', 'CQR@0.1_zero', 'CQR@0.1_short_of_long')]
s = ''
for lab, c, m, z, sl in rows:
    s += f'{lab} & {ci3(PS[c])} & {PS[m][0]:.2f} & ' + (f'{100*PS[z][0]:.1f}' if z else '--') + ' & ' + (f'{100*PS[sl][0]:.1f}' if sl else '--') + ' \\\\\n'
w('tab_pred', s)
s = ''
for a in ['0.05', '0.1', '0.2']:
    s += f'$1-\\alpha={1-float(a):.2f}$ & ' + ' & '.join(ci3(PS[f'{n}@{a}']) for n in ['ENS', 'QR', 'CP', 'CQR']) + ' \\\\\n'
w('tab_pred_alpha', s)
# conditional
s = ''
labs = [('age0', 'HELLO age 0~s'), ('age1', 'HELLO age 1~s'), ('age2-4', 'HELLO age 2--4~s'), ('d<0.5', '$\\hat d/R<0.5$'),
        ('d0.5-0.8', '$0.5\\le\\hat d/R<0.8$'), ('d>=0.8', '$\\hat d/R\\ge0.8$')]
ntot = PS['n'][0]
for k, lab in labs:
    c = PS['cond_' + k]
    s += f"{lab} & {100*c['n'][0]/ntot:.1f} & {c['meanY'][0]:.1f} & {c['LET'][0]:.3f} & {ci3(c['CQR'])} & {c['CQR_meanL'][0]:.1f} \\\\\n"
w('tab_conditional', s)
# ablation
def fmt_corr(c):
    if c >= 0.01: return f'{c:.2f}'
    e = int(np.floor(np.log10(c))); return f'${c/10**e:.1f}\\times10^{{{e}}}$'
s = ''
for k, v in A.items():
    s += f"{(k[0].upper()+k[1:]).replace('%','\\%')} & {ci3(v['raw_cov'])} & {ci3(v['cqr_cov'])} & {fmt_corr(v['correction_s'])} & {v['raw_meanL'][0]:.2f} & {v['cqr_meanL'][0]:.2f} \\\\\n"
w('tab_ablation', s)

# ---------- main routing: absolute (test traces) ----------
M = S['main']; MS = M['summary']; ch = M['chosen']; cq = ch['CQR']
def lab(k):
    if k == 'ORACLE': return 'Oracle (true lifetime)'
    if k == 'HOP': return 'Shortest hop'
    fam, par = k.split(' ', 1)
    name = {'LET': 'LET', 'ML': 'ML point', 'Ens': 'Bootstrap ens.', 'CP': 'CP on point', 'QR': 'Raw QR', 'CQR': 'CQR'}[fam]
    return name + ', ' + par.replace('a=', '$\\alpha$=').replace('m=', '$m$=')
order = ['HOP', 'LET m=2', 'LET m=4', 'LET m=6', 'LET m=9', 'ML m=4', 'ML m=9', 'Ens m=4', 'CP a=0.1', 'CP a=0.3', 'QR a=0.1',
         'CQR a=0.05', 'CQR a=0.1', 'CQR a=0.2', 'CQR a=0.3', 'ORACLE']
s = ''
for k in order:
    v = MS[k]; bold = k == cq
    cells = [f"{v['PDR'][0]:.3f}", f"{v['PDR_uncond'][0]:.3f}", f"{v['breaks'][0]:.2f}", f"{v['disc_re'][0]:.2f}", f"{v['disc_pro'][0]:.2f}",
             f"{v['hops'][0]:.2f}"]
    if bold: cells = ['\\textbf{' + c + '}' for c in cells]
    s += (('\\textbf{' + lab(k) + '}') if bold else lab(k)) + ' & ' + ' & '.join(cells) + ' \\\\\n'
    if k in ('HOP', 'QR a=0.1', 'CQR a=0.3'): s += '\\midrule\n'
w('tab_main_abs', s)
# paired
pr = M['paired_vs_' + cq]
s = ''
for b in ['LET m=2', 'LET m=4', 'LET m=6', 'LET m=9', ch['ML'], 'Ens m=4', ch['CP'], 'QR a=0.1']:
    d = pr[b]
    pdr, wins, n = d['PDR']; br, bw, _ = d['breaks']; di, dw, _ = d['disc']
    rel_b = 100 * br[0] / MS[b]['breaks'][0]; rel_d = 100 * di[0] / MS[b]['disc'][0]
    s += f"{lab(b)} & {pm(pdr,100)} ({wins}/{n}) & {pm(br)} ({rel_b:+.1f}\\%) & {pm(di)} ({rel_d:+.1f}\\%) \\\\\n"
w('tab_paired', s)
# selection effect
hb = M['hopbins']
s = ''
for k in ['LET m=4', ch['ML'], 'QR a=0.1', cq, 'CQR a=0.05']:
    v = MS[k]
    s += f"{lab(k)} & {v['cov_all'][0]:.3f} & {v['link_cov_used'][0]:.3f} & {v['path_cov'][0]:.3f} & " + \
         ' & '.join(f"{100*hb[k][h]['kept_fail']:.2f}" for h in ['1', '2', '3', '4', '5+']) + ' \\\\\n'
w('tab_selected', s)

# ---------- shift / noretrain ----------
for mode in ['shift', 'noretrain']:
    if mode not in S: continue
    D = S[mode]; DS = D['summary']; ref = 'CQR static a=0.1'; pp = D['paired_vs_' + ref]
    s = ''
    for k in ['ORACLE', 'LET m=4', 'ML m=9', 'QR a=0.1', ref, 'CQR+ACI ideal', 'CQR+ACI obs', 'QL+ACI obs']:
        v = DS[k]
        name = {'ORACLE': 'Oracle', 'LET m=4': 'LET, $m$=4', 'ML m=9': 'ML point, $m$=9', 'QR a=0.1': 'Raw QR',
                ref: 'CQR static', 'CQR+ACI ideal': 'CQR+ACI, ideal fb.', 'CQR+ACI obs': 'CQR+ACI, obs. fb.',
                'QL+ACI obs': 'QL-ACI, obs. fb.'}[k]
        dp = '--' if k == ref else pm(pp[k]['PDR'][0], -100) + f" ({pp[k]['PDR'][2]-pp[k]['PDR'][1]}/{pp[k]['PDR'][2]})"
        cov = '--' if k == 'ORACLE' else f"{v['cov_all'][0]:.3f}"
        covu = '--' if k == 'ORACLE' else f"{v['link_cov_used'][0]:.3f}"
        s += f"{name} & {cov} & {covu} & {v['PDR'][0]:.3f} & {v['breaks'][0]:.2f} & {v['disc'][0]:.2f} & {dp} \\\\\n"
    w(f'tab_{mode}', s)

# ---------- swarm size ----------
if 'N30' in S or 'N80' in S:
    s = ''
    for k in ['HOP', 'LET m=4', 'ML m=9', 'Ens m=4', 'CQR a=0.1', 'CQR a=0.2', 'ORACLE']:
        cells = []
        for mode in ['N30', 'N80']:
            if mode not in S: cells += ['--'] * 3; continue
            v = S[mode]['summary'][k]; pp = S[mode]['paired_vs_CQR a=0.1']
            dp = '--' if k == 'CQR a=0.1' else pm(pp[k]['PDR'][0], 100) + f" ({pp[k]['PDR'][1]}/{pp[k]['PDR'][2]})"
            cells += [f"{v['PDR'][0]:.3f}", f"{v['breaks'][0]:.2f}", dp]
        s += lab(k) + ' & ' + ' & '.join(cells) + ' \\\\\n'
    w('tab_swarm', s)

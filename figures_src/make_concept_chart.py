"""Figure 1 as a data-style chart (drawn at the exact printed size: 5.46 in = text width)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.5, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': False, 'hatch.linewidth': 0.6, 'pdf.fonttype': 42})
ORANGE, BLUE, GREY, RED, GREEN = '#eb6834', '#2a78d6', '#3d3d3d', '#d62728', '#1a9850'
fig, axs = plt.subplots(1, 2, figsize=(5.46, 2.35), sharey=True)
links = ['Link B', 'Link A']                      # bottom to top
true = [5, 22]
panels = [('(a) Point prediction (LET / ML)', [20, 20], ORANGE, 'Predicted lifetime $\\hat Y$'),
          ('(b) Calibrated lower bound (proposed)', [3, 15], BLUE, 'Lower bound $L$ ($1-\\alpha=0.9$)')]
h = 0.34
for ax, (title, est, col, lab) in zip(axs, panels):
    y = range(len(links))
    ax.barh([k + h / 2 for k in y], est, height=h, color='white', edgecolor=col, hatch='////', linewidth=1.0, label=lab)
    ax.barh([k - h / 2 for k in y], true, height=h, color=GREY, edgecolor=GREY, label='True residual lifetime $Y$')
    for k in y:
        ax.text(est[k] + 0.5, k + h / 2, f'{est[k]} s', va='center', fontsize=8, color=col)
        off = 1.3 if (k == 0 and col == ORANGE) else 0.5
        ax.text(true[k] + off, k - h / 2, f'{true[k]} s', va='center', fontsize=8, color=GREY)
    ax.set_xlim(0, 30); ax.set_ylim(-0.6, 1.6)
    ax.set_yticks(list(y)); ax.set_yticklabels(links)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.set_xlabel('Time from now (s)')
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.tick_params(axis='y', length=0)
    ax.axvline(2, color='#7f7f7f', ls=':', lw=0.9)
axs[0].text(2.3, -0.52, '$T_u$', fontsize=8, color='#555555')
# outcome markers
axs[0].annotate('unexpected\nbreak at 5 s', xy=(8.2, -h / 2), xytext=(12, -0.38), fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8), va='center')
axs[0].plot(5, -h / 2, marker='X', color=RED, ms=6, mec='white', mew=0.5, zorder=5)
axs[1].annotate('$L_B=3$ s < $Y_B$:\nswitched in time', xy=(6.2, h / 2), xytext=(10.5, 0.47), fontsize=7.5, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.8), va='center')
handles = [Patch(facecolor='white', edgecolor=ORANGE, hatch='////', label='Point prediction $\\hat Y$'),
           Patch(facecolor='white', edgecolor=BLUE, hatch='////', label='Calibrated lower bound $L$'),
           Patch(facecolor=GREY, edgecolor=GREY, label='True lifetime $Y$'),
           Line2D([0], [0], color='#7f7f7f', ls=':', label='Next route check $T_u$')]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7.5, frameon=True, edgecolor='#555555', fancybox=False,
           handlelength=1.6, columnspacing=1.2, borderpad=0.4)
fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.0)
fig.savefig('../figures/fig_concept.pdf'); fig.savefig('fig_concept_chart.png', dpi=250)
print('ok')

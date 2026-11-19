"""Рисунки для научной статьи (ширина колонки ~8.6 см, векторный PDF)."""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
sys.path.insert(0, '.')
from spline_imu import demo_keyframes, ModelA, ModelB, ModelV

OUT = '../out/'
FIG = '../figures/'
COL = {'A': '#8a8984', 'B': '#eb6834', 'V': '#2a78d6'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'lines.linewidth': 1.2, 'legend.frameon': False})
W = 3.35

kf = demo_keyframes()
models = {'A': ModelA(kf), 'B': ModelB(kf), 'V': ModelV(kf)}
t = np.linspace(0.0005, 9.9995, 8000)

# --- угловое ускорение по оси x ---
fig, ax = plt.subplots(figsize=(W, 1.9))
for k, lab in (('A', 'А'), ('B', 'Б'), ('V', 'В')):
    al = models[k].eval(t)[4][:, 0]
    ax.plot(t, al, color=COL[k], label=lab, lw=1.0 if k == 'A' else 1.2)
for x in kf.dt * np.arange(1, kf.K):
    ax.axvline(x, color=GRID, lw=0.5, zorder=0)
ax.set_xlabel('$t$, с')
ax.set_ylabel(r'$\alpha_{C,x}$, рад/с$^2$')
ax.legend(loc='upper left', ncol=3, fontsize=7, handlelength=1.4,
          columnspacing=1.0, bbox_to_anchor=(0.0, 1.13))
fig.tight_layout(pad=0.3)
fig.savefig(FIG + 'alpha.pdf')
plt.close(fig)

# --- сходимость ---
res = json.load(open(OUT + 'convergence.json'))


def sel(m, g, s):
    rr = [r for r in res if r['model'] == m and r['grid'] == g
          and r['scheme'] == s]
    return sorted(rr, key=lambda r: r['f'])


fig, ax = plt.subplots(figsize=(W, 2.5))
series = [('A', 'shifted', 'А', '-', 'o'),
          ('B', 'shifted', 'Б', '-', 'o'),
          ('V', 'shifted', 'В, сдвиг', '-', 'o'),
          ('V', 'aligned', 'В, в узлах', '--', 's')]
for m, g, lab, ls, mk in series:
    rr = sel(m, g, 'rk4')
    f = np.array([r['f'] for r in rr])
    e = np.array([r['ep'] for r in rr])
    if m == 'V' and g == 'aligned':
        keep = f <= 400
        f, e = f[keep], e[keep]
    ax.plot(f, e, ls, color=COL[m], marker=mk, ms=3, lw=1.1)
    if m == 'A':
        ax.text(f[2], e[2] * 5, lab, color=INK, ha='center', va='bottom')
    else:
        ax.text(f[-1] * 1.15, e[-1], lab, color=INK, va='center')
for order, y0, ff in ((1, 2.4e-4, [50, 1600.0]), (2, 2.2e-5, [50, 1600.0]),
                      (4, 4.5e-9, [50, 400.0])):
    ff = np.array(ff)
    ax.plot(ff, y0 * (ff / 50) ** (-order), ':', color=INK2, lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks([50, 100, 200, 400, 800, 1600])
ax.set_xticklabels(['50', '100', '200', '400', '800', '1600'])
ax.set_xlim(40, 4200)
ax.set_ylim(1e-13, 3e2)
ax.spines['bottom'].set_bounds(40, 1600)
ax.set_xlabel('$f$, Гц')
ax.set_ylabel('СКО $e_p$, м')
fig.tight_layout(pad=0.3)
fig.savefig(FIG + 'convergence.pdf')
plt.close(fig)
print('ok')

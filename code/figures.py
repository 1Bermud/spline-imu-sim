import numpy as np, sys, json
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt
from spline_imu import *
OUT = '../out/'
COL = {'A': '#8a8984', 'B': '#eb6834', 'V': '#2a78d6'}
LAB = {'A': 'А: отрезки + SLERP', 'B': 'Б: кубический сплайн + SQUAD', 'V': 'В: B-сплайн на SO(3)'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2,
    'legend.frameon': False, 'figure.dpi': 100, 'savefig.dpi': 200})
kf = demo_keyframes(); rig = demo_rig()
models = {'A': ModelA(kf), 'B': ModelB(kf), 'V': ModelV(kf)}
t = np.linspace(0.0005, 9.9995, 10000)
ev = {k: m.eval(t) for k, m in models.items()}
knots = kf.dt * np.arange(1, kf.K)

# --- Рис. 1: угловая скорость и угловое ускорение ---
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)
for k in ('A', 'B', 'V'):
    a1.plot(t, ev[k][3][:, 0], color=COL[k], label=LAB[k], lw=1.8 if k != 'A' else 1.4)
    a2.plot(t, ev[k][4][:, 0], color=COL[k], lw=1.8 if k != 'A' else 1.4)
for ax in (a1, a2):
    for x in knots: ax.axvline(x, color=GRID, lw=0.8, zorder=0)
a1.set_ylabel('ω_x, рад/с'); a2.set_ylabel('α_x, рад/с²'); a2.set_xlabel('время, с   (вертикальные линии — узлы, ключевые позы через 0,5 с)')
a1.set_title('Угловая скорость и угловое ускорение камеры, ось x', loc='left', color=INK, fontsize=11)
a1.legend(loc='upper center', fontsize=8.5, ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=1.5)

fig.tight_layout(); fig.savefig(OUT + 'fig1_omega_alpha.png'); plt.close(fig)

# --- Рис. 2: вклад рычага в показания акселерометра ---
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for k in ('B', 'V'):
    Rwc, _, _, om, al = ev[k]
    r = rig.r
    lever = np.cross(al, r) + np.cross(om, np.cross(om, r))          # в системе C
    lever_B = (rig.R_BC @ lever[..., None])[..., 0] * 1000           # в системе B, мм/с²
    ax.plot(t, lever_B[:, 2], color=COL[k], label=LAB[k], lw=1.8)
for x in knots: ax.axvline(x, color=GRID, lw=0.8, zorder=0)
ax.set_ylabel('мм/с²'); ax.set_xlabel('время, с')
ax.set_title('Вклад рычага α×r + ω×(ω×r) в акселерометр, ось z IMU, |r| = 8,5 см', loc='left', color=INK, fontsize=11)
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(OUT + 'fig2_lever.png'); plt.close(fig)

# --- Рис. 3: сходимость обратного интегрирования ---
res = json.load(open(OUT + 'convergence.json'))
sel = lambda m, g, s: sorted([r for r in res if r['model'] == m and r['grid'] == g and r['scheme'] == s], key=lambda r: r['f'])
fig, ax = plt.subplots(figsize=(7.6, 4.8))
series = [('A', 'shifted', 'А: ≈ 8 м при любой частоте', '-', 'o'),
          ('B', 'shifted', 'Б: порядок 1', '-', 'o'),
          ('V', 'shifted', 'В, сетка ≠ узлы: порядок 2', '-', 'o'),
          ('V', 'aligned', 'В, сетка = узлы: порядок 4', '--', 's')]
for m, g, lab, ls, mk in series:
    rr = sel(m, g, 'rk4')
    f = np.array([r['f'] for r in rr]); e = np.array([r['ep'] for r in rr])
    if m == 'V' and g == 'aligned':
        keep = f <= 400; f, e = f[keep], e[keep]     # дальше — ошибки округления
    ax.plot(f, e, ls, color=COL[m], marker=mk, ms=5, lw=1.8)
    if m == 'A':
        ax.text(f[2], e[2] * 4, lab, color=INK, fontsize=9, ha='center', va='bottom')
    else:
        ax.text(f[-1] * 1.12, e[-1], lab, color=INK, fontsize=9, va='center')
for order, y0, ff in ((1, 2.4e-4, [50, 1600.0]), (2, 2.2e-5, [50, 1600.0]), (4, 4.5e-9, [50, 400.0])):
    ff = np.array(ff)
    ax.plot(ff, y0 * (ff / 50) ** (-order), ':', color=INK2, lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks([50, 100, 200, 400, 800, 1600]); ax.set_xticklabels(['50', '100', '200', '400', '800', '1600'])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xlim(40, 6500); ax.set_ylim(1e-13, 3e2)
ax.set_xlabel('частота IMU, Гц'); ax.set_ylabel('СКО ошибки положения, м')
ax.set_title('Обратное интегрирование (РК4) идеальных показаний, 10 с', loc='left', color=INK, fontsize=11)
ax.spines['bottom'].set_bounds(40, 1600)
ax.text(0.0, -0.17, 'пунктир — опорные наклоны h¹, h², h⁴', transform=ax.transAxes, color=INK2, fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT + 'fig3_convergence.png'); plt.close(fig)

# --- таблица порядков ---
for m in 'ABV':
    for g in ('aligned', 'shifted'):
        for s in ('euler', 'rk4'):
            rr = sel(m, g, s); e = {r['f']: r['ep'] for r in rr}
            print(m, g, s, 'e(200Hz)=%.2e' % e[200], 'order50-200=%.2f' % (np.log2(e[50] / e[200]) / 2))

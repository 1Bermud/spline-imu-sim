import numpy as np, sys, json, time
sys.path.insert(0, '.')
from spline_imu import *
kf = demo_keyframes(); rig = demo_rig()
models = {'A': ModelA(kf), 'B': ModelB(kf), 'V': ModelV(kf)}
freqs = [50, 100, 200, 400, 800, 1600]
res = []
T0 = time.time()
for name, M in models.items():
    for grid in ('aligned', 'shifted'):
        for scheme in ('euler', 'rk4'):
            for f in freqs:
                h = 1.0 / f
                ts = 0.0 if grid == 'aligned' else 0.37 * h
                te = ts + h * int((kf.tK - 1e-9 - ts) / h)
                ep, eR = run_consistency(rig, M, ts, te, h, scheme)
                res.append(dict(model=name, grid=grid, scheme=scheme, f=f, ep=ep, eR=eR))
                print(name, grid, scheme, f, '%.3e %.3e' % (ep, eR), '%.0fs' % (time.time()-T0), flush=True)
# РК4 на сетке в узлах, конец шага -- предел слева (проверка: дело в модели или в отсчёте в узле)
for name in ('B', 'V'):
    for f in freqs:
        h = 1.0 / f
        te = h * int((kf.tK - 1e-9) / h)
        ep, eR = run_consistency(rig, models[name], 0.0, te, h, 'rk4', left_limit=True)
        res.append(dict(model=name, grid='aligned', scheme='rk4_left', f=f, ep=ep, eR=eR))
        print(name, 'aligned rk4_left', f, '%.3e %.3e' % (ep, eR), flush=True)
json.dump(res, open('../out/convergence.json', 'w'), indent=1)

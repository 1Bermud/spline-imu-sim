"""Проверка шумовой части генератора по вариации Аллана.

Неподвижный датчик: идеальные показания равны нулю (гироскоп) и константе
(акселерометр -- константа на вариацию Аллана не влияет). Шум добавляется
функцией add_noise генератора. По перекрывающейся вариации Аллана
оцениваются интенсивности белого шума N и случайного блуждания K по модели
sigma^2(tau) = N^2 / tau + K^2 tau / 3 и сравниваются с заданными.
"""
import json
import sys
import time
import numpy as np
sys.path.insert(0, '.')
from spline_imu import add_noise

F = 200.0                    # Гц
T = 3 * 3600.0               # с
# тестовые интенсивности, порядка типичных MEMS-датчиков
SIG = dict(g=1.7e-4, bg=2.0e-5, a=2.0e-3, ba=3.0e-3)


def adev(x, h, ms):
    """Перекрывающаяся вариация Аллана по отсчётам скорости x (N, 3)."""
    theta = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0) * h])
    out = []
    for m in ms:
        tau = m * h
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out.append((d ** 2).mean(axis=0) / (2 * tau ** 2))
    return np.array([m * h for m in ms]), np.array(out)


def fit_nk(tau, var):
    """МНК по модели var = N^2/tau + K^2 tau/3 (относительные невязки)."""
    A = np.stack([1 / tau, tau / 3], axis=1) / var[:, None]
    coef, *_ = np.linalg.lstsq(A, np.ones_like(var), rcond=None)
    return np.sqrt(coef)


if __name__ == '__main__':
    t0 = time.time()
    rng = np.random.default_rng(1)
    n = int(T * F)
    h = 1 / F
    zeros = np.zeros((n, 3))
    om, ac = add_noise(zeros, zeros, h, SIG['g'], SIG['a'], SIG['bg'],
                       SIG['ba'], rng)
    ms = np.unique(np.logspace(0, np.log10(n // 10), 60).astype(int))
    tau, vg = adev(om, h, ms)
    _, va = adev(ac, h, ms)
    res = {}
    for name, v, sn, sk in (('gyro', vg, 'g', 'bg'), ('accel', va, 'a', 'ba')):
        est = np.array([fit_nk(tau, v[:, k]) for k in range(3)])
        res[name] = dict(N_set=SIG[sn], K_set=SIG[sk],
                         N_est=est[:, 0].tolist(), K_est=est[:, 1].tolist())
        print(name, 'N: set %.3e est %s' % (SIG[sn], np.round(est[:, 0], 7)),
              '| K: set %.3e est %s' % (SIG[sk], np.round(est[:, 1], 7)))
    res['tau'] = tau.tolist()
    res['adev_gyro'] = np.sqrt(vg).tolist()
    res['adev_accel'] = np.sqrt(va).tolist()
    res['T_hours'] = T / 3600
    res['f_hz'] = F
    json.dump(res, open('../out/allan.json', 'w'), indent=1)
    print('%.0f s' % (time.time() - t0))

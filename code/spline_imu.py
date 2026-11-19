"""Генератор синтетических показаний IMU по траектории камеры.

Три модели движения по одним и тем же ключевым позам камеры:
  A — кусочно-линейное положение + SLERP для ориентации;
  B — кубический интерполяционный сплайн + SQUAD;
  V — кубический B-сплайн в R^3 + кумулятивный кубический B-сплайн на SO(3).

Соглашения: W — мир (z вверх), C — камера, B — IMU; R_XY переводит координаты
из Y в X; g_W = (0, 0, -9.81). Python 3.11, numpy, scipy.
"""
import numpy as np
from scipy.interpolate import CubicSpline

G_W = np.array([0.0, 0.0, -9.81])


# ---------- SO(3) ----------
def hat(v):
    v = np.asarray(v, float)
    O = np.zeros(v.shape[:-1] + (3, 3))
    O[..., 0, 1], O[..., 0, 2] = -v[..., 2], v[..., 1]
    O[..., 1, 0], O[..., 1, 2] = v[..., 2], -v[..., 0]
    O[..., 2, 0], O[..., 2, 1] = -v[..., 1], v[..., 0]
    return O


def vee(M):
    return np.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], axis=-1)


def Exp(phi):
    """Формула Родрига; работает с массивами (..., 3)."""
    phi = np.asarray(phi, float)
    th = np.linalg.norm(phi, axis=-1)[..., None, None]
    K = hat(phi)
    small = th < 1e-8
    th_s = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th**2 / 6, np.sin(th_s) / th_s)
    b = np.where(small, 0.5 - th**2 / 24, (1 - np.cos(th_s)) / th_s**2)
    return np.eye(3) + a * K + b * (K @ K)


def Log(R):
    R = np.asarray(R, float)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)[..., None]
    w = vee(R - np.swapaxes(R, -1, -2)) / 2
    small = th < 1e-8
    th_s = np.where(small, 1.0, th)
    return np.where(small, w, w * th_s / np.sin(th_s))


def polar(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


# ---------- кумулятивный базис (формула (8) статьи) ----------
def cum_basis(u):
    u = np.asarray(u, float)
    B = np.stack([(5 + 3*u - 3*u**2 + u**3) / 6,
                  (1 + 3*u + 3*u**2 - 2*u**3) / 6,
                  u**3 / 6], axis=-1)
    dB = np.stack([(3 - 6*u + 3*u**2) / 6,
                   (3 + 6*u - 6*u**2) / 6,
                   u**2 / 2], axis=-1)
    ddB = np.stack([u - 1, 1 - 2*u, u], axis=-1)
    return B, dB, ddB


M_POS = np.array([[1, -3, 3, -1],
                  [4, 0, -6, 3],
                  [1, 3, 3, -3],
                  [0, 0, 0, 1]]) / 6.0


class Keyframes:
    """Ключевые позы камеры на равномерной сетке t_k = t0 + k*dt."""

    def __init__(self, t0, dt, R, p, v0, vK, w0=None, wK=None):
        self.t0, self.dt = t0, dt
        self.R, self.p = np.asarray(R), np.asarray(p)
        self.v0, self.vK = np.asarray(v0), np.asarray(vK)
        # угловые скорости камеры на концах (в её системе), нужны для уточнения
        self.w0 = None if w0 is None else np.asarray(w0)
        self.wK = None if wK is None else np.asarray(wK)
        self.K = len(p) - 1
        self.tK = t0 + self.K * dt

    def seg(self, t, ref=None):
        """Номер отрезка i и локальный параметр u (u в [0, 1) внутри отрезка).
        Если задан ref, отрезок берётся по моментам ref, а u считается для t:
        так формула отрезка продолжается за его границу (нужно для пределов
        слева и справа в узле)."""
        x = (np.asarray(t, float) - self.t0) / self.dt
        xr = x if ref is None else (np.asarray(ref, float) - self.t0) / self.dt
        i = np.clip(np.floor(xr).astype(int), 0, self.K - 1)
        return i, x - i


# ---------- модель V: B-сплайны ----------
class ModelV:
    name = "V"

    def __init__(self, kf, refine=False):
        self.kf = kf
        K, dt = kf.K, kf.dt
        # контрольные точки c_1..c_{K+1}: трёхдиагональная система (IV-C)
        n = K + 1
        A = np.zeros((n, n))
        rhs = 6 * kf.p.copy()
        for k in range(n):
            A[k, k] = 4
            if k > 0:
                A[k, k - 1] = 1
            if k < n - 1:
                A[k, k + 1] = 1
        A[0, 1] = 2
        A[-1, -2] = 2
        rhs[0] += 2 * dt * kf.v0
        rhs[-1] -= 2 * dt * kf.vK
        c_in = np.linalg.solve(A, rhs)
        c0 = c_in[1] - 2 * dt * kf.v0
        cK2 = c_in[-2] + 2 * dt * kf.vK
        self.c = np.vstack([c0, c_in, cK2])                   # c_0..c_{K+2}
        # контрольные вращения: Q_{k+1} = R_k (начальное приближение IV-C)
        R = kf.R
        Q0 = R[0] @ Exp(-Log(R[0].T @ R[1]))
        QK2 = R[-1] @ Exp(Log(R[-2].T @ R[-1]))
        self.Q = np.concatenate([Q0[None], R, QK2[None]])     # Q_0..Q_{K+2}
        self.d = Log(np.swapaxes(self.Q[:-1], -1, -2) @ self.Q[1:])
        self.refine_info = None
        if refine:
            self.refine()

    def _set_Q(self, Q):
        self.Q = Q
        self.d = Log(np.swapaxes(Q[:-1], -1, -2) @ Q[1:])

    def key_residuals(self):
        """Невязки по ключевым вращениям [рад] и угловым скоростям на концах."""
        kf = self.kf
        tk = kf.t0 + kf.dt * np.arange(kf.K + 1)
        ref = np.minimum(tk, kf.tK - 0.5 * kf.dt)     # t_K -- конец отрезка K-1
        Rm, _, _, om, _ = self.eval(tk, ref)
        r_rot = Log(np.swapaxes(kf.R, -1, -2) @ Rm)
        return r_rot, om[0], om[-1]

    def refine(self):
        """Задача 1 для ориентации: K+1 условий R(t_k)=R_k и два условия на
        угловую скорость на концах (как v_0, v_K для положения). Неизвестных
        3(K+3), уравнений столько же; решается методом Левенберга-Марквардта."""
        from scipy.optimize import least_squares
        kf = self.kf
        if kf.w0 is None or kf.wK is None:
            raise ValueError("для уточнения нужны угловые скорости на концах")
        Q_init = self.Q.copy()

        def fun(x):
            self._set_Q(Q_init @ Exp(x.reshape(-1, 3)))
            r_rot, w0, wK = self.key_residuals()
            return np.concatenate([r_rot.ravel(), kf.dt * (w0 - kf.w0),
                                   kf.dt * (wK - kf.wK)])

        r0 = fun(np.zeros(3 * len(Q_init)))
        sol = least_squares(fun, np.zeros(3 * len(Q_init)), method='lm',
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        fun(sol.x)
        r_rot, w0, wK = self.key_residuals()
        self.refine_info = dict(
            init_max_rot=float(np.linalg.norm(r0[:3 * (kf.K + 1)].reshape(-1, 3), axis=1).max()),
            max_rot=float(np.linalg.norm(r_rot, axis=1).max()),
            max_w_end=float(max(np.linalg.norm(w0 - kf.w0), np.linalg.norm(wK - kf.wK))),
            nfev=int(sol.nfev), status=int(sol.status))
        return self.refine_info

    def eval(self, t, ref=None):
        kf = self.kf
        t = np.atleast_1d(t)
        i, u = kf.seg(t, ref)
        dt = kf.dt
        C = np.stack([self.c[i + j] for j in range(4)], axis=-1)    # (N,3,4)
        psi = lambda a, b, c_, d: np.stack([a, b, c_, d], -1)[..., None]
        one, zero = np.ones_like(u), np.zeros_like(u)
        p = (C @ M_POS @ psi(one, u, u**2, u**3))[..., 0]
        acc = (C @ M_POS @ psi(zero, zero, 2*one, 6*u))[..., 0] / dt**2
        B, dB, ddB = cum_basis(u)
        dB, ddB = dB / dt, ddB / dt**2
        Rm = self.Q[i].copy()
        om = np.zeros((len(t), 3))
        al = np.zeros((len(t), 3))
        for j in range(3):
            dj = self.d[i + j]
            Aj = Exp(B[:, j:j+1] * dj)
            AjT = np.swapaxes(Aj, -1, -2)
            w = dB[:, j:j+1] * dj
            om_rot = (AjT @ om[..., None])[..., 0]
            al = (AjT @ al[..., None])[..., 0] + np.cross(om_rot, w) + ddB[:, j:j+1] * dj
            om = om_rot + w
            Rm = Rm @ Aj
        return Rm, p, acc, om, al


# ---------- общие утилиты для моделей A и B ----------
def slerp(P, Q, u):
    return P @ Exp(u[..., None] * Log(np.swapaxes(P, -1, -2) @ Q))


def numeric_rates(Rfun, t, eps=1e-3, seg=None):
    """omega, alpha в собственной системе по 5-точечным разностям.
    Rfun(t, seg) вычисляет поворот по формуле отрезка seg (гладкое продолжение
    за его границы), поэтому шаблон не перескакивает через узел."""
    kf_seg = Rfun.seg_of(t) if seg is None else seg

    def omega_at(tt):
        offs = np.array([-2, -1, 1, 2]) * eps
        w = np.array([1, -8, 8, -1]) / (12 * eps)
        Rd = sum(wk * Rfun(tt + ok, kf_seg) for wk, ok in zip(w, offs))
        return vee(np.swapaxes(Rfun(tt, kf_seg), -1, -2) @ Rd)

    om = omega_at(t)
    offs = np.array([-2, -1, 1, 2]) * eps
    w = np.array([1, -8, 8, -1]) / (12 * eps)
    al = sum(wk * omega_at(t + ok) for wk, ok in zip(w, offs))
    return om, al


# ---------- модель A: отрезки + SLERP ----------
class ModelA:
    name = "A"

    def __init__(self, kf):
        self.kf = kf

    def eval(self, t, ref=None):
        kf = self.kf
        t = np.atleast_1d(t)
        i, u = kf.seg(t, ref)
        p = kf.p[i] + u[:, None] * (kf.p[i + 1] - kf.p[i])
        acc = np.zeros_like(p)                       # внутри отрезка p'' = 0
        d = Log(np.swapaxes(kf.R[i], -1, -2) @ kf.R[i + 1])
        Rm = kf.R[i] @ Exp(u[:, None] * d)
        om = d / kf.dt                               # постоянна на отрезке
        al = np.zeros_like(om)
        return Rm, p, acc, om, al


# ---------- модель B: кубический сплайн + SQUAD ----------
class ModelB:
    name = "B"

    def __init__(self, kf):
        self.kf = kf
        tk = kf.t0 + kf.dt * np.arange(kf.K + 1)
        self.cs = CubicSpline(tk, kf.p, bc_type=((1, kf.v0), (1, kf.vK)))
        R = kf.R
        a = [R[0]]
        for k in range(1, kf.K):
            Rt = R[k].T
            a.append(R[k] @ Exp(-(Log(Rt @ R[k + 1]) + Log(Rt @ R[k - 1])) / 4))
        a.append(R[-1])
        self.a = np.array(a)
        model = self

        def Rfun(tt, seg):
            u = (tt - kf.t0) / kf.dt - seg
            q0, q1 = R[seg], R[seg + 1]
            s0, s1 = model.a[seg], model.a[seg + 1]
            return slerp(slerp(q0, q1, u), slerp(s0, s1, u), 2 * u * (1 - u))

        Rfun.seg_of = lambda tt: kf.seg(tt)[0]
        self.Rfun = Rfun

    def eval(self, t, ref=None):
        t = np.atleast_1d(t)
        i, _ = self.kf.seg(t, ref)
        Rm = self.Rfun(t, i)
        # кусочный многочлен отрезка i (а не того, куда попало t)
        x = t[:, None] - self.cs.x[i][:, None]
        c = self.cs.c[:, i, :]                       # (4, N, 3)
        p = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
        acc = 6 * c[0] * x + 2 * c[1]
        om, al = numeric_rates(self.Rfun, t, seg=i)
        return Rm, p, acc, om, al


# ---------- пересчёт камера -> IMU и модель измерений ----------
class Rig:
    def __init__(self, R_BC, p_BC):
        self.R_BC = np.asarray(R_BC)
        self.p_BC = np.asarray(p_BC)
        self.r = -self.R_BC.T @ self.p_BC           # положение IMU в системе C

    def imu_state(self, model, t, ref=None):
        Rwc, pwc, awc, omc, alc = model.eval(t, ref)
        r = self.r
        Rwb = Rwc @ self.R_BC.T
        pwb = pwc + (Rwc @ r)
        rot_term = np.cross(alc, r) + np.cross(omc, np.cross(omc, r))
        awb = awc + (Rwc @ rot_term[..., None])[..., 0]
        omb = (self.R_BC @ omc[..., None])[..., 0]
        return Rwb, pwb, awb, omb

    def ideal_imu(self, model, t, ref=None):
        Rwb, _, awb, omb = self.imu_state(model, t, ref)
        f = (np.swapaxes(Rwb, -1, -2) @ (awb - G_W)[..., None])[..., 0]
        return omb, f


def add_noise(omega, acc, h, sg, sa, sbg, sba, rng, bg0=None, ba0=None):
    """Дискретизация шума по формуле (11) статьи."""
    n = len(omega)
    bg = np.zeros(3) if bg0 is None else bg0
    ba = np.zeros(3) if ba0 is None else ba0
    BG = np.empty((n, 3)); BA = np.empty((n, 3))
    for k in range(n):
        BG[k], BA[k] = bg, ba
        bg = bg + sbg * np.sqrt(h) * rng.standard_normal(3)
        ba = ba + sba * np.sqrt(h) * rng.standard_normal(3)
    om = omega + BG + sg / np.sqrt(h) * rng.standard_normal((n, 3))
    ac = acc + BA + sa / np.sqrt(h) * rng.standard_normal((n, 3))
    return om, ac


# ---------- обратное интегрирование ----------
def integrate_euler(R0, v0, p0, om, acc, h):
    """Схема (12) статьи, первый порядок."""
    n = len(om)
    R, v, p = R0.copy(), v0.copy(), p0.copy()
    P = np.empty((n, 3)); RR = np.empty((n, 3, 3))
    P[0], RR[0] = p, R
    for k in range(n - 1):
        a = R @ acc[k] + G_W
        p = p + v * h + 0.5 * a * h * h
        v = v + a * h
        R = R @ Exp(om[k] * h)
        P[k + 1], RR[k + 1] = p, R
    return RR, P


def integrate_rk4(R0, v0, p0, om_half, acc_half, h):
    """РК4; измерения нужны в узлах сетки с шагом h/2 (индексы 0,1,2,...)."""
    n = (len(om_half) - 1) // 2 + 1
    R, v, p = R0.copy(), v0.copy(), p0.copy()
    P = np.empty((n, 3)); RR = np.empty((n, 3, 3))
    P[0], RR[0] = p, R

    def f(R, v, k):
        return R @ hat(om_half[k]), R @ acc_half[k] + G_W, v

    for s in range(n - 1):
        k0, k1, k2 = 2 * s, 2 * s + 1, 2 * s + 2
        dR1, dv1, dp1 = f(R, v, k0)
        dR2, dv2, dp2 = f(R + 0.5*h*dR1, v + 0.5*h*dv1, k1)
        dR3, dv3, dp3 = f(R + 0.5*h*dR2, v + 0.5*h*dv2, k1)
        dR4, dv4, dp4 = f(R + h*dR3, v + h*dv3, k2)
        R = polar(R + h/6 * (dR1 + 2*dR2 + 2*dR3 + dR4))
        v = v + h/6 * (dv1 + 2*dv2 + 2*dv3 + dv4)
        p = p + h/6 * (dp1 + 2*dp2 + 2*dp3 + dp4)
        P[s + 1], RR[s + 1] = p, R
    return RR, P


def integrate_rk4_lr(R0, v0, p0, om_half, acc_half, om_left, acc_left, h):
    """РК4, где последняя стадия шага берёт предел слева (om_left, acc_left)."""
    n = (len(om_half) - 1) // 2 + 1
    R, v, p = R0.copy(), v0.copy(), p0.copy()
    P = np.empty((n, 3)); RR = np.empty((n, 3, 3))
    P[0], RR[0] = p, R
    for s in range(n - 1):
        k0, k1, k2 = 2 * s, 2 * s + 1, 2 * s + 2
        dR1, dv1, dp1 = R @ hat(om_half[k0]), R @ acc_half[k0] + G_W, v
        R2, v2 = R + 0.5*h*dR1, v + 0.5*h*dv1
        dR2, dv2, dp2 = R2 @ hat(om_half[k1]), R2 @ acc_half[k1] + G_W, v2
        R3, v3 = R + 0.5*h*dR2, v + 0.5*h*dv2
        dR3, dv3, dp3 = R3 @ hat(om_half[k1]), R3 @ acc_half[k1] + G_W, v3
        R4, v4 = R + h*dR3, v + h*dv3
        dR4, dv4, dp4 = R4 @ hat(om_left[k2]), R4 @ acc_left[k2] + G_W, v4
        R = polar(R + h/6 * (dR1 + 2*dR2 + 2*dR3 + dR4))
        v = v + h/6 * (dv1 + 2*dv2 + 2*dv3 + dv4)
        p = p + h/6 * (dp1 + 2*dp2 + 2*dp3 + dp4)
        P[s + 1], RR[s + 1] = p, R
    return RR, P


def imu_velocity(rig, model, t, eps=1e-3):
    """Скорость IMU в начальный момент: 5-точечная разность по формуле отрезка
    (нужна только для начального условия; погрешность ~1e-13 м/с)."""
    offs = np.array([-2, -1, 1, 2]) * eps
    w = np.array([1, -8, 8, -1]) / (12 * eps)
    i0, _ = model.kf.seg(np.array([t]))
    pts = []
    for o in offs:
        tt = np.array([t + o])
        if model.kf.seg(tt)[0][0] != i0[0]:
            raise RuntimeError("шаблон пересёк узел; уменьшите eps")
        pts.append(rig.imu_state(model, tt)[1][0])
    return sum(wk * pk for wk, pk in zip(w, pts))


def run_consistency(rig, model, t_start, t_end, h, scheme, left_limit=False):
    """Интегрирует идеальные показания модели и сравнивает с её же траекторией.
    Возвращает RMS ошибки положения [м] и ориентации [рад].
    left_limit=True (только РК4): в конце шага берётся значение по формуле того
    отрезка, где лежит середина шага, то есть предел слева в узле. Реальный
    датчик так не умеет: в каждый момент у него одно показание."""
    n = int(round((t_end - t_start) / h)) + 1
    s = t_start + h * np.arange(n)
    Rwb, pwb, _, _ = rig.imu_state(model, s)
    v0 = imu_velocity(rig, model, t_start)
    if scheme == "euler":
        om, ac = rig.ideal_imu(model, s)
        RR, P = integrate_euler(Rwb[0], v0, pwb[0], om, ac, h)
    else:
        s2 = t_start + (h / 2) * np.arange(2 * n - 1)
        om, ac = rig.ideal_imu(model, s2)
        if left_limit:
            ends = s2[2::2]
            om_e, ac_e = rig.ideal_imu(model, ends, ref=ends - h / 2)
            om_l, ac_l = om.copy(), ac.copy()
            om_l[2::2], ac_l[2::2] = om_e, ac_e
            RR, P = integrate_rk4_lr(Rwb[0], v0, pwb[0], om, ac, om_l, ac_l, h)
        else:
            RR, P = integrate_rk4(Rwb[0], v0, pwb[0], om, ac, h)
    ep = np.linalg.norm(P - pwb, axis=1)
    eR = np.linalg.norm(Log(np.swapaxes(RR, -1, -2) @ Rwb), axis=1)
    return np.sqrt(np.mean(ep**2)), np.sqrt(np.mean(eR**2))


# ---------- тестовая траектория ----------
def demo_keyframes(K=20, dt=0.5):
    """Восьмёрка с колебаниями по высоте, рысканием и креном."""
    def pos(t):
        return np.stack([2.0*np.sin(0.4*t), 1.5*np.sin(0.8*t), 1.0 + 0.3*np.sin(0.6*t)], -1)

    def vel(t):
        return np.stack([0.8*np.cos(0.4*t), 1.2*np.cos(0.8*t), 0.18*np.cos(0.6*t)], -1)

    def rotvec(t):
        return np.stack([0.15*np.sin(0.9*t), 0.2*np.sin(0.7*t), 0.6*np.sin(0.3*t)], -1)

    def body_rate(t, eps=1e-3):
        """Угловая скорость Exp(rotvec(t)) в собственной системе (5-точечная
        разность по аналитической функции, погрешность ~1e-13)."""
        offs = np.array([-2, -1, 1, 2]) * eps
        w = np.array([1, -8, 8, -1]) / (12 * eps)
        Rd = sum(wk * Exp(rotvec(t + o)) for wk, o in zip(w, offs))
        return vee(Exp(rotvec(t)).T @ Rd)

    tk = dt * np.arange(K + 1)
    R = Exp(rotvec(tk))
    return Keyframes(0.0, dt, R, pos(tk), vel(tk[0]), vel(tk[-1]),
                     body_rate(tk[0]), body_rate(tk[-1]))


def demo_rig():
    """Камера смещена относительно IMU на 8,5 см и повёрнута на 10° по рысканию."""
    return Rig(Exp(np.array([0.0, 0.0, np.deg2rad(10)])), np.array([0.08, 0.0, 0.03]))

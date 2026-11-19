import numpy as np
from scipy.spatial.transform import Rotation as Rot

G_W = np.array([0.0, 0.0, -9.81])


def Exp(phi):
    return Rot.from_rotvec(phi).as_matrix()


def Log(R):
    return Rot.from_matrix(R).as_rotvec()


def cum_basis(u):
    """Кумулятивный базис и его производные по u."""
    B = np.array([(5 + 3*u - 3*u**2 + u**3) / 6,
                  (1 + 3*u + 3*u**2 - 2*u**3) / 6,
                  u**3 / 6])
    dB = np.array([(3 - 6*u + 3*u**2) / 6,
                   (3 + 6*u - 6*u**2) / 6,
                   u**2 / 2])
    ddB = np.array([u - 1, 1 - 2*u, u])
    return B, dB, ddB


def rotation_spline(Q, t, t0, dt):
    """R, omega, alpha кумулятивного B-сплайна на SO(3) в момент t."""
    i = min(int((t - t0) // dt), len(Q) - 4)
    u = (t - t0) / dt - i
    B, dB, ddB = cum_basis(u)
    dB, ddB = dB / dt, ddB / dt**2
    R, om, al = Q[i].copy(), np.zeros(3), np.zeros(3)
    for j in range(3):
        d = Log(Q[i + j].T @ Q[i + j + 1])
        A = Exp(B[j] * d)
        w = dB[j] * d
        om_rot = A.T @ om
        al = A.T @ al + np.cross(om_rot, w) + ddB[j] * d
        om = om_rot + w
        R = R @ A
    return R, om, al


def ideal_imu(R_wc, acc_wc, om_c, al_c, R_bc, r):
    """Идеальные показания IMU по движению камеры.
    r -- положение IMU в системе камеры."""
    lever = np.cross(al_c, r) + np.cross(om_c, np.cross(om_c, r))
    acc_wb = acc_wc + R_wc @ lever
    R_wb = R_wc @ R_bc.T
    gyro = R_bc @ om_c
    accel = R_wb.T @ (acc_wb - G_W)
    return gyro, accel

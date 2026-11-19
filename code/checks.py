import numpy as np, sys
sys.path.insert(0, '.')
from spline_imu import *
kf = demo_keyframes(); V = ModelV(kf); A = ModelA(kf); B = ModelB(kf)
t = np.linspace(0.13, 9.7, 37); e = 1e-5
R, p, a, om, al = V.eval(t)
Rp, pp, _, omp, _ = V.eval(t+e); Rm, pm, _, omm, _ = V.eval(t-e)
_, _, _, _, _ = V.eval(t)
print('V: p_k interp err', np.abs(V.eval(kf.t0+kf.dt*np.arange(kf.K)+0)[1]-kf.p[:-1]).max())
print('V: vel at ends', V.eval(np.array([0.0]))[1], )
print('V: acc vs FD', np.abs(a-(pp-2*p+pm)/e**2).max())
omfd = vee(np.swapaxes(R,-1,-2)@(Rp-Rm)/(2*e))
print('V: omega vs FD', np.abs(om-omfd).max(), ' |om| max', np.abs(om).max())
print('V: alpha vs FD', np.abs(al-(omp-omm)/(2*e)).max(), ' |al| max', np.abs(al).max())
print('B vs V position (same function?)', np.abs(B.eval(t)[1]-p).max())
knots = kf.dt*np.arange(1,kf.K); d=1e-9
for M in (A,B,V):
    L = M.eval(knots-d); Rr = M.eval(knots+d)
    print(M.name, 'jump at knots: R %.2e  acc %.2e  omega %.2e  alpha %.2e' % tuple(np.abs(L[k]-Rr[k]).max() for k in (0,2,3,4)))
rig = demo_rig()
for M in (A,B,V):
    oL,fL = rig.ideal_imu(M, knots-d); oR,fR = rig.ideal_imu(M, knots+d)
    print(M.name, 'IMU jump: gyro %.3e rad/s, accel %.3e m/s^2' % (np.abs(oL-oR).max(), np.abs(fL-fR).max()))
print('r =', rig.r)

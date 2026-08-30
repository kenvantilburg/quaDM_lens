"""How the stochastic and acceleration channels compare, and why the comparison looks
different in Fig. matter_power than in Fig. SNR.

Both curves are "SNR = 1" thresholds, but on different statistics over different bands:
  stoch:  Delta^2 such that C~(omega_min) = sigma_delta_theta^2 tau/N   [lowest mode only]
  acc:    Delta^2 such that int C~ omega^4 F_2 domega/2pi = 720 sigma_delta_theta^2/(N tau^4)

For the white-truncated line-of-sight spectrum of Fig. matter_power the two land within
~10% of one another (acc/stoch = 0.88 at their respective optima, 1.08 at a common
k = 1e8/Mpc; with the single-plane kernel these read 0.76 and 1.25). The reason is
that each channel's optimum sits where the spectrum's
frequency cutoff omega_cut = k mu_tilde D falls BELOW the band edge 2 pi/tau, so both
statistics are fed almost entirely by the unresolved band and differ only by the
order-unity kernel factor 720/int u^3 F_2(u) du = 2.4 computed below.

For the lens-bound monochromatic halos of Fig. SNR the acceleration channel pulls ahead,
and by more and more as M_L grows (acc/stoch ~ 0.8, 2, 3e3 at M_L = 1e-5, 1e-4, 1e-3
M_Solar): the halo crossing frequency mu_tilde/gamma_L drops below the band edge, which
removes the resolved stochastic modes entirely while the acceleration statistic keeps
integrating the low-frequency power that remains.
"""
import numpy as np
from scipy.integrate import quad
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from sensitivity_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import params_B1422_231 as pB

tau = 10*year; N_obs = 300; sigma_delta_theta = 0.1*muas
omega_min = 2*np.pi/tau

def smear_F2(u):
    """F_2(omega tau) of Eq. (F_2_smearing)"""
    u = np.atleast_1d(u).astype(float)
    out = np.ones_like(u)
    m = u > 1e-2
    out[m] = 14400*(6*u[m]*np.cos(u[m]/2) + (-12+u[m]**2)*np.sin(u[m]/2))**2/u[m]**10
    return out

# ---- the kernel-only identity: ratio -> 720/int_u3_F2 when cutoff >> band edge ----
int_u3_F2 = quad(lambda u: u**3*smear_F2(u)[0], 0, 200, limit=800)[0]
print('int_u3_F2 = int u^3 F_2(u) du = %.3f   ->   720/int_u3_F2 = %.2f' % (int_u3_F2, 720/int_u3_F2))

# ---- line-of-sight configuration (galaxy lens B1422+231, image B tangential) ----
res = np.load('macro_lens_results.npz')
B_tang = np.max(np.abs(res['eigvals_fit'][1]))   # tangential eigenvalue B_t^I, image B
mu_t = np.sqrt((B_tang*np.linalg.norm(pB.v_lens_fid)/pB.D_lens)**2 + pB.mu_L_int**2)
vec_a = np.logspace(np.log10(0.999), np.log10(1/(1+pB.z_source)), 2000)
vec_D = cosmo.angular_diameter_distance(1/vec_a-1).value*Mpc/vec_a
D_src = pB.d_source/(1/(1+pB.z_source))
iL = np.argmin(np.abs(vec_D - pB.D_lens))

X_los = sweep_offplane(vec_D, pB.D_lens, D_src)            # tent-shaped sweep lever arm
B_eff_los = B_offplane(vec_D, pB.D_lens, D_src, B_tang)    # (1-beta) B + beta

def I_M(omega, k):
    """Angular kernel of Eq. (C_tilde_1) for a white spectrum truncated at k, Delta^2 = 1"""
    x = (omega/(mu_t*np.maximum(X_los, 1e-30)))/k
    out = np.zeros_like(x)
    m = x <= 1
    out[m] = 3*np.pi/k**3*(np.arccos(np.clip(x[m],-1,1)) + x[m]*np.sqrt(1-x[m]**2))
    return out
def Ctilde(omega, k):  # C~(omega) of Eq. (C_tilde_1) at unit Delta^2 per e-fold
    integ = B_eff_los**2*(1-vec_D/D_src)**2*I_M(omega,k)*(4*np.pi*G_N*rho_M_0/vec_a)**2
    return 4/omega*np.trapezoid(integ, vec_D)

N_white = sigma_delta_theta**2*tau/N_obs
s2_acc_noise = 720*sigma_delta_theta**2/(tau**4*N_obs)
vec_omega = omega_min*np.logspace(-6, 3, 1200)

def thresholds(k):
    # Below k ~ omega_min/(mu_tilde X_max) = omega_min/(mu_tilde chi_L) no plane along the
    # sightline sweeps fast enough to put power in the lowest mode, and Cstoch vanishes:
    # the off-plane kernel puts a hard low-k wall on the stochastic channel that the
    # single-plane kernel did not have (it swept the far sightline at chi_S mu_tilde).
    Cstoch = Ctilde(omega_min, k)
    D2_stoch = N_white/Cstoch if Cstoch > 0 else np.inf
    vecC = np.array([Ctilde(w, k) for w in vec_omega])
    # factor 2 for the negative frequencies (C~ is two-sided), as in matter_power.ipynb
    # and every other script here; without it the acceleration threshold comes out 2x high.
    s2_sig = np.trapezoid(2*vecC*vec_omega**4*smear_F2(vec_omega*tau)/(2*np.pi), vec_omega)
    D2_acc = s2_acc_noise/s2_sig
    omega_cut = k*mu_t*X_los[iL]
    return D2_stoch, D2_acc, omega_cut/omega_min

# scan k to find each channel's optimum
vec_k = np.logspace(6.0, 9.5, 60)*Mpc**-1
rows = np.array([thresholds(k) for k in vec_k])
i_st = np.argmin(rows[:,0]); i_ac = np.argmin(rows[:,1])
print('\nstoch  min Delta^2 = %.3g at k=%.2e h/Mpc, omega_cut/omega_min=%.2f'
      % (rows[i_st,0], vec_k[i_st]/(cosmo.h*Mpc**-1), rows[i_st,2]))
print('acc    min Delta^2 = %.3g at k=%.2e h/Mpc, omega_cut/omega_min=%.2f'
      % (rows[i_ac,1], vec_k[i_ac]/(cosmo.h*Mpc**-1), rows[i_ac,2]))
print('ratio acc/stoch at respective minima = %.2f' % (rows[i_ac,1]/rows[i_st,0]))
print('ratio acc/stoch at a COMMON high k (1e8/Mpc):')
d2s,d2a,r = thresholds(1e8*Mpc**-1); print('   %.3g / %.3g = %.2f (omega_cut/omega_min=%.1f)'%(d2a,d2s,d2a/d2s,r))

# ---- per-mode SNR spectrum at the stochastic optimum: the redness of C~ ~ 1/omega
# means only the lowest few DFT modes carry weight (cf. N_eff ~ 6 in Sec. II E) ----
k = vec_k[i_st]
n = np.arange(1, 60)
lam = np.array([Ctilde(nn*omega_min, k) for nn in n])/N_white
D2_stoch = rows[i_st,0]
lam *= D2_stoch   # normalize to the stochastic-threshold amplitude (lambda_1 = 1)
print('\nAt stoch optimum (lambda normalized so lowest mode=1):')
print('  lambda_1,2,3,5,10 =', np.round(lam[[0,1,2,4,9]], 3))
frac_resolved = 1.0  # by construction acc lives below omega_min; quantify:
# fraction of acceleration SIGNAL variance from unresolved band (omega<omega_min)
k_ac = vec_k[i_ac]
vecC = np.array([Ctilde(w, k_ac) for w in vec_omega])
w_ac = vecC*vec_omega**4*smear_F2(vec_omega*tau)/(2*np.pi)
below = vec_omega < omega_min
print('acc signal fraction from unresolved band (omega<2pi/tau) = %.2f'
      % (np.trapezoid(w_ac[below], vec_omega[below])/np.trapezoid(w_ac, vec_omega)))

# ===== lens-bound monochromatic halos (Eq. (C)) for comparison: here the acceleration
# channel wins once the halo crossing frequency mu_tilde/gamma_L drops below the band edge =====
print('\n===== lens-bound one-halo (Gaussian profile), image B =====')
kappa_L = 0.5*res['kappa_fit'][1]
def Ctilde_halo(omega, M_L, rho_s):
    r_L = (M_L/(4*np.pi*np.sqrt(np.e)*rho_s))**(1/3)
    gamma_L = r_L/pB.d_lens
    tE = theta_E(M_L, pB.d_lens, pB.d_source, pB.d_lens_source)
    xk = gamma_L/mu_t*omega
    Cij = kappa_L*tE**2/omega*C_ij_integral(xk, 0.0)  # zeta=0 diag piece scale
    return Cij[0,0]
for M_L in [1e-5, 1e-4, 1e-3]:
    M = M_L*M_Solar; rho_s = 1*M_Solar/pc**3
    Cst = Ctilde_halo(omega_min, M, rho_s)
    vecC = np.array([Ctilde_halo(w, M, rho_s) for w in vec_omega])
    s2sig = np.trapezoid(2*vecC*vec_omega**4*smear_F2(vec_omega*tau)/(2*np.pi), vec_omega)
    r_L = (M/(4*np.pi*np.sqrt(np.e)*rho_s))**(1/3)
    omega_halo = mu_t/(r_L/pB.d_lens)   # crossing frequency mu/gamma
    below = vec_omega < omega_min
    w = vecC*vec_omega**4*smear_F2(vec_omega*tau)
    fbelow = np.trapezoid(w[below], vec_omega[below])/np.trapezoid(w, vec_omega)
    print('M_L=%.0e: SNR_stoch(1mode)=%.2g  SNR_acc=%.2g  ratio acc/stoch=%.2g | '
          'omega_cross/omega_min=%.2f, acc frac unresolved=%.2f'
          % (M_L, Cst/N_white, s2sig/s2_acc_noise, (s2sig/s2_acc_noise)/(Cst/N_white),
             omega_halo/omega_min, fbelow))

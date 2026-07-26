"""Why does the matter_power figure make the STOCHASTIC curve ~18x more sensitive
(lower min Delta^2) than the ACCELERATION curve, while the lens-bound one-halo
figure has acceleration as the stronger channel?

Two threshold definitions (matter_power.ipynb):
  stoch:  Delta^2 s.t. C_tilde(omega_min) = N_white          [single lowest mode]
  acc:    Delta^2 s.t. int C_tilde omega^4 F2 domega/2pi = 720 sigma^2/(N tau^4)
Both are 'SNR=1' on ONE statistic, but different statistics & different bands.
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

t_int = 10*year; N_obs = 300; sigma = 0.1*muas
omega_min = 2*np.pi/t_int
rho_m0 = cosmo.Om0*rho_crit

def F2(u):
    u = np.atleast_1d(u).astype(float)
    out = np.ones_like(u)
    m = u > 1e-2
    out[m] = 14400*(6*u[m]*np.cos(u[m]/2) + (-12+u[m]**2)*np.sin(u[m]/2))**2/u[m]**10
    return out

# ---- the kernel-only identity: ratio -> 720/J when cutoff >> band edge ----
J = quad(lambda u: u**3*F2(u)[0], 0, 200, limit=800)[0]
print('J = int u^3 F2(u) du = %.3f   ->   720/J = %.2f' % (J, 720/J))

# ---- LOS system (galaxy lens, image B tangential) ----
res = np.load('macro_lens_results.npz')
B = np.max(np.abs(res['eigvals_fit'][1]))
mu_t = np.sqrt((B*np.linalg.norm(pB.v_lens_fid)/pB.D_lens)**2 + pB.mu_int_drift**2)
vec_a = np.logspace(np.log10(0.999), np.log10(1/(1+pB.z_source)), 2000)
vec_D = cosmo.angular_diameter_distance(1/vec_a-1).value*Mpc/vec_a
D_src = pB.d_source/(1/(1+pB.z_source))
iL = np.argmin(np.abs(vec_D - pB.D_lens))

def I_lm(omega, kc):
    x = (omega/(mu_t*vec_D))/kc
    out = np.zeros_like(x)
    m = x <= 1
    out[m] = 3*np.pi/kc**3*(np.arccos(np.clip(x[m],-1,1)) + x[m]*np.sqrt(1-x[m]**2))
    return out
def Ctilde(omega, kc):  # Delta^2 = 1
    integ = (1-vec_D/D_src)**2*I_lm(omega,kc)*(4*np.pi*G_N*rho_m0/vec_a)**2
    return B**2*4/omega*np.trapezoid(integ, vec_D)

N_white = sigma**2*t_int/N_obs
s2_acc_noise = 720*sigma**2/(t_int**4*N_obs)
vec_omega = omega_min*np.logspace(-6, 3, 1200)

def thresholds(kc):
    Cstoch = Ctilde(omega_min, kc)
    D2_stoch = N_white/Cstoch
    vecC = np.array([Ctilde(w, kc) for w in vec_omega])
    s2_sig = np.trapezoid(vecC*vec_omega**4*F2(vec_omega*t_int)/(2*np.pi), vec_omega)
    D2_acc = s2_acc_noise/s2_sig
    omega_cut = kc*mu_t*vec_D[iL]
    return D2_stoch, D2_acc, omega_cut/omega_min

# scan k_c to find each channel's optimum
kcs = np.logspace(6.0, 9.5, 60)*Mpc**-1
rows = np.array([thresholds(kc) for kc in kcs])
i_st = np.argmin(rows[:,0]); i_ac = np.argmin(rows[:,1])
print('\nstoch  min Delta^2 = %.3g at k=%.2e h/Mpc, omega_cut/omega_min=%.2f'
      % (rows[i_st,0], kcs[i_st]/(cosmo.h*Mpc**-1), rows[i_st,2]))
print('acc    min Delta^2 = %.3g at k=%.2e h/Mpc, omega_cut/omega_min=%.2f'
      % (rows[i_ac,1], kcs[i_ac]/(cosmo.h*Mpc**-1), rows[i_ac,2]))
print('ratio acc/stoch at respective minima = %.2f' % (rows[i_ac,1]/rows[i_st,0]))
print('ratio acc/stoch at a COMMON high k_c (1e8/Mpc):')
d2s,d2a,r = thresholds(1e8*Mpc**-1); print('   %.3g / %.3g = %.2f (omega_cut/omega_min=%.1f)'%(d2a,d2s,d2a/d2s,r))

# ---- per-mode SNR spectrum at the stochastic optimum: where does the power live? ----
kc = kcs[i_st]
n = np.arange(1, 60)
lam = np.array([Ctilde(nn*omega_min, kc) for nn in n])/N_white
D2_stoch = rows[i_st,0]
lam *= D2_stoch   # normalize to the stochastic-threshold amplitude (lambda_1 = 1)
print('\nAt stoch optimum (lambda normalized so lowest mode=1):')
print('  lambda_1,2,3,5,10 =', np.round(lam[[0,1,2,4,9]], 3))
frac_resolved = 1.0  # by construction acc lives below omega_min; quantify:
# fraction of acceleration SIGNAL variance from unresolved band (omega<omega_min)
kc_ac = kcs[i_ac]
vecC = np.array([Ctilde(w, kc_ac) for w in vec_omega])
w_ac = vecC*vec_omega**4*F2(vec_omega*t_int)/(2*np.pi)
below = vec_omega < omega_min
print('acc signal fraction from unresolved band (omega<2pi/tau) = %.2f'
      % (np.trapezoid(w_ac[below], vec_omega[below])/np.trapezoid(w_ac, vec_omega)))

# ================= lens-bound (Gaussian halos) crossover =================
print('\n===== lens-bound one-halo (Gaussian profile), image B =====')
kappa_L = 0.5*res['kappa_fit'][1]
def Ctilde_halo(omega, M_L, rho_s):
    r_L = (M_L/(4*np.pi*np.sqrt(np.e)*rho_s))**(1/3)
    gamma = r_L/pB.d_lens
    tE = theta_E(M_L, pB.d_lens, pB.d_source, pB.d_lens_source)
    xk = gamma/mu_t*omega
    Cij = kappa_L*tE**2/omega*C_ij_integral(xk, 0.0)  # zeta=0 diag piece scale
    return Cij[0,0]
for M_L in [1e-5, 1e-4, 1e-3]:
    M = M_L*M_Solar; rho_s = 1*M_Solar/pc**3
    Cst = Ctilde_halo(omega_min, M, rho_s)
    vecC = np.array([Ctilde_halo(w, M, rho_s) for w in vec_omega])
    s2sig = np.trapezoid(vecC*vec_omega**4*F2(vec_omega*t_int)/(2*np.pi), vec_omega)
    r_L = (M/(4*np.pi*np.sqrt(np.e)*rho_s))**(1/3)
    omega_halo = mu_t/(r_L/pB.d_lens)   # crossing frequency mu/gamma
    below = vec_omega < omega_min
    w = vecC*vec_omega**4*F2(vec_omega*t_int)
    fbelow = np.trapezoid(w[below], vec_omega[below])/np.trapezoid(w, vec_omega)
    print('M_L=%.0e: SNR_stoch(1mode)=%.2g  SNR_acc=%.2g  ratio acc/stoch=%.2g | '
          'omega_cross/omega_min=%.2f, acc frac unresolved=%.2f'
          % (M_L, Cst/N_white, s2sig/s2_acc_noise, (s2sig/s2_acc_noise)/(Cst/N_white),
             omega_halo/omega_min, fbelow))

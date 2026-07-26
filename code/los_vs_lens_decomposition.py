"""Decompose the gap between the lens-bound halo-spectrum sensitivity (Fig. SNR,
variance SNR ~ 13 at the CDM prediction) and the line-of-sight matter-power
sensitivity (Fig. matter_power, min detectable Delta^2 orders of magnitude above
the CDM band), for the galaxy lens B1422+231 in the acceleration channel.

Chain (one change at a time, identical kernels throughout):
  S1  : lens-plane monochromatic clumps (f_sub kappa at one mass), Eq.-C machinery,
        image A only  -> anchors to the published SNR numbers.
  S1' : the same population as a thin slab at D_L, evaluated in the C_tilde_1
        (line-of-sight) scalar machinery -> internal consistency check vs S1.
  S2  : the same clumps (same mass, size, profile) spread along the line of sight
        at the COSMIC MEAN density, carrying f = 0.5 of all matter.
        F_bias = S1'/S2 : the biased-column factor.
  S3  : the line of sight carrying the actual pure-CDM one-halo power spectrum
        (Sheth-Tormen x NFW, no cutoff; same machinery, general Delta^2(k)).
        F_pop = S2/S3 : the population-efficiency factor (mass function spread +
        NFW profile dilution vs. all mass in compact sweet-spot clumps).
Total:  S1'/S3 = F_bias * F_pop  should reproduce the figure-to-figure gap.
"""
import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.interpolate import interp1d
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from sensitivity_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import params_B1422_231 as pB

# ---------------------------------------------------------------- survey + system
t_int = 10 * year
N_obs = 300
sigma_dth = 0.1 * muas
sigma2_acc_noise = 720 * sigma_dth**2 / (t_int**4 * N_obs)   # instrument-only floor

res = np.load('macro_lens_results.npz')
inv_jacobian_fit = res['inv_jacobian_fit']; kappa_fit = res['kappa_fit']
eigvals = res['eigvals_fit']
vec_mu = mu_rel(np.zeros(2)*km/second, pB.v_lens_fid, np.zeros(2)*km/second,
                pB.z_lens, pB.z_source, pB.d_lens, pB.d_source, pB.d_lens_source)
vec_mu_tilde = np.tensordot(inv_jacobian_fit, vec_mu, axes=1)
mu_tilde_A = np.sqrt(np.linalg.norm(vec_mu_tilde[0])**2 + pB.mu_int_drift**2)
zeta_A = np.arctan2(vec_mu_tilde[0, 1], vec_mu_tilde[0, 0])
B_A = np.max(np.abs(eigvals[0]))          # tangential eigenvalue of image A
kappa_L_A = 0.5 * kappa_fit[0]            # f_sub = 0.5 of the image-A convergence
Sig_cr = Sigma_crit(pB.d_lens, pB.d_source, pB.d_lens_source)
a_L = 1 / (1 + pB.z_lens)

# monochromatic benchmark clump: acceleration-channel sweet spot
rho_s = 1.0 * M_Solar / pc**3
M_L = 2e-4 * M_Solar
r_L = (M_L / (4*np.pi*np.sqrt(np.e)*rho_s))**(1/3)
gamma_L = r_L / pB.d_lens
tE = theta_E(M_L, pB.d_lens, pB.d_source, pB.d_lens_source)
print('image A: |B| = %.2f, mu_tilde = %.2f muas/yr, kappa_L = %.3f' %
      (B_A, mu_tilde_A/muasy, kappa_L_A))
print('clump: M_L = %.1e Msun, r_L = %.3f pc, theta_E = %.3g muas, gamma_L = %.1f muas'
      % (M_L/M_Solar, r_L/pc, tE/muas, gamma_L/muas))
print('Sigma_cr = %.4g Msun/pc^2 -> Sigma_sub = %.4g Msun/pc^2'
      % (Sig_cr/(M_Solar/pc**2), kappa_L_A*Sig_cr/(M_Solar/pc**2)))

def smear_fac_acc(a):
    return np.heaviside(-a + 1e-2, 0) + np.heaviside(a - 1e-2, 0) * 14400 * (
        6*a*np.cos(a/2) + (-12 + a**2)*np.sin(a/2))**2 / a**10

vec_omega = 2*np.pi/t_int * np.logspace(-7, 2, 600)

def acc_var_from_C(vec_C):
    """acceleration variance from a one-sided PSD sampled on vec_omega."""
    return np.trapezoid(2 * vec_C * vec_omega**4 * smear_fac_acc(vec_omega*t_int)
                        / (2*np.pi), vec_omega)

# ==================================================================== S1: Eq.-C
x_k = gamma_L / mu_tilde_A * vec_omega
C_ij = kappa_L_A * tE**2 / vec_omega * C_ij_integral_src(
    x_k, np.zeros_like(x_k), zeta_A)                       # (2,2,n_omega)
BmatA = inv_jacobian_fit[0]
C_pq_A = np.einsum('pi,ijm,qj->pqm', BmatA, C_ij, BmatA)
acc_mat = np.zeros((2, 2))
for p in range(2):
    for q in range(2):
        acc_mat[p, q] = acc_var_from_C(C_pq_A[p, q])
S1 = np.max(np.real(np.linalg.eigvals(acc_mat)))
print('\nS1  (Eq. C, image A, lens slab):        %.3e muas^2/yr^4 -> SNR = %.1f'
      % (S1/muasyy**2, S1/sigma2_acc_noise))

# =============================================== scalar LOS machinery (C_tilde_1)
vec_a_los = np.logspace(np.log10(0.999), np.log10(1/(1 + pB.z_source)), 2000)
vec_z_los = 1/vec_a_los - 1
vec_D = cosmo.angular_diameter_distance(vec_z_los).value * Mpc * (1 + vec_z_los)
D_src = pB.d_source * (1 + pB.z_source)
D_L_com = pB.d_lens * (1 + pB.z_lens)
rho_m0 = cosmo.Om0 * rho_crit
iL = np.argmin(np.abs(vec_D - D_L_com))

kc_grid = np.logspace(2, np.log10(3e11), 700) * Mpc**-1     # top-hat decomposition grid

def I_general(omega, mu_t, D_arr, Delta2_fun):
    """Angular kernel for an arbitrary spectrum, as a superposition of WHITE shells.

    The notebook kernel I_lm(k_c, Delta_c) is the response to a white spectrum
    P = P0 theta(k_c - k) with P0 = 6 pi^2 Delta_c / k_c^3, since
    arccos x + x sqrt(1-x^2) = int_{cos phi > x} cos^2 phi dphi.  Any decreasing
    P(k) is a positive superposition of such shells: I = sum_i (dP_i/2pi) g(x_i).
    Delta2_fun is the dimensionless spectrum; P(k) = 2 pi^2 Delta^2(k)/k^3."""
    P = 2*np.pi**2 * Delta2_fun(kc_grid) / kc_grid**3
    w = P[:-1] - P[1:]                                      # white-shell amplitudes (>= 0)
    kmid = np.sqrt(kc_grid[:-1] * kc_grid[1:])
    x = (omega / (mu_t * D_arr))[:, None] / kmid[None, :]
    g = np.where(x <= 1, np.arccos(np.clip(x, -1, 1)) + x*np.sqrt(np.clip(1 - x**2, 0, 1)), 0.0)
    return np.sum(w[None, :] / (2*np.pi) * g, axis=1)

def C_los(omega, Bt, mu_t, Delta2_fun):
    # comoving Poisson factor: rho_bar(z) a^2 = rho_m,0 (1+z) = rho_m,0 / a
    Ig = I_general(omega, mu_t, vec_D, Delta2_fun)
    integrand = (1 - vec_D/D_src)**2 * Ig * (4*np.pi*G_N*rho_m0/vec_a_los)**2
    return Bt**2 * 4/omega * np.trapezoid(integrand, vec_D)

def C_slab(omega, Bt, mu_t, Delta2_fun, Sigma_com, f_norm):
    """Same integrand collapsed onto the lens plane, carrying comoving column Sigma_com
       of clumps whose cosmic-mean version (fraction f_norm of all matter) is Delta2_fun."""
    Ig = I_general(omega, mu_t, np.array([vec_D[iL]]), Delta2_fun)[0]
    kern = (1 - vec_D[iL]/D_src)**2 * Ig * (4*np.pi*G_N*rho_m0/vec_a_los[iL])**2
    return Bt**2 * 4/omega * kern * Sigma_com / (f_norm * rho_m0)

# clump spectrum (comoving; clump physical size r_L mapped at the lens redshift)
f_clump = 0.5
r_L_com = r_L * (1 + pB.z_lens)
def Delta2_clump(k):
    return k**3 * f_clump * M_L * np.exp(-(k*r_L_com)**2) / (2*np.pi**2 * rho_m0)

# ------------------------------------------------------------------- S1' and S2
Sigma_com = a_L**2 * kappa_L_A * Sig_cr
vec_C_slab = np.array([C_slab(w, B_A, mu_tilde_A, Delta2_clump, Sigma_com, f_clump)
                       for w in vec_omega])
S1p = acc_var_from_C(vec_C_slab)
print("S1' (C_tilde_1, slab at D_L):           %.3e muas^2/yr^4 -> SNR = %.1f   [S1'/S1 = %.2f]"
      % (S1p/muasyy**2, S1p/sigma2_acc_noise, S1p/S1))

vec_C_cosmo = np.array([C_los(w, B_A, mu_tilde_A, Delta2_clump) for w in vec_omega])
S2 = acc_var_from_C(vec_C_cosmo)
print('S2  (same clumps, cosmic mean, f=0.5):  %.3e muas^2/yr^4 -> SNR = %.2e'
      % (S2/muasyy**2, S2/sigma2_acc_noise))

# equivalent column of the cosmic line of sight (same kernel, same clumps)
Sigma_eff_com = Sigma_com * S2 / S1p
print('    -> effective LOS column = %.3g Msun/pc^2 (comoving; lens slab: %.4g)'
      % (Sigma_eff_com/(M_Solar/pc**2), Sigma_com/(M_Solar/pc**2)))
print('    F_bias = S1\'/S2 = %.1f' % (S1p/S2))

# ------------------------------------------------- S3: pure-CDM one-halo spectrum
# (Sheth-Tormen x NFW, no free-streaming cutoff -- same construction as the figure)
data = np.loadtxt("Delta2_extended_output.csv", delimiter=",", skiprows=1)
kk_ext, k_h_ext, D2_lin_ext, D2_nl_ext, D2_nl_30 = data.T
rho_crit_0 = rho_m0 / cosmo.Om0
delta_c = 1.686
_int_lnD2lin = interp1d(np.log(kk_ext), np.log(D2_lin_ext), bounds_error=False,
                        fill_value=(np.log(D2_lin_ext[0]), np.log(D2_lin_ext[-1])))
def Delta2_lin(k):
    return np.exp(_int_lnD2lin(np.log(k * Mpc)))
def _W_tophat(x):
    x = np.asarray(x, float); w = np.ones_like(x); g = x > 1e-3; xx = x[g]
    w[g] = 3*(np.sin(xx) - xx*np.cos(xx))/xx**3
    return w
def sigma_of_M(M):
    R = (3*M/(4*np.pi*rho_m0))**(1/3)
    f = lambda lk: Delta2_lin(np.exp(lk)*Mpc**-1) * _W_tophat(np.exp(lk)*Mpc**-1*R)**2
    return np.sqrt(quad(f, np.log(1e-4), np.log(1e11), limit=400)[0])
vec_M_sig = np.logspace(-14, 16, 120) * M_Solar
_int_lnsig = interp1d(np.log(vec_M_sig),
                      np.log([sigma_of_M(M) for M in vec_M_sig]), fill_value='extrapolate')
def sigma_M(M): return np.exp(_int_lnsig(np.log(M)))
def dlnsig(M):
    h = 1e-3
    return (np.log(sigma_M(M*np.exp(h))) - np.log(sigma_M(M*np.exp(-h))))/(2*h)
def dn_dlnM(M):
    nu = delta_c/sigma_M(M); A_ST, a_ST, p_ST = 0.3222, 0.707, 0.3
    nu_p = np.sqrt(a_ST)*nu
    f_nu = A_ST*np.sqrt(2/np.pi)*nu_p*np.exp(-nu_p**2/2)*(1 + nu_p**(-2*p_ST))
    return (rho_m0/M)*f_nu*abs(dlnsig(M))
def conc_nocut(M):
    return np.clip(c_Einasto(M, h=cosmo.h), 2, 200)
def r_200f(M):
    return (3*M/(4*np.pi*200*rho_crit_0))**(1/3)
def u_NFW(k, M):
    c = conc_nocut(M); r_s = r_200f(M)/c; x = k*r_s
    mc = np.log(1+c) - c/(1+c)
    si, ci = sp.special.sici(x); si1, ci1 = sp.special.sici((1+c)*x)
    return (np.sin(x)*(si1-si) + np.cos(x)*(ci1-ci) - np.sin(c*x)/((1+c)*x))/mc

_lnM = np.log(np.logspace(-13, 16, 250)*M_Solar)
_kk = np.logspace(2, np.log10(3e11), 500) * Mpc**-1
_D2 = np.zeros_like(_kk)
dlnM = _lnM[1] - _lnM[0]
for lnM in _lnM:
    M = np.exp(lnM)
    _D2 += dn_dlnM(M)*M**2*u_NFW(_kk, M)**2*dlnM
_D2 *= _kk**3/(2*np.pi**2)/rho_m0**2
_int_D2CDM = interp1d(np.log(_kk), np.log(_D2), bounds_error=False, fill_value=-np.inf)
def Delta2_CDM(k):
    return np.exp(_int_D2CDM(np.log(k)))
kq = np.array([1e7, 3e7, 5e7, 1e8]) * Mpc**-1
print('\npure-CDM one-halo band (Wang, no cutoff): Delta^2 =',
      np.round(Delta2_CDM(kq), 0), 'at k =', kq*Mpc, '/Mpc')

vec_C_CDM = np.array([C_los(w, B_A, mu_tilde_A, Delta2_CDM) for w in vec_omega])
S3 = acc_var_from_C(vec_C_CDM)
print('S3  (LOS, pure-CDM one-halo spectrum):  %.3e muas^2/yr^4 -> SNR = %.2e'
      % (S3/muasyy**2, S3/sigma2_acc_noise))
print('    F_pop  = S2/S3  = %.1f' % (S2/S3))
print('    TOTAL  = S1\'/S3 = %.3g  (= F_bias x F_pop)' % (S1p/S3))

# ------------------------------------------------- interpretive sub-factors of F_pop
print('\nsub-factors of F_pop:')
for M in [1e-4*M_Solar, 1e-2*M_Solar, 1e0*M_Solar]:
    f_lnM = dn_dlnM(M)*M/rho_m0
    print('  mass fraction per e-fold at M = %.0e Msun: %.4f' % (M/M_Solar, f_lnM))
kpk = np.sqrt(1.5)/r_L_com
print('  Delta^2_clump peak = %.3g at k = %.3g /Mpc  vs  Delta^2_CDM there = %.3g'
      % (Delta2_clump(kpk), kpk*Mpc, Delta2_CDM(kpk)))
# NFW spread: an NFW halo whose scale radius matches the clump size
for c_test in [30., 60.]:
    x = 1.0
    mc = np.log(1+c_test) - c_test/(1+c_test)
    si, ci = sp.special.sici(x); si1, ci1 = sp.special.sici((1+c_test)*x)
    u1 = (np.sin(x)*(si1-si) + np.cos(x)*(ci1-ci) - np.sin(c_test*x)/((1+c_test)*x))/mc
    print('  |u_NFW(k r_s = 1)|^2 for c = %d: %.3g   (Gaussian clump: %.2f)'
          % (c_test, u1**2, np.exp(-1.5)))

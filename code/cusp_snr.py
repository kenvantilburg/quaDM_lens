"""Acceleration-channel SNR (instrumental noise only) of the PROMPT-CUSP population
bound to the lens, for B1422+231 (EPIC, 0.1 muas) and J1029+2623 (1 muas survey).

Same machinery as impact_numbers.py: cusps modeled as Gaussian-cutoff cusp halos
with M = M_cusp, effective size r_s ~ r_cusp/2 (half-mass matched; results are
insensitive since cusps are nearly point-like at the relevant impact parameters),
at a lens-plane convergence kappa_cusp = f_cusp * kappa_fit.
"""
import sys, os
CODE = "/Users/kenvt/Library/CloudStorage/Dropbox/projects/1 astrometric-microlensing-quasars/quaDM_lens/code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from sensitivity_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def smear_fac_acc(a):
    return np.heaviside(-a + 1e-2, 0) + np.heaviside(a - 1e-2, 0) * 14400 * (
        6*a*np.cos(a/2) + (-12 + a**2)*np.sin(a/2))**2 / a**10

t_int = 10 * year
N_obs = 300
list_omega_out = np.logspace(-6, 2, int(5e2)) * 2*np.pi/t_int
Delta_omega_out = list_omega_out[1:] - list_omega_out[:-1]
smear_acc = smear_fac_acc(list_omega_out[1:]*t_int)

def acc_var_population(kappa_l, inv_jacobian_fit, mu_tilde, zeta, d_lens_v,
                       d_source_v, d_lens_source_v, rho_s, r_s, theta_src_I=None):
    """max-over-pairs largest-eigenvalue differential acceleration variance
    for a population of identical (rho_s, r_s) lenses with column kappa_l."""
    n_im = len(kappa_l)
    if theta_src_I is None:
        theta_src_I = np.zeros(n_im)
    M = 4*np.pi*np.exp(0.5)*rho_s*r_s**3
    gamma = r_s / d_lens_v
    tE = theta_E(M, d_lens_v, d_source_v, d_lens_source_v)
    C_ij = np.zeros((n_im, 2, 2, len(list_omega_out)))
    for i in range(n_im):
        x_k = gamma/mu_tilde[i]*list_omega_out
        x_src = theta_src_I[i]/mu_tilde[i]*list_omega_out
        C_ij[i] = kappa_l[i]*tE**2/list_omega_out * C_ij_integral_src(x_k, x_src, zeta[i])
    C_pq = np.einsum('aqj,apjm->apqm', inv_jacobian_fit,
                     np.einsum('api,aijm->apjm', inv_jacobian_fit, C_ij))
    pairs = [(i, j) for i in range(n_im) for j in range(i+1, n_im)]
    best = 0.
    for (i, j) in pairs:
        dC = C_pq[i] + C_pq[j]
        acc = 2*np.sum(smear_acc*Delta_omega_out/(2*np.pi)*list_omega_out[1:]**4*dC[:, :, 1:], axis=-1)
        eig = np.max(np.real(np.linalg.eigvals(acc)))
        best = max(best, eig)
    return M, best

# fiducial cusp population
M_cusp = 1e-6 * M_Solar
r_cusp = 5e-3 * pc
f_cusp = 0.01 * 0.5      # ~1% of DM mass in cusps as formed, x f_surv ~ 0.5

# ============================================================ B1422 (EPIC)
print("="*72); print("B1422+231, EPIC (sigma=0.1 muas, tau=10 yr, N=300)"); print("="*72)
from params_B1422_231 import *
res = np.load('macro_lens_results.npz')
inv_jacobian_fit = res['inv_jacobian_fit']; kappa_fit = res['kappa_fit']
vec_mu = mu_rel(np.zeros(2)*km/second, v_lens_fid, np.zeros(2)*km/second,
                z_lens, z_source, d_lens, d_source, d_lens_source)
vec_mu_tilde = np.tensordot(inv_jacobian_fit, vec_mu, axes=1)
# sweep rate: bulk magnified by B, internal subhalo motion UNMAGNIFIED (kinematics fix)
mu_tilde = np.sqrt(np.linalg.norm(vec_mu_tilde, axis=1)**2 + mu_int_drift**2)
zeta = np.arctan2(vec_mu_tilde[:, 1], vec_mu_tilde[:, 0])
sigma2_acc_noise = 720 * (0.1*muas)**2 / (t_int**4 * N_obs)

# magnified source size per image (tangential stretch of theta_500)
eigvals_B = np.linalg.eigvals(inv_jacobian_fit)
B_tang = np.max(np.abs(eigvals_B), axis=1)
theta_src_I_B = B_tang * theta_source
print("kappa_fit:", np.round(kappa_fit, 3))
print("B_tang:", np.round(B_tang, 2), " theta_src_I [muas]:", np.round(theta_src_I_B/muas, 3))
print("cusp angular size gamma = %.2f muas" % (r_cusp/d_lens/muas))
print("mu_tilde [muas/yr]:", np.round(mu_tilde/muasy, 2))

for rs_eff, lab in [(0.5*r_cusp, "r_s = r_cusp/2 (half-mass match)"),
                    (r_cusp,     "r_s = r_cusp"),
                    (0.1*r_cusp, "r_s = r_cusp/10 (point-like limit)")]:
    rho_s = M_cusp / (4*np.pi*np.exp(0.5)*rs_eff**3)
    M, var = acc_var_population(f_cusp*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                                d_lens, d_source, d_lens_source, rho_s, rs_eff,
                                theta_src_I=theta_src_I_B)
    snr = var / sigma2_acc_noise
    print("%-36s SNR = %.3f  (f_cusp=%.3f, M=%.1e Msun)  -> SNR=1 needs f_cusp x %.1f, or sigma = %.3f muas"
          % (lab, snr, f_cusp, M/M_Solar, 1/snr, 0.1*np.sqrt(snr)))

# no-source-size check
rho_s = M_cusp / (4*np.pi*np.exp(0.5)*(0.5*r_cusp)**3)
_, var0 = acc_var_population(f_cusp*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                             d_lens, d_source, d_lens_source, rho_s, 0.5*r_cusp)
print("(point-source check, no finite-src suppression: SNR = %.3f)" % (var0/sigma2_acc_noise))

# ===================================================== J1029 (1 muas survey)
print(); print("="*72); print("SDSS J1029+2623, optical survey (sigma=1 muas)"); print("="*72)
import params_J1029_2623 as pj
vec_mu_J = mu_rel(np.zeros(2)*km/second, pj.v_lens_fid, np.zeros(2)*km/second,
                  pj.z_lens, pj.z_source, pj.d_lens, pj.d_source, pj.d_lens_source)
vec_mu_tilde_J = np.tensordot(pj.inv_jacobian_fit, vec_mu_J, axes=1)
# sweep rate: bulk magnified by B, internal cluster-orbital motion UNMAGNIFIED (kinematics fix)
mu_tilde_J = np.sqrt(np.linalg.norm(vec_mu_tilde_J, axis=1)**2 + pj.mu_int_drift**2)
zeta_J = np.arctan2(vec_mu_tilde_J[:, 1], vec_mu_tilde_J[:, 0])
sigma2_acc_noise_J = 720 * (1.0*muas)**2 / (t_int**4 * N_obs)
print("cusp angular size gamma = %.2f muas" % (r_cusp/pj.d_lens/muas))
print("theta_src_I [muas]:", np.round(pj.theta_source_I/muas, 3))
print("mu_tilde [muas/yr]:", np.round(mu_tilde_J/muasy, 2))

for rs_eff, lab in [(0.5*r_cusp, "r_s = r_cusp/2 (half-mass match)"),
                    (r_cusp,     "r_s = r_cusp")]:
    rho_s = M_cusp / (4*np.pi*np.exp(0.5)*rs_eff**3)
    M, var = acc_var_population(f_cusp*pj.kappa_fit, pj.inv_jacobian_fit, mu_tilde_J, zeta_J,
                                pj.d_lens, pj.d_source, pj.d_lens_source, rho_s, rs_eff,
                                theta_src_I=pj.theta_source_I)
    snr = var / sigma2_acc_noise_J
    print("%-36s SNR = %.2e  -> SNR=1 needs f_cusp x %.0f, or sigma = %.3f muas"
          % (lab, snr, 1/snr, 1.0*np.sqrt(snr)))

# =============== cross-check: reproduce along-CDM subhalo SNR at M=1e-6 Msun
print(); print("cross-check B1422: f_sub=0.5 CDM halo at M=1e-6 Msun (Einasto-pred rho_s, r_s)")
list_M_200 = 10**np.arange(-10., 8, 0.1) * M_Solar
r_s_p = r_s_Einasto_200(list_M_200); rho_s_p = 20*rho_s_Einasto_200(list_M_200)
M_s_p = M_enc_Einasto(r_s_p, rho_s_p, r_s_p)
i6 = np.argmin(np.abs(np.log10(M_s_p/M_Solar) + 6))
rho_eff = M_s_p[i6]/(4*np.pi*np.exp(0.5)*r_s_p[i6]**3)
M, var = acc_var_population(0.5*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                            d_lens, d_source, d_lens_source, rho_eff, r_s_p[i6],
                            theta_src_I=theta_src_I_B)
print("M_s = %.2e Msun, r_s = %.2e pc -> SNR = %.1f"
      % (M_s_p[i6]/M_Solar, r_s_p[i6]/pc, var/sigma2_acc_noise))

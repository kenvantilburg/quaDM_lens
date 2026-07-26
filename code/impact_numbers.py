"""Headline numbers for the q-lensing manuscript impact revision (2026-07).

Computes, with the same machinery as sensitivity.ipynb / sensitivity_J1029.py /
matter_power.ipynb:
  1. B1422+231: acceleration- and stochastic-channel SNR evaluated ALONG the
     fiducial CDM rho_s(M) prediction curve (the black dashed line of the SNR
     figure) -> the paper's headline "SNR at the standard CDM population".
     Quoted both instrument-limited (certified-clean epochs) and with the
     post-subtraction stellar microlensing residual noise (campaign of the
     companion paper rescaled to the survey parameters assumed here:
     theta_fit = 10 muas x (N/33)^(1/4), worst-case unfitted star +
     fitted-star residual at image A, adopted per image and doubled per pair).
  2. SDSS J1029+2623: the same for the sigma = 1 muas optical survey.
  3. Line-of-sight-only (mean cosmological density, f_sub-independent)
     acceleration SNR for both fiducials, for a scale-invariant Delta^2 up to a
     cutoff k_c -- evaluated at the prompt-cusp plateau (Delta^2 = 4e5) and the
     halo-model level (Delta^2 = 4e3), feeding the "robustness" paragraph.
  4. J1029 nearest-unfitted-star worst-case acceleration (cross-check of the
     Sec. III D caveat).

Run:  python impact_numbers.py
"""
import numpy as np
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from sensitivity_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ---------------------------------------------------------------- smearing
def smear_fac_acc(a):
    return np.heaviside(-a + 1e-2, 0) + np.heaviside(a - 1e-2, 0) * 14400 * (
        6*a*np.cos(a/2) + (-12 + a**2)*np.sin(a/2))**2 / a**10

t_int = 10 * year
N_obs = 300
list_omega_out = np.logspace(-6, 2, int(5e2)) * 2*np.pi/t_int
Delta_omega_out = list_omega_out[1:] - list_omega_out[:-1]
smear_acc = smear_fac_acc(list_omega_out[1:]*t_int)

def acc_pairs(kappa_l, inv_jacobian_fit, mu_tilde, zeta, d_lens, d_source,
              d_lens_source, rho_s_grid, r_s_grid, theta_src_I=None):
    """largest-eigval differential acceleration covariance on a (rho_s, r_s) grid."""
    n_im = len(kappa_l)
    if theta_src_I is None:
        theta_src_I = np.zeros(n_im)
    arr_r, arr_rho = np.meshgrid(r_s_grid, rho_s_grid)
    arr_M = 4*np.pi*np.exp(0.5)*arr_rho*arr_r**3
    arr_gamma = arr_r / d_lens
    arr_tE = theta_E(arr_M, d_lens, d_source, d_lens_source)
    arr_C_ij = np.zeros((n_im, len(rho_s_grid), len(r_s_grid), 2, 2, len(list_omega_out)))
    for i in range(n_im):
        x_k = arr_gamma[:, :, None]/mu_tilde[i]*list_omega_out
        x_src = theta_src_I[i]/mu_tilde[i]*list_omega_out
        arr_C_ij[i] = kappa_l[i]*arr_tE[:, :, None, None, None]**2/list_omega_out \
            * np.transpose(C_ij_integral_src(x_k, x_src*np.ones_like(x_k), zeta[i]), (2, 3, 0, 1, 4))
    arr_C_pq = np.einsum('aqj,abcpjm->abcpqm', inv_jacobian_fit,
                         np.einsum('api,abcijm->abcpjm', inv_jacobian_fit, arr_C_ij))
    pairs = [(i, j) for i in range(n_im) for j in range(i+1, n_im)]
    acc = np.zeros((len(pairs), len(rho_s_grid), len(r_s_grid), 2, 2))
    for p, (i, j) in enumerate(pairs):
        dC = arr_C_pq[i] + arr_C_pq[j]
        acc[p] = 2*np.sum(smear_acc*Delta_omega_out/(2*np.pi)*list_omega_out[1:]**4*dC[:, :, :, :, 1:], axis=-1)
    eig = np.real(np.max(np.linalg.eigvals(np.transpose(acc, (1, 2, 0, 3, 4))), axis=-1))
    return arr_M, eig, pairs   # eig: (rho, r, pair)

# CDM prediction curve (same construction as sensitivity.ipynb cell 26)
list_M_200 = 10**np.arange(-10., 8, 0.1) * M_Solar
list_r_s_pred = r_s_Einasto_200(list_M_200)
list_rho_s_pred = 20*rho_s_Einasto_200(list_M_200)
list_M_s_pred = M_enc_Einasto(list_r_s_pred, list_rho_s_pred, list_r_s_pred)

def snr_along_cdm(arr_M, arr_snr, rho_s_grid):
    """Interpolate log10(SNR) grid onto the CDM (M_s_pred, rho_s_pred) curve."""
    from scipy.interpolate import RegularGridInterpolator
    logM = np.log10(arr_M[0]/M_Solar)   # r_s axis at fixed rho: M varies; grid regular in (rho, r_s)
    # interpolate in (log rho_s, log M) using the fact M = 4 pi e^.5 rho r^3
    # -> at fixed rho row, M column is monotonic; build interpolator on (logrho, logM_row0) instead:
    # simpler: for each prediction point, find nearest rho row, then interp in M.
    out = []
    for Mp, rp in zip(list_M_s_pred, list_rho_s_pred):
        if not (rho_s_grid[0] <= rp <= rho_s_grid[-1]):
            out.append(np.nan); continue
        irho = np.argmin(np.abs(np.log10(rho_s_grid/rp)))
        Mrow = arr_M[irho]
        if not (Mrow[0] <= Mp <= Mrow[-1]):
            out.append(np.nan); continue
        out.append(np.interp(np.log10(Mp/M_Solar), np.log10(Mrow/M_Solar), arr_snr[irho]))
    return np.asarray(out)

rho_s_grid = 10**np.arange(-2., 5.01, 0.04) * M_Solar/pc**3
r_s_grid = 10**np.arange(-4.5, 1.5, 0.04) * pc

# ============================================================ B1422 (EPIC)
print("="*72); print("B1422+231, EPIC fiducial (sigma=0.1 muas, tau=10yr, N=300)"); print("="*72)
from params_B1422_231 import *
res = np.load('macro_lens_results.npz')
inv_jacobian_fit = res['inv_jacobian_fit']; kappa_fit = res['kappa_fit']
v_lens = v_lens_fid    # BULK fiducial, |v_L| ~ 865 km/s (see params_B1422_231)
vec_mu = mu_rel(np.zeros(2)*km/second, v_lens, np.zeros(2)*km/second,
                z_lens, z_source, d_lens, d_source, d_lens_source)
vec_mu_tilde = np.tensordot(inv_jacobian_fit, vec_mu, axes=1)
# sweep rate: bulk term magnified by B, internal subhalo motion UNMAGNIFIED (kinematics fix)
mu_tilde = np.sqrt(np.linalg.norm(vec_mu_tilde, axis=1)**2 + mu_int_drift**2)
zeta = np.arctan2(vec_mu_tilde[:, 1], vec_mu_tilde[:, 0])
kappa_l = 0.5 * kappa_fit
sigma_dth = 0.1 * muas
sigma2_acc_noise = 720 * sigma_dth**2 / (t_int**4 * N_obs)

arr_M_B, eig_B, pairs_B = acc_pairs(kappa_l, inv_jacobian_fit, mu_tilde, zeta,
                                    d_lens, d_source, d_lens_source, rho_s_grid, r_s_grid)
acc_B = np.max(eig_B, axis=-1)
snr_B = acc_B / sigma2_acc_noise
snr_cdm_B = snr_along_cdm(arr_M_B, snr_B, rho_s_grid)
m = np.isfinite(snr_cdm_B)
print("acceleration SNR along CDM rho_s(M), instrument-limited (certified-clean):")
print("  max = %.1f at M = %.2g Msun"
      % (np.nanmax(snr_cdm_B), list_M_s_pred[m][np.nanargmax(snr_cdm_B[m])]/M_Solar))
det = list_M_s_pred[m][snr_cdm_B[m] > 1]
if det.size:
    print("  SNR > 1 for M in [%.1e, %.1e] Msun" % (det[0]/M_Solar, det[-1]/M_Solar))
print("mu_tilde [muas/yr]:", np.round(mu_tilde/muasy, 2))
print("instrument floor = %.2e muas^2/yr^4" % (sigma2_acc_noise/muasyy**2))

# ---- post-subtraction stellar microlensing residual (Sec. III C): the campaign of the
# companion paper rescaled to the survey parameters assumed here (theta_fit ~ theta_E*
# SNR_1^(1/2) n_obs^(1/4); unfitted worst case ~ theta_fit^-6; fitted residual ~
# sigma_c^2/(n_obs tau^4)). Only image A was studied; its budget is adopted as the
# per-image estimate and doubled for the differential pair.
theta_fit = 10*muas * (N_obs/33)**0.25
s2_unf_A = np.abs(np.linalg.det(inv_jacobian_fit[0])) * 4*theta_E_star**4 * mu_tilde[0]**4 / theta_fit**6
s2_fit_A = 1.96e-4*muasyy**2 * (33/N_obs) * ((8*year)/t_int)**4
s2_star_A = s2_unf_A + s2_fit_A
sigma2_acc_eff = sigma2_acc_noise + 2*s2_star_A
print("theta_fit = %.3g muas; image-A stellar residual = %.3g (unf %.2g + fit %.2g) muas^2/yr^4"
      % (theta_fit/muas, s2_star_A/muasyy**2, s2_unf_A/muasyy**2, s2_fit_A/muasyy**2))
print("effective pair noise = %.3g muas^2/yr^4 (%.1f x instr)"
      % (sigma2_acc_eff/muasyy**2, sigma2_acc_eff/sigma2_acc_noise))
snr_B_star = np.max(eig_B, axis=-1) / sigma2_acc_eff
snr_cdm_B_star = snr_along_cdm(arr_M_B, snr_B_star, rho_s_grid)
ms = np.isfinite(snr_cdm_B_star)
print("acceleration SNR along CDM rho_s(M), WITH stellar residual noise:")
print("  max = %.2f at M = %.2g Msun"
      % (np.nanmax(snr_cdm_B_star), list_M_s_pred[ms][np.nanargmax(snr_cdm_B_star[ms])]/M_Solar))
dets = list_M_s_pred[ms][snr_cdm_B_star[ms] > 1]
if dets.size:
    print("  SNR > 1 for M in [%.1e, %.1e] Msun" % (dets[0]/M_Solar, dets[-1]/M_Solar))
else:
    print("  SNR < 1 everywhere along the CDM curve")

# ===================================================== J1029 (1 muas survey)
print(); print("="*72); print("SDSS J1029+2623, optical survey (sigma=1 muas)"); print("="*72)
import importlib, params_J1029_2623 as pj
importlib.reload(pj)
v_lens_J = pj.v_lens_fid   # EFFECTIVE fiducial, |v_L| ~ 552 km/s: bulk 474 (+) moving-clump
                           # member-halo term sqrt(2)*f_gal*sigma_v (see params_J1029_2623)
vec_mu_J = mu_rel(np.zeros(2)*km/second, v_lens_J, np.zeros(2)*km/second,
                  pj.z_lens, pj.z_source, pj.d_lens, pj.d_source, pj.d_lens_source)
vec_mu_tilde_J = np.tensordot(pj.inv_jacobian_fit, vec_mu_J, axes=1)
# sweep rate: bulk term magnified by B, internal cluster-orbital motion UNMAGNIFIED (kinematics fix)
mu_tilde_J = np.sqrt(np.linalg.norm(vec_mu_tilde_J, axis=1)**2 + pj.mu_int_drift**2)
zeta_J = np.arctan2(vec_mu_tilde_J[:, 1], vec_mu_tilde_J[:, 0])
kappa_l_J = 0.5 * pj.kappa_fit
sigma_dth_J = 1.0 * muas
sigma2_acc_noise_J = 720 * sigma_dth_J**2 / (t_int**4 * N_obs)

arr_M_J, eig_J, _ = acc_pairs(kappa_l_J, pj.inv_jacobian_fit, mu_tilde_J, zeta_J,
                              pj.d_lens, pj.d_source, pj.d_lens_source,
                              rho_s_grid, r_s_grid, theta_src_I=pj.theta_source_I)
acc_J = np.max(eig_J, axis=-1)
snr_J = acc_J / sigma2_acc_noise_J
snr_cdm_J = snr_along_cdm(arr_M_J, snr_J, rho_s_grid)
mJ = np.isfinite(snr_cdm_J)
print("acceleration SNR along CDM rho_s(M): max = %.1f at M = %.2g Msun"
      % (np.nanmax(snr_cdm_J), list_M_s_pred[mJ][np.nanargmax(snr_cdm_J[mJ])]/M_Solar))
detJ = list_M_s_pred[mJ][snr_cdm_J[mJ] > 1]
if detJ.size:
    print("SNR > 1 for M in [%.1e, %.1e] Msun" % (detJ[0]/M_Solar, detJ[-1]/M_Solar))
print("mu_tilde [muas/yr]:", np.round(mu_tilde_J/muasy, 2))
print("instrument floor = %.2e muas^2/yr^4" % (sigma2_acc_noise_J/muasyy**2))

# =================================== LOS-only (mean cosmological density)
print(); print("="*72); print("Line-of-sight-only acceleration SNR (mean cosmological density)"); print("="*72)
# top-hat scale-invariant Delta^2 up to k_c along the LOS (matter_power.ipynb machinery)
def C_pq_los(omega, Bmax, mu_t, k_c, Delta_c, z_source_v, d_source_v):
    vec_a = np.logspace(np.log10(0.999), np.log10(1/(1+z_source_v)), 2000)
    vec_z = 1/vec_a - 1
    vec_D = cosmo.angular_diameter_distance(vec_z).value * Mpc * (1+vec_z)  # comoving
    D_src = d_source_v * (1+z_source_v)
    rho_m0 = cosmo.Om0 * rho_crit
    k_tilde = omega / (mu_t * vec_D)
    x = k_tilde / k_c
    Ilm = np.zeros_like(x)
    msk = x <= 1
    Ilm[msk] = 3*np.pi*Delta_c/k_c**3 * (np.arccos(np.clip(x[msk], -1, 1)) + x[msk]*np.sqrt(1-x[msk]**2))
    integrand = (1 - vec_D/D_src)**2 * Ilm * (4*np.pi*G_N*rho_m0/vec_a)**2  # rho_bar(z) a^2 = rho_m,0 (1+z)
    return Bmax**2 * 4/omega * np.trapezoid(integrand, vec_D)

def sigma2_acc_los(Bmax, mu_t, k_c, Delta_c, z_s, d_s):
    vec_o = 2*np.pi/t_int * np.logspace(-8, 2, 800)
    vec_C = np.asarray([C_pq_los(o, Bmax, mu_t, k_c, Delta_c, z_s, d_s) for o in vec_o])
    return np.trapezoid(vec_C * vec_o**4 * smear_fac_acc(vec_o*t_int)/(2*np.pi), vec_o)

for tag, Bmax, mu_t, z_s, d_s, s2n in [
        ("B1422/EPIC (B=7.5, mu=%.2f muas/yr, sigma=0.1)" % (mu_tilde[0]/muasy),
         7.5, mu_tilde[0], z_source, d_source, sigma2_acc_noise),
        ("J1029/survey (B=22.6, mu=%.2f muas/yr, sigma=1)" % (mu_tilde_J[2]/muasy),
         22.6, mu_tilde_J[2], pj.z_source, pj.d_source, sigma2_acc_noise_J)]:
    for lab, kc, D2 in [("prompt-cusp plateau", 1e9/Mpc, 4e5),
                        ("halo model",          1e8/Mpc, 4e3)]:
        s2 = sigma2_acc_los(Bmax, mu_t, kc, D2, z_s, d_s)
        print("%s | %s (Delta^2=%.0e, k_c=%.0e/Mpc): acc var = %.2e muas^2/yr^4, SNR = %.2g"
              % (tag, lab, D2, kc*Mpc, s2/muasyy**2, s2/s2n))

# ============================= J1029 nearest-star worst case (cross-check)
print(); print("="*72); print("J1029 nearest-unfitted-star acceleration (worst case)"); print("="*72)
tE = pj.theta_E_star
n_star = pj.kappa_star * pj.Sigma_crit_val / pj.M_star  # surface number density (mass/area /M)
for P, lab in [(0.5, "median"), (0.25, "75% clean")]:
    # nearest-star distance with P(no star closer): N(theta)= -ln P
    N_t = -np.log(P)
    theta_near = np.sqrt(N_t / (np.pi * n_star)) / pj.d_lens
    Bmax = 22.6
    acc = 2*Bmax*tE**2*mu_tilde_J[2]**2/theta_near**3
    print("%s nearest star at %.1f muas -> |acc|^2 <= %.2e muas^2/yr^4"
          % (lab, theta_near/muas, acc**2/muasyy**2))

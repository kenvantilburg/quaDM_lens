"""Headline SNRs quoted in the text and in Tab. tab:bottomline of the paper.

Same machinery as sensitivity.ipynb and matter_power.ipynb, collected in one script:
  1. B1422+231: acceleration-channel variance SNR evaluated ALONG the fiducial CDM
     rho_s(M_L) relation (the black dashed line of Fig. SNR) -> the paper's headline
     "SNR at the standard CDM population". Quoted both instrument-limited (certifiably
     star-free epochs) and with the worst-case post-subtraction stellar microlensing
     residual of Sec. III C (the campaign of the companion paper rescaled to the survey
     assumed here: theta_fit = 10 muas x (N/33)^(1/4), unfitted-star bound plus
     fitted-star residual at image A, adopted per image and doubled for the pair).
  2. SDSS J1029+2623: the same for the sigma_delta_theta = 1 muas optical survey.
  3. Line-of-sight-only acceleration SNR (mean cosmological density, hence independent
     of f_sub) for both systems, for a white spectrum truncated at k carrying Delta^2
     per e-fold -- evaluated at the prompt-cusp plateau (Delta^2 = 4e5) and the
     halo-model level (Delta^2 = 4e3). Cf. los_full_spectrum_snr.py, which folds the
     full predicted spectra through the same kernel.
  4. J1029 nearest-unfitted-star worst-case acceleration (the Sec. III D caveat).

Notation follows the paper (M_L, r_L, rho_s, kappa_L, gamma_L, theta_E,L, tau, N_obs,
sigma_delta_theta, F_2 -> smear_F2). SNRs are variance ratios S/N throughout.

Run:  python impact_numbers.py
"""
import numpy as np
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from sensitivity_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ------------------------------------------- acceleration-estimator smearing factor F_2
def smear_F2(a):
    """F_2(omega tau) of Eq. (F_2_smearing); -> 1 as a -> 0, ~ a^-6 for a >> 1."""
    return np.heaviside(-a + 1e-2, 0) + np.heaviside(a - 1e-2, 0) * 14400 * (
        6*a*np.cos(a/2) + (-12 + a**2)*np.sin(a/2))**2 / a**10

tau = 10 * year
N_obs = 300
# Frequency grid for the continuum omega-integrals of Eq. (observable_2). It deliberately
# extends far BELOW the DFT band edge 2 pi / tau, where the omega^4 F_2 kernel peaks.
vec_omega_wide = np.logspace(-6, 2, int(5e2)) * 2*np.pi/tau
d_omega_wide = vec_omega_wide[1:] - vec_omega_wide[:-1]
vec_F2 = smear_F2(vec_omega_wide[1:]*tau)

def acc_pairs(kappa_L, inv_jacobian_fit, mu_tilde, zeta, d_lens, d_source,
              d_lens_source, rho_s_grid, r_L_grid, theta_src_I=None):
    """Largest-eigenvalue differential acceleration covariance, Eq. (observable_2).

    Evaluated on the (rho_s, r_L) grid of Fig. SNR for every image pair: build C~^I_pq
    from Eq. (C), sum it over the two images of each pair, then take the smeared
    omega^4 moment. The factor 2 counts negative frequencies (C~ is two-sided).
    Returns (M_L grid, largest eigenvalue indexed [rho_s, r_L, pair], pair list).
    """
    n_im = len(kappa_L)
    if theta_src_I is None:
        theta_src_I = np.zeros(n_im)
    arr_r, arr_rho = np.meshgrid(r_L_grid, rho_s_grid)
    arr_M = 4*np.pi*np.exp(0.5)*arr_rho*arr_r**3
    arr_gamma = arr_r / d_lens
    arr_tE = theta_E(arr_M, d_lens, d_source, d_lens_source)
    arr_C_ij = np.zeros((n_im, len(rho_s_grid), len(r_L_grid), 2, 2, len(vec_omega_wide)))
    for i in range(n_im):
        x_k = arr_gamma[:, :, None]/mu_tilde[i]*vec_omega_wide
        x_src = theta_src_I[i]/mu_tilde[i]*vec_omega_wide
        arr_C_ij[i] = kappa_L[i]*arr_tE[:, :, None, None, None]**2/vec_omega_wide \
            * np.transpose(C_ij_integral_src(x_k, x_src*np.ones_like(x_k), zeta[i]), (2, 3, 0, 1, 4))
    arr_C_pq = np.einsum('aqj,abcpjm->abcpqm', inv_jacobian_fit,
                         np.einsum('api,abcijm->abcpjm', inv_jacobian_fit, arr_C_ij))
    pairs = [(i, j) for i in range(n_im) for j in range(i+1, n_im)]
    acc = np.zeros((len(pairs), len(rho_s_grid), len(r_L_grid), 2, 2))
    for p, (i, j) in enumerate(pairs):
        dC = arr_C_pq[i] + arr_C_pq[j]
        acc[p] = 2*np.sum(vec_F2*d_omega_wide/(2*np.pi)*vec_omega_wide[1:]**4*dC[:, :, :, :, 1:], axis=-1)
    eig = np.real(np.max(np.linalg.eigvals(np.transpose(acc, (1, 2, 0, 3, 4))), axis=-1))
    return arr_M, eig, pairs   # eig: (rho, r, pair)

# Fiducial LambdaCDM rho_s(M_s) relation -- the gray band of Fig. SNR, built exactly as in
# sensitivity.ipynb: NFW subhalos carrying the median concentration c_200(M_200, x_sub) of
# Eq. (7) of Moline+2017, with r_s = r_200/c_200, rho_s = rho(r_s) (the convention of
# Sec. III B), and the mass taken as that enclosed within r_s. The band spans
# x_sub = R_sub/R_200^host = 1 (field halos) down to 0.01; each system is additionally
# evaluated at its own x_sub -- 0.03 at B1422+231's images, 0.13 at J1029's (see the
# params files). This replaces the Wang+2020 field-halo relation with its ad hoc factor-20
# renormalization of rho_s; the Moline+ relation lands on the rho_s ~ 0.1-1 M_Solar/pc^3
# of Sec. IV A unaided.
list_M_200 = 10**np.arange(-8., 6.01, 0.05) * M_Solar

def cdm_curve(x_sub):
    """(M_s, rho_s) along the fiducial CDM relation at host distance x_sub."""
    M_s, rho_s, _ = scale_params_NFW_Moline(list_M_200, x_sub)
    return M_s, rho_s

def snr_along_cdm(arr_M, arr_snr, rho_s_grid, curve):
    """Interpolate the SNR grid onto a CDM (M_s, rho_s) curve.

    The grid is regular in (rho_s, r_L), so M_L varies along each row. For every point of
    the prediction curve we snap to the nearest rho_s row, then interpolate in log M_L
    along it; points off the grid come back as NaN.
    """
    list_M_s_pred, list_rho_s_pred = curve
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

def report_cdm(arr_M, arr_snr, rho_s_grid, x_sub_list):
    """Print the peak SNR and the SNR > 1 mass range along the CDM band."""
    for x_sub in x_sub_list:
        Ms, rs = cdm_curve(x_sub)
        snr = snr_along_cdm(arr_M, arr_snr, rho_s_grid, (Ms, rs))
        m = np.isfinite(snr)
        det = Ms[m][snr[m] > 1]
        rng = ("SNR>1 for M_s in [%.1e, %.1e] Msun" % (det[0]/M_Solar, det[-1]/M_Solar)
               if det.size else "SNR < 1 everywhere")
        print("  x_sub = %5.3g:  max SNR = %7.2f at M_s = %.2g Msun;  %s"
              % (x_sub, np.nanmax(snr), Ms[m][np.nanargmax(snr[m])]/M_Solar, rng))

rho_s_grid = 10**np.arange(-2., 5.01, 0.04) * M_Solar/pc**3
r_L_grid = 10**np.arange(-4.5, 1.5, 0.04) * pc

# ============================================================ B1422 (EPIC)
print("="*72); print("B1422+231, EPIC fiducial (sigma=0.1 muas, tau=10yr, N=300)"); print("="*72)
from params_B1422_231 import *
res = np.load('macro_lens_results.npz')
inv_jacobian_fit = res['inv_jacobian_fit']; kappa_fit = res['kappa_fit']
v_lens = v_lens_fid    # bulk fiducial, |v_L| ~ 865 km/s (see params_B1422_231)
vec_mu = mu_rel(np.zeros(2)*km/second, v_lens, np.zeros(2)*km/second,
                z_lens, z_source, d_lens, d_source, d_lens_source)
vec_mu_tilde = np.tensordot(inv_jacobian_fit, vec_mu, axes=1)
# sweep rate: bulk term magnified by B^I, the subhalos' own motion mu_L unmagnified
mu_tilde = np.sqrt(np.linalg.norm(vec_mu_tilde, axis=1)**2 + mu_L_int**2)
zeta = np.arctan2(vec_mu_tilde[:, 1], vec_mu_tilde[:, 0])
kappa_L = 0.5 * kappa_fit   # f_sub = 0.5 of the convergence at each image
sigma_delta_theta = 0.1 * muas
sigma2_acc_noise = 720 * sigma_delta_theta**2 / (tau**4 * N_obs)

arr_M_B, eig_B, pairs_B = acc_pairs(kappa_L, inv_jacobian_fit, mu_tilde, zeta,
                                    d_lens, d_source, d_lens_source, rho_s_grid, r_L_grid)
acc_B = np.max(eig_B, axis=-1)
snr_B = acc_B / sigma2_acc_noise
print("acceleration SNR along CDM rho_s(M_s), instrument-limited (certified-clean):")
report_cdm(arr_M_B, snr_B, rho_s_grid, [1., 0.1, x_sub_fid, 0.01])
print("mu_tilde [muas/yr]:", np.round(mu_tilde/muasy, 2))
print("instrument floor = %.2e muas^2/yr^4" % (sigma2_acc_noise/muasyy**2))

# ---- worst-case post-subtraction stellar microlensing residual (Sec. III C): the
# 33-epoch, 8-yr campaign of the companion paper rescaled to the survey assumed here via
# Eq. (theta_fit), theta_fit ~ theta_E,* SNR_1^(1/2) N^(1/4); the unfitted-star bound then
# falls as theta_fit^-6 and the fitted-star residual as sigma^2/(N tau^4). Only image A
# was studied there; its budget is adopted per image and doubled for the pair.
theta_fit = 10*muas * (N_obs/33)**0.25
s2_unf_A = np.abs(np.linalg.det(inv_jacobian_fit[0])) * 4*theta_E_star**4 * mu_tilde[0]**4 / theta_fit**6
s2_fit_A = 1.96e-4*muasyy**2 * (33/N_obs) * ((8*year)/tau)**4
s2_star_A = s2_unf_A + s2_fit_A
sigma2_acc_eff = sigma2_acc_noise + 2*s2_star_A
print("theta_fit = %.3g muas; image-A stellar residual = %.3g (unf %.2g + fit %.2g) muas^2/yr^4"
      % (theta_fit/muas, s2_star_A/muasyy**2, s2_unf_A/muasyy**2, s2_fit_A/muasyy**2))
print("effective pair noise = %.3g muas^2/yr^4 (%.1f x instr)"
      % (sigma2_acc_eff/muasyy**2, sigma2_acc_eff/sigma2_acc_noise))
snr_B_star = np.max(eig_B, axis=-1) / sigma2_acc_eff
print("acceleration SNR along CDM rho_s(M_s), WITH stellar residual noise:")
report_cdm(arr_M_B, snr_B_star, rho_s_grid, [1., 0.1, x_sub_fid, 0.01])

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
mu_tilde_J = np.sqrt(np.linalg.norm(vec_mu_tilde_J, axis=1)**2 + pj.mu_L_int**2)
zeta_J = np.arctan2(vec_mu_tilde_J[:, 1], vec_mu_tilde_J[:, 0])
kappa_L_J = 0.5 * pj.kappa_fit
sigma_delta_theta_J = 1.0 * muas
sigma2_acc_noise_J = 720 * sigma_delta_theta_J**2 / (tau**4 * N_obs)

arr_M_J, eig_J, _ = acc_pairs(kappa_L_J, pj.inv_jacobian_fit, mu_tilde_J, zeta_J,
                              pj.d_lens, pj.d_source, pj.d_lens_source,
                              rho_s_grid, r_L_grid, theta_src_I=pj.theta_src_I)
acc_J = np.max(eig_J, axis=-1)
snr_J = acc_J / sigma2_acc_noise_J
print("acceleration SNR along CDM rho_s(M_s):")
report_cdm(arr_M_J, snr_J, rho_s_grid, [1., 0.1, pj.x_sub_fid, 0.01])
print("mu_tilde [muas/yr]:", np.round(mu_tilde_J/muasy, 2))
print("instrument floor = %.2e muas^2/yr^4" % (sigma2_acc_noise_J/muasyy**2))

# =================================== LOS-only (mean cosmological density)
print(); print("="*72); print("Line-of-sight-only acceleration SNR (mean cosmological density)"); print("="*72)
# White spectrum truncated at k, normalized so Delta2 is the power per e-fold near k;
# folded through the line-of-sight kernel of Eq. (C_tilde_1) (matter_power.ipynb machinery).
def C_pq_los(omega, B_tang, mu_t, k, Delta2, z_source_v, d_source_v):
    """C~_pq(omega) of Eq. (C_tilde_1) for P_delta = 6 pi^2 Delta2/k^3 theta(k - k').

    The 1e0 normalization is such that Delta2 is the power integrated over one e-fold
    below k. The phi integral is analytic: with x = k_perp,min/k, only |phi| < arccos x
    passes the truncation and int dphi/2pi cos^2 phi = [arccos x + x sqrt(1-x^2)]/2pi.
    B_tang is the tangential eigenvalue B_t^I of B^I, applied as a scalar to the component of
    C~ along the image motion (the dominant one).
    """
    vec_a = np.logspace(np.log10(0.999), np.log10(1/(1+z_source_v)), 2000)
    vec_z = 1/vec_a - 1
    vec_D = cosmo.angular_diameter_distance(vec_z).value * Mpc * (1+vec_z)  # comoving
    D_src = d_source_v * (1+z_source_v)
    k_perp_min = omega / (mu_t * vec_D)
    x = k_perp_min / k
    I_M = np.zeros_like(x)
    msk = x <= 1
    I_M[msk] = 3*np.pi*Delta2/k**3 * (np.arccos(np.clip(x[msk], -1, 1)) + x[msk]*np.sqrt(1-x[msk]**2))
    integrand = (1 - vec_D/D_src)**2 * I_M * (4*np.pi*G_N*rho_M_0/vec_a)**2  # rho_bar(z) a^2 = rho_m,0 (1+z)
    return B_tang**2 * 4/omega * np.trapezoid(integrand, vec_D)

def sigma2_acc_los(B_tang, mu_t, k, Delta2, z_s, d_s):
    """Acceleration variance of Eq. (observable_2) for the line-of-sight PSD above."""
    vec_o = 2*np.pi/tau * np.logspace(-8, 2, 800)
    vec_C = np.asarray([C_pq_los(o, B_tang, mu_t, k, Delta2, z_s, d_s) for o in vec_o])
    # factor 2 for the negative frequencies (two-sided PSD), as in acc_pairs above
    return np.trapezoid(2 * vec_C * vec_o**4 * smear_F2(vec_o*tau)/(2*np.pi), vec_o)

# B1422 is evaluated at image A (tangential |B| = 7.5); Fig. matter_power instead uses the
# brighter image B, |B| = 10.2, so its curves sit correspondingly lower in Delta^2.
for tag, B_tang, mu_t, z_s, d_s, s2n in [
        ("B1422/EPIC (B=7.5, mu_tilde=%.2f muas/yr, sigma=0.1)" % (mu_tilde[0]/muasy),
         7.5, mu_tilde[0], z_source, d_source, sigma2_acc_noise),
        ("J1029/survey (B=22.6, mu_tilde=%.2f muas/yr, sigma=1)" % (mu_tilde_J[2]/muasy),
         22.6, mu_tilde_J[2], pj.z_source, pj.d_source, sigma2_acc_noise_J)]:
    for lab, k, D2 in [("prompt-cusp plateau", 1e9/Mpc, 4e5),
                        ("halo model",          1e8/Mpc, 4e3)]:
        s2 = sigma2_acc_los(B_tang, mu_t, k, D2, z_s, d_s)
        print("%s | %s (Delta^2=%.0e, k=%.0e/Mpc): acc var = %.2e muas^2/yr^4, SNR = %.2g"
              % (tag, lab, D2, k*Mpc, s2/muasyy**2, s2/s2n))

# ===== J1029 worst-case acceleration from the nearest unfitted star (Sec. III D) =====
# A single identifiable perturber rather than a stochastic background: quoted at the
# median nearest-neighbour separation and at the 25th percentile.
print(); print("="*72); print("J1029 nearest-unfitted-star acceleration (worst case)"); print("="*72)
tE = pj.theta_E_star
n_star = pj.kappa_star * pj.Sigma_crit_val / pj.M_star  # stellar surface number density
for P, lab in [(0.5, "median"), (0.25, "75% clean")]:
    # nearest-star distance with P(no star closer): N(theta)= -ln P
    N_t = -np.log(P)
    theta_near = np.sqrt(N_t / (np.pi * n_star)) / pj.d_lens
    B_tang = 22.6
    acc = 2*B_tang*tE**2*mu_tilde_J[2]**2/theta_near**3
    print("%s nearest star at %.1f muas -> |acc|^2 <= %.2e muas^2/yr^4"
          % (lab, theta_near/muas, acc**2/muasyy**2))

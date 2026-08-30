"""Acceleration-channel SNR of the lens-bound PROMPT-CUSP population (App. Prompt Cusps).

Instrumental noise only, for B1422+231 (EPIC, 0.1 muas) and SDSS J1029+2623 (1 muas).
Same machinery as impact_numbers.py: the cusps are modeled as Gaussian-cutoff cusp halos
of mass M_cusp and effective size r_L ~ r_cusp/2 (half-mass matched; the answer is
insensitive to this choice because the cusps are nearly point-like at the impact
parameters that matter), carrying a lens-plane convergence kappa_L = f_cusp kappa^I.
The finite optical source size is included via theta_src^I.
"""
import sys, os
# run from anywhere: the sibling modules and the ../figs paths are resolved relative
# to this file, not to the working directory
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from sensitivity_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def smear_F2(a):
    """F_2(omega tau) of Eq. (F_2_smearing)"""
    return np.heaviside(-a + 1e-2, 0) + np.heaviside(a - 1e-2, 0) * 14400 * (
        6*a*np.cos(a/2) + (-12 + a**2)*np.sin(a/2))**2 / a**10

tau = 10 * year
N_obs = 300
vec_omega_wide = np.logspace(-6, 2, int(5e2)) * 2*np.pi/tau
d_omega_wide = vec_omega_wide[1:] - vec_omega_wide[:-1]
vec_F2 = smear_F2(vec_omega_wide[1:]*tau)

def acc_var_population(kappa_L, inv_jacobian_fit, mu_tilde, zeta, d_lens_v,
                       d_source_v, d_lens_source_v, rho_s, r_L, theta_src_I=None):
    """Max-over-pairs largest-eigenvalue differential acceleration variance of
    Eq. (observable_2), for a population of identical (rho_s, r_L) lenses carrying
    convergence kappa_L at each image."""
    n_im = len(kappa_L)
    if theta_src_I is None:
        theta_src_I = np.zeros(n_im)
    M = 4*np.pi*np.exp(0.5)*rho_s*r_L**3
    gamma_L = r_L / d_lens_v
    tE = theta_E(M, d_lens_v, d_source_v, d_lens_source_v)
    C_ij = np.zeros((n_im, 2, 2, len(vec_omega_wide)))
    for i in range(n_im):
        x_k = gamma_L/mu_tilde[i]*vec_omega_wide
        x_src = theta_src_I[i]/mu_tilde[i]*vec_omega_wide
        C_ij[i] = kappa_L[i]*tE**2/vec_omega_wide * C_ij_integral_src(x_k, x_src, zeta[i])
    C_pq = np.einsum('aqj,apjm->apqm', inv_jacobian_fit,
                     np.einsum('api,aijm->apjm', inv_jacobian_fit, C_ij))
    pairs = [(i, j) for i in range(n_im) for j in range(i+1, n_im)]
    best = 0.
    for (i, j) in pairs:
        dC = C_pq[i] + C_pq[j]
        acc = 2*np.sum(vec_F2*d_omega_wide/(2*np.pi)*vec_omega_wide[1:]**4*dC[:, :, 1:], axis=-1)
        eig = np.max(np.real(np.linalg.eigvals(acc)))
        best = max(best, eig)
    return M, best

# ===================================== true r^{-3/2} cusp form factor and its phi integral
# The Gaussian-cutoff proxy above is adequate only while the cusps are UNRESOLVED. Once
# r_cusp exceeds the sweep scale mu_tilde/omega it is far too pessimistic: a Gaussian
# truncation kills the form factor as exp{-(k r)^2/2}, whereas the real rho ~ r^{-3/2} cusp
# has F ~ 1.88 (k r_cusp)^{-3/2}, a power law. The kinetic-decoupling sweep below drives
# r_cusp up by more than an order of magnitude, so it needs the true profile.
ratio_cusp = 500.0                     # r_cusp/r_core, set by the primordial phase-space density

def _cusp_FT2(k, r_cusp, r_core):
    """|rho~(k)|^2 k^3/(8 pi^3 A^2) for a truncated, cored r^{-3/2} cusp (App. Nonlinear Matter Power Spectrum Predictions).

    Identical to matter_power.ipynb and los_full_spectrum_snr.py; -> 1 on the plateau.
    """
    Sc, _ = sp.special.fresnel(np.sqrt(2*k*r_cusp/np.pi))
    So, _ = sp.special.fresnel(np.sqrt(2*k*r_core/np.pi))
    I_cusp = np.sqrt(2*np.pi/k)*(Sc - So)
    I_core = r_core**-1.5*(np.sin(k*r_core) - k*r_core*np.cos(k*r_core))/k**2
    return ((4*np.pi/k)*(I_core + I_cusp))**2*k**3/(8*np.pi**3)

def F2_cusp(y, ratio=ratio_cusp):
    """Squared halo form factor F^2 = |rho~(k)/M_cusp|^2 as a function of y = k r_cusp.

    With M_cusp = (8 pi/3) A r_cusp^{3/2}, F^2 = (9 pi/8) _cusp_FT2 / y^3. Limits:
    F -> 1 for y << 1 (unresolved), F -> 1.88 y^{-3/2} on the plateau, cut off by the
    core for y >~ ratio. Plays the role of F_G_cusp(k gamma_L)^2 = exp{-(k gamma_L)^2}.
    """
    y = np.asarray(y, dtype=float)
    out = np.ones_like(y)
    m = y > 1e-6
    out[m] = (9*np.pi/8)*_cusp_FT2(y[m], 1.0, 1.0/ratio)/y[m]**3
    return out

# phi grid for the integral of Eq. (C); the integrand vanishes at +-pi/2 for any F that
# decays, so an interior uniform grid converges quickly.
_phi = np.linspace(-np.pi/2, np.pi/2, 2001)[1:-1]
_cos_phi = np.cos(_phi)

def C_ij_integral_profile(x_k, x_src, zeta, F2):
    """int dphi Q^I_ij(zeta,phi) F^2[x_k/cos phi] |W~[x_src/cos phi]|^2 of Eq. (C), for an
    ARBITRARY halo form factor F2(y), done numerically. The Gaussian source form factor
    |W~|^2 = exp{-(x_src/cos phi)^2} keeps the same 1/cos(phi) structure as F.

    Reduces to C_ij_integral_src (pi erfc) when F2 = exp(-y^2); checked at import below.
    """
    x_k = np.atleast_1d(x_k); x_src = np.atleast_1d(x_src)
    y = x_k[:, None]/_cos_phi[None, :]
    w = F2(y)*np.exp(-(x_src[:, None]/_cos_phi[None, :])**2)
    int_d = np.trapezoid(w, _phi, axis=-1)
    int_c = np.trapezoid(w*np.cos(2*_phi), _phi, axis=-1)
    return np.asarray([[int_d + np.cos(2*zeta)*int_c, np.sin(2*zeta)*int_c],
                       [np.sin(2*zeta)*int_c, int_d - np.cos(2*zeta)*int_c]])

def acc_var_population_profile(kappa_L, inv_jacobian_fit, mu_tilde, zeta, d_lens_v,
                               d_source_v, d_lens_source_v, M_L, r_L, F2,
                               theta_src_I=None):
    """acc_var_population for a population of (M_L, r_L) lenses with form factor F2(k r_L)."""
    n_im = len(kappa_L)
    if theta_src_I is None:
        theta_src_I = np.zeros(n_im)
    gamma_L = r_L/d_lens_v
    tE = theta_E(M_L, d_lens_v, d_source_v, d_lens_source_v)
    C_ij = np.zeros((n_im, 2, 2, len(vec_omega_wide)))
    for i in range(n_im):
        x_k = gamma_L/mu_tilde[i]*vec_omega_wide
        x_src = theta_src_I[i]/mu_tilde[i]*vec_omega_wide
        C_ij[i] = kappa_L[i]*tE**2/vec_omega_wide*C_ij_integral_profile(x_k, x_src, zeta[i], F2)
    C_pq = np.einsum('aqj,apjm->apqm', inv_jacobian_fit,
                     np.einsum('api,aijm->apjm', inv_jacobian_fit, C_ij))
    best = 0.
    for i in range(n_im):
        for j in range(i+1, n_im):
            dC = C_pq[i] + C_pq[j]
            acc = 2*np.sum(vec_F2*d_omega_wide/(2*np.pi)*vec_omega_wide[1:]**4*dC[:, :, 1:], axis=-1)
            best = max(best, np.max(np.real(np.linalg.eigvals(acc))))
    return best

# self-check: the isotropic piece of the numerical phi integral must reproduce the exact
# closed form int dphi exp{-(x/cos phi)^2} = pi erfc(x). (The anisotropic int_c has no closed
# form and is only interpolated in sensitivity_functions, so it is excluded here.)
_xt = np.array([0.03, 0.3, 1.0, 3.0])
_M = C_ij_integral_profile(_xt, 0*_xt, 0.0, lambda y: np.exp(-y**2))
_num = 0.5*(_M[0, 0] + _M[1, 1])
print("phi-integral self-check (numeric/pi erfc):",
      np.round(_num/(np.pi*sp.special.erfc(_xt)), 4))
print("F_cusp(y) at y = 1e-3, 1, 10, 100:",
      np.round(np.sqrt(F2_cusp(np.array([1e-3, 1., 10., 100.]))), 4),
      " [plateau 1.88 y^-3/2 ->", np.round(1.88*np.array([10., 100.])**-1.5, 4), "]")

# Fiducial cusp population for the 100 GeV WIMP (App. Prompt Cusps): M_cusp ~ 1e-6 M_sun,
# outer radius r_cusp ~ 5e-3 pc.
M_cusp = 1e-6 * M_Solar
r_cusp = 5e-3 * pc
f_cusp = 0.01 * 0.5      # ~1% of the DM mass in cusps as formed, times f_surv ~ 0.5

# ============================================================ B1422 (EPIC)
print("="*72); print("B1422+231, EPIC (sigma=0.1 muas, tau=10 yr, N=300)"); print("="*72)
from params_B1422_231 import *
res = np.load('macro_lens_results.npz')
inv_jacobian_fit = res['inv_jacobian_fit']; kappa_fit = res['kappa_fit']
vec_mu = mu_rel(np.zeros(2)*km/second, v_lens_fid, np.zeros(2)*km/second,
                z_lens, z_source, d_lens, d_source, d_lens_source)
vec_mu_tilde = np.tensordot(inv_jacobian_fit, vec_mu, axes=1)
# sweep rate: bulk magnified by B, internal subhalo motion UNMAGNIFIED (kinematics fix)
mu_tilde = np.sqrt(np.linalg.norm(vec_mu_tilde, axis=1)**2 + mu_L_int**2)
zeta = np.arctan2(vec_mu_tilde[:, 1], vec_mu_tilde[:, 0])
sigma2_acc_noise = 720 * (0.1*muas)**2 / (tau**4 * N_obs)

# magnified source size per image (tangential stretch of theta_src)
eigvals_B = np.linalg.eigvals(inv_jacobian_fit)
B_tang = np.max(np.abs(eigvals_B), axis=1)
theta_src_I_B = B_tang * theta_src
print("kappa_fit:", np.round(kappa_fit, 3))
print("B_tang:", np.round(B_tang, 2), " theta_src_I [muas]:", np.round(theta_src_I_B/muas, 3))
print("cusp angular size gamma_L = %.2f muas" % (r_cusp/d_lens/muas))
print("mu_tilde [muas/yr]:", np.round(mu_tilde/muasy, 2))

for r_L_eff, lab in [(0.5*r_cusp, "r_L = r_cusp/2 (half-mass match)"),
                    (r_cusp,     "r_L = r_cusp"),
                    (0.1*r_cusp, "r_L = r_cusp/10 (point-like limit)")]:
    rho_s = M_cusp / (4*np.pi*np.exp(0.5)*r_L_eff**3)
    M, var = acc_var_population(f_cusp*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                                d_lens, d_source, d_lens_source, rho_s, r_L_eff,
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
mu_tilde_J = np.sqrt(np.linalg.norm(vec_mu_tilde_J, axis=1)**2 + pj.mu_L_int**2)
zeta_J = np.arctan2(vec_mu_tilde_J[:, 1], vec_mu_tilde_J[:, 0])
sigma2_acc_noise_J = 720 * (1.0*muas)**2 / (tau**4 * N_obs)
print("cusp angular size gamma_L = %.2f muas" % (r_cusp/pj.d_lens/muas))
print("theta_src_I [muas]:", np.round(pj.theta_src_I/muas, 3))
print("mu_tilde [muas/yr]:", np.round(mu_tilde_J/muasy, 2))

for r_L_eff, lab in [(0.5*r_cusp, "r_L = r_cusp/2 (half-mass match)"),
                    (r_cusp,     "r_L = r_cusp")]:
    rho_s = M_cusp / (4*np.pi*np.exp(0.5)*r_L_eff**3)
    M, var = acc_var_population(f_cusp*pj.kappa_fit, pj.inv_jacobian_fit, mu_tilde_J, zeta_J,
                                pj.d_lens, pj.d_source, pj.d_lens_source, rho_s, r_L_eff,
                                theta_src_I=pj.theta_src_I)
    snr = var / sigma2_acc_noise_J
    print("%-36s SNR = %.2e  -> SNR=1 needs f_cusp x %.0f, or sigma = %.3f muas"
          % (lab, snr, 1/snr, 1.0*np.sqrt(snr)))

# =============== cross-check: reproduce along-CDM subhalo SNR at M=1e-6 Msun
print(); print("cross-check B1422: f_sub=0.5 CDM halo at M=1e-6 Msun (NFW + Moline+2017 c_200,")
print("                   at B1422's own x_sub = 0.03; cf. sensitivity.ipynb)")
list_M_200 = 10**np.arange(-8., 6.01, 0.05) * M_Solar
M_s_p, rho_s_p, r_s_p = scale_params_NFW_Moline(list_M_200, x_sub_fid)
i6 = np.argmin(np.abs(np.log10(M_s_p/M_Solar) + 6))
rho_eff = M_s_p[i6]/(4*np.pi*np.exp(0.5)*r_s_p[i6]**3)
M, var = acc_var_population(0.5*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                            d_lens, d_source, d_lens_source, rho_eff, r_s_p[i6],
                            theta_src_I=theta_src_I_B)
print("M_s = %.2e Msun, r_s = %.2e pc -> SNR = %.1f"
      % (M_s_p[i6]/M_Solar, r_s_p[i6]/pc, var/sigma2_acc_noise))

# ================================ kinetic-decoupling sweep (Sec. Beyond Cold Dark Matter)
# Both cusp properties are set by the damping cutoff, M_cut ~ 4e-6 M_sun
# (T_kd/30 MeV)^-3 of Eq. (Mcut-Tkd) -- acoustic damping dominates free streaming for this
# relic, hence the T_kd^-3 scaling; the cusp mass below is the Delos & White 2023 median,
# a fixed fraction of M_cut.  With A ~ a_coll^-3/2 R^3/2 and r_cusp ~ 0.11 a_coll R,
# the collapse time cancels in the mass: M_cusp = (8 pi/3) A r_cusp^3/2 ~ R^3 ~ M_cut
# exactly, while r_cusp ~ M_cut^(1/3) (up to the weak drift of a_coll with mass).  The mass
# fraction f_cusp and the plateau amplitude are set by the collapse of order-unity peaks and
# are nearly cutoff-independent, so they are held fixed here.  A LOWER T_kd therefore makes
# the cusps heavier (SNR ~ f_cusp M_cusp) but also larger, which eventually costs response.
print(); print("="*72)
print("kinetic-decoupling sweep: M_cusp ~ T_kd^-3, r_cusp ~ T_kd^-1, f_cusp fixed")
print("="*72)
# The true r^{-3/2} form factor is used here (F2_cusp): over this sweep the cusps grow from
# unresolved to several times the sweep scale, exactly where the Gaussian-cutoff proxy stops
# being trustworthy. The proxy is shown alongside to expose the size of that modeling choice.
print("%7s %11s %11s %8s | %9s %9s %9s | %9s"
      % ("T_kd", "M_cusp", "r_cusp", "gamma", "B1422", "B1422", "B1422", "J1029"))
print("%7s %11s %11s %8s | %9s %9s %9s | %9s"
      % ("[MeV]", "[M_sun]", "[pc]", "[muas]", "r^-3/2", "Gauss", "T_kd^-3", "r^-3/2"))
snr_fid_B = None
for T_kd in [30., 20., 15., 12., 10., 8., 6., 5., 4., 3., 2.]:
    s = (T_kd/30.)**-3                       # M_cut / M_cut,fid = M_cusp / M_cusp,fid
    M_c = 1e-6 * s * M_Solar
    r_c = 5e-3 * s**(1/3) * pc
    var_B = acc_var_population_profile(f_cusp*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                                       d_lens, d_source, d_lens_source, M_c, r_c, F2_cusp,
                                       theta_src_I=theta_src_I_B)
    var_J = acc_var_population_profile(f_cusp*pj.kappa_fit, pj.inv_jacobian_fit, mu_tilde_J,
                                       zeta_J, pj.d_lens, pj.d_source, pj.d_lens_source,
                                       M_c, r_c, F2_cusp, theta_src_I=pj.theta_src_I)
    # Gaussian-cutoff proxy at the half-mass radius, for comparison only
    r_L_eff = 0.5*r_c
    _, var_G = acc_var_population(f_cusp*kappa_fit, inv_jacobian_fit, mu_tilde, zeta,
                                  d_lens, d_source, d_lens_source,
                                  M_c/(4*np.pi*np.exp(0.5)*r_L_eff**3), r_L_eff,
                                  theta_src_I=theta_src_I_B)
    snr_B = var_B/sigma2_acc_noise
    if snr_fid_B is None: snr_fid_B = snr_B
    print("%7.0f %11.1e %11.1e %8.2f | %9.3f %9.3f %9.3f | %9.2e"
          % (T_kd, M_c/M_Solar, r_c/pc, r_c/d_lens/muas, snr_B,
             var_G/sigma2_acc_noise, snr_fid_B*s, var_J/sigma2_acc_noise_J))

# ============================ occupancy: is the Gaussian (PSD) description still valid?
# Sec. II rests the PSD treatment of Eq. (C) on a large column occupancy
# N_col = kappa_L Sigma_cr r_L^2 / M_L, and warns that the count that actually matters is
# the WEIGHTED effective number of perturbers, reduced for cuspy profiles. Prompt cusps are
# exactly that case, and lowering T_kd makes them heavier but RARER (n ~ 1/M_cusp ~ T_kd^3),
# so the sweep above walks toward the nearest-neighbour-dominated regime.
#
# Weighting: for a lens at impact parameter b from the swept image path, the fitted
# differential acceleration is ~ alpha(b) v^2/b^2, so the per-encounter contribution to the
# variance is s(b) = [alpha(b)/b^2]^2 with alpha = M_2D(<b)/b. Inside the cusp
# M_2D ~ b^{3/2}, hence alpha ~ b^{1/2} and s ~ b^-3: dV/dln b ~ b^-1 is dominated by the
# SMALLEST impact parameter, which is pinned by the magnified source size b_src (NOT by
# r_cusp). Encounters along the campaign path have a uniform density dN = 2 n L db out to
# b ~ L = mu_tilde tau, beyond which the encounter is slower than the campaign and the
# omega^4 F_2 kernel discards it. The effective number of contributors is then the inverse
# participation ratio N_perturb = (int s dN)^2 / int s^2 dN; N_perturb ~ 1 is the point at
# which the nearest perturber alone matters as much as all the others combined. This is
# the "weighted effective number of perturbers" of Sec. II C -- NOT the paper's N_eff,
# which is the effective number of DFT MODES, sum_a lambda_a/(1+lambda_a) (Sec. II E).
i_B = 1                                             # image B drives the pair variance
Sigma_cr_v = Sigma_crit(d_lens, d_source, d_lens_source)
Sigma_L = f_cusp*kappa_fit[i_B]*Sigma_cr_v          # projected mass density in cusps
b_src = theta_src_I_B[i_B]*d_lens                   # magnified source size, as a length
L_path = mu_tilde[i_B]*tau*d_lens                   # swept path length over the campaign

def M_2D_cusp(b, r_c):
    """Projected mass inside cylinder radius b for rho = A r^-3/2 truncated at r_c,
    normalized to M_cusp = 1. -> 1 for b >= r_c."""
    b = np.atleast_1d(b).astype(float)
    out = np.ones_like(b)
    for i, bv in enumerate(b):
        if bv >= r_c:
            continue
        Rp = np.linspace(1e-12*r_c, bv, 300)
        zc = np.sqrt(np.clip(r_c**2 - Rp**2, 0, None))
        Sig = np.array([2*np.trapezoid((Rp[j]**2 + np.linspace(0, zc[j], 200)**2)**-0.75,
                                       np.linspace(0, zc[j], 200)) for j in range(len(Rp))])
        out[i] = np.trapezoid(2*np.pi*Rp*Sig, Rp)/((8*np.pi/3)*r_c**1.5)
    return out

def N_perturb_population(M_L, r_L, M_2D_fun):
    """Inverse participation ratio of the acceleration variance over encounter impact
    parameter, for lenses of mass M_L and size r_L at surface density Sigma_L."""
    n = Sigma_L/M_L                                 # lenses per unit area
    b = np.logspace(np.log10(b_src) - 2, np.log10(L_path), 220)
    b_eff = np.sqrt(b**2 + b_src**2)                # source size smooths the deflection
    s = (M_2D_fun(b_eff, r_L)/b_eff/b_eff**2)**2    # [alpha(b)/b^2]^2, alpha = M_2D/b
    s = s/s.max()                                   # N_perturb is invariant under s -> c s;
    bb = b/b_src                                    # rescale both to avoid over/underflow
    # N_perturb = (int s dN)^2 / int s^2 dN with dN = 2 n L db, so the 2 n L prefactor survives
    # once as an overall factor: N_perturb = 2 n L b_src (int s dbb)^2 / int s^2 dbb.
    int_s = np.trapezoid(s, bb); int_s2 = np.trapezoid(s**2, bb)
    return 2*n*L_path*b_src*int_s**2/int_s2, 2*n*L_path*b_src, Sigma_L*r_L**2/M_L

print(); print("="*72)
print("occupancy of the LENS-BOUND cusps (B1422 image B): when does the PSD description fail?")
print("="*72)
print("b_src = %.2e pc (%.2f muas), swept path L = %.2e pc (%.1f muas), Sigma_L = %.2f M_sun/pc^2"
      % (b_src/pc, b_src/d_lens/muas, L_path/pc, L_path/d_lens/muas, Sigma_L/(M_Solar/pc**2)))
print("%7s %11s %11s | %9s %11s %9s"
      % ("T_kd", "M_cusp", "r_cusp", "N_col", "N(<b_src)", "N_perturb"))
print("%7s %11s %11s | %9s %11s %9s"
      % ("[MeV]", "[M_sun]", "[pc]", "Sec. II", "swept", "weighted"))
for T_kd in [30., 20., 15., 12., 10., 9., 8., 6., 5., 4., 3., 2.]:
    s_ = (T_kd/30.)**-3
    M_c = 1e-6*s_*M_Solar; r_c = 5e-3*s_**(1/3)*pc
    N_perturb, N_src, N_col = N_perturb_population(M_c, r_c, M_2D_cusp)
    print("%7.0f %11.1e %11.2e | %9.1f %11.2f %9.2f"
          % (T_kd, M_c/M_Solar, r_c/pc, N_col, N_src, N_perturb))

# Validation: Sec. II states the weighted count is "comparable to N_col for the smooth,
# Gaussian-cutoff profiles adopted below". Run the SAME estimator on that profile
# (rho ~ exp[-r^2/2 r_L^2]/r, so M_2D rises ~ b^2 at small b and saturates) and check that
# it is NOT nearest-neighbour dominated where the cusps are. Note N_perturb and N_col use
# different normalizations (swept path vs crossing-time column), so compare trends and
# orders of magnitude, not absolute values.
def M_2D_smooth(b, r_L):
    """Projected mass inside b for rho ~ exp(-r^2/2 r_L^2)/r, normalized to M_L = 1."""
    b = np.atleast_1d(b).astype(float)
    u = b/r_L
    return 1 - np.exp(-u**2/2)          # exact for this profile: M_2D/M_L = 1 - e^{-u^2/2}
print()
print("validation, smooth Gaussian-cutoff profile at the same (M_L, r_L) [Sec. II claim:")
print("weighted count comparable to N_col, i.e. NOT nearest-neighbour dominated]:")
print("%7s %11s %11s | %9s %11s %9s" % ("T_kd", "M_L", "r_L", "N_col", "N(<b_src)", "N_perturb"))
for T_kd in [30., 9., 6., 3., 2.]:
    s_ = (T_kd/30.)**-3
    M_c = 1e-6*s_*M_Solar; r_c = 5e-3*s_**(1/3)*pc
    N_perturb, N_src, N_col = N_perturb_population(M_c, r_c, M_2D_smooth)
    print("%7.0f %11.1e %11.2e | %9.1f %11.2f %9.2f"
          % (T_kd, M_c/M_Solar, r_c/pc, N_col, N_src, N_perturb))

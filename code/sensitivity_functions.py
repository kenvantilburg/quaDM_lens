"""Substructure profiles, form factors and noise levels for the astrometric
weak-lensing forecasts of Van Tilburg & Kaplan.

Notation follows the paper:
  M_L, r_L, rho_s, gamma_L = r_L/d_L, kappa_L, theta_E,L  -- the microhalo population
    (gamma_L uses the angular-diameter distance d_lens, the paper's D_L, not the
     comoving D_lens; see the distance convention in macro_lens_functions)
  C_ij / C_pq             -- the PSD C~^I_pq(omega) of Eq. (C), before / after B^I
  x_k = omega gamma_L / mu_tilde^I, x_src = omega theta_src^I / mu_tilde^I
  tau, N_obs, sigma_delta_theta -- survey baseline, epochs, per-epoch precision
The fiducial LambdaCDM rho_s(M_s) prediction band of Fig. SNR comes from the NFW +
Moline+2017 helpers (scale_params_NFW_Moline); the Wang+2020 Einasto helpers feed the
one-halo field-halo spectrum of matter_power.ipynb and the line-of-sight scripts. The
signal itself uses the Gaussian-cutoff 1/r cusp of Sec. III B (form factor F_G_cusp).
"""
from natural_units_GeV import *
from preamble import *
from macro_lens_functions import *

##### DM profiles #####
# Einasto profile, rho(r) = rho_s exp{-(2/alpha)[(r/r_s)^alpha - 1]}, used only for the
# LambdaCDM concentration-mass prediction of Sec. IV A; alpha = 0.16 is the standard
# low-mass value. The astrometric signal itself uses the cusp profile further below.
def rho_Einasto(r, rho_s, r_s, alpha=0.16):
    """Mass density of Einasto profile (rho_s = rho(r_s))"""
    return rho_s * np.exp(-2./alpha * ((r/r_s)**alpha - 1))

def M_enc_Einasto(r, rho_s, r_s, alpha=0.16):
    """Mass enclosed within radius r of Einasto profile"""
    prefac = rho_s * r_s**3 * 2**(2-3/alpha) * np.exp(2/alpha) * np.pi * alpha**(-1+3/alpha)
    return prefac * sp.special.gamma(3/alpha) * sp.special.gammainc(3/alpha, 2./alpha * (r/r_s)**alpha)

def M_Einasto(rho_s, r_s, alpha=0.16):
    """Total mass of Einasto profile (r -> infinity)"""
    prefac = rho_s * r_s**3 * 2**(2-3/alpha) * np.exp(2/alpha) * np.pi * alpha**(-1+3/alpha)
    return prefac * sp.special.gamma(3/alpha)

def c_Einasto(M_200, h=0.7, M_cut=1e-20 * M_Solar):
    """Concentration c_200(M_200) at z = 0 from Wang et al. 2020 (Nature 585, 39).

    Fifth-order polynomial in ln(M_200 h/M_Solar) times the cutoff suppression
    exp[-0.529 (M_cut/M_200)^(1/3)], which truncates the relation near the damping
    cutoff mass of a thermal relic. The default M_cut is negligibly small, i.e.
    purely cold DM with no cutoff; pass the WIMP M_cut for the cutoff variant.
    """
    c_Einasto_coeff = np.asarray([27.112, -0.381, -1.853e-3, -4.141e-4, -4.334e-6, 3.208e-7, -0.529])
    x = np.log(M_200 / (h**-1 * M_Solar))
    pol = 0
    for i in range(6):
        pol += c_Einasto_coeff[i] * x**i
    return np.exp(c_Einasto_coeff[6] * (M_cut/M_200)**(1/3)) * pol

def r_200(M_200,h=0.7):
    """Radius of sphere with mean density 200 times the critical density"""
    return (3 * M_200 / (4 * np.pi * 200 * (h/0.7)**2 * rho_crit))**(1/3)

def r_s_Einasto_200(M_200, h=0.7, M_cut = 1e-20 * M_Solar):
    """Einasto scale radius r_s = r_200 / c_200(M_200)"""
    return r_200(M_200) / c_Einasto(M_200, h, M_cut)

# Scale density: solve M_enc(r_200) = M_200 for rho_s. M_enc is linear in rho_s, so the
# ratio M_200 / M_enc(r_200; rho_s = 1 M_Solar/pc^3) is that solution in those units.
vec_M_200 = np.logspace(-12,12,1000) * M_Solar
vec_r_200 = r_200(vec_M_200)
vec_r_s_Einasto = r_s_Einasto_200(vec_M_200)
vec_M_enc = M_enc_Einasto(vec_r_200, M_Solar / pc**3, vec_r_s_Einasto)
int_rho_s_Einasto_200 = interp1d(vec_M_200, vec_M_200/vec_M_enc * M_Solar / pc**3,
                                 bounds_error=False, fill_value=0)
def rho_s_Einasto_200(M_200):
    """Einasto scale density rho_s = rho(r_s) implied by c_200(M_200) and r_200.
    Gives rho_s ~ 0.03 (0.018) M_Solar/pc^3 at M_200 = 10^-6 (1) M_Solar.
    """
    return int_rho_s_Einasto_200(M_200)

##### Fiducial LambdaCDM scale-density relation: NFW + Moline+2017 concentrations #####
# This block replaces the Wang+2020 Einasto bookkeeping above for the black dashed /
# gray-band CDM prediction of figs/SNR.pdf. Two reasons: (i) the relevant halos are
# SUBhalos of the lens galaxy, which are systematically more concentrated than field
# halos of the same mass, and (ii) the earlier construction needed an ad hoc factor 20
# on rho_s to land on the rho_s ~ 0.1-1 M_Solar/pc^3 of the collapse-redshift argument
# of Sec. IV A. The Moline+2017 relation reproduces that range with no renormalization.
def m_NFW(c):
    """NFW mass integral m(c) = ln(1+c) - c/(1+c); M(<c r_s) = 4 pi rho_char r_s^3 m(c)."""
    return np.log(1+c) - c/(1+c)

def c_200_Moline(M_200, x_sub=1., h=0.7):
    """Median subhalo concentration c_200(m_200, x_sub), Eq. (7) of Moline et al. 2017
    (MNRAS 466, 4974), calibrated on Via Lactea II + ELVIS (plus BolshoiP and the
    Ishiyama 2014 microhalos at x_sub = 1) at z = 0:

        c_200 = c_0 [1 + sum_{i=1..3} (a_i log10(m_200 / 10^8 h^-1 M_Solar))^i]
                    x [1 + b log10(x_sub)],

    with c_0 = 19.9, a_i = {-0.195, 0.089, 0.089}, b = -0.54 (their Tab. 2). Note the
    unusual nesting: the a_i multiply the log BEFORE the power i is taken.

    x_sub = R_sub/R_200^host is the subhalo's distance from the host centre in units of
    the host virial radius; x_sub = 1 recovers field-halo concentrations. The fit is
    quoted for 10^-6 h^-1 M_Solar < m_200 < 10^15 h^-1 M_Solar and diverges
    logarithmically as x_sub -> 0, so it should not be pushed below x_sub ~ 0.01.
    """
    c_0 = 19.9
    a_i = np.asarray([-0.195, 0.089, 0.089])
    b = -0.54
    y = np.log10(M_200 / (1e8 * h**-1 * M_Solar))
    pol = 1.
    for i in (1, 2, 3):
        pol = pol + (a_i[i-1] * y)**i
    return c_0 * pol * (1 + b * np.log10(x_sub))

def scale_params_NFW_Moline(M_200, x_sub=1., h=0.7):
    """(M_s, rho_s, r_s) of an NFW subhalo of virial mass M_200 at host distance x_sub.

    Conventions match Sec. III B of the paper, where rho_s is the density AT the scale
    radius and r_L plays the role of r_s:
        r_s    = r_200 / c_200,
        rho_s  = rho_NFW(r_s) = rho_char/4 = (50/3) rho_crit c_200^3 / m(c_200),
        M_s    = M(<r_s) = M_200 m(1)/m(c_200),   m(1) = ln 2 - 1/2.
    The paper's Gaussian-cutoff cusp instead carries a TOTAL mass
    M_L = 4 pi sqrt(e) rho_s r_s^3 = 2.13 M_s at the same (r_s, rho_s); that O(1)
    convention offset is smaller than the width of the x_sub band and is not corrected.
    """
    c = c_200_Moline(M_200, x_sub, h)
    r_s = r_200(M_200, h) / c
    rho_s = (200/3) * (h/0.7)**2 * rho_crit * c**3 / m_NFW(c) / 4   # rho_NFW(r_s)
    M_s = M_200 * m_NFW(1.) / m_NFW(c)
    return M_s, rho_s, r_s

vec_x_b = np.logspace(-4,4,int(1e3))
vec_Sigma_x_b = np.zeros_like(vec_x_b)
for i,x_b in enumerate(vec_x_b):
    vec_Sigma_x_b[i] = quad(lambda z: 2*rho_Einasto(np.sqrt(x_b**2 + z**2), 1, 1), 0, 1e4)[0]
int_Sigma_x_b = interp1d(vec_x_b, vec_Sigma_x_b, bounds_error=False, fill_value=0) # scale-free surface mass density

vec_m_x_b = odeint(lambda m,x_b : 2 * np.pi * x_b * int_Sigma_x_b(x_b), 0, vec_x_b)[:,0] # scale-free mass enclosed within x_b
int_m_x_b = interp1d(vec_x_b, vec_m_x_b, bounds_error=False, fill_value=(0,vec_m_x_b[-1])) # scale-free mass enclosed within x_b

int_alpha_x_b = interp1d(vec_x_b, vec_m_x_b / (np.pi * vec_x_b), bounds_error=False, fill_value=0) # scale-free deflection angle

def Sigma_Einasto(b, rho_s, r_s):
    """Projected surface mass density at impact parameter b (alpha = 0.16)"""
    return rho_s * r_s * int_Sigma_x_b(b/r_s)

def Sigma_Einasto_200(b, M_200):
    """Sigma_Einasto for the (r_s, rho_s) implied by c_200(M_200)"""
    r_s = r_s_Einasto_200(M_200)
    rho_s = rho_s_Einasto_200(M_200)
    return Sigma_Einasto(b, rho_s, r_s)

def alpha_Einasto(rho_s, r_s, d_l, d_s, d_ls, arr_theta):
    """Reduced deflection angle vector alpha(theta) of an Einasto halo (alpha = 0.16)"""
    theta = np.linalg.norm(arr_theta,axis=0)
    return rho_s * r_s**2 / d_l * int_alpha_x_b(theta*d_l/r_s) / Sigma_crit(d_l,d_s,d_ls) * arr_theta / (theta + 1e-30) # 1e-30 to avoid division by zero

def alpha_Einasto_200(M_200, d_l, d_s, d_ls, arr_theta):
    """alpha_Einasto for the (r_s, rho_s) implied by c_200(M_200)"""
    r_s = r_s_Einasto_200(M_200)
    rho_s = rho_s_Einasto_200(M_200)
    return alpha_Einasto(rho_s, r_s, d_l, d_s, d_ls, arr_theta)

##### Halo and source form factors, and the phi integral of Eq. (C) #####
def F_G_cusp(k_gamma_L):
    """Halo form factor F(k gamma_L) of Eq. (form) for the fiducial profile.

    Sec. III B: a 1/r cusp with a Gaussian cutoff, rho(r) ~ exp{-r^2/2 r_L^2}/r, for
    which the enclosed-mass integral of Eq. (form) collapses to F = exp{-(k gamma_L)^2/2}
    and the total mass is finite, M_L = 4 pi sqrt(e) rho_s r_L^3.
    """
    return np.exp(-k_gamma_L**2/2)

def F_G_cusp_integral_d(x_k):
    """Isotropic piece of the phi integral of Eq. (C): int dphi F[x_k/cos phi]^2.

    Exact for F = F_G_cusp: int_{-pi/2}^{pi/2} dphi exp{-(x_k/cos phi)^2} = pi erfc(x_k).
    """
    return np.pi * sp.special.erfc(x_k)

def F_G_cusp_integral_c(x_k):
    """Anisotropic piece: int dphi cos(2 phi) F[x_k/cos phi]^2.

    No closed form; this interpolation between the small- and large-x_k limits is
    accurate to better than 7% over the whole range, and multiplies the (already
    subleading) cos 2 zeta terms of Q^I.
    """
    return (6/5) / (1 + 1/x_k) * np.pi * sp.special.erfc(x_k)

def C_ij_integral(x_k, zeta):
    """The phi integral of Eq. (C): int dphi Q^I_ij(zeta, phi) F[x_k/cos phi]^2.

    With Q^I of Eq. (Q), the 2x2 result is built from the isotropic (int_d) and
    cos-2phi-weighted (int_c) pieces above. Multiply by kappa_L theta_E,L^2/|omega|
    and contract with B^I on both indices to obtain C~^I_pq(omega).
    Arguments: x_k = |omega| gamma_L / mu_tilde^I, zeta = position angle of mu_tilde^I.
    """
    int_d = F_G_cusp_integral_d(x_k)
    int_c = F_G_cusp_integral_c(x_k)
    return np.asarray([[int_d + np.cos(2*zeta) * int_c, np.sin(2*zeta) * int_c],
                        [np.sin(2*zeta) * int_c, int_d - np.cos(2*zeta) * int_c]])

def C_ij_integral_src(x_k, x_src, zeta):
    """C_ij_integral including the finite-source form factor of Eq. (finite_source).

    The source form factor |W~^I|^2 is the squared Fourier transform of the image
    surface-brightness profile (the quasar form factor of Galanis+ 2023,
    arXiv:2307.06989). For a Gaussian source and the Gaussian-cutoff cusp halo it
    enters the phi integrand with the same 1/cos(phi) structure as F, so the two
    scales add in quadrature inside the closed form:
        int dphi F[x_k/c_phi]^2 |W~[x_src/c_phi]|^2 = pi erfc(sqrt(x_k^2 + x_src^2)),
    i.e. gamma_L^2 -> gamma_L^2 + (theta_src^I)^2 (App. SDSS J1029+2623).
    Here x_src = |omega| theta_src^I / mu_tilde^I, with theta_src^I the source size
    magnified by B^I. Point source: x_src = 0.
    """
    return C_ij_integral(np.sqrt(x_k**2 + x_src**2), zeta)

def C_inst(omega, tau=10*year, N_obs=100, sigma_delta_theta=muas):
    """White instrumental noise PSD of Eq. (noise_1): sigma_delta_theta^2 tau / N.

    Per DFT mode of a campaign of N_obs equally spaced epochs over a baseline tau,
    at per-epoch relative light-centroiding precision sigma_delta_theta. Not
    EPIC-specific: also used for the 1 muas cluster-lens survey of Sec. III D.

    Modes below the fundamental omega = 2 pi / tau are inaccessible and are returned
    at an effectively infinite noise level (the 1e-50 floor). The fundamental itself
    IS measurable and must be counted, hence np.heaviside(x, 1), which evaluates the
    n = 1 bin (x = 0) to 1 rather than dropping it; matter_power.ipynb and
    los_full_spectrum_snr.py include the same mode by evaluating at 1.001*omega_min.
    """
    return tau / N_obs * sigma_delta_theta**2 / (1e-50 + np.heaviside(omega  * tau / (2*np.pi) - 1,1))


##### off-plane (multi-plane) corrections to the line-of-sight kernel #####
# Eq. (C_tilde_1) as written keeps the single lens-plane Jacobian B^I outside the sightline
# integral and sweeps every plane at chi mu_tilde^I, i.e. it treats a perturber at comoving
# distance chi as if it sat at chi_L. Solving the two-plane lens equation to first order in
# the perturber's deflection (App. Matter Power Spectrum) gives instead
#     delta theta = [(1 - beta) B^I + beta 1] alpha_perturber ,
#     beta(chi) = (chi_L - chi) chi_S / [chi_L (chi_S - chi)]   (0 behind the lens),
# and a beam whose transverse comoving velocity is PIECEWISE LINEAR, pinned at v_o at the
# observer, at chi_L mu_tilde^I + v_L at the lens plane, and at v_S on the source. Dropping
# the unmagnified drift (v_o, v_L, v_S; a few per cent of chi_L mu_tilde^I and worth < 1% of
# the integral), the magnified sweep is the tent function sweep_offplane below.
# Each error lives on one side of the lens plane: the amplitude is exact behind it, the
# sweep rate exact in front of it, and both are overestimates.
def sweep_offplane(D, D_lens, D_source):
    """Comoving lever arm X(chi) replacing chi in the sweep rate of Eq. (C_tilde_1).

    The beam's transverse comoving velocity at comoving distance chi is mu_tilde^I X(chi)
    (plus unmagnified drift), with the tent function
        X(chi) = chi_L min[ chi/chi_L , (chi_S - chi)/(chi_S - chi_L) ] ,
    which rises to the lens plane exactly as the naive chi does and then falls back to zero
    at the source: the beam re-converges onto the quasar, so the MAGNIFIED part of the sweep
    must vanish there. The naive chi instead extrapolates the magnified sweep past the lens.
    All distances comoving (the uppercase D_* convention of macro_lens_functions).
    """
    return D_lens * np.minimum(D / D_lens, (D_source - D) / (D_source - D_lens))

def B_offplane(D, D_lens, D_source, B_tang):
    """Effective (scalar) inverse Jacobian B_eff(chi) replacing B_t^I in Eq. (C_tilde_1).

    B_eff = (1 - beta) B_t^I + beta with the standard multi-plane ratio
        beta(chi) = (chi_L - chi) chi_S / [chi_L (chi_S - chi)]  for chi < chi_L, else 0.
    A perturber behind the lens shifts the source-plane position directly and is amplified
    by the full B^I (beta = 0); one in front is partly "pre-lensed" by the macrolens, and at
    the observer (beta = 1) its deflection is not magnified at all (B_eff = 1).
    """
    beta = np.where(D < D_lens,
                    (D_lens - D) * D_source / (D_lens * np.maximum(D_source - D, 1e-30)),
                    0.0)
    return (1 - beta) * B_tang + beta

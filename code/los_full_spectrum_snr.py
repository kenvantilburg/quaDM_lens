"""Variance SNRs of the PREDICTED matter power spectra along the line of sight.

Folds each predicted Delta^2_delta(k) -- the pure-CDM one-halo band (no cutoff), the
fiducial WIMP total (damping-cutoff halos plus prompt cusps), and the cusp
component alone -- through the line-of-sight kernel of Eq. (C_tilde_1) at the MEAN
cosmological density, i.e. counting only field halos and no lens-bound substructure.
These are the "guaranteed variance" numbers quoted in Sec. IV B.

Two systems, both over tau = 10 yr with N = 300 epochs: the galaxy lens B1422+231
(image B, tangential |B| ~ 10.2, sigma_delta_theta = 0.1 muas) and the cluster lens
SDSS J1029+2623 (image C, |B| ~ 22.6, 1 muas). Two channels: the stochastic excess
variance in the lowest accessible DFT mode (Eq. (observable_1)) and the differential
acceleration (Eq. (observable_2)). Conventions as in matter_power.ipynb; unlike the
per-mode sensitivity curves there, the spectra here are the actual Delta^2_delta(k)
rather than a per-e-fold amplitude."""
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
import params_J1029_2623 as pJ

tau = 10 * year
N_obs = 300
omega_min = 2*np.pi/tau
rho_crit_0 = rho_crit                    # 3 H_0^2 m_Pl^2, the units-module value
delta_c = 1.686

# ------------------------------------------------------------------ spectra
data = np.loadtxt("Delta2_extended_output.csv", delimiter=",", skiprows=1)
kk_ext, k_h_ext, D2_lin_ext, D2_nl_ext, D2_nl_30 = data.T
k_cut = 1.06e6 * Mpc**-1
M_cut = (4*np.pi/3) * rho_M_0 * (np.pi/k_cut)**3
_int_lnD2lin = interp1d(np.log(kk_ext), np.log(D2_lin_ext), bounds_error=False,
                        fill_value=(np.log(D2_lin_ext[0]), np.log(D2_lin_ext[-1])))
def Delta2_lin(k, cutoff=True):
    out = np.exp(_int_lnD2lin(np.log(k * Mpc)))
    return out * np.exp(-2*(k/k_cut)**2) if cutoff else out
def _W_tophat(x):
    x = np.asarray(x, float); w = np.ones_like(x); g = x > 1e-3; xx = x[g]
    w[g] = 3*(np.sin(xx) - xx*np.cos(xx))/xx**3
    return w
def sigma_of_M(M, cutoff):
    R = (3*M/(4*np.pi*rho_M_0))**(1/3)
    f = lambda lk: Delta2_lin(np.exp(lk)*Mpc**-1, cutoff) * _W_tophat(np.exp(lk)*Mpc**-1*R)**2
    return np.sqrt(quad(f, np.log(1e-4), np.log(1e11), limit=400)[0])
vec_M_sig = np.logspace(-14, 16, 120) * M_Solar
_int_lnsig = {c: interp1d(np.log(vec_M_sig),
                          np.log([sigma_of_M(M, c) for M in vec_M_sig]),
                          fill_value='extrapolate') for c in (True, False)}
def sigma_M(M, cutoff): return np.exp(_int_lnsig[cutoff](np.log(M)))
def dlnsig(M, cutoff):
    h = 1e-3
    return (np.log(sigma_M(M*np.exp(h), cutoff)) - np.log(sigma_M(M*np.exp(-h), cutoff)))/(2*h)
def dn_dlnM(M, cutoff):
    nu = delta_c/sigma_M(M, cutoff); A_ST, a_ST, p_ST = 0.3222, 0.707, 0.3
    nu_p = np.sqrt(a_ST)*nu
    f_nu = A_ST*np.sqrt(2/np.pi)*nu_p*np.exp(-nu_p**2/2)*(1 + nu_p**(-2*p_ST))
    return (rho_M_0/M)*f_nu*abs(dlnsig(M, cutoff))
_scp14_coeff = [37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7]
def _scp14_pol(M):
    x = np.log(M/(cosmo.h**-1*M_Solar))
    return sum(ci*x**i for i, ci in enumerate(_scp14_coeff))
def conc(M, kind, cutoff):
    base = c_Einasto(M, h=cosmo.h, M_cut=(M_cut if cutoff else 1e-20*M_Solar)) if kind == 'Wang' \
        else (np.exp(-0.529*(M_cut/M)**(1/3)) if cutoff else 1.0)*_scp14_pol(M)
    return np.clip(base, 2, 200)
def r_200f(M):
    return (3*M/(4*np.pi*200*rho_crit_0))**(1/3)
def u_NFW(k, M, kind, cutoff):
    c = conc(M, kind, cutoff); r_s = r_200f(M)/c; x = k*r_s
    mc = np.log(1+c) - c/(1+c)
    si, ci = sp.special.sici(x); si1, ci1 = sp.special.sici((1+c)*x)
    return (np.sin(x)*(si1-si) + np.cos(x)*(ci1-ci) - np.sin(c*x)/((1+c)*x))/mc

_lnM = np.log(np.logspace(-13, 16, 250)*M_Solar)
_kk = np.logspace(2, np.log10(3e11), 400) * Mpc**-1
def make_D2_PS(kind, cutoff):
    D2 = np.zeros_like(_kk); dlnM = _lnM[1] - _lnM[0]
    for lnM in _lnM:
        M = np.exp(lnM)
        D2 += dn_dlnM(M, cutoff)*M**2*u_NFW(_kk, M, kind, cutoff)**2*dlnM
    return D2*_kk**3/(2*np.pi**2)/rho_M_0**2

# prompt cusps (as in matter_power.ipynb cell 31)
z5_cusp, f_surv, ratio_cusp = 31.0, 0.5, 500.0
r_cusp_med = 5e-3 * pc
def _cusp_FT2(k, r_cusp, r_core):
    Sc, _ = sp.special.fresnel(np.sqrt(2*k*r_cusp/np.pi))
    So, _ = sp.special.fresnel(np.sqrt(2*k*r_core/np.pi))
    I_cusp = np.sqrt(2*np.pi/k)*(Sc - So)
    I_core = r_core**-1.5*(np.sin(k*r_core) - k*r_core*np.cos(k*r_core))/k**2
    return ((4*np.pi/k)*(I_core + I_cusp))**2*k**3/(8*np.pi**3)
def Delta2_cusp(k, r_cusp=r_cusp_med):
    """Prompt-cusp one-halo term. Only the LEFT EDGE of the plateau, k ~ 1/r_cusp, moves
    with the damping cutoff: the plateau amplitude is anchored to the cusp
    annihilation rate of Delos & White 2023 and is nearly cutoff-independent."""
    k = np.atleast_1d(k).astype(float)
    plateau = 160*f_surv*(1 + z5_cusp)**3/(0.531 + np.log(ratio_cusp))
    lnr = np.log(r_cusp) + np.linspace(-1.5, 1.5, 21)*0.6
    w = np.exp(-((lnr - np.log(r_cusp))/0.6)**2/2); w /= w.sum()
    shape = np.zeros_like(k)
    for lr, wi in zip(lnr, w):
        shape += wi*_cusp_FT2(k, np.exp(lr), np.exp(lr)/ratio_cusp)
    return plateau*shape

D2_CDM_lo = make_D2_PS('Wang', False)
D2_CDM_hi = make_D2_PS('SCP14', False)
D2_WIMP_lo = make_D2_PS('Wang', True) + Delta2_cusp(_kk)
D2_WIMP_hi = make_D2_PS('SCP14', True) + Delta2_cusp(_kk)
D2_cusp_only = Delta2_cusp(_kk)
def interp_D2(D2):
    i = interp1d(np.log(_kk), np.log(np.clip(D2, 1e-30, None)),
                 bounds_error=False, fill_value=-np.inf)
    return lambda k: np.exp(i(np.log(k)))

spectra = [('pure-CDM band, low (Wang)', interp_D2(D2_CDM_lo)),
           ('pure-CDM band, high (SCP14)', interp_D2(D2_CDM_hi)),
           ('WIMP total, low', interp_D2(D2_WIMP_lo)),
           ('WIMP total, high', interp_D2(D2_WIMP_hi)),
           ('prompt cusps only', interp_D2(D2_cusp_only))]

# --------------------------------------- line-of-sight kernel of Eq. (C_tilde_1)
k_grid = np.logspace(2, np.log10(3e11), 600) * Mpc**-1
def I_M_general(omega, mu_t, X_arr, Delta2_fun):
    """int dphi/2pi M_11 P_delta of Eq. (C_tilde_1), for an arbitrary P_delta(k).

    Any decreasing P(k) is a positive superposition of white spectra truncated at k_i,
    P = sum_i dP_i theta(k_i - k), and each of those has the closed-form angular kernel
    [arccos x + x sqrt(1-x^2)]/2pi with x = k_perp,min/k_i (see matter_power.ipynb).
    X_arr is the comoving lever arm of sweep_offplane, i.e. the beam's transverse sweep
    velocity at that plane is mu_t X_arr (the naive kernel uses X = chi throughout).
    """
    P = 2*np.pi**2 * Delta2_fun(k_grid) / k_grid**3
    w = P[:-1] - P[1:]                       # white-shell amplitudes, >= 0 for decreasing P
    kmid = np.sqrt(k_grid[:-1]*k_grid[1:])
    x = (omega/(mu_t*np.maximum(X_arr, 1e-30)))[:, None]/kmid[None, :]
    g = np.where(x <= 1, np.arccos(np.clip(x, -1, 1)) + x*np.sqrt(np.clip(1 - x**2, 0, 1)), 0.0)
    return np.sum(w[None, :]/(2*np.pi)*g, axis=1)

def smear_F2(a):
    """F_2(omega tau) of Eq. (F_2_smearing)"""
    return np.heaviside(-a + 1e-2, 0) + np.heaviside(a - 1e-2, 0)*14400*(
        6*a*np.cos(a/2) + (-12 + a**2)*np.sin(a/2))**2/a**10
vec_omega = 2*np.pi/tau*np.logspace(-7, 2, 500)

def make_los(params):
    """Comoving line-of-sight grid from the observer to that system's source redshift"""
    vec_a = np.logspace(np.log10(0.999), np.log10(1/(1 + params.z_source)), 2000)
    vec_z = 1/vec_a - 1
    vec_D = cosmo.angular_diameter_distance(vec_z).value*Mpc*(1 + vec_z)
    D_src = params.d_source*(1 + params.z_source)
    return vec_a, vec_D, D_src, params.D_lens

def C_los(omega, B_tang, mu_t, Delta2_fun, los):
    """C~(omega) of Eq. (C_tilde_1): the angular kernel above times the lensing
    efficiency [(D_S - D)/D_S]^2 and the Poisson factor (4 pi G rho_m,0/a)^2, integrated
    over comoving distance. B_tang is the tangential eigenvalue of B^I, applied as a scalar.

    Off-plane (multi-plane) corrections, notes/off_plane_los_kernel.md: the amplification
    B_offplane(chi) and the sweep lever arm sweep_offplane(chi) both live INSIDE the
    integral -- a perturber away from the lens plane is neither amplified by the full B^I
    (in front) nor swept at chi mu_tilde (behind)."""
    vec_a, vec_D, D_src, D_L = los
    X = sweep_offplane(vec_D, D_L, D_src)
    B_eff = B_offplane(vec_D, D_L, D_src, B_tang)
    Ig = I_M_general(omega, mu_t, X, Delta2_fun)
    integrand = B_eff**2*(1 - vec_D/D_src)**2*Ig*(4*np.pi*G_N*rho_M_0/vec_a)**2
    return 4/omega*np.trapezoid(integrand, vec_D)

# ------------------------------------------------------------------ systems
res = np.load('macro_lens_results.npz')
B_tang_gal = np.max(np.abs(res['eigvals_fit'][1]))     # image B tangential, ~10.2
mu_gal = np.sqrt((B_tang_gal*np.linalg.norm(pB.v_lens_fid)/pB.D_lens)**2 + pB.mu_L_int**2)
B_tang_clu = np.max(np.abs(pJ.eigvals_fit[2]))         # image C tangential, ~22.6
mu_clu = np.sqrt((B_tang_clu*np.linalg.norm(pJ.v_lens_fid)/pJ.D_lens)**2 + pJ.mu_L_int**2)
# worst-case post-subtraction stellar residual (Sec. III C): image-A budget, doubled
# for the differential pair; see impact_numbers.py for its derivation.
s2_star_pair = 2*1.25e-5*muasyy**2

systems = [('B1422 galaxy, EPIC 0.1 muas', pB, B_tang_gal, mu_gal, 0.1*muas, s2_star_pair),
           ('J1029 cluster, 1 muas', pJ, B_tang_clu, mu_clu, 1.0*muas, 0.0)]

for name, par, B_tang, mu_t, sig, s2star in systems:
    los = make_los(par)
    C_noise = sig**2*tau/N_obs
    s2_acc_noise = 720*sig**2/(tau**4*N_obs)
    print('\n===== %s  (B = %.1f, mu_tilde = %.2f muas/yr) =====' % (name, B_tang, mu_t/muasy))
    for lab, D2f in spectra:
        C_sig = C_los(1.001*omega_min, B_tang, mu_t, D2f, los)
        vec_C = np.array([C_los(w, B_tang, mu_t, D2f, los) for w in vec_omega])
        s2_acc = np.trapezoid(2*vec_C*vec_omega**4*smear_F2(vec_omega*tau)/(2*np.pi),
                              vec_omega)
        line = '  %-28s stoch SNR = %8.3g | acc SNR = %8.3g' \
            % (lab, C_sig/C_noise, s2_acc/s2_acc_noise)
        if s2star > 0:
            line += ' (with stars: %.3g)' % (s2_acc/(s2_acc_noise + s2star))
        print(line)

# ============================== kinetic-decoupling sweep of the prompt-cusp plateau
# r_cusp ~ M_cut^(1/3) ~ T_kd^-1 (Eq. Mcut-Tkd), at fixed plateau amplitude: a lower
# decoupling temperature slides the whole plateau LEFT in k, toward the sensitivity
# minimum of Fig. matter_power, without changing its height.
print('\n' + '='*72)
print('kinetic-decoupling sweep, LINE-OF-SIGHT cusps: r_cusp ~ T_kd^-1, plateau fixed')
print('='*72)
print('%7s %11s %11s | %19s | %19s' % ('T_kd', 'M_cut', 'r_cusp', 'B1422 EPIC 0.1 muas',
                                       'J1029 1 muas'))
print('%7s %11s %11s | %9s %9s | %9s %9s' % ('[MeV]', '[M_sun]', '[pc]', 'stoch', 'acc',
                                             'stoch', 'acc'))
for T_kd in [30., 20., 15., 12., 10., 8., 6., 5., 4., 3., 2.]:
    r_c = r_cusp_med*(T_kd/30.)**-1
    D2f = interp_D2(Delta2_cusp(_kk, r_cusp=r_c))
    row = '%7.0f %11.1e %11.1e' % (T_kd, M_cut/M_Solar*(T_kd/30.)**-3, r_c/pc)
    for name, par, B_tang, mu_t, sig, s2star in systems:
        los = make_los(par)
        C_sig = C_los(1.001*omega_min, B_tang, mu_t, D2f, los)
        vec_C = np.array([C_los(w, B_tang, mu_t, D2f, los) for w in vec_omega])
        s2_acc = np.trapezoid(2*vec_C*vec_omega**4*smear_F2(vec_omega*tau)/(2*np.pi),
                              vec_omega)
        row += ' | %9.3g %9.3g' % (C_sig/(sig**2*tau/N_obs),
                                   s2_acc/(720*sig**2/(tau**4*N_obs)))
    print(row)

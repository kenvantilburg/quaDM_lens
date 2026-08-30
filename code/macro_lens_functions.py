"""Macro-lensing geometry, potentials and deflections for Van Tilburg & Kaplan.

DISTANCE CONVENTION. Lowercase `d_*` are angular-diameter distances -- these are the
paper's D_L, D_S, D_LS (Eq. (mu), Sigma_cr, theta_E,L; App. Macrolensing Models).
Uppercase `D_*` are the corresponding comoving distances, D_* = (1+z) d_*, which the
paper writes out explicitly where it needs them (e.g. the (1+z_L) D_L that converts a
transverse velocity at the lens into a proper motion, and the comoving line-of-sight
integral of App. Matter Power Spectrum). The two are kept distinct because mixing them
costs factors of (1+z).

Other symbols follow the paper: theta^I image positions, beta source position, psi the
lensing potential, alpha the reduced deflection, kappa / gamma_1,2 convergence and
shear, J the lensing Jacobian, B^I = (J^I)^-1 its inverse, and A = det B^I the
signed magnification. NB the paper reserves A for the magnification, never the Jacobian.
"""
from natural_units_GeV import *
from preamble import *

##### distances and scales #####
def integrand_d_A(z, Omega_M = 0.3, Omega_Lambda = 0.7):
    """Integrand for the angular diameter distance as a function of redshift 'z' for a flat universe."""
    return  1/np.sqrt(Omega_M*(1+z)**3 + Omega_Lambda)

def d_A(z, H_0 = 70 * km / (second * Mpc), Omega_M = 0.3, Omega_Lambda = 0.7):
    """Angular diameter distance (the paper's D) as a function of redshift 'z', flat universe."""
    return H_0**-1 * (1+z)**-1 * sp.integrate.quad(integrand_d_A, 0, z, args=(Omega_M, Omega_Lambda))[0]
d_A_vectorized = np.vectorize(d_A)

def D_A(z, H_0 = 70 * km / (second * Mpc), Omega_M = 0.3, Omega_Lambda = 0.7):
    """Comoving distance (1+z) d_A(z), flat universe."""
    return d_A(z, H_0, Omega_M, Omega_Lambda) * (1+z)

def mu_rel(v_o, v_l, v_s, z_l, z_s, d_l, d_s, d_ls):
    """Source-plane relative proper motion mu of Eq. (mu), from the transverse peculiar
    velocities of observer (v_o), lens (v_l) and source (v_s). Magnify by B^I to get the
    image proper motion mu_tilde^I = B^I mu of Eq. (mu_tilde)."""
    return v_o / (1 + z_l) * d_ls / (d_l * d_s) + v_s / (1 + z_s) / d_s - v_l / (1 + z_l) / d_l

def Sigma_crit(d_l,d_s,d_ls):
    """Critical surface mass density Sigma_cr = (4 pi G D_L)^-1 D_S/D_LS (Sec. II C)"""
    return 1 / (4*np.pi*G_N) * d_s / (d_l * d_ls)

def theta_E(M_L, d_l, d_s, d_ls):
    """Angular Einstein radius theta_E,L of a lens of mass M_L (Sec. II C)"""
    return np.sqrt(4 * G_N * M_L * d_ls / (d_l * d_s))

##### Light profiles #####
# Stellar component of the lens galaxy, used for kappa_* in Tab. tab:macro (App. B1422+231).
def I_DV(R,I_eff=1,R_eff=1):
    """De Vaucouleurs surface brightness, I(R_eff) = I_eff (Sersic n = 4)"""
    return I_eff * np.exp(-7.669 * ((R/R_eff)**(1/4) - 1) )

vec_R = np.logspace(-4,3,int(1e4))
vec_I = I_DV(vec_R)
vec_R_half = vec_R[vec_R < 1]
sum_DV_tot = np.sum((vec_R[1:] - vec_R[:-1]) * 2*np.pi * vec_R[:-1] * I_DV(vec_R[:-1]))
sum_DV_half = np.sum((vec_R_half[1:] - vec_R_half[:-1]) * 2*np.pi * vec_R_half[:-1] * I_DV(vec_R_half[:-1]))

def kappa_DV(x,y,ell,theta_eff,PA_major,M_Sal,d_l,d_s,d_ls):
    """Stellar convergence kappa_* of a de Vaucouleurs profile at image-plane (x,y).

    Offsets x,y are measured from the galaxy center in arcsec. Shape parameters:
    ellipticity 'ell' (axis ratio q = 1 - ell), major-axis position angle 'PA_major',
    effective (half-light) radius 'theta_eff' in arcsec (the geometric mean of the
    major and minor effective radii, as in Sluse+ 2012). 'M_Sal' is the total stellar
    mass for a Salpeter IMF; the profile is normalized to it via sum_DV_tot.

    CONVENTION: 'PA_major' is the astronomical position angle, EAST OF NORTH. With
    x = Delta RA (East-positive) and y = Delta Dec, the rotation below combined with
    the 1/q weighting of x_p places the major axis along (sin PA_major, cos PA_major),
    which is exactly the E-of-N direction. The two apparently-inverted pieces (the
    rotation sense, and the fact that y_p rather than x_p is the major axis) cancel;
    do not "fix" either one in isolation. Same convention in kappa_SIE/kappa_SIE_f.
    NB lenstronomy instead measures its angle counterclockwise from +x: phi = 90 deg
    - PA_major.
    """
    kappa_eff = M_Sal / (theta_eff * arcsec * d_l)**2 / Sigma_crit(d_l,d_s,d_ls) # arcsec to correct for units
    x_p = x * np.cos(PA_major) - y * np.sin(PA_major)
    y_p = x * np.sin(PA_major) + y * np.cos(PA_major)
    q = 1 - ell
    arg = (np.sqrt((x_p**2 / q + y_p**2 * q))/theta_eff)**(1/4)
    return sum_DV_tot**-1 * kappa_eff * np.exp(-7.669 * (arg - 1)) 

##### lensing potentials #####
# Analytic potentials psi(theta), with beta = theta - grad psi. The SIS/SIE forms are
# reference implementations; the B1422+231 macro-model itself is fit with lenstronomy
# (macro_lens.ipynb). Angles are in arcsec unless a distance argument is present.
def psi_smooth(kappa_0, gamma_1, gamma_2,arr_theta):
    """Local expansion of the macro-lens potential: constant convergence plus shear.

    Reproduces a uniform lensing Jacobian J = [[1-k-g1, -g2], [-g2, 1-k+g1]] about the
    image position -- the macro-model input B^I = (J^I)^-1 of the microlensing maps.
    """
    arr_psi_kappa = (kappa_0/2) * np.linalg.norm(arr_theta,axis=0)**2
    arr_psi_gamma_1 = (gamma_1/2) * (arr_theta[0]**2 - arr_theta[1]**2) 
    arr_psi_gamma_2 = gamma_2 * arr_theta[0] * arr_theta[1]
    return arr_psi_kappa + arr_psi_gamma_1 + arr_psi_gamma_2

def kappa_SIS(theta,theta_E):
    """Convergence of a singular isothermal sphere of Einstein radius theta_E"""
    return theta_E / (2 * theta)

def kappa_SIE(x,y,theta_E,ell,PA_major):
    """Convergence of a singular isothermal ellipsoid, ellipticity ell (axis ratio 1-ell).

    'PA_major' is the major-axis position angle east of north; see kappa_DV."""
    x_p = x * np.cos(PA_major) - y * np.sin(PA_major)
    y_p = x * np.sin(PA_major) + y * np.cos(PA_major)
    q = 1 - ell # axis ratio
    return theta_E / (2 * np.sqrt(q * y_p**2 + x_p**2 / q))

def kappa_SIE_f(x,y,theta_E,f,PA_major):
    """Convergence of an SIE parametrized by the axis ratio f = 1 - ell directly.

    'PA_major' is the major-axis position angle east of north; see kappa_DV."""
    x_p = x * np.cos(PA_major) - y * np.sin(PA_major)
    y_p = x * np.sin(PA_major) + y * np.cos(PA_major)
    Delta = np.sqrt(x_p**2 + f**2 * y_p**2)
    return theta_E * np.sqrt(f) / (2 * Delta)

def psi_SIS(theta,theta_E):
    """Lensing potential of a singular isothermal sphere"""
    return theta_E * theta

def psi_SIE(x,y,theta_E,ell,PA_major):
    """Lensing potential of an SIE (Kormann+ 1994), ellipticity ell"""
    x_p = x * np.cos(PA_major) - y * np.sin(PA_major)
    y_p = x * np.sin(PA_major) + y * np.cos(PA_major)
    q = 1 - ell # axis ratio
    p = np.sqrt(q**2 * y_p**2 + x_p**2)
    return theta_E * np.sqrt(q) / np.sqrt(1-q**2) * (y_p * np.arctan(np.sqrt(1-q**2) * y_p / p) + x_p * np.arctanh(np.sqrt(1-q**2) * x_p / p))

def psi_SIE_f(x,y,theta_E,f,PA_major):
    """Lensing potential of an SIE parametrized by the axis ratio f = 1 - ell"""
    x_p = x * np.cos(PA_major) - y * np.sin(PA_major)
    y_p = x * np.sin(PA_major) + y * np.cos(PA_major)
    f_p = np.sqrt(1-f**2)
    x_abs = np.sqrt(x_p**2 + y_p**2)
    cos_phi = x_p / x_abs
    sin_phi = y_p / x_abs
    return theta_E * np.sqrt(f)/f_p * (sin_phi * np.arcsin(f_p * sin_phi) + cos_phi * np.arcsinh(f_p / f * cos_phi))

def psi_point(M_L,d_l,d_s,d_ls,arr_theta):
    """Lensing potential theta_E,L^2 ln(theta) of a point mass M_L (a microlensing star).

    Regularized at theta below one grid step to keep the FFT maps finite.
    """
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    theta_E_L = theta_E(M_L, d_l,d_s,d_ls)
    theta = np.linalg.norm(arr_theta,axis=0)
    theta[theta<theta_step] = theta_step
    return theta_E_L**2 * np.log(theta)

def psi_star(kappa_star,M_star,d_l,d_s,d_ls,arr_theta,theta_max,N_step):
    """Potential of a Poisson realization of stars of mass M_star at convergence kappa_*.

    Uniform positions in the (2 theta_max)^2 field, monochromatic mass function; returns
    the summed potential on the theta grid and the drawn star positions. A direct
    realization of the stellar background of Sec. III C, for illustration only: the
    forecasts never use it, because a point-lens field is neither Gaussian nor
    perturbative and is handled star by star in the companion paper instead.
    """
    vec_delta_theta = np.linspace(-2*theta_max, 2*theta_max, 2*N_step+1)
    arr_delta_theta = np.asarray(np.meshgrid(vec_delta_theta, vec_delta_theta, indexing='ij'))
    M_tot_star = kappa_star * Sigma_crit(d_l,d_s,d_ls) * (2 * theta_max * d_l)**2
    N_star = np.random.poisson(M_tot_star / M_star)
    arr_rand_idx = np.random.randint(low=0, high=N_step+1, size=(N_star,2))

    arr_psi_star = np.zeros((N_step+1, N_step+1))
    arr_theta_star = np.zeros((N_star,2))
    arr_psi_star_delta = psi_point(M_star,d_l,d_s,d_ls,arr_delta_theta)
    for i in range(N_star):
        idx = arr_rand_idx[i]
        arr_theta_star[i] = arr_theta[:,-idx[0]-1,-idx[1]-1]
        arr_psi_star += arr_psi_star_delta[idx[0]:idx[0]+N_step+1,idx[1]:idx[1]+N_step+1]
    return arr_psi_star, arr_theta_star

##### lensing deflection #####
def alpha_from_psi(arr_psi,arr_theta):
    """Reduced deflection alpha_i = d psi / d theta_i, by finite differences on the grid"""
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    return np.asarray(np.gradient(arr_psi,theta_step))

def J_from_alpha(arr_alpha,arr_theta):
    """Lensing Jacobian J_ij = delta_ij - d alpha_i / d theta_j (so B^I = (J^I)^-1 and
    the signed magnification is A^I = det B^I)"""
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    arr_d_alpha =  np.asarray([np.gradient(arr_alpha[0],theta_step),
                           np.gradient(arr_alpha[1],theta_step)])
    arr_identity = np.asarray([[np.ones_like(arr_theta[0]),np.zeros_like(arr_theta[0])],
                               [np.zeros_like(arr_theta[0]),np.ones_like(arr_theta[0])]])
    return arr_identity - arr_d_alpha


##### source intensities #####
def source_intensity_uniform_disk(arr_beta, beta_s):
    """Unit-normalized surface brightness of a uniform source disk of angular radius beta_s"""
    arr_beta_norm = np.linalg.norm(arr_beta, axis=0)
    area = np.pi * beta_s**2
    source_intensity = np.where(arr_beta_norm < beta_s, 1/area, 0)
    return  source_intensity
from natural_units_GeV import *
from preamble import *
from macro_lens_functions import *

##### Linear power spectrum #####


##### DM profiles #####
def rho_Einasto(r, rho_s, r_s, alpha=0.16):
    """Mass density of Einasto profile"""
    return rho_s * np.exp(-2./alpha * ((r/r_s)**alpha - 1))

def M_enc_Einasto(r, rho_s, r_s, alpha=0.16):
    """Mass enclosed within radius r of Einasto profile"""
    prefac = rho_s * r_s**3 * 2**(2-3/alpha) * np.exp(2/alpha) * np.pi * alpha**(-1+3/alpha)
    return prefac * sp.special.gamma(3/alpha) * sp.special.gammainc(3/alpha, 2./alpha * (r/r_s)**alpha)

def M_Einasto(rho_s, r_s, alpha=0.16):
    """Mass of Einasto profile"""
    prefac = rho_s * r_s**3 * 2**(2-3/alpha) * np.exp(2/alpha) * np.pi * alpha**(-1+3/alpha)
    return prefac * sp.special.gamma(3/alpha)

def c_Einasto(M_200,h=0.7,M_fs = 1e-20 * M_Solar):
    c_Einasto_coeff = np.asarray([27.112, -0.381, -1.853e-3, -4.141e-4, -4.334e-6, 3.208e-7, -0.529])
    x = np.log(M_200 / (h**-1 * M_Solar))
    pol = 0
    for i in range(6):
        pol += c_Einasto_coeff[i] * x**i
    return np.exp(c_Einasto_coeff[6] * (M_fs/M_200)**(1/3)) * pol

def r_200(M_200,h=0.7):
    """Radius of sphere with mean density 200 times the critical density"""
    return (3 * M_200 / (4 * np.pi * 200 * (h/0.7)**2 * rho_crit))**(1/3)

def r_s_Einasto_200(M_200, h=0.7, M_fs = 1e-20 * M_Solar):
    """Scale radius of Einasto profile"""
    return r_200(M_200) / c_Einasto(M_200, h, M_fs)

vec_M_200 = np.logspace(-12,12,1000) * M_Solar
vec_r_200 = r_200(vec_M_200)
vec_r_s_Einasto = r_s_Einasto_200(vec_M_200)
vec_M_enc = M_enc_Einasto(vec_r_200, M_Solar / pc**3, vec_r_s_Einasto)
int_rho_s_Einasto_200 = interp1d(vec_M_200, vec_M_200/vec_M_enc * M_Solar / pc**3,
                                 bounds_error=False, fill_value=0) 
def rho_s_Einasto_200(M_200):
    """Scale density of Einasto profile"""
    return int_rho_s_Einasto_200(M_200)

vec_x_b = np.logspace(-4,4,int(1e3))
vec_Sigma_x_b = np.zeros_like(vec_x_b)
for i,x_b in enumerate(vec_x_b):
    vec_Sigma_x_b[i] = quad(lambda z: 2*rho_Einasto(np.sqrt(x_b**2 + z**2), 1, 1), 0, 1e4)[0]
int_Sigma_x_b = interp1d(vec_x_b, vec_Sigma_x_b, bounds_error=False, fill_value=0) # scale-free surface mass density

vec_m_x_b = odeint(lambda m,x_b : 2 * np.pi * x_b * int_Sigma_x_b(x_b), 0, vec_x_b)[:,0] # scale-free mass enclosed within x_b
int_m_x_b = interp1d(vec_x_b, vec_m_x_b, bounds_error=False, fill_value=(0,vec_m_x_b[-1])) # scale-free mass enclosed within x_b

int_alpha_x_b = interp1d(vec_x_b, vec_m_x_b / (np.pi * vec_x_b), bounds_error=False, fill_value=0) # scale-free deflection angle

def Sigma_Einasto(b, rho_s, r_s):
    """Surface mass density of Einasto profile at alpha = 0.16"""
    return rho_s * r_s * int_Sigma_x_b(b/r_s)

def Sigma_Einasto_200(b, M_200):
    """Surface mass density of Einasto profile at alpha = 0.16"""
    r_s = r_s_Einasto_200(M_200)
    rho_s = rho_s_Einasto_200(M_200)
    return Sigma_Einasto(b, rho_s, r_s)

def alpha_Einasto(rho_s, r_s, d_l, d_s, d_ls, arr_theta):
    """Deflection angle of Einasto profile at alpha = 0.16"""
    theta = np.linalg.norm(arr_theta,axis=0)
    return rho_s * r_s**2 / d_l * int_alpha_x_b(theta*d_l/r_s) / Sigma_crit(d_l,d_s,d_ls) * arr_theta / (theta + 1e-30) # 1e-30 to avoid division by zero

def alpha_Einasto_200(M_200, d_l, d_s, d_ls, arr_theta):
    """Deflection angle of Einasto profile at alpha = 0.16"""
    r_s = r_s_Einasto_200(M_200)
    rho_s = rho_s_Einasto_200(M_200)
    return alpha_Einasto(rho_s, r_s, d_l, d_s, d_ls, arr_theta)

# ##### DM Fourier transforms #####
# vec_x_b = np.logspace(-4,4,int(1e3))
# vec_x_k = np.logspace(-3,6,int(5e2))
# vec_m_x_b = int_m_x_b(vec_x_b)
# vec_F_x_k = np.zeros_like(vec_x_k)

# for i_k, x_k in enumerate(tqdm(vec_x_k)):
#     a = np.min([1e-4, 1e-4 * x_k])
#     b = np.max([1e3])
#     vec_F_x_k[i_k] = quad(lambda x_b: sp.special.jv(1,x_b) * int_m_x_b(x_b / x_k) / vec_m_x_b[-1],
#                           a=a, b=b,
#                           epsabs=1e-7,epsrel=1e-5,limit=2000)[0]
# # form factor for Einasto profile
# int_F_x_k = interp1d(vec_x_k, vec_F_x_k, bounds_error=False, fill_value='extrapolate') 

# ##### Analytic power spectrum
# vec_x_k = np.logspace(-5,5,int(1e3))
# vec_F_x_k_integral_d = np.zeros_like(vec_x_k)
# vec_F_x_k_integral_c = np.zeros_like(vec_x_k)
# epsabs = 1e-7
# epsrel = 1e-5
# limit = 1000
# for i_k, x_k in enumerate(tqdm(vec_x_k)):
#     vec_F_x_k_integral_d[i_k] = quad(lambda phi: (int_F_x_k(x_k / np.cos(phi)))**2,
#                           a=-np.pi/2, b=np.pi/2,
#                           epsabs=epsabs,epsrel=epsrel,limit=limit)[0]
#     vec_F_x_k_integral_c[i_k] = quad(lambda phi: np.cos(2*phi) * (int_F_x_k(x_k / np.cos(phi)))**2,
#                           a=-np.pi/2, b=np.pi/2,
#                           epsabs=epsabs,epsrel=epsrel,limit=limit)[0]
# int_F_x_k_integral_d = interp1d(vec_x_k, vec_F_x_k_integral_d, bounds_error=False, fill_value=(vec_F_x_k_integral_d[0],0)) # integral of F^2 over phi
# int_F_x_k_integral_c = interp1d(vec_x_k, vec_F_x_k_integral_c, bounds_error=False, fill_value=(0,0)) # integral of F^2 * cos(2*phi) over phi

def F_G_cusp(k):
    """Form factor for gaussian profile with 1/r cusp"""
    return np.exp(-k**2/2)

def F_G_cusp_integral_d(x_k):
    """Diagonal piece of F^2 integral"""
    return np.pi * sp.special.erfc(x_k)

def F_G_cusp_integral_c(x_k):
    """Cosine piece of F^2 cos(2phi) integral. Form is approximate but works to 5%"""
    return (6/5) / (1 + 1/x_k) * np.pi * sp.special.erfc(x_k)

def C_ij_integral(x_k, zeta):
    """Integral for analytic power spectrum"""
    int_d = F_G_cusp_integral_d(x_k)
    int_c = F_G_cusp_integral_c(x_k)
    return np.asarray([[int_d + np.cos(2*zeta) * int_c, np.sin(2*zeta) * int_c],
                        [np.sin(2*zeta) * int_c, int_d - np.cos(2*zeta) * int_c]])

def C_ij_integral_src(x_k, x_src, zeta):
    """C_ij_integral including a finite-source form factor.

    The source form factor |W~|^2 is the squared Fourier transform of the image
    surface-brightness profile (the effective quasar form factor W of Galanis+ 2023,
    arXiv:2307.06989). For a Gaussian source surface-brightness profile and the
    Gaussian-cutoff cusp halo F(k)=exp(-k^2/2), it multiplies the
    phi-integrand with the same 1/cos(phi) structure as the halo form factor,
    so the two add in quadrature inside the closed-form integral:
        int dphi F[x_k/c_phi]^2 |W~[x_src/c_phi]|^2 = pi*erfc(sqrt(x_k^2+x_src^2)).
    Here x_k = omega*gamma_L/mu_tilde and x_src = omega*theta_src^I/mu_tilde, with
    theta_src^I the (magnified) image source size. Point source: x_src=0.
    """
    return C_ij_integral(np.sqrt(x_k**2 + x_src**2), zeta)

def C_EPIC(omega, t_int=10*year, N_obs=100, sigma_delta_theta=muas):
    """Noise power spectrum of EPIC"""
    return t_int / N_obs * sigma_delta_theta**2 / (1e-50 + np.heaviside(omega  * t_int / (2*np.pi) - 1,0))


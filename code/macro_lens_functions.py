from natural_units_GeV import *
from preamble import *

##### distances and scales #####
def integrand_d_A(z, Omega_M = 0.3, Omega_Lambda = 0.7):
    """Integrand for the angular diameter distance as a function of redshift 'z' for a flat universe."""
    return  1/np.sqrt(Omega_M*(1+z)**3 + Omega_Lambda)

def d_A(z, H_0 = 70 * km / (second * Mpc), Omega_M = 0.3, Omega_Lambda = 0.7):
    """Angular diameter distance as a function of redshift 'z' for a flat universe."""
    return H_0**-1 * (1+z)**-1 * sp.integrate.quad(integrand_d_A, 0, z, args=(Omega_M, Omega_Lambda))[0]
d_A_vectorized = np.vectorize(d_A)

def D_A(z, H_0 = 70 * km / (second * Mpc), Omega_M = 0.3, Omega_Lambda = 0.7):
    """Comoving angular diameter distance as a function of redshift 'z' for a flat universe."""
    return d_A(z, H_0, Omega_M, Omega_Lambda) * (1+z)

def mu_rel(v_o, v_l, v_s, z_l, z_s, d_l, d_s, d_ls):
    """Proper motion of the source relative to the lens."""
    return v_o / (1 + z_l) * d_ls / (d_l * d_s) + v_s / (1 + z_s) / d_s - v_l / (1 + z_l) / d_l

def Sigma_crit(d_l,d_s,d_ls):
    """Critical surface mass density"""
    return 1 / (4*np.pi*G_N) * d_s / (d_l * d_ls)

def theta_E(M_l, d_l, d_s, d_ls):
    """Einstein radius"""
    return np.sqrt(4 * G_N * M_l * d_ls / (d_l * d_s))

##### Light profiles #####
def I_DV(R,I_eff=1,R_eff=1):
    """De Vaucouleurs light profile"""
    return I_eff * np.exp(-7.669 * ((R/R_eff)**(1/4) - 1) )

vec_R = np.logspace(-4,3,int(1e4))
vec_I = I_DV(vec_R)
vec_R_half = vec_R[vec_R < 1]
sum_DV_tot = np.sum((vec_R[1:] - vec_R[:-1]) * 2*np.pi * vec_R[:-1] * I_DV(vec_R[:-1]))
sum_DV_half = np.sum((vec_R_half[1:] - vec_R_half[:-1]) * 2*np.pi * vec_R_half[:-1] * I_DV(vec_R_half[:-1]))

def kappa_DV(x,y,ell,theta_eff,theta_e,M_Sal,d_l,d_s,d_ls):
    """De Vaucouleurs stellar convergence in x,y coordinates (image plane) 
    with eccentricity 'ecc' and rotation angle 'theta_e', effective radius 'theta_eff',
    total stellar mass M_Sal, lens and source distances d_l and d_s"""
    kappa_eff = M_Sal / (theta_eff * arcsec * d_l)**2 / Sigma_crit(d_l,d_s,d_ls) # arcsec to correct for units
    x_p = x * np.cos(theta_e) - y * np.sin(theta_e)
    y_p = x * np.sin(theta_e) + y * np.cos(theta_e)
    q = 1 - ell
    arg = (np.sqrt((x_p**2 / q + y_p**2 * q))/theta_eff)**(1/4)
    return sum_DV_tot**-1 * kappa_eff * np.exp(-7.669 * (arg - 1)) 

##### lensing potentials #####
def psi_smooth(kappa_0, gamma_1, gamma_2,arr_theta):
    """Smooth lensing potential from constant convergence and shear"""
    arr_psi_kappa = (kappa_0/2) * np.linalg.norm(arr_theta,axis=0)**2
    arr_psi_gamma_1 = (gamma_1/2) * (arr_theta[0]**2 - arr_theta[1]**2) 
    arr_psi_gamma_2 = gamma_2 * arr_theta[0] * arr_theta[1]
    return arr_psi_kappa + arr_psi_gamma_1 + arr_psi_gamma_2

def kappa_SIS(theta,theta_E):
    """Convergence of SIS"""
    return theta_E / (2 * theta)

def kappa_SIE(x,y,theta_E,ell,theta_e):
    """Convergence of SIE"""
    x_p = x * np.cos(theta_e) - y * np.sin(theta_e)
    y_p = x * np.sin(theta_e) + y * np.cos(theta_e)
    q = 1 - ell # axis ratio
    return theta_E / (2 * np.sqrt(q * y_p**2 + x_p**2 / q))

def kappa_SIE_f(x,y,theta_E,f,theta_e):
    """Convergence of SIE"""
    x_p = x * np.cos(theta_e) - y * np.sin(theta_e)
    y_p = x * np.sin(theta_e) + y * np.cos(theta_e)
    Delta = np.sqrt(x_p**2 + f**2 * y_p**2)
    return theta_E * np.sqrt(f) / (2 * Delta)

def psi_SIS(theta,theta_E):
    """Lensing potential of SIS"""
    return theta_E * theta

def psi_SIE(x,y,theta_E,ell,theta_e):
    """Lensing potential of SIE"""
    x_p = x * np.cos(theta_e) - y * np.sin(theta_e)
    y_p = x * np.sin(theta_e) + y * np.cos(theta_e)
    q = 1 - ell # axis ratio
    p = np.sqrt(q**2 * y_p**2 + x_p**2)
    return theta_E * np.sqrt(q) / np.sqrt(1-q**2) * (y_p * np.arctan(np.sqrt(1-q**2) * y_p / p) + x_p * np.arctanh(np.sqrt(1-q**2) * x_p / p))

def psi_SIE_f(x,y,theta_E,f,theta_e):
    """Lensing potential of SIE"""
    x_p = x * np.cos(theta_e) - y * np.sin(theta_e)
    y_p = x * np.sin(theta_e) + y * np.cos(theta_e)
    f_p = np.sqrt(1-f**2)
    x_abs = np.sqrt(x_p**2 + y_p**2)
    cos_phi = x_p / x_abs
    sin_phi = y_p / x_abs
    return theta_E * np.sqrt(f)/f_p * (sin_phi * np.arcsin(f_p * sin_phi) + cos_phi * np.arcsinh(f_p / f * cos_phi))

def psi_point(M_l,D_l,D_s,arr_theta):
    """Lensing potential of point mass"""
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    theta_E_l = theta_Ein(M_l, D_l, D_s)
    theta = np.linalg.norm(arr_theta,axis=0)
    theta[theta<theta_step] = theta_step
    return theta_E_l**2 * np.log(theta)

def psi_star(kappa_star,M_star,D_l,D_s,arr_theta,theta_max,N_step):
    """Lensing potential of a collection of point masses with uniform spatial distribution and monochromatic mass function"""
    vec_delta_theta = np.linspace(-2*theta_max, 2*theta_max, 2*N_step+1)
    arr_delta_theta = np.asarray(np.meshgrid(vec_delta_theta, vec_delta_theta, indexing='ij'))
    M_tot_star = kappa_star * Sigma_crit(D_l,D_s) * (2 * theta_max * D_l)**2
    N_star = np.random.poisson(M_tot_star / M_star)
    arr_rand_idx = np.random.randint(low=0, high=N_step+1, size=(N_star,2))

    arr_psi_star = np.zeros((N_step+1, N_step+1))
    arr_theta_star = np.zeros((N_star,2))
    arr_psi_star_delta = psi_point(M_star,D_l,D_s,arr_delta_theta)
    for i in range(N_star):
        idx = arr_rand_idx[i]
        arr_theta_star[i] = arr_theta[:,-idx[0]-1,-idx[1]-1]
        arr_psi_star += arr_psi_star_delta[idx[0]:idx[0]+N_step+1,idx[1]:idx[1]+N_step+1]
    return arr_psi_star, arr_theta_star

def psi_NFW(rho_s,r_s,D_l,D_s,arr_theta):
    """Lensing potential of NFW halo"""
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    kappa_s = rho_s * r_s / Sigma_crit(D_l,D_s)
    x = np.linalg.norm(arr_theta,axis=0) / (r_s / D_l)
    x[x<theta_step / (r_s / D_l)] = theta_step / (r_s / D_l)
    return np.real((r_s/D_l)**2 * 4 * kappa_s * ((1/2) * (np.log(x/2+0j))**2 - 2 * (np.arctanh(np.sqrt((1-x)/(1+x)+0j)))**2 ) )

# def alpha_DM_Einasto(kappa_DM,rho_s,r_s,D_l,D_s,arr_theta,theta_max,N_step):
#     """Lensing deflection of a collection of DM Einasto halos with uniform spatial distribution and monochromatic mass function"""
#     M_DM = M_Einasto(rho_s,r_s)
#     vec_delta_theta = np.linspace(-2*theta_max, 2*theta_max, 2*N_step+1)
#     arr_delta_theta = np.asarray(np.meshgrid(vec_delta_theta, vec_delta_theta, indexing='ij'))
#     M_tot_DM = kappa_DM * Sigma_crit(D_l,D_s) * (2 * theta_max * D_l)**2
#     N_DM = np.random.poisson(M_tot_DM / M_DM)
#     arr_rand_idx = np.random.randint(low=0, high=N_step+1, size=(N_DM,2))

#     arr_alpha_DM = np.zeros((2,N_step+1, N_step+1))
#     arr_kappa_DM = np.zeros((N_step+1, N_step+1))
#     arr_theta_DM = np.zeros((N_DM,2))
#     arr_alpha_DM_delta = alpha_Einasto(rho_s,r_s,D_l,D_s,arr_delta_theta)
#     arr_kappa_DM_delta = Sigma_Einasto(D_l * np.linalg.norm(arr_delta_theta,axis=0), rho_s, r_s) / Sigma_crit(D_l,D_s)
#     for i in tqdm(range(N_DM)):
#         idx = arr_rand_idx[i]
#         arr_theta_DM[i] = arr_theta[:,-idx[0]-1,-idx[1]-1]
#         arr_alpha_DM += arr_alpha_DM_delta[:,idx[0]:idx[0]+N_step+1,idx[1]:idx[1]+N_step+1]
#         arr_kappa_DM += arr_kappa_DM_delta[idx[0]:idx[0]+N_step+1,idx[1]:idx[1]+N_step+1]
#     return arr_alpha_DM, arr_kappa_DM, arr_theta_DM

def psi_DM_NFW(kappa_s,rho_s,r_s,D_l,D_s,arr_theta,theta_max,N_step):
    """Lensing potential of a collection of DM NFW halos with uniform spatial distribution and monochromatic mass function"""
    M_s = 2 * np.pi * (np.log(4) - 1) * rho_s * r_s**3
    vec_delta_theta = np.linspace(-2*theta_max, 2*theta_max, 2*N_step+1)
    arr_delta_theta = np.asarray(np.meshgrid(vec_delta_theta, vec_delta_theta, indexing='ij'))
    M_tot_s = kappa_s * Sigma_crit(D_l,D_s) * (2 * theta_max * D_l)**2
    N_DM = np.random.poisson(M_tot_s / M_s)
    arr_rand_idx = np.random.randint(low=0, high=N_step+1, size=(N_DM,2))

    arr_psi_DM = np.zeros((N_step+1, N_step+1))
    arr_theta_DM = np.zeros((N_DM,2))
    arr_psi_DM_delta = psi_NFW(rho_s,r_s,D_l,D_s,arr_delta_theta)
    for i in tqdm(range(N_DM)):
        idx = arr_rand_idx[i]
        arr_theta_DM[i] = arr_theta[:,-idx[0]-1,-idx[1]-1]
        arr_psi_DM += arr_psi_DM_delta[idx[0]:idx[0]+N_step+1,idx[1]:idx[1]+N_step+1]
    return arr_psi_DM, arr_theta_DM

##### lensing deflection #####
def alpha_from_psi(arr_psi,arr_theta):
    """Reduced deflection angle from lensing potential"""
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    return np.asarray(np.gradient(arr_psi,theta_step))

def A_from_alpha(arr_alpha,arr_theta):
    """Jacobian of lensing map from deflection angle"""
    theta_step = arr_theta[0,1,0] - arr_theta[0,0,0]
    arr_d_A =  np.asarray([np.gradient(arr_alpha[0],theta_step),
                           np.gradient(arr_alpha[1],theta_step)])
    arr_identity = np.asarray([[np.ones_like(arr_theta[0]),np.zeros_like(arr_theta[0])],
                               [np.zeros_like(arr_theta[0]),np.ones_like(arr_theta[0])]])
    return arr_identity - arr_d_A


##### source intensities #####
def source_intensity_uniform_disk(arr_beta, beta_s):
    """Fractional flux distribution for a uniform disk on the source plane"""
    arr_beta_norm = np.linalg.norm(arr_beta, axis=0)
    area = np.pi * beta_s**2
    source_intensity = np.where(arr_beta_norm < beta_s, 1/area, 0)
    return  source_intensity
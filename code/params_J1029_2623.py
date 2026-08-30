"""Observed inputs for the cluster lens SDSS J1029+2623 (App. SDSS J1029+2623).

A three-image quasar split by 22.5" by a foreground galaxy CLUSTER -- the cluster-lens
analogue of the galaxy-lensed B1422+231 in params_B1422_231.py, feeding sensitivity.ipynb
through the same interface. No lens model is refit here: the convergence and shear at the
three images are taken directly from the cluster-scale reconstruction of Acebron+ 2022,
evaluated at the quasar redshift. Distances follow the macro_lens_functions convention
(lowercase d_* angular-diameter = the paper's D_L, D_S, D_LS; uppercase D_* comoving).
"""
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import pandas as pd
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# redshifts and distances (d_L ~ 1.37, d_S ~ 1.71, d_LS ~ 1.03 Gpc)
z_lens = 0.588      # cluster redshift
z_source = 2.1992   # quasar redshift
d_lens = cosmo.angular_diameter_distance(z_lens).value * Mpc
d_source = cosmo.angular_diameter_distance(z_source).value * Mpc
d_lens_source = cosmo.angular_diameter_distance(z_lens, z_source).value * Mpc
D_lens = d_lens * (1 + z_lens)
D_source = d_source * (1 + z_source)
D_lens_source = D_source - D_lens
npt.assert_almost_equal(d_lens/Mpc, d_A(z_lens)/Mpc, decimal=10) # consistency with manual calc
npt.assert_almost_equal(d_source/Mpc, d_A(z_source)/Mpc, decimal=10)
Sigma_crit_val = Sigma_crit(d_lens, d_source, d_lens_source)
length_per_arcsec = d_lens * arcsec # transverse physical length subtended by 1" at the lens

# observed (*_o) and model-predicted (*_p) image positions [deg] and the best-fit
# convergence/shear at the QSO redshift (Acebron+2022). We use the observed rows.
data = {
    "ID": ["A_o", "B_o", "C_o", "A_p", "B_p", "C_p"],
    "RA": [157.3081015, 157.3093576, 157.3095761, 157.3080822, 157.3093822, 157.3095682],
    "Dec": [26.3883036, 26.3944634, 26.3939843, 26.3882944, 26.3944311, 26.3939969],
    "kappa": [0.4990, 0.4587, 0.5053, 0.4990, 0.4623, 0.5090],
    "gamma1": [0.0702, -0.3324, -0.4406, 0.0702, -0.3418, -0.4415],
    "gamma2": [-0.3085, 0.3692, 0.3103, -0.3085, 0.3654, 0.3202],
}
df = pd.DataFrame(data)

labels = ['A', 'B', 'C']
kappa_fit = df['kappa'].to_numpy()[:3]
gamma1_fit = df['gamma1'].to_numpy()[:3]
gamma2_fit = df['gamma2'].to_numpy()[:3]

# image positions (RA, Dec) in arcsec offsets from image A, with the RA axis on the sky
coord = SkyCoord(df['RA'].to_numpy()[:3], df['Dec'].to_numpy()[:3], unit='deg', frame='icrs')
cos_dec = np.cos(np.deg2rad(df['Dec'].to_numpy()[0]))
theta = np.zeros((3, 2))
theta[:, 0] = (df['RA'].to_numpy()[:3] - df['RA'].to_numpy()[0]) * 3600 * cos_dec  # arcsec
theta[:, 1] = (df['Dec'].to_numpy()[:3] - df['Dec'].to_numpy()[0]) * 3600          # arcsec

# Lensing Jacobian J = I - H and its inverse B^I = (J^I)^-1, in the convention of
#   J = [[1-kappa-gamma1, -gamma2], [-gamma2, 1-kappa+gamma1]],
# matching macro_lens.ipynb and App. Macrolensing Models. mag_fit is the signed
# magnification A^I = det B^I of Tab. tab:macro-cluster (the paper's A, not mu; A is
# never the Jacobian in this work).
jacobian_fit = np.zeros((3, 2, 2))
for i in range(3):
    jacobian_fit[i] = [[1 - kappa_fit[i] - gamma1_fit[i], -gamma2_fit[i]],
                       [-gamma2_fit[i], 1 - kappa_fit[i] + gamma1_fit[i]]]
inv_jacobian_fit = np.linalg.inv(jacobian_fit)
mag_fit = 1 / np.linalg.det(jacobian_fit)
eig_fit = np.linalg.eig(inv_jacobian_fit)
eigvals_fit = eig_fit[0]
eigvecs_fit = eig_fit[1]

# ---- velocity budget of Eq. (mu) ("Expected motion" paragraph of Sec. III D and
# "Velocity scale" of App. SDSS J1029+2623; computed in mu_expected.py, CMB frame) ----
# observer v_o: CMB dipole transverse component at this LOS, 212 km/s (apex sep. 34.9 deg).
# lens v_L:  bulk peculiar velocity of the cluster, ~300 km/s per axis (linear theory).
#            The cluster carries its subhalos with it, so this is a BULK velocity.
# source v_S: quasar host peculiar velocity ~300 km/s (1D).
# Together: bulk-only <|mu|^2>^{1/2} = 0.046 muas/yr -> equivalent bulk transverse
# velocity (1+z_L) d_L <|mu|^2>^{1/2} = 474 km/s, giving mu_tilde^{B,C} ~ 1.0 muas/yr.
#
# KINEMATICS. The sweep rate of image I across a given subhalo's deflection field is
# d/dt[theta^I - theta_L] = B^I mu - mu_L. Only the macro-image motion is magnified by
# B^I; a microhalo's own orbital motion in the cluster (sigma_v ~ 1000 km/s per axis, set
# by the 22.5" splitting, theta_E ~ 11-18") displaces the DEFLECTOR, not the image, and
# enters UNMAGNIFIED:
#   mu_sweep = sqrt(|B^I mu|^2 + 2 (sigma_v/[(1+z_L) d_L])^2),
# a ~1% quadrature correction for the bright images. NB adding sigma_v to the velocity
# BEFORE magnifying (v = sqrt(474^2 + 2*1000^2) ~ 1490 km/s) would boost the microhalos'
# own orbital motion by B ~ 22, overestimating the sweep rate ~3x and the acceleration
# variance by ~30-100x.
#
# MOVING-CLUMP TERM. The macro-image responds to the motion of every mass component of
# the deflector, weighted by that component's share of the local deflection gradient:
# differentiating alpha = sum_c alpha_c(theta - theta_c(t)) for a multi-component lens,
#   d theta^I/dt = B^I [mu - sum_c H_c dmu_c],   H_c = grad alpha_c at the image.
# A virialized, phase-mixed halo has a STATIC potential despite its fast-moving
# particles, so only CLUMPED components count. The hierarchy is cluster > galaxy-scale
# halos > microhalos: the member-galaxy halos (intermediate level) orbit at the cluster
# dispersion sigma_v and supply a fraction f_gal of the local convergence-plus-shear at
# the images (kappa + gamma ~ 0.95 at B,C; nearby members within ~5-10", plus the known
# ~1e8 Msun radio-anomaly perturber near image B [Kratzer+ 2011], in a dynamically
# disturbed cluster). We estimate f_gal ~ 0.1-0.4, fiducial 0.2 (uncertain by ~x2).
# Member orbits (~Gyr) are quasi-static over tau = 10 yr, so this term is a constant
# addition to the image velocity and injects no spurious acceleration noise:
#   v_eff = sqrt(v_bulk^2 + 2 (f_gal sigma_v)^2) ~ 550 km/s at the fiducial f_gal,
# spanning ~500-740 km/s over f_gal = 0.1-0.4.
sigma_v_cluster = 1000 * km / second   # cluster internal velocity dispersion sigma_v (1D)
v_lens_bulk = 474 * km / second        # bulk (center-of-mass) transverse velocity
f_gal = 0.2                            # moving-clump share of the local deflection gradient
v_lens_eff = np.sqrt(v_lens_bulk**2 + 2 * (f_gal * sigma_v_cluster)**2)  # v_eff ~ 552 km/s
mu_L_int = np.sqrt(2) * sigma_v_cluster / ((1 + z_lens) * d_lens)  # mu_L, unmagnified rms 2D
v_lens_fid = v_lens_eff / np.sqrt(2) * np.asarray([1., 1.])  # fiducial forecast direction

# ---- stellar microlensing background (App. "Stellar component") ----
# Images form ~90-100 kpc from the cluster center, in the dark-matter-dominated outskirts
# where only the intracluster light (ICL) contributes stars. Morishita+2017 Fig. 5 (the
# six HFF clusters, M_500 ~ 1.2-1.8e15 M_sun) reads log Sigma_*/(M_sun/kpc^2) ~ 6.3-6.7 at
# 100 kpc, i.e. Sigma_* ~ 2-5 M_sun/pc^2; against Sigma_cr = 2021 M_sun/pc^2 that is
# kappa_* ~ 0.001-0.0025, and those clusters are ~10x more massive than this one
# (M_vir ~ 2e14 M_sun), so the reference is conservative. Zibetti+2005 stacks of
# lower-mass clusters give several times less. Microlensing of the cluster lens
# SDSS J1004+4112 gives kappa_* ~ 0.02-0.07 (Fores-Toribio+2024b), but at image radii of
# only 40-70 kpc. Still more than an order of magnitude below B1422+231 (kappa_* ~ 0.05).
kappa_star = 0.003                 # fiducial stellar convergence kappa_* at the images
M_star = 0.3 * M_Solar             # characteristic microlens mass M_*
theta_E_star = theta_E(M_star, d_lens, d_source, d_lens_source)  # theta_E,*

# ---- host-halo location of the lensing subhalos, x_sub = R_sub / R_200^cluster ----
# Needed by the Moline+2017 concentration relation behind the fiducial rho_s(M_s) band of
# Fig. SNR (see sensitivity_functions.scale_params_NFW_Moline). The images sit ~95 kpc
# from the cluster center; for M_vir ~ 2e14 M_Solar (Oguri+2012) the virial radius at
# z_L is ~1 Mpc, so the projected x_perp ~ 0.1. Weighting the line of sight by an NFW
# subhalo number density with c_host = 5 raises the median 3D distance by ~50%.
M_200_cluster = 2e14 * M_Solar     # Oguri+2012 virial mass
R_img_cluster = 95 * kpc           # image distance from the cluster center (App. J1029)
rho_crit_z = rho_crit * (0.3 * (1 + z_lens)**3 + 0.7)
R_200_cluster = (3 * M_200_cluster / (4 * np.pi * 200 * rho_crit_z))**(1/3)
c_host = 5.
_r_s_host = R_200_cluster / c_host
_vec_z_los = np.concatenate([[0], np.logspace(-5, 0, 2000) * R_200_cluster])
_r = np.sqrt(R_img_cluster**2 + _vec_z_los**2)
_w = 1 / ((_r/_r_s_host) * (1 + _r/_r_s_host)**2)
_cdf = np.concatenate([[0], np.cumsum(0.5*(_w[1:]+_w[:-1]) * np.diff(_vec_z_los))])
x_sub_perp = R_img_cluster / R_200_cluster
x_sub_fid = np.interp(0.5*_cdf[-1], _cdf, _r) / R_200_cluster
print('R_200^cluster = %.0f kpc, x_sub = %.3f (projected %.3f)'
      % (R_200_cluster/kpc, x_sub_fid, x_sub_perp))

# ---- source sizes for the finite-source form factor of Eq. (finite_source) ----
# The form factor |W~^I|^2 acts as a high-pass filter in halo angular size: a halo of
# angular size gamma_L is washed out only for gamma_L < theta_src^I, the source size
# MAGNIFIED by B^I. For the Gaussian source and Gaussian-cutoff cusp halo the two scales
# add in quadrature (C_ij_integral_src), so theta_src^I = B_tang * theta_src with B_tang
# the tangential (largest) eigenvalue of B^I -- the same direction as mu_tilde^I.
B_tang = np.max(np.abs(eigvals_fit), axis=1)   # tangential stretch per image (~22.5 at B,C)

# Optical/UV accretion-disk continuum: compact (~1 light-day), so it clips only
# M_L < 1e-5 M_sun. NB this is a plain geometric size, unlike the temperature-defined
# R_500 of params_B1422_231 (which carries an extra (1+z_S)^(4/3)).
R_src = 1.5e15 * cm                # optical continuum radius (~1 light-day)
theta_src = R_src / d_source        # unlensed theta_src ~ 0.06 muas
theta_src_I = B_tang * theta_src   # magnified theta_src^I ~ 1.3 muas at B,C

# Compact VLBI radio cores are far larger and suppress all but the heaviest halos,
# leaving only the acceleration channel (the "Radio interferometry" paragraph, Sec. III D).
R_src_radio = np.asarray([0.01, 0.1, 1.0]) * pc
theta_src_radio = R_src_radio / d_source          # unlensed radio sizes
theta_src_radio_I = np.outer(B_tang, theta_src_radio)  # [image, core] magnified, ~27-2700 muas

# ---- VLBI light-centroiding precision (radio channel) ----
# Fiducial relative astrometric precision for very-long-baseline interferometry:
# ~10 muas with current phase-referenced VLBI, ~1 muas a forward-looking target
# (ngVLA long baselines / space VLBI).
sigma_delta_theta_VLBI = 10.0 * muas
sigma_delta_theta_VLBI_future = 1.0 * muas

if __name__ == "__main__":
    print(df.to_string(index=False))
    print(f"\nd_L={d_lens/Gpc:.3f} Gpc, d_S={d_source/Gpc:.3f} Gpc, d_LS={d_lens_source/Gpc:.3f} Gpc")
    print(f"scale at lens = {length_per_arcsec/kpc:.2f} kpc/arcsec")
    print(f"Sigma_crit = {Sigma_crit_val/(M_Solar/pc**2):.0f} M_sun/pc^2")
    for i in range(3):
        print(f"  image {labels[i]}: A={mag_fit[i]:+.1f}, |B| eigs={np.sort(np.abs(eigvals_fit[i]))}")
    print(f"theta_E_star(0.3 Msun) = {theta_E_star/muas:.2f} muas")

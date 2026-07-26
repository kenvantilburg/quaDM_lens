from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import pandas as pd
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# SDSS J1029+2623: a three-image quasar lensed by a galaxy CLUSTER (image
# separation 22.5"). Lens model (convergence & shear at the QSO redshift) from
# the Acebron+2022 reconstruction. This is the cluster-lens analogue of the
# (galaxy-lensed) B1422+231 system in params_B1422_231.py.

# redshifts and distances
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

# lensing Jacobian A = I - H and its inverse B = A^{-1}, standard convention
#   A = [[1-kappa-gamma1, -gamma2], [-gamma2, 1-kappa+gamma1]]
# matching macro_lens.ipynb / App. (Macro-lensing Model) of the paper.
jacobian_fit = np.zeros((3, 2, 2))
for i in range(3):
    jacobian_fit[i] = [[1 - kappa_fit[i] - gamma1_fit[i], -gamma2_fit[i]],
                       [-gamma2_fit[i], 1 - kappa_fit[i] + gamma1_fit[i]]]
inv_jacobian_fit = np.linalg.inv(jacobian_fit)
mag_fit = 1 / np.linalg.det(jacobian_fit)
eig_fit = np.linalg.eig(inv_jacobian_fit)
eigvals_fit = eig_fit[0]
eigvecs_fit = eig_fit[1]

# ---- cluster velocity scale (relevant for lensing) ----
# Velocity budget (computed in mu_expected.py; peculiar velocities in CMB frame):
# observer: CMB dipole transverse component at this LOS: 212 km/s (apex sep. 34.9 deg).
# lens:     bulk peculiar velocity of the cluster, ~300 km/s per axis (linear theory).
#           The cluster carries its subhalos with it, so this is a BULK velocity.
# source:   quasar host peculiar velocity ~300 km/s (1D).
# Expected bulk-only <|mu|^2>^{1/2} = 0.046 muas/yr -> equivalent bulk transverse
# velocity (1+z_L) d_L |mu| = 474 km/s: this sets the macro-image proper motion
# mu_tilde = B mu, mu_tilde^{B,C} ~ 1 muas/yr -- the rate at which the images sweep
# across the subhalo field.
#
# KINEMATICS (referee fix, 2026-07): the sweep rate of image I across a subhalo's
# deflection field is d/dt[theta^I - theta_h] = B mu_eff - mu_h,int. Only the
# macro-image motion is magnified by B; a subhalo's own orbital motion in the cluster
# (sigma_v ~ 1000 km/s 1D, set by the ~22.5" splitting [theta_E ~ 15"]) moves the
# DEFLECTOR, not the image, and so enters UNMAGNIFIED:
#   mu_sweep = sqrt(|B mu_eff|^2 + 2 (sigma_v/((1+z_L) d_L))^2).
# (Adding sigma_v to the velocity BEFORE magnifying, v = sqrt(474^2+2*1000^2)
# ~ 1490 km/s, would wrongly boost the microhalos' orbital motion by B ~ 22: a
# factor ~3 overestimate of the sweep rate and ~30-100 in acceleration variance.)
#
# MACRO-IMAGE (EFFECTIVE) VELOCITY, 2026-07-18: the macro-image responds to the
# motion of every mass component of the deflector, weighted by that component's
# share of the local deflection gradient: differentiating the lens equation for a
# multi-component lens, alpha = sum_c alpha_c(theta - theta_c(t)), gives
#   d theta^I/dt = B [mu - sum_c H_c dmu_c],   H_c = grad alpha_c at the image.
# A virialized, phase-mixed halo has a STATIC potential despite its fast-moving
# particles, so only CLUMPED components count. The hierarchy is cluster > galaxy-
# scale halos > microhalos: the member-galaxy halos (intermediate level) orbit at
# the cluster dispersion sigma_v and contribute a fraction f_gal of the local
# convergence+shear at the images (nearby members within ~5-10", plus the known
# ~1e8 Msun radio-anomaly perturber near image B [Kratzer+2011], in a dynamically
# disturbed cluster). Estimate f_gal ~ 0.1-0.4, fiducial 0.2 (uncertain x2).
# Member orbits (~Gyr) are quasi-static over tau = 10 yr, so this term acts as a
# constant addition to the image velocity (no spurious acceleration noise):
#   v_eff = sqrt(v_bulk^2 + 2 (f_gal sigma_v)^2) ~ 550 km/s (fiducial)
# spanning ~500-740 km/s for f_gal = 0.1-0.4.
sigma_v_cluster = 1000 * km / second   # internal velocity dispersion (1D)
v_lens_bulk = 474 * km / second        # bulk (COM) effective transverse velocity
f_gal = 0.2                            # moving-clump fraction of local deflection gradient
v_lens_eff = np.sqrt(v_lens_bulk**2 + 2 * (f_gal * sigma_v_cluster)**2)  # ~552 km/s
mu_int_drift = np.sqrt(2) * sigma_v_cluster / ((1 + z_lens) * d_lens)  # unmagnified, rms 2D
v_lens_fid = v_lens_eff / np.sqrt(2) * np.asarray([1., 1.])  # fiducial forecast vector

# ---- stellar microlensing background ----
# Images form at ~90-100 kpc from the cluster center, in the dark-matter-dominated
# outskirts where only the intracluster light / BCG outskirts contribute stars.
# Sigma_* ~ few M_sun/pc^2 and Sigma_crit ~ 2000 M_sun/pc^2 give a small
# stellar convergence -- much lower than in a galaxy lens (kappa_* ~ 0.05).
kappa_star = 0.003                 # fiducial stellar convergence at the images
M_star = 0.3 * M_Solar             # characteristic microlens mass
theta_E_star = theta_E(M_star, d_lens, d_source, d_lens_source)

# ---- source sizes (finite-source washout) ----
# Optical/UV accretion-disk continuum: compact (~1 light-day), safely below the
# optimal substructure size. Radio (VLBI core): much larger -- see below.
# The finite-source form factor acts as a high-pass filter in halo angular size:
# a halo of angular size gamma_L is washed out only when gamma_L < theta_src^I,
# the *magnified* image source size. theta_src^I = B_tang * theta_src, with
# B_tang the tangential (largest) eigenvalue of B^I = inv_jacobian_fit (the same
# direction as the image motion mu_tilde).
B_tang = np.max(np.abs(eigvals_fit), axis=1)   # tangential magnification per image

R_source = 1.5e15 * cm             # optical continuum radius (~1 light-day)
theta_source = R_source / d_source # unlensed angular size
theta_source_I = B_tang * theta_source  # magnified optical source size per image

# fiducial compact VLBI radio core radii (parsec scale); much larger than optical
R_source_radio = np.asarray([0.01, 0.1, 1.0]) * pc
theta_source_radio = R_source_radio / d_source          # unlensed radio sizes
theta_source_radio_I = np.outer(B_tang, theta_source_radio)  # [image, core] magnified

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
        print(f"  image {labels[i]}: mu={mag_fit[i]:+.1f}, |B| eigs={np.sort(np.abs(eigvals_fit[i]))}")
    print(f"theta_E_star(0.3 Msun) = {theta_E_star/muas:.2f} muas")

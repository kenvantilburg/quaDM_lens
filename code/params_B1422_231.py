from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# B1422+231 parameters from Cosmograil 2012

# redshifts and distances
z_lens = 0.34
z_source = 3.62
d_lens = cosmo.angular_diameter_distance(z_lens).value * Mpc
d_source = cosmo.angular_diameter_distance(z_source).value * Mpc
D_lens = d_lens * (1 + z_lens)
D_source = d_source * (1 + z_source)
D_lens_source = D_source - D_lens
d_lens_source = D_lens_source / (1 + z_source)
npt.assert_almost_equal(d_lens/Mpc, d_A(z_lens)/Mpc, decimal=10) # check that the distance is consistent with manual calculation
npt.assert_almost_equal(d_source/Mpc, d_A(z_source)/Mpc, decimal=10) # check that the distance is consistent with manual calculation

# coordinates of images and galaxy/group
coord_ref = SkyCoord('14h24m38.09s', '+22d56m00.60s', frame='icrs') # reference coordinate
coord_ref = np.asarray([coord_ref.ra.degree, coord_ref.dec.degree]) * degree # in rad
cos_dec = np.cos(coord_ref[1]) # cosine of the declination
theta = np.asarray([[0.3860,0.3169],[0.,0.],[-0.3360,-0.7516],[0.9470,-0.8012]]) # lens image locations in ra, dec (in arcsec)
print('theta: \n',theta)
theta_lens = np.asarray([0.7321,-0.6390]) # lens galaxy location in ra, dec (in arcsec)
labels = ['A','B','C','D']

# flux
flux_ratios = np.array([1, 1.116, 0.59, 0.032]) # flux ratios between the images (relative to A)

# source size derived for this system in the companion paper (Moran & Van Tilburg),
# by flux-normalizing a thin-disk blackbody model (Galanis+ 2023, arXiv:2307.06989) to
# the observed brightness of image A. R_500 is the source-frame, face-on disk radius at
# which T = T_500 = hc/(lambda_500 k_B), lambda_500 = 500 nm (CASTLES-based value adopted
# there; the Gaia-based cross-check gives 7.54e15 cm). Replaces the Mosquera 2011 estimate.
R_source = 6.93e15 * cm # physical radius R_500
# The temperature-defined R_500 carries an extra (1+z_source)^(4/3) factor relative to the
# plain geometric size, so the unlensed 500 nm angular radius is R_500/(d_source*(1+z)^(4/3))
# (companion paper Eq. for theta_500). This reproduces their theta_500,lensed = 5.13e-13 rad.
theta_500 = R_source / (d_source * (1 + z_source)**(4/3)) # unlensed angular radius at 500 nm, face-on
theta_source = theta_500 # angular radius of source in rad
M_star = 0.3 * M_Solar 
#theta_E_star = np.sqrt(4 * G_N * M_star * (1+z_lens) * D_lens_source / (D_lens*D_source))
theta_E_star = theta_E(M_star, d_lens, d_source, d_lens_source)
R_E_star = theta_E_star * d_source # physical Einstein radius of solar mass on source plane
#print('R_source = ', str(R_source/(1e15*cm)) + 'e15 cm')
#print('R_E_star = ', str(R_E_star/(1e16*cm))[0:5] + 'e16 cm')
print('theta_500 = ', str(theta_500/muas)[0:5] + ' muas = '+str(theta_500*1e12)[0:5] + 'e-12 rad')
print('theta_E_star = ', str(theta_E_star/muas)[0:5] + ' muas')

# ---- velocity budget (computed in mu_expected.py; peculiar velocities in CMB frame) ----
# observer: CMB dipole 369.8 km/s (Planck 2018) -> transverse component at this LOS:
#   306 km/s, PA 243.5 deg E of N (apex separation 55.8 deg); a KNOWN vector.
# lens G:   orbital motion in its group, sigma_1D ~ 500 km/s (Kundic+ 1997: ~550;
#   Momcheva+ 2006: ~470), plus ~300 km/s (1D) bulk flow of the group -> 583 km/s/axis.
#   The galaxy carries its subhalos with it, so this is a BULK velocity.
# source:   quasar host peculiar velocity ~300 km/s (1D); suppressed by 1/[(1+z_S) d_S].
# Expected bulk-only <|mu|^2>^{1/2} = 0.136 muas/yr, i.e. an equivalent bulk transverse
# velocity (1+z_L) d_L |mu| = 865 km/s: this sets the macro-image proper motion
# mu_tilde = B mu, the rate at which the image sweeps across the subhalo field.
#
# KINEMATICS (referee fix, 2026-07): the sweep rate of image I across a subhalo's
# deflection field is d/dt[theta^I - theta_h] = B mu_bulk - mu_h,int. Only the bulk
# term is magnified by B; a subhalo's own orbital motion (sigma_int ~ 150 km/s 1D
# for the galaxy halo) moves the DEFLECTOR, not the image, and so enters UNMAGNIFIED:
#   mu_sweep = sqrt(|B mu_bulk|^2 + 2 (sigma_int/((1+z_L) d_L))^2).
# The internal drift, mu_int_drift ~ 0.03 muas/yr, is negligible against
# |B mu_bulk| ~ 1 muas/yr. (The former convention added sigma_int to the velocity
# BEFORE magnifying, v_stoch = sqrt(865^2+2*150^2) = 890 km/s, wrongly boosting the
# internal motion by B; immaterial here, but a factor ~3 in mu_sweep for J1029.)
# MACRO-IMAGE (EFFECTIVE) VELOCITY note, 2026-07-18: distinct components of the
# deflector moving relative to one another add B-magnified image motion weighted
# by their share of the local deflection gradient (see params_J1029_2623.py).
# Here the local field at the images is dominated by G itself, whose orbital
# motion (500 km/s/axis) is already in the bulk budget; the OTHER group members
# enter through the external shear, gamma_ext ~ 0.17, with relative dispersion
# sqrt(2)*sigma_grp -> sqrt(2)*0.17*707 ~ 170 km/s (2D), a ~2% quadrature
# correction to 865 km/s: negligible, so the bulk-only fiducial stands.
sigma_int_1D = 150 * km / second   # subhalo internal velocity dispersion (1D)
v_lens_bulk = 865 * km / second    # macro-image (bulk) effective transverse velocity
mu_int_drift = np.sqrt(2) * sigma_int_1D / ((1 + z_lens) * d_lens)  # unmagnified, rms 2D
v_lens_fid = v_lens_bulk / np.sqrt(2) * np.asarray([1., 1.])  # fiducial forecast vector (bulk)

# light profile de Vaucouleurs
M_Sal = 10**(10.83) * M_Solar
ell_star = 0.39 #1 - 0.33/0.51; print("ell_star =",ell_star)
theta_eff_star = 0.41 # in units of arcsec
theta_e_star = -58.9 * degree


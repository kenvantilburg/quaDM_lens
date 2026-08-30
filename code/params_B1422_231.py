"""Observed inputs for the galaxy lens B1422+231 (App. B1422+231 of the paper).

Astrometry, photometry and light-profile shape are the deconvolved HST/NICMOS F160W
(H-band) measurements of Sluse et al. 2012 (COSMOGRAIL, Tabs. 3 and 4); the SIE + shear
macro-model itself is fit in macro_lens.ipynb. Distances follow the convention of
macro_lens_functions: lowercase d_* are angular-diameter distances (the paper's
D_L, D_S, D_LS), uppercase D_* the corresponding comoving distances.
"""
from preamble import *
from natural_units_GeV import *
from macro_lens_functions import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# redshifts and distances (d_L ~ 1.0, d_S ~ 1.5, d_LS ~ 1.2 Gpc)
z_lens = 0.34
z_source = 3.62
d_lens = cosmo.angular_diameter_distance(z_lens).value * Mpc
d_source = cosmo.angular_diameter_distance(z_source).value * Mpc
D_lens = d_lens * (1 + z_lens)
D_source = d_source * (1 + z_source)
D_lens_source = D_source - D_lens          # comoving distances are additive (flat cosmology)
d_lens_source = D_lens_source / (1 + z_source)
npt.assert_almost_equal(d_lens/Mpc, d_A(z_lens)/Mpc, decimal=10)      # astropy vs. d_A above
npt.assert_almost_equal(d_source/Mpc, d_A(z_source)/Mpc, decimal=10)

# coordinates of images and galaxy/group
coord_ref = SkyCoord('14h24m38.09s', '+22d56m00.60s', frame='icrs') # reference coordinate
coord_ref = np.asarray([coord_ref.ra.degree, coord_ref.dec.degree]) * degree # in rad
cos_dec = np.cos(coord_ref[1]) # cosine of the declination
# image positions theta^I in (RA, Dec) arcsec offsets from the brightest image B (Tab. tab:macro)
theta = np.asarray([[0.3860,0.3169],[0.,0.],[-0.3360,-0.7516],[0.9470,-0.8012]])
print('theta: \n',theta)
theta_lens = np.asarray([0.7321,-0.6390]) # lens galaxy G location in ra, dec (in arcsec)
labels = ['A','B','C','D']

# flux
flux_ratios = np.array([1, 1.116, 0.59, 0.032]) # observed flux ratios relative to image A

# source size derived for this system in the companion paper (Moran & Van Tilburg),
# by flux-normalizing a thin-disk blackbody model (Galanis+ 2023, arXiv:2307.06989) to
# the observed brightness of image A. R_500 is the source-frame, face-on disk radius at
# which T = T_500 = hc/(lambda_500 k_B), lambda_500 = 500 nm (CASTLES-based value adopted
# there). Agrees at the ~10% level with the Mosquera+2011 microlensing disk size once
# that is rescaled to 500 nm and face-on orientation (App. B1422+231).
R_src = 6.93e15 * cm # physical radius R_500
# The temperature-defined R_500 carries an extra (1+z_source)^(4/3) factor relative to the
# plain geometric size, so the unlensed 500 nm angular radius is R_500/(d_source*(1+z)^(4/3)).
theta_500 = R_src / (d_source * (1 + z_source)**(4/3)) # unlensed angular radius at 500 nm, face-on
theta_src = theta_500              # theta_src of Sec. II D: ~0.04 muas, magnified to ~0.3 muas at image A
M_star = 0.3 * M_Solar             # characteristic microlens mass M_* (Sec. III C)
theta_E_star = theta_E(M_star, d_lens, d_source, d_lens_source)  # theta_E,* ~ 1.4 muas
R_E_star = theta_E_star * d_source # theta_E,* projected onto the source plane, ~3e16 cm ~ 0.01 pc
print('theta_500 = ', str(theta_500/muas)[0:5] + ' muas = '+str(theta_500*1e12)[0:5] + 'e-12 rad')
print('theta_E_star = ', str(theta_E_star/muas)[0:5] + ' muas')

# ---- host-halo location of the lensing subhalos, x_sub = R_sub / R_200^G (Sec. IV A) ----
# The Moline+2017 concentration relation behind the fiducial rho_s(M_s) band of Fig. SNR
# depends on where in its host a subhalo sits, x_sub = R_sub/R_200^host, so we need that
# for the four images. For G we take the isothermal velocity dispersion implied by the
# fitted Einstein radius theta_E = 0.776" (App. B1422),
#   sigma_SIS = c [theta_E d_S / (4 pi d_LS)]^(1/2),
# and the SIS virial radius at the lens redshift from
#   M_200 = 2 sigma^2 R_200 / G_N = (4 pi/3) 200 rho_c(z_L) R_200^3.
theta_E_G = 0.776 * arcsec         # best-fit SIE Einstein radius of G (macro_lens.ipynb)
sigma_SIS = np.sqrt(theta_E_G * d_source / (4 * np.pi * d_lens_source))   # c = 1 units
rho_crit_z = rho_crit * (0.3 * (1 + z_lens)**3 + 0.7)                     # rho_c(z_L)
R_200_G = np.sqrt(2 * sigma_SIS**2 * 3 / (4 * np.pi * 200 * rho_crit_z * G_N))
M_200_G = 2 * sigma_SIS**2 * R_200_G / G_N

# Projected image-to-G separations, and their 3D counterparts. Subhalos anywhere along
# the column contribute, so the relevant 3D radius exceeds the projected one; we weight
# the line of sight by an NFW subhalo number density n(r) with host concentration
# c_host = 6 and quote the weighted median R_sub. (Moline+'s fit is calibrated for
# x_sub > 0.01 and diverges logarithmically below that, so it is not pushed further.)
R_perp_images = np.linalg.norm(theta - theta_lens, axis=1) * arcsec * d_lens
c_host = 6.
_r_s_host = R_200_G / c_host
_vec_z_los = np.concatenate([[0], np.logspace(-5, 0, 2000) * R_200_G])
x_sub_perp = R_perp_images / R_200_G
x_sub_images = np.zeros_like(x_sub_perp)
for _i, _R in enumerate(R_perp_images):
    _r = np.sqrt(_R**2 + _vec_z_los**2)
    _w = 1 / ((_r/_r_s_host) * (1 + _r/_r_s_host)**2)          # NFW number density
    _cdf = np.concatenate([[0], np.cumsum(0.5*(_w[1:]+_w[:-1]) * np.diff(_vec_z_los))])
    x_sub_images[_i] = np.interp(0.5*_cdf[-1], _cdf, _r) / R_200_G
print('sigma_SIS = %.0f km/s, R_200^G = %.0f kpc, M_200^G = %.1e M_Solar'
      % (sigma_SIS/(km/second), R_200_G/(1e3*pc), M_200_G/M_Solar))
print('x_sub (projected / LOS-weighted median):',
      ', '.join('%s %.3f/%.3f' % (l, xp, x) for l, xp, x in zip(labels, x_sub_perp, x_sub_images)))
x_sub_fid = 0.03   # fiducial for the bright images A, B, C (D is ~2x smaller)

# ---- velocity budget of Eq. (mu) (computed in mu_expected.py; peculiar velocities in
# the CMB frame; see the "Expected motion" paragraph of Sec. III A) ----
# observer v_o: CMB dipole 369.8 km/s (Planck 2018) -> transverse component at this LOS
#   306 km/s, PA 243.5 deg E of N (apex separation 55.8 deg); a KNOWN vector.
# lens v_L:  G's orbital motion in its group, sigma_grp ~ 500 km/s per axis (Kundic+ 1997:
#   ~550; Momcheva+ 2006: ~470), in quadrature with a ~300 km/s (1D) bulk flow of the
#   group itself -> 583 km/s per axis. G carries its subhalos with it, so this is BULK.
# source v_S: quasar host peculiar velocity ~300 km/s (1D); suppressed by 1/[(1+z_S) d_S].
# Together: <|mu|^2>^{1/2} = 0.136 muas/yr, i.e. an equivalent bulk transverse velocity
# (1+z_L) d_L <|mu|^2>^{1/2} = 865 km/s. This sets the magnified image proper motion
# mu_tilde^I = B^I mu of Eq. (mu_tilde), the rate at which image I sweeps the subhalo field.
#
# KINEMATICS. The sweep rate of image I across a given subhalo's deflection field is
# d/dt[theta^I - theta_L] = B^I mu - mu_L (footnote to Sec. II C). Only the bulk term is
# magnified by B^I; a subhalo's own orbital motion (sigma_int ~ 150 km/s per axis in the
# galaxy halo) displaces the DEFLECTOR, not the image, and enters UNMAGNIFIED:
#   mu_sweep = sqrt(|B^I mu|^2 + 2 (sigma_int/[(1+z_L) d_L])^2).
# The internal drift mu_L_int ~ 0.03 muas/yr is negligible against |B^I mu| ~ 1 muas/yr
# (a <1% quadrature correction, as quoted in Sec. II C).
#
# MOVING-CLUMP TERM. Macroscopic components of the deflector moving relative to one
# another do add B-magnified image motion, weighted by their share of the local
# deflection gradient (this is the f_gal term of params_J1029_2623.py). Here the local
# field is dominated by G itself, whose orbital motion is already in the bulk budget; the
# other group members enter only through the external shear gamma_ext ~ 0.17, giving
# sqrt(2) gamma_ext sigma_grp ~ 120 km/s -- a ~1% quadrature correction to 865 km/s (the
# sqrt(2) is the two-component projection |H dmu| for a traceless shear H = diag(g,-g)
# acting on an isotropic dmu of per-axis dispersion sigma_grp = 500 km/s). So
# the bulk-only fiducial stands for this system.
sigma_int_1D = 150 * km / second   # subhalo internal velocity dispersion sigma_int (1D)
v_lens_bulk = 865 * km / second    # fiducial bulk transverse velocity |v_L|
mu_L_int = np.sqrt(2) * sigma_int_1D / ((1 + z_lens) * d_lens)  # mu_L, unmagnified rms 2D
# fiducial forecast direction: v_L = [612, 612] km/s, as quoted for Figs. CtildeA and acc
v_lens_fid = v_lens_bulk / np.sqrt(2) * np.asarray([1., 1.])

# ---- stellar light profile (de Vaucouleurs), for kappa_* in Tab. tab:macro ----
# Total Salpeter-IMF stellar mass, and the F160W light-profile shape of Sluse+ 2012 Tab. 3:
# ellipticity 0.39 (axis ratio 1:0.61), effective radius 0.41", major axis at -58.9 deg.
# PA_major_star is a position angle EAST OF NORTH (Sluse+'s convention), which is also
# the convention kappa_DV consumes -- see its docstring. Do not convert it to
# lenstronomy's angle phi = 90 deg - PA; that is only needed for the SIE/shear guess in
# macro_lens.ipynb, where the converted angles are named phi_ell / phi_gamma.
M_Sal = 10**(10.83) * M_Solar
ell_star = 0.39
theta_eff_star = 0.41 # in units of arcsec, geometric-mean effective radius
PA_major_star = -58.9 * degree


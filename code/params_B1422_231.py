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

# (unlensed) source size from Mosquera 2011 -- radius at which  T = lambda_rest with lambda_obs = 814 nm at cos i = 1/2
R_source = 2.29e15 * cm # physical radius
theta_source = R_source / d_source # angular radius of source in rad
theta_500 = (500/814)**(4/3) * theta_source / np.sqrt(2) # angular radius where T = lambda_rest with lambda_obs = 500 nm, assuming T ~ R^(-3/4), face-on
M_star = 0.3 * M_Solar 
#theta_E_star = np.sqrt(4 * G_N * M_star * (1+z_lens) * D_lens_source / (D_lens*D_source))
theta_E_star = theta_E(M_star, d_lens, d_source, d_lens_source)
R_E_star = theta_E_star * d_source # physical Einstein radius of solar mass on source plane
#print('R_source = ', str(R_source/(1e15*cm)) + 'e15 cm')
#print('R_E_star = ', str(R_E_star/(1e16*cm))[0:5] + 'e16 cm')
print('theta_500 = ', str(theta_500/muas)[0:5] + ' muas = '+str(theta_500*1e12)[0:5] + 'e-12 rad')
print('theta_E_star = ', str(theta_E_star/muas)[0:5] + ' muas')

# light profile de Vaucouleurs
M_Sal = 10**(10.83) * M_Solar
ell_star = 0.39 #1 - 0.33/0.51; print("ell_star =",ell_star)
theta_eff_star = 0.41 # in units of arcsec
theta_e_star = -58.9 * degree


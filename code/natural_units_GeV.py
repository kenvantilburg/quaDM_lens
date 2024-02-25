import math
import numpy as np

# Notebook for user defined units. All dimensional quantities are in GeV = 1
GeV = 1; 
TeV = 1e3 * GeV;
MeV = 1e-3 * GeV; keV = 1e-3 * MeV; eV = 1e-3 * keV; meV = 1e-3 * eV; 

kg = 5.6096 * 10**26 * GeV
gram = 1e-3 * kg
meter = 1/(0.1973e-15 * GeV)
cm = 1e-2 * meter; mm = 1e-3 * meter; nm = 1e-9 * meter; Angstrom  = 1e-10 * meter; fm = 1e-15 * meter; 
km = 1e3 * meter; 

second = 2.99792458e8 * meter
ps = 1e-12 * second; fs = 1e-15 * second;
Hz = second**-1
kHz = 1e3 * Hz; MHz = 1e6 * Hz; GHz = 1e9 * Hz; THz = 1e12 * Hz; mHz = 1e-3 * Hz;
hour = 3600 * second
year = 365*24 * hour

Joule = kg * meter**2 * second**-2; erg = 10**-7 * Joule; Watt = Joule * second**-1;
Newton = kg * meter * second**-2; Pa = kg * meter**-1 * second**-2; GPa = 10**9 * Pa;
Kelvin = 1.3806505e-23 * Joule;
sigma_B = np.pi**2 / 60;

N_Avo = 6.022e23;
bar = 1e5 * Pa;
Torr = 1/750.061683 * bar;

M_Planck = 1.2209e19 * GeV
G_N = M_Planck**-2
m_Planck = M_Planck/np.sqrt(8*np.pi)

kpc = 3261 * year; pc = 1e-3 * kpc; Mpc = 1e3 * kpc; Gpc = 1e3 * Mpc
AU = 150 * 1e6 * km;
R_Solar = 695.51e6 * meter; M_Solar = 1.98e30 * kg; lum_Solar = 3.827e26 * Watt;
M_Earth = 5.972e24 * kg;
M_Moon = 0.012300 * M_Earth; 
M_Jupiter = 1.898e27 * kg; a_Jupiter = 5.2044 * AU;
M_Venus = 4.867e24 * kg; a_Venus = 0.72333 * AU;
rho_Earth = 5.59 * kg/(10 * cm)**3;
R_Earth = 6371 * km;

H_0 = 70 * km / (second * Mpc)
Omega_M = 0.3
Omega_Lambda = 0.7
z_eq = 3250
H_eq = H_0*(1+z_eq)**(3/2) 
aeq = 1/(1 + z_eq)
rho_crit = 3 * H_0**2 * m_Planck**2
rho_M_0 = rho_crit * Omega_M
rho_Lambda_0 = rho_crit * Omega_Lambda
rho_DM_0 = 0.23 * rho_crit
rho_DM_G = 0.4 * GeV * cm**-3
v_0_DM = 235 * km/second
sigma_v_DM = v_0_DM / np.sqrt(2)

degree = np.pi / 180; arcmin = degree/60; arcsec = arcmin/60;
mas = 1e-3 * arcsec; muas = 1e-6 * arcsec;
masy = mas/year
muasy = muas/year
muasyy = muas/year**2

Tesla = 195 * eV**2; Gauss = 1e-4 * Tesla;
alpha_EM = 1/137;
e_EM  = np.sqrt(4 * np.pi * alpha_EM); Coulomb = (5.28 * 10**-19)**-1;
Volt = eV/e_EM;
Ampere = Coulomb/second; Ohm = Volt/Ampere; Farad = Coulomb/Volt; 
Henry = Joule/Ampere**2;
Debye = 3.33564 * 1e-30 * Coulomb * meter;

m_proton = 0.938 * GeV; MPion = 134.976 * MeV;
m_electron = 511. * keV;
m_up = 2.01 * MeV;
m_down = 4.79 * MeV;
m_higgs = 125 * GeV; v_EW = 246 * GeV;
mu_nuclear = e_EM/(2 * m_proton); mu_bohr = e_EM/(2 * m_electron); a_0 = 1/(alpha_EM * m_electron);
amu = 1.67377 * 1e-27 * kg;
barn = 1e-28 * meter**2; pb = 1e-12 * barn; fb = 1e-15 * barn;
G_F = 1.166 * 1e-5 * GeV**-2;

theta_axion = 4.1e-19
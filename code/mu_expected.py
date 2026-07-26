"""
Expected (unlensed) relative proper motion mu of the source w.r.t. the lens,
Eq. (eq:mu) of the paper, for the two benchmark systems:
  * galaxy lens  B1422+231       (z_L = 0.34,  z_S = 3.62)
  * cluster lens SDSS J1029+2623 (z_L = 0.588, z_S = 2.199)

    mu = v_S/[(1+z_S) d_S] - v_L/[(1+z_L) d_L] + v_o d_LS (1+z_S)/[(1+z_L) d_L d_S (1+z_S)]

with angular-diameter distances (Kayser, Refsdal & Stabell 1986), as implemented
in macro_lens_functions.mu_rel.

Velocity budget (peculiar velocities in the CMB frame):
  v_o -- KNOWN vector: solar-system barycenter velocity w.r.t. the CMB frame,
         369.82 km/s toward galactic (l,b) = (264.021, 48.253) deg
         [Planck 2018, arXiv:1807.06205]. Only the component transverse to the
         line of sight enters; its sky direction is the great-circle direction
         from the target toward the dipole apex.
  v_L -- BULK peculiar velocity of the deflector (random, quoted as 1D rms):
         B1422+231: lens galaxy G orbits in a rich compact group with measured
         line-of-sight dispersion sigma_grp ~ 550 km/s (Kundic+ 1997,
         AJ 114, 2276) to ~ 470 km/s (Momcheva+ 2006, ApJ 641, 169, with >2x
         the members); we adopt 500 km/s, added in quadrature with a ~300 km/s
         (1D) large-scale bulk flow of the group itself (linear theory).
         J1029+2623: bulk peculiar velocity of the cluster as a whole,
         ~300 km/s (1D) (linear theory / N-body, e.g. Sheth & Diaferio 2001).
  v_S -- quasar host-halo peculiar velocity, ~300 km/s (1D); its contribution
         is strongly suppressed by the 1/[(1+z_S) d_S] weight.

KINEMATICS (referee fix, 2026-07): the macro-image proper motion
mu_tilde^I = B^I mu is set by the BULK velocities only (the mu computed here).
The sweep rate of image I across a subhalo's deflection field is
    d/dt[theta^I - theta_h] = B^I mu_bulk - mu_h,int :
only the bulk term is magnified by B^I, because bulk source/lens/observer
motion moves the IMAGE through the lens plane, whereas a subhalo's own orbital
motion (sigma_int ~ 150 km/s 1D for a galaxy halo, ~ 1000 km/s for the
cluster) moves the DEFLECTOR and enters UNMAGNIFIED:
    mu_sweep = sqrt( |B^I mu_bulk|^2 + 2 (sigma_int/((1+z_L) d_L))^2 ).
The unmagnified internal drift (~0.03 muas/yr galaxy, ~0.14 muas/yr cluster)
is subdominant to |B mu_bulk| ~ 1 muas/yr in both benchmark systems.
(The former convention, v_eff,stoch = sqrt(v_bulk^2 + 2 sigma_int^2) applied
BEFORE the B magnification, wrongly boosted the internal motion by B -- a
factor ~3 overestimate of the J1029 sweep rate.)

Run:  conda run -n quaDM python mu_expected.py
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from natural_units_GeV import *
from macro_lens_functions import mu_rel
import params_B1422_231 as pB
import params_J1029_2623 as pJ

# ------------------------------------------------------------------ CMB dipole
V_DIPOLE = 369.82 * km / second  # Planck 2018
APEX = SkyCoord(l=264.021 * u.deg, b=48.253 * u.deg, frame='galactic').icrs


def dipole_transverse(target):
    """Transverse component of the observer's CMB-frame velocity at a given
    line of sight, in the (RA*cos Dec, Dec) sky basis (east positive)."""
    psi = target.separation(APEX).rad          # angle LOS <-> apex
    pa = target.position_angle(APEX).rad       # east of north, target -> apex
    v_perp = V_DIPOLE * np.sin(psi)
    return v_perp * np.array([np.sin(pa), np.cos(pa)]), psi, pa


# ------------------------------------------------------------------ per system
def report(name, z_l, z_s, d_l, d_s, d_ls, target,
           sigma_L_1D, sigma_S_1D, sigma_int_1D, inv_jacs, labels, v_fid):
    print("\n" + "=" * 72)
    print(f"{name}:  z_L = {z_l}, z_S = {z_s}")
    print(f"  d_L = {d_l/Gpc:.3f} Gpc, d_S = {d_s/Gpc:.3f} Gpc, "
          f"d_LS = {d_ls/Gpc:.3f} Gpc")

    # known observer (CMB dipole) term
    v_o_vec, psi, pa = dipole_transverse(target)
    mu_o = mu_rel(v_o_vec, np.zeros(2), np.zeros(2), z_l, z_s, d_l, d_s, d_ls)
    print(f"  CMB dipole: apex separation = {np.degrees(psi):.1f} deg, "
          f"|v_o,perp| = {np.linalg.norm(v_o_vec)/(km/second):.0f} km/s, "
          f"PA(->apex) = {np.degrees(pa):.1f} deg E of N")
    print(f"  observer term:  mu_o = ({mu_o[0]/muasy:+.4f}, {mu_o[1]/muasy:+.4f}) "
          f"muas/yr, |mu_o| = {np.linalg.norm(mu_o)/muasy:.4f} muas/yr")

    # random terms: per-axis rms of the mu contributions
    w_L = 1 / ((1 + z_l) * d_l)
    w_S = 1 / ((1 + z_s) * d_s)
    mu_L_axis = sigma_L_1D * w_L
    mu_S_axis = sigma_S_1D * w_S
    print(f"  lens term:      rms per axis = {mu_L_axis/muasy:.4f} muas/yr  "
          f"(sigma_L,1D = {sigma_L_1D/(km/second):.0f} km/s)")
    print(f"  source term:    rms per axis = {mu_S_axis/muasy:.4f} muas/yr  "
          f"(sigma_S,1D = {sigma_S_1D/(km/second):.0f} km/s)")

    # expected magnitude: known vector + isotropic 2D Gaussian terms
    var_axis = mu_L_axis**2 + mu_S_axis**2
    mu_rms = np.sqrt(np.linalg.norm(mu_o)**2 + 2 * var_axis)
    v_bulk_eff = mu_rms * (1 + z_l) * d_l
    print(f"  ==> expected <|mu|^2>^1/2 = {mu_rms/muasy:.4f} muas/yr")
    print(f"      equivalent bulk transverse velocity at the lens: "
          f"(1+z_L) d_L |mu| = {v_bulk_eff/(km/second):.0f} km/s "
          f"(paper fiducial: {v_fid:.0f} km/s)")

    # unmagnified internal-dispersion drift of the subhalos (adds in quadrature
    # to the MAGNIFIED bulk sweep |B mu| at the mu_tilde level, NOT in velocity)
    mu_int = np.sqrt(2) * sigma_int_1D / ((1 + z_l) * d_l)
    print(f"      internal-dispersion drift (UNMAGNIFIED): mu_int = "
          f"{mu_int/muasy:.3f} muas/yr  (sigma_int,1D = "
          f"{sigma_int_1D/(km/second):.0f} km/s)")
    print("      [sweep rate per image: mu_sweep = sqrt(|B mu_bulk|^2 + mu_int^2); "
          "mu_int is NOT boosted by B]")
    v_stoch = v_bulk_eff  # fiducial forecast velocity = bulk only

    # magnified image motions
    print("  magnified image motions mu_tilde^I = B^I mu:")
    for lab, B in zip(labels, inv_jacs):
        mu_t_o = B @ mu_o                                   # dipole-only part
        rms2 = mu_t_o @ mu_t_o + var_axis * np.sum(B * B)   # + random part
        print(f"    image {lab}:  dipole-only ({mu_t_o[0]/muasy:+.3f}, "
              f"{mu_t_o[1]/muasy:+.3f}) muas/yr;  expected rms "
              f"|mu_tilde| = {np.sqrt(rms2)/muasy:.3f} muas/yr")

    return mu_rms, v_bulk_eff, v_stoch


# ------------------------------------------------------------------ inputs
coord_B1422 = SkyCoord('14h24m38.09s', '+22d56m00.60s', frame='icrs')
coord_J1029 = SkyCoord(pJ.df['RA'][0] * u.deg, pJ.df['Dec'][0] * u.deg,
                       frame='icrs')

B_jacs_B1422 = np.load('macro_lens_results.npz')['inv_jacobian_fit']

SIGMA_GRP = 500 * km / second     # G's orbital motion in its group (1D)
SIGMA_FLOW = 300 * km / second    # large-scale bulk flow, group/cluster (1D)
SIGMA_SRC = 300 * km / second     # quasar host peculiar velocity (1D)

sigma_L_B1422 = np.sqrt(SIGMA_GRP**2 + SIGMA_FLOW**2)
sigma_L_J1029 = SIGMA_FLOW

if __name__ == "__main__":
    report("B1422+231 (galaxy lens)", pB.z_lens, pB.z_source,
           pB.d_lens, pB.d_source, pB.d_lens_source, coord_B1422,
           sigma_L_B1422, SIGMA_SRC, 150 * km / second,
           B_jacs_B1422, pB.labels, v_fid=700)

    report("SDSS J1029+2623 (cluster lens)", pJ.z_lens, pJ.z_source,
           pJ.d_lens, pJ.d_source, pJ.d_lens_source, coord_J1029,
           sigma_L_J1029, SIGMA_SRC, 1000 * km / second,
           pJ.inv_jacobian_fit, pJ.labels, v_fid=1000)

    # ------------------------------------------------- sanity anchor (paper)
    print("\n" + "=" * 72)
    print("Sanity anchor (App. macro-cluster of the paper): J1029 with "
          "|v_L| = 1000 km/s bulk,\nv_o = v_S = 0, should give "
          "mu ~ 0.1 muas/yr and mu_tilde^{B,C} ~ 2 muas/yr:")
    v_l = np.array([1000 / np.sqrt(2), 1000 / np.sqrt(2)]) * km / second
    mu_anchor = mu_rel(np.zeros(2), v_l, np.zeros(2), pJ.z_lens, pJ.z_source,
                       pJ.d_lens, pJ.d_source, pJ.d_lens_source)
    print(f"  |mu| = {np.linalg.norm(mu_anchor)/muasy:.3f} muas/yr")
    for lab, B in zip(pJ.labels, pJ.inv_jacobian_fit):
        print(f"  image {lab}: |mu_tilde| = "
              f"{np.linalg.norm(B @ mu_anchor)/muasy:.2f} muas/yr")

    # -------------------------------------- sensitivity to the sigma choices
    print("\n" + "=" * 72)
    print("Sensitivity to velocity-budget choices (expected <|mu|^2>^1/2, "
          "muas/yr;\nequivalent bulk velocity in km/s in parentheses):")
    print(f"{'budget':<10}{'sigma_grp':>10}{'sigma_flow':>11}{'sigma_src':>10}"
          f"{'B1422+231':>18}{'J1029+2623':>18}")
    for tag, s_grp, s_flow, s_src in [("low",     400, 200, 200),
                                      ("central", 500, 300, 300),
                                      ("high",    600, 400, 400)]:
        row = []
        for (z_l, z_s, d_l, d_s, d_ls, target, is_group) in [
                (pB.z_lens, pB.z_source, pB.d_lens, pB.d_source,
                 pB.d_lens_source, coord_B1422, True),
                (pJ.z_lens, pJ.z_source, pJ.d_lens, pJ.d_source,
                 pJ.d_lens_source, coord_J1029, False)]:
            s_L = (np.sqrt(s_grp**2 + s_flow**2) if is_group else s_flow) \
                * km / second
            v_o_vec, _, _ = dipole_transverse(target)
            mu_o = mu_rel(v_o_vec, np.zeros(2), np.zeros(2),
                          z_l, z_s, d_l, d_s, d_ls)
            var_axis = (s_L / ((1 + z_l) * d_l))**2 \
                + (s_src * km / second / ((1 + z_s) * d_s))**2
            mu_rms = np.sqrt(np.linalg.norm(mu_o)**2 + 2 * var_axis)
            row.append(f"{mu_rms/muasy:.3f} "
                       f"({mu_rms*(1+z_l)*d_l/(km/second):.0f})")
        print(f"{tag:<10}{s_grp:>10}{s_flow:>11}{s_src:>10}"
              f"{row[0]:>18}{row[1]:>18}")

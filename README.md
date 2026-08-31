# quaDM_lens

**Astrometric weak lensing of multiply imaged quasars by substellar dark matter halos.**

Code and figures for

> K. Van Tilburg and D. E. Kaplan,
> *Discovering Substellar Dark Matter Halos with Astrometric Weak Lensing of
> Multiply Imaged Quasars*, [arXiv:2608.27557](https://arxiv.org/abs/2608.27557).

Every figure in the paper is produced by a notebook in [`code/`](code); the map from
figure to notebook is in [Figures and tables](#figures-and-tables) below, and the
code-file icon in each figure caption of the paper links straight to that notebook. The
figures themselves are in [`figs/`](figs).

---

## What this computes

A strongly lensed quasar's macro-images sweep across the dark matter substructure of the
lens (and of the line of sight) at $\sim10^2$ – $10^3\ \mathrm{km\,s^{-1}}$. The
collective, stochastic deflection by many substellar dark matter halos makes each image
centroid execute a random walk with a calculable, red-tilted power spectrum. This
repository contains the calculations behind the forecast in the manuscript.

1. **Macrolensing model** — the local convergence $\kappa$, shear $(\gamma_1,\gamma_2)$
   and inverse magnification tensor $B^\mathrm{I}$ at each image, plus the smooth stellar
   convergence $\kappa_*$.
2. **Astrometric response** — the power spectral density
   $\widetilde{C}^\mathrm{I}_{pq}(\omega)$ of the image-centroid random walk induced by a
   monochromatic population of halos, and its smeared $\omega^4$ moment, the differential
   angular-acceleration covariance
   $\langle \delta\ddot\theta^{\mathrm{I},\mathrm{I}'}_{0,p}\,
   \delta\ddot\theta^{\mathrm{I},\mathrm{I}'}_{0,q}\rangle$.
3. **Survey sensitivity** — those signals against the instrumental noise floor of a
   $\tau$-year, $N$-epoch differential-astrometry campaign at per-epoch precision
   $\sigma_{\delta\theta}$, and against the post-subtraction stellar microlensing
   residual.
4. **Discovery potential** — the same reach expressed as a bound on the small-scale
   matter power spectrum $\Delta^2_\delta(k)$, alongside the pure-CDM and thermal-WIMP
   predictions.

Two benchmark systems are carried through in parallel everywhere:

| | **B1422+231** | **SDSS J1029+2623** |
|---|---|---|
| lens | galaxy | galaxy **cluster** |
| $z_\mathrm{L}$, $z_\mathrm{S}$ | 0.34, 3.62 | 0.588, 2.199 |
| images | 4 (A,B,C,D) | 3 (A,B,C) |
| image separation | $\sim1''$ | $22.5''$ (A vs B/C), $\approx 2''$ (B–C pair) |
| brightest-pair magnification | $\lvert A\rvert \approx 7$–9 | $\lvert A\rvert \approx 22$ |
| stellar convergence $\kappa_*$ | $\approx 0.05$ | $\approx 0.003$ |
| effective lens velocity | 865 km/s (bulk) | 550 km/s (bulk + moving clumps) |
| assumed precision $\sigma_{\delta\theta}$ | $0.1\,\mu\mathrm{as}$ (EPIC) | $1\,\mu\mathrm{as}$ |
| parameters | [`code/params_B1422_231.py`](code/params_B1422_231.py) | [`code/params_J1029_2623.py`](code/params_J1029_2623.py) |

The galaxy-lens and cluster-lens versions of a given figure live in the **same** notebook,
as Part I and Part II; the two systems are computed by identical code and differ only
through their parameter modules.

---

## Headline numbers

Reproducing Table I of the paper — the differential angular-acceleration
channel for the fiducial $\tau = 10\,\mathrm{yr}$, $N = 300$ campaigns, at
$\rho_s = 1\,M_\odot\,\mathrm{pc^{-3}}$ and substructure fraction
$f_\mathrm{sub} = 0.5$:

| | **B1422+231** | **SDSS J1029+2623** |
|---|---|---|
| image pair | A,B | B,C |
| $\sigma_{\delta\theta}$ $[\mu\mathrm{as}]$ | 0.1 | 1 |
| peak $\langle \delta\ddot\theta_0^2\rangle$ $[\mu\mathrm{as^2\,yr^{-4}}]$ | $6.1\times10^{-5}$ | $3.3\times10^{-4}$ |
| instrumental floor $[\mu\mathrm{as^2\,yr^{-4}}]$ | $2.4\times10^{-6}$ | $2.4\times10^{-4}$ |
| stellar residual $[\mu\mathrm{as^2\,yr^{-4}}]$ | $2.5\times10^{-5}$ | — |
| peak SNR (clean $\vert$ stars) | $25\ \vert\ 2.2$ | $1.4$ |
| $M_{\mathrm{L,peak}}$ $[M_\odot]$ | $2\times10^{-4}$ | $4\times10^{-4}$ |
| $\theta_\mathrm{fit}$ $[\mu\mathrm{as}]$ | 17 | — |
| stars within $\theta_\mathrm{fit}$ | 7.5 | 0.3 |

Evaluated *along* the fiducial $\Lambda$CDM $\rho_s(M_s)$ relation instead of at the
best-matched monochromatic point, the same campaigns give peak variance
$\mathrm{SNR} \approx 23\ \vert\ 2.0$ (clean $\vert$ stars) for B1422+231 and
$\approx 0.7$ for SDSS J1029+2623. Both sets are printed by
[`code/impact_numbers.py`](code/impact_numbers.py); all SNRs are **variance** ratios, not
Gaussian significances.

---

## Figures and tables

Figure numbers refer to the paper; paths are relative to the repository root.

| Fig. | File | Notebook | Content |
|---|---|---|---|
| 1 | [`figs/macrolens-B1422.pdf`](figs/macrolens-B1422.pdf) | [`code/macro_lens.ipynb`](code/macro_lens.ipynb) | B1422+231 macro-lens configuration: images, source, critical curve, caustic, distortion ellipses, $\kappa_*$ contours |
| 2 | [`figs/concept.pdf`](figs/concept.pdf) | [`code/concept.ipynb`](code/concept.ipynb) | Simulated realization of the signal: sky-plane random walk and the induced apparent differential acceleration |
| 3 | [`figs/CtildeA.pdf`](figs/CtildeA.pdf) | [`code/sensitivity.ipynb`](code/sensitivity.ipynb) (Part I) | Stochastic PSD $\widetilde{C}^\mathrm{A}_{pq}(\omega)$, galaxy lens |
| 4 | [`figs/acc.pdf`](figs/acc.pdf) | [`code/sensitivity.ipynb`](code/sensitivity.ipynb) (Part I) | Differential acceleration covariance vs $M_\mathrm{L}$, galaxy lens |
| 5 | [`figs/CtildeC-J1029.pdf`](figs/CtildeC-J1029.pdf) | [`code/sensitivity.ipynb`](code/sensitivity.ipynb) (Part II) | Stochastic PSD $\widetilde{C}^\mathrm{C}_{pq}(\omega)$, cluster lens — analogue of Fig. 3 |
| 6 | [`figs/acc-J1029.pdf`](figs/acc-J1029.pdf) | [`code/sensitivity.ipynb`](code/sensitivity.ipynb) (Part II) | Differential acceleration covariance vs $M_\mathrm{L}$, cluster lens — analogue of Fig. 4 |
| 7 | [`figs/SNR.pdf`](figs/SNR.pdf) | [`code/sensitivity.ipynb`](code/sensitivity.ipynb) (Part I) | Projected reach in the $(M_\mathrm{L}, \rho_s)$ plane, all three channels |
| 8 | [`figs/matter_power.pdf`](figs/matter_power.pdf) | [`code/matter_power.ipynb`](code/matter_power.ipynb) | Matter power spectrum $\Delta^2_\delta(k)$: line-of-sight reach of **both** campaigns, plus the lens-bound thresholds for B1422+231 |

Tables:

| Table | Where | Produced by |
|---|---|---|
| `tab:bottomline` | Sec. III D | [`code/impact_numbers.py`](code/impact_numbers.py) and the printed summaries of [`code/sensitivity.ipynb`](code/sensitivity.ipynb) (both parts) |
| `tab:macro` (B1422+231) | App. Macrolensing Models, B1422+231 | [`code/macro_lens.ipynb`](code/macro_lens.ipynb), end of Part I |
| `tab:macro-cluster` (SDSS J1029+2623) | App. Macrolensing Models, SDSS J1029+2623 | [`code/macro_lens.ipynb`](code/macro_lens.ipynb), end of Part II |

One figure in the repository is not in the paper:
[`figs/requirements.pdf`](figs/requirements.pdf), from
[`code/requirements.ipynb`](code/requirements.ipynb), plots the SNR of the table above
against the per-epoch precision $\sigma_{\delta\theta}$ as a summary of how the reach scales with astrometric capability. 
It was cut from the final draft because it obscured the latent variable of source brightness (with which different observatories' SNRs scale differently).

---

## Repository layout

```
code/
  # ---- notebooks: one per figure group (run in this order) ----
  macro_lens.ipynb      Part I  B1422+231 SIE+shear fit with lenstronomy -> macro_lens_results.npz,
                                figs/macrolens-B1422.pdf, Table tab:macro
                        Part II SDSS J1029+2623 cluster model read-out, Table tab:macro-cluster
  sensitivity.ipynb     Part I  galaxy lens  -> figs/CtildeA.pdf, figs/acc.pdf, figs/SNR.pdf
                        Part II cluster lens -> figs/CtildeC-J1029.pdf, figs/acc-J1029.pdf
  concept.ipynb         simulated signal realization -> figs/concept.pdf
  matter_power.ipynb    matter power spectrum + line-of-sight reach -> figs/matter_power.pdf
  requirements.ipynb    reach vs astrometric precision -> figs/requirements.pdf (not in the paper)
  Pnl.ipynb             CLASS + halofit -> Delta2_output.csv, Delta2_extended_output.csv
                        (needs `classy`; the CSVs are committed, so this is optional)

  # ---- shared modules ----
  preamble.py               imports, matplotlib/LaTeX setup, color palettes
  natural_units_GeV.py      natural-unit constants and conversions (GeV-based)
  macro_lens_functions.py   lensing geometry: distances, Sigma_crit, theta_E, mu_rel,
                            de Vaucouleurs kappa_*, stellar and smooth lens potentials
  sensitivity_functions.py  halo profiles (Einasto, Gaussian-cutoff cusp), form factors,
                            C_ij_integral / C_ij_integral_src, instrumental noise PSD,
                            off-plane corrections to the line-of-sight kernel
  params_B1422_231.py       galaxy-lens system parameters, velocity budget, source size
  params_J1029_2623.py      cluster-lens system parameters (Acebron+2022 model)

  # ---- standalone analysis scripts (print numbers quoted in the text) ----
  impact_numbers.py             headline SNRs of Tab. tab:bottomline, both systems,
                                instrument-limited and with the stellar residual
  mu_expected.py                relative proper-motion / velocity budget for both systems
  cusp_snr.py                   acceleration SNR of the lens-bound prompt-cusp population,
                                and its kinetic-decoupling-temperature sweep
  los_full_spectrum_snr.py      line-of-sight SNRs of the predicted CDM / WIMP / cusp spectra
  los_vs_lens_decomposition.py  reconciles the lens-bound and line-of-sight sensitivities
  stoch_vs_acc_channels.py      why the stochastic and acceleration channels rank differently
                                in the lens-bound vs line-of-sight figures

  # ---- data ----
  macro_lens_results.npz        B1422+231 macro-lens outputs (written by macro_lens.ipynb)
  Delta2_output.csv             CLASS linear + halofit nonlinear spectrum
  Delta2_extended_output.csv    the same, extrapolated to k = 1e10 h/Mpc

figs/     the eight paper figures, plus requirements.pdf
```

The standalone scripts only print to stdout; they write no files. Like the notebooks,
they resolve their inputs (`macro_lens_results.npz`, the `Delta2_*.csv` spectra) relative
to `code/`, so run them from there, e.g.:

```bash
cd code
python impact_numbers.py
```

---

## Installation

```bash
git clone https://github.com/kenvantilburg/quaDM_lens.git
cd quaDM_lens
conda env create -f environment.yml
conda activate quaDM
jupyter lab
```

Requirements beyond the conda environment:

- **A LaTeX installation** with `latex` and `dvipng` on the `PATH`. All figures use
  matplotlib's `text.usetex = True` (set in `code/preamble.py`).
- **`classy`** (the CLASS Python wrapper), only if you want to regenerate the
  power-spectrum CSVs with `Pnl.ipynb`. Not needed otherwise —
  `Delta2_output.csv` and `Delta2_extended_output.csv` are committed.

---

## Reproducing the figures

Run from within `code/` (the notebooks use paths relative to it, and write to `../figs/`):

```bash
cd code
jupyter nbconvert --to notebook --execute --inplace macro_lens.ipynb
jupyter nbconvert --to notebook --execute --inplace sensitivity.ipynb
jupyter nbconvert --to notebook --execute --inplace concept.ipynb
jupyter nbconvert --to notebook --execute --inplace matter_power.ipynb
jupyter nbconvert --to notebook --execute --inplace requirements.ipynb
```

`macro_lens.ipynb` must run first: it writes `macro_lens_results.npz`, which
`sensitivity.ipynb` and `concept.ipynb` both read. The other notebooks are independent of
one another. Total runtime is a few minutes on a laptop.

**Within `macro_lens.ipynb` and `sensitivity.ipynb`, run the cells top to bottom.**
Part II of each notebook begins with `from params_J1029_2623 import *`, which deliberately
rebinds the system-level names (`d_lens`, `z_lens`, `kappa_fit`, `inv_jacobian_fit`,
`labels`, `kappa_star`, …) from the galaxy lens to the cluster lens. Re-running a Part I
cell after Part II therefore requires re-running Part I from the top. This is what lets the
two systems share one code path, rather than one system's code being a copy of the other's.

`requirements.ipynb` is a thin summary layer: it takes the signal and noise budget at the
reference precision from `sensitivity.ipynb` and `impact_numbers.py` as hard-coded numbers
(with provenance comments) and rescales them in $\sigma_{\delta\theta}$. Update those
numbers in step with a re-run of `sensitivity.ipynb`.

---

## Conventions

- **Units.** Everything internal is in natural units built on GeV
  (`natural_units_GeV.py`); conversions such as `muas`, `year`, `pc`, `M_Solar`,
  `km/second` are applied at input and output only.
- **Lensing Jacobian.** $J^\mathrm{I} = \left(\begin{smallmatrix}
  1-\kappa-\gamma_1 & -\gamma_2 \\ -\gamma_2 & 1-\kappa+\gamma_1
  \end{smallmatrix}\right)$ and $B^\mathrm{I} = (J^\mathrm{I})^{-1}$, with
  $A = \det B^\mathrm{I}$ the signed magnification. Both parameter modules build
  `jacobian_fit` / `inv_jacobian_fit` this way, so downstream code is system-agnostic.
- **Halo profile.** A Gaussian-cutoff $1/r$ cusp with
  $M_\mathrm{L} = 4\pi\sqrt{e}\,\rho_s r_\mathrm{L}^3$ and form factor
  $F(k\gamma_\mathrm{L}) = e^{-(k\gamma_\mathrm{L})^2/2}$, whose angular integrals close in
  terms of $\mathrm{erfc}$ (`C_ij_integral`). A finite source enters in quadrature,
  $x_k \to \sqrt{x_k^2 + x_\mathrm{src}^2}$ (`C_ij_integral_src`). $r_\mathrm{L}$ is the
  microhalo radius throughout; $r_s$ is reserved for the NFW/Einasto scale radius of the
  $\Lambda$CDM prediction curves. The one exception is the prompt cusps: as in App. Prompt
  Cusps, they use the truncated, cored $r^{-3/2}$ transform instead of the Gaussian-cutoff
  form factor (`F2_cusp` in `cusp_snr.py`, `_cusp_FT2` in `matter_power.ipynb`), which
  matters once $k\,r_\mathrm{cusp}\gtrsim1$.
- **Sweep rate.** The image sweeps across a halo's deflection field at
  $\mathrm{d}(\theta^\mathrm{I}-\theta_\mathrm{L})/\mathrm{d}t =
  B^\mathrm{I}\mu - \mu_\mathrm{L}$: only the *bulk* lens motion $\mu$ is magnified by
  $B^\mathrm{I}$; a halo's own orbital motion $\mu_\mathrm{L}$ moves the deflector rather
  than the image and so enters **unmagnified** (`mu_L_int` in both parameter modules).
- **Subscript `L`.** As in the paper, `_lens` / `z_lens` / `d_lens` refer to the **macro**
  lens (galaxy or cluster), while the suffix `_L` (`M_L`, `r_L`, `gamma_L`, `kappa_L`,
  `theta_E_L`, `mu_L_int`) refers to a **microhalo**. The paper overloads the subscript
  $\mathrm{L}$ for both; the code keeps them apart.
- **Position angles.** `PA_*` are astronomical position angles, **east of north** (the
  convention of Sluse+ 2012 and of `kappa_DV` / `kappa_SIE`). `phi_*` are `lenstronomy`
  angles, counterclockwise from $+\Delta\mathrm{RA}$, $\phi = 90^\circ - \mathrm{PA}$.
- **SNR.** All quoted SNRs are *variance* signal-to-noise ratios (signal covariance over
  noise covariance on the same statistic), and should not be interpreted as
  Gaussian-$\sigma$ significances.

### Symbols

The paper is the ground truth for notation; the code either uses the same letter or a
spelled-out version of it. The non-obvious correspondences:

| Paper | Code | Meaning |
|---|---|---|
| $J^\mathrm{I}_{ij}$ | `jacobian_fit` | lensing Jacobian $\partial\varphi_i/\partial\theta_j$ |
| $B^\mathrm{I}_{ij}$ | `inv_jacobian_fit` | inverse Jacobian $(J^\mathrm{I})^{-1}$ |
| $A^\mathrm{I}$ | `mag_fit` | signed magnification $\det B^\mathrm{I}$ |
| $B^\mathrm{I}_\mathrm{t}$ | `B_tang` | tangential eigenvalue of $B^\mathrm{I}$ |
| $H = \nabla\alpha$ | `hessian_fit` | deflection gradient, $J = \mathbb{1} - H$ |
| $\mu$, $\widetilde{\mu}^\mathrm{I}$, $\zeta^\mathrm{I}$ | `vec_mu`, `mu_tilde`, `zeta` | source-plane proper motion, image proper motion, image pm's position angle |
| $D_\mathrm{L},D_\mathrm{S},D_\mathrm{LS}$ | `d_lens`, `d_source`, `d_lens_source` | **angular-diameter** distances (lowercase `d`) |
| $\chi_\mathrm{L},\chi_\mathrm{S}$ | `D_lens`, `D_source` | **comoving** distances (uppercase `D`) |
| $\widetilde{C}^\mathrm{I}_{pq}(\omega)$ | `C_ij` / `C_pq` | PSD before / after contraction with $B^\mathrm{I}$ |
| $\mathcal{F}_1,\mathcal{F}_2$ | `smear_F1`, `smear_F2` | estimator smearing factors |
| $\tau, N, \sigma_{\delta\theta}$ | `tau`, `N_obs`, `sigma_delta_theta` | survey baseline, epochs, precision |
| $\theta_\mathrm{src}$, $\theta^\mathrm{I}_\mathrm{src}$ | `theta_src`, `theta_src_I` | unlensed / magnified source size |
| $\theta_{\mathrm{E},*}$, $M_*$, $\theta_\mathrm{fit}$ | `theta_E_star`, `M_star`, `theta_fit` | stellar microlensing |
| $x_\mathrm{sub}$ | `x_sub_fid` | host-centric distance $R_\mathrm{sub}/R_{200}^\mathrm{host}$ of the Moline+2017 relation |
| $\overline{\rho}_{m,0}$ | `rho_M_0` | mean matter density today (from `natural_units_GeV`) |
| $\mathcal{K}_\mathrm{S}$, $\Lambda_\mathrm{S}$ | `K_S`, `Lambda_S` | line-of-sight Limber integrals |
| $N_\mathrm{eff}$ | — | effective number of **DFT modes** (Sec. II E); not to be confused with the weighted count of contributing perturbers behind Sec. II C's Gaussianity argument, which is `N_perturb` in `cusp_snr.py` |

---

## License

[MIT](LICENSE). If you use this code, please cite the paper. `Claude Code` was used for
some of the code development and documentation, but Ken Van Tilburg has manually reviewed
all lines of code and bears responsibility for any mistakes. If you see any, please notify
me at `kenvt@stanford.edu`!

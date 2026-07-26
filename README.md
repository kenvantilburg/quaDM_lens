# quaDM_lens

**Astrometric weak lensing of multiply-imaged quasars by substellar dark matter halos.**

Code and figures for

> K. Van Tilburg and D. E. Kaplan,
> *Discovering Substellar Dark Matter Halos with Astrometric Weak Lensing of
> Multiply-Imaged Quasars*.

Every figure in the paper is produced by a notebook in [`code/`](code); the map
from figure to notebook is in [Figures](#figures) below, and each figure caption
in the paper links directly to its notebook.

---

## What this computes

A strongly lensed quasar's macro-images sweep across the dark matter substructure
of the lens (and of the line of sight) at $\sim10^2$–$10^3\ \mathrm{km\,s^{-1}}$.
The collective deflection by many substellar halos makes each image centroid
execute a random walk with a calculable, red-tilted power spectrum. The code here
turns that statement into a forecast:

1. **Macro-lens model** — the local convergence $\kappa$, shear
   $(\gamma_1,\gamma_2)$ and inverse magnification tensor $B^\mathrm{I}$ at each
   image, plus the smooth stellar convergence $\kappa_*$.
2. **Astrometric response** — the power spectral density
   $\widetilde{C}^\mathrm{I}_{pq}(\omega)$ of the image-centroid random walk
   induced by a monochromatic population of halos, and its smeared $\omega^4$
   moment, the differential angular-acceleration covariance
   $\langle \delta\ddot\theta^{\mathrm{I},\mathrm{I}'}_{0,p}\,
   \delta\ddot\theta^{\mathrm{I},\mathrm{I}'}_{0,q}\rangle$.
3. **Survey sensitivity** — those signals against the instrumental noise floor of
   a $\tau$-year, $N$-epoch differential-astrometry campaign at per-epoch
   precision $\sigma_{\delta\theta}$, and against the post-subtraction stellar
   microlensing residual.
4. **Cosmological read-across** — the same reach expressed as a bound on the
   small-scale matter power spectrum $\Delta^2_\delta(k)$, alongside the
   pure-CDM and thermal-WIMP predictions.

Two benchmark systems are carried through in parallel everywhere:

| | **B1422+231** | **SDSS J1029+2623** |
|---|---|---|
| lens | galaxy | galaxy **cluster** |
| $z_\mathrm{L}$, $z_\mathrm{S}$ | 0.34, 3.62 | 0.588, 2.199 |
| images | 4 (A,B,C,D) | 3 (A,B,C) |
| image separation | $\sim1''$ | $22.5''$ |
| brightest-pair magnification | $\lvert A\rvert \approx 7$–9 | $\lvert A\rvert \approx 22$ |
| stellar convergence $\kappa_*$ | $\approx 0.05$ | $\approx 0.003$ |
| effective lens velocity | 865 km/s (bulk) | 552 km/s (bulk + moving clumps) |
| assumed precision $\sigma_{\delta\theta}$ | $0.1\ \mu$as (EPIC) | $1\ \mu$as |
| parameters | [`code/params_B1422_231.py`](code/params_B1422_231.py) | [`code/params_J1029_2623.py`](code/params_J1029_2623.py) |

The galaxy-lens and cluster-lens versions of a given figure live in the **same**
notebook, as Part I and Part II, so the two systems are computed by identical
code and differ only through their parameter modules.

---

## Figures

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
| 8 | [`figs/matter_power.pdf`](figs/matter_power.pdf) | [`code/matter_power.ipynb`](code/matter_power.ipynb) | Matter power spectrum $\Delta^2_\delta(k)$ with the line-of-sight reach of **both** campaigns |
| 9 | [`figs/requirements.pdf`](figs/requirements.pdf) | [`code/requirements.ipynb`](code/requirements.ipynb) | $\Lambda$CDM-prediction SNR vs astrometric precision for **both** systems |

The appendix tables are printed by [`code/macro_lens.ipynb`](code/macro_lens.ipynb):
`tab:macro` (B1422+231) at the end of Part I, `tab:macro-cluster`
(SDSS J1029+2623) at the end of Part II.

The animations in `figs/animation_lensing_*.mp4` illustrate stellar microlensing
of the quasar image and are produced by
[`code/micro_lens.ipynb`](code/micro_lens.ipynb); they do not appear in the paper.

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
  requirements.ipynb    reach vs astrometric precision -> figs/requirements.pdf
  Pnl.ipynb             CLASS + halofit -> Delta2_output.csv, Delta2_extended_output.csv
                        (needs `classy`; the CSVs are committed, so this is optional)
  micro_lens.ipynb      stellar-microlensing image illustrations and animations (needs ffmpeg)

  # ---- shared modules ----
  preamble.py               imports, matplotlib/LaTeX setup, color palettes
  natural_units_GeV.py      natural-unit constants and conversions (GeV-based)
  macro_lens_functions.py   lensing geometry: distances, Sigma_crit, theta_E, mu_rel,
                            de Vaucouleurs kappa_*, stellar and smooth lens potentials
  sensitivity_functions.py  halo profiles (Einasto, Gaussian-cutoff cusp), form factors,
                            C_ij_integral / C_ij_integral_src, instrumental noise PSD
  params_B1422_231.py       galaxy-lens system parameters, velocity budget, source size
  params_J1029_2623.py      cluster-lens system parameters (Acebron+2022 model)

  # ---- standalone analysis scripts (print numbers quoted in the text) ----
  impact_numbers.py             headline SNRs along the fiducial CDM rho_s(M) relation,
                                both systems, instrument-limited and with the stellar residual
  mu_expected.py                relative proper-motion / velocity budget for both systems
  cusp_snr.py                   acceleration SNR of the lens-bound prompt-cusp population
  los_full_spectrum_snr.py      line-of-sight SNRs of the predicted CDM / WIMP / cusp spectra
  los_vs_lens_decomposition.py  reconciles the lens-bound and line-of-sight sensitivities
  stoch_vs_acc_channels.py      why the stochastic and acceleration channels rank differently
                                in the lens-bound vs line-of-sight figures

  # ---- data ----
  macro_lens_results.npz        B1422+231 macro-lens outputs (written by macro_lens.ipynb)
  Delta2_output.csv             CLASS linear + halofit nonlinear spectrum
  Delta2_extended_output.csv    the same, extrapolated to k = 1e10 h/Mpc

figs/     all paper figures (PDF) and the microlensing animations (MP4)
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

- **A LaTeX installation** with `latex` and `dvipng` on the `PATH`. All figures
  use matplotlib's `text.usetex = True` (set in `code/preamble.py`).
- **`ffmpeg`**, only for the animation cells of `micro_lens.ipynb`.
- **`classy`** (the CLASS Python wrapper), only if you want to regenerate the
  power-spectrum CSVs with `Pnl.ipynb`. Not needed otherwise —
  `Delta2_output.csv` and `Delta2_extended_output.csv` are committed.

---

## Reproducing the figures

Run from within `code/` (the notebooks use paths relative to it, and write to
`../figs/`):

```bash
cd code
jupyter nbconvert --to notebook --execute --inplace macro_lens.ipynb
jupyter nbconvert --to notebook --execute --inplace sensitivity.ipynb
jupyter nbconvert --to notebook --execute --inplace concept.ipynb
jupyter nbconvert --to notebook --execute --inplace matter_power.ipynb
jupyter nbconvert --to notebook --execute --inplace requirements.ipynb
```

`macro_lens.ipynb` must run first: it writes `macro_lens_results.npz`, which
`sensitivity.ipynb`, `concept.ipynb` and `micro_lens.ipynb` all read. The other
notebooks are independent of one another. Total runtime is a few minutes on a
laptop.

**Within `macro_lens.ipynb` and `sensitivity.ipynb`, run the cells top to bottom.**
Part II of each notebook begins with `from params_J1029_2623 import *`, which
deliberately rebinds the system-level names (`d_lens`, `z_lens`, `kappa_fit`,
`inv_jacobian_fit`, `labels`, `kappa_star`, …) from the galaxy lens to the
cluster lens. Re-running a Part I cell after Part II therefore requires
re-running Part I from the top. This is what lets the two systems share one code
path, rather than one system's code being a copy of the other's.

`requirements.ipynb` is a thin summary layer: it takes the signal and noise
budget at the reference precision from `sensitivity.ipynb` and
`impact_numbers.py` as hard-coded numbers (with provenance comments) and rescales
them in $\sigma_{\delta\theta}$. Update those numbers in step with a re-run of
`sensitivity.ipynb`.

---

## Conventions

- **Units.** Everything internal is in natural units built on GeV
  (`natural_units_GeV.py`); conversions such as `muas`, `year`, `pc`, `M_Solar`,
  `km/second` are applied at input and output only.
- **Lensing Jacobian.** $A^\mathrm{I} = \left(\begin{smallmatrix}
  1-\kappa-\gamma_1 & -\gamma_2 \\ -\gamma_2 & 1-\kappa+\gamma_1
  \end{smallmatrix}\right)$ and $B^\mathrm{I} = (A^\mathrm{I})^{-1}$, with
  $\det B^\mathrm{I}$ the signed magnification. Both parameter modules build
  `jacobian_fit` / `inv_jacobian_fit` this way, so downstream code is
  system-agnostic.
- **Halo profile.** A Gaussian-cutoff $1/r$ cusp with
  $M_\mathrm{L} = 4\pi\sqrt{e}\,\rho_s r_s^3$ and form factor
  $F(k) = e^{-k^2/2}$, whose angular integrals close in terms of $\mathrm{erfc}$
  (`C_ij_integral`). A finite source enters in quadrature,
  $x_k \to \sqrt{x_k^2 + x_\mathrm{src}^2}$ (`C_ij_integral_src`).
- **Sweep rate.** The image sweeps across a halo's deflection field at
  $\mathrm{d}(\theta^\mathrm{I}-\theta_h)/\mathrm{d}t =
  B^\mathrm{I}\mu_\mathrm{bulk} - \mu_{h,\mathrm{int}}$: only the *bulk* lens
  motion is magnified by $B^\mathrm{I}$; a halo's own orbital motion moves the
  deflector rather than the image and so enters **unmagnified**.
- **SNR.** All quoted SNRs are *variance* signal-to-noise ratios (signal
  covariance over noise covariance on the same statistic), not Gaussian-$\sigma$
  significances.

---

## License

[MIT](LICENSE). If you use this code, please cite the paper.

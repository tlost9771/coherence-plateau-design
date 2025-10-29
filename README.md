# Coherence Plateau Figures

This repository generates all figures (PDF) for the manuscript:

**Closed-Form Design of Coherence Plateaus under Correlated Non-Markovian Noise and Continuous Drive**  
**Author:** Sajjad Saei — Department of Theoretical Physics and Astrophysics, University of Tabriz, Tabriz, Iran

## What is included
- `make_figures.py`: single entry-point script that reproduces all figures as vector **PDF** under `./figures/`.
- `requirements.txt`: minimal Python dependencies (NumPy, Matplotlib).
- `Makefile`: convenience target to regenerate the figures.

## Figures produced
The script generates the following PDFs (one plot per figure, Matplotlib only, no explicit colors set):
- `fig2a_lambda_vs_t.pdf`: uniform attenuation \Lambda(t,\mu) vs time for \mu in {0.0, 0.5, 0.9}.
- `fig2b_heatmap_plateau.pdf`: heatmap of surrogate plateau width \Delta t_plat over (\Omega_R, \mu) at fixed \Gamma.
- `fig3_plateau_local_quad.pdf`: local plateau window around a stationary point (quadratic tolerance band).
- `fig4a_design_map_mu05.pdf`: design map of \Delta t_plat over (\Omega_R, \Delta) at \mu=0.5.
- `fig4b_criterion_regions.pdf`: analytical feasibility boundary from the sufficient plateau criterion.
- `fig4c_opt_omega_vs_mu.pdf`: optimal drive amplitude \Omega_R^* vs \mu.
- `fig4d_opt_delta_vs_mu.pdf`: optimal detuning \Delta^* vs \mu.
- `fig4e_opt_width_vs_mu.pdf`: maximum achievable \Delta t_plat vs \mu.
- `fig5a_rec_bound_vs_t.pdf`: REC-bound factor \Lambda(t,\mu) with/without drive.
- `fig5b_qfi_proxy_vs_t.pdf`: proxy \Phi(t)=\Lambda^2 vs time with/without drive.
- `fig5c_qfi_proxy_gain_vs_t.pdf`: gain ratio \Phi/\Phi_0 highlighting the plateau advantage.
- `fig6a_width_vs_mu_multiN.pdf`: \Delta t_plat vs \mu for N=1,2,3 under the uniform-N law.
- `fig6b_vtype_map.pdf`: V-type qutrit map of \Delta t_plat over (\Omega_R, w_d=\sin^2\theta).
- `fig6c_width_vs_N_at_mu05.pdf`: \max \Delta t_plat vs N at \mu=0.5.

> All plots are single-panel PDFs to comply with APS reprint constraints on float congestion.

## Reproducing the figures

### 1) Create and activate an environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Generate the PDFs
```bash
python make_figures.py
# or customize the bath linewidth:
python make_figures.py --Gamma 0.6
```

All PDFs will appear under `./figures/`.

## Notes
- The models are analytic surrogates consistent with the manuscript's resolvent picture (pseudomode/PMME)
  and the uniform attenuation law for spatial collectivity. They are designed for clarity and reproducibility.
- Each figure uses a single Matplotlib plot, avoids seaborn, and does not specify any explicit colors.
- Parameters (\Gamma, grid sizes, thresholds) can be adjusted in `make_figures.py` if needed.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py (robust, dimensionless-compliant)

Generates all PDF figures for the manuscript:

  "Closed-Form Design of Coherence Plateaus under Correlated Non-Markovian Noise and Continuous Drive"
  Author: Sajjad Saei (Department of Theoretical Physics and Astrophysics, University of Tabriz, Tabriz, Iran)

Key guarantees (per journal/figure policy):
  - Headless backend (Agg)
  - One plot per figure; no seaborn; no explicit colors
  - Plain text/Unicode labels/titles (no TeX rendering)
  - Dimensionless presentation:
      * τ = Γ t on time axes
      * Ω_R/Γ and Δ/Γ on design maps and boundaries
      * Colorbars report Δτ_plat (not Δt_plat)
  - Sufficient-criterion boundary uses κ(μ)=1+μ in dimensionless form: (Ω_R/Γ)^2 / (1+(Δ/Γ)^2) ≥ κ(μ)
"""
from __future__ import annotations
import argparse, sys, os, traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt


# ========================= Core surrogate models =========================

def driven_surrogate_amplitude(t: np.ndarray, Omega_R: float=3.0, Delta: float=0.0, Gamma: float=0.6) -> np.ndarray:
    """Under-damped drive-dressed envelope consistent with a cubic-resolvent picture (surrogate)."""
    Om_eff_sq = max(Omega_R**2 + Delta**2 - (Gamma/2.0)**2, 0.0)
    Om_eff = np.sqrt(Om_eff_sq)
    # canonical second-order underdamped-like envelope
    A = np.exp(-Gamma*t/2.0) * (np.cos(Om_eff*t) + (Gamma/(2.0*Om_eff+1e-12))*np.sin(Om_eff*t))
    return np.clip(np.abs(A), 0.0, 1.0)

def uniform_attenuation(A: np.ndarray, mu: float) -> np.ndarray:
    """Λ(t, μ) = (1-μ) A^2 + μ A  (single-/two-qubit sectors)."""
    return (1.0-mu)*A**2 + mu*A

def uniform_attenuation_N(A: np.ndarray, mu: float, N: int) -> np.ndarray:
    """N-qubit GHZ/Bell-type sector: (1-μ) A^N + μ A."""
    return (1.0-mu)*A**N + mu*A

def vtype_qutrit_lambda(t: np.ndarray, Omega_R: float, Delta: float, Gamma: float, theta: float) -> np.ndarray:
    """V-type qutrit proxy: dark weight w_d = sin^2(theta); bright branch ~ driven amplitude surrogate."""
    w_d = np.sin(theta)**2
    A_b = driven_surrogate_amplitude(t, Omega_R=Omega_R, Delta=Delta, Gamma=Gamma)
    Lam = w_d*1.0 + (1.0-w_d)*A_b
    return np.clip(Lam, 0.0, 1.0)

# ---------- Plateau width helpers ----------
def plateau_width_from_series_time(t: np.ndarray, y: np.ndarray, eps_time: float=5e-3, y_min: float=0.2) -> float:
    """Longest contiguous window (in time units of t) with |dy/dt|<=eps_time and y>=y_min."""
    dt = t[1]-t[0]
    d = np.gradient(y, dt)
    mask = (np.abs(d) <= eps_time) & (y >= y_min)
    best = cnt = 0
    for m in mask:
        cnt = cnt + 1 if m else 0
        best = max(best, cnt)
    return best*dt

def plateau_width_dimless(t: np.ndarray, y: np.ndarray, Gamma: float, eps_tau: float=5e-3, y_min: float=0.2) -> float:
    """
    Dimensionless plateau width Δτ_plat using the τ-derivative threshold |∂_τ y|<=eps_tau.

    Since ∂_τ y = (1/Γ) ∂_t y, the time-derivative threshold is eps_time = eps_tau / Γ.
    Returned width is Δτ_plat = Γ * Δt_plat.
    """
    eps_time = eps_tau / max(Gamma, 1e-12)
    width_t = plateau_width_from_series_time(t, y, eps_time=eps_time, y_min=y_min)
    return width_t * Gamma  # Δτ = Γ Δt

def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ========================= Figure generators (dimensionless presentation) =========================

def fig2a_lambda_vs_t(outpath: str, Gamma: float, Omega_R: float, Delta: float, mus, T: float, N: int):
    t = np.linspace(0, T, N)
    tau = Gamma * t
    A = driven_surrogate_amplitude(t, Omega_R=Omega_R, Delta=Delta, Gamma=Gamma)
    fig = plt.figure(figsize=(6.4, 4.2))
    for mu in mus:
        Lam = uniform_attenuation(A, mu)
        plt.plot(tau, Lam, label=f"μ={mu:.1f}")
    plt.xlabel("τ (dimensionless)")
    plt.ylabel("Λ(τ, μ)")
    plt.title("Uniform attenuation vs τ (several μ)")
    plt.legend()
    save_fig(fig, outpath)

def fig2b_heatmap_plateau(outpath: str, Gamma: float, Omega_grid, mu_grid, T: float, N: int, eps_tau: float=5e-3):
    t = np.linspace(0, T, N)
    heat = np.zeros((len(mu_grid), len(Omega_grid)))
    for i, mu in enumerate(mu_grid):
        for j, Om in enumerate(Omega_grid):
            A = driven_surrogate_amplitude(t, Omega_R=Om, Delta=0.0, Gamma=Gamma)
            Lam = uniform_attenuation(A, mu)
            heat[i, j] = plateau_width_dimless(t, Lam, Gamma=Gamma, eps_tau=eps_tau, y_min=0.2)
    fig = plt.figure(figsize=(6.6, 4.8))
    extent = [Omega_grid[0]/Gamma, Omega_grid[-1]/Gamma, mu_grid[0], mu_grid[-1]]
    plt.imshow(heat, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Ω_R / Γ")
    plt.ylabel("μ")
    plt.title("Δτ_plat vs (Ω_R/Γ, μ)")
    cb = plt.colorbar()
    cb.set_label("Δτ_plat")
    save_fig(fig, outpath)

def fig3_plateau_local_quad(outpath: str, Gamma: float, Omega_R: float, Delta: float, mu: float, T: float, N: int):
    t = np.linspace(0, T, N)
    tau = Gamma * t
    A = driven_surrogate_amplitude(t, Omega_R=Omega_R, Delta=Delta, Gamma=Gamma)
    Lam = uniform_attenuation(A, mu)
    dLam_dt = np.gradient(Lam, t[1]-t[0])
    idx = int(np.argmin(np.abs(dLam_dt)))
    # local tolerance band in Λ for display
    eta = 0.02
    center = Lam[idx]
    mask = np.abs(Lam - center) <= eta
    left = idx
    while left > 0 and mask[left-1]:
        left -= 1
    right = idx
    while right < len(t)-1 and mask[right+1]:
        right += 1
    fig = plt.figure(figsize=(6.4, 4.0))
    plt.plot(tau, Lam, label="Λ(τ, μ)")
    plt.axvspan(tau[left], tau[right], alpha=0.2)
    plt.xlabel("τ (dimensionless)")
    plt.ylabel("Λ(τ, μ)")
    plt.title("Local plateau window around a stationary point")
    save_fig(fig, outpath)

def fig4a_design_map_mu05(outpath: str, Gamma: float, mu: float, T: float, N: int, Om_grid, De_grid, eps_tau: float=5e-3):
    t = np.linspace(0, T, N)
    heat = np.zeros((len(De_grid), len(Om_grid)))
    for i, De in enumerate(De_grid):
        for j, Om in enumerate(Om_grid):
            A = driven_surrogate_amplitude(t, Omega_R=Om, Delta=De, Gamma=Gamma)
            Lam = uniform_attenuation(A, mu)
            heat[i, j] = plateau_width_dimless(t, Lam, Gamma=Gamma, eps_tau=eps_tau, y_min=0.2)
    fig = plt.figure(figsize=(6.4, 4.8))
    extent = [Om_grid[0]/Gamma, Om_grid[-1]/Gamma, De_grid[0]/Gamma, De_grid[-1]/Gamma]
    plt.imshow(heat, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Ω_R / Γ")
    plt.ylabel("Δ / Γ")
    plt.title(f"Design map: Δτ_plat for μ={mu:.2f} (dimensionless)")
    cb = plt.colorbar()
    cb.set_label("Δτ_plat")
    save_fig(fig, outpath)

def fig4b_criterion_regions(outpath: str, mus, De_tilde_span=(-3.0, 3.0), num=600):
    def kappa_mu(mu): return 1.0 + mu  # within bounds [1,2]
    De_tilde = np.linspace(De_tilde_span[0], De_tilde_span[1], num)
    fig = plt.figure(figsize=(6.4, 4.8))
    for mu in mus:
        kappa = kappa_mu(mu)
        Om_tilde_boundary = np.sqrt((1.0 + De_tilde**2) * kappa)
        plt.plot(Om_tilde_boundary, De_tilde, label=f"μ={mu:.1f}")
    plt.xlabel("Ω_R / Γ")
    plt.ylabel("Δ / Γ")
    plt.title("Plateau criterion boundary (dimensionless)")
    plt.legend()
    save_fig(fig, outpath)

def fig4cde_opts_vs_mu(out_Om: str, out_De: str, out_W: str, Gamma: float, T: float, N: int, mus, Om_grid, De_grid, eps_tau: float=5e-3):
    t = np.linspace(0, T, N)
    Om_star = np.zeros_like(mus, dtype=float)
    De_star = np.zeros_like(mus, dtype=float)
    W_star  = np.zeros_like(mus, dtype=float)
    for i, mu in enumerate(mus):
        bestW = -1.0
        bestOm = bestDe = None
        for De in De_grid:
            for Om in Om_grid:
                A = driven_surrogate_amplitude(t, Omega_R=Om, Delta=De, Gamma=Gamma)
                Lam = uniform_attenuation(A, mu)
                width_tau = plateau_width_dimless(t, Lam, Gamma=Gamma, eps_tau=eps_tau, y_min=0.2)
                if width_tau > bestW:
                    bestW, bestOm, bestDe = width_tau, Om, De
        W_star[i], Om_star[i], De_star[i] = bestW, bestOm/Gamma, bestDe/Gamma  # store dimensionless
    # Ω*/Γ vs μ
    fig = plt.figure(figsize=(6.4, 4.2))
    plt.plot(mus, Om_star, marker="o", linewidth=1.0)
    plt.xlabel("μ"); plt.ylabel("Ω_R*/Γ")
    plt.title("Optimal drive amplitude (dimensionless) vs μ")
    save_fig(fig, out_Om)
    # Δ*/Γ vs μ
    fig = plt.figure(figsize=(6.4, 4.2))
    plt.plot(mus, De_star, marker="o", linewidth=1.0)
    plt.xlabel("μ"); plt.ylabel("Δ*/Γ")
    plt.title("Optimal detuning (dimensionless) vs μ")
    save_fig(fig, out_De)
    # max Δτ_plat vs μ
    fig = plt.figure(figsize=(6.4, 4.2))
    plt.plot(mus, W_star, marker="o", linewidth=1.0)
    plt.xlabel("μ"); plt.ylabel("max Δτ_plat")
    plt.title("Maximum achievable plateau width vs μ (dimensionless)")
    save_fig(fig, out_W)

def fig5_metrology(outA: str, outB: str, outC: str, Gamma: float, mu: float, T: float, N: int):
    t = np.linspace(0, T, N)
    tau = Gamma * t
    A_drive = driven_surrogate_amplitude(t, Omega_R=3.0, Delta=0.0, Gamma=Gamma)
    A_nodrv = np.exp(-Gamma*t/2.0)
    Lam_drive = uniform_attenuation(A_drive, mu)
    Lam_nodrv = uniform_attenuation(A_nodrv, mu)

    # (a) REC envelope factor Λ(τ, μ)
    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(tau, Lam_drive, label="Λ(τ, μ) with drive")
    plt.plot(tau, Lam_nodrv, label="Λ0(τ, μ) no drive", linestyle="--")
    plt.xlabel("τ (dimensionless)")
    plt.ylabel("Λ(τ, μ)  (REC envelope factor)")
    plt.title("REC envelope: C_r ≤ Λ · C_r(0)")
    plt.legend()
    save_fig(fig, outA)

    # (b) Φ(τ) = Λ^2 (proxy)
    Phi_drive = Lam_drive**2
    Phi_nodrv = Lam_nodrv**2
    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(tau, Phi_drive, label="Φ(τ) = Λ^2 with drive")
    plt.plot(tau, Phi_nodrv, label="Φ0(τ) = Λ0^2 no drive", linestyle="--")
    plt.xlabel("τ (dimensionless)")
    plt.ylabel("Φ(τ) (proxy)")
    plt.title("Coherence-squared proxy for phase-like tasks")
    plt.legend()
    save_fig(fig, outB)

    # (c) Relative proxy gain
    gain = np.divide(Phi_drive, Phi_nodrv, out=np.ones_like(Phi_drive), where=Phi_nodrv>1e-12)
    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(tau, gain, label="proxy gain = Φ / Φ0")
    plt.xlabel("τ (dimensionless)")
    plt.ylabel("proxy gain")
    plt.title("Relative proxy gain inside plateau window")
    plt.legend()
    save_fig(fig, outC)

def fig6_multi_and_vtype(outA: str, outB: str, outC: str, Gamma: float, T: float, N: int, eps_tau: float=5e-3):
    t = np.linspace(0, T, N)
    mu_grid = np.linspace(0.0, 0.95, 40)
    A = driven_surrogate_amplitude(t, Omega_R=3.0, Delta=0.0, Gamma=Gamma)

    # (a) Δτ_plat vs μ for N=1,2,3
    fig = plt.figure(figsize=(6.4, 4.2))
    for Nq in [1,2,3]:
        widths_tau = []
        for mu in mu_grid:
            Lam = uniform_attenuation_N(A, mu, Nq)
            widths_tau.append(plateau_width_dimless(t, Lam, Gamma=Gamma, eps_tau=eps_tau, y_min=0.2))
        widths_tau = np.array(widths_tau)
        plt.plot(mu_grid, widths_tau, label=f"N={Nq}")
    plt.xlabel("μ"); plt.ylabel("Δτ_plat")
    plt.title("Plateau width vs μ for N=1,2,3 (dimensionless)")
    plt.legend()
    save_fig(fig, outA)

    # (b) V-type qutrit map: Δτ_plat vs (Ω_R/Γ, w_d)
    Om_grid = np.linspace(0.2, 5.0, 61)
    th_grid = np.linspace(0.0, np.pi/2, 61)
    heat = np.zeros((len(th_grid), len(Om_grid)))
    for i, th in enumerate(th_grid):
        for j, Om in enumerate(Om_grid):
            Lam = vtype_qutrit_lambda(t, Omega_R=Om, Delta=0.0, Gamma=Gamma, theta=th)
            heat[i, j] = plateau_width_dimless(t, Lam, Gamma=Gamma, eps_tau=eps_tau, y_min=0.2)
    fig = plt.figure(figsize=(6.4, 4.8))
    extent = [Om_grid[0]/Gamma, Om_grid[-1]/Gamma, 0.0, 1.0]
    plt.imshow(heat, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Ω_R / Γ"); plt.ylabel("w_d = sin^2 θ")
    plt.title("V-type qutrit: Δτ_plat vs (Ω_R/Γ, w_d)")
    cb = plt.colorbar(); cb.set_label("Δτ_plat")
    save_fig(fig, outB)

    # (c) max Δτ_plat vs N at μ=0.5 (optimized over Ω_R, Δ)
    mu = 0.5
    Ns = list(range(1, 7))
    Om_grid2 = np.linspace(0.2, 5.0, 41)
    De_grid2 = np.linspace(-2.5, 2.5, 41)
    widths_tau = []
    for Nq in Ns:
        best_tau = 0.0
        for De in De_grid2:
            for Om in Om_grid2:
                A2 = driven_surrogate_amplitude(t, Omega_R=Om, Delta=De, Gamma=Gamma)
                Lam = uniform_attenuation_N(A2, mu, Nq)
                w_tau = plateau_width_dimless(t, Lam, Gamma=Gamma, eps_tau=eps_tau, y_min=0.2)
                if w_tau > best_tau:
                    best_tau = w_tau
        widths_tau.append(best_tau)
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.plot(Ns, widths_tau, marker="o")
    plt.xlabel("N"); plt.ylabel("max Δτ_plat")
    plt.title("Maximum plateau width vs N at μ=0.5 (dimensionless)")
    save_fig(fig, outC)


# ========================= Runner with logging =========================

def _safe(name, fn, *args, **kwargs):
    try:
        print(f"[make_figures] Generating {name} ...", flush=True)
        fn(*args, **kwargs)
        print(f"[make_figures] -> {name} done.", flush=True)
    except Exception as e:
        print(f"[make_figures] !! {name} FAILED: {e}", flush=True)
        traceback.print_exc()

def run_all(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Global grids
    if args.fast:
        T, N = 6.0, 1600
        Omega_grid = np.linspace(0.4, 4.0, 31)
        mu_grid    = np.linspace(0.0, 0.95, 31)
        De_grid    = np.linspace(-2.0, 2.0, 31)
        mus_small  = np.linspace(0.0, 0.9, 10)
    else:
        T, N = 8.0, 3200
        Omega_grid = np.linspace(0.2, 5.0, 61)
        mu_grid    = np.linspace(0.0, 0.95, 61)
        De_grid    = np.linspace(-3.0, 3.0, 61)
        mus_small  = np.linspace(0.0, 0.95, 20)

    # Fig.2
    _safe("fig2a", fig2a_lambda_vs_t, os.path.join(outdir, "fig2a_lambda_vs_t.pdf"),
          Gamma=args.Gamma, Omega_R=3.0, Delta=0.0, mus=(0.0,0.5,0.9), T=T, N=N)
    _safe("fig2b", fig2b_heatmap_plateau, os.path.join(outdir, "fig2b_heatmap_plateau.pdf"),
          Gamma=args.Gamma, Omega_grid=Omega_grid, mu_grid=mu_grid, T=T, N=N)

    # Fig.3
    _safe("fig3", fig3_plateau_local_quad, os.path.join(outdir, "fig3_plateau_local_quad.pdf"),
          Gamma=args.Gamma, Omega_R=3.0, Delta=0.0, mu=0.5, T=T, N=N)

    # Fig.4
    _safe("fig4a", fig4a_design_map_mu05, os.path.join(outdir, "fig4a_design_map_mu05.pdf"),
          Gamma=args.Gamma, mu=0.5, T=T, N=N, Om_grid=Omega_grid, De_grid=De_grid)
    _safe("fig4b", fig4b_criterion_regions, os.path.join(outdir, "fig4b_criterion_regions.pdf"),
          mus=(0.0,0.5,0.9))
    _safe("fig4cde", fig4cde_opts_vs_mu,
          os.path.join(outdir, "fig4c_opt_omega_vs_mu.pdf"),
          os.path.join(outdir, "fig4d_opt_delta_vs_mu.pdf"),
          os.path.join(outdir, "fig4e_opt_width_vs_mu.pdf"),
          Gamma=args.Gamma, T=T, N=N, mus=mus_small, Om_grid=Omega_grid, De_grid=De_grid)

    # Fig.5
    _safe("fig5", fig5_metrology,
          os.path.join(outdir, "fig5a_rec_bound_vs_t.pdf"),
          os.path.join(outdir, "fig5b_qfi_proxy_vs_t.pdf"),
          os.path.join(outdir, "fig5c_qfi_proxy_gain_vs_t.pdf"),
          Gamma=args.Gamma, mu=0.5, T=T, N=N)

    # Fig.6
    _safe("fig6", fig6_multi_and_vtype,
          os.path.join(outdir, "fig6a_width_vs_mu_multiN.pdf"),
          os.path.join(outdir, "fig6b_vtype_map.pdf"),
          os.path.join(outdir, "fig6c_width_vs_N_at_mu05.pdf"),
          Gamma=args.Gamma, T=T, N=N)

    # Summary
    print("\n[make_figures] Summary")
    manifest = [
        "fig2a_lambda_vs_t.pdf",
        "fig2b_heatmap_plateau.pdf",
        "fig3_plateau_local_quad.pdf",
        "fig4a_design_map_mu05.pdf",
        "fig4b_criterion_regions.pdf",
        "fig4c_opt_omega_vs_mu.pdf",
        "fig4d_opt_delta_vs_mu.pdf",
        "fig4e_opt_width_vs_mu.pdf",
        "fig5a_rec_bound_vs_t.pdf",
        "fig5b_qfi_proxy_vs_t.pdf",
        "fig5c_qfi_proxy_gain_vs_t.pdf",
        "fig6a_width_vs_mu_multiN.pdf",
        "fig6b_vtype_map.pdf",
        "fig6c_width_vs_N_at_mu05.pdf",
    ]
    for name in manifest:
        status = "OK" if os.path.exists(os.path.join(outdir, name)) else "MISSING"
        print(f"  - {name}: {status}")

def main():
    ap = argparse.ArgumentParser(description="Generate all manuscript figures as PDF (robust, dimensionless).")
    ap.add_argument("--Gamma", type=float, default=0.6, help="Bath linewidth Γ (default 0.6)")
    ap.add_argument("--outdir", type=str, default="figures", help="Output directory for PDFs")
    ap.add_argument("--fast", action="store_true", help="Use smaller grids for a quick sanity run")
    args = ap.parse_args()
    run_all(args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py (robust)

Generates all PDF figures for the manuscript:

  "Closed-Form Design of Coherence Plateaus under Correlated Non-Markovian Noise and Continuous Drive"
  Author: Sajjad Saei (Department of Theoretical Physics and Astrophysics, University of Tabriz, Tabriz, Iran)

Improvements:
  - Headless backend (Agg)
  - try/except per figure + progress logs + summary
  - --fast flag for quick sanity runs
  - No seaborn; one plot per figure; no explicit colors
  - Titles/labels use plain text/Unicode (no TeX)
"""
from __future__ import annotations
import argparse, sys, os, traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt


# ========================= Core surrogate models =========================

def driven_surrogate_amplitude(t: np.ndarray, Omega_R: float=3.0, Delta: float=0.0, Gamma: float=0.6) -> np.ndarray:
    """Under-damped drive-dressed envelope consistent with a cubic resolvent picture."""
    Om_eff_sq = max(Omega_R**2 + Delta**2 - (Gamma/2.0)**2, 0.0)
    Om_eff = np.sqrt(Om_eff_sq)
    A = np.exp(-Gamma*t/2.0) * (np.cos(Om_eff*t) + (Gamma/(2.0*Om_eff+1e-12))*np.sin(Om_eff*t))
    return np.clip(np.abs(A), 0.0, 1.0)

def uniform_attenuation(A: np.ndarray, mu: float) -> np.ndarray:
    """Lambda(t, mu) = (1-mu) A^2 + mu A"""
    return (1.0-mu)*A**2 + mu*A

def uniform_attenuation_N(A: np.ndarray, mu: float, N: int) -> np.ndarray:
    """N-qubit GHZ/Bell-type off-diagonal generalization: (1-mu) A^N + mu A"""
    return (1.0-mu)*A**N + mu*A

def vtype_qutrit_lambda(t: np.ndarray, Omega_R: float, Delta: float, Gamma: float, theta: float) -> np.ndarray:
    """V-type qutrit proxy: dark weight w_d = sin^2(theta); bright branch ~ driven amplitude."""
    w_d = np.sin(theta)**2
    A_b = driven_surrogate_amplitude(t, Omega_R=Omega_R, Delta=Delta, Gamma=Gamma)
    Lam = w_d*1.0 + (1.0-w_d)*A_b
    return np.clip(Lam, 0.0, 1.0)

def plateau_width_from_series(t: np.ndarray, y: np.ndarray, eps: float=5e-3, y_min: float=0.2) -> float:
    """Longest contiguous window with |dy/dt|<=eps and y>=y_min."""
    dt = t[1]-t[0]
    d = np.gradient(y, dt)
    mask = (np.abs(d) <= eps) & (y >= y_min)
    best = cnt = 0
    for m in mask:
        cnt = cnt + 1 if m else 0
        best = max(best, cnt)
    return best*dt

def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ========================= Figure generators =========================

def fig2a_lambda_vs_t(outpath: str, Gamma: float, Omega_R: float, Delta: float, mus, T: float, N: int):
    t = np.linspace(0, T, N)
    A = driven_surrogate_amplitude(t, Omega_R=Omega_R, Delta=Delta, Gamma=Gamma)
    fig = plt.figure(figsize=(6.4, 4.2))
    for mu in mus:
        Lam = uniform_attenuation(A, mu)
        plt.plot(t, Lam, label=f"mu={mu:.1f}")
    plt.xlabel("t (arb. units)")
    plt.ylabel("Lambda(t, mu)")
    plt.title("Uniform attenuation vs time (several mu)")
    plt.legend()
    save_fig(fig, outpath)

def fig2b_heatmap_plateau(outpath: str, Gamma: float, Omega_grid, mu_grid, T: float, N: int):
    t = np.linspace(0, T, N)
    heat = np.zeros((len(mu_grid), len(Omega_grid)))
    for i, mu in enumerate(mu_grid):
        for j, Om in enumerate(Omega_grid):
            A = driven_surrogate_amplitude(t, Omega_R=Om, Delta=0.0, Gamma=Gamma)  # <- uses passed Gamma
            Lam = uniform_attenuation(A, mu)
            heat[i, j] = plateau_width_from_series(t, Lam, eps=5e-3, y_min=0.2)
    fig = plt.figure(figsize=(6.6, 4.8))
    extent = [Omega_grid[0], Omega_grid[-1], mu_grid[0], mu_grid[-1]]
    plt.imshow(heat, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Omega_R")
    plt.ylabel("mu")
    plt.title("Surrogate plateau width vs (Omega_R, mu)")
    cb = plt.colorbar()
    cb.set_label("Delta t_plat")
    save_fig(fig, outpath)

def fig3_plateau_local_quad(outpath: str, Gamma: float, Omega_R: float, Delta: float, mu: float, T: float, N: int):
    t = np.linspace(0, T, N)
    A = driven_surrogate_amplitude(t, Omega_R=Omega_R, Delta=Delta, Gamma=Gamma)
    Lam = uniform_attenuation(A, mu)
    dLam = np.gradient(Lam, t[1]-t[0])
    idx = int(np.argmin(np.abs(dLam)))
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
    plt.plot(t, Lam, label="Lambda(t, mu)")
    plt.axvspan(t[left], t[right], alpha=0.2)
    plt.xlabel("t (arb. units)")
    plt.ylabel("Lambda(t, mu)")
    plt.title("Local plateau window around a stationary point")
    save_fig(fig, outpath)

def fig4a_design_map_mu05(outpath: str, Gamma: float, mu: float, T: float, N: int, Om_grid, De_grid):
    t = np.linspace(0, T, N)
    heat = np.zeros((len(De_grid), len(Om_grid)))
    for i, De in enumerate(De_grid):
        for j, Om in enumerate(Om_grid):
            A = driven_surrogate_amplitude(t, Omega_R=Om, Delta=De, Gamma=Gamma)
            Lam = uniform_attenuation(A, mu)
            heat[i, j] = plateau_width_from_series(t, Lam)
    fig = plt.figure(figsize=(6.4, 4.8))
    extent = [Om_grid[0], Om_grid[-1], De_grid[0], De_grid[-1]]
    plt.imshow(heat, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Omega_R")
    plt.ylabel("Delta")
    plt.title(f"Design map: Delta t_plat for mu={mu:.2f}, Gamma={Gamma:.2f}")
    cb = plt.colorbar()
    cb.set_label("Delta t_plat")
    save_fig(fig, outpath)

def fig4b_criterion_regions(outpath: str, Gamma: float, mus):
    def alpha_mu(mu): return 1.0 + mu
    De = np.linspace(-3.0, 3.0, 600)
    fig = plt.figure(figsize=(6.4, 4.8))
    for mu in mus:
        alpha = alpha_mu(mu)
        Om_boundary = np.sqrt(Gamma*(De**2 + Gamma**2)/alpha)
        plt.plot(Om_boundary, De, label=f"mu={mu:.1f}")
    plt.xlabel("Omega_R")
    plt.ylabel("Delta")
    plt.title("Plateau criterion boundary")
    plt.legend()
    save_fig(fig, outpath)

def fig4cde_opts_vs_mu(out_Om: str, out_De: str, out_W: str, Gamma: float, T: float, N: int, mus, Om_grid, De_grid):
    t = np.linspace(0, T, N)
    Om_star = np.zeros_like(mus)
    De_star = np.zeros_like(mus)
    W_star  = np.zeros_like(mus)
    for i, mu in enumerate(mus):
        bestW = -1.0
        bestOm = bestDe = None
        for De in De_grid:
            for Om in Om_grid:
                A = driven_surrogate_amplitude(t, Omega_R=Om, Delta=De, Gamma=Gamma)
                Lam = uniform_attenuation(A, mu)
                width = plateau_width_from_series(t, Lam)
                if width > bestW:
                    bestW, bestOm, bestDe = width, Om, De
        W_star[i], Om_star[i], De_star[i] = bestW, bestOm, bestDe

    fig = plt.figure(figsize=(6.4, 4.2))
    plt.plot(mus, Om_star, marker="o", linewidth=1.0)
    plt.xlabel("mu"); plt.ylabel("Omega_R*")
    plt.title("Optimal drive amplitude vs mu")
    save_fig(fig, out_Om)

    fig = plt.figure(figsize=(6.4, 4.2))
    plt.plot(mus, De_star, marker="o", linewidth=1.0)
    plt.xlabel("mu"); plt.ylabel("Delta*")
    plt.title("Optimal detuning vs mu")
    save_fig(fig, out_De)

    fig = plt.figure(figsize=(6.4, 4.2))
    plt.plot(mus, W_star, marker="o", linewidth=1.0)
    plt.xlabel("mu"); plt.ylabel("max Delta t_plat")
    plt.title("Maximum achievable plateau width vs mu")
    save_fig(fig, out_W)

def fig5_metrology(outA: str, outB: str, outC: str, Gamma: float, mu: float, T: float, N: int):
    t = np.linspace(0, T, N)
    A_drive = driven_surrogate_amplitude(t, Omega_R=3.0, Delta=0.0, Gamma=Gamma)
    A_nodrv = np.exp(-Gamma*t/2.0)
    Lam_drive = uniform_attenuation(A_drive, mu)
    Lam_nodrv = uniform_attenuation(A_nodrv, mu)

    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(t, Lam_drive, label="Lambda_eff(t, mu) with drive")
    plt.plot(t, Lam_nodrv, label="Lambda0(t, mu) no drive", linestyle="--")
    plt.xlabel("t (arb. units)")
    plt.ylabel("Lambda(t, mu)  (REC bound factor)")
    plt.title("REC bound:  Cr(t) ≤ Lambda(t, μ) · Cr(0)")
    plt.legend()
    save_fig(fig, outA)

    Phi_drive = Lam_drive**2
    Phi_nodrv = Lam_nodrv**2
    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(t, Phi_drive, label="Phi(t)=Lambda^2 with drive")
    plt.plot(t, Phi_nodrv, label="Phi0(t)=Lambda0^2 no drive", linestyle="--")
    plt.xlabel("t (arb. units)")
    plt.ylabel("Phi(t) (QFI proxy)")
    plt.title("Coherence-squared proxy for phase metrology")
    plt.legend()
    save_fig(fig, outB)

    gain = np.divide(Phi_drive, Phi_nodrv, out=np.ones_like(Phi_drive), where=Phi_nodrv>1e-12)
    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(t, gain, label="proxy gain = Phi / Phi0")
    plt.xlabel("t (arb. units)")
    plt.ylabel("proxy gain")
    plt.title("Relative metrological proxy gain inside plateau")
    plt.legend()
    save_fig(fig, outC)

def fig6_multi_and_vtype(outA: str, outB: str, outC: str, Gamma: float, T: float, N: int):
    t = np.linspace(0, T, N)
    mu_grid = np.linspace(0.0, 0.95, 40)
    A = driven_surrogate_amplitude(t, Omega_R=3.0, Delta=0.0, Gamma=Gamma)

    fig = plt.figure(figsize=(6.4, 4.2))
    for Nq in [1,2,3]:
        widths = []
        for mu in mu_grid:
            Lam = uniform_attenuation_N(A, mu, Nq)
            widths.append(plateau_width_from_series(t, Lam))
        widths = np.array(widths)
        plt.plot(mu_grid, widths, label=f"N={Nq}")
    plt.xlabel("mu"); plt.ylabel("Delta t_plat")
    plt.title("Plateau width vs mu for N=1,2,3")
    plt.legend()
    save_fig(fig, outA)

    Om_grid = np.linspace(0.2, 5.0, 61)
    th_grid = np.linspace(0.0, np.pi/2, 61)
    heat = np.zeros((len(th_grid), len(Om_grid)))
    for i, th in enumerate(th_grid):
        for j, Om in enumerate(Om_grid):
            Lam = vtype_qutrit_lambda(t, Omega_R=Om, Delta=0.0, Gamma=Gamma, theta=th)
            heat[i, j] = plateau_width_from_series(t, Lam)
    fig = plt.figure(figsize=(6.4, 4.8))
    extent = [Om_grid[0], Om_grid[-1], 0.0, 1.0]
    plt.imshow(heat, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Omega_R"); plt.ylabel("w_d = sin^2(theta)")
    plt.title("V-type qutrit: Delta t_plat vs (Omega_R, w_d)")
    cb = plt.colorbar(); cb.set_label("Delta t_plat")
    save_fig(fig, outB)

    mu = 0.5
    Ns = list(range(1, 7))
    Om_grid2 = np.linspace(0.2, 5.0, 41)
    De_grid2 = np.linspace(-2.5, 2.5, 41)
    widths = []
    for Nq in Ns:
        best = 0.0
        for De in De_grid2:
            for Om in Om_grid2:
                A2 = driven_surrogate_amplitude(t, Omega_R=Om, Delta=De, Gamma=Gamma)
                Lam = uniform_attenuation_N(A2, mu, Nq)
                w = plateau_width_from_series(t, Lam)
                if w > best:
                    best = w
        widths.append(best)
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.plot(Ns, widths, marker="o")
    plt.xlabel("N"); plt.ylabel("max Delta t_plat")
    plt.title("Maximum plateau width vs N at mu=0.5")
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
          Gamma=args.Gamma, mus=(0.0,0.5,0.9))
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
    ap = argparse.ArgumentParser(description="Generate all manuscript figures as PDF (robust).")
    ap.add_argument("--Gamma", type=float, default=0.6, help="Bath linewidth (default 0.6)")
    ap.add_argument("--outdir", type=str, default="figures", help="Output directory for PDFs")
    ap.add_argument("--fast", action="store_true", help="Use smaller grids for a quick sanity run")
    args = ap.parse_args()
    run_all(args)

if __name__ == "__main__":
    main()

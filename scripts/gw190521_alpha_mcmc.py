# scripts/gw190521_alpha_mcmc.py
"""
MCMC sencillo para el ringdown de GW190521 (H1) con el modelo:

  h(t) = A220 * exp(-t / tau_220( alpha_flow )) * cos( 2π f_220(alpha_flow) t + phi220 )

donde f_220 y tau_220 se obtienen de las fórmulas GR (Berti et al.)
modificadas por alpha_flow (f -> f*(1+alpha_flow), tau -> tau*(1+beta_flow) con beta_flow=0).

Se muestrean los parámetros:
  - A220
  - phi220
  - alpha_flow

con priors:
  - A220 ~ Uniform[-10|A0|, +10|A0|]   (A0 = máximo de la señal normalizada)
  - phi220 ~ Uniform[-π, π]
  - alpha_flow ~ Uniform[-0.5, 0.5]

Se usa una verosimilitud gaussiana simple:

  ln L = -0.5 * sum( (h_data_norm - h_model)^2 )

(sigma^2 absorbido en la definición de S).

El script imprime el intervalo de credibilidad 5%-95% para alpha_flow
y guarda un histograma en data/gw190521_alpha_posterior.png
"""

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============
# Parámetros físicos como en gw190521_edr_ringdown.py
# ============

G_SI    = 6.67430e-11
C_SI    = 2.99792458e8
MSUN_SI = 1.98847e30

M_FINAL_SOLAR = 142.0
A_FINAL       = 0.72


def geom_time_from_mass_solar(m_solar: float) -> float:
    return G_SI * m_solar * MSUN_SI / C_SI**3


def qnm_220_GR(m_final_solar: float, a_final: float) -> tuple[float, float]:
    f1, f2, f3 = 1.5251, -1.1568, 0.1292
    g1, g2, g3 = 0.0889, -0.0260, 0.3350

    M_geom = geom_time_from_mass_solar(m_final_solar)
    one_minus_a = 1.0 - a_final

    omegaR_M = f1 + f2 * (one_minus_a**f3)
    omegaI_M = g1 + g2 * (one_minus_a**g3)

    omegaR_SI = omegaR_M / M_geom
    omegaI_SI = omegaI_M / M_geom

    f_Hz  = omegaR_SI / (2.0 * np.pi)
    tau_s = 1.0 / abs(omegaI_SI)
    return f_Hz, tau_s


def qnm_EDR_220(alpha_flow: float = 0.0) -> tuple[float, float]:
    """
    Devuelve f_220^EDR (Hz) y tau_220^EDR (s) para GW190521, con la
    corrección alpha_flow sobre la frecuencia. beta_flow=0 aquí.
    """
    f_GR, tau_GR = qnm_220_GR(M_FINAL_SOLAR, A_FINAL)
    f_EDR   = f_GR   * (1.0 + alpha_flow)
    tau_EDR = tau_GR * (1.0)   # sin corrección en tau
    return f_EDR, tau_EDR


def ringdown_model_M1(t: np.ndarray,
                      A220: float,
                      phi220: float,
                      alpha_flow: float) -> np.ndarray:
    """
    Modelo M1: GR + alpha_flow (220 corregido, sin modo X).
    """
    f220_EDR, tau220_EDR = qnm_EDR_220(alpha_flow)
    omega220 = 2.0 * np.pi * f220_EDR
    return A220 * np.exp(-t / tau220_EDR) * np.cos(omega220 * t + phi220)


def load_strain_from_hdf5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        for key in ["strain/Strain", "strain/strain", "Strain/Strain"]:
            if key in f:
                dset = f[key]
                break
        else:
            raise KeyError("No se encontró dataset de strain en el HDF5.")
        strain = dset[()]
        t0 = dset.attrs.get("Xstart", 0.0)
        dt = dset.attrs.get("Xspacing", 1.0 / 4096.0)
    n = len(strain)
    times = t0 + np.arange(n) * dt
    return times, strain


def find_h1_file_for_gw190521(data_dir: str) -> str:
    patterns = [
        "H-H1_GWOSC_4KHZ_R1-1242442952-32.hdf5",
        "H-*GW190521*.hdf5",
        "H-H1*GW190521*.hdf5",
        "H1*GW190521*.hdf5",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(data_dir, pat)))
    if not candidates:
        candidates.extend(glob.glob(os.path.join(data_dir, "H-*190521*.hdf5")))
    if not candidates:
        all_h1 = glob.glob(os.path.join(data_dir, "H-*.hdf5"))
        print("Archivos H-*.hdf5 en data/:")
        for f in all_h1:
            print(" ", os.path.basename(f))
        raise FileNotFoundError("No se encontró archivo H1 de GW190521 en data/.")
    chosen = sorted(candidates)[0]
    print(f"Archivo H1 para GW190521 seleccionado: {chosen}")
    return chosen


# ============
# Likelihood y priors
# ============

def log_prior(theta: np.ndarray, A0_abs: float) -> float:
    """
    Priors uniformes:

      A220     ~ U[-10|A0|, 10|A0|]
      phi220   ~ U[-π, π]
      alpha_fl ~ U[-0.5, 0.5]
    """
    A220, phi220, alpha_flow = theta
    if not (-10*A0_abs <= A220 <= 10*A0_abs):
        return -np.inf
    if not (-np.pi <= phi220 <= np.pi):
        return -np.inf
    if not (-0.5 <= alpha_flow <= 0.5):
        return -np.inf
    # Todos uniformes -> constante (que podemos tomar como 0)
    return 0.0


def log_likelihood(theta: np.ndarray,
                   t_rel: np.ndarray,
                   h_norm: np.ndarray) -> float:
    A220, phi220, alpha_flow = theta
    model = ringdown_model_M1(t_rel, A220, phi220, alpha_flow)
    resid = h_norm - model
    S = np.sum(resid**2)
    return -0.5 * S


def metropolis_hastings(n_samples: int,
                        init_theta: np.ndarray,
                        proposal_scales: np.ndarray,
                        t_rel: np.ndarray,
                        h_norm: np.ndarray,
                        A0_abs: float,
                        burn_in: int = 2000) -> np.ndarray:
    """
    MCMC Metropolis–Hastings muy simple.
    """
    ndim = len(init_theta)
    samples = np.zeros((n_samples, ndim))

    theta = init_theta.copy()
    lp = log_prior(theta, A0_abs)
    ll = log_likelihood(theta, t_rel, h_norm)
    post = lp + ll

    rng = np.random.default_rng()

    accepted = 0
    for i in range(n_samples + burn_in):
        proposal = theta + proposal_scales * rng.normal(size=ndim)
        lp_prop = log_prior(proposal, A0_abs)
        if np.isneginf(lp_prop):
            # Rechazar inmediatamente
            accept = False
        else:
            ll_prop = log_likelihood(proposal, t_rel, h_norm)
            post_prop = lp_prop + ll_prop
            log_alpha = post_prop - post
            accept = (np.log(rng.random()) < log_alpha)

        if accept:
            theta = proposal
            lp = lp_prop
            ll = ll_prop
            post = post_prop
            if i >= burn_in:
                accepted += 1

        if i >= burn_in:
            samples[i - burn_in] = theta

        if (i+1) % 2000 == 0:
            print(f"Iteración {i+1}/{n_samples + burn_in}")

    acc_rate = accepted / n_samples
    print(f"Tasa de aceptación ~ {acc_rate:.3f}")
    return samples


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data")
    h1_file = find_h1_file_for_gw190521(data_dir)
    t, h = load_strain_from_hdf5(h1_file)
    h = h - np.mean(h)

    # Ventana de ringdown: 1 a 30 ms después del pico (igual que en gw190521_edr_ringdown.py)
    idx_peak = np.argmax(np.abs(h))
    t_peak = t[idx_peak]
    t_start = t_peak + 0.001
    t_end   = t_peak + 0.030
    mask = (t >= t_start) & (t <= t_end)
    t_rd = t[mask]
    h_rd = h[mask]
    t_rel = t_rd - t_rd[0]

    h_rms = np.sqrt(np.mean(h_rd**2))
    h_norm = h_rd / h_rms if h_rms != 0 else h_rd

    print(f"Ventana GW190521(H1): duración ~ {t_rel[-1] - t_rel[0]:.6f} s, puntos = {len(t_rel)}")
    print(f"RMS de la ventana: {h_rms:.3e}")

    # Estimaciones iniciales
    A0 = float(np.max(h_norm))
    A0_abs = abs(A0)
    init_theta = np.array([A0, 0.0, -0.5])  # empezamos cerca de tu mejor ajuste
    proposal_scales = np.array([0.1*A0_abs if A0_abs > 0 else 0.1,
                                0.1,
                                0.02])      # tunear si hace falta

    print("Theta inicial (A220, phi220, alpha_flow):", init_theta)

    n_samples = 20000  # se puede aumentar si quieres cadenas más largas
    samples = metropolis_hastings(
        n_samples=n_samples,
        init_theta=init_theta,
        proposal_scales=proposal_scales,
        t_rel=t_rel,
        h_norm=h_norm,
        A0_abs=A0_abs,
        burn_in=5000,
    )

    A_chain      = samples[:, 0]
    phi_chain    = samples[:, 1]
    alpha_chain  = samples[:, 2]

    # Resumen para alpha_flow
    alpha_med = np.median(alpha_chain)
    alpha_p05 = np.percentile(alpha_chain, 5)
    alpha_p95 = np.percentile(alpha_chain, 95)

    print("\n=== Posterior de alpha_flow (GW190521 H1, modelo M1) ===")
    print(f"Mediana     : {alpha_med:.3e}")
    print(f"IC 5%–95%   : [{alpha_p05:.3e}, {alpha_p95:.3e}]")
    print(f"Valor GR (0) está {'FUERA' if not (alpha_p05 <= 0.0 <= alpha_p95) else 'DENTRO'} del IC 5%–95%")

    # Histograma de alpha_flow
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(alpha_chain, bins=50, density=True, color="C2", alpha=0.7)
    ax.axvline(0.0, color="k", ls="--", label="GR (alpha=0)")
    ax.axvline(alpha_med, color="C1", ls="-", label="Mediana")
    ax.set_xlabel("alpha_flow")
    ax.set_ylabel("Posterior (densidad)")
    ax.set_title("Posterior de alpha_flow (GW190521 H1, M1)")
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(repo_root, "data", "gw190521_alpha_posterior.png")
    fig.savefig(out_png, dpi=150)
    print(f"Histograma de alpha_flow guardado en {out_png}")


if __name__ == "__main__":
    main()

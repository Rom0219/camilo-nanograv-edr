# scripts/gw150914_edr_ringdown.py
"""
Análisis de ringdown de GW150914 con el modelo EDR (modo 220 + modo X).

Requiere que los archivos HDF5 de GW150914 estén en data/, generados por
scripts/download_data.py
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constantes físicas (SI)
G_SI    = 6.67430e-11       # m^3 kg^-1 s^-2
C_SI    = 2.99792458e8      # m s^-1
MSUN_SI = 1.98847e30        # kg

# Parámetros del remanente de GW150914 (aprox.)
M_FINAL_SOLAR = 68.0
A_FINAL       = 0.67


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

    f_Hz = omegaR_SI / (2.0 * np.pi)
    tau_s = 1.0 / abs(omegaI_SI)
    return f_Hz, tau_s


def omega_horizon(m_final_solar: float, a_final: float) -> float:
    M_geom = geom_time_from_mass_solar(m_final_solar)
    a = a_final
    rp_geom = M_geom * (1.0 + np.sqrt(1.0 - a**2))
    omega_H = a / (2.0 * M_geom * rp_geom)
    return omega_H


def qnm_EDR_parameters(alpha_flow: float = 0.0,
                       beta_flow: float = 0.0,
                       k_flow: float = 1.0,
                       gamma_flow: float = 10.0) -> tuple[float, float, float, float]:
    f220_GR, tau220_GR = qnm_220_GR(M_FINAL_SOLAR, A_FINAL)

    f220_EDR   = f220_GR   * (1.0 + alpha_flow)
    tau220_EDR = tau220_GR * (1.0 + beta_flow)

    omega_H = omega_horizon(M_FINAL_SOLAR, A_FINAL)
    fX_Hz = (k_flow * omega_H) / (2.0 * np.pi)
    M_geom = geom_time_from_mass_solar(M_FINAL_SOLAR)
    tauX_s = gamma_flow * M_geom

    return f220_EDR, tau220_EDR, fX_Hz, tauX_s


def ringdown_model_EDR(t: np.ndarray,
                       A220: float, phi220: float,
                       AX: float, phiX: float,
                       alpha_flow: float,
                       k_flow: float = 1.0,
                       gamma_flow: float = 10.0) -> np.ndarray:
    f220_EDR, tau220_EDR, fX_Hz, tauX_s = qnm_EDR_parameters(
        alpha_flow=alpha_flow,
        beta_flow=0.0,
        k_flow=k_flow,
        gamma_flow=gamma_flow,
    )

    omega220 = 2.0 * np.pi * f220_EDR
    omegaX   = 2.0 * np.pi * fX_Hz

    h220 = A220 * np.exp(-t / tau220_EDR) * np.cos(omega220 * t + phi220)
    hX   = AX   * np.exp(-t / tauX_s)    * np.cos(omegaX   * t + phiX)

    return h220 + hX


def load_strain_from_hdf5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        dset = f["strain/Strain"]
        strain = dset[()]
        t0 = dset.attrs["Xstart"]
        dt = dset.attrs["Xspacing"]

    n = len(strain)
    times = t0 + np.arange(n) * dt
    return times, strain


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data")
    h1_file = os.path.join(data_dir, "H-H1_LOSC_4_V1-1126259446-32.hdf5")

    if not os.path.exists(h1_file):
        raise FileNotFoundError(
            f"No se encontró {h1_file}. Ejecuta primero scripts/download_data.py"
        )

    t, h = load_strain_from_hdf5(h1_file)

    # Filtro muy simple: quitar media (para este ejemplo básico)
    h = h - np.mean(h)

    # Seleccionar ventana de ringdown alrededor del máximo
    idx_peak = np.argmax(np.abs(h))
    t_peak = t[idx_peak]

    t_start = t_peak + 0.002  # 2 ms después del pico
    t_end   = t_peak + 0.030  # hasta 30 ms después del pico

    mask = (t >= t_start) & (t <= t_end)
    t_rd = t[mask]
    h_rd = h[mask]

    t_rel = t_rd - t_rd[0]

    print(f"Ventana de ringdown: duración ~ {t_rel[-1] - t_rel[0]:.6f} s, puntos = {len(t_rel)}")

    # Ajuste
    k_flow_example     = 1.0
    gamma_flow_example = 10.0

    def model_to_fit(t_arr, A220, phi220, AX, phiX, alpha_flow):
        return ringdown_model_EDR(
            t_arr,
            A220=A220,
            phi220=phi220,
            AX=AX,
            phiX=phiX,
            alpha_flow=alpha_flow,
            k_flow=k_flow_example,
            gamma_flow=gamma_flow_example,
        )

    A220_0   = np.max(h_rd)
    phi220_0 = 0.0
    AX_0     = 0.1 * A220_0
    phiX_0   = 0.0
    alpha_0  = 0.0

    p0 = [A220_0, phi220_0, AX_0, phiX_0, alpha_0]

    bounds = (
        [-10*A220_0, -2*np.pi, -10*A220_0, -2*np.pi, -0.5],
        [ 10*A220_0,  2*np.pi,  10*A220_0,  2*np.pi,  0.5],
    )

    print("Ajustando modelo EDR...")
    popt, pcov = curve_fit(model_to_fit, t_rel, h_rd, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    A220_fit, phi220_fit, AX_fit, phiX_fit, alpha_fit = popt

    print("\nResultados del ajuste (H1, ventana de ringdown):")
    print(f"A220       = {A220_fit:.3e} ± {perr[0]:.3e}")
    print(f"phi220     = {phi220_fit:.3f} ± {perr[1]:.3f} rad")
    print(f"AX         = {AX_fit:.3e} ± {perr[2]:.3e}")
    print(f"phiX       = {phiX_fit:.3f} ± {perr[3]:.3f} rad")
    print(f"alpha_flow = {alpha_fit:.3e} ± {perr[4]:.3e}")

    h_fit = model_to_fit(t_rel, *popt)

    # Gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(t_rel * 1000, h_rd, label="Datos H1 (ringdown)", lw=1)
    plt.plot(t_rel * 1000, h_fit, label="Modelo EDR ajustado", lw=2)
    plt.xlabel("Tiempo desde inicio de la ventana [ms]")
    plt.ylabel("Strain")
    plt.title("GW150914 (H1): Ajuste de ringdown con modelo EDR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(repo_root, "data", "gw150914_edr_ringdown.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nFigura guardada en {out_png}")


if __name__ == "__main__":
    main()
